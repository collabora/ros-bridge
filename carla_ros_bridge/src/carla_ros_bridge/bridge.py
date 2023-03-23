#!/usr/bin/env python
#
# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
Rosbridge class:

Class that handle communication between CARLA and ROS
"""

import math
import os
import numpy as np
import pkg_resources

try:
    import queue
except ImportError:
    import Queue as queue
import sys
from distutils.version import LooseVersion
from threading import Thread, Lock, Event

import carla
import cv2
from geometry_msgs.msg import Pose

from transforms3d.euler import euler2quat

import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode

from carla_ros_bridge.actor import Actor
from carla_ros_bridge.actor_factory import ActorFactory
from carla_ros_bridge.carla_status_publisher import CarlaStatusPublisher
from carla_ros_bridge.debug_helper import DebugHelper
from carla_ros_bridge.ego_vehicle import EgoVehicle
from carla_ros_bridge.world_info import WorldInfo

from carla_msgs.msg import CarlaControl, CarlaWeatherParameters
from carla_msgs.srv import SpawnObject, DestroyObject, GetBlueprints
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CompressedImage
from carla_ros_bridge.bounding_boxes import ClientSideBoundingBoxes
from carla_ros_bridge.camera import Camera, RgbCamera
from carla_ros_bridge.lidar import Lidar
from carla_ros_bridge.lidar_to_rgb import LidarToRGB
from carla_ros_bridge.vehicle import Vehicle
from carla_ros_bridge.tf_sensor import TFSensor
from carla_ros_bridge.bev import BevFusion
from carla_common.transforms import carla_rotation_to_quaternion


class CarlaRosBridge(CompatibleNode):

    """
    Carla Ros bridge
    """

    with open(os.path.join(os.path.dirname(__file__), "CARLA_VERSION")) as f:
        CARLA_VERSION = f.read()[:-1]

    # in synchronous mode, if synchronous_mode_wait_for_vehicle_control_command is True,
    # wait for this time until a next tick is triggered.
    VEHICLE_CONTROL_TIMEOUT = 1.
    BB_COLOR = (92,41,131)
    BB_COLOR_WALKERS = (64,186,47)

    def __init__(self):
        """
        Constructor

        :param carla_world: carla world object
        :type carla_world: carla.World
        :param params: dict of parameters, see settings.yaml
        :type params: dict
        """
        super(CarlaRosBridge, self).__init__("ros_bridge_node")

    # pylint: disable=attribute-defined-outside-init
    def initialize_bridge(self, carla_world, params):
        """
        Initialize the bridge
        """
        self.parameters = params
        self.carla_world = carla_world

        self.ros_timestamp = roscomp.ros_timestamp()
        self.callback_group = roscomp.callback_groups.ReentrantCallbackGroup()

        self.synchronous_mode_update_thread = None
        self.shutdown = Event()

        self.carla_settings = carla_world.get_settings()
        if not self.parameters["passive"]:
            # workaround: settings can only applied within non-sync mode
            if self.carla_settings.synchronous_mode:
                self.carla_settings.synchronous_mode = False
                carla_world.apply_settings(self.carla_settings)

            self.loginfo("synchronous_mode: {}".format(
                self.parameters["synchronous_mode"]))
            self.carla_settings.synchronous_mode = self.parameters["synchronous_mode"]
            self.loginfo("fixed_delta_seconds: {}".format(
                self.parameters["fixed_delta_seconds"]))
            self.carla_settings.fixed_delta_seconds = self.parameters["fixed_delta_seconds"]
            carla_world.apply_settings(self.carla_settings)

        self.loginfo("Parameters:")
        for key in self.parameters:
            self.loginfo("  {}: {}".format(key, self.parameters[key]))

        # active sync mode in the ros bridge only if CARLA world is configured in sync mode and
        # passive mode is not enabled.
        self.sync_mode = self.carla_settings.synchronous_mode and not self.parameters["passive"]
        if self.carla_settings.synchronous_mode and self.parameters["passive"]:
            self.loginfo(
                "Passive mode is enabled and CARLA world is configured in synchronous mode. This configuration requires another client ticking the CARLA world.")

        self.carla_control_queue = queue.Queue()

        # actor factory
        self.actor_factory = ActorFactory(self, carla_world, self.sync_mode)

        # add world info
        self.world_info = WorldInfo(carla_world=self.carla_world, node=self)
        # add debug helper
        self.debug_helper = DebugHelper(carla_world.debug, self)

        # Communication topics
        self.clock_publisher = self.new_publisher(Clock, 'clock', 10)

        self.status_publisher = CarlaStatusPublisher(
            self.carla_settings.synchronous_mode,
            self.carla_settings.fixed_delta_seconds,
            self)

        # for waiting for ego vehicle control commands in synchronous mode,
        # their ids are maintained in a list.
        # Before tick(), the list is filled and the loop waits until the list is empty.
        self._all_vehicle_control_commands_received = Event()
        self._expected_ego_vehicle_control_command_ids = []
        self._expected_ego_vehicle_control_command_ids_lock = Lock()
        
        # To spawn pseudo tf sensor
        self.ego_vehicles = {}

        if self.sync_mode:
            self.carla_run_state = CarlaControl.PLAY

            self.carla_control_subscriber = \
                self.new_subscription(CarlaControl, "/carla/control",
                                      lambda control: self.carla_control_queue.put(control.command),
                                      qos_profile=10, callback_group=self.callback_group)

            self.bevfusion = BevFusion(
                self.parameters["bevfusion_config"],
                ckpt=self.parameters["bevfusion_ckpt"],
                calibrated_sensor_cfg=self.parameters["calibrated_sensor_cfg"])

            # for bbox image publish
            self.n_rgb_cams = 0
            self.rgb_cams = {}
            self.boxes_lidar_publishers = {}
            self.boxes_lidar_publishers_pred = {}
            
            # lidar overlay RGB
            self.lidar = {}
            self.lidar_to_camera = None

            self.synchronous_mode_update_thread = Thread(
                target=self._synchronous_mode_update)
            self.synchronous_mode_update_thread.start()
        else:
            self.timestamp_last_run = 0.0

            self.actor_factory.start()

            # register callback to update actors
            self.on_tick_id = self.carla_world.on_tick(self._carla_time_tick)

        # services configuration.
        self._registered_actors = []
        self.spawn_object_service = self.new_service(SpawnObject, "/carla/spawn_object",
                                                     self.spawn_object)
        self.destroy_object_service = self.new_service(DestroyObject, "/carla/destroy_object",
                                                       self.destroy_object)

        self.get_blueprints_service = self.new_service(GetBlueprints, "/carla/get_blueprints",
                                                       self.get_blueprints, callback_group=self.callback_group)

        self.carla_weather_subscriber = \
            self.new_subscription(CarlaWeatherParameters, "/carla/weather_control",
                                  self.on_weather_changed, qos_profile=10, callback_group=self.callback_group)

    def spawn_object(self, req, response=None):
        response = roscomp.get_service_response(SpawnObject)
        if not self.shutdown.is_set():
            try:
                id_ = self.actor_factory.spawn_actor(req)
                self._registered_actors.append(id_)
                response.id = id_
            except Exception as e:
                self.logwarn("Error spawning object '{}': {}".format(req.type, e))
                response.id = -1
                response.error_string = str(e)
        else:
            response.id = -1
            response.error_string = 'Bridge is shutting down, object will not be spawned.'
        return response
    
    def create_spawn_point(self, x, y, z, roll, pitch, yaw):
        spawn_point = Pose()
        spawn_point.position.x = x
        spawn_point.position.y = y
        spawn_point.position.z = z
        quat = euler2quat(math.radians(roll), math.radians(pitch), math.radians(yaw))

        spawn_point.orientation.w = quat[0]
        spawn_point.orientation.x = quat[1]
        spawn_point.orientation.y = quat[2]
        spawn_point.orientation.z = quat[3]
        return spawn_point
    
    def spawn_pseudo_sensor(self):
        if len(self.ego_vehicles) == 0:
            if not self.get_ego_vehicles():
                self.logwarn("No ego vehicles found cannot spawn pesudo sensors.")
                return
        
        for attached_vehicle_id, has_tf_sensor in self.ego_vehicles.items():
            if has_tf_sensor:
                continue
            try:
                sensor_transform = self.create_spawn_point(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                # create request
                spawn_object_request = roscomp.get_service_request(SpawnObject)
                spawn_object_request.type = "sensor.pseudo.tf"
                spawn_object_request.id = "tf"
                spawn_object_request.attach_to = attached_vehicle_id
                spawn_object_request.transform = sensor_transform
                spawn_object_request.random_pose = False  # never set a random pose for a sensor

                response = self.spawn_object(spawn_object_request)
                response_id = response.id
                if response_id == -1:
                    raise RuntimeError(response.error_string)
                else:
                    self.ego_vehicles[attached_vehicle_id] = True
            except RuntimeError as e:
                self.logerr(
                        "Sensor {} for vehicle {} will not be spawned: {}".format(
                        "sensor.pseudo.tf", attached_vehicle_id,  e))
                continue

    def get_ego_vehicles_and_tf_sensors(self):
        """Gets all ego vehicles from the world actors dict.
        """
        for id_, actor in self.actor_factory.actors.items():
            if not isinstance(actor, Vehicle): continue
            if not actor.carla_actor.attributes.get('role_name') \
                    in self.parameters['ego_vehicle']['role_name']:
                continue
        
            if id_ not in self.ego_vehicles.keys():
                self.ego_vehicles[id_] = False
            
        # Check if tf sensor alreday exists for any of the ego vehicles
        for _, actor in self.actor_factory.actors.items():
            if not isinstance(actor, TFSensor): continue
            self.loginfo(f"Found TFSensor for ego vehicle with id={actor.parent.uid}")
            if actor.parent.uid not in self.ego_vehicles.keys():
                self.ego_vehicles[id_] = False
            self.ego_vehicles[actor.parent.uid] = True

    def destroy_object(self, req, response=None):
        response = roscomp.get_service_response(DestroyObject)
        destroyed_actors = self.actor_factory.destroy_actor(req.id)
        response.success = bool(destroyed_actors)
        for actor in destroyed_actors:
            if actor in self._registered_actors:
                self._registered_actors.remove(actor)
        return response

    def get_blueprints(self, req):
        response = roscomp.get_service_response(GetBlueprints)
        if req.filter:
            bp_filter = req.filter
        else:
            bp_filter = "*"

        response.blueprints = [
            bp.id for bp in self.carla_world.get_blueprint_library().filter(bp_filter)]
        response.blueprints.extend(self.actor_factory.get_pseudo_sensor_types())
        response.blueprints.sort()
        return response

    def on_weather_changed(self, weather_parameters):
        """
        Callback on new weather parameters
        :return:
        """
        if not self.carla_world:
            return
        self.loginfo("Applying weather parameters...")
        weather = carla.WeatherParameters()
        weather.cloudiness = weather_parameters.cloudiness
        weather.precipitation = weather_parameters.precipitation
        weather.precipitation_deposits = weather_parameters.precipitation_deposits
        weather.wind_intensity = weather_parameters.wind_intensity
        weather.fog_density = weather_parameters.fog_density
        weather.fog_distance = weather_parameters.fog_distance
        weather.wetness = weather_parameters.wetness
        weather.sun_azimuth_angle = weather_parameters.sun_azimuth_angle
        weather.sun_altitude_angle = weather_parameters.sun_altitude_angle
        self.carla_world.set_weather(weather)

    def process_run_state(self):
        """
        process state changes
        """
        command = None

        # get last command
        while not self.carla_control_queue.empty():
            command = self.carla_control_queue.get()

        while command is not None and roscomp.ok():
            self.carla_run_state = command

            if self.carla_run_state == CarlaControl.PAUSE:
                # wait for next command
                self.loginfo("State set to PAUSED")
                self.status_publisher.set_synchronous_mode_running(False)
                command = self.carla_control_queue.get()
            elif self.carla_run_state == CarlaControl.PLAY:
                self.loginfo("State set to PLAY")
                self.status_publisher.set_synchronous_mode_running(True)
                return
            elif self.carla_run_state == CarlaControl.STEP_ONCE:
                self.loginfo("Execute single step.")
                self.status_publisher.set_synchronous_mode_running(True)
                self.carla_control_queue.put(CarlaControl.PAUSE)
                return

    def get_rgb_cams_and_publishers(self):
        """Gets all the RGB cameras and lidar sensor for
        all the Ego Vehicles in the simulation.
        """
        # Get ego vehicles
        self.get_ego_vehicles_and_tf_sensors()

        # Fetch RGB cameras and create publishers
        for _, actor in self.actor_factory.actors.items():
            if not isinstance(actor, RgbCamera): continue
            id_ = actor.get_id()
            parent = actor.parent
            if parent not in self.boxes_lidar_publishers.keys():
                self.boxes_lidar_publishers[parent] = {}
                self.boxes_lidar_publishers_pred[parent] = {}
                
            if parent not in self.rgb_cams.keys():
                self.rgb_cams[parent] = {}

            if id_ not in self.boxes_lidar_publishers[parent].keys():
                self.boxes_lidar_publishers[parent][id_] = self.new_publisher(
                    CompressedImage, 
                    actor.get_topic_prefix() + '/' + 'bboxes_lidar',
                    qos_profile=10)
                if 'rgb_view' not in actor.get_topic_prefix():
                    self.boxes_lidar_publishers_pred[parent][id_] = self.new_publisher(
                        CompressedImage, 
                        actor.get_topic_prefix() + '/' + 'bboxes_lidar_pred',
                        qos_profile=10)
                self.rgb_cams[parent][id_] = actor
                self.n_rgb_cams += 1

        # Fetch lidar sensors 
        for _, actor in self.actor_factory.actors.items():
            if not isinstance(actor, Lidar): continue
            parent = actor.parent
            if parent not in self.lidar.keys():
                self.lidar[parent] = actor

        # Initialize lidar to camera 
        if self.lidar_to_camera is None:
            self.lidar_to_camera = LidarToRGB()
    
    def publish_bboxes_and_lidar_overlay(self):
        """Publishes bboxes and lidar overlay RGB.
        """
        if self.n_rgb_cams != len(self.carla_world.get_actors().filter('sensor.camera.rgb')) or \
            len(self.lidar.keys()) != len(self.carla_world.get_actors().filter('sensor.lidar.ray_cast')) or \
            self.lidar_to_camera is None:
            self.get_rgb_cams_and_publishers()
            if self.n_rgb_cams != len(self.carla_world.get_actors().filter('sensor.camera.rgb')): return

        if self.lidar_to_camera is None:
            self.logwarn("Lidar overlay.. couldnt setup lidar_to_rgb.")

        vehicles = self.carla_world.get_actors().filter('vehicle.*')
        walkers = self.carla_world.get_actors().filter('walker.*')

        for parent, rgb_cams in self.rgb_cams.items():
            # if parent is None: return
            # check if lidar exists for the same parent
            curr_lidar = None
            if parent in self.lidar:
                curr_lidar = self.lidar[parent]
            else:
                self.logwarn(f"lidar is None for {parent}")

            # bevfusion inference
            # prepare data
            images = {}
            frame = None
            for id_, rgb_cam in rgb_cams.items():
                frame = rgb_cam.get_frame()
                try:
                    images[rgb_cam.get_topic_prefix().split('/')[-1]] = cv2.cvtColor(rgb_cam.get_image(), cv2.COLOR_BGR2RGB)
                except:
                    return
            ego2global_location = parent.carla_actor.get_transform().location
            ego2global_translation = [ego2global_location.x, -ego2global_location.y, ego2global_location.z]
            ego2global_translation[-1] = 0.
            ego2global_rotation = carla_rotation_to_quaternion(parent.carla_actor.get_transform().rotation)

            if curr_lidar is not None:
                lidar_data, lidar_frame = curr_lidar.get_lidar_world_with_ring_index()
                if lidar_frame != frame: 
                    self.logwarn(
                        f"Bevfusion.. RGB {rgb_cam.get_topic_prefix().split('/')[-1]} and lidar not in sync.")
                    continue
                self.loginfo((ego2global_translation, ego2global_rotation))
                output_images = self.bevfusion([images, lidar_data, ego2global_translation, ego2global_rotation])
                if output_images is not None:
                    for id_, rgb_cam in rgb_cams.items():
                        header = rgb_cam.get_msg_header()
                        img = output_images[rgb_cam.get_topic_prefix().split('/')[-1]]
                        # publish bboxes
                        img_msg_bboxes_lidar = Camera.cv_bridge.cv2_to_compressed_imgmsg(img)
                        img_msg_bboxes_lidar.header = header
                        try:
                            self.boxes_lidar_publishers_pred[parent][id_].publish(img_msg_bboxes_lidar)
                        except Exception as e:
                            self.logwarn(f"Publish bboxes with lidar exception {e}")

                        # publish original imagewith lidar overlay
                        img = images[rgb_cam.get_topic_prefix().split('/')[-1]]
                        img = img[:, :, ::-1]

                        # draw lidar points
                        image_w = float(rgb_cam.carla_actor.attributes.get("image_size_x"))
                        image_h = float(rgb_cam.carla_actor.attributes.get("image_size_y"))
                        img = self.lidar_to_camera.lidar_overlay(
                            curr_lidar, curr_lidar.get_lidar_data(), img, rgb_cam, image_w, image_h)

                        # publish bboxes with lidar overlay
                        img_msg_bboxes_lidar = Camera.cv_bridge.cv2_to_compressed_imgmsg(img)
                        img_msg_bboxes_lidar.header = header
                        try:
                            self.boxes_lidar_publishers[parent][id_].publish(img_msg_bboxes_lidar)
                        except Exception as e:
                            self.logwarn(f"Publish bboxes with lidar exception {e}")

    def _synchronous_mode_update(self):
        """
        execution loop for synchronous mode
        """
        while not self.shutdown.is_set() and roscomp.ok():
            self.process_run_state()

            if self.parameters['synchronous_mode_wait_for_vehicle_control_command']:
                # fill list of available ego vehicles
                self._expected_ego_vehicle_control_command_ids = []
                with self._expected_ego_vehicle_control_command_ids_lock:
                    for actor_id, actor in self.actor_factory.actors.items():
                        if isinstance(actor, EgoVehicle):
                            self._expected_ego_vehicle_control_command_ids.append(
                                actor_id)

            self.actor_factory.update_available_objects()

            # Spawn pseudo tf sensor if not exist
            if len(self.ego_vehicles.keys()):
                if list(self.ego_vehicles.values()).count(False) > 0:
                    self.spawn_pseudo_sensor()

            frame = self.carla_world.tick()
            
            world_snapshot = self.carla_world.get_snapshot()

            self.status_publisher.set_frame(frame)
            
            # TODO; (ROS1) fix 'Clock' object has no attribute '_buff'
            self.update_clock(world_snapshot.timestamp)
            
            self.logdebug("Tick for frame {} returned. Waiting for sensor data...".format(
                frame))
            self._update(frame, world_snapshot.timestamp.elapsed_seconds)
            self.publish_bboxes_and_lidar_overlay()
            self.logdebug("Waiting for sensor data finished.")

            if self.parameters['synchronous_mode_wait_for_vehicle_control_command']:
                # wait for all ego vehicles to send a vehicle control command
                if self._expected_ego_vehicle_control_command_ids:
                    if not self._all_vehicle_control_commands_received.wait(CarlaRosBridge.VEHICLE_CONTROL_TIMEOUT):
                        self.logwarn("Timeout ({}s) while waiting for vehicle control commands. "
                                     "Missing command from actor ids {}".format(CarlaRosBridge.VEHICLE_CONTROL_TIMEOUT,
                                                                                self._expected_ego_vehicle_control_command_ids))
                    self._all_vehicle_control_commands_received.clear()

    def _carla_time_tick(self, carla_snapshot):
        """
        Private callback registered at carla.World.on_tick()
        to trigger cyclic updates.

        After successful locking the update mutex
        (only perform trylock to respect bridge processing time)
        the clock and the children are updated.
        Finally the ROS messages collected to be published are sent out.

        :param carla_timestamp: the current carla time
        :type carla_timestamp: carla.Timestamp
        :return:
        """
        if not self.shutdown.is_set():
            if self.timestamp_last_run < carla_snapshot.timestamp.elapsed_seconds:
                self.timestamp_last_run = carla_snapshot.timestamp.elapsed_seconds
                self.update_clock(carla_snapshot.timestamp)
                self.status_publisher.set_frame(carla_snapshot.frame)
                self._update(carla_snapshot.frame,
                             carla_snapshot.timestamp.elapsed_seconds)

    def _update(self, frame_id, timestamp):
        """
        update all actors
        :return:
        """
        self.world_info.update(frame_id, timestamp)
        self.actor_factory.update_actor_states(frame_id, timestamp)

    def _ego_vehicle_control_applied_callback(self, ego_vehicle_id):
        if not self.sync_mode or \
                not self.parameters['synchronous_mode_wait_for_vehicle_control_command']:
            return
        with self._expected_ego_vehicle_control_command_ids_lock:
            if ego_vehicle_id in self._expected_ego_vehicle_control_command_ids:
                self._expected_ego_vehicle_control_command_ids.remove(
                    ego_vehicle_id)
            else:
                self.logwarn(
                    "Unexpected vehicle control command received from {}".format(ego_vehicle_id))
            if not self._expected_ego_vehicle_control_command_ids:
                self._all_vehicle_control_commands_received.set()

    def update_clock(self, carla_timestamp):
        """
        perform the update of the clock

        :param carla_timestamp: the current carla time
        :type carla_timestamp: carla.Timestamp
        :return:
        """
        if roscomp.ok():
            self.ros_timestamp = roscomp.ros_timestamp(carla_timestamp.elapsed_seconds, from_sec=True)
            self.clock_publisher.publish(Clock(clock=self.ros_timestamp))

    def destroy(self):
        """
        Function to destroy this object.

        :return:
        """
        self.loginfo("Shutting down...")
        self.shutdown.set()
        if not self.sync_mode:
            if self.on_tick_id:
                self.carla_world.remove_on_tick(self.on_tick_id)
            self.actor_factory.thread.join()
        else:
            self.synchronous_mode_update_thread.join()
        self.loginfo("Object update finished.")
        self.debug_helper.destroy()
        self.status_publisher.destroy()
        self.destroy_service(self.spawn_object_service)
        self.destroy_service(self.destroy_object_service)
        self.destroy_subscription(self.carla_weather_subscriber)
        self.carla_control_queue.put(CarlaControl.STEP_ONCE)

        for uid in self._registered_actors:
            self.actor_factory.destroy_actor(uid)
        self.actor_factory.update_available_objects()
        self.actor_factory.clear()
        super(CarlaRosBridge, self).destroy()


def main(args=None):
    """
    main function for carla simulator ROS bridge
    maintaining the communication client and the CarlaBridge object
    """
    roscomp.init("bridge", args=args)

    carla_bridge = None
    carla_world = None
    carla_client = None
    executor = None
    parameters = {}

    executor = roscomp.executors.MultiThreadedExecutor()
    carla_bridge = CarlaRosBridge()
    executor.add_node(carla_bridge)

    roscomp.on_shutdown(carla_bridge.destroy)

    parameters['host'] = carla_bridge.get_param('host', 'localhost')
    parameters['port'] = carla_bridge.get_param('port', 2000)
    parameters['timeout'] = carla_bridge.get_param('timeout', 2)
    parameters['passive'] = carla_bridge.get_param('passive', False)
    parameters['synchronous_mode'] = carla_bridge.get_param('synchronous_mode', True)
    parameters['synchronous_mode_wait_for_vehicle_control_command'] = carla_bridge.get_param(
        'synchronous_mode_wait_for_vehicle_control_command', False)
    parameters['fixed_delta_seconds'] = carla_bridge.get_param('fixed_delta_seconds', 0.05)
    parameters['register_all_sensors'] = carla_bridge.get_param('register_all_sensors', True)
    parameters['town'] = carla_bridge.get_param('town', 'Town01')
    role_name = carla_bridge.get_param('ego_vehicle_role_name',
                                       ["hero", "ego_vehicle", "hero1", "hero2", "hero3"])
    parameters["ego_vehicle"] = {"role_name": role_name}
    
    parameters["bevfusion_ckpt"] = carla_bridge.get_param('bevfusion_ckpt', None)
    parameters["bevfusion_config"] = carla_bridge.get_param('bevfusion_config', None)
    parameters["calibrated_sensor_cfg"] = carla_bridge.get_param('calibrated_sensor_cfg', None)

    carla_bridge.loginfo("Trying to connect to {host}:{port}".format(
        host=parameters['host'], port=parameters['port']))

    try:
        carla_client = carla.Client(
            host=parameters['host'],
            port=parameters['port'])
        carla_client.set_timeout(parameters['timeout'])

        # check carla version
        dist = pkg_resources.get_distribution("carla")
        if LooseVersion(dist.version) != LooseVersion(CarlaRosBridge.CARLA_VERSION):
            carla_bridge.logfatal("CARLA python module version {} required. Found: {}".format(
                CarlaRosBridge.CARLA_VERSION, dist.version))
            sys.exit(1)

        if LooseVersion(carla_client.get_server_version()) != \
           LooseVersion(carla_client.get_client_version()):
            carla_bridge.logwarn(
                "Version mismatch detected: You are trying to connect to a simulator that might be incompatible with this API. Client API version: {}. Simulator API version: {}"
                .format(carla_client.get_client_version(),
                        carla_client.get_server_version()))

        carla_world = carla_client.get_world()

        if "town" in parameters and not parameters['passive']:
            if parameters["town"].endswith(".xodr"):
                carla_bridge.loginfo(
                    "Loading opendrive world from file '{}'".format(parameters["town"]))
                with open(parameters["town"]) as od_file:
                    data = od_file.read()
                carla_world = carla_client.generate_opendrive_world(str(data))
            else:
                if carla_world.get_map().name != parameters["town"]:
                    carla_bridge.loginfo("Loading town '{}' (previous: '{}').".format(
                        parameters["town"], carla_world.get_map().name))
                    carla_world = carla_client.load_world(parameters["town"])
            carla_world.tick()

        carla_bridge.initialize_bridge(carla_client.get_world(), parameters)
        carla_bridge.spin()

    except (IOError, RuntimeError) as e:
        carla_bridge.logerr("Error: {}".format(e))
    except KeyboardInterrupt:
        pass
    finally:
        roscomp.shutdown()
        del carla_world
        del carla_client


if __name__ == "__main__":
    main()
