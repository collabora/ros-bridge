FROM ros:noetic-ros-base

# setup ros bridge
RUN mkdir -p /opt/carla-ros-bridge/src
WORKDIR /opt/carla-ros-bridge/

COPY requirements.txt /opt/carla-ros-bridge/
COPY install_dependencies.sh /opt/carla-ros-bridge/

RUN /bin/bash -c 'source /opt/ros/noetic/setup.bash; \
                 rosdep update'
RUN /bin/bash -c 'source /opt/ros/noetic/setup.bash; \
                  bash /opt/carla-ros-bridge/install_dependencies.sh'

COPY . /opt/carla-ros-bridge/src

RUN /bin/bash -c 'source /opt/ros/noetic/setup.bash && \
                  catkin_make install'

WORKDIR /studio

CMD ["caddy", "file-server", "--listen", ":8080"]