xhost +local:root
xhost +local:docker

docker run --rm -it \
    --user root \
    --name 'pypose_imu' \
    --net=host \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -e DISPLAY=unix$DISPLAY \
    --volume=$(pwd)/../:/project \
    --volume=/media/gskim/Ext_SSD/data/kitti/raw:/data \
    pypose:imu \
    /bin/bash -c 'cd /project/python; python3 main.py; bash'
