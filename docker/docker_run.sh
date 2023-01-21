xhost +local:root
xhost +local:docker

docker run --rm -it \
    --user root \
    --name 'llio' \
    --net=host \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:ro \
    -e DISPLAY=unix$DISPLAY \
    --volume=$(pwd)/../:/project \
    --volume=/media/gskim/Ext_SSD/data/kitti/raw:/data \
    pypose:llio \
    /bin/bash -c 'cd /project/python; python3 main.py; bash'
