DIR=/home/abuzarmahmood/Desktop/img_conv_net
docker run -v $DIR:$DIR --gpus all -it --rm tensorflow/tensorflow:latest-gpu bash
