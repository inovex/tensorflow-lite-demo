#!/bin/bash
# Absolute path to this script, e.g. /home/user/bin/foo.sh
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz -P $DIR
tar -C $DIR -xzvf $DIR/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz

rm $DIR/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz