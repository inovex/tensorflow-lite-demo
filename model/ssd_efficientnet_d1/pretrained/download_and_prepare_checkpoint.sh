  #!/bin/bash
# Absolute path to this script, e.g. /home/user/bin/foo.sh
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d1_coco17_tpu-32.tar.gz -P $DIR
tar -C $DIR -xzvf $DIR/efficientdet_d1_coco17_tpu-32.tar.gz

rm $DIR/efficientdet_d1_coco17_tpu-32.tar.gz