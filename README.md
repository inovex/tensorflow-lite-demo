# tensorflow-lite-demo
Object detection model for car detection using the [cars196 dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).

This repository contains the accompanying code for the blog post [Deep Learning for mobile devices with TensorFlow Lite: Custom Object Detector]() on the inovex blog.

## How to get started

I highly recommend reading the blog post, since it gives a concise overview of all the steps that are required to train an object detection model using the TensorFlow Object Detection API. However, if you are just interested in the code, you can start with the following steps.
 
### Prerequisites

  * Python>=3.7
  * TensorFlow >= 2.3.0
  * TensorFlow Datasets >= 2.1.0
  * ProtoBuf Compiler (protoc) >= 3.0
 
### Install the TensorFlow Object Detection API

The TF Object Detection API is integrated as a Git submodule in this repository. You can install it by issuing the following commands:

  1. Initialize the submodule: `git submodule init`
  2. Update the submodule: `git submodule update`
  3. Move to the object detection API: `cd od-api/model/research`
  4. Install the API:
  
    # Compile protos.
    protoc object_detection/protos/*.proto --python_out=.
    # Install TensorFlow Object Detection API.
    cp object_detection/packages/tf2/setup.py .
    python -m pip install --use-feature=2020-resolver .
    
    
You can test your installation via:

    python object_detection/builders/model_builder_tf2_test.py

### Prepare checkpoints

Training the model from a fine-tuning checkpoint requires you to download and extract a checkpoint matching the model architecture. You can do so by running the following script: 

`./model/ssd_mobilenet_v2/pretrained/download_and_prepare_checkpoint.sh` 


### Prepare input data

The TensorFlow Object Detection API requires the data to be stored in a specific structure. The following script downloads the cars196 dataset and transforms it into a structure that can be read by the object detection API.

`python data/cars196_to_pascal_voc_format.py --output_path=data --num_shards=10`

### Start model training

Issue the following command from `od-api/models/research`:

    PIPELINE_CONFIG_PATH=../../../model/ssd_mobilenet_v2/ssd_mobilenet_v2.config
    MODEL_DIR={Wherever/you/want/your/model/to/be/saved}
    python object_detection/model_main_tf2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --alsologtostderr
    

