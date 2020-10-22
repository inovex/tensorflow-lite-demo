# tensorflow-lite-demo
Object detection model for car detection on mobile edge devices.


## Prepare checkpoints.

`./model/ssd_efficientdet_d7/pretrained/download_and_prepare_checkpoint.sh` 


## Prepare input data

    `python data/cars196_to_pascal_voc_format.py --output_path=data --num_shards=10`