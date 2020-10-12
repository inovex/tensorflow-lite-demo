"""
Created by Robin Baumann <https://github.com/RobinBaumann> at October 08, 2020.
"""
import os

from absl import app
from absl import flags
import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import tensorflow as tf
import tensorflow_datasets as tfds

flags.DEFINE_integer('num_shards', 1,
                     'Number of files to spread the dataset across.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

CATEGORY_INDEX = label_map_util.create_category_index_from_labelmap(
    os.path.join(os.path.dirname(__file__), 'cars196_labelmap.pbtxt')
)


def create_tf_example(example):
  height, width = example['image'].shape[:-1]
  filename = b''  # Filename of the image. Empty if image is not from file
  encoded_image_data = tf.image.encode_jpeg(example['image']).numpy()  # Encoded image
  image_format = b'jpeg'

  ymins = [example['bbox'][0]]
  xmins = [example['bbox'][1]]
  ymaxs = [example['bbox'][2]]
  xmaxs = [example['bbox'][3]]

  # add 1 to label, since it was stored as [0, 195] instead of [1, 196]
  label = example['label'].numpy() + 1
  classes_text = [CATEGORY_INDEX[label][
                    'name'].encode('utf-8')]  # List of string class name of bounding box
  classes = [label]  # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))

  return tf_example


def write_sharded_tf_records(examples, output_filebase, num_shards):
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filebase, num_shards)
    for index, example in examples.enumerate():
      tf_example = create_tf_example(example)
      output_shard_index = index % num_shards
      output_tfrecords[output_shard_index].write(tf_example.SerializeToString())


def main(_):
  (train_examples, test_examples) = tfds.load(
      'cars196',
      split=['train', 'test']
  )

  train_filebase = os.path.join(FLAGS.output_path, 'train_cars196.record')
  test_filebase = os.path.join(FLAGS.output_path, 'test_cars196.record')
  write_sharded_tf_records(train_examples, train_filebase, FLAGS.num_shards)
  write_sharded_tf_records(test_examples, test_filebase, FLAGS.num_shards)


if __name__ == '__main__':
  app.run(main)
