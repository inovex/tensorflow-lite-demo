"""
Created by Robin Baumann <mail@robin-baumann.com> at 29.04.20.
"""


import tensorflow as tf
import tensorflow_model_optimization as tfmot


def get_model(in_shape, num_classes, qat=True):

    base_model = tf.keras.applications.EfficientNetB7(include_top=False,
                                                      input_shape=in_shape,
                                                      weights='imagenet')
    outputs = tf.keras.layers.Dense(num_classes,
                                    activation='softmax')(base_model.output)
    model = tf.keras.models.Model(inputs=base_model.inputs,
                                  outputs=outputs)

    if qat:
        model = tfmot.quantization.keras.quantize_model(model)

    return model
