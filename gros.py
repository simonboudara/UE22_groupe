import tensorflow as tf

base_model = tf.keras.applications.VGG16(input_shape=(360,363,3), include_top=False, weights='imagenet')
base_model.summary()