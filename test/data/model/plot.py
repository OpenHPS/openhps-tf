import tensorflow as tf
from tensorflow.keras.utils import plot_model
import os

# Load the model from the .pb file using TFSMLayer
current_dir = os.path.dirname(__file__)
model_file = os.path.join(current_dir, 'output/')
model = tf.keras.Sequential([
    tf.keras.layers.TFSMLayer(model_file, call_endpoint='serving_default')
])
model.build()

# Plot the model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)