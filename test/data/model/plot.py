import tensorflow as tf
from tensorflow.keras.utils import plot_model
import os

# Load the model from the .pb file using tf.saved_model.load
current_dir = os.path.dirname(__file__)
model_file = os.path.join(current_dir, 'model.H5')
model = tf.keras.models.load_model(model_file)

# Plot the model
plot_model(model, to_file=os.path.join(current_dir, 'output/model_plot.png'), show_shapes=True, show_layer_names=True)
