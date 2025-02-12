import tensorflow as tf
import os
import visualkeras

# Load the model from the .h5 file using keras.models.load_model
current_dir = os.path.dirname(__file__)
model_file = os.path.join(current_dir, 'test.h5')
model = tf.keras.models.load_model(model_file)
model.build()
model.summary()
visualkeras.layered_view(model, to_file=os.path.join(current_dir, 'model_visualization.png'))