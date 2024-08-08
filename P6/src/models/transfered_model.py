# models/transfered_model.py

from models.model import Model
#from tensorflow.keras import Sequential, layers
#from tensorflow.keras.models import load_model
#from tensorflow.keras.optimizers import RMSprop, Adam
from keras import Sequential, layers
from keras.models import load_model
from keras.optimizers import RMSprop, Adam

class TransferedModel(Model):
    def __init__(self, input_shape, categories_count):
        super().__init__(input_shape, categories_count)
        self.base_model = load_model('model.keras')

    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model for transfer learning
        for layer in self.base_model.layers[:-1]:
            layer.trainable = False

        model = Sequential(self.base_model.layers[:-1])
        model.add(layers.Dense(categories_count, activation='softmax'))
        self.model = model

    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
