from models.model import Model
from keras import Sequential, layers, models
from keras.layers.preprocessing import Rescaling
from keras.optimizers import RMSprop, Adam

class TransferedModel(Model):
    def _define_model(self, input_shape, categories_count): #Yazmyn Claimed Task
        # Your code goes here
        # you have to initialize self.model to a keras model
        # load your basic model with keras's load_model function
        # freeze the weights of the loaded model to make sure the training doesn't affect them
        # (check the number of total params, trainable params and non-trainable params in your summary generated by train_transfer.py)
        # use this model by removing the last layer, adding dense layers and an output layer
         # Load the pre-trained basic model
        base_model = models.load_model('path_to_your_saved_basic_model.h5')
        
        # Freeze the layers of the base model to prevent them from being trained
        for layer in base_model.layers:
            layer.trainable = False
        
        # Remove the last layer of the base model
        base_model = models.Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
        
        # Add new dense layers and the output layer
        self.model = Sequential([
            base_model,
            layers.Dense(64, activation='relu'),
            layers.Dense(categories_count, activation='softmax')
        ])
    
    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        pass
