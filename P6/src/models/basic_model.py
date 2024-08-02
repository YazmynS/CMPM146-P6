from models.model import Model
from keras import Sequential, layers
from keras.layers.preprocessing import Rescaling
from keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count): #Yazmyn Claimed Task
        # you have to initialize self.model to a keras model
        self.model = Sequential([
            # Normalize the input data to the range [0, 1]
            Rescaling(1./255, input_shape=input_shape),
            
            # First convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
            layers.Conv2D(32, (3, 3), activation='relu'),
            # Max pooling layer to reduce spatial dimensions
            layers.MaxPooling2D((2, 2)),
            
            # Second convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
            layers.Conv2D(64, (3, 3), activation='relu'),
            # Max pooling layer to reduce spatial dimensions
            layers.MaxPooling2D((2, 2)),
            
            # Third convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
            layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Flatten the 2D feature maps into a 1D vector
            layers.Flatten(),
            # Fully connected layer with 64 units and ReLU activation
            layers.Dense(64, activation='relu'),
            # Output layer with softmax activation for classification
            layers.Dense(categories_count, activation='softmax')
        ])
    
    def _compile_model(self): 
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        pass