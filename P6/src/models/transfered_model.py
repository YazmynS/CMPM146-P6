# models/transfered_model.py
#from tensorflow.keras import Sequential, layers
#from tensorflow.keras.models import load_model
#from tensorflow.keras.optimizers import RMSprop, Adam
from keras import Sequential, layers, Input
from keras.models import load_model, Model
from keras.optimizers import RMSprop, Adam


class TransferedModel(Model):
    def __init__(self, input_shape, categories_count):
        super().__init__(Input(shape=input_shape))
        self.base_model = load_model('testModel.keras')
        self._define_model(input_shape,categories_count)
        self._compile_model()

    def _define_model(self, input_shape, categories_count):
        
        # Step 1: Eliminate the final softmax layer
        self.base_model = Model(inputs=self.base_model.input, outputs=self.base_model.layers[-2].output)

        # Step 2: Freeze all parameters in the remainder
        for layer in self.base_model.layers:
            layer.trainable = False

        # Step 3: Bolt on one or more fully connected layers
        model = Sequential([
            self.base_model,
            layers.Dense(128, activation='relu'),   # Example of a fully connected layer
            layers.Dense(categories_count, activation='softmax')  # New softmax layer for classification
        ])

        # Assign the modified model to the instance variable
        self.model = model



    def _compile_model(self):
        # Your code goes here
        # you have to compile the keras model, similar to the example in the writeup
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )


    def print_summary(self):
        # Print the summary of the model
        self.model.summary()

    def train_model(self, train_dataset, validation_dataset, epochs):
        history = self.model.fit(
            x=train_dataset,
            epochs=epochs,
            verbose="auto",
            validation_data=validation_dataset
        )

        return history
