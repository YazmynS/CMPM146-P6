from keras import Sequential, layers, Model
from keras.models import load_model
from keras.optimizers import RMSprop
import numpy as np
from sklearn.metrics import confusion_matrix

class RandomModel:
    def __init__(self, input_shape, categories_count):
        self.base_model = load_model('testModel.keras')
        self.model = self._define_model(input_shape, categories_count)
        self._compile_model()

    def _define_model(self, input_shape, categories_count):
        # Step 1: Remove the last softmax layer from the pre-trained model
        self.base_model = Model(inputs=self.base_model.input, outputs=self.base_model.layers[-2].output)

        # Step 2: Freeze the base model layers
        for layer in self.base_model.layers:
            layer.trainable = False

        # Step 3: Randomize the weights of the base model
        self._randomize_layers(self.base_model)

        # Step 4: Add new layers on top of the base model
        model = Sequential([
            self.base_model,
            layers.Dense(128, activation='relu'),
            layers.Dense(categories_count, activation='softmax')
        ])

        return model

    def _compile_model(self):
        # Compile the model
        self.model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def _randomize_layers(self, model):
        for layer in model.layers:
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                original_shape = layer.kernel.shape
                random_weights = np.random.standard_normal(original_shape)
                layer.kernel.assign(random_weights)
            if hasattr(layer, 'bias') and layer.bias is not None:
                original_shape = layer.bias.shape
                random_bias = np.random.standard_normal(original_shape)
                layer.bias.assign(random_bias)

    def print_summary(self):
        self.model.summary()

    def train_model(self, train_dataset, validation_dataset, epochs):
        return self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=validation_dataset
        )

    def evaluate(self, test_dataset):
        return self.model.evaluate(test_dataset)

    def get_confusion_matrix(self, test_dataset):
        prediction = self.model.predict(test_dataset)
        labels = np.concatenate([y for x, y in test_dataset], axis=0)
        y_pred = np.argmax(prediction, axis=-1)
        y = np.argmax(labels, axis=-1)
        return confusion_matrix(y, y_pred)
