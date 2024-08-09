from keras import Sequential, layers, Model
from keras.models import load_model
from keras.optimizers import RMSprop

class TransferedModel:
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

        # Step 3: Add new layers on top of the base model
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
        # Implement confusion matrix logic here if needed
        pass
