from preprocess import get_transfer_datasets
from models.transfered_model import TransferedModel
from models.random_model import RandomModel
from config import image_size, categories
import matplotlib.pyplot as plt
import time

input_shape = (150, 150, 3)
categories_count = 2

models = {
    'transfered_model': TransferedModel,
    'random_model': RandomModel,
}

def plot_history_diff(initial_hist, transfered_hist):
    val_acc_initial = initial_hist.history['val_accuracy']
    val_acc_tranfered = transfered_hist.history['val_accuracy']

    epochs_initial = range(1, len(val_acc_initial) + 1)
    epochs_transfered = range(1, len(val_acc_initial) + 1)
    assert epochs_initial == epochs_transfered, "The two models have been tried with different epochs"

    plt.figure(figsize=(24, 6))
    plt.plot(epochs_initial, val_acc_initial, 'b', label='Initial Model Accuracy')
    plt.plot(epochs_initial, val_acc_tranfered, 'r', label='Transfered Model Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()

if __name__ == "__main__":
    epochs = 3
    print('* Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_transfer_datasets()
    histories = []
    for name, model_class in models.items():
        print(f'* Training {name} for {epochs} epochs')
        model = model_class(input_shape, categories_count)
        model.print_summary()
        history = model.train_model(train_dataset, validation_dataset, epochs)
        histories.append(history)
        print(f'* Evaluating {name}')
        model.evaluate(test_dataset)
        print(f'* Confusion Matrix for {name}')
        print(model.get_confusion_matrix(test_dataset))
    assert len(histories) == 2, "The number of trained models is not equal to two"
    plot_history_diff(*histories)
