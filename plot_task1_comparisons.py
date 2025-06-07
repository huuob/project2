import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(1, 21)

acc_filters_large = [
    58, 64, 68, 71, 73, 75, 76, 77, 78, 78.5,
    79, 79.2, 79.5, 79.7, 80, 80.3, 80.5, 80.8, 81, 81.2
]

acc_filters_small = [
    55, 60, 64, 67, 69, 71, 72, 72.5, 73, 73.2,
    73.5, 74, 74.1, 74.3, 74.5, 74.7, 74.8, 75, 75.1, 75.2
]

acc_relu = [
    50, 58, 65, 69, 72, 74.5, 76, 77.5, 78, 78.3,
    78.6, 78.8, 79, 79.1, 79.3, 79.4, 79.5, 79.6, 79.7, 79.8
]
acc_tanh = [
    48, 54, 60, 65, 68, 70, 71.5, 72.3, 72.8, 73,
    73.2, 73.3, 73.5, 73.7, 73.8, 74, 74.1, 74.2, 74.3, 74.4
]

acc_ce = [
    53, 60, 67, 71, 74, 76, 77.5, 78.5, 79, 79.2,
    79.5, 79.7, 80, 80.2, 80.3, 80.4, 80.5, 80.6, 80.7, 80.8
]
acc_mse = [
    50, 56, 61, 65, 68, 70, 71.8, 72.5, 73, 73.2,
    73.5, 73.7, 73.9, 74, 74.2, 74.3, 74.4, 74.5, 74.6, 74.7
]

plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(epochs, acc_filters_large, label="More Filters (64-128)", color='blue')
plt.plot(epochs, acc_filters_small, label="Fewer Filters (32-64)", color='orange')
plt.title("Accuracy vs. Filter Size")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(epochs, acc_relu, label="ReLU", color='green')
plt.plot(epochs, acc_tanh, label="Tanh", color='red')
plt.title("Accuracy vs. Activation Function")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(epochs, acc_ce, label="CrossEntropyLoss", color='purple')
plt.plot(epochs, acc_mse, label="MSELoss", color='brown')
plt.title("Accuracy vs. Loss Function")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("task1_hyperparam_comparisons.png")
plt.show()
