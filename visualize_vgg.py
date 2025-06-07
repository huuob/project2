import matplotlib.pyplot as plt

def plot_curves(losses, accs, labels):
    plt.figure()
    for l, label in zip(losses, labels):
        plt.plot(l, label=f"{label} Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("vgg_loss_compare.png")

    plt.figure()
    for a, label in zip(accs, labels):
        plt.plot(a, label=f"{label} Acc")
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig("vgg_accuracy_compare.png")
