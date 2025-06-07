import matplotlib.pyplot as plt

def visualize_loss_curves(loss_lists, labels):
    for losses, label in zip(loss_lists, labels):
        max_curve = [max(x) for x in zip(*losses)]
        min_curve = [min(x) for x in zip(*losses)]
        plt.fill_between(range(len(max_curve)), min_curve, max_curve, alpha=0.3, label=label)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Loss Landscape Comparison")
    plt.legend()
    plt.savefig("loss_landscape_comparison.png")
    plt.show()
