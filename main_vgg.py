from vgg import VGG_A, VGG_BN
from train_vgg import train_vgg
from visualize_vgg import plot_curves

if __name__ == "__main__":
    losses_A, accs_A = train_vgg(VGG_A(), "VGG_A")
    losses_BN, accs_BN = train_vgg(VGG_BN(), "VGG_BN")

    plot_curves([losses_A, losses_BN], [accs_A, accs_BN], labels=["VGG_A", "VGG_BN"])
