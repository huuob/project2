import torch
import matplotlib.pyplot as plt
from model import SimpleCNN_Variant

def visualize_filters(path="model_adam_relu_ce.pth"):
    model = SimpleCNN_Variant()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    filters = model.conv1.weight.data

    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < filters.shape[0]:
            filter_img = filters[i].permute(1, 2, 0).numpy()
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())
            ax.imshow(filter_img)
            ax.axis('off')
    plt.suptitle("Conv1 Filters")
    plt.savefig("conv1_filters.png")
    plt.show()
