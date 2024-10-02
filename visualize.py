import matplotlib.pyplot as plt
import torch

def visualize_predictions(model, test_loader):
    model.eval()
    images, labels = next(iter(test_loader))
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    fig = plt.figure(figsize=(10, 10))
    for idx in range(9):
        ax = fig.add_subplot(3, 3, idx + 1)
        ax.imshow(images[idx].numpy().squeeze(), cmap='gray')
        ax.set_title(f'Predicted: {predicted[idx].item()}')
        ax.axis('off')
    plt.show()

