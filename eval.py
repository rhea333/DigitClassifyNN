import torch

def evaluate_model(model, test_loader):
    model.eval() 
    correct = 0
    total = 0
    with torch.no_grad(): 
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')
