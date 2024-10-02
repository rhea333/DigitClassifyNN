import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loaders
from model import SimpleNN
from train import train_model
from evaluate import evaluate_model
from visualize import visualize_predictions

if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders()

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, epochs=5)

    evaluate_model(model, test_loader)

    visualize_predictions(model, test_loader)
