import torch
from torch.optim import SGD
from shared.utils import Net, load_data, test, train


def main():
    # Hyper-parameters
    num_epochs = 10
    learning_rate = 0.2
    batch_size = 64

    print("Centralized PyTorch training")
    print("Loading data...")
    trainloader, testloader, _ = load_data(batch_size=batch_size)

    # Selecting model and optimizer
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(DEVICE)
    print(f"Training model for {num_epochs} epochs.")
    optimizer = SGD(params=net.parameters(), lr=learning_rate)

    # Train model
    loss, accuracy = train(
        net=net,
        trainloader=trainloader,
        optimizer=optimizer,
        device=DEVICE,
        epochs=num_epochs,
    )
    print(f"Final Training - Loss: {loss:0.4f} | Accuracy: {accuracy:04f}")

    # Evaluate model
    print("Evaluate model")
    loss, accuracy = test(net=net, testloader=testloader, device=DEVICE)
    print(f"Final Validation - Loss: {loss:0.4f} | Accuracy: {accuracy:04f}")


if __name__ == "__main__":
    main()
