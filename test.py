import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image

if __name__ == '__main__':
    ''''
    # 1. Prepare the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    '''

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 2. Define the model
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    '''
    #testing changes
    net = Net()

    # 3. Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 4. Training loop
    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # 5. Evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
    torch.save(net.state_dict(), 'model.pth')
'''
    
    net = Net()
    # Load the saved model
    net.load_state_dict(torch.load('model.pth'))
    net.eval()

    input_size = next(net.parameters()).shape[1:]
    print(next(net.parameters()).shape[1:], 'param')

    # Define the transformations for preprocessing the image
    transform = transforms.Compose([
        transforms.Resize((input_size[1:])),  # Resize the image to match the input size of the model
        transforms.ToTensor(),         # Convert the image to a tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
    ])

    # Load the image
    image_path = 'boat.jpg'  # Provide the path to your image
    image = Image.open(image_path)

    # Preprocess the image
    input_image = transform(image).unsqueeze(0)  # Add a batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = net(input_image)
        print(outputs)

    # Interpret the predictions
    _, predicted = torch.max(outputs, 1)
    predicted_class = classes[predicted.item()]

    # Get the prediction confidence (probability) for the predicted class
    softmax = nn.Softmax(dim=1)
    probabilities = softmax(outputs)
    prediction_confidence = probabilities[0, predicted.item()].item() * 100

    print("Predicted class:", predicted_class)
    print("Prediction confidence: {:.2f}%".format(prediction_confidence))

    print('Predicted class:', predicted_class)