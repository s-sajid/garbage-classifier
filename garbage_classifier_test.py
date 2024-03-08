#!/usr/bin/env python
# coding: utf-8

import torch
import glob
import matplotlib.pylab as plt
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from PIL import Image
from torchvision.models import resnet18
from torchvision import transforms, models
from sklearn.model_selection import StratifiedShuffleSplit

import warnings
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    print("CUDA is available on this machine.")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Found CUDA-enabled device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available on this machine.")


class TorchVisionDataset(Dataset):
    def __init__(self, data_dic, transform=None):
        self.file_paths = data_dic["X"]
        self.labels = data_dic["Y"]
        self.transform = transform
        self.corrupted_count = 0

    def __len__(self):
        return len(self.file_paths)

    
    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_paths[idx]

        try:
            image = Image.open(file_path)

            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            return image, label
        
        except Exception as e:
            self.corrupted_count += 1
            print(self.corrupted_count)
            return torch.zeros(3, 224, 224), label


# images = glob.glob("data/*/*/*")
images = glob.glob("test-data/*/*")
images = np.array(images)
labels = np.array([f.split("\\")[-2] for f in images])

print(len(images))
print(labels.size)
print(images[0])
print(labels[0])

classes = np.unique(labels).flatten()
print(classes)
labels_int = np.zeros(labels.size, dtype=np.int64)
print(labels_int.shape)

for ii, jj in enumerate(classes):
    labels_int[labels == jj] = ii 


for ii in range(4):
    print("Label ", ii, ":", (labels_int == ii).sum())


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=10)
sss.get_n_splits(images, labels_int)
dev_index, test_index = next(sss.split(images, labels_int))

dev_images = images[dev_index]
dev_labels = labels_int[dev_index]

test_images = images[test_index]
test_labels = labels_int[test_index] 


sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=10)
sss2.get_n_splits(dev_images, dev_labels)
train_index, val_index = next(sss2.split(dev_images, dev_labels))

train_images = images[train_index]
train_labels = labels_int[train_index]

val_images = images[val_index]
val_labels = labels_int[val_index] 


print("Train set:", len(train_images))
print("Val set:", len(val_images))
print("Test set:", len(test_images))


train_set = {"X": train_images, "Y": train_labels}
val_set = {"X": val_images, "Y": val_labels}
test_set = {"X": test_images, "Y": test_labels}


torchvision_transform = transforms.Compose([transforms.Resize((224,224)),\
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.ToTensor(),transforms.Normalize(mean=[0.4120, 0.3768, 0.3407],std=[0.2944,0.2759,0.2598])])

torchvision_transform_test = transforms.Compose([transforms.Resize((224,224)),\
    transforms.ToTensor(),transforms.Normalize(mean=[0.4120, 0.3768, 0.3407],std=[0.2944,0.2759,0.2598])])


train_dataset = TorchVisionDataset(train_set, transform=torchvision_transform)
val_dataset = TorchVisionDataset(val_set, transform=torchvision_transform)
test_dataset = TorchVisionDataset(test_set, transform=torchvision_transform_test)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=0)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=0)

def get_dataset_stats(data_loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in data_loader:
        data = data[0]  # Get the images to compute the statistics
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std


print(get_dataset_stats(trainloader))


train_iterator = iter(trainloader)
train_batch = next(train_iterator)


print(train_batch[0].size())
print(train_batch[1].size())


plt.figure()
plt.imshow(train_batch[0].numpy()[16].transpose(1,2,0))
plt.show()


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


net = ResNet18(4)
net.to(device)


criterion = nn.CrossEntropyLoss() # Loss function
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
scheduler = ExponentialLR(optimizer, gamma=0.9)


nepochs = 10
PATH = './resnet18.pth' # Path to save the best model

best_loss = 1e+20
patience = 3  # Number of epochs to wait before early stopping
count = 0

for epoch in range(nepochs):  # loop over the dataset multiple times
    # Training Loop
    train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)  # Move data to GPU
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    print(f'{epoch + 1},  train loss: {train_loss / i:.3f},', end=' ')
    scheduler.step()
    
    val_loss = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(valloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)  # Move data to GPU
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
        print(f'val loss: {val_loss / (i + 1):.3f}') 
        
        # Early stopping
        if val_loss < best_loss:
            print("Saving model")
            torch.save(net.state_dict(), PATH)
            best_loss = val_loss
            count = 0  # Reset count if we found a better model
        else:
            count += 1
            if count >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
        
print('Finished Training')

net = ResNet18(4)
net.load_state_dict(torch.load(PATH))


class_correct = [0 for _ in range(4)]
class_total = [0 for _ in range(4)]
overall_correct = 0
overall_total = 0

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1
            overall_correct += (predicted[i] == label).item()
            overall_total += 1

for i in range(4):
    accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f'Accuracy of class {i}: {accuracy:.2f} %')

overall_accuracy = 100 * overall_correct / overall_total if overall_total > 0 else 0
print(f'Overall accuracy: {overall_accuracy:.2f} %')


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Initialize variables to store true labels and predicted labels
true_labels = []
predicted_labels = []

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.numpy())
        predicted_labels.extend(predicted.numpy())

classes = ['Black', 'Blue', 'Green', 'Other']

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Create confusion matrix display
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap=plt.cm.Greens)
plt.title('Confusion Matrix')
plt.show()
