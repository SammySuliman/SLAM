import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from models.pointnet_classifier import PointNetClassifier  # Assuming your model is defined in a separate file
from ModelNet40 import ModelNet40

# Define your model architecture
model = PointNetClassifier()

# Define the additional layer
additional_layer = nn.Sequential(
    nn.BatchNorm1d(40), # Assuming the input size is 40 from the previous layer
    nn.ReLU(),
    nn.Dropout(0.7),
    nn.Linear(40, 2))

# Load the pre-trained weights
pretrained_weights_path = 'models/weights.pth'
pretrained_dict = torch.load(pretrained_weights_path, map_location=torch.device('cpu'))

# Load the weights into your model
model.load_state_dict(pretrained_dict)

# Optionally, freeze layers if needed
for name, param in model.named_parameters():
    param.requires_grad = False

# Append the additional layer to the existing classifier
model.classifier.add_module('additional_layer', additional_layer)
# -----------------------------------------------------------------------------

# Define transforms to preprocess the data
transform = transforms.Compose([
    transforms.ToTensor(),
    ])

# Specify the path to your dataset
training_path = "preprocessing/OFF_files/train"
validation_path = "preprocessing/OFF_files/test"

# Create a DataLoader for training
training_set = ModelNet40(training_path)
validation_set = ModelNet40(validation_path, test=True)
training_loader = DataLoader(training_set, batch_size = 32, shuffle = True)
validation_loader = DataLoader(validation_set, batch_size = 32)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
parameters = list(model.parameters())
optimizer = optim.Adam(parameters)
running_vloss = 0

def train_one_epoch(epoch_index, tb_writer):
    print('new epoch')
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, (inputs, labels) in enumerate(training_loader):
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs[0], labels.float())
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 937 == 0:
            last_loss = running_loss / 937 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
    return last_loss
    
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = criterion(voutputs[0], vlabels.float())
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_output/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    # Disable gradient computation and reduce memory consumption.
    epoch_number += 1