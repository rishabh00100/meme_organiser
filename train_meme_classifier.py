"""
	Script to train a classification model to classify an image as:
		- Meme
		- Non Meme

	Code source: 
	https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5

	This code was modified to run on Google Colab
"""

# Mount the Google drive base directory on the instance of Google Colab
from google.colab import drive
drive.mount('/content/drive')

# Code block to extract the ZIP file containg the training dataset
import zipfile
zip_ref = zipfile.ZipFile("/content/drive/My Drive/sample_dataset.zip", 'r')
zip_ref.extractall("/content/drive/My Drive/sample_dataset/")
zip_ref.close()
print("completed")

# Library imports
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def load_split_train_test(datadir, valid_size = .2):
	train_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),])
	test_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),])    
	train_data = datasets.ImageFolder(datadir,       
					transform=train_transforms)
	test_data = datasets.ImageFolder(datadir,
					transform=test_transforms)   
	num_train = len(train_data)
	indices = list(range(num_train))
	split = int(np.floor(valid_size * num_train))
	np.random.shuffle(indices)
	from torch.utils.data.sampler import SubsetRandomSampler
	train_idx, test_idx = indices[split:], indices[:split]
	train_sampler = SubsetRandomSampler(train_idx)
	test_sampler = SubsetRandomSampler(test_idx)
	trainloader = torch.utils.data.DataLoader(train_data,
				   sampler=train_sampler, batch_size=64)
	testloader = torch.utils.data.DataLoader(test_data,
				   sampler=test_sampler, batch_size=64)
	return trainloader, testloader
	

data_dir = '/content/drive/My Drive/sample_dataset/sample_dataset'

trainloader, testloader = load_split_train_test(data_dir, .2)
print("[*] Following classes were found in the training dataset: {}".format(trainloader.dataset.classes))

#---------------------------------------------------------------
#						Load the model
#---------------------------------------------------------------

# Choose CUDA to train if available, else choose CPU
device = torch.device("cuda" if torch.cuda.is_available() 
								  else "cpu")
model = models.resnet50(pretrained=True)
print("[*] Model architecture\n")
print(model)

# Freeze the pre-trained layers
for param in model.parameters():
	param.requires_grad = False

# Setup the fully connected layer for transfer learning
model.fc = nn.Sequential(nn.Linear(2048, 512),
								 nn.ReLU(),
								 nn.Dropout(0.2),
								 nn.Linear(512, 10),
								 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)

#---------------------------------------------------------------
#						Train the model
#---------------------------------------------------------------

epochs = 20
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []

for epoch in range(epochs):
	for inputs, labels in trainloader:
		steps += 1
		inputs, labels = inputs.to(device), labels.to(device)
		optimizer.zero_grad()
		logps = model.forward(inputs)
		loss = criterion(logps, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		
		if steps % print_every == 0:
			test_loss = 0
			accuracy = 0
			model.eval()
			with torch.no_grad():
				for inputs, labels in testloader:
					inputs, labels = inputs.to(device), labels.to(device)
					logps = model.forward(inputs)
					batch_loss = criterion(logps, labels)
					test_loss += batch_loss.item()
					
					ps = torch.exp(logps)
					top_p, top_class = ps.topk(1, dim=1)
					equals = top_class == labels.view(*top_class.shape)
					accuracy = accuracy + torch.mean(equals.type(torch.FloatTensor)).item()
			train_losses.append(running_loss/len(trainloader))
			test_losses.append(test_loss/len(testloader))                    
			print(f"Epoch {epoch+1}/{epochs}.. "
				  f"Train loss: {running_loss/print_every:.3f}.. "
				  f"Test loss: {test_loss/len(testloader):.3f}.. "
				  f"Test accuracy: {accuracy/len(testloader):.3f}")
			running_loss = 0
			model.train()

# Save the trained model to local directory
torch.save(model, '/content/drive/My Drive/meme_classifier.pth')









