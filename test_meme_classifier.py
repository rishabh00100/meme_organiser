"""
	Script to train a classification model to classify an image as:
		- yes_meme
		- no_meme

	Code source: 
	https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5

	This code was modified to run on Google Colab
"""

import os, sys
import matplotlib.pyplot as plt
import argparse

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import warnings
import time
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Inference script for meme classification')
parser.add_argument('--path', default='meme_classifier_openSource.pth',
                    help='name of the model .pth file. Ex: model1.pth')

args = parser.parse_args()

def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

def get_random_images(num):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes
    # print("data.classes", data.classes)
    # print("data.class_to_idx", data.class_to_idx)
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, 
                   sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    return images, labels, classes

cwd = os.path.abspath(os.getcwd())
data_dir = os.path.join(cwd, "resource/test_imgs")

# Load model in eval mode for testing
test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                     ])
print("[*] Loading Model")
start_time = time.time()

try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model=torch.load(os.path.join(cwd, args.path))
    else:
        device = torch.device("cpu")
        model=torch.load(os.path.join(cwd, args.path), map_location='cpu')
except Exception as e:
    print(e)
    link = "https://drive.google.com/file/d/1oU1E5LyGQVKDZzgueycjdE15W6kaVxet/view"
    print("Model not found. Please download the model from {} and save it in the $ROOT as meme_classifier_openSource.pth".format(link))
    sys.exit()

model.to(device)
model.eval()
print("[*] Model successfully loaded in {:.2f} secs".format(time.time() - start_time))

# Run model predictions on random images
num_imgs = 25
to_pil = transforms.ToPILImage()
images, labels, classes = get_random_images(num_imgs)
fig=plt.figure(figsize=(20,20))
counter = 0
time_pred = 0
while counter < num_imgs:
    row_num = 1
    for ii in range(5):
        image = to_pil(images[counter])
        pred_start_time = time.time()
        index = predict_image(image)
        pred_time = time.time() - pred_start_time
        time_pred += pred_time
        sub = fig.add_subplot(5, 5, counter+1)
        res = int(labels[counter]) == index
        sub.set_title("Annotation: " + str(classes[index]) + "\nPredicted correctly:" + str(res))
        plt.axis('off')
        plt.imshow(image)
        counter += 1
    row_num += 1

print("[*] Average time of prediction: {:.2f} secs / image".format(time_pred/25))
path = os.path.join(cwd, 'output.png')
fig.savefig(os.path.join(cwd, 'output.png'), bbox_inches='tight')
plt.close(fig)
print("[*] Output saved at: {}".format(path))