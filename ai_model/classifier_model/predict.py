import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from PIL import Image
from pathlib import Path
from architecture import ResNet


transformations = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor()]
)

model_path = "./garbage.pth"

model = ResNet()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

from torchvision import transforms
from PIL import Image

def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    preprocessed_image = transform(image).unsqueeze(0)
    return preprocessed_image

def predict_image(model, preprocessed_image):
    with torch.no_grad():
        model.eval()
        output = model(preprocessed_image)
        _, predicted_class = torch.max(output, 1)
        return predicted_class

def predict_external_image(image_name, loaded_model):
    image = Image.open(Path("./" + image_name))

    example_image = transformations(image)
    print("The image resembles", predict_image(example_image, loaded_model) + ".")