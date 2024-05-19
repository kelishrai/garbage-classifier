import torch
import os
from torchvision import transforms
from PIL import Image
import torch
from datetime import datetime

from prediction.architecture.arch import ResNet

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(os.getcwd())

from torchvision import transforms
from PIL import Image

def preprocess_image(image_path):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = os.getcwd() + "/static/image_"+timestamp+"." + "jpg"
    image.save(file_path, format=image.format)
    transform = transformations
    preprocessed_image = transform(image).unsqueeze(0)
    return preprocessed_image, file_path.split('/')[-1]

def predict_image(preprocessed_image, model_name, model_path):
    with torch.no_grad():
        model = ResNet(model_name) 
        if model_name == "18":
            model_path = os.path.join(os.getcwd(), "prediction/output-models/resnet18.pth")
        elif model_name == "34":
            model_path = os.path.join(os.getcwd(), "prediction/output-models/resnet34.pth")
        elif model_name == "50":
            model_path = os.path.join(os.getcwd(), "prediction/output-models/resnet50.pth")
        print(model_path)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        output = model(preprocessed_image)
        _, predicted_class = torch.max(output, 1)
        return predicted_class


