import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import os
print("parth",os.getcwd())
data_dir = "../garbage-large/"

classes = os.listdir(data_dir)

from torchvision import transforms

transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


dataset = ImageFolder(data_dir, transform=transformations)
print(len(dataset.classes))
print(dataset.classes)

def count_images_in_subfolders(root_folder):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    counts = []

    for subfolder in subfolders:
        image_count = count_images_in_folder(subfolder)
        counts.append((os.path.basename(subfolder), image_count))
        print(f"Subfolder: {os.path.basename(subfolder)}, Number of Images: {image_count}")

    total_images = sum(count for _, count in counts)
    print(f"Total number of images: {total_images}")

    # Plotting
    plot_counts(counts)

def count_images_in_folder(folder):
    image_count = sum(1 for entry in os.scandir(folder) if entry.is_file())
    return image_count

def plot_counts(counts):
    subfolders, image_counts = zip(*counts)
    plt.figure(figsize=(12, 6))  
    plt.bar(subfolders, image_counts)
    plt.xlabel('Subfolders')
    plt.ylabel('Number of Images')
    plt.title('Image Count in Subfolders')
    plt.xticks(rotation=45, ha='right', fontsize=8)  
    plt.tight_layout()  
    plt.savefig("dataset.png")

if __name__ == "__main__":
    root_folder = data_dir # Replace with the path to your folder
    count_images_in_subfolders(root_folder)