import torch
import torch.nn as nn
import torch.nn.functional as F
from image_preprocess import dataset
from architecture import ResNet18, ResNet50, ResNet34
from sklearn.metrics import confusion_matrix as cm

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))



class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        preds = torch.argmax(out, dim=1)  # Get predicted labels
        return {"val_loss": loss.detach(), "val_acc": acc, "val_preds": preds, "val_targets": labels}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        
        # Combine predictions and targets
        batch_preds = [x["val_preds"] for x in outputs]
        batch_targets = [x["val_targets"] for x in outputs]
        preds = torch.cat(batch_preds, dim=0)
        targets = torch.cat(batch_targets, dim=0)
        
        # Calculate confusion matrix
        confusion_matrix = cm(targets.cpu().numpy(), preds.cpu().numpy(), labels=range(len(dataset.classes)))
        
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item(), "confusion_matrix": confusion_matrix}

    def epoch_end(self, epoch, result):
        print(
            "Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch + 1, result["train_loss"], result["val_loss"], result["val_acc"]
            )
        )


class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = ResNet34()
        self.network = ResNet18()
        

    def forward(self, xb):
        x = self.network(xb)
        return F.relu(x)