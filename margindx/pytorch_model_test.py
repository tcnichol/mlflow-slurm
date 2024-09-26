import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor
from dotenv import load_dotenv
import mlflow.sklearn
load_dotenv()
import mlflow
import sys
import os
from torchvision.transforms import transforms
from PIL import Image

from mlflow.server import get_app_client

mlflow.set_tracking_uri(uri="https://mlflow.margindx.software-dev.ncsa.illinois.edu")
mlflow.set_experiment("MLflow Pytorch Model Testing")

test_image_path = os.path.join(os.getcwd(), 'margindx', 'image1.png')

# tracking_uri = "https://mlflow.margindx.software-dev.ncsa.illinois.edu/"
#
# auth_client = get_app_client("basic-auth", tracking_uri=tracking_uri)
# auth_client.create_user(username="margindx", password="noghie8bai1Foopi0xah0faib8Achi")
# auth_client.update_user_admin(username="margindx", is_admin=True)

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

test_dataloader = DataLoader(test_data, batch_size=64)

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(10),  # 10 classes in total.
        )

    def forward(self, x):
        return self.model(x)

device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = nn.CrossEntropyLoss()
metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
model = ImageClassifier().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def evaluate(dataloader, model, loss_fn, metrics_fn, epoch):
    """Evaluate the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the eval data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        loss_fn: a callable, the loss function.
        metrics_fn: a callable, the metrics function.
        epoch: an integer, the current epoch number.
    """
    num_batches = len(dataloader)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            eval_loss += loss_fn(pred, y).item()
            eval_accuracy += metrics_fn(pred, y)

    eval_loss /= num_batches
    eval_accuracy /= num_batches
    mlflow.log_metric("eval_loss", f"{eval_loss:2f}", step=epoch)
    mlflow.log_metric("eval_accuracy", f"{eval_accuracy:2f}", step=epoch)

    print(f"Eval metrics: \nAccuracy: {eval_accuracy:.2f}, Avg loss: {eval_loss:2f} \n")


# Set our tracking server uri for logging
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080"
# mlflow.set_experiment("/mlflow-pytorch-quickstart")

first_model_path = os.path.join(os.getcwd(), 'margindx',"model_1.pth")
second_model_path = os.path.join(os.getcwd(), 'margindx', "model_2.pth")
# test_model_1 = mlflow.pyfunc.load_model(first_model_path)
model_1 = torch.load(first_model_path)
evaluate(test_dataloader, model_1, loss_fn, metric_fn, epoch=0)
print("How was first model?")

print("How was second modal?")
loaded_model_2 = torch.load(second_model_path)
evaluate(test_dataloader, loaded_model_2, loss_fn, metric_fn, epoch=0)





print(f"Done with the pytorch experiment")
