import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor
from dotenv import load_dotenv
load_dotenv()
import mlflow
import sys

from mlflow.server import get_app_client

mlflow.set_tracking_uri(uri="https://mlflow.margindx.software-dev.ncsa.illinois.edu")
mlflow.set_experiment("MLflow Pytorch Example")


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

print(f"Image size: {training_data[0][0].shape}")
print(f"Size of training dataset: {len(training_data)}")
print(f"Size of test dataset: {len(test_data)}")

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

epochs = 3
if len(sys.argv[:]) > 1:
    epochs = int(sys.argv[1])
print(f"The epochs are {epochs}")

try_cuda = False
if len(sys.argv[:]) > 2:
    try_cuda = bool(sys.argv[2])
print(f"Try cuda is {try_cuda}")

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


# Set our tracking server uri for logging
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080"
mlflow.set_experiment("/mlflow-pytorch-quickstart")

# Get cpu or gpu for training.
if try_cuda:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = "cpu"

print(f"The device is {device}")


def train(dataloader, model, loss_fn, metrics_fn, optimizer, epoch):
    """Train the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the training data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        loss_fn: a callable, the loss function.
        metrics_fn: a callable, the metrics function.
        optimizer: an instance of `torch.optim.Optimizer`, the optimizer used for training.
        epoch: an integer, the current epoch number.
    """
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        accuracy = metrics_fn(pred, y)

        # Backpropagation.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch
            step = batch // 100 * (epoch + 1)
            mlflow.log_metric("loss", f"{loss:2f}", step=step)
            mlflow.log_metric("accuracy", f"{accuracy:2f}", step=step)
            print(f"loss: {loss:2f} accuracy: {accuracy:2f} [{current} / {len(dataloader)}]")

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

loss_fn = nn.CrossEntropyLoss()
metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
model = ImageClassifier().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

with mlflow.start_run() as run:
    params = {
        "epochs": epochs,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "loss_function": loss_fn.__class__.__name__,
        "metric_function": metric_fn.__class__.__name__,
        "optimizer": "SGD",
    }
    # Log training parameters.
    mlflow.log_params(params)

    # Log model summary.
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt")

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, metric_fn, optimizer, epoch=t)
        evaluate(test_dataloader, model, loss_fn, metric_fn, epoch=0)

    # Save the trained model to MLflow.
    mlflow.pytorch.log_model(model, "model")

logged_model = f"runs:/{run.info.run_id}/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)

outputs = loaded_model.predict(training_data[0][0][None, :].numpy())





print(f"Done with the pytorch experiment")
