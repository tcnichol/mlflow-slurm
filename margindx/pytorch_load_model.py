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

model_name = 'runs:/66ae7996ac0e42558f99a18104c35f16/model'


