import os

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
import cv2
import numpy as np
import PIL
from PIL import Image
from mlflow.server import get_app_client
from torchvision.transforms import transforms


def image_to_tensor(image_path):
    # Open the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale

    # Resize the image to 28x28
    img = img.resize((28, 28))

    # Convert the image to a tensor
    transform = transforms.ToTensor()
    tensor = transform(img)

    # Add the batch and channel dimensions
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor
def convert_to_grayscale(image_path):
    """Converts an image to grayscale and resizes it to 28x28."""

    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to 28x28
    resized_image = cv2.resize(gray_image, (28, 28))

    return resized_image

def resize_to_28x28(img):
    img_h, img_w = img.shape
    dim_size_max = max(img.shape)

    if dim_size_max == img_w:
        im_h = (26 * img_h) // img_w
        if im_h <= 0 or img_w <= 0:
            print("Invalid Image Dimention: ", im_h, img_w, img_h)
        tmp_img = cv2.resize(img, (26,im_h),0,0,cv2.INTER_NEAREST)
    else:
        im_w = (26 * img_w) // img_h
        if im_w <= 0 or img_h <= 0:
            print("Invalid Image Dimention: ", im_w, img_w, img_h)
        tmp_img = cv2.resize(img, (im_w, 26),0,0,cv2.INTER_NEAREST)

    out_img = np.zeros((28, 28), dtype=np.ubyte)

    nb_h, nb_w = out_img.shape
    na_h, na_w = tmp_img.shape
    y_min = (nb_w) // 2 - (na_w // 2)
    y_max = y_min + na_w
    x_min = (nb_h) // 2 - (na_h // 2)
    x_max = x_min + na_h

    out_img[x_min:x_max, y_min:y_max] = tmp_img

    return out_img


def convert_image(image_path):
    # Open the image
    img = Image.open(image_path)

    # Resize the image to 28x28
    img = img.resize((28, 28))

    # Convert the image to grayscale
    img = img.convert('L')

    # Convert the image to a numpy array
    img_array = np.array(img, dtype=np.uint8)

    return img_array

mlflow.set_tracking_uri(uri="https://mlflow.margindx.software-dev.ncsa.illinois.edu")
mlflow.set_experiment("MLflow Pytorch Example")

# https://stackoverflow.com/questions/59097657/in-pytorch-how-to-test-simple-image-with-my-loaded-mo

test_image_path = os.path.join(os.getcwd(), 'margindx', 'image1.png')

grayscale_image = convert_to_grayscale(test_image_path)

print(os.path.exists(test_image_path))
test_image = Image.open(test_image_path)
# Define a transform to convert PIL
# image to a Torch tensor
transform = transforms.Compose([
    transforms.PILToTensor()
])

# transform = transforms.PILToTensor()
# Convert the PIL image to Torch tensor
# img_tensor = transform(image)

# image_array = convert_image(test_image)


model_uri = 'runs:/66ae7996ac0e42558f99a18104c35f16/model'
model_path = 'mlflow-artifacts:/5/fea4c50287f947c496721788237344fd/artifacts/model/data/model.pth'

# model = torch.load(model_path)

loaded_model = mlflow.pytorch.load_model(model_uri)
model_parameters = loaded_model.parameters()
all_parameters = [x for x in model_parameters]

try:
    loaded_model.eval()
    print(f"We loaded the model")
except Exception as e:
    print(f"We could not load the model")
    print(e)

# batch = torch.tensor(image_array / 255).unsqueeze(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

# image = Image.open(test_image_path)
#
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Adjust as needed
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# image = transform(image)
# image = image.unsqueeze(0)  # Add batch dimension
# converted_image = convert_image(test_image_path)
# test_image_array = np.array(test_image)
# other_converted_image = resize_to_28x28(test_image_array)


grayscale_image = convert_to_grayscale(test_image_path)
tensor_input = image_to_tensor(test_image_path)
tensor_input_array = tensor_input.numpy()
loaded_model_2 = mlflow.pyfunc.load_model(model_uri)
loaded_model.eval()
with torch.no_grad():
    output = loaded_model_2.predict(tensor_input_array)
    print('here')

# Get the predicted class
_, predicted = torch.max(output, 1)
print(f"Predicted class: {predicted.item()}")