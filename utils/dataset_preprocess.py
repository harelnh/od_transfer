import torch
from tqdm import tqdm
import torchvision
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torchvision import models, transforms, datasets
from PIL import Image
import numpy as np
import pickle
import os

model = models.resnet101(pretrained=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(DEVICE)
emb_dim = 2048

def pooling_output(x):
    global model
    for layer_name, layer in model._modules.items():
        x = layer(x)
        if layer_name == 'avgpool':
            break
    return x



def embed_dataset(dataloader):

    embeddings = np.array([]).reshape(0,emb_dim)
    labels = []
    with torch.no_grad():
        model.eval()
        for inputs, labels_ in tqdm(dataloader):
            result = pooling_output(inputs.to(DEVICE))
            embeddings = np.concatenate((embeddings, result.cpu().numpy().squeeze()), axis=0)
            labels = labels + list(labels_.detach().cpu().numpy())
            torch.cuda.empty_cache()

    return embeddings, labels


def dataset_2_thumbnails(root_path, dim = 64):

    transforms_ = transforms.Compose([
        transforms.Resize(size=[dim, dim]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = torchvision.datasets.ImageFolder(root_path,
                                               transforms_)  # our custom dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
    arr = []
    for inputs, _ in dataloader:
        arr.append(inputs)
    tensor = torch.cat(arr)
    return tensor