import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
from utils.dataset_preprocess import embed_dataset, dataset_2_thumbnails

if not os.path.exists('runs'):
  os.mkdir('runs')

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/jewellery_resnet101_experiment')

transforms_ = transforms.Compose([
    transforms.Resize(size=[224, 224], interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

root_path = 'data/Jewellery-Classification-master/dataset/training'
dataset = torchvision.datasets.ImageFolder('data/Jewellery-Classification-master/dataset/training',
                                           transforms_)  # our custom dataset
classes = dataset.classes
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
embeddings, labels_idx = embed_dataset(dataloader)
embeddings = np.array(embeddings).squeeze()
labels = [classes[idx] for idx in labels_idx]

dataiter = iter(dataloader)
images, _ = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)

thumbnail_tensor = dataset_2_thumbnails(root_path, dim = 128)
writer.add_embedding(embeddings,metadata=labels,label_img=thumbnail_tensor)
writer.close()

pass

# tensor board commands:
# 1. tensorboard --logdir=runs
# 2. fuser 6006/tcp -k