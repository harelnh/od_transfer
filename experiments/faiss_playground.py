import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torchvision import models, transforms, datasets
from PIL import Image
import numpy as np
import pickle
import os
import faiss


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    Source: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

transforms_ = transforms.Compose([
    transforms.Resize(size=[224, 224], interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
dataset = ImageFolderWithPaths('data/ring_dataset', transforms_) # our custom dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models.resnet50(pretrained=True)

def pooling_output(x):
    global model
    for layer_name, layer in model._modules.items():
        x = layer(x)
        if layer_name == 'avgpool':
            break
    return x

# iterate over data
image_paths_cache_path = 'resources/cache/public_jewellery_image_paths.pkl'
index_path = 'resources/cache/public_jewellery_index'
imgs_path = []

if os.path.exists(image_paths_cache_path):
    with open(image_paths_cache_path, 'rb') as f:
        imgs_path = pickle.load(f)
    index = faiss.read_index(index_path)
else:
    descriptors = []
    model.to(DEVICE)
    with torch.no_grad():
        model.eval()
        for inputs, labels, paths in dataloader:
            result = pooling_output(inputs.to(DEVICE))
            descriptors.append(result.cpu().view(1, -1).numpy())
            imgs_path.append(paths)
            torch.cuda.empty_cache()

    index = faiss.IndexFlatL2(2048)
    descriptors = np.vstack(descriptors)
    index.add(descriptors)

    faiss.write_index(index, index_path)
    with open(image_paths_cache_path,'wb') as f:
        pickle.dump(imgs_path, f)

query_image = 'data/query_images/rings/Screen Shot 2020-11-13 at 10.02.43.png'
img = Image.open(query_image).convert('RGB')

input_tensor = transforms_(img)
input_tensor = input_tensor.view(1, *input_tensor.shape)
with torch.no_grad():
    model.eval()
    query_descriptors = pooling_output(input_tensor.to(DEVICE)).cpu().numpy()
    distance, indices = index.search(query_descriptors.reshape(1, 2048), 9)

fig, ax = plt.subplots(3, 3, figsize=(15,15))
for file_index, ax_i in zip(indices[0], np.array(ax).flatten()):
    ax_i.imshow(plt.imread(imgs_path[file_index][0]))

plt.show()