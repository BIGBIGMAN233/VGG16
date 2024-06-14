<<<<<<< HEAD
import torch
from PIL import Image
import torchvision
from module import *

# 读取图片
image_path = 'img/猫2.png'
image = Image.open(image_path)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)
image = torch.reshape(image, [1, 3, 32, 32])
model = torch.load('results/vgg49--best.pth', map_location=torch.device('cpu'))
model.eval()
with torch.no_grad():
    output = model(image)

mydict = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5,
        'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

dict_new=dict([val,key] for key,val in mydict.items())
print(output)
print(output.argmax(1))
=======
import torch
from PIL import Image
import torchvision
from module import *

# 读取图片
image_path = 'img/猫2.png'
image = Image.open(image_path)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)
image = torch.reshape(image, [1, 3, 32, 32])
model = torch.load('results/vgg49--best.pth', map_location=torch.device('cpu'))
model.eval()
with torch.no_grad():
    output = model(image)

mydict = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5,
        'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

dict_new=dict([val,key] for key,val in mydict.items())
print(output)
print(output.argmax(1))
>>>>>>> a341e64481184ad887d7b06f3f36ca64c9313329
