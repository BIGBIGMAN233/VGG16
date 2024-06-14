<<<<<<< HEAD
import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x=self.model(x)
        return x

if __name__ == '__main__':
    vgg=VGG16()
    x=torch.ones([64,3,32,32])
    output=vgg(x)
    print(output.size())
=======
import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x=self.model(x)
        return x

if __name__ == '__main__':
    vgg=VGG16()
    x=torch.ones([64,3,32,32])
    output=vgg(x)
    print(output.size())
>>>>>>> a341e64481184ad887d7b06f3f36ca64c9313329
