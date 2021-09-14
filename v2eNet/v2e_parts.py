

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools



class inputDown(nn.Module):

    def __init__(self,input_nc,output_nc):
        super().__init__()
        self.conv=nn.Conv2d(input_nc, output_nc, kernel_size=4,stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)

class down(nn.Module):
    def __init__(self,input_nc,output_nc):
        super().__init__()
        norm=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        self.downConv=nn.Sequential(
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(input_nc, output_nc, kernel_size=4,stride=2, padding=1, bias=False),
            norm(output_nc)
            )

    def forward(self, x):
        return self.downConv(x)

class downBottleNeck(nn.Module):
    def __init__(self,input_nc,output_nc):
        super().__init__()
        self.downBN=nn.Sequential(
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(input_nc, output_nc, kernel_size=4,stride=2, padding=1, bias=False),
            nn.ReLU(True)
            #nn.LeakyReLU(0.2,True)
            )

    def forward(self, x):
        return self.downBN(x)

class upBottleNeck(nn.Module):
    def __init__(self,input_nc,output_nc):
        super().__init__()
        norm=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        self.upBN=nn.Sequential(
            nn.ConvTranspose2d(input_nc, output_nc,kernel_size=4, stride=2,padding=1, bias=False),
            norm(output_nc)
            )

    def forward(self, x1,x2):
        x1=self.upBN(x1)
        return torch.cat([x2, x1], dim=1)

class up(nn.Module):
    def __init__(self,input_nc,output_nc,dropOut=False):
        super().__init__()
        norm=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        if dropOut:
            self.upConv=nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(input_nc * 2, output_nc,kernel_size=4, stride=2,padding=1, bias=False),
                norm(output_nc),
                nn.Dropout(0.5)
                )
        else:
            self.upConv=nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(input_nc * 2, output_nc,kernel_size=4, stride=2,padding=1, bias=False),
                norm(output_nc)
                )

    def forward(self, x1,x2):
        x1=self.upConv(x1)
        return torch.cat([x2, x1], dim=1)

class outUp(nn.Module):
    def __init__(self,input_nc,output_nc):
        super().__init__()
        self.out=nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(input_nc * 2, output_nc,kernel_size=4, stride=2,padding=1),
            nn.Tanh()
            )

    def forward(self, x):
        return self.out(x)

class inputEmbedding(nn.Module):
    def __init__(self,input_nc,output_nc,middle=2048):
        super().__init__()
        self.outputNc=output_nc
        self.embed=nn.Sequential(
            nn.Linear(input_nc,middle),
            nn.LeakyReLU(0.2,True),
            nn.Linear(middle,output_nc**2)
            )

    def forward(self, x):
        x=self.embed(x)
        return x.reshape(-1,1,self.outputNc,self.outputNc)