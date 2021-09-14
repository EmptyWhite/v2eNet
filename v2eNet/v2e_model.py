

import torch.nn.functional as F

from .v2e_parts import *


class v2e(nn.Module):
    def __init__(self, input_nc, output_nc,embeddingNum,inputSize):
        super(v2e, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.embeddingNum=embeddingNum

        
        self.inputEmbedding=inputEmbedding(embeddingNum,inputSize)
        self.down1=inputDown(input_nc+1,64)   #(64,128,128)
        self.down2=down(64,128)             #(128,64,64)
        self.down3=down(128,256)            #(256,32,32)
        self.down4=down(256,512)            #(512,16,16)
        self.down5=down(512,512)            #(512,8,8)
        self.down6=down(512,512)            #(512,4,4)
        self.down7=down(512,512)            #(512,2,2)
        self.down8=downBottleNeck(512,512)  #(512,1,1)

        self.up1=upBottleNeck(512,512)      #(512,2,2)
        self.up2=up(512,512,dropOut=True)   #(512,4,4)
        self.up3=up(512,512,dropOut=True)   #(512,8,8)
        self.up4=up(512,512,dropOut=True)   #(512,16,16)
        self.up5=up(512,256)                #(256,32,32)
        self.up6=up(256,128)                #(128,64,64)
        self.up7=up(128,64)                 #(64,128,128)
        self.up8=outUp(64,output_nc)        #(output_nc,256,256)

    def forward(self, x,y):
        
        y=self.inputEmbedding(y)

        x=torch.cat([x, y], dim=1)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7)
        x = self.up1(x8, x7)
        x = self.up2(x, x6)
        x = self.up3(x, x5)
        x = self.up4(x, x4)
        x = self.up5(x, x3)
        x = self.up6(x, x2)
        x = self.up7(x, x1)
        out = self.up8(x)
        return out
