
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from utils.dataPreprocess import dataNormalize,dataNormalizeDelta

class faceData(Dataset):
    def __init__(self,dataDir,expression,delta,embedding_nc):

        self.dataDir=dataDir
        self.dataNames=os.listdir(self.dataDir)
        self.expression=expression
        self.embedding_nc=embedding_nc
        self.delta=delta
        if self.delta:
            print('use delta')

        logging.info(f'Creating dataset with {len(self.dataNames)} examples')

    def __len__(self):
        return len(self.dataNames)

    @classmethod
    def preprocess(cls, pil_img):

        # HWC to CHW
        img_trans = pil_img.transpose((2, 0, 1))

        return img_trans

    def __getitem__(self, i):

        dataName=os.path.join(self.dataDir,self.dataNames[i])

        data=np.load(dataName)

        neu=data['vetMapNeu']
        exp=data['vetMapExp']
        if self.expression=='expParam':
            Expression=data['expParam']
            exp,neu,_,_=dataNormalize(exp,neu,0,0)
        elif self.expression=='embedding':
            Expression=data['embedding']
            exp,neu,_,_=dataNormalize(exp,neu,0,0)
        
        if self.delta:
            exp=exp-neu
            exp=dataNormalizeDelta(exp)

        neu = self.preprocess(neu)
        exp = self.preprocess(exp)
        Expression=Expression.reshape(-1)

        return {
            'neu': torch.from_numpy(neu).type(torch.FloatTensor),
            'exp': torch.from_numpy(exp).type(torch.FloatTensor),
            'Expression': torch.from_numpy(Expression).type(torch.FloatTensor),
            'path':self.dataNames[i]
        }