
#test for expression retargeting
#testA is the expression embedding folder, testB is the face model folder

import os
from v2eNet import v2e
from utils.gridModel import gridModel
import numpy as np
import cv2
from utils.dataPreprocess import deDataNormalizeDelta
import torch
from utils.faceData import faceData
from torch.utils.data import DataLoader

def saveResult(resultFolder,images,imgPath,faceGrid,delta):
    
    neu=images['neu'].cpu().numpy()[0] #neutral
    neu=np.transpose(neu,(1,2,0))

    expg=images['out'].cpu().detach().numpy()[0] #generated model
    expg=np.transpose(expg,(1,2,0))
    dataName=imgPath

    if delta:

        expg=deDataNormalizeDelta(expg)
        expg=expg+neu

    vetExpg=faceGrid.map2Model(expg)
    faceGrid.saveFaceModel(vetExpg,os.path.join(resultFolder,dataName))
    pass

def saveData(resultFolder,data,saveName,faceGrid):
    #save data to model
    
    img=data.numpy()[0]
    img=np.transpose(img,(1,2,0))
    vets=faceGrid.map2Model(img)
    faceGrid.saveFaceModel(vets,os.path.join(resultFolder,saveName))
    pass

if __name__ == '__main__':
    
    expression='expParam'
    delta=True
    embedding_nc=14
    loadModel='checkpoints/ckpt.pth'

    dataRoot='dataset/'
    saveFolder='results'

    dataFolder='testA'
    dataA=faceData(dataRoot+dataFolder, expression,delta,embedding_nc)
    datasetA = DataLoader(dataA, 1, shuffle=False,num_workers=0)

    dataFolder='testA'
    dataB=faceData(dataRoot+dataFolder, expression,delta,embedding_nc)
    datasetB = DataLoader(dataB, 1, shuffle=False,num_workers=0)

    model =v2e(input_nc=3, output_nc=3,embeddingNum=embedding_nc,inputSize=256)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: '+str(device))
    model.to(device=device)
    model.load_state_dict(torch.load(loadModel, map_location=device))
    print('Load model from: '+loadModel)
    
    model.eval()

    
    modelTemplate='dataset/meryShow.obj'
    faceArea=np.array([152,871,192,868])/1024 #face area in uv space for mery
    faceGrid=gridModel(modelTemplate=modelTemplate,faceArea=faceArea)
    #save models

    for i, dataA in enumerate(datasetA):
        if delta:
            deltaB=deDataNormalizeDelta(dataA['exp'])
            saveData(dataRoot+saveFolder,dataA['neu']+deltaB,'A'+str(i),faceGrid)
        else:
            saveData(dataRoot+saveFolder,dataA['exp'],'A'+str(i),faceGrid)
    #for j, dataB in enumerate(datasetB):
    #    saveData(dataRoot+saveFolder,dataB['neu'],'B'+str(j),faceGrid)

    for i, dataA in enumerate(datasetA):
        for j, dataB in enumerate(datasetB):
            
            if not i==j:
                continue

            neu=dataB['neu']
            exp=dataB['exp']
            embedding=dataA['Expression']
    
            neu = neu.to(device=device, dtype=torch.float32)
            exp = exp.to(device=device, dtype=torch.float32)
            embedding=embedding.to(device=device, dtype=torch.float32)

            output = model(neu,embedding)
            
            images={'neu':neu,'exp':exp,'out':output}

            saveResult(dataRoot+saveFolder,images,str(i)+'-'+str(j),faceGrid,delta)
            print('Finish{0}-{1}({2}-{3})'.format(i+1,j+1,len(datasetA),len(datasetB)))
