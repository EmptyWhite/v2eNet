

import numpy as np
import cv2
import os
import random

def main():
    
    #computeMaxMin()

    computeDeltaMaxMin()


    pass

def computeMaxMin():
    dataFolder='dataset/train/'
    fileList=os.listdir(dataFolder)
    modelMax=float('-inf')
    modelMin=float('inf')
    expP=[]
    for i,file in enumerate(fileList):
        data=np.load(dataFolder+file)
        exp=data['vetMapExp']
        neu=data['vetMapNeu']
        expParam=data['expParam']
        modelMax=max((modelMax,np.max(exp),np.max(neu)))
        modelMin=min((modelMin,np.min(exp),np.min(neu)))
        expP.append(expParam)
        print('\r processing, finish{0}/{1}'.format(i,len(fileList)),end='')

    print('')
    print('modelMax:'+str(modelMax))
    print('modelMin:'+str(modelMin))

    pass

def computeDeltaMaxMin():
    dataFolder='dataset/train/'
    fileList=os.listdir(dataFolder)
    
    deltaMax=float('-inf')
    deltaMin=float('inf')

    for i,file in enumerate(fileList):
        data=np.load(dataFolder+file)
        exp=data['vetMapExp']
        neu=data['vetMapNeu']
        exp,neu,_,_=dataNormalize(exp,neu,0,0)
        delta=exp-neu
        deltaMax=max(deltaMax,np.max(delta))
        deltaMin=min(deltaMin,np.min(delta))
        print('\r processing, finish{0}/{1}'.format(i,len(fileList)),end='')

    print('')
    print('deltaMax:'+str(deltaMax))
    print('deltaMin:'+str(deltaMin))
    pass



def dataNormalize(exp,neu,expParam,embedding):
    # max min is acquired by computeMaxMin()
    modelMax=15
    modelMin=-16
    expParamMax=5.5
    expParamMin=-11.1
    embeddingMax=1.31
    embeddingMin=-1.43
    exp=(exp-modelMin)/(modelMax-modelMin)*2-1
    neu=(neu-modelMin)/(modelMax-modelMin)*2-1
    expParam=(expParam-expParamMin)/(expParamMax-expParamMin)*2-1
    embedding=(embedding-embeddingMin)/(embeddingMax-embeddingMin)*2-1
    return exp,neu,expParam,embedding


def dataNormalizeDelta(delta):
    #max min is acquired by computeDeltaMaxMin()
    deltaMin=-0.353
    deltaMax=0.182
    deltaN=(delta-deltaMin)/(deltaMax-deltaMin)*2-1
    return deltaN

def deDataNormalizeDelta(deltaN):    
    #max min is acquired by computeDeltaMaxMin()
    deltaMin=-0.353
    deltaMax=0.182
    delta=(deltaN+1)/2*(deltaMax-deltaMin)+deltaMin
    return delta




if __name__=='__main__':
    main()