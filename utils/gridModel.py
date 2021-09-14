import numpy as np
import cv2
from scipy.interpolate import griddata
from utils.funcs import readObj,saveObj


class gridModel(object):
    """transform the model and grid in uv space"""

    def __init__(self, modelTemplate='dataset/BFM.obj', faceArea=np.array([26,228,0,225])/256):
                
        meshSize=(256j,256j)#number of the grid(xnum,ynum)

       
        #face area in uv space (xmin,xmax,ymin,ymax)
        self.faceArea=faceArea
        self.faceGrid=np.mgrid[self.faceArea[2]:self.faceArea[3]:meshSize[1],self.faceArea[0]:self.faceArea[1]:meshSize[0]]
        
        #load model
        vets,self.uvo,self.vn,self.face=readObj(modelTemplate)
        uvo=self.uvo[:,:2]
        self.uv=uvo*[1,-1]+[0,1] #flip y

        pass

    def model2map(self,vets):

        grid_y, grid_x=self.faceGrid

        mStr='linear'
        #mStr='cubic'
        map=griddata(self.uv, vets,  (grid_x, grid_y), method=mStr,fill_value=0)

    
        return map
    
    def map2Model(self,map):
    
        xmin,xmax,ymin,ymax=self.faceArea    

        vets=np.zeros((self.uv.shape[0],3),dtype=float)
        vetIdx=(self.uv[:,0]>=xmin)&(self.uv[:,0]<=xmax)&(self.uv[:,1]>=ymin)&(self.uv[:,1]<=ymax)
        imgIdx=(self.uv[vetIdx]-[xmin,ymin])*[map.shape[1]-1,map.shape[0]-1]/[xmax-xmin,ymax-ymin]

        interPoints=[]
        for i in range(len(imgIdx)):
    
            iIdx=imgIdx[i]        
            ulx=np.floor(iIdx[0]).astype(int)
            uly=np.floor(iIdx[1]).astype(int)
            ul=(ulx,uly,map[uly,ulx])
            
            urx=np.floor(iIdx[0]).astype(int)+1
            ury=np.floor(iIdx[1]).astype(int)
            ur=(urx,ury,map[ury,urx])
            
            dlx=np.floor(iIdx[0]).astype(int)
            dly=np.floor(iIdx[1]).astype(int)+1
            dl=(dlx,dly,map[dly,dlx])
            
            drx=np.floor(iIdx[0]).astype(int)+1
            dry=np.floor(iIdx[1]).astype(int)+1
            dr=(drx,dry,map[dry,drx])
    
            points=np.array(((ulx,uly),(urx,ury),(dlx,dly),(drx,dry)))
            value=np.array([map[uly,ulx],map[ury,urx],map[dly,dlx],map[dry,drx]])
    
            interPoints.append(griddata(points, value,  (iIdx[0], iIdx[1]), method='linear'))
    
        vets[vetIdx]=np.array(interPoints)
        return vets

    def saveFaceModel(self,vets,objName):
        saveObj(vets,self.uvo,self.vn,self.face,objName)