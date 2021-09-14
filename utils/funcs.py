import numpy as np
import cv2
import os

def readObj(objName):
    #Load obj models
    v = []
    f = []
    vn = []
    vt = []
    with open(objName) as obj_file:
        obj_lines = obj_file.readlines()
    for line in obj_lines:
        lineData = line.split()
        if len(lineData):
            if lineData[0] == "v":
                vertexPoint = []
                vertexPoint.append((float)(lineData[1]))
                vertexPoint.append((float)(lineData[2]))
                vertexPoint.append((float)(lineData[3]))
                v.append(vertexPoint)
            
            if lineData[0] == "vt":
                uvPoint = []
                for lineD in lineData[1:]:
                    uvPoint.append((float)(lineD))
                vt.append(uvPoint)
            
            if lineData[0] == "vn":
                nPoint = []
                nPoint.append((float)(lineData[1]))
                nPoint.append((float)(lineData[2]))
                nPoint.append((float)(lineData[3]))
                vn.append(nPoint)
            
            if lineData[0] == "f":
                f.append(line.rstrip())


    vertices = np.array(v)
    face = np.array(f)
    normal = np.array(vn)
    uv = np.array(vt)

    return vertices,uv,normal,face

def saveObj(vertices,uvs,normals,faces,fileName):
    if not fileName[-4:]==".obj":
        fileName+=".obj"
    with open(fileName,"w") as obj_file:
        obj_file.write("#points:{0:d},faces:{1:d}\n\n".format(len(vertices),len(faces)))
        for point in vertices:
            obj_file.write("v {0} {1} {2}\n".format(point[0],point[1],point[2]))
        if len(uvs):
            obj_file.write("\n")
            for vt in uvs:
                obj_file.write("vt {0} {1}\n".format(vt[0],vt[1]))
        if len(normals):
            obj_file.write("\n")
            for vn in normals:
                obj_file.write("vn {0} {1} {2}\n".format(vn[0],vn[1],vn[2]))
        obj_file.write("\n")
        for face in faces:
            obj_file.write(face+"\n")
    print("save model"+fileName)