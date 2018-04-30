import scipy.io as sio
import torch
import torch.nn as nn
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

def predictPoul(filename,net):
  donnees = sio.loadmat(filename)
  xD = donnees.get('x')
  yD = donnees.get('y')
  zD = donnees.get('z')
  winLen = 75
  stepSize = 15
  output = np.array([])
  index = np.arange(winLen,np.size(xD),stepSize)
  for idx in index:
    feat = np.array([])
    x =pd.Series(xD[0,idx-winLen:idx])
    y =pd.Series(yD[0,idx-winLen:idx])
    z =pd.Series(zD[0,idx-winLen:idx])
    meanData = np.array([x.mean(),y.mean(),z.mean()])
    feat = np.append(feat,meanData)
    stdData = [x.std(),y.std(),z.std()]
    feat = np.append(feat,stdData)
    feat = np.append(feat,x.corr(y))
    feat = np.append(feat,x.corr(z))
    feat = np.append(feat,y.corr(z))
    feat = np.expand_dims(feat,axis=0)
    inputs = torch.Tensor(feat)
    inputs = torch.autograd.Variable(inputs.cuda())
    outputs = net(inputs)
    _,outputs = torch.max(outputs.data,1)
    output = np.append(output, outputs)
    
  somme = 0
  indexFound = []
  value = np.array([])
  for i in range(np.size(output,0)-1) :
    value = np.append(value,np.zeros((1,74)))
    if (output[i+1]==1):
      somme += 1
    else:
      somme = 0
      value=np.append(value,0)
    if somme == 10:
      indexFound.append(index[i])
      value=np.append(value,1)
    else:
      value=np.append(value,0)
  indexFound = np.array(indexFound)
  array = np.zeros((np.size(xD),1))
  if indexFound.size != 0 :
    array[indexFound]=1
  return array,indexFound
  
if __name__ == "__main__":
  net = torch.load('neuNet1')
  inDir1 = "C:/Users/Phat/Documents/Stage/matFile/"
  filenames = glob.glob(inDir1+"/*.mat")
  outputFile = {}
  for filename in filenames:
    x,idxF = predictPoul(filename,net)
    path,name = os.path.split(filename)
    name,ext = os.path.splitext(name)
    outputFile[name]=idxF
    print(name, idxF)
    
  np.save('resultat1.npy',outputFile)
