import numpy as np
import datetime as dt
import os

def flectureFilePoulinage(filein):
  m=0
  filein=filein[:27]+"/"+filein[28:len(filein)]
  for i in range(len(filein)):
    if filein[i]=='/':
      m=i
  nom = filein[m+1:len(filein)]
  
  if len(nom)>50:
    jour = np.int8(nom[18:20])
    mois = np.int8(nom[20:22])
    annee = np.int8(nom[22:24])
    heure = np.int(nom[25:27])
    minute = np.int8(nom[27:29])
    second = np.int8(nom[29:31])
    datedeb = np.array([annee+2000, mois, jour, heure, minute, second])
    
    jour = np.int8(nom[32:34])
    mois = np.int8(nom[34:36])
    annee = np.int8(nom[36:38])
    heure = np.int(nom[39:41])
    minute = np.int8(nom[41:43])
    second = np.int8(nom[43:45])
    datefoal = np.array([annee+2000, mois, jour, heure, minute, second])
    
  else:
    datedeb = np.array([2011, 1, 1, 0, 0, 0])
    datefoal = np.array([2011, 1, 1, 1, 0, 0])
    
  binaryFile = open(filein,"rb")
  data = binaryFile.read()
  data = list(data)
  data = np.array(data)
  n = np.floor(len(data)/8).astype(np.int)
  data = np.reshape(data,(n,8))
  octet = data[:,6]
  
  del data
  tab = np.ones(len(octet),dtype=np.byte)*255
  detect = np.bitwise_and(128,octet)*(octet!=tab)    
  vitesse = np.bitwise_and(64,octet)*(octet!=tab)    
  
  newtimeH = np.zeros(n)
  c=dt.datetime(datedeb[0],datedeb[1],datedeb[2],datedeb[3],datedeb[4],datedeb[5])
  d=dt.datetime(datefoal[0],datefoal[1],datefoal[2],datefoal[3],datefoal[4],datefoal[5])
  
  Tinit = 24*(c-d)/dt.timedelta(days=1)
  HPose = (24*(c-d)/dt.timedelta(days=1))+0.5
  IndPose = n
  PoseFind = 0
  HMinus = -2.5
  HPlus = 0
  IndMinus = 1
  IndPlus = 1
  MinusFind = 0
  PlusFind = 0
  vitesse_compense = np.zeros(n,dtype = np.int8)
  for i in range(n):
    if vitesse[i]==64:
      Tinit = Tinit+0.25/3600
      vitesse_compense[i]=1
    else:
      Tinit = Tinit+0.25*16/3600
    
    newtimeH[i]=Tinit
    
    if PoseFind == 0:
      if Tinit>HPose:
        IndPose = i
        PoseFind = 1
    
    if MinusFind == 0:
      if Tinit>HMinus:
        IndMinus = i
        MinusFind = 1
    
    if PlusFind ==0:
      if Tinit>HPlus:
        IndPlus = i
        PlusFind = 1
        
  if IndPlus == 1:
    IndPlus = n
    
  del octet
  nom,_ = os.path.splitext(nom)
  return IndPlus, IndMinus, nom

if __name__ == '__main__':
  filein = "D:/Data/SmartFoal/util_2016/0005_0048_ILIC_04_060416_174500_120416_020500_16130101_2RAS18TF01.dat"
  IndPlus,IndMinus,filename = flectureFilePoulinage(filein)
    
    
    
      
      