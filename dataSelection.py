import pandas as pd
import numpy as np
import glob
import os
import scipy.io as sio
import cv2
import random

dataDir = "C:/Users/Phat/Documents/Stage/newData/"

filenames = glob.glob(dataDir+"/*.csv")

def featuresSelec(df,startPoint,endPoint):
    startPoint = np.int(np.floor(startPoint/16))
    endPoint = np.int(np.floor(endPoint/16))
    winLen = 75
    intervT = list(range(startPoint,endPoint))
    intervF = list(range(winLen,startPoint))
    groundTruth = []
    features = np.zeros((1,9))
    idxT = random.sample(intervT,50)
    idxF = random.sample(intervF,50)
    for i in idxT :
        feat = featureExtract(df[i-winLen:i])
        features = np.concatenate((features,feat),axis = 0)
        groundTruth.append(1)
    for j in idxF:
        feat = featureExtract(df[j-winLen:j])
        features = np.concatenate((features,feat),axis = 0)
        groundTruth.append(0)
    return features[1:,:], groundTruth
  
  
def featureExtract(data):
    x = data['x']
    y = data['y']
    z = data['z']
#    algo10 = data['algo10'].div(data['algo10'].max())
    feat = np.array([])
    meanData = np.array([x.mean(),y.mean(),z.mean()])
    feat = np.append(feat,meanData)
    stdData = [x.std(),y.std(),z.std()]
    feat = np.append(feat,stdData)
#    xycorr = x.corr(y)
#    xzcorr = x.corr(z)
#    yzcorr = y.corr(z)
#    algo10Auto = algo10.autocorr(5)
    feat = np.append(feat,x.corr(y))
    feat = np.append(feat,x.corr(z))
    feat = np.append(feat,y.corr(z))
#    feat = np.append(feat,algo10.autocorr(5))
    feat = np.expand_dims(feat,axis=0)
    return feat
    
feat = np.zeros((1,9))
ground = []

###############################################################################
filein = dataDir+"/0001_0000_0000_04_020513_190000_050513_051000_12390168_0000000000.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,815100,821000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0001_0001_0000_04_020513_184000_030513_010000_12390162_0000000000.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,81000,84000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0001_0002_0000_04_190413_125940_210413_220000_12390191_0000000000.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,812000,815600)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0001_0005_0000_04_280413_190000_300413_071500_12390167_0000000000.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,502000,510000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0001_0007_0000_04_030513_210000_060513_064500_12390203_0000000000.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,812000,821000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0004_0001_0PAC_04_030214_160000_050214_014500_12390177_1RAS04SP01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,479000,480500)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0004_0002_0REI_04_110214_080000_120214_024500_12390177_2RAS00TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,262000,264000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0004_0004_0MAY_04_010314_182815_020314_050000_12390192_2RAS00PS01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,144500,146500)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0004_0010_0PRU_04_310314_183000_310314_211000_12390222_1RAS00TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,29500,31800)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0004_0014_0VIC_04_190414_092700_190414_194000_DL121001_2RAS01TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,144500,147121)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0004_0020_0LAM_04_260414_122500_260414_223000_DL121001_3GP_00TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,142500,145200)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0003_0TEQ_04_070515_204500_120515_050000_DL150016_1RAS02TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,1479000,1484000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0004_0KEP_04_030515_213500_080515_020300_DL150018_1RAS0CTF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,1444500,1447000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0005_0PRE_04_030515_173000_030515_214300_DL150015_1RAS02TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,57000,61000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0007_0LIM_04_290415_181500_020515_112000_DL150015_1RAS04TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,933500,936500)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0009_0ENA_04_290415_182400_020515_211000_DL150018_1RAS03TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,1072000,1076500)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0011_0QUA_04_290415_174500_080515_233000_DL150020_1RAS04TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,3189000,3193000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0012_0ARZ_04_240415_174500_300415_230100_DL150016_2RAS02PE01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,2145000,2149440)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0015_0REI_04_210415_164500_220415_205000_DL150016_3GP_03TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,402000,404400)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0017_0VIV_04_200415_172500_240415_021000_DL150020_1RAS02TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,1158000,1162800)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0018_0LAR_04_160415_203000_270415_021000_DL150015_1RAS02PS01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,3532000,3536000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0020_0PEP_04_140415_202500_160415_005300_DL150020_1RAS05TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,406000,409920)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0022_0SER_04_080415_172500_130415_201600_DL150020_3PCM00TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,1762000,1766300)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0023_0RIK_04_070415_164500_080415_072600_DL150020_1RAS06TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,209000,210500)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0025_0TRA_04_130415_175500_240415_190000_DL150018_3PME03TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,3808000,3816000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0026_0SYB_04_090415_174500_200415_215000_DL150016_2RAS03TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,3858000,3860400)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0027_0FOR_04_030415_174500_130415_003300_DL150015_2RAS0ATF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,3205000,3208320)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0028_0QUI_04_030415_162500_040415_224500_DL150020_1RAS04TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,434000,436800)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0030_0NEW_04_300515_174000_100615_010200_DL150016_2GP_02SF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,3558000,3561500)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0031_0NIS_04_310515_205500_010615_050000_DL150015_1RAS06TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,110000,116401)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0032_0ENT_04_190615_181000_010715_001500_DL150015_0RAS0CTF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,3884000,3889200)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0033_0PIS_04_060615_174500_160615_002500_DL150020_1RAS06TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,3201000,3206000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0034_0PAS_04_170615_171500_170615_195000_DL150016_3GP_03SP01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,34500,36000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0035_0JIP_04_180615_214500_190615_004000_DL150020_2PME08TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,41000,42000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0036_0SIN_04_100515_204500_220515_024000_DL150015_3PME09PS01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,3875000,3884000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0037_0ULA_04_100515_174500_190515_135000_DL150018_3GP_04PE01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,3051000,3054000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0103_VICT_04_140417_170500_180417_052000_16130111_2RAS13TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,1207500,1209841)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0110_LUNA_04_040517_181500_050517_044900_16160125_1RAS19TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,146000,148000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0122_VACE_04_070517_105000_100517_043200_16130112_2RAS11TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,937000,942630)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0007_0052_DULA_04_150616_090000_200616_144000_16160125_1RAS1ITF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,1800000,1803500)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0007_0053_HERB_04_300516_084500_040616_214000_16130121_1RAS0CTF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,1904000,1908000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0007_0063_VINT_04_110317_201500_250317_173500_16130116_2RAS02TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,4773800,4774800)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0007_0088_QUEM_04_080517_175200_180517_012000_16130117_2RAS10TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,3211500,3213000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0010_0004_RIZI_04_110416_185000_150416_214000_16130115_2RAS11PS01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,1414000,1419945)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0010_0006_LUCE_04_050516_205000_060516_051500_16130115_2RAS13PS01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,113000,117800)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0044_0HAR_04_300515_173000_060615_150800_DL150020_3PME09TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,2375000,2380000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0006_0003_0ISM_04_110515_171500_150515_030000_DL150014_2RAS00TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,1170000,1177200)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0006_0007_0ORC_04_110515_174500_120515_023500_DL150010_2RAS00TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,123000,127200)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0007_0001_0QUI_04_160415_170000_160415_213000_DL150006_2RAS06TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,59000,64000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0007_0002_0DAS_04_160415_170000_180415_040000_DL150007_1RAS0CTF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,497000,502000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0007_0003_0VAH_04_160415_170500_180415_213500_DL150008_2RAS01TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,750000,752800)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0007_0009_0ODY_04_230415_174000_290415_053000_DL150007_2RAS03TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,1891000,1895000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0007_0010_0OQU_04_230415_173500_280415_234500_DL150006_1RAS05TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,1815000,1816800)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0007_0012_0PER_04_280415_163500_290415_214500_DL150005_2RAS00TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,410000,420000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0007_0013_0FAT_04_280415_164500_290415_205500_DL150008_1RAS00TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,403000,405601)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0004_0001_0PAC_04_030214_160000_050214_014500_12390177_1RAS04SP01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,479000,480500)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0004_0020_0LAM_04_260414_122500_260414_223000_DL121001_3GP_00TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,142500,145200)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0043_0OSM_04_180515_180500_200515_234000_DL150016_2RAS08TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,768400,770800)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0006_0005_0CAM_04_110515_170000_110515_231000_DL150012_0RAS00PS01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,85560,87480)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0051_TRAL_04_140416_173000_150416_204000_16130103_3RED14TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,385700,388500)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0056_POCA_04_240416_171600_240416_190000_16130104_2RAS15TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,20190,23290)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0065_SONA_04_270516_182000_290516_203200_16130105_2RAS02TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,712500,716000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0073_COST_04_130416_180500_210416_121500_16130107_2RAS11TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,2675500,2677200)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0080_DIAM_04_050416_174800_080416_210300_16130110_3PME04PS01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,1077000,1084000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0084_BILL_04_140616_074500_150616_104000_16130111_2RAS16SP01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,381600,384300)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0100_TERR_04_020317_162000_030317_185600_16130111_2RAS12TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,377000,378800)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0005_0108_BILL_04_120617_205500_140617_052500_16130111_2RAS12SP01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,462000,464500)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0007_0041_TAQU_04_210416_081500_210416_140000_16130123_0RAS16TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,79640,82170)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0007_0084_TAQU_04_210417_083100_210417_214500_16130117_2RAS10TF01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,186500,189100)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0012_0005_NINA_04_210217_230000_110317_210000_16130103_2RAS00PS01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,6185000,6189000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
filein = dataDir+"/0012_0017_NIST_04_140217_155000_200217_193000_16130110_2RAS10PS01.csv"
df = pd.read_csv(filein,delimiter=',')
features,groundTruth = featuresSelec(df,2121500,2123000)
feat = np.concatenate((feat,features),axis = 0)
ground = ground + groundTruth

###############################################################################
###############################################################################

#groundX = np.array(ground)
#idxT = np.nonzero(groundX)
#idxT = idxT[0]
#featT = []
#groundT = []
#for idx in idxT:
#  featT.append(feat[idx])
#  groundT.append(1)
#
#feat = feat + featT*2
#ground = ground + groundT*2

feat = feat[1:,:]

colNames = ['meanX', 'meanY', 'meanZ', 'stdX', 'stdY', 'stdZ', 'xycorr', 'xzcorr', 'yzcorr', 'Label']
df = pd.DataFrame(feat,columns = colNames[:9])
df = df.fillna(method='pad',limit = 1)

featuresTraining = np.array(df)

maxFeats = df.max(axis = 0)
minFeats = df.min(axis = 0)

maxFeats = np.array(maxFeats)
minFeats = np.array(minFeats)

normalizedFeats = (featuresTraining-minFeats)/(maxFeats-minFeats)
ground = np.array(ground)
ground = np.expand_dims(ground,axis=1)


fullData = np.concatenate((normalizedFeats,ground),axis=1)

fullData = pd.DataFrame(fullData,columns = colNames)
fullData = fullData.fillna(method='pad',limit = 1)
fullData = np.array(fullData)

np.save('trainingData.npy',fullData)
np.save('maxFeats.npy',maxFeats)
np.save('minFeats.npy',minFeats)