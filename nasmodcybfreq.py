import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from operator import itemgetter
import os

import mlfunctions as ml
import columns as cl

COLUMNS= cl.returnCOLUMNS()
dtype=cl.returndtype()
BANDFEATURES=cl.returnBandFeatures()
LABEL = "CYBClaimCount"
EXPOSURE="CYBExposure"

filenametrain="nasmod.csv"

print("Starting Program")
print(datetime.datetime.now())

data_train = pd.read_csv(filenametrain, skipinitialspace=True, skiprows=1, names=COLUMNS, dtype=dtype)


data_train=data_train[data_train[EXPOSURE]>0]

CONTFEATURES = ['NumberPhysicians','Revenue','BothCoverages','CyberLimit','MedefenseLimit']
CATFEATURES = ['ProgramType','CyberLimit_Cat','MedefenseLimit_Cat']



DUMMYFEATURES=[]
for feature in CATFEATURES:
	dummiesTrain=pd.get_dummies(data_train[feature],prefix=feature)
	dummiesTrain_temp=pd.concat([dummiesTrain,data_train[EXPOSURE]],axis=1)
	for column in dummiesTrain_temp:
		dummiesTrain_temp[column] = np.where(dummiesTrain_temp[column]==1, dummiesTrain_temp[EXPOSURE], 0)
	dummiesTrain_temp = dummiesTrain_temp.drop(EXPOSURE, 1)
	temp=dummiesTrain_temp.sum()	
	dummiesTrain=dummiesTrain[temp[temp<temp.max()].index.values]
	if(len(dummiesTrain.columns)>0):
		dummiesTrain.columns = dummiesTrain.columns.str.replace('\s+', '_')
		data_train=pd.concat([data_train,dummiesTrain],axis=1)
		DUMMYFEATURES += list(dummiesTrain)

# print(DUMMYFEATURES)


# Create custom variables for GLM
data_train['NumberPhysicians_missing']=np.where(data_train['NumberPhysicians']==0, 1, 0)
data_train['Revenue_missing']=np.where(data_train['Revenue']<=0, 1, 0)
data_train['log_CyberLimit']=np.log(data_train['CyberLimit'])
data_train['log_Revenue']=np.where(data_train['Revenue']<=0,-1,np.log(data_train['Revenue']))
data_train['log_NumberPhysicians']=np.where(data_train['NumberPhysicians']<=0,-1,np.log(data_train['NumberPhysicians']))
data_train['ProgramType_Nonopen']=np.where(data_train['ProgramType'] != 'open', 1, 0)



formula = LABEL+' ~  NumberPhysicians_missing + log_NumberPhysicians '
formula += ' + log_Revenue   + Revenue_missing  + log_CyberLimit'
# formula += ' + ProgramType_omprog + ProgramType_prog + ProgramType_rein + ProgramType_open + ProgramType_reinbu  '
# formula += ' + MedefenseLimit_Cat_01_10k + MedefenseLimit_Cat_02_25k + MedefenseLimit_Cat_03_50k + MedefenseLimit_Cat_04_100k + MedefenseLimit_Cat_05_250k + MedefenseLimit_Cat_06_500k + MedefenseLimit_Cat_07_1000k + MedefenseLimit_Cat_08_2000k' 
# formula += ' + CyberLimit_Cat_01_10k + CyberLimit_Cat_02_25k + CyberLimit_Cat_03_50k + CyberLimit_Cat_04_100k + CyberLimit_Cat_05_150k + CyberLimit_Cat_06_250k + CyberLimit_Cat_07_500k + CyberLimit_Cat_08_750k + CyberLimit_Cat_10_2000k + CyberLimit_Cat_11_2500k + CyberLimit_Cat_12_3000k + CyberLimit_Cat_13_4000k + CyberLimit_Cat_14_5000k + CyberLimit_Cat_15_6000k + CyberLimit_Cat_16_7500k + CyberLimit_Cat_17_8000k + CyberLimit_Cat_18_10000k + CyberLimit_Cat_19_15000k + CyberLimit_Cat_20_20000k'

prediction_glm = ml.GLM(data_train, CONTFEATURES,CATFEATURES,LABEL,EXPOSURE,formula,'cybfreq/',0,'poisson')


data_train['weight'] = data_train[EXPOSURE]
data_train['predictedValue'] = prediction_glm
data_train['actualValue'] = data_train[LABEL]

ALLCONTFEATURES = cl.returnCONTFEATURES()
ALLCATFEATURES = cl.returnCATFEATURES()

for feature in ALLCONTFEATURES:
	print(feature)
	cutoffs=BANDFEATURES[feature]
	data_train[feature+'Band']=data_train[feature].apply(ml.applyband,args=(cutoffs,))
	ml.actualvsfittedbyfactor(data_train,feature+'Band','cybfreq/')

for feature in ALLCATFEATURES:
	ml.actualvsfittedbyfactor(data_train,feature,'cybfreq/')

print(datetime.datetime.now())
print("Ending Program")