import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from operator import itemgetter
import os

import mlfunctions as ml
import columns as cl


print("Starting Program")
print(datetime.datetime.now())


COLUMNS= cl.returnCOLUMNS()
dtype=cl.returndtype()
BANDFEATURES=cl.returnBandFeatures()


def modelnas(LABEL,EXPOSURE,filenametrain,folder,errordist,formula):

	data_train = pd.read_csv(filenametrain, skipinitialspace=True, skiprows=1, names=COLUMNS, dtype=dtype)

	data_train=data_train[data_train[EXPOSURE]>0]

	CONTFEATURES = ['NumberPhysicians','Revenue','BothCoverages','CyberLimit','MedefenseLimit']
	# CATFEATURES = ['ProgramType','CyberLimit_Cat','MedefenseLimit_Cat']
	CATFEATURES = []


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
	data_train['missing_NumberPhysicians']=np.where(data_train['NumberPhysicians']<=0, 1, 0)
	data_train['missing_Revenue']=np.where(data_train['Revenue']<=0, 1, 0)
	data_train['log_CyberLimit']=np.where(data_train['CyberLimit']<=0,-1,np.log(data_train['CyberLimit']))
	data_train['log_MedefenseLimit']=np.where(data_train['MedefenseLimit']<=0,-1,np.log(data_train['MedefenseLimit']))
	data_train['log_Revenue']=np.where(data_train['Revenue']<=0,-1,np.log(data_train['Revenue']))
	data_train['log_NumberPhysicians']=np.where(data_train['NumberPhysicians']<=0,-1,np.log(data_train['NumberPhysicians']))
	data_train['ProgramType_Nonopen']=np.where(data_train['ProgramType'] != 'open', 1, 0)


	prediction_glm = ml.GLM(data_train, CONTFEATURES,CATFEATURES,LABEL,EXPOSURE,formula,folder,0,errordist)


	data_train['weight'] = data_train[EXPOSURE]
	data_train['predictedValue'] = prediction_glm*data_train['weight']
	data_train['actualValue'] = data_train[LABEL]

	ALLCONTFEATURES = cl.returnCONTFEATURES()
	ALLCATFEATURES = cl.returnCATFEATURES()

	for feature in ALLCONTFEATURES:
		print(feature)
		cutoffs=BANDFEATURES[feature]
		data_train[feature+'Band']=data_train[feature].apply(ml.applyband,args=(cutoffs,))
		ml.actualvsfittedbyfactor(data_train,feature+'Band',folder)

	for feature in ALLCATFEATURES:
		ml.actualvsfittedbyfactor(data_train,feature,folder)

modelnas(
	LABEL = "CYBClaimCount",
	EXPOSURE="CYBExposure",
	filenametrain="nasmod.csv",
	folder = 'cybfreq/',
	errordist='poisson',
	formula = ' CYBClaimCount ~ missing_Revenue + log_Revenue + log_CyberLimit '
)

modelnas(LABEL = "CYBClaimCost",
	EXPOSURE="CYBClaimCount",
	filenametrain="nasmod.csv",
	folder = 'cybsev/',
	errordist='gamma',
	formula = 'CYBClaimCost ~ missing_NumberPhysicians + log_NumberPhysicians '
)

modelnas(
	LABEL = "MEDClaimCount",
	EXPOSURE="MEDExposure",
	filenametrain="nasmod.csv",
	folder = 'medfreq/',
	errordist='poisson',
	formula = 'MEDClaimCount ~  missing_Revenue + log_Revenue + BothCoverages '
)

modelnas(
	LABEL = "MEDClaimCost",
	EXPOSURE="MEDClaimCount",
	filenametrain="nasmod.csv",
	folder = 'medsev/',
	errordist='gamma',
	formula = 'MEDClaimCost ~ missing_Revenue + log_Revenue  + log_MedefenseLimit  '
)


# formula = LABEL+' ~  missing_NumberPhysicians + missing_Revenue + log_CyberLimit + log_MedefenseLimit + log_Revenue + log_NumberPhysicians + BothCoverages '


print(datetime.datetime.now())
print("Ending Program")