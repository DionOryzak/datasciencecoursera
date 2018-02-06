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

filenametrain="mock.csv"
# filenametest="acertaremediation.csv"

print("Starting Program")
print(datetime.datetime.now())

data_train = pd.read_csv(filenametrain, skipinitialspace=True, skiprows=1, names=COLUMNS, dtype=dtype)
# data_test = pd.read_csv(filenametest, skipinitialspace=True, skiprows=1, names=COLUMNS, dtype=dtype)
print('train records: '+str(len(data_train)))
# print('test records: '+str(len(data_test)))

# avgweight = data_train[WEIGHT].mean()
# print(avgweight)

# data_train[WEIGHT]=data_train[WEIGHT]/avgweight

CONTFEATURES = cl.returnCONTFEATURES()
CATFEATURES = cl.returnCATFEATURES()

# Keep only records where Miles > 0 AND Reported_Accidents >= 0
data_train=data_train.loc[data_train['Miles'] >0 & data_train['Reported_Accidents'] >=0 ]

# Rename City #N/A to "City NA"
data_train['City']= np.where(data_train['City']=='#N/A',"City NA",data_train['City'])

# Slice Segment, city and Product
data_train.Segment = df.Segment.str[8:]
data_train.City = df.City.str[5:]
data_train.Product = df.Product.str[8:]

DUMMYFEATURES=[]
for feature in CATFEATURES:
	dummiesTrain=pd.get_dummies(data_train_segment[feature],prefix=feature)
	temp=dummiesTrain.sum()	
	dummiesTrain=dummiesTrain[temp[temp<temp.max()].index.values]
	if(len(dummiesTrain.columns)>0):
		dummiesTrain.columns = dummiesTrain.columns.str.replace('\s+', '_')
		data_train_segment=pd.concat([data_train_segment,dummiesTrain],axis=1)
		DUMMYFEATURES += list(dummiesTrain)
print(DUMMYFEATURES)

# data_train_segment['ASSETVALUE_BAND_02_0_5m_1m'] = data_train_segment['ASSETVALUE_BAND_02:_0.5m_-_1.0m']
# data_train_segment['ASSETVALUE_BAND_03_1m_1_5m'] = data_train_segment['ASSETVALUE_BAND_03:_1.0m_-_1.5m']
# data_train_segment['ASSETVALUE_BAND_04_1_5m_2m'] = data_train_segment['ASSETVALUE_BAND_04:_1.5m_-_2.0m']
# data_train_segment['ASSETVALUE_BAND_05_2m_2_5m'] = data_train_segment['ASSETVALUE_BAND_05:_2.0m_-_2.5m']
# data_train_segment['ASSETVALUE_BAND_06_2_5m_3m'] = data_train_segment['ASSETVALUE_BAND_06:_2.5m_-_3.0m']
# data_train_segment['ASSETVALUE_BAND_07_3m_3_5m'] = data_train_segment['ASSETVALUE_BAND_07:_3.0m_-_3.5m']
# data_train_segment['ASSETVALUE_BAND_08_3_5m_5m'] = data_train_segment['ASSETVALUE_BAND_08:_3.5m_-_5.0m']
# data_train_segment['ASSETVALUE_BAND_09_5m_10m'] = data_train_segment['ASSETVALUE_BAND_09:_5.0m_-_10.0m']
# data_train_segment['ASSETVALUE_BAND_10_10m_20m'] = data_train_segment['ASSETVALUE_BAND_10:_10.0m_-_20.0m']
# data_train_segment['ASSETVALUE_BAND_11_20m_high'] = data_train_segment['ASSETVALUE_BAND_11:_20.0m_-_high']
# data_train_segment['ASSETVALUE_BAND_99_Unknown'] = data_train_segment['ASSETVALUE_BAND_99:_Unknown']
# # ASSETVALUE_BAND_02_0_5m_1m + ASSETVALUE_BAND_03_1m_1_5m + ASSETVALUE_BAND_04_1_5m_2m + ASSETVALUE_BAND_05_2m_2_5m + ASSETVALUE_BAND_06_2_5m_3m + ASSETVALUE_BAND_07_3m_3_5m + ASSETVALUE_BAND_08_3_5m_5m + ASSETVALUE_BAND_09_5m_10m + ASSETVALUE_BAND_10_10m_20m + ASSETVALUE_BAND_11_20m_high + ASSETVALUE_BAND_99_Unknown

# data_train_segment['RiskState_grp1']=(data_train_segment['RiskState_SA']+data_train_segment['RiskState_TAS']+data_train_segment['RiskState_VIC'])
# data_train_segment['ASSETVALUE_BAND_grp1']=(data_train_segment['ASSETVALUE_BAND_06_2_5m_3m']+data_train_segment['ASSETVALUE_BAND_07_3m_3_5m']+data_train_segment['ASSETVALUE_BAND_08_3_5m_5m']+data_train_segment['ASSETVALUE_BAND_99_Unknown'])
# data_train_segment['ASSETVALUE_BAND_grp2']=(data_train_segment['ASSETVALUE_BAND_06_2_5m_3m']+data_train_segment['ASSETVALUE_BAND_07_3m_3_5m']+data_train_segment['ASSETVALUE_BAND_08_3_5m_5m']+data_train_segment['ASSETVALUE_BAND_09_5m_10m']+data_train_segment['ASSETVALUE_BAND_10_10m_20m']+data_train_segment['ASSETVALUE_BAND_11_20m_high'])

# modeltype='wf'
# LABEL = 'LABEL_'+modeltype
# WEIGHT = 'WEIGHT_'+modeltype
# if not os.path.exists('glm_'+modeltype+'_'+segment+'/'):
# 	os.makedirs('glm_'+modeltype+'_'+segment+'/')
# formula = 'LABEL_'+modeltype+' ~   QTR_2 + QTR_3 + cd_coverage_04C + cd_coverage_04D + cd_coverage_17C + cd_coverage_22A + cd_coverage_24A + RiskState_grp1 + ASSETVALUE_BAND_03_1m_1_5m + ASSETVALUE_BAND_04_1_5m_2m + ASSETVALUE_BAND_05_2m_2_5m + ASSETVALUE_BAND_06_2_5m_3m + ASSETVALUE_BAND_07_3m_3_5m + ASSETVALUE_BAND_08_3_5m_5m + ASSETVALUE_BAND_09_5m_10m + ASSETVALUE_BAND_10_10m_20m + ASSETVALUE_BAND_11_20m_high + Business_Prop_Hazard_Missing '
# prediction_glm_train, prediction_glm_test = ml.GLM(data_train_segment, data_train_segment, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm_'+modeltype+'_'+segment+'/',0,'poisson')
# # ml.avecharts(data_train_segment,WEIGHT,prediction_glm_train,LABEL,CONTFEATURES,BANDFEATURES,CATFEATURES,segment,modeltype)
# prediction_glm_train.to_csv('glm_'+modeltype+'_'+segment+'/'+modeltype+'_'+segment+'.csv')



print(datetime.datetime.now())
print("Ending Program")
