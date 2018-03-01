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
LABEL = "lossratio"
WEIGHT = "LCCL_OCP"

filenametrain="TRAIN1.csv"
filenametest="TEST1.csv"

print("Starting Program")
print(datetime.datetime.now())

data_train = pd.read_csv(filenametrain, skipinitialspace=True, skiprows=1, names=COLUMNS, dtype=dtype)
data_test = pd.read_csv(filenametest, skipinitialspace=True, skiprows=1, names=COLUMNS, dtype=dtype)


avgweight = data_train[WEIGHT].mean()
# print(avgweight)

data_train[WEIGHT]=data_train[WEIGHT]/avgweight
data_test[WEIGHT]= data_test[WEIGHT]/avgweight


# text_file = open('ENG_FUEL_CD.txt', "w")
# text_file.write(str(set(data_train['ENG_FUEL_CD'])))
# text_file.close() 

# text_file = open('ENG_MDL_CD.txt', "w")
# text_file.write(str(set(data_train['ENG_MDL_CD'])))
# text_file.close()  

# text_file = open('MAK_NM.txt', "w")
# text_file.write(str(set(data_train['MAK_NM'])))
# text_file.close()  

# text_file = open('NADA_BODY1.txt', "w")
# text_file.write(str(set(data_train['NADA_BODY1'])))
# text_file.close()  

# text_file = open('PLNT_CD.txt', "w")
# text_file.write(str(set(data_train['PLNT_CD'])))
# text_file.close()  


# # Step 1
# # Run initital GLM

# CONTFEATURES = []
# CATFEATURES = ['TRK_CAB_CNFG_CD','ENG_MDL_CD']
# # Include only catfeatures needed for GLM

# DUMMYFEATURES=[]
# for feature in CATFEATURES:
# 	dummiesTrain=pd.get_dummies(data_train[feature],prefix=feature)
# 	dummiesTest=pd.get_dummies(data_test[feature],prefix=feature)
# 	temp=dummiesTrain.sum()	
# 	# If column appears in data_train but not data_test we need to add it to data_test as a column of zeroes
# 	for k in ml.returnNotMatches(dummiesTrain.columns.values,dummiesTest.columns.values)[0]:
# 		dummiesTest[k]=0	
# 	dummiesTrain=dummiesTrain[temp[temp<temp.max()].index.values]
# 	dummiesTest=dummiesTest[temp[temp<temp.max()].index.values]
# 	if(len(dummiesTrain.columns)>0):
# 		dummiesTrain.columns = dummiesTrain.columns.str.replace('\s+', '_')
# 		dummiesTest.columns = dummiesTest.columns.str.replace('\s+', '_')
# 		data_train=pd.concat([data_train,dummiesTrain],axis=1)
# 		data_test=pd.concat([data_test,dummiesTest],axis=1)
# 		DUMMYFEATURES += list(dummiesTrain)

# # print(data_train.groupby(['TRK_CAB_CNFG_CD'], as_index=False)[WEIGHT].sum())

# # Create custom variables for GLM
# data_train['TRK_GRSS_VEH_WGHT_RATG_CD_gt6']=np.where(data_train['TRK_GRSS_VEH_WGHT_RATG_CD']>6, 1, 0)
# data_test['TRK_GRSS_VEH_WGHT_RATG_CD_gt6'] =np.where( data_test['TRK_GRSS_VEH_WGHT_RATG_CD']>6, 1, 0)
# data_train['TRK_CAB_CNFG_CD_CONSUVSPE']=(data_train['TRK_CAB_CNFG_CD_CON']+data_train['TRK_CAB_CNFG_CD_SUV']+data_train['TRK_CAB_CNFG_CD_SPE'])
# data_test['TRK_CAB_CNFG_CD_CONSUVSPE']=(data_test['TRK_CAB_CNFG_CD_CON']+data_test['TRK_CAB_CNFG_CD_SUV']+data_test['TRK_CAB_CNFG_CD_SPE'])
# data_train['TRK_CAB_CNFG_CD_CRWTHICFWHCB']=(data_train['TRK_CAB_CNFG_CD_CRW']+data_train['TRK_CAB_CNFG_CD_THI']+data_train['TRK_CAB_CNFG_CD_CFW']+data_train['TRK_CAB_CNFG_CD_HCB'])
# data_test['TRK_CAB_CNFG_CD_CRWTHICFWHCB']=(data_test['TRK_CAB_CNFG_CD_CRW']+data_test['TRK_CAB_CNFG_CD_THI']+data_test['TRK_CAB_CNFG_CD_CFW']+data_test['TRK_CAB_CNFG_CD_HCB'])
# data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_020105']+data_train['ENG_MDL_CD_165230']+data_train['ENG_MDL_CD_020120']+data_train['ENG_MDL_CD_165225']+data_train['ENG_MDL_CD_220019']+data_train['ENG_MDL_CD_140167']+data_train['ENG_MDL_CD_165185']+data_train['ENG_MDL_CD_195050']+data_train['ENG_MDL_CD_245010']+data_train['ENG_MDL_CD_140330']+data_train['ENG_MDL_CD_020026']+data_train['ENG_MDL_CD_140173']+data_train['ENG_MDL_CD_195001']+data_train['ENG_MDL_CD_010038']+data_train['ENG_MDL_CD_140027']+data_train['ENG_MDL_CD_020075']+data_train['ENG_MDL_CD_020072']+data_train['ENG_MDL_CD_080012']+data_train['ENG_MDL_CD_090011']+data_train['ENG_MDL_CD_180012']+data_train['ENG_MDL_CD_160030']+data_train['ENG_MDL_CD_140168']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_020085']+data_train['ENG_MDL_CD_120010']+data_train['ENG_MDL_CD_160020']+data_train['ENG_MDL_CD_110030']+data_train['ENG_MDL_CD_090020']+data_train['ENG_MDL_CD_090077']+data_train['ENG_MDL_CD_090051']+data_train['ENG_MDL_CD_180020']+data_train['ENG_MDL_CD_180010']+data_train['ENG_MDL_CD_020060']+data_train['ENG_MDL_CD_050020']+data_train['ENG_MDL_CD_180052']+data_train['ENG_MDL_CD_150020']+data_train['ENG_MDL_CD_090045']+data_train['ENG_MDL_CD_090043']+data_train['ENG_MDL_CD_140090']+data_train['ENG_MDL_CD_140220']+data_train['ENG_MDL_CD_140115']+data_train['ENG_MDL_CD_210080']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_140025']+data_train['ENG_MDL_CD_140200']+data_train['ENG_MDL_CD_140055']+data_train['ENG_MDL_CD_040010']+data_train['ENG_MDL_CD_020100']+data_train['ENG_MDL_CD_020090']+data_train['ENG_MDL_CD_165160']+data_train['ENG_MDL_CD_090040']+data_train['ENG_MDL_CD_210050']+data_train['ENG_MDL_CD_140195']+data_train['ENG_MDL_CD_200010']+data_train['ENG_MDL_CD_150010']+data_train['ENG_MDL_CD_210030']+data_train['ENG_MDL_CD_160060']+data_train['ENG_MDL_CD_150060']+data_train['ENG_MDL_CD_010050']+data_train['ENG_MDL_CD_210100']+data_train['ENG_MDL_CD_140215']+data_train['ENG_MDL_CD_140050']+data_train['ENG_MDL_CD_210070']+data_train['ENG_MDL_CD_040030']+data_train['ENG_MDL_CD_210040']+data_train['ENG_MDL_CD_140125']+data_train['ENG_MDL_CD_210090']+data_train['ENG_MDL_CD_210020']+data_train['ENG_MDL_CD_140020']+data_train['ENG_MDL_CD_140045']+data_train['ENG_MDL_CD_140040']+data_train['ENG_MDL_CD_140234']+data_train['ENG_MDL_CD_210060']+data_train['ENG_MDL_CD_165007']+data_train['ENG_MDL_CD_210165']+data_train['ENG_MDL_CD_165030']+data_train['ENG_MDL_CD_160012'])
# data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_140160']+data_train['ENG_MDL_CD_140191']+data_train['ENG_MDL_CD_140210']+data_train['ENG_MDL_CD_140028']+data_train['ENG_MDL_CD_165110']+data_train['ENG_MDL_CD_140205']+data_train['ENG_MDL_CD_110050']+data_train['ENG_MDL_CD_090070']+data_train['ENG_MDL_CD_165090']+data_train['ENG_MDL_CD_090075']+data_train['ENG_MDL_CD_090030']+data_train['ENG_MDL_CD_210142']+data_train['ENG_MDL_CD_210143']+data_train['ENG_MDL_CD_180040']+data_train['ENG_MDL_CD_090010']+data_train['ENG_MDL_CD_220028']+data_train['ENG_MDL_CD_110028']+data_train['ENG_MDL_CD_080044']+data_train['ENG_MDL_CD_165140']+data_train['ENG_MDL_CD_150003']+data_train['ENG_MDL_CD_160022']+data_train['ENG_MDL_CD_210130']+data_train['ENG_MDL_CD_090085']+data_train['ENG_MDL_CD_140095']+data_train['ENG_MDL_CD_180030']+data_train['ENG_MDL_CD_080080']+data_train['ENG_MDL_CD_020010']+data_train['ENG_MDL_CD_080015']+data_train['ENG_MDL_CD_070025']+data_train['ENG_MDL_CD_165200']+data_train['ENG_MDL_CD_080100']+data_train['ENG_MDL_CD_160040']+data_train['ENG_MDL_CD_165040']+data_train['ENG_MDL_CD_220034']+data_train['ENG_MDL_CD_110040']+data_train['ENG_MDL_CD_090100']+data_train['ENG_MDL_CD_090071']+data_train['ENG_MDL_CD_140150']+data_train['ENG_MDL_CD_160025']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_165170']+data_train['ENG_MDL_CD_070018']+data_train['ENG_MDL_CD_160065']+data_train['ENG_MDL_CD_220029']+data_train['ENG_MDL_CD_180050']+data_train['ENG_MDL_CD_165070']+data_train['ENG_MDL_CD_200020']+data_train['ENG_MDL_CD_140182']+data_train['ENG_MDL_CD_140065']+data_train['ENG_MDL_CD_110070']+data_train['ENG_MDL_CD_220023']+data_train['ENG_MDL_CD_020091']+data_train['ENG_MDL_CD_180016']+data_train['ENG_MDL_CD_165130']+data_train['ENG_MDL_CD_165057']+data_train['ENG_MDL_CD_010033']+data_train['ENG_MDL_CD_220026']+data_train['ENG_MDL_CD_020027']+data_train['ENG_MDL_CD_165035']+data_train['ENG_MDL_CD_210122']+data_train['ENG_MDL_CD_010006']+data_train['ENG_MDL_CD_195022'])
# data_train['ENG_MDL_CD_grp3']=(data_train['ENG_MDL_CD_010030']+data_train['ENG_MDL_CD_140190']+data_train['ENG_MDL_CD_140175']+data_train['ENG_MDL_CD_165010']+data_train['ENG_MDL_CD_165100']+data_train['ENG_MDL_CD_010039']+data_train['ENG_MDL_CD_030060']+data_train['ENG_MDL_CD_140181']+data_train['ENG_MDL_CD_030020']+data_train['ENG_MDL_CD_165055']+data_train['ENG_MDL_CD_140145']+data_train['ENG_MDL_CD_165020']+data_train['ENG_MDL_CD_080020']+data_train['ENG_MDL_CD_140165']+data_train['ENG_MDL_CD_070035']+data_train['ENG_MDL_CD_210120']+data_train['ENG_MDL_CD_140130']+data_train['ENG_MDL_CD_140080']+data_train['ENG_MDL_CD_140135']+data_train['ENG_MDL_CD_090047']+data_train['ENG_MDL_CD_140194']+data_train['ENG_MDL_CD_140185']+data_train['ENG_MDL_CD_010080']+data_train['ENG_MDL_CD_210110']+data_train['ENG_MDL_CD_030015']+data_train['ENG_MDL_CD_030080']+data_train['ENG_MDL_CD_030030']+data_train['ENG_MDL_CD_030051']+data_train['ENG_MDL_CD_030105']+data_train['ENG_MDL_CD_030110']+data_train['ENG_MDL_CD_030070']+data_train['ENG_MDL_CD_030047']+data_train['ENG_MDL_CD_030040']+data_train['ENG_MDL_CD_030115']+data_train['ENG_MDL_CD_030050']+data_train['ENG_MDL_CD_030045'])
# data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_020105']+data_test['ENG_MDL_CD_165230']+data_test['ENG_MDL_CD_020120']+data_test['ENG_MDL_CD_165225']+data_test['ENG_MDL_CD_220019']+data_test['ENG_MDL_CD_140167']+data_test['ENG_MDL_CD_165185']+data_test['ENG_MDL_CD_195050']+data_test['ENG_MDL_CD_245010']+data_test['ENG_MDL_CD_140330']+data_test['ENG_MDL_CD_020026']+data_test['ENG_MDL_CD_140173']+data_test['ENG_MDL_CD_195001']+data_test['ENG_MDL_CD_010038']+data_test['ENG_MDL_CD_140027']+data_test['ENG_MDL_CD_020075']+data_test['ENG_MDL_CD_020072']+data_test['ENG_MDL_CD_080012']+data_test['ENG_MDL_CD_090011']+data_test['ENG_MDL_CD_180012']+data_test['ENG_MDL_CD_160030']+data_test['ENG_MDL_CD_140168']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_020085']+data_test['ENG_MDL_CD_120010']+data_test['ENG_MDL_CD_160020']+data_test['ENG_MDL_CD_110030']+data_test['ENG_MDL_CD_090020']+data_test['ENG_MDL_CD_090077']+data_test['ENG_MDL_CD_090051']+data_test['ENG_MDL_CD_180020']+data_test['ENG_MDL_CD_180010']+data_test['ENG_MDL_CD_020060']+data_test['ENG_MDL_CD_050020']+data_test['ENG_MDL_CD_180052']+data_test['ENG_MDL_CD_150020']+data_test['ENG_MDL_CD_090045']+data_test['ENG_MDL_CD_090043']+data_test['ENG_MDL_CD_140090']+data_test['ENG_MDL_CD_140220']+data_test['ENG_MDL_CD_140115']+data_test['ENG_MDL_CD_210080']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_140025']+data_test['ENG_MDL_CD_140200']+data_test['ENG_MDL_CD_140055']+data_test['ENG_MDL_CD_040010']+data_test['ENG_MDL_CD_020100']+data_test['ENG_MDL_CD_020090']+data_test['ENG_MDL_CD_165160']+data_test['ENG_MDL_CD_090040']+data_test['ENG_MDL_CD_210050']+data_test['ENG_MDL_CD_140195']+data_test['ENG_MDL_CD_200010']+data_test['ENG_MDL_CD_150010']+data_test['ENG_MDL_CD_210030']+data_test['ENG_MDL_CD_160060']+data_test['ENG_MDL_CD_150060']+data_test['ENG_MDL_CD_010050']+data_test['ENG_MDL_CD_210100']+data_test['ENG_MDL_CD_140215']+data_test['ENG_MDL_CD_140050']+data_test['ENG_MDL_CD_210070']+data_test['ENG_MDL_CD_040030']+data_test['ENG_MDL_CD_210040']+data_test['ENG_MDL_CD_140125']+data_test['ENG_MDL_CD_210090']+data_test['ENG_MDL_CD_210020']+data_test['ENG_MDL_CD_140020']+data_test['ENG_MDL_CD_140045']+data_test['ENG_MDL_CD_140040']+data_test['ENG_MDL_CD_140234']+data_test['ENG_MDL_CD_210060']+data_test['ENG_MDL_CD_165007']+data_test['ENG_MDL_CD_210165']+data_test['ENG_MDL_CD_165030']+data_test['ENG_MDL_CD_160012'])
# data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_140160']+data_test['ENG_MDL_CD_140191']+data_test['ENG_MDL_CD_140210']+data_test['ENG_MDL_CD_140028']+data_test['ENG_MDL_CD_165110']+data_test['ENG_MDL_CD_140205']+data_test['ENG_MDL_CD_110050']+data_test['ENG_MDL_CD_090070']+data_test['ENG_MDL_CD_165090']+data_test['ENG_MDL_CD_090075']+data_test['ENG_MDL_CD_090030']+data_test['ENG_MDL_CD_210142']+data_test['ENG_MDL_CD_210143']+data_test['ENG_MDL_CD_180040']+data_test['ENG_MDL_CD_090010']+data_test['ENG_MDL_CD_220028']+data_test['ENG_MDL_CD_110028']+data_test['ENG_MDL_CD_080044']+data_test['ENG_MDL_CD_165140']+data_test['ENG_MDL_CD_150003']+data_test['ENG_MDL_CD_160022']+data_test['ENG_MDL_CD_210130']+data_test['ENG_MDL_CD_090085']+data_test['ENG_MDL_CD_140095']+data_test['ENG_MDL_CD_180030']+data_test['ENG_MDL_CD_080080']+data_test['ENG_MDL_CD_020010']+data_test['ENG_MDL_CD_080015']+data_test['ENG_MDL_CD_070025']+data_test['ENG_MDL_CD_165200']+data_test['ENG_MDL_CD_080100']+data_test['ENG_MDL_CD_160040']+data_test['ENG_MDL_CD_165040']+data_test['ENG_MDL_CD_220034']+data_test['ENG_MDL_CD_110040']+data_test['ENG_MDL_CD_090100']+data_test['ENG_MDL_CD_090071']+data_test['ENG_MDL_CD_140150']+data_test['ENG_MDL_CD_160025']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_165170']+data_test['ENG_MDL_CD_070018']+data_test['ENG_MDL_CD_160065']+data_test['ENG_MDL_CD_220029']+data_test['ENG_MDL_CD_180050']+data_test['ENG_MDL_CD_165070']+data_test['ENG_MDL_CD_200020']+data_test['ENG_MDL_CD_140182']+data_test['ENG_MDL_CD_140065']+data_test['ENG_MDL_CD_110070']+data_test['ENG_MDL_CD_220023']+data_test['ENG_MDL_CD_020091']+data_test['ENG_MDL_CD_180016']+data_test['ENG_MDL_CD_165130']+data_test['ENG_MDL_CD_165057']+data_test['ENG_MDL_CD_010033']+data_test['ENG_MDL_CD_220026']+data_test['ENG_MDL_CD_020027']+data_test['ENG_MDL_CD_165035']+data_test['ENG_MDL_CD_210122']+data_test['ENG_MDL_CD_010006']+data_test['ENG_MDL_CD_195022'])
# data_test['ENG_MDL_CD_grp3']=(data_test['ENG_MDL_CD_010030']+data_test['ENG_MDL_CD_140190']+data_test['ENG_MDL_CD_140175']+data_test['ENG_MDL_CD_165010']+data_test['ENG_MDL_CD_165100']+data_test['ENG_MDL_CD_010039']+data_test['ENG_MDL_CD_030060']+data_test['ENG_MDL_CD_140181']+data_test['ENG_MDL_CD_030020']+data_test['ENG_MDL_CD_165055']+data_test['ENG_MDL_CD_140145']+data_test['ENG_MDL_CD_165020']+data_test['ENG_MDL_CD_080020']+data_test['ENG_MDL_CD_140165']+data_test['ENG_MDL_CD_070035']+data_test['ENG_MDL_CD_210120']+data_test['ENG_MDL_CD_140130']+data_test['ENG_MDL_CD_140080']+data_test['ENG_MDL_CD_140135']+data_test['ENG_MDL_CD_090047']+data_test['ENG_MDL_CD_140194']+data_test['ENG_MDL_CD_140185']+data_test['ENG_MDL_CD_010080']+data_test['ENG_MDL_CD_210110']+data_test['ENG_MDL_CD_030015']+data_test['ENG_MDL_CD_030080']+data_test['ENG_MDL_CD_030030']+data_test['ENG_MDL_CD_030051']+data_test['ENG_MDL_CD_030105']+data_test['ENG_MDL_CD_030110']+data_test['ENG_MDL_CD_030070']+data_test['ENG_MDL_CD_030047']+data_test['ENG_MDL_CD_030040']+data_test['ENG_MDL_CD_030115']+data_test['ENG_MDL_CD_030050']+data_test['ENG_MDL_CD_030045'])


# if not os.path.exists('glm/'):
# 	os.makedirs('glm/')
# formula= 'lossratio ~  cko_mdl_yr + Liab_Ex_Veh_Age + TRK_GRSS_VEH_WGHT_RATG_CD_gt6 + cko_fuel_d2  +  TRK_CAB_CNFG_CD_CONSUVSPE + TRK_CAB_CNFG_CD_CRWTHICFWHCB  + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 '
# prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm/',1)



# # Step 2
# # Run CONTFEATURES GBM

# ALLCONTFEATURES = cl.returnCONTFEATURES()
# CONTFEATURES = []
# CATFEATURES = ['TRK_CAB_CNFG_CD','ENG_MDL_CD']
# # Include only catfeatures needed for GLM

# DUMMYFEATURES=[]
# for feature in CATFEATURES:
# 	dummiesTrain=pd.get_dummies(data_train[feature],prefix=feature)
# 	dummiesTest=pd.get_dummies(data_test[feature],prefix=feature)
# 	temp=dummiesTrain.sum()	
# 	# If column appears in data_train but not data_test we need to add it to data_test as a column of zeroes
# 	for k in ml.returnNotMatches(dummiesTrain.columns.values,dummiesTest.columns.values)[0]:
# 		dummiesTest[k]=0	
# 	dummiesTrain=dummiesTrain[temp[temp<temp.max()].index.values]
# 	dummiesTest=dummiesTest[temp[temp<temp.max()].index.values]
# 	if(len(dummiesTrain.columns)>0):
# 		dummiesTrain.columns = dummiesTrain.columns.str.replace('\s+', '_')
# 		dummiesTest.columns = dummiesTest.columns.str.replace('\s+', '_')
# 		data_train=pd.concat([data_train,dummiesTrain],axis=1)
# 		data_test=pd.concat([data_test,dummiesTest],axis=1)
# 		DUMMYFEATURES += list(dummiesTrain)

# # Create custom variables for GLM
# data_train['TRK_GRSS_VEH_WGHT_RATG_CD_gt6']=np.where(data_train['TRK_GRSS_VEH_WGHT_RATG_CD']>6, 1, 0)
# data_test['TRK_GRSS_VEH_WGHT_RATG_CD_gt6'] =np.where( data_test['TRK_GRSS_VEH_WGHT_RATG_CD']>6, 1, 0)
# data_train['TRK_CAB_CNFG_CD_CONSUVSPE']=(data_train['TRK_CAB_CNFG_CD_CON']+data_train['TRK_CAB_CNFG_CD_SUV']+data_train['TRK_CAB_CNFG_CD_SPE'])
# data_test['TRK_CAB_CNFG_CD_CONSUVSPE']=(data_test['TRK_CAB_CNFG_CD_CON']+data_test['TRK_CAB_CNFG_CD_SUV']+data_test['TRK_CAB_CNFG_CD_SPE'])
# data_train['TRK_CAB_CNFG_CD_CRWTHICFWHCB']=(data_train['TRK_CAB_CNFG_CD_CRW']+data_train['TRK_CAB_CNFG_CD_THI']+data_train['TRK_CAB_CNFG_CD_CFW']+data_train['TRK_CAB_CNFG_CD_HCB'])
# data_test['TRK_CAB_CNFG_CD_CRWTHICFWHCB']=(data_test['TRK_CAB_CNFG_CD_CRW']+data_test['TRK_CAB_CNFG_CD_THI']+data_test['TRK_CAB_CNFG_CD_CFW']+data_test['TRK_CAB_CNFG_CD_HCB'])
# data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_020105']+data_train['ENG_MDL_CD_165230']+data_train['ENG_MDL_CD_020120']+data_train['ENG_MDL_CD_165225']+data_train['ENG_MDL_CD_220019']+data_train['ENG_MDL_CD_140167']+data_train['ENG_MDL_CD_165185']+data_train['ENG_MDL_CD_195050']+data_train['ENG_MDL_CD_245010']+data_train['ENG_MDL_CD_140330']+data_train['ENG_MDL_CD_020026']+data_train['ENG_MDL_CD_140173']+data_train['ENG_MDL_CD_195001']+data_train['ENG_MDL_CD_010038']+data_train['ENG_MDL_CD_140027']+data_train['ENG_MDL_CD_020075']+data_train['ENG_MDL_CD_020072']+data_train['ENG_MDL_CD_080012']+data_train['ENG_MDL_CD_090011']+data_train['ENG_MDL_CD_180012']+data_train['ENG_MDL_CD_160030']+data_train['ENG_MDL_CD_140168']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_020085']+data_train['ENG_MDL_CD_120010']+data_train['ENG_MDL_CD_160020']+data_train['ENG_MDL_CD_110030']+data_train['ENG_MDL_CD_090020']+data_train['ENG_MDL_CD_090077']+data_train['ENG_MDL_CD_090051']+data_train['ENG_MDL_CD_180020']+data_train['ENG_MDL_CD_180010']+data_train['ENG_MDL_CD_020060']+data_train['ENG_MDL_CD_050020']+data_train['ENG_MDL_CD_180052']+data_train['ENG_MDL_CD_150020']+data_train['ENG_MDL_CD_090045']+data_train['ENG_MDL_CD_090043']+data_train['ENG_MDL_CD_140090']+data_train['ENG_MDL_CD_140220']+data_train['ENG_MDL_CD_140115']+data_train['ENG_MDL_CD_210080']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_140025']+data_train['ENG_MDL_CD_140200']+data_train['ENG_MDL_CD_140055']+data_train['ENG_MDL_CD_040010']+data_train['ENG_MDL_CD_020100']+data_train['ENG_MDL_CD_020090']+data_train['ENG_MDL_CD_165160']+data_train['ENG_MDL_CD_090040']+data_train['ENG_MDL_CD_210050']+data_train['ENG_MDL_CD_140195']+data_train['ENG_MDL_CD_200010']+data_train['ENG_MDL_CD_150010']+data_train['ENG_MDL_CD_210030']+data_train['ENG_MDL_CD_160060']+data_train['ENG_MDL_CD_150060']+data_train['ENG_MDL_CD_010050']+data_train['ENG_MDL_CD_210100']+data_train['ENG_MDL_CD_140215']+data_train['ENG_MDL_CD_140050']+data_train['ENG_MDL_CD_210070']+data_train['ENG_MDL_CD_040030']+data_train['ENG_MDL_CD_210040']+data_train['ENG_MDL_CD_140125']+data_train['ENG_MDL_CD_210090']+data_train['ENG_MDL_CD_210020']+data_train['ENG_MDL_CD_140020']+data_train['ENG_MDL_CD_140045']+data_train['ENG_MDL_CD_140040']+data_train['ENG_MDL_CD_140234']+data_train['ENG_MDL_CD_210060']+data_train['ENG_MDL_CD_165007']+data_train['ENG_MDL_CD_210165']+data_train['ENG_MDL_CD_165030']+data_train['ENG_MDL_CD_160012'])
# data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_140160']+data_train['ENG_MDL_CD_140191']+data_train['ENG_MDL_CD_140210']+data_train['ENG_MDL_CD_140028']+data_train['ENG_MDL_CD_165110']+data_train['ENG_MDL_CD_140205']+data_train['ENG_MDL_CD_110050']+data_train['ENG_MDL_CD_090070']+data_train['ENG_MDL_CD_165090']+data_train['ENG_MDL_CD_090075']+data_train['ENG_MDL_CD_090030']+data_train['ENG_MDL_CD_210142']+data_train['ENG_MDL_CD_210143']+data_train['ENG_MDL_CD_180040']+data_train['ENG_MDL_CD_090010']+data_train['ENG_MDL_CD_220028']+data_train['ENG_MDL_CD_110028']+data_train['ENG_MDL_CD_080044']+data_train['ENG_MDL_CD_165140']+data_train['ENG_MDL_CD_150003']+data_train['ENG_MDL_CD_160022']+data_train['ENG_MDL_CD_210130']+data_train['ENG_MDL_CD_090085']+data_train['ENG_MDL_CD_140095']+data_train['ENG_MDL_CD_180030']+data_train['ENG_MDL_CD_080080']+data_train['ENG_MDL_CD_020010']+data_train['ENG_MDL_CD_080015']+data_train['ENG_MDL_CD_070025']+data_train['ENG_MDL_CD_165200']+data_train['ENG_MDL_CD_080100']+data_train['ENG_MDL_CD_160040']+data_train['ENG_MDL_CD_165040']+data_train['ENG_MDL_CD_220034']+data_train['ENG_MDL_CD_110040']+data_train['ENG_MDL_CD_090100']+data_train['ENG_MDL_CD_090071']+data_train['ENG_MDL_CD_140150']+data_train['ENG_MDL_CD_160025']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_165170']+data_train['ENG_MDL_CD_070018']+data_train['ENG_MDL_CD_160065']+data_train['ENG_MDL_CD_220029']+data_train['ENG_MDL_CD_180050']+data_train['ENG_MDL_CD_165070']+data_train['ENG_MDL_CD_200020']+data_train['ENG_MDL_CD_140182']+data_train['ENG_MDL_CD_140065']+data_train['ENG_MDL_CD_110070']+data_train['ENG_MDL_CD_220023']+data_train['ENG_MDL_CD_020091']+data_train['ENG_MDL_CD_180016']+data_train['ENG_MDL_CD_165130']+data_train['ENG_MDL_CD_165057']+data_train['ENG_MDL_CD_010033']+data_train['ENG_MDL_CD_220026']+data_train['ENG_MDL_CD_020027']+data_train['ENG_MDL_CD_165035']+data_train['ENG_MDL_CD_210122']+data_train['ENG_MDL_CD_010006']+data_train['ENG_MDL_CD_195022'])
# data_train['ENG_MDL_CD_grp3']=(data_train['ENG_MDL_CD_010030']+data_train['ENG_MDL_CD_140190']+data_train['ENG_MDL_CD_140175']+data_train['ENG_MDL_CD_165010']+data_train['ENG_MDL_CD_165100']+data_train['ENG_MDL_CD_010039']+data_train['ENG_MDL_CD_030060']+data_train['ENG_MDL_CD_140181']+data_train['ENG_MDL_CD_030020']+data_train['ENG_MDL_CD_165055']+data_train['ENG_MDL_CD_140145']+data_train['ENG_MDL_CD_165020']+data_train['ENG_MDL_CD_080020']+data_train['ENG_MDL_CD_140165']+data_train['ENG_MDL_CD_070035']+data_train['ENG_MDL_CD_210120']+data_train['ENG_MDL_CD_140130']+data_train['ENG_MDL_CD_140080']+data_train['ENG_MDL_CD_140135']+data_train['ENG_MDL_CD_090047']+data_train['ENG_MDL_CD_140194']+data_train['ENG_MDL_CD_140185']+data_train['ENG_MDL_CD_010080']+data_train['ENG_MDL_CD_210110']+data_train['ENG_MDL_CD_030015']+data_train['ENG_MDL_CD_030080']+data_train['ENG_MDL_CD_030030']+data_train['ENG_MDL_CD_030051']+data_train['ENG_MDL_CD_030105']+data_train['ENG_MDL_CD_030110']+data_train['ENG_MDL_CD_030070']+data_train['ENG_MDL_CD_030047']+data_train['ENG_MDL_CD_030040']+data_train['ENG_MDL_CD_030115']+data_train['ENG_MDL_CD_030050']+data_train['ENG_MDL_CD_030045'])
# data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_020105']+data_test['ENG_MDL_CD_165230']+data_test['ENG_MDL_CD_020120']+data_test['ENG_MDL_CD_165225']+data_test['ENG_MDL_CD_220019']+data_test['ENG_MDL_CD_140167']+data_test['ENG_MDL_CD_165185']+data_test['ENG_MDL_CD_195050']+data_test['ENG_MDL_CD_245010']+data_test['ENG_MDL_CD_140330']+data_test['ENG_MDL_CD_020026']+data_test['ENG_MDL_CD_140173']+data_test['ENG_MDL_CD_195001']+data_test['ENG_MDL_CD_010038']+data_test['ENG_MDL_CD_140027']+data_test['ENG_MDL_CD_020075']+data_test['ENG_MDL_CD_020072']+data_test['ENG_MDL_CD_080012']+data_test['ENG_MDL_CD_090011']+data_test['ENG_MDL_CD_180012']+data_test['ENG_MDL_CD_160030']+data_test['ENG_MDL_CD_140168']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_020085']+data_test['ENG_MDL_CD_120010']+data_test['ENG_MDL_CD_160020']+data_test['ENG_MDL_CD_110030']+data_test['ENG_MDL_CD_090020']+data_test['ENG_MDL_CD_090077']+data_test['ENG_MDL_CD_090051']+data_test['ENG_MDL_CD_180020']+data_test['ENG_MDL_CD_180010']+data_test['ENG_MDL_CD_020060']+data_test['ENG_MDL_CD_050020']+data_test['ENG_MDL_CD_180052']+data_test['ENG_MDL_CD_150020']+data_test['ENG_MDL_CD_090045']+data_test['ENG_MDL_CD_090043']+data_test['ENG_MDL_CD_140090']+data_test['ENG_MDL_CD_140220']+data_test['ENG_MDL_CD_140115']+data_test['ENG_MDL_CD_210080']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_140025']+data_test['ENG_MDL_CD_140200']+data_test['ENG_MDL_CD_140055']+data_test['ENG_MDL_CD_040010']+data_test['ENG_MDL_CD_020100']+data_test['ENG_MDL_CD_020090']+data_test['ENG_MDL_CD_165160']+data_test['ENG_MDL_CD_090040']+data_test['ENG_MDL_CD_210050']+data_test['ENG_MDL_CD_140195']+data_test['ENG_MDL_CD_200010']+data_test['ENG_MDL_CD_150010']+data_test['ENG_MDL_CD_210030']+data_test['ENG_MDL_CD_160060']+data_test['ENG_MDL_CD_150060']+data_test['ENG_MDL_CD_010050']+data_test['ENG_MDL_CD_210100']+data_test['ENG_MDL_CD_140215']+data_test['ENG_MDL_CD_140050']+data_test['ENG_MDL_CD_210070']+data_test['ENG_MDL_CD_040030']+data_test['ENG_MDL_CD_210040']+data_test['ENG_MDL_CD_140125']+data_test['ENG_MDL_CD_210090']+data_test['ENG_MDL_CD_210020']+data_test['ENG_MDL_CD_140020']+data_test['ENG_MDL_CD_140045']+data_test['ENG_MDL_CD_140040']+data_test['ENG_MDL_CD_140234']+data_test['ENG_MDL_CD_210060']+data_test['ENG_MDL_CD_165007']+data_test['ENG_MDL_CD_210165']+data_test['ENG_MDL_CD_165030']+data_test['ENG_MDL_CD_160012'])
# data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_140160']+data_test['ENG_MDL_CD_140191']+data_test['ENG_MDL_CD_140210']+data_test['ENG_MDL_CD_140028']+data_test['ENG_MDL_CD_165110']+data_test['ENG_MDL_CD_140205']+data_test['ENG_MDL_CD_110050']+data_test['ENG_MDL_CD_090070']+data_test['ENG_MDL_CD_165090']+data_test['ENG_MDL_CD_090075']+data_test['ENG_MDL_CD_090030']+data_test['ENG_MDL_CD_210142']+data_test['ENG_MDL_CD_210143']+data_test['ENG_MDL_CD_180040']+data_test['ENG_MDL_CD_090010']+data_test['ENG_MDL_CD_220028']+data_test['ENG_MDL_CD_110028']+data_test['ENG_MDL_CD_080044']+data_test['ENG_MDL_CD_165140']+data_test['ENG_MDL_CD_150003']+data_test['ENG_MDL_CD_160022']+data_test['ENG_MDL_CD_210130']+data_test['ENG_MDL_CD_090085']+data_test['ENG_MDL_CD_140095']+data_test['ENG_MDL_CD_180030']+data_test['ENG_MDL_CD_080080']+data_test['ENG_MDL_CD_020010']+data_test['ENG_MDL_CD_080015']+data_test['ENG_MDL_CD_070025']+data_test['ENG_MDL_CD_165200']+data_test['ENG_MDL_CD_080100']+data_test['ENG_MDL_CD_160040']+data_test['ENG_MDL_CD_165040']+data_test['ENG_MDL_CD_220034']+data_test['ENG_MDL_CD_110040']+data_test['ENG_MDL_CD_090100']+data_test['ENG_MDL_CD_090071']+data_test['ENG_MDL_CD_140150']+data_test['ENG_MDL_CD_160025']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_165170']+data_test['ENG_MDL_CD_070018']+data_test['ENG_MDL_CD_160065']+data_test['ENG_MDL_CD_220029']+data_test['ENG_MDL_CD_180050']+data_test['ENG_MDL_CD_165070']+data_test['ENG_MDL_CD_200020']+data_test['ENG_MDL_CD_140182']+data_test['ENG_MDL_CD_140065']+data_test['ENG_MDL_CD_110070']+data_test['ENG_MDL_CD_220023']+data_test['ENG_MDL_CD_020091']+data_test['ENG_MDL_CD_180016']+data_test['ENG_MDL_CD_165130']+data_test['ENG_MDL_CD_165057']+data_test['ENG_MDL_CD_010033']+data_test['ENG_MDL_CD_220026']+data_test['ENG_MDL_CD_020027']+data_test['ENG_MDL_CD_165035']+data_test['ENG_MDL_CD_210122']+data_test['ENG_MDL_CD_010006']+data_test['ENG_MDL_CD_195022'])
# data_test['ENG_MDL_CD_grp3']=(data_test['ENG_MDL_CD_010030']+data_test['ENG_MDL_CD_140190']+data_test['ENG_MDL_CD_140175']+data_test['ENG_MDL_CD_165010']+data_test['ENG_MDL_CD_165100']+data_test['ENG_MDL_CD_010039']+data_test['ENG_MDL_CD_030060']+data_test['ENG_MDL_CD_140181']+data_test['ENG_MDL_CD_030020']+data_test['ENG_MDL_CD_165055']+data_test['ENG_MDL_CD_140145']+data_test['ENG_MDL_CD_165020']+data_test['ENG_MDL_CD_080020']+data_test['ENG_MDL_CD_140165']+data_test['ENG_MDL_CD_070035']+data_test['ENG_MDL_CD_210120']+data_test['ENG_MDL_CD_140130']+data_test['ENG_MDL_CD_140080']+data_test['ENG_MDL_CD_140135']+data_test['ENG_MDL_CD_090047']+data_test['ENG_MDL_CD_140194']+data_test['ENG_MDL_CD_140185']+data_test['ENG_MDL_CD_010080']+data_test['ENG_MDL_CD_210110']+data_test['ENG_MDL_CD_030015']+data_test['ENG_MDL_CD_030080']+data_test['ENG_MDL_CD_030030']+data_test['ENG_MDL_CD_030051']+data_test['ENG_MDL_CD_030105']+data_test['ENG_MDL_CD_030110']+data_test['ENG_MDL_CD_030070']+data_test['ENG_MDL_CD_030047']+data_test['ENG_MDL_CD_030040']+data_test['ENG_MDL_CD_030115']+data_test['ENG_MDL_CD_030050']+data_test['ENG_MDL_CD_030045'])


# if not os.path.exists('glm/'):
# 	os.makedirs('glm/')
# formula= 'lossratio ~  cko_mdl_yr + Liab_Ex_Veh_Age + TRK_GRSS_VEH_WGHT_RATG_CD_gt6 + cko_fuel_d2  +  TRK_CAB_CNFG_CD_CONSUVSPE + TRK_CAB_CNFG_CD_CRWTHICFWHCB  + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 '
# prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm/',0)

# data_train['weight'] = data_train[WEIGHT]
# data_train['predictedValue'] = prediction_glm_train*data_train['weight']
# data_train['actualValue'] = data_train[LABEL]*data_train['weight']
# data_test['weight'] = data_test[WEIGHT]
# data_test['predictedValue'] = prediction_glm_test*data_test['weight']
# data_test['actualValue'] = data_test[LABEL]*data_test['weight']

# y_train_residuals=np.asarray(data_train[LABEL]-prediction_glm_train)
# y_test_residuals=np.asarray(data_test[LABEL]-prediction_glm_test)

# numCONTFEATURES=len(ALLCONTFEATURES)
# print(numCONTFEATURES)

# # Set numFEATURESATATIME to 1 for initial run through, commenting out elements in CONTFEATURES that cause a conflict. 
# # When it runs all the way through set numFEATURESATATIME to 9

# numFEATURESATATIME=9
# for i in range(0,int((numCONTFEATURES/numFEATURESATATIME))) :

# 	if i<int((numCONTFEATURES/numFEATURESATATIME))-1:
# 		upper=i*numFEATURESATATIME+numFEATURESATATIME
# 	else:
# 		upper= numCONTFEATURES
# 	CONTFEATURES= ALLCONTFEATURES[i*numFEATURESATATIME:upper]


# 	X_train = np.asarray(data_train[CONTFEATURES])
# 	X_test = np.asarray(data_test[CONTFEATURES])

# 	if (numFEATURESATATIME>1) & (not os.path.exists('gbm'+str(i)+'/')) :
# 		os.makedirs('gbm'+str(i)+'/')
# 	print(CONTFEATURES)
# 	# try:
# 	prediction_residuals_train, prediction_residuals_test = ml.gbm(X_train, X_test, y_train_residuals, y_test_residuals,CONTFEATURES,[],'gbm'+str(i)+'/')
# 	# except: 
# 	# 	print ("ERROR with this GBM!!!!")
# 	# 	print(CONTFEATURES)

# for feature in ALLCONTFEATURES:
# 	print(feature)
# 	cutoffs=BANDFEATURES[feature]
# 	data_train[feature+'Band']=data_train[feature].apply(ml.applyband,args=(cutoffs,))
# 	data_test[feature+'Band']=data_test[feature].apply(ml.applyband,args=(cutoffs,))
# 	ml.actualvsfittedbyfactor(data_train,data_test,feature+'Band','glm/')	



# # # Step 3
# # # Run GLM from result of CONTFEATURES GBM

# CONTFEATURES = []
# CATFEATURES = ['TRK_CAB_CNFG_CD','ENG_MDL_CD']
# # Include only catfeatures needed for GLM

# DUMMYFEATURES=[]
# for feature in CATFEATURES:
# 	dummiesTrain=pd.get_dummies(data_train[feature],prefix=feature)
# 	dummiesTest=pd.get_dummies(data_test[feature],prefix=feature)
# 	temp=dummiesTrain.sum()	
# 	# If column appears in data_train but not data_test we need to add it to data_test as a column of zeroes
# 	for k in ml.returnNotMatches(dummiesTrain.columns.values,dummiesTest.columns.values)[0]:
# 		dummiesTest[k]=0	
# 	dummiesTrain=dummiesTrain[temp[temp<temp.max()].index.values]
# 	dummiesTest=dummiesTest[temp[temp<temp.max()].index.values]
# 	if(len(dummiesTrain.columns)>0):
# 		dummiesTrain.columns = dummiesTrain.columns.str.replace('\s+', '_')
# 		dummiesTest.columns = dummiesTest.columns.str.replace('\s+', '_')
# 		data_train=pd.concat([data_train,dummiesTrain],axis=1)
# 		data_test=pd.concat([data_test,dummiesTest],axis=1)
# 		DUMMYFEATURES += list(dummiesTrain)

# # Create custom variables for GLM
# data_train['TRK_GRSS_VEH_WGHT_RATG_CD_gt6']=np.where(data_train['TRK_GRSS_VEH_WGHT_RATG_CD']>6, 1, 0)
# data_test['TRK_GRSS_VEH_WGHT_RATG_CD_gt6'] =np.where( data_test['TRK_GRSS_VEH_WGHT_RATG_CD']>6, 1, 0)
# data_train['TRK_CAB_CNFG_CD_CONSUVSPE']=(data_train['TRK_CAB_CNFG_CD_CON']+data_train['TRK_CAB_CNFG_CD_SUV']+data_train['TRK_CAB_CNFG_CD_SPE'])
# data_test['TRK_CAB_CNFG_CD_CONSUVSPE']=(data_test['TRK_CAB_CNFG_CD_CON']+data_test['TRK_CAB_CNFG_CD_SUV']+data_test['TRK_CAB_CNFG_CD_SPE'])
# data_train['TRK_CAB_CNFG_CD_CRWTHICFWHCB']=(data_train['TRK_CAB_CNFG_CD_CRW']+data_train['TRK_CAB_CNFG_CD_THI']+data_train['TRK_CAB_CNFG_CD_CFW']+data_train['TRK_CAB_CNFG_CD_HCB'])
# data_test['TRK_CAB_CNFG_CD_CRWTHICFWHCB']=(data_test['TRK_CAB_CNFG_CD_CRW']+data_test['TRK_CAB_CNFG_CD_THI']+data_test['TRK_CAB_CNFG_CD_CFW']+data_test['TRK_CAB_CNFG_CD_HCB'])
# data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_020105']+data_train['ENG_MDL_CD_165230']+data_train['ENG_MDL_CD_020120']+data_train['ENG_MDL_CD_165225']+data_train['ENG_MDL_CD_220019']+data_train['ENG_MDL_CD_140167']+data_train['ENG_MDL_CD_165185']+data_train['ENG_MDL_CD_195050']+data_train['ENG_MDL_CD_245010']+data_train['ENG_MDL_CD_140330']+data_train['ENG_MDL_CD_020026']+data_train['ENG_MDL_CD_140173']+data_train['ENG_MDL_CD_195001']+data_train['ENG_MDL_CD_010038']+data_train['ENG_MDL_CD_140027']+data_train['ENG_MDL_CD_020075']+data_train['ENG_MDL_CD_020072']+data_train['ENG_MDL_CD_080012']+data_train['ENG_MDL_CD_090011']+data_train['ENG_MDL_CD_180012']+data_train['ENG_MDL_CD_160030']+data_train['ENG_MDL_CD_140168']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_020085']+data_train['ENG_MDL_CD_120010']+data_train['ENG_MDL_CD_160020']+data_train['ENG_MDL_CD_110030']+data_train['ENG_MDL_CD_090020']+data_train['ENG_MDL_CD_090077']+data_train['ENG_MDL_CD_090051']+data_train['ENG_MDL_CD_180020']+data_train['ENG_MDL_CD_180010']+data_train['ENG_MDL_CD_020060']+data_train['ENG_MDL_CD_050020']+data_train['ENG_MDL_CD_180052']+data_train['ENG_MDL_CD_150020']+data_train['ENG_MDL_CD_090045']+data_train['ENG_MDL_CD_090043']+data_train['ENG_MDL_CD_140090']+data_train['ENG_MDL_CD_140220']+data_train['ENG_MDL_CD_140115']+data_train['ENG_MDL_CD_210080']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_140025']+data_train['ENG_MDL_CD_140200']+data_train['ENG_MDL_CD_140055']+data_train['ENG_MDL_CD_040010']+data_train['ENG_MDL_CD_020100']+data_train['ENG_MDL_CD_020090']+data_train['ENG_MDL_CD_165160']+data_train['ENG_MDL_CD_090040']+data_train['ENG_MDL_CD_210050']+data_train['ENG_MDL_CD_140195']+data_train['ENG_MDL_CD_200010']+data_train['ENG_MDL_CD_150010']+data_train['ENG_MDL_CD_210030']+data_train['ENG_MDL_CD_160060']+data_train['ENG_MDL_CD_150060']+data_train['ENG_MDL_CD_010050']+data_train['ENG_MDL_CD_210100']+data_train['ENG_MDL_CD_140215']+data_train['ENG_MDL_CD_140050']+data_train['ENG_MDL_CD_210070']+data_train['ENG_MDL_CD_040030']+data_train['ENG_MDL_CD_210040']+data_train['ENG_MDL_CD_140125']+data_train['ENG_MDL_CD_210090']+data_train['ENG_MDL_CD_210020']+data_train['ENG_MDL_CD_140020']+data_train['ENG_MDL_CD_140045']+data_train['ENG_MDL_CD_140040']+data_train['ENG_MDL_CD_140234']+data_train['ENG_MDL_CD_210060']+data_train['ENG_MDL_CD_165007']+data_train['ENG_MDL_CD_210165']+data_train['ENG_MDL_CD_165030']+data_train['ENG_MDL_CD_160012'])
# data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_140160']+data_train['ENG_MDL_CD_140191']+data_train['ENG_MDL_CD_140210']+data_train['ENG_MDL_CD_140028']+data_train['ENG_MDL_CD_165110']+data_train['ENG_MDL_CD_140205']+data_train['ENG_MDL_CD_110050']+data_train['ENG_MDL_CD_090070']+data_train['ENG_MDL_CD_165090']+data_train['ENG_MDL_CD_090075']+data_train['ENG_MDL_CD_090030']+data_train['ENG_MDL_CD_210142']+data_train['ENG_MDL_CD_210143']+data_train['ENG_MDL_CD_180040']+data_train['ENG_MDL_CD_090010']+data_train['ENG_MDL_CD_220028']+data_train['ENG_MDL_CD_110028']+data_train['ENG_MDL_CD_080044']+data_train['ENG_MDL_CD_165140']+data_train['ENG_MDL_CD_150003']+data_train['ENG_MDL_CD_160022']+data_train['ENG_MDL_CD_210130']+data_train['ENG_MDL_CD_090085']+data_train['ENG_MDL_CD_140095']+data_train['ENG_MDL_CD_180030']+data_train['ENG_MDL_CD_080080']+data_train['ENG_MDL_CD_020010']+data_train['ENG_MDL_CD_080015']+data_train['ENG_MDL_CD_070025']+data_train['ENG_MDL_CD_165200']+data_train['ENG_MDL_CD_080100']+data_train['ENG_MDL_CD_160040']+data_train['ENG_MDL_CD_165040']+data_train['ENG_MDL_CD_220034']+data_train['ENG_MDL_CD_110040']+data_train['ENG_MDL_CD_090100']+data_train['ENG_MDL_CD_090071']+data_train['ENG_MDL_CD_140150']+data_train['ENG_MDL_CD_160025']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_165170']+data_train['ENG_MDL_CD_070018']+data_train['ENG_MDL_CD_160065']+data_train['ENG_MDL_CD_220029']+data_train['ENG_MDL_CD_180050']+data_train['ENG_MDL_CD_165070']+data_train['ENG_MDL_CD_200020']+data_train['ENG_MDL_CD_140182']+data_train['ENG_MDL_CD_140065']+data_train['ENG_MDL_CD_110070']+data_train['ENG_MDL_CD_220023']+data_train['ENG_MDL_CD_020091']+data_train['ENG_MDL_CD_180016']+data_train['ENG_MDL_CD_165130']+data_train['ENG_MDL_CD_165057']+data_train['ENG_MDL_CD_010033']+data_train['ENG_MDL_CD_220026']+data_train['ENG_MDL_CD_020027']+data_train['ENG_MDL_CD_165035']+data_train['ENG_MDL_CD_210122']+data_train['ENG_MDL_CD_010006']+data_train['ENG_MDL_CD_195022'])
# data_train['ENG_MDL_CD_grp3']=(data_train['ENG_MDL_CD_010030']+data_train['ENG_MDL_CD_140190']+data_train['ENG_MDL_CD_140175']+data_train['ENG_MDL_CD_165010']+data_train['ENG_MDL_CD_165100']+data_train['ENG_MDL_CD_010039']+data_train['ENG_MDL_CD_030060']+data_train['ENG_MDL_CD_140181']+data_train['ENG_MDL_CD_030020']+data_train['ENG_MDL_CD_165055']+data_train['ENG_MDL_CD_140145']+data_train['ENG_MDL_CD_165020']+data_train['ENG_MDL_CD_080020']+data_train['ENG_MDL_CD_140165']+data_train['ENG_MDL_CD_070035']+data_train['ENG_MDL_CD_210120']+data_train['ENG_MDL_CD_140130']+data_train['ENG_MDL_CD_140080']+data_train['ENG_MDL_CD_140135']+data_train['ENG_MDL_CD_090047']+data_train['ENG_MDL_CD_140194']+data_train['ENG_MDL_CD_140185']+data_train['ENG_MDL_CD_010080']+data_train['ENG_MDL_CD_210110']+data_train['ENG_MDL_CD_030015']+data_train['ENG_MDL_CD_030080']+data_train['ENG_MDL_CD_030030']+data_train['ENG_MDL_CD_030051']+data_train['ENG_MDL_CD_030105']+data_train['ENG_MDL_CD_030110']+data_train['ENG_MDL_CD_030070']+data_train['ENG_MDL_CD_030047']+data_train['ENG_MDL_CD_030040']+data_train['ENG_MDL_CD_030115']+data_train['ENG_MDL_CD_030050']+data_train['ENG_MDL_CD_030045'])
# data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_020105']+data_test['ENG_MDL_CD_165230']+data_test['ENG_MDL_CD_020120']+data_test['ENG_MDL_CD_165225']+data_test['ENG_MDL_CD_220019']+data_test['ENG_MDL_CD_140167']+data_test['ENG_MDL_CD_165185']+data_test['ENG_MDL_CD_195050']+data_test['ENG_MDL_CD_245010']+data_test['ENG_MDL_CD_140330']+data_test['ENG_MDL_CD_020026']+data_test['ENG_MDL_CD_140173']+data_test['ENG_MDL_CD_195001']+data_test['ENG_MDL_CD_010038']+data_test['ENG_MDL_CD_140027']+data_test['ENG_MDL_CD_020075']+data_test['ENG_MDL_CD_020072']+data_test['ENG_MDL_CD_080012']+data_test['ENG_MDL_CD_090011']+data_test['ENG_MDL_CD_180012']+data_test['ENG_MDL_CD_160030']+data_test['ENG_MDL_CD_140168']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_020085']+data_test['ENG_MDL_CD_120010']+data_test['ENG_MDL_CD_160020']+data_test['ENG_MDL_CD_110030']+data_test['ENG_MDL_CD_090020']+data_test['ENG_MDL_CD_090077']+data_test['ENG_MDL_CD_090051']+data_test['ENG_MDL_CD_180020']+data_test['ENG_MDL_CD_180010']+data_test['ENG_MDL_CD_020060']+data_test['ENG_MDL_CD_050020']+data_test['ENG_MDL_CD_180052']+data_test['ENG_MDL_CD_150020']+data_test['ENG_MDL_CD_090045']+data_test['ENG_MDL_CD_090043']+data_test['ENG_MDL_CD_140090']+data_test['ENG_MDL_CD_140220']+data_test['ENG_MDL_CD_140115']+data_test['ENG_MDL_CD_210080']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_140025']+data_test['ENG_MDL_CD_140200']+data_test['ENG_MDL_CD_140055']+data_test['ENG_MDL_CD_040010']+data_test['ENG_MDL_CD_020100']+data_test['ENG_MDL_CD_020090']+data_test['ENG_MDL_CD_165160']+data_test['ENG_MDL_CD_090040']+data_test['ENG_MDL_CD_210050']+data_test['ENG_MDL_CD_140195']+data_test['ENG_MDL_CD_200010']+data_test['ENG_MDL_CD_150010']+data_test['ENG_MDL_CD_210030']+data_test['ENG_MDL_CD_160060']+data_test['ENG_MDL_CD_150060']+data_test['ENG_MDL_CD_010050']+data_test['ENG_MDL_CD_210100']+data_test['ENG_MDL_CD_140215']+data_test['ENG_MDL_CD_140050']+data_test['ENG_MDL_CD_210070']+data_test['ENG_MDL_CD_040030']+data_test['ENG_MDL_CD_210040']+data_test['ENG_MDL_CD_140125']+data_test['ENG_MDL_CD_210090']+data_test['ENG_MDL_CD_210020']+data_test['ENG_MDL_CD_140020']+data_test['ENG_MDL_CD_140045']+data_test['ENG_MDL_CD_140040']+data_test['ENG_MDL_CD_140234']+data_test['ENG_MDL_CD_210060']+data_test['ENG_MDL_CD_165007']+data_test['ENG_MDL_CD_210165']+data_test['ENG_MDL_CD_165030']+data_test['ENG_MDL_CD_160012'])
# data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_140160']+data_test['ENG_MDL_CD_140191']+data_test['ENG_MDL_CD_140210']+data_test['ENG_MDL_CD_140028']+data_test['ENG_MDL_CD_165110']+data_test['ENG_MDL_CD_140205']+data_test['ENG_MDL_CD_110050']+data_test['ENG_MDL_CD_090070']+data_test['ENG_MDL_CD_165090']+data_test['ENG_MDL_CD_090075']+data_test['ENG_MDL_CD_090030']+data_test['ENG_MDL_CD_210142']+data_test['ENG_MDL_CD_210143']+data_test['ENG_MDL_CD_180040']+data_test['ENG_MDL_CD_090010']+data_test['ENG_MDL_CD_220028']+data_test['ENG_MDL_CD_110028']+data_test['ENG_MDL_CD_080044']+data_test['ENG_MDL_CD_165140']+data_test['ENG_MDL_CD_150003']+data_test['ENG_MDL_CD_160022']+data_test['ENG_MDL_CD_210130']+data_test['ENG_MDL_CD_090085']+data_test['ENG_MDL_CD_140095']+data_test['ENG_MDL_CD_180030']+data_test['ENG_MDL_CD_080080']+data_test['ENG_MDL_CD_020010']+data_test['ENG_MDL_CD_080015']+data_test['ENG_MDL_CD_070025']+data_test['ENG_MDL_CD_165200']+data_test['ENG_MDL_CD_080100']+data_test['ENG_MDL_CD_160040']+data_test['ENG_MDL_CD_165040']+data_test['ENG_MDL_CD_220034']+data_test['ENG_MDL_CD_110040']+data_test['ENG_MDL_CD_090100']+data_test['ENG_MDL_CD_090071']+data_test['ENG_MDL_CD_140150']+data_test['ENG_MDL_CD_160025']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_165170']+data_test['ENG_MDL_CD_070018']+data_test['ENG_MDL_CD_160065']+data_test['ENG_MDL_CD_220029']+data_test['ENG_MDL_CD_180050']+data_test['ENG_MDL_CD_165070']+data_test['ENG_MDL_CD_200020']+data_test['ENG_MDL_CD_140182']+data_test['ENG_MDL_CD_140065']+data_test['ENG_MDL_CD_110070']+data_test['ENG_MDL_CD_220023']+data_test['ENG_MDL_CD_020091']+data_test['ENG_MDL_CD_180016']+data_test['ENG_MDL_CD_165130']+data_test['ENG_MDL_CD_165057']+data_test['ENG_MDL_CD_010033']+data_test['ENG_MDL_CD_220026']+data_test['ENG_MDL_CD_020027']+data_test['ENG_MDL_CD_165035']+data_test['ENG_MDL_CD_210122']+data_test['ENG_MDL_CD_010006']+data_test['ENG_MDL_CD_195022'])
# data_test['ENG_MDL_CD_grp3']=(data_test['ENG_MDL_CD_010030']+data_test['ENG_MDL_CD_140190']+data_test['ENG_MDL_CD_140175']+data_test['ENG_MDL_CD_165010']+data_test['ENG_MDL_CD_165100']+data_test['ENG_MDL_CD_010039']+data_test['ENG_MDL_CD_030060']+data_test['ENG_MDL_CD_140181']+data_test['ENG_MDL_CD_030020']+data_test['ENG_MDL_CD_165055']+data_test['ENG_MDL_CD_140145']+data_test['ENG_MDL_CD_165020']+data_test['ENG_MDL_CD_080020']+data_test['ENG_MDL_CD_140165']+data_test['ENG_MDL_CD_070035']+data_test['ENG_MDL_CD_210120']+data_test['ENG_MDL_CD_140130']+data_test['ENG_MDL_CD_140080']+data_test['ENG_MDL_CD_140135']+data_test['ENG_MDL_CD_090047']+data_test['ENG_MDL_CD_140194']+data_test['ENG_MDL_CD_140185']+data_test['ENG_MDL_CD_010080']+data_test['ENG_MDL_CD_210110']+data_test['ENG_MDL_CD_030015']+data_test['ENG_MDL_CD_030080']+data_test['ENG_MDL_CD_030030']+data_test['ENG_MDL_CD_030051']+data_test['ENG_MDL_CD_030105']+data_test['ENG_MDL_CD_030110']+data_test['ENG_MDL_CD_030070']+data_test['ENG_MDL_CD_030047']+data_test['ENG_MDL_CD_030040']+data_test['ENG_MDL_CD_030115']+data_test['ENG_MDL_CD_030050']+data_test['ENG_MDL_CD_030045'])

# data_train['cko_max_gvwc_lt11k']=np.where(data_train['cko_max_gvwc']<11000,1,0)
# data_train['cko_min_msrp_ge150k']=np.where(data_train['cko_min_msrp']>=150000,1,0)
# data_train['cko_max_msrp_ge150k']=np.where(data_train['cko_max_msrp']>=150000,1,0)
# data_train['cko_wheelbase_mean_lt130']=np.where(data_train['cko_wheelbase_mean']<130,1,0)
# data_train['cko_wheelbase_max_150bw300']=np.where((150<=data_train['cko_wheelbase_max']) & (data_train['cko_wheelbase_max']<300),1,0)
# data_train['NADA_MSRP1_ge150k']=np.where(data_train['NADA_MSRP1']>=150000,1,0)
# data_train['WHL_CNT_4spline9']=np.minimum(np.maximum(4,data_train['WHL_CNT']),9)
# data_train['WHL_DRVN_CNT_ge6']=np.where(data_train['WHL_DRVN_CNT']>=6,1,0)

# data_test['cko_max_gvwc_lt11k']=np.where(data_test['cko_max_gvwc']<11000,1,0)
# data_test['cko_min_msrp_ge150k']=np.where(data_test['cko_min_msrp']>=150000,1,0)
# data_test['cko_max_msrp_ge150k']=np.where(data_test['cko_max_msrp']>=150000,1,0)
# data_test['cko_wheelbase_mean_lt130']=np.where(data_test['cko_wheelbase_mean']<130,1,0)
# data_test['cko_wheelbase_max_150bw300']=np.where((150<=data_test['cko_wheelbase_max']) & (data_test['cko_wheelbase_max']<300),1,0)
# data_test['NADA_MSRP1_ge150k']=np.where(data_test['NADA_MSRP1']>=150000,1,0)
# data_test['WHL_CNT_4spline9']=np.minimum(np.maximum(4,data_test['WHL_CNT']),9)
# data_test['WHL_DRVN_CNT_ge6']=np.where(data_test['WHL_DRVN_CNT']>=6,1,0)

# formulabase= 'lossratio ~  TRK_GRSS_VEH_WGHT_RATG_CD_gt6 + cko_fuel_d2  +  TRK_CAB_CNFG_CD_CONSUVSPE + TRK_CAB_CNFG_CD_CRWTHICFWHCB   '

# # Originally included these by mistake.
# # + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 cko_mdl_yr + Liab_Ex_Veh_Age + 
# forms=[]
# forms.append('  + BODY_STYLE_CD_d4 + BODY_STYLE_CD_d5 + BODY_STYLE_CD_d3 + BODY_STYLE_CD_d1 + body_cg + cko_eng_cylinders_d2 + cko_fuel_d2   ')
# forms.append('  + cko_eng_disp_min_d1 + cko_weight_minMS + duty_typ_me + door_cnt_bin + DOOR_CNT + DOOR_CNTMS + ENG_MFG_CD_d1 + ENG_MFG_CD_d2 + ENG_MFG_CD_d3 + ENG_MFG_CD_d7  ')
# forms.append('  + ENG_TRK_DUTY_TYP_CD_d1 + ENG_VLVS_PER_CLNDR_d1  + ENG_VLVS_PER_CLNDR_d2 + ENG_VLVS_TOTLMS + ENTERTAIN_CD_127   ')
# forms.append('  + fuel_gas + fuel_flx + mak_frght + MAK_NM_d2 + MAK_NM_d4 + SHIP_WGHT_LBSMS + RSTRNT_TYP_CD_d1 + TRANS_CD_d1 + TRK_CAB_CNFG_CD_d4 + TRK_CAB_CNFG_CD_d6 + turbo_super  ')
# forms.append('  + cko_max_gvwc_lt11k + cko_max_gvwcMS + cko_min_msrp_ge150k + cko_max_msrp_ge150k + cko_max_msrpMS + cko_min_msrpMS  ')
# forms.append('  + cko_wheelbase_mean_lt130 + cko_wheelbase_max_150bw300 + cko_wheelbase_maxMS + NADA_MSRP1_ge150k + WHL_CNT_4spline9 + WHL_DRVN_CNT_ge6  ')

# forms.append('  + ENG_MFG_CD_d3 + ENG_MFG_CD_d7 + ENG_TRK_DUTY_TYP_CD_d1 + fuel_gas + turbo_super + cko_max_msrpMS + cko_min_msrpMS + NADA_MSRP1_ge150k + WHL_DRVN_CNT_ge6 ')
# forms.append('  + ENG_MFG_CD_d3 + ENG_MFG_CD_d7  + fuel_gas + turbo_super + cko_max_msrpMS + cko_min_msrpMS + NADA_MSRP1_ge150k  ')
# forms.append('  + ENG_MFG_CD_d3 + ENG_MFG_CD_d7  + fuel_gas + turbo_super + cko_max_msrpMS   ')
# forms.append('  + ENG_MFG_CD_d7  + fuel_gas + turbo_super + cko_max_msrpMS   ')
# forms.append('  + ENG_MFG_CD_d3 + ENG_MFG_CD_d7  + fuel_gas + cko_max_msrpMS   ')
# forms.append('  + ENG_MFG_CD_d7  + fuel_gas + cko_max_msrpMS   ')
# # 12
# forms.append('  + cko_wheelbase_mean_lt130 + cko_eng_cylinders_d2 + ENG_MFG_CD_d7 + ENG_TRK_DUTY_TYP_CD_d1 + fuel_gas + mak_frght + SHIP_WGHT_LBSMS + cko_max_gvwc_lt11k ')
# forms.append('  + cko_wheelbase_mean_lt130 + cko_eng_cylinders_d2 + ENG_MFG_CD_d7 + ENG_TRK_DUTY_TYP_CD_d1 + fuel_gas + SHIP_WGHT_LBSMS + cko_max_gvwc_lt11k ')


# for i in range(13,len(forms)):
# 	formula=formulabase+forms[i]
# 	if not os.path.exists('glm'+str(i)+'/'):
# 		os.makedirs('glm'+str(i)+'/')
# 	prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm'+str(i)+'/',0)



# # Step 4
# # Run CATFEATURES GBM

# ALLCATFEATURES = cl.returnCATFEATURES()
# CONTFEATURES = []
# CATFEATURES = ['TRK_CAB_CNFG_CD','ENG_MDL_CD']
# # Include only catfeatures needed for GLM

# DUMMYFEATURES=[]
# for feature in CATFEATURES:
# 	dummiesTrain=pd.get_dummies(data_train[feature],prefix=feature)
# 	dummiesTest=pd.get_dummies(data_test[feature],prefix=feature)
# 	temp=dummiesTrain.sum()	
# 	# If column appears in data_train but not data_test we need to add it to data_test as a column of zeroes
# 	for k in ml.returnNotMatches(dummiesTrain.columns.values,dummiesTest.columns.values)[0]:
# 		dummiesTest[k]=0	
# 	dummiesTrain=dummiesTrain[temp[temp<temp.max()].index.values]
# 	dummiesTest=dummiesTest[temp[temp<temp.max()].index.values]
# 	if(len(dummiesTrain.columns)>0):
# 		dummiesTrain.columns = dummiesTrain.columns.str.replace('\s+', '_')
# 		dummiesTest.columns = dummiesTest.columns.str.replace('\s+', '_')
# 		data_train=pd.concat([data_train,dummiesTrain],axis=1)
# 		data_test=pd.concat([data_test,dummiesTest],axis=1)
# 		DUMMYFEATURES += list(dummiesTrain)

# # Create custom variables for GLM
# data_train['TRK_GRSS_VEH_WGHT_RATG_CD_gt6']=np.where(data_train['TRK_GRSS_VEH_WGHT_RATG_CD']>6, 1, 0)
# data_test['TRK_GRSS_VEH_WGHT_RATG_CD_gt6'] =np.where( data_test['TRK_GRSS_VEH_WGHT_RATG_CD']>6, 1, 0)
# data_train['TRK_CAB_CNFG_CD_CONSUVSPE']=(data_train['TRK_CAB_CNFG_CD_CON']+data_train['TRK_CAB_CNFG_CD_SUV']+data_train['TRK_CAB_CNFG_CD_SPE'])
# data_test['TRK_CAB_CNFG_CD_CONSUVSPE']=(data_test['TRK_CAB_CNFG_CD_CON']+data_test['TRK_CAB_CNFG_CD_SUV']+data_test['TRK_CAB_CNFG_CD_SPE'])
# data_train['TRK_CAB_CNFG_CD_CRWTHICFWHCB']=(data_train['TRK_CAB_CNFG_CD_CRW']+data_train['TRK_CAB_CNFG_CD_THI']+data_train['TRK_CAB_CNFG_CD_CFW']+data_train['TRK_CAB_CNFG_CD_HCB'])
# data_test['TRK_CAB_CNFG_CD_CRWTHICFWHCB']=(data_test['TRK_CAB_CNFG_CD_CRW']+data_test['TRK_CAB_CNFG_CD_THI']+data_test['TRK_CAB_CNFG_CD_CFW']+data_test['TRK_CAB_CNFG_CD_HCB'])
# data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_020105']+data_train['ENG_MDL_CD_165230']+data_train['ENG_MDL_CD_020120']+data_train['ENG_MDL_CD_165225']+data_train['ENG_MDL_CD_220019']+data_train['ENG_MDL_CD_140167']+data_train['ENG_MDL_CD_165185']+data_train['ENG_MDL_CD_195050']+data_train['ENG_MDL_CD_245010']+data_train['ENG_MDL_CD_140330']+data_train['ENG_MDL_CD_020026']+data_train['ENG_MDL_CD_140173']+data_train['ENG_MDL_CD_195001']+data_train['ENG_MDL_CD_010038']+data_train['ENG_MDL_CD_140027']+data_train['ENG_MDL_CD_020075']+data_train['ENG_MDL_CD_020072']+data_train['ENG_MDL_CD_080012']+data_train['ENG_MDL_CD_090011']+data_train['ENG_MDL_CD_180012']+data_train['ENG_MDL_CD_160030']+data_train['ENG_MDL_CD_140168']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_020085']+data_train['ENG_MDL_CD_120010']+data_train['ENG_MDL_CD_160020']+data_train['ENG_MDL_CD_110030']+data_train['ENG_MDL_CD_090020']+data_train['ENG_MDL_CD_090077']+data_train['ENG_MDL_CD_090051']+data_train['ENG_MDL_CD_180020']+data_train['ENG_MDL_CD_180010']+data_train['ENG_MDL_CD_020060']+data_train['ENG_MDL_CD_050020']+data_train['ENG_MDL_CD_180052']+data_train['ENG_MDL_CD_150020']+data_train['ENG_MDL_CD_090045']+data_train['ENG_MDL_CD_090043']+data_train['ENG_MDL_CD_140090']+data_train['ENG_MDL_CD_140220']+data_train['ENG_MDL_CD_140115']+data_train['ENG_MDL_CD_210080']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_140025']+data_train['ENG_MDL_CD_140200']+data_train['ENG_MDL_CD_140055']+data_train['ENG_MDL_CD_040010']+data_train['ENG_MDL_CD_020100']+data_train['ENG_MDL_CD_020090']+data_train['ENG_MDL_CD_165160']+data_train['ENG_MDL_CD_090040']+data_train['ENG_MDL_CD_210050']+data_train['ENG_MDL_CD_140195']+data_train['ENG_MDL_CD_200010']+data_train['ENG_MDL_CD_150010']+data_train['ENG_MDL_CD_210030']+data_train['ENG_MDL_CD_160060']+data_train['ENG_MDL_CD_150060']+data_train['ENG_MDL_CD_010050']+data_train['ENG_MDL_CD_210100']+data_train['ENG_MDL_CD_140215']+data_train['ENG_MDL_CD_140050']+data_train['ENG_MDL_CD_210070']+data_train['ENG_MDL_CD_040030']+data_train['ENG_MDL_CD_210040']+data_train['ENG_MDL_CD_140125']+data_train['ENG_MDL_CD_210090']+data_train['ENG_MDL_CD_210020']+data_train['ENG_MDL_CD_140020']+data_train['ENG_MDL_CD_140045']+data_train['ENG_MDL_CD_140040']+data_train['ENG_MDL_CD_140234']+data_train['ENG_MDL_CD_210060']+data_train['ENG_MDL_CD_165007']+data_train['ENG_MDL_CD_210165']+data_train['ENG_MDL_CD_165030']+data_train['ENG_MDL_CD_160012'])
# data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_140160']+data_train['ENG_MDL_CD_140191']+data_train['ENG_MDL_CD_140210']+data_train['ENG_MDL_CD_140028']+data_train['ENG_MDL_CD_165110']+data_train['ENG_MDL_CD_140205']+data_train['ENG_MDL_CD_110050']+data_train['ENG_MDL_CD_090070']+data_train['ENG_MDL_CD_165090']+data_train['ENG_MDL_CD_090075']+data_train['ENG_MDL_CD_090030']+data_train['ENG_MDL_CD_210142']+data_train['ENG_MDL_CD_210143']+data_train['ENG_MDL_CD_180040']+data_train['ENG_MDL_CD_090010']+data_train['ENG_MDL_CD_220028']+data_train['ENG_MDL_CD_110028']+data_train['ENG_MDL_CD_080044']+data_train['ENG_MDL_CD_165140']+data_train['ENG_MDL_CD_150003']+data_train['ENG_MDL_CD_160022']+data_train['ENG_MDL_CD_210130']+data_train['ENG_MDL_CD_090085']+data_train['ENG_MDL_CD_140095']+data_train['ENG_MDL_CD_180030']+data_train['ENG_MDL_CD_080080']+data_train['ENG_MDL_CD_020010']+data_train['ENG_MDL_CD_080015']+data_train['ENG_MDL_CD_070025']+data_train['ENG_MDL_CD_165200']+data_train['ENG_MDL_CD_080100']+data_train['ENG_MDL_CD_160040']+data_train['ENG_MDL_CD_165040']+data_train['ENG_MDL_CD_220034']+data_train['ENG_MDL_CD_110040']+data_train['ENG_MDL_CD_090100']+data_train['ENG_MDL_CD_090071']+data_train['ENG_MDL_CD_140150']+data_train['ENG_MDL_CD_160025']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_165170']+data_train['ENG_MDL_CD_070018']+data_train['ENG_MDL_CD_160065']+data_train['ENG_MDL_CD_220029']+data_train['ENG_MDL_CD_180050']+data_train['ENG_MDL_CD_165070']+data_train['ENG_MDL_CD_200020']+data_train['ENG_MDL_CD_140182']+data_train['ENG_MDL_CD_140065']+data_train['ENG_MDL_CD_110070']+data_train['ENG_MDL_CD_220023']+data_train['ENG_MDL_CD_020091']+data_train['ENG_MDL_CD_180016']+data_train['ENG_MDL_CD_165130']+data_train['ENG_MDL_CD_165057']+data_train['ENG_MDL_CD_010033']+data_train['ENG_MDL_CD_220026']+data_train['ENG_MDL_CD_020027']+data_train['ENG_MDL_CD_165035']+data_train['ENG_MDL_CD_210122']+data_train['ENG_MDL_CD_010006']+data_train['ENG_MDL_CD_195022'])
# data_train['ENG_MDL_CD_grp3']=(data_train['ENG_MDL_CD_010030']+data_train['ENG_MDL_CD_140190']+data_train['ENG_MDL_CD_140175']+data_train['ENG_MDL_CD_165010']+data_train['ENG_MDL_CD_165100']+data_train['ENG_MDL_CD_010039']+data_train['ENG_MDL_CD_030060']+data_train['ENG_MDL_CD_140181']+data_train['ENG_MDL_CD_030020']+data_train['ENG_MDL_CD_165055']+data_train['ENG_MDL_CD_140145']+data_train['ENG_MDL_CD_165020']+data_train['ENG_MDL_CD_080020']+data_train['ENG_MDL_CD_140165']+data_train['ENG_MDL_CD_070035']+data_train['ENG_MDL_CD_210120']+data_train['ENG_MDL_CD_140130']+data_train['ENG_MDL_CD_140080']+data_train['ENG_MDL_CD_140135']+data_train['ENG_MDL_CD_090047']+data_train['ENG_MDL_CD_140194']+data_train['ENG_MDL_CD_140185']+data_train['ENG_MDL_CD_010080']+data_train['ENG_MDL_CD_210110']+data_train['ENG_MDL_CD_030015']+data_train['ENG_MDL_CD_030080']+data_train['ENG_MDL_CD_030030']+data_train['ENG_MDL_CD_030051']+data_train['ENG_MDL_CD_030105']+data_train['ENG_MDL_CD_030110']+data_train['ENG_MDL_CD_030070']+data_train['ENG_MDL_CD_030047']+data_train['ENG_MDL_CD_030040']+data_train['ENG_MDL_CD_030115']+data_train['ENG_MDL_CD_030050']+data_train['ENG_MDL_CD_030045'])
# data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_020105']+data_test['ENG_MDL_CD_165230']+data_test['ENG_MDL_CD_020120']+data_test['ENG_MDL_CD_165225']+data_test['ENG_MDL_CD_220019']+data_test['ENG_MDL_CD_140167']+data_test['ENG_MDL_CD_165185']+data_test['ENG_MDL_CD_195050']+data_test['ENG_MDL_CD_245010']+data_test['ENG_MDL_CD_140330']+data_test['ENG_MDL_CD_020026']+data_test['ENG_MDL_CD_140173']+data_test['ENG_MDL_CD_195001']+data_test['ENG_MDL_CD_010038']+data_test['ENG_MDL_CD_140027']+data_test['ENG_MDL_CD_020075']+data_test['ENG_MDL_CD_020072']+data_test['ENG_MDL_CD_080012']+data_test['ENG_MDL_CD_090011']+data_test['ENG_MDL_CD_180012']+data_test['ENG_MDL_CD_160030']+data_test['ENG_MDL_CD_140168']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_020085']+data_test['ENG_MDL_CD_120010']+data_test['ENG_MDL_CD_160020']+data_test['ENG_MDL_CD_110030']+data_test['ENG_MDL_CD_090020']+data_test['ENG_MDL_CD_090077']+data_test['ENG_MDL_CD_090051']+data_test['ENG_MDL_CD_180020']+data_test['ENG_MDL_CD_180010']+data_test['ENG_MDL_CD_020060']+data_test['ENG_MDL_CD_050020']+data_test['ENG_MDL_CD_180052']+data_test['ENG_MDL_CD_150020']+data_test['ENG_MDL_CD_090045']+data_test['ENG_MDL_CD_090043']+data_test['ENG_MDL_CD_140090']+data_test['ENG_MDL_CD_140220']+data_test['ENG_MDL_CD_140115']+data_test['ENG_MDL_CD_210080']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_140025']+data_test['ENG_MDL_CD_140200']+data_test['ENG_MDL_CD_140055']+data_test['ENG_MDL_CD_040010']+data_test['ENG_MDL_CD_020100']+data_test['ENG_MDL_CD_020090']+data_test['ENG_MDL_CD_165160']+data_test['ENG_MDL_CD_090040']+data_test['ENG_MDL_CD_210050']+data_test['ENG_MDL_CD_140195']+data_test['ENG_MDL_CD_200010']+data_test['ENG_MDL_CD_150010']+data_test['ENG_MDL_CD_210030']+data_test['ENG_MDL_CD_160060']+data_test['ENG_MDL_CD_150060']+data_test['ENG_MDL_CD_010050']+data_test['ENG_MDL_CD_210100']+data_test['ENG_MDL_CD_140215']+data_test['ENG_MDL_CD_140050']+data_test['ENG_MDL_CD_210070']+data_test['ENG_MDL_CD_040030']+data_test['ENG_MDL_CD_210040']+data_test['ENG_MDL_CD_140125']+data_test['ENG_MDL_CD_210090']+data_test['ENG_MDL_CD_210020']+data_test['ENG_MDL_CD_140020']+data_test['ENG_MDL_CD_140045']+data_test['ENG_MDL_CD_140040']+data_test['ENG_MDL_CD_140234']+data_test['ENG_MDL_CD_210060']+data_test['ENG_MDL_CD_165007']+data_test['ENG_MDL_CD_210165']+data_test['ENG_MDL_CD_165030']+data_test['ENG_MDL_CD_160012'])
# data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_140160']+data_test['ENG_MDL_CD_140191']+data_test['ENG_MDL_CD_140210']+data_test['ENG_MDL_CD_140028']+data_test['ENG_MDL_CD_165110']+data_test['ENG_MDL_CD_140205']+data_test['ENG_MDL_CD_110050']+data_test['ENG_MDL_CD_090070']+data_test['ENG_MDL_CD_165090']+data_test['ENG_MDL_CD_090075']+data_test['ENG_MDL_CD_090030']+data_test['ENG_MDL_CD_210142']+data_test['ENG_MDL_CD_210143']+data_test['ENG_MDL_CD_180040']+data_test['ENG_MDL_CD_090010']+data_test['ENG_MDL_CD_220028']+data_test['ENG_MDL_CD_110028']+data_test['ENG_MDL_CD_080044']+data_test['ENG_MDL_CD_165140']+data_test['ENG_MDL_CD_150003']+data_test['ENG_MDL_CD_160022']+data_test['ENG_MDL_CD_210130']+data_test['ENG_MDL_CD_090085']+data_test['ENG_MDL_CD_140095']+data_test['ENG_MDL_CD_180030']+data_test['ENG_MDL_CD_080080']+data_test['ENG_MDL_CD_020010']+data_test['ENG_MDL_CD_080015']+data_test['ENG_MDL_CD_070025']+data_test['ENG_MDL_CD_165200']+data_test['ENG_MDL_CD_080100']+data_test['ENG_MDL_CD_160040']+data_test['ENG_MDL_CD_165040']+data_test['ENG_MDL_CD_220034']+data_test['ENG_MDL_CD_110040']+data_test['ENG_MDL_CD_090100']+data_test['ENG_MDL_CD_090071']+data_test['ENG_MDL_CD_140150']+data_test['ENG_MDL_CD_160025']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_165170']+data_test['ENG_MDL_CD_070018']+data_test['ENG_MDL_CD_160065']+data_test['ENG_MDL_CD_220029']+data_test['ENG_MDL_CD_180050']+data_test['ENG_MDL_CD_165070']+data_test['ENG_MDL_CD_200020']+data_test['ENG_MDL_CD_140182']+data_test['ENG_MDL_CD_140065']+data_test['ENG_MDL_CD_110070']+data_test['ENG_MDL_CD_220023']+data_test['ENG_MDL_CD_020091']+data_test['ENG_MDL_CD_180016']+data_test['ENG_MDL_CD_165130']+data_test['ENG_MDL_CD_165057']+data_test['ENG_MDL_CD_010033']+data_test['ENG_MDL_CD_220026']+data_test['ENG_MDL_CD_020027']+data_test['ENG_MDL_CD_165035']+data_test['ENG_MDL_CD_210122']+data_test['ENG_MDL_CD_010006']+data_test['ENG_MDL_CD_195022'])
# data_test['ENG_MDL_CD_grp3']=(data_test['ENG_MDL_CD_010030']+data_test['ENG_MDL_CD_140190']+data_test['ENG_MDL_CD_140175']+data_test['ENG_MDL_CD_165010']+data_test['ENG_MDL_CD_165100']+data_test['ENG_MDL_CD_010039']+data_test['ENG_MDL_CD_030060']+data_test['ENG_MDL_CD_140181']+data_test['ENG_MDL_CD_030020']+data_test['ENG_MDL_CD_165055']+data_test['ENG_MDL_CD_140145']+data_test['ENG_MDL_CD_165020']+data_test['ENG_MDL_CD_080020']+data_test['ENG_MDL_CD_140165']+data_test['ENG_MDL_CD_070035']+data_test['ENG_MDL_CD_210120']+data_test['ENG_MDL_CD_140130']+data_test['ENG_MDL_CD_140080']+data_test['ENG_MDL_CD_140135']+data_test['ENG_MDL_CD_090047']+data_test['ENG_MDL_CD_140194']+data_test['ENG_MDL_CD_140185']+data_test['ENG_MDL_CD_010080']+data_test['ENG_MDL_CD_210110']+data_test['ENG_MDL_CD_030015']+data_test['ENG_MDL_CD_030080']+data_test['ENG_MDL_CD_030030']+data_test['ENG_MDL_CD_030051']+data_test['ENG_MDL_CD_030105']+data_test['ENG_MDL_CD_030110']+data_test['ENG_MDL_CD_030070']+data_test['ENG_MDL_CD_030047']+data_test['ENG_MDL_CD_030040']+data_test['ENG_MDL_CD_030115']+data_test['ENG_MDL_CD_030050']+data_test['ENG_MDL_CD_030045'])


# formula = 'lossratio ~  cko_mdl_yr + Liab_Ex_Veh_Age + TRK_GRSS_VEH_WGHT_RATG_CD_gt6 + cko_fuel_d2  +  TRK_CAB_CNFG_CD_CONSUVSPE + TRK_CAB_CNFG_CD_CRWTHICFWHCB  + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + ENG_MFG_CD_d7  + fuel_gas + cko_max_msrpMS   '
# prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm/',2)

# data_train['weight'] = data_train[WEIGHT]
# data_train['predictedValue'] = prediction_glm_train*data_train['weight']
# data_train['actualValue'] = data_train[LABEL]*data_train['weight']
# data_test['weight'] = data_test[WEIGHT]
# data_test['predictedValue'] = prediction_glm_test*data_test['weight']
# data_test['actualValue'] = data_test[LABEL]*data_test['weight']

# y_train_residuals=np.asarray(data_train[LABEL]-prediction_glm_train)
# y_test_residuals=np.asarray(data_test[LABEL]-prediction_glm_test)

# numCATFEATURES=len(ALLCATFEATURES)
# print(numCATFEATURES)

# # Set numCATFEATURES to 1 initially, commenting out any CATFEATURES elements that conflict. 
# # When no CATFEATURES elements conflict anymore then set to 10
# numFEATURESATATIME=10
# for i in range(0,int((numCATFEATURES/numFEATURESATATIME))) :

# 	# print(i)
# 	if i<int((numCATFEATURES/numFEATURESATATIME))-1:
# 		upper=i*numFEATURESATATIME+numFEATURESATATIME
# 	else:
# 		upper= numCATFEATURES
# 	# print(upper)
# 	CATFEATURES= ALLCATFEATURES[i*numFEATURESATATIME:upper]

# 	X_train = data_train[WEIGHT]
# 	X_test = data_test[WEIGHT]

# 	DUMMYFEATURES=[]
# 	for feature in CATFEATURES:
# 		dummiesTrain=pd.get_dummies(data_train[feature],prefix=feature)
# 		dummiesTest=pd.get_dummies(data_test[feature],prefix=feature)
# 		temp=dummiesTrain.sum()	
# 		# If column appears in data_train but not data_test we need to add it to data_test as a column of zeroes
# 		for k in ml.returnNotMatches(dummiesTrain.columns.values,dummiesTest.columns.values)[0]:
# 			dummiesTest[k]=0	
# 		dummiesTrain=dummiesTrain[temp[temp<temp.max()].index.values]
# 		dummiesTest=dummiesTest[temp[temp<temp.max()].index.values]
# 		# print(dummiesTrain.columns)
# 		if(len(dummiesTrain.columns)>0):
# 			dummiesTrain.columns = dummiesTrain.columns.str.replace('\s+', '_')
# 			dummiesTest.columns = dummiesTest.columns.str.replace('\s+', '_')
# 			X_train=pd.concat([X_train,dummiesTrain],axis=1)
# 			X_test=pd.concat([X_test,dummiesTest],axis=1)
# 			DUMMYFEATURES += list(dummiesTrain)

# 	if(len(DUMMYFEATURES)>0):
# 		X_train = np.asarray(X_train[DUMMYFEATURES])
# 		X_test = np.asarray(X_test[DUMMYFEATURES])

# 		# print(y_train_residuals)
# 		# print(y_test_residuals)
# 		# print(X_train)
# 		# print(X_test)
# 		if (numFEATURESATATIME>1) & (not os.path.exists('gbm'+str(i)+'/')):
# 			os.makedirs('gbm'+str(i)+'/')
# 		print(CATFEATURES)
# 		try:
# 			prediction_residuals_train, prediction_residuals_test = ml.gbm(X_train, X_test, y_train_residuals, y_test_residuals,CONTFEATURES,DUMMYFEATURES,'gbm'+str(i)+'/')
# 		except: 
# 			print ("ERROR with this GBM!!!!")
# 			print(CATFEATURES)
# for feature in ALLCATFEATURES:
# 	ml.actualvsfittedbyfactor(data_train,data_test,feature,'glm/')	



# # Step 5
# # Run GLM from result of CATFEATURES GBM

# CONTFEATURES = []
# CATFEATURES = ['ENG_MDL_CD','TRK_CAB_CNFG_CD']
# CATFAARRAY = []
# CATFAARRAY.append(['ABS_BRK_DESC','BODY_STYLE_CD'])
# CATFAARRAY.append(['BODY_STYLE_DESC','cko_4wd','cko_antitheft','cko_dtrl'])
# CATFAARRAY.append(['cko_eng_cylinders','cko_fuel','cko_max_gvwc_Tr','cko_overdrive','cko_semi_flag','cko_turbo_super','DR_LGHT_OPT_CD','DRV_TYP_CD'])
# CATFAARRAY.append(['ENG_ASP_SUP_CHGR_CD','ENG_ASP_TRBL_CHGR_CD','ENG_ASP_VVTL_CD','ENG_BLCK_TYP_CD','ENG_CBRT_TYP_CD','ENG_CLNDR_RTR_CNT','ENG_FUEL_CD','ENG_FUEL_INJ_TYP_CD','ENG_HEAD_CNFG_CD'])
# CATFAARRAY.append(['ENG_MFG_CD'])
# CATFAARRAY.append(['ENG_TRK_DUTY_TYP_CD','ENTERTAIN_CD','ENTERTAIN_GP','MAK_NM'])
# CATFAARRAY.append(['MFG_DESC','NADA_BODY1'])
# CATFAARRAY.append(['OPT1_ENTERTAIN_CD','PLNT_CD','PLNT_CNTRY_NM','PWR_BRK_OPT_CD','REAR_TIRE_SIZE_DESC'])
# CATFAARRAY.append(['RSTRNT_TYP_CD','TLT_STRNG_WHL_OPT_CD','TRANS_CD','TRANS_OVERDRV_IND','TRK_BRK_TYP_CD','TRK_FRNT_AXL_CD','VINA_BODY_TYPE_CD'])
# CATFAARRAY.append(['cko_4wd','cko_fuel','ENG_FUEL_CD','ENG_MDL_CD','MAK_NM','NADA_BODY1','PLNT_CD','PWR_BRK_OPT_CD'])
# CATFAARRAY.append(['cko_fuel','ENG_FUEL_CD','MAK_NM','NADA_BODY1','PLNT_CD'])
# CATFAARRAY.append([])
# CATFAARRAY.append(['PLNT_CD'])
# CATFAARRAY.append(['PLNT_CD','NADA_BODY1'])
# CATFAARRAY.append(['PLNT_CD','MAK_NM'])
# CATFAARRAY.append(['PLNT_CD','ENG_FUEL_CD'])
# CATFAARRAY.append(['PLNT_CD','MAK_NM','ENG_FUEL_CD'])
# CATFAARRAY.append(['NADA_BODY1'])
# CATFAARRAY.append(['MAK_NM'])
# CATFAARRAY.append(['ENG_FUEL_CD'])
# CATFAARRAY.append(['PLNT_CD'])
# CATFAARRAY.append([])
# CATFAARRAY.append(['NADA_BODY1','MAK_NM','ENG_FUEL_CD','PLNT_CD'])
# CATFAARRAY.append(['NADA_BODY1','MAK_NM','ENG_FUEL_CD','PLNT_CD'])
# # 24
# CATFAARRAY.append(['NADA_BODY1','MAK_NM','ENG_FUEL_CD','PLNT_CD'])
# CATFAARRAY.append(['NADA_BODY1','MAK_NM','ENG_FUEL_CD','PLNT_CD'])
# CATFAARRAY.append(['NADA_BODY1','MAK_NM','ENG_FUEL_CD','PLNT_CD'])
# CATFAARRAY.append(['NADA_BODY1','MAK_NM','ENG_FUEL_CD','PLNT_CD'])
# CATFAARRAY.append(['NADA_BODY1','MAK_NM','ENG_FUEL_CD','PLNT_CD'])
# CATFAARRAY.append(['NADA_BODY1','MAK_NM','ENG_FUEL_CD','PLNT_CD'])
# CATFAARRAY.append(['NADA_BODY1','MAK_NM','ENG_FUEL_CD','PLNT_CD'])
# CATFAARRAY.append(['NADA_BODY1','MAK_NM','ENG_FUEL_CD','PLNT_CD'])
# CATFAARRAY.append(['NADA_BODY1','MAK_NM','ENG_FUEL_CD','PLNT_CD'])
# CATFAARRAY.append(['NADA_BODY1','MAK_NM','ENG_FUEL_CD','PLNT_CD'])
# CATFAARRAY.append(['NADA_BODY1','MAK_NM','ENG_FUEL_CD','PLNT_CD'])
# CATFAARRAY.append(['NADA_BODY1','MAK_NM','ENG_FUEL_CD','PLNT_CD'])
# CATFAARRAY.append(['NADA_BODY1','MAK_NM','ENG_FUEL_CD','PLNT_CD'])
# CATFAARRAY.append(['NADA_BODY1','MAK_NM','ENG_FUEL_CD','PLNT_CD'])

# DUMMYFEATURES=[]
# for feature in CATFEATURES:
# 	dummiesTrain=pd.get_dummies(data_train[feature],prefix=feature)
# 	dummiesTest=pd.get_dummies(data_test[feature],prefix=feature)
# 	temp=dummiesTrain.sum()	
# 	# If column appears in data_train but not data_test we need to add it to data_test as a column of zeroes
# 	for k in ml.returnNotMatches(dummiesTrain.columns.values,dummiesTest.columns.values)[0]:
# 		dummiesTest[k]=0	
# 	dummiesTrain=dummiesTrain[temp[temp<temp.max()].index.values]
# 	dummiesTest=dummiesTest[temp[temp<temp.max()].index.values]	
# 	if(len(dummiesTrain.columns)>0):
# 		dummiesTrain.columns = dummiesTrain.columns.str.replace('\s+', '_')
# 		dummiesTest.columns = dummiesTest.columns.str.replace('\s+', '_')
# 		data_train=pd.concat([data_train,dummiesTrain],axis=1)
# 		data_test=pd.concat([data_test,dummiesTest],axis=1)
# 		DUMMYFEATURES += list(dummiesTrain)

# # print(DUMMYFEATURES)		

# # Create custom variables for GLM
# data_train['TRK_GRSS_VEH_WGHT_RATG_CD_gt6']=np.where(data_train['TRK_GRSS_VEH_WGHT_RATG_CD']>6, 1, 0)
# data_test['TRK_GRSS_VEH_WGHT_RATG_CD_gt6'] =np.where( data_test['TRK_GRSS_VEH_WGHT_RATG_CD']>6, 1, 0)
# data_train['cko_wheelbase_mean_lt130']=np.where(data_train['cko_wheelbase_mean']<130,1,0)
# data_train['cko_max_gvwc_lt11k']=np.where(data_train['cko_max_gvwc']<11000,1,0)
# data_test['cko_wheelbase_mean_lt130']=np.where(data_test['cko_wheelbase_mean']<130,1,0)
# data_test['cko_max_gvwc_lt11k']=np.where(data_test['cko_max_gvwc']<11000,1,0)
# data_train['TRK_CAB_CNFG_CD_CONSUVSPE']=(data_train['TRK_CAB_CNFG_CD_CON']+data_train['TRK_CAB_CNFG_CD_SUV']+data_train['TRK_CAB_CNFG_CD_SPE'])
# data_test['TRK_CAB_CNFG_CD_CONSUVSPE']=(data_test['TRK_CAB_CNFG_CD_CON']+data_test['TRK_CAB_CNFG_CD_SUV']+data_test['TRK_CAB_CNFG_CD_SPE'])
# data_train['TRK_CAB_CNFG_CD_CRWTHICFWHCB']=(data_train['TRK_CAB_CNFG_CD_CRW']+data_train['TRK_CAB_CNFG_CD_THI']+data_train['TRK_CAB_CNFG_CD_CFW']+data_train['TRK_CAB_CNFG_CD_HCB'])
# data_test['TRK_CAB_CNFG_CD_CRWTHICFWHCB']=(data_test['TRK_CAB_CNFG_CD_CRW']+data_test['TRK_CAB_CNFG_CD_THI']+data_test['TRK_CAB_CNFG_CD_CFW']+data_test['TRK_CAB_CNFG_CD_HCB'])
# # data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_020105']+data_train['ENG_MDL_CD_165230']+data_train['ENG_MDL_CD_020120']+data_train['ENG_MDL_CD_165225']+data_train['ENG_MDL_CD_220019']+data_train['ENG_MDL_CD_140167']+data_train['ENG_MDL_CD_165185']+data_train['ENG_MDL_CD_195050']+data_train['ENG_MDL_CD_245010']+data_train['ENG_MDL_CD_140330']+data_train['ENG_MDL_CD_020026']+data_train['ENG_MDL_CD_140173']+data_train['ENG_MDL_CD_195001']+data_train['ENG_MDL_CD_010038']+data_train['ENG_MDL_CD_140027']+data_train['ENG_MDL_CD_020075']+data_train['ENG_MDL_CD_020072']+data_train['ENG_MDL_CD_080012']+data_train['ENG_MDL_CD_090011']+data_train['ENG_MDL_CD_180012']+data_train['ENG_MDL_CD_160030']+data_train['ENG_MDL_CD_140168']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_020085']+data_train['ENG_MDL_CD_120010']+data_train['ENG_MDL_CD_160020']+data_train['ENG_MDL_CD_110030']+data_train['ENG_MDL_CD_090020']+data_train['ENG_MDL_CD_090077']+data_train['ENG_MDL_CD_090051']+data_train['ENG_MDL_CD_180020']+data_train['ENG_MDL_CD_180010']+data_train['ENG_MDL_CD_020060']+data_train['ENG_MDL_CD_050020']+data_train['ENG_MDL_CD_180052']+data_train['ENG_MDL_CD_150020']+data_train['ENG_MDL_CD_090045']+data_train['ENG_MDL_CD_090043']+data_train['ENG_MDL_CD_140090']+data_train['ENG_MDL_CD_140220']+data_train['ENG_MDL_CD_140115']+data_train['ENG_MDL_CD_210080']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_140025']+data_train['ENG_MDL_CD_140200']+data_train['ENG_MDL_CD_140055']+data_train['ENG_MDL_CD_040010']+data_train['ENG_MDL_CD_020100']+data_train['ENG_MDL_CD_020090']+data_train['ENG_MDL_CD_165160']+data_train['ENG_MDL_CD_090040']+data_train['ENG_MDL_CD_210050']+data_train['ENG_MDL_CD_140195']+data_train['ENG_MDL_CD_200010']+data_train['ENG_MDL_CD_150010']+data_train['ENG_MDL_CD_210030']+data_train['ENG_MDL_CD_160060']+data_train['ENG_MDL_CD_150060']+data_train['ENG_MDL_CD_010050']+data_train['ENG_MDL_CD_210100']+data_train['ENG_MDL_CD_140215']+data_train['ENG_MDL_CD_140050']+data_train['ENG_MDL_CD_210070']+data_train['ENG_MDL_CD_040030']+data_train['ENG_MDL_CD_210040']+data_train['ENG_MDL_CD_140125']+data_train['ENG_MDL_CD_210090']+data_train['ENG_MDL_CD_210020']+data_train['ENG_MDL_CD_140020']+data_train['ENG_MDL_CD_140045']+data_train['ENG_MDL_CD_140040']+data_train['ENG_MDL_CD_140234']+data_train['ENG_MDL_CD_210060']+data_train['ENG_MDL_CD_165007']+data_train['ENG_MDL_CD_210165']+data_train['ENG_MDL_CD_165030']+data_train['ENG_MDL_CD_160012'])
# # data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_140160']+data_train['ENG_MDL_CD_140191']+data_train['ENG_MDL_CD_140210']+data_train['ENG_MDL_CD_140028']+data_train['ENG_MDL_CD_165110']+data_train['ENG_MDL_CD_140205']+data_train['ENG_MDL_CD_110050']+data_train['ENG_MDL_CD_090070']+data_train['ENG_MDL_CD_165090']+data_train['ENG_MDL_CD_090075']+data_train['ENG_MDL_CD_090030']+data_train['ENG_MDL_CD_210142']+data_train['ENG_MDL_CD_210143']+data_train['ENG_MDL_CD_180040']+data_train['ENG_MDL_CD_090010']+data_train['ENG_MDL_CD_220028']+data_train['ENG_MDL_CD_110028']+data_train['ENG_MDL_CD_080044']+data_train['ENG_MDL_CD_165140']+data_train['ENG_MDL_CD_150003']+data_train['ENG_MDL_CD_160022']+data_train['ENG_MDL_CD_210130']+data_train['ENG_MDL_CD_090085']+data_train['ENG_MDL_CD_140095']+data_train['ENG_MDL_CD_180030']+data_train['ENG_MDL_CD_080080']+data_train['ENG_MDL_CD_020010']+data_train['ENG_MDL_CD_080015']+data_train['ENG_MDL_CD_070025']+data_train['ENG_MDL_CD_165200']+data_train['ENG_MDL_CD_080100']+data_train['ENG_MDL_CD_160040']+data_train['ENG_MDL_CD_165040']+data_train['ENG_MDL_CD_220034']+data_train['ENG_MDL_CD_110040']+data_train['ENG_MDL_CD_090100']+data_train['ENG_MDL_CD_090071']+data_train['ENG_MDL_CD_140150']+data_train['ENG_MDL_CD_160025']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_165170']+data_train['ENG_MDL_CD_070018']+data_train['ENG_MDL_CD_160065']+data_train['ENG_MDL_CD_220029']+data_train['ENG_MDL_CD_180050']+data_train['ENG_MDL_CD_165070']+data_train['ENG_MDL_CD_200020']+data_train['ENG_MDL_CD_140182']+data_train['ENG_MDL_CD_140065']+data_train['ENG_MDL_CD_110070']+data_train['ENG_MDL_CD_220023']+data_train['ENG_MDL_CD_020091']+data_train['ENG_MDL_CD_180016']+data_train['ENG_MDL_CD_165130']+data_train['ENG_MDL_CD_165057']+data_train['ENG_MDL_CD_010033']+data_train['ENG_MDL_CD_220026']+data_train['ENG_MDL_CD_020027']+data_train['ENG_MDL_CD_165035']+data_train['ENG_MDL_CD_210122']+data_train['ENG_MDL_CD_010006']+data_train['ENG_MDL_CD_195022'])
# # data_train['ENG_MDL_CD_grp3']=(data_train['ENG_MDL_CD_010030']+data_train['ENG_MDL_CD_140190']+data_train['ENG_MDL_CD_140175']+data_train['ENG_MDL_CD_165010']+data_train['ENG_MDL_CD_165100']+data_train['ENG_MDL_CD_010039']+data_train['ENG_MDL_CD_030060']+data_train['ENG_MDL_CD_140181']+data_train['ENG_MDL_CD_030020']+data_train['ENG_MDL_CD_165055']+data_train['ENG_MDL_CD_140145']+data_train['ENG_MDL_CD_165020']+data_train['ENG_MDL_CD_080020']+data_train['ENG_MDL_CD_140165']+data_train['ENG_MDL_CD_070035']+data_train['ENG_MDL_CD_210120']+data_train['ENG_MDL_CD_140130']+data_train['ENG_MDL_CD_140080']+data_train['ENG_MDL_CD_140135']+data_train['ENG_MDL_CD_090047']+data_train['ENG_MDL_CD_140194']+data_train['ENG_MDL_CD_140185']+data_train['ENG_MDL_CD_010080']+data_train['ENG_MDL_CD_210110']+data_train['ENG_MDL_CD_030015']+data_train['ENG_MDL_CD_030080']+data_train['ENG_MDL_CD_030030']+data_train['ENG_MDL_CD_030051']+data_train['ENG_MDL_CD_030105']+data_train['ENG_MDL_CD_030110']+data_train['ENG_MDL_CD_030070']+data_train['ENG_MDL_CD_030047']+data_train['ENG_MDL_CD_030040']+data_train['ENG_MDL_CD_030115']+data_train['ENG_MDL_CD_030050']+data_train['ENG_MDL_CD_030045'])
# # data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_020105']+data_test['ENG_MDL_CD_165230']+data_test['ENG_MDL_CD_020120']+data_test['ENG_MDL_CD_165225']+data_test['ENG_MDL_CD_220019']+data_test['ENG_MDL_CD_140167']+data_test['ENG_MDL_CD_165185']+data_test['ENG_MDL_CD_195050']+data_test['ENG_MDL_CD_245010']+data_test['ENG_MDL_CD_140330']+data_test['ENG_MDL_CD_020026']+data_test['ENG_MDL_CD_140173']+data_test['ENG_MDL_CD_195001']+data_test['ENG_MDL_CD_010038']+data_test['ENG_MDL_CD_140027']+data_test['ENG_MDL_CD_020075']+data_test['ENG_MDL_CD_020072']+data_test['ENG_MDL_CD_080012']+data_test['ENG_MDL_CD_090011']+data_test['ENG_MDL_CD_180012']+data_test['ENG_MDL_CD_160030']+data_test['ENG_MDL_CD_140168']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_020085']+data_test['ENG_MDL_CD_120010']+data_test['ENG_MDL_CD_160020']+data_test['ENG_MDL_CD_110030']+data_test['ENG_MDL_CD_090020']+data_test['ENG_MDL_CD_090077']+data_test['ENG_MDL_CD_090051']+data_test['ENG_MDL_CD_180020']+data_test['ENG_MDL_CD_180010']+data_test['ENG_MDL_CD_020060']+data_test['ENG_MDL_CD_050020']+data_test['ENG_MDL_CD_180052']+data_test['ENG_MDL_CD_150020']+data_test['ENG_MDL_CD_090045']+data_test['ENG_MDL_CD_090043']+data_test['ENG_MDL_CD_140090']+data_test['ENG_MDL_CD_140220']+data_test['ENG_MDL_CD_140115']+data_test['ENG_MDL_CD_210080']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_140025']+data_test['ENG_MDL_CD_140200']+data_test['ENG_MDL_CD_140055']+data_test['ENG_MDL_CD_040010']+data_test['ENG_MDL_CD_020100']+data_test['ENG_MDL_CD_020090']+data_test['ENG_MDL_CD_165160']+data_test['ENG_MDL_CD_090040']+data_test['ENG_MDL_CD_210050']+data_test['ENG_MDL_CD_140195']+data_test['ENG_MDL_CD_200010']+data_test['ENG_MDL_CD_150010']+data_test['ENG_MDL_CD_210030']+data_test['ENG_MDL_CD_160060']+data_test['ENG_MDL_CD_150060']+data_test['ENG_MDL_CD_010050']+data_test['ENG_MDL_CD_210100']+data_test['ENG_MDL_CD_140215']+data_test['ENG_MDL_CD_140050']+data_test['ENG_MDL_CD_210070']+data_test['ENG_MDL_CD_040030']+data_test['ENG_MDL_CD_210040']+data_test['ENG_MDL_CD_140125']+data_test['ENG_MDL_CD_210090']+data_test['ENG_MDL_CD_210020']+data_test['ENG_MDL_CD_140020']+data_test['ENG_MDL_CD_140045']+data_test['ENG_MDL_CD_140040']+data_test['ENG_MDL_CD_140234']+data_test['ENG_MDL_CD_210060']+data_test['ENG_MDL_CD_165007']+data_test['ENG_MDL_CD_210165']+data_test['ENG_MDL_CD_165030']+data_test['ENG_MDL_CD_160012'])
# # data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_140160']+data_test['ENG_MDL_CD_140191']+data_test['ENG_MDL_CD_140210']+data_test['ENG_MDL_CD_140028']+data_test['ENG_MDL_CD_165110']+data_test['ENG_MDL_CD_140205']+data_test['ENG_MDL_CD_110050']+data_test['ENG_MDL_CD_090070']+data_test['ENG_MDL_CD_165090']+data_test['ENG_MDL_CD_090075']+data_test['ENG_MDL_CD_090030']+data_test['ENG_MDL_CD_210142']+data_test['ENG_MDL_CD_210143']+data_test['ENG_MDL_CD_180040']+data_test['ENG_MDL_CD_090010']+data_test['ENG_MDL_CD_220028']+data_test['ENG_MDL_CD_110028']+data_test['ENG_MDL_CD_080044']+data_test['ENG_MDL_CD_165140']+data_test['ENG_MDL_CD_150003']+data_test['ENG_MDL_CD_160022']+data_test['ENG_MDL_CD_210130']+data_test['ENG_MDL_CD_090085']+data_test['ENG_MDL_CD_140095']+data_test['ENG_MDL_CD_180030']+data_test['ENG_MDL_CD_080080']+data_test['ENG_MDL_CD_020010']+data_test['ENG_MDL_CD_080015']+data_test['ENG_MDL_CD_070025']+data_test['ENG_MDL_CD_165200']+data_test['ENG_MDL_CD_080100']+data_test['ENG_MDL_CD_160040']+data_test['ENG_MDL_CD_165040']+data_test['ENG_MDL_CD_220034']+data_test['ENG_MDL_CD_110040']+data_test['ENG_MDL_CD_090100']+data_test['ENG_MDL_CD_090071']+data_test['ENG_MDL_CD_140150']+data_test['ENG_MDL_CD_160025']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_165170']+data_test['ENG_MDL_CD_070018']+data_test['ENG_MDL_CD_160065']+data_test['ENG_MDL_CD_220029']+data_test['ENG_MDL_CD_180050']+data_test['ENG_MDL_CD_165070']+data_test['ENG_MDL_CD_200020']+data_test['ENG_MDL_CD_140182']+data_test['ENG_MDL_CD_140065']+data_test['ENG_MDL_CD_110070']+data_test['ENG_MDL_CD_220023']+data_test['ENG_MDL_CD_020091']+data_test['ENG_MDL_CD_180016']+data_test['ENG_MDL_CD_165130']+data_test['ENG_MDL_CD_165057']+data_test['ENG_MDL_CD_010033']+data_test['ENG_MDL_CD_220026']+data_test['ENG_MDL_CD_020027']+data_test['ENG_MDL_CD_165035']+data_test['ENG_MDL_CD_210122']+data_test['ENG_MDL_CD_010006']+data_test['ENG_MDL_CD_195022'])
# # data_test['ENG_MDL_CD_grp3']=(data_test['ENG_MDL_CD_010030']+data_test['ENG_MDL_CD_140190']+data_test['ENG_MDL_CD_140175']+data_test['ENG_MDL_CD_165010']+data_test['ENG_MDL_CD_165100']+data_test['ENG_MDL_CD_010039']+data_test['ENG_MDL_CD_030060']+data_test['ENG_MDL_CD_140181']+data_test['ENG_MDL_CD_030020']+data_test['ENG_MDL_CD_165055']+data_test['ENG_MDL_CD_140145']+data_test['ENG_MDL_CD_165020']+data_test['ENG_MDL_CD_080020']+data_test['ENG_MDL_CD_140165']+data_test['ENG_MDL_CD_070035']+data_test['ENG_MDL_CD_210120']+data_test['ENG_MDL_CD_140130']+data_test['ENG_MDL_CD_140080']+data_test['ENG_MDL_CD_140135']+data_test['ENG_MDL_CD_090047']+data_test['ENG_MDL_CD_140194']+data_test['ENG_MDL_CD_140185']+data_test['ENG_MDL_CD_010080']+data_test['ENG_MDL_CD_210110']+data_test['ENG_MDL_CD_030015']+data_test['ENG_MDL_CD_030080']+data_test['ENG_MDL_CD_030030']+data_test['ENG_MDL_CD_030051']+data_test['ENG_MDL_CD_030105']+data_test['ENG_MDL_CD_030110']+data_test['ENG_MDL_CD_030070']+data_test['ENG_MDL_CD_030047']+data_test['ENG_MDL_CD_030040']+data_test['ENG_MDL_CD_030115']+data_test['ENG_MDL_CD_030050']+data_test['ENG_MDL_CD_030045'])

# # Engine model groups from iteration 12 onwards
# # data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_020105']+data_train['ENG_MDL_CD_165230']+data_train['ENG_MDL_CD_020120']+data_train['ENG_MDL_CD_165225']+data_train['ENG_MDL_CD_220019']+data_train['ENG_MDL_CD_010070']+data_train['ENG_MDL_CD_165185']+data_train['ENG_MDL_CD_245010']+data_train['ENG_MDL_CD_020026']+data_train['ENG_MDL_CD_140167']+data_train['ENG_MDL_CD_195001']+data_train['ENG_MDL_CD_030090']+data_train['ENG_MDL_CD_010038']+data_train['ENG_MDL_CD_140173']+data_train['ENG_MDL_CD_140027']+data_train['ENG_MDL_CD_080012']+data_train['ENG_MDL_CD_090077']+data_train['ENG_MDL_CD_140168']+data_train['ENG_MDL_CD_020085']+data_train['ENG_MDL_CD_090051']+data_train['ENG_MDL_CD_020060']+data_train['ENG_MDL_CD_160020']+data_train['ENG_MDL_CD_120010']+data_train['ENG_MDL_CD_110030']+data_train['ENG_MDL_CD_160030']+data_train['ENG_MDL_CD_180020']+data_train['ENG_MDL_CD_210110']+data_train['ENG_MDL_CD_090045']+data_train['ENG_MDL_CD_180010']+data_train['ENG_MDL_CD_180052']+data_train['ENG_MDL_CD_090020']+data_train['ENG_MDL_CD_090043']+data_train['ENG_MDL_CD_210080']+data_train['ENG_MDL_CD_140090']+data_train['ENG_MDL_CD_160060']+data_train['ENG_MDL_CD_140025']+data_train['ENG_MDL_CD_140115']+data_train['ENG_MDL_CD_090040']+data_train['ENG_MDL_CD_140055']+data_train['ENG_MDL_CD_020090']+data_train['ENG_MDL_CD_140220']+data_train['ENG_MDL_CD_020100']+data_train['ENG_MDL_CD_165160']+data_train['ENG_MDL_CD_210050']+data_train['ENG_MDL_CD_110070']+data_train['ENG_MDL_CD_050020']+data_train['ENG_MDL_CD_210030']+data_train['ENG_MDL_CD_040030']+data_train['ENG_MDL_CD_180040']+data_train['ENG_MDL_CD_140215']+data_train['ENG_MDL_CD_210100']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_210070']+data_train['ENG_MDL_CD_010050']+data_train['ENG_MDL_CD_140125']+data_train['ENG_MDL_CD_040010']+data_train['ENG_MDL_CD_140050']+data_train['ENG_MDL_CD_210040']+data_train['ENG_MDL_CD_210090']+data_train['ENG_MDL_CD_210020']+data_train['ENG_MDL_CD_140020']+data_train['ENG_MDL_CD_140040']+data_train['ENG_MDL_CD_140234']+data_train['ENG_MDL_CD_030060']+data_train['ENG_MDL_CD_140045']+data_train['ENG_MDL_CD_210060']+data_train['ENG_MDL_CD_030020']+data_train['ENG_MDL_CD_040020']+data_train['ENG_MDL_CD_165007'])
# # data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_150020']+data_train['ENG_MDL_CD_150060']+data_train['ENG_MDL_CD_150010'])
# # data_train['ENG_MDL_CD_grp3']=(data_train['ENG_MDL_CD_210165']+data_train['ENG_MDL_CD_090030']+data_train['ENG_MDL_CD_165030']+data_train['ENG_MDL_CD_180030']+data_train['ENG_MDL_CD_200010']+data_train['ENG_MDL_CD_140195']+data_train['ENG_MDL_CD_140210']+data_train['ENG_MDL_CD_160012'])
# # data_train['ENG_MDL_CD_grp4']=(data_train['ENG_MDL_CD_140105']+data_train['ENG_MDL_CD_165140']+data_train['ENG_MDL_CD_080080']+data_train['ENG_MDL_CD_140065']+data_train['ENG_MDL_CD_090011']+data_train['ENG_MDL_CD_140205']+data_train['ENG_MDL_CD_030047']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_195050']+data_train['ENG_MDL_CD_020072']+data_train['ENG_MDL_CD_220028']+data_train['ENG_MDL_CD_030050']+data_train['ENG_MDL_CD_090070']+data_train['ENG_MDL_CD_140095']+data_train['ENG_MDL_CD_165090']+data_train['ENG_MDL_CD_220034']+data_train['ENG_MDL_CD_210130']+data_train['ENG_MDL_CD_140185']+data_train['ENG_MDL_CD_165010']+data_train['ENG_MDL_CD_070020'])
# # data_train['ENG_MDL_CD_grp5']=(data_train['ENG_MDL_CD_210122']+data_train['ENG_MDL_CD_140150']+data_train['ENG_MDL_CD_110050']+data_train['ENG_MDL_CD_180050']+data_train['ENG_MDL_CD_080015']+data_train['ENG_MDL_CD_070025']+data_train['ENG_MDL_CD_200020']+data_train['ENG_MDL_CD_140035']+data_train['ENG_MDL_CD_090100']+data_train['ENG_MDL_CD_030080']+data_train['ENG_MDL_CD_160022']+data_train['ENG_MDL_CD_160040']+data_train['ENG_MDL_CD_090085']+data_train['ENG_MDL_CD_165170']+data_train['ENG_MDL_CD_180014']+data_train['ENG_MDL_CD_160025'])
# # data_train['ENG_MDL_CD_grp6']=(data_train['ENG_MDL_CD_165070']+data_train['ENG_MDL_CD_010006']+data_train['ENG_MDL_CD_165040']+data_train['ENG_MDL_CD_080100']+data_train['ENG_MDL_CD_010075']+data_train['ENG_MDL_CD_140182']+data_train['ENG_MDL_CD_140191']+data_train['ENG_MDL_CD_020091']+data_train['ENG_MDL_CD_110040']+data_train['ENG_MDL_CD_090075']+data_train['ENG_MDL_CD_140193']+data_train['ENG_MDL_CD_195070']+data_train['ENG_MDL_CD_220029'])
# # data_train['ENG_MDL_CD_grp7']=(data_train['ENG_MDL_CD_090071']+data_train['ENG_MDL_CD_030015']+data_train['ENG_MDL_CD_140080']+data_train['ENG_MDL_CD_180012']+data_train['ENG_MDL_CD_180016']+data_train['ENG_MDL_CD_020010']+data_train['ENG_MDL_CD_140186']+data_train['ENG_MDL_CD_165130']+data_train['ENG_MDL_CD_110010']+data_train['ENG_MDL_CD_160065']+data_train['ENG_MDL_CD_195060']+data_train['ENG_MDL_CD_110028']+data_train['ENG_MDL_CD_140028']+data_train['ENG_MDL_CD_020075']+data_train['ENG_MDL_CD_165057']+data_train['ENG_MDL_CD_140199']+data_train['ENG_MDL_CD_165200']+data_train['ENG_MDL_CD_165031']+data_train['ENG_MDL_CD_165035'])
# # data_train['ENG_MDL_CD_grp8']=(data_train['ENG_MDL_CD_070018']+data_train['ENG_MDL_CD_160013']+data_train['ENG_MDL_CD_140085']+data_train['ENG_MDL_CD_165120']+data_train['ENG_MDL_CD_110065']+data_train['ENG_MDL_CD_165215']+data_train['ENG_MDL_CD_220023']+data_train['ENG_MDL_CD_090053']+data_train['ENG_MDL_CD_165150']+data_train['ENG_MDL_CD_010033']+data_train['ENG_MDL_CD_070022']+data_train['ENG_MDL_CD_165009']+data_train['ENG_MDL_CD_140250']+data_train['ENG_MDL_CD_010012']+data_train['ENG_MDL_CD_010015']+data_train['ENG_MDL_CD_010020']+data_train['ENG_MDL_CD_165180']+data_train['ENG_MDL_CD_165051'])
# # data_train['ENG_MDL_CD_grp9']=(data_train['ENG_MDL_CD_140181']+data_train['ENG_MDL_CD_165055']+data_train['ENG_MDL_CD_210143']+data_train['ENG_MDL_CD_080020']+data_train['ENG_MDL_CD_140165']+data_train['ENG_MDL_CD_030035']+data_train['ENG_MDL_CD_140160']+data_train['ENG_MDL_CD_070035']+data_train['ENG_MDL_CD_090050']+data_train['ENG_MDL_CD_140140']+data_train['ENG_MDL_CD_020027'])
# # data_train['ENG_MDL_CD_grp10']=(data_train['ENG_MDL_CD_140145']+data_train['ENG_MDL_CD_030045']+data_train['ENG_MDL_CD_140130']+data_train['ENG_MDL_CD_090047']+data_train['ENG_MDL_CD_140135']+data_train['ENG_MDL_CD_140194']+data_train['ENG_MDL_CD_010080'])
# # data_train['ENG_MDL_CD_grp11']=(data_train['ENG_MDL_CD_150003']+data_train['ENG_MDL_CD_150150']+data_train['ENG_MDL_CD_150005']+data_train['ENG_MDL_CD_150007'])

# # data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_020105']+data_test['ENG_MDL_CD_165230']+data_test['ENG_MDL_CD_020120']+data_test['ENG_MDL_CD_165225']+data_test['ENG_MDL_CD_220019']+data_test['ENG_MDL_CD_010070']+data_test['ENG_MDL_CD_165185']+data_test['ENG_MDL_CD_245010']+data_test['ENG_MDL_CD_020026']+data_test['ENG_MDL_CD_140167']+data_test['ENG_MDL_CD_195001']+data_test['ENG_MDL_CD_030090']+data_test['ENG_MDL_CD_010038']+data_test['ENG_MDL_CD_140173']+data_test['ENG_MDL_CD_140027']+data_test['ENG_MDL_CD_080012']+data_test['ENG_MDL_CD_090077']+data_test['ENG_MDL_CD_140168']+data_test['ENG_MDL_CD_020085']+data_test['ENG_MDL_CD_090051']+data_test['ENG_MDL_CD_020060']+data_test['ENG_MDL_CD_160020']+data_test['ENG_MDL_CD_120010']+data_test['ENG_MDL_CD_110030']+data_test['ENG_MDL_CD_160030']+data_test['ENG_MDL_CD_180020']+data_test['ENG_MDL_CD_210110']+data_test['ENG_MDL_CD_090045']+data_test['ENG_MDL_CD_180010']+data_test['ENG_MDL_CD_180052']+data_test['ENG_MDL_CD_090020']+data_test['ENG_MDL_CD_090043']+data_test['ENG_MDL_CD_210080']+data_test['ENG_MDL_CD_140090']+data_test['ENG_MDL_CD_160060']+data_test['ENG_MDL_CD_140025']+data_test['ENG_MDL_CD_140115']+data_test['ENG_MDL_CD_090040']+data_test['ENG_MDL_CD_140055']+data_test['ENG_MDL_CD_020090']+data_test['ENG_MDL_CD_140220']+data_test['ENG_MDL_CD_020100']+data_test['ENG_MDL_CD_165160']+data_test['ENG_MDL_CD_210050']+data_test['ENG_MDL_CD_110070']+data_test['ENG_MDL_CD_050020']+data_test['ENG_MDL_CD_210030']+data_test['ENG_MDL_CD_040030']+data_test['ENG_MDL_CD_180040']+data_test['ENG_MDL_CD_140215']+data_test['ENG_MDL_CD_210100']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_210070']+data_test['ENG_MDL_CD_010050']+data_test['ENG_MDL_CD_140125']+data_test['ENG_MDL_CD_040010']+data_test['ENG_MDL_CD_140050']+data_test['ENG_MDL_CD_210040']+data_test['ENG_MDL_CD_210090']+data_test['ENG_MDL_CD_210020']+data_test['ENG_MDL_CD_140020']+data_test['ENG_MDL_CD_140040']+data_test['ENG_MDL_CD_140234']+data_test['ENG_MDL_CD_030060']+data_test['ENG_MDL_CD_140045']+data_test['ENG_MDL_CD_210060']+data_test['ENG_MDL_CD_030020']+data_test['ENG_MDL_CD_040020']+data_test['ENG_MDL_CD_165007'])
# # data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_150020']+data_test['ENG_MDL_CD_150060']+data_test['ENG_MDL_CD_150010'])
# # data_test['ENG_MDL_CD_grp3']=(data_test['ENG_MDL_CD_210165']+data_test['ENG_MDL_CD_090030']+data_test['ENG_MDL_CD_165030']+data_test['ENG_MDL_CD_180030']+data_test['ENG_MDL_CD_200010']+data_test['ENG_MDL_CD_140195']+data_test['ENG_MDL_CD_140210']+data_test['ENG_MDL_CD_160012'])
# # data_test['ENG_MDL_CD_grp4']=(data_test['ENG_MDL_CD_140105']+data_test['ENG_MDL_CD_165140']+data_test['ENG_MDL_CD_080080']+data_test['ENG_MDL_CD_140065']+data_test['ENG_MDL_CD_090011']+data_test['ENG_MDL_CD_140205']+data_test['ENG_MDL_CD_030047']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_195050']+data_test['ENG_MDL_CD_020072']+data_test['ENG_MDL_CD_220028']+data_test['ENG_MDL_CD_030050']+data_test['ENG_MDL_CD_090070']+data_test['ENG_MDL_CD_140095']+data_test['ENG_MDL_CD_165090']+data_test['ENG_MDL_CD_220034']+data_test['ENG_MDL_CD_210130']+data_test['ENG_MDL_CD_140185']+data_test['ENG_MDL_CD_165010']+data_test['ENG_MDL_CD_070020'])
# # data_test['ENG_MDL_CD_grp5']=(data_test['ENG_MDL_CD_210122']+data_test['ENG_MDL_CD_140150']+data_test['ENG_MDL_CD_110050']+data_test['ENG_MDL_CD_180050']+data_test['ENG_MDL_CD_080015']+data_test['ENG_MDL_CD_070025']+data_test['ENG_MDL_CD_200020']+data_test['ENG_MDL_CD_140035']+data_test['ENG_MDL_CD_090100']+data_test['ENG_MDL_CD_030080']+data_test['ENG_MDL_CD_160022']+data_test['ENG_MDL_CD_160040']+data_test['ENG_MDL_CD_090085']+data_test['ENG_MDL_CD_165170']+data_test['ENG_MDL_CD_180014']+data_test['ENG_MDL_CD_160025'])
# # data_test['ENG_MDL_CD_grp6']=(data_test['ENG_MDL_CD_165070']+data_test['ENG_MDL_CD_010006']+data_test['ENG_MDL_CD_165040']+data_test['ENG_MDL_CD_080100']+data_test['ENG_MDL_CD_010075']+data_test['ENG_MDL_CD_140182']+data_test['ENG_MDL_CD_140191']+data_test['ENG_MDL_CD_020091']+data_test['ENG_MDL_CD_110040']+data_test['ENG_MDL_CD_090075']+data_test['ENG_MDL_CD_140193']+data_test['ENG_MDL_CD_195070']+data_test['ENG_MDL_CD_220029'])
# # data_test['ENG_MDL_CD_grp7']=(data_test['ENG_MDL_CD_090071']+data_test['ENG_MDL_CD_030015']+data_test['ENG_MDL_CD_140080']+data_test['ENG_MDL_CD_180012']+data_test['ENG_MDL_CD_180016']+data_test['ENG_MDL_CD_020010']+data_test['ENG_MDL_CD_140186']+data_test['ENG_MDL_CD_165130']+data_test['ENG_MDL_CD_110010']+data_test['ENG_MDL_CD_160065']+data_test['ENG_MDL_CD_195060']+data_test['ENG_MDL_CD_110028']+data_test['ENG_MDL_CD_140028']+data_test['ENG_MDL_CD_020075']+data_test['ENG_MDL_CD_165057']+data_test['ENG_MDL_CD_140199']+data_test['ENG_MDL_CD_165200']+data_test['ENG_MDL_CD_165031']+data_test['ENG_MDL_CD_165035'])
# # data_test['ENG_MDL_CD_grp8']=(data_test['ENG_MDL_CD_070018']+data_test['ENG_MDL_CD_160013']+data_test['ENG_MDL_CD_140085']+data_test['ENG_MDL_CD_165120']+data_test['ENG_MDL_CD_110065']+data_test['ENG_MDL_CD_165215']+data_test['ENG_MDL_CD_220023']+data_test['ENG_MDL_CD_090053']+data_test['ENG_MDL_CD_165150']+data_test['ENG_MDL_CD_010033']+data_test['ENG_MDL_CD_070022']+data_test['ENG_MDL_CD_165009']+data_test['ENG_MDL_CD_140250']+data_test['ENG_MDL_CD_010012']+data_test['ENG_MDL_CD_010015']+data_test['ENG_MDL_CD_010020']+data_test['ENG_MDL_CD_165180']+data_test['ENG_MDL_CD_165051'])
# # data_test['ENG_MDL_CD_grp9']=(data_test['ENG_MDL_CD_140181']+data_test['ENG_MDL_CD_165055']+data_test['ENG_MDL_CD_210143']+data_test['ENG_MDL_CD_080020']+data_test['ENG_MDL_CD_140165']+data_test['ENG_MDL_CD_030035']+data_test['ENG_MDL_CD_140160']+data_test['ENG_MDL_CD_070035']+data_test['ENG_MDL_CD_090050']+data_test['ENG_MDL_CD_140140']+data_test['ENG_MDL_CD_020027'])
# # data_test['ENG_MDL_CD_grp10']=(data_test['ENG_MDL_CD_140145']+data_test['ENG_MDL_CD_030045']+data_test['ENG_MDL_CD_140130']+data_test['ENG_MDL_CD_090047']+data_test['ENG_MDL_CD_140135']+data_test['ENG_MDL_CD_140194']+data_test['ENG_MDL_CD_010080'])
# # data_test['ENG_MDL_CD_grp11']=(data_test['ENG_MDL_CD_150003']+data_test['ENG_MDL_CD_150150']+data_test['ENG_MDL_CD_150005']+data_test['ENG_MDL_CD_150007'])



# formulabase = 'lossratio ~  TRK_GRSS_VEH_WGHT_RATG_CD_gt6 + cko_fuel_d2  +  TRK_CAB_CNFG_CD_CONSUVSPE + TRK_CAB_CNFG_CD_CRWTHICFWHCB  + cko_wheelbase_mean_lt130 + cko_eng_cylinders_d2 + ENG_MFG_CD_d7 + ENG_TRK_DUTY_TYP_CD_d1 + fuel_gas + SHIP_WGHT_LBSMS + cko_max_gvwc_lt11k    '
# # cko_mdl_yr + Liab_Ex_Veh_Age + cko_max_msrpMS
# forms=[]
# forms.append(' + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + ABS_BRK_DESC_REAR_WHEEL_ST + BODY_STYLE_CD_CG + BODY_STYLE_CD_CH + BODY_STYLE_CD_FT + BODY_STYLE_CD_GG + BODY_STYLE_CD_IC + BODY_STYLE_CD_MH + BODY_STYLE_CD_PK + BODY_STYLE_CD_TR + BODY_STYLE_CD_YY  ')
# forms.append(' + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + BODY_STYLE_DESC_CUTAWAY + BODY_STYLE_DESC_GARBAGE_REFUSE + BODY_STYLE_DESC_TRACTOR_TRUCK + BODY_STYLE_DESC_VAN_CARGO + cko_4wd_4WD + cko_antitheft_ALARM + cko_antitheft_PSV_DIS + cko_dtrl_N + cko_dtrl_S  ')
# forms.append(' + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + cko_eng_cylinders_4 + cko_eng_cylinders_8 + cko_fuel_GAS + cko_fuel_NGAS + cko_fuel_U + cko_max_gvwc_Tr_02_le20k + cko_overdrive_Y + cko_semi_flag_Y + cko_turbo_super_Y + DR_LGHT_OPT_CD_N + DR_LGHT_OPT_CD_S + DRV_TYP_CD_RWD  ')
# forms.append(' + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + ENG_ASP_SUP_CHGR_CD_N + ENG_ASP_TRBL_CHGR_CD_Y + ENG_ASP_VVTL_CD_N + ENG_ASP_VVTL_CD_Y + ENG_BLCK_TYP_CD_I + ENG_BLCK_TYP_CD_V + ENG_CBRT_TYP_CD_C + ENG_CBRT_TYP_CD_F + ENG_CLNDR_RTR_CNT__ + ENG_CLNDR_RTR_CNT_4 + ENG_CLNDR_RTR_CNT_8 + ENG_FUEL_CD_G + ENG_FUEL_INJ_TYP_CD_R + ENG_HEAD_CNFG_CD_OHV + ENG_HEAD_CNFG_CD_SOHC  ')
# forms.append(' + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + ENG_MDL_CD_010007 + ENG_MDL_CD_010035 + ENG_MDL_CD_010037 + ENG_MDL_CD_010045 + ENG_MDL_CD_020040 + ENG_MDL_CD_020070 + ENG_MDL_CD_020078 + ENG_MDL_CD_020080 + ENG_MDL_CD_020085 + ENG_MDL_CD_020090 + ENG_MDL_CD_020093 + ENG_MDL_CD_020094 + ENG_MDL_CD_020100 + ENG_MDL_CD_020105 + ENG_MDL_CD_030035 + ENG_MDL_CD_030040 + ENG_MDL_CD_030045 + ENG_MDL_CD_030047 + ENG_MDL_CD_030050 + ENG_MDL_CD_030051 + ENG_MDL_CD_030060 + ENG_MDL_CD_030070 + ENG_MDL_CD_030080 + ENG_MDL_CD_030090 + ENG_MDL_CD_030105 + ENG_MDL_CD_030110 + ENG_MDL_CD_030115 + ENG_MDL_CD_040010 + ENG_MDL_CD_040020 + ENG_MDL_CD_040030 + ENG_MDL_CD_070015 + ENG_MDL_CD_090047 + ENG_MDL_CD_165009 + ENG_MDL_CD_165195 + ENG_MDL_CD_195022 + ENG_MDL_CD_220023 + ENG_MFG_CD_010 + ENG_MFG_CD_030  ')
# forms.append(' + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + ENG_TRK_DUTY_TYP_CD_HV + ENG_TRK_DUTY_TYP_CD_U + ENTERTAIN_CD_A + ENTERTAIN_GP_1_2_9 + ENTERTAIN_GP_A + MAK_NM_AMERICAN_LA_FRANCE + MAK_NM_AUTOCAR + MAK_NM_AUTOCAR_LLC + MAK_NM_BERING + MAK_NM_BLUE_BIRD + MAK_NM_CAPACITY_OF_TEXAS + MAK_NM_CATERPILLAR + MAK_NM_CHEVROLET + MAK_NM_DODGE + MAK_NM_FREIGHTLINER + MAK_NM_HENDRICKSON + MAK_NM_IC_CORPORATION + MAK_NM_INDIANA_PHOENIX + MAK_NM_INTERNATIONAL + MAK_NM_ISUZU + MAK_NM_IVECO + MAK_NM_JOHN_DEERE + MAK_NM_KALMAR + MAK_NM_KENWORTH + MAK_NM_LAFORZA + MAK_NM_LES_AUTOBUS_MCI + MAK_NM_LODAL + MAK_NM_MACK + MAK_NM_MARMON_HERRINGTON + MAK_NM_MAXIM + MAK_NM_MERCEDES_BENZ + MAK_NM_MITSUBISHI_FUSO_TRUCK_OF_AMERICA_INC + MAK_NM_MOTOR_COACH_INDUSTRIES + MAK_NM_NEW_FLYER + MAK_NM_NISSAN_DIESEL + MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_ + MAK_NM_OTTAWA + MAK_NM_PETERBILT + MAK_NM_RAM + MAK_NM_WESTERN_STAR_AUTO_CAR  ')
# forms.append(' + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + MFG_DESC_CHRYSLER_GROUP_LLC + MFG_DESC_DAIMLER + MFG_DESC_GENERAL_MOTORS + MFG_DESC_NAVISTAR_INTERNATIONAL + NADA_BODY1_1 + NADA_BODY1_2 + NADA_BODY1_3 + NADA_BODY1_4 + NADA_BODY1_5 + NADA_BODY1_6 + NADA_BODY1_7 + NADA_BODY1_8 + NADA_BODY1_9 + NADA_BODY1_A + NADA_BODY1_C + NADA_BODY1_G + NADA_BODY1_L + NADA_BODY1_M + NADA_BODY1_R + NADA_BODY1_T + NADA_BODY1_U + NADA_BODY1_V  ')
# forms.append(' + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + OPT1_ENTERTAIN_CD_7 + PLNT_CD_A + PLNT_CD_D + PLNT_CD_G + PLNT_CD_J + PLNT_CD_L + PLNT_CD_P + PLNT_CD_S + PLNT_CD_U + PLNT_CD_V + PLNT_CD_W + PLNT_CD_Y + PLNT_CNTRY_NM_CANADA + PLNT_CNTRY_NM_FRANCE + PLNT_CNTRY_NM_GERMANY + PLNT_CNTRY_NM_ITALY + PLNT_CNTRY_NM_JAPAN + PLNT_CNTRY_NM_MEXICO + PLNT_CNTRY_NM_U + PLNT_CNTRY_NM_UNITED_KINGDO + PWR_BRK_OPT_CD_S + REAR_TIRE_SIZE_DESC_16R245  ')
# forms.append(' + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + RSTRNT_TYP_CD_7 + RSTRNT_TYP_CD_B + RSTRNT_TYP_CD_M + RSTRNT_TYP_CD_Z + TLT_STRNG_WHL_OPT_CD_S + TRANS_CD_A + TRANS_OVERDRV_IND_Y + TRK_BRK_TYP_CD_AOH + TRK_CAB_CNFG_CD_TLO + TRK_FRNT_AXL_CD_SF + TRK_FRNT_AXL_CD_U + VINA_BODY_TYPE_CD_CH + VINA_BODY_TYPE_CD_TB + VINA_BODY_TYPE_CD_TR + VINA_BODY_TYPE_CD_U + VINA_BODY_TYPE_CD_YY ')

# forms.append(' + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + cko_4wd_4WD + cko_fuel_GAS + ENG_FUEL_CD_G + ENG_MDL_CD_020085 + ENG_MDL_CD_020090 + ENG_MDL_CD_020100 + ENG_MDL_CD_020105 + ENG_MDL_CD_030060 + ENG_MDL_CD_040030 + ENG_MDL_CD_195022 + MAK_NM_CAPACITY_OF_TEXAS + MAK_NM_JOHN_DEERE + MAK_NM_LES_AUTOBUS_MCI + MAK_NM_MERCEDES_BENZ + MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_ + MAK_NM_OTTAWA + NADA_BODY1_4 + NADA_BODY1_C + PLNT_CD_U + PWR_BRK_OPT_CD_S ')
# forms.append(' + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + cko_fuel_GAS + ENG_FUEL_CD_G + ENG_MDL_CD_020085 + ENG_MDL_CD_020090 + ENG_MDL_CD_020100 + ENG_MDL_CD_020105 + ENG_MDL_CD_030060 + ENG_MDL_CD_040030 + ENG_MDL_CD_195022 + MAK_NM_JOHN_DEERE + MAK_NM_LES_AUTOBUS_MCI + MAK_NM_MERCEDES_BENZ + MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_ + MAK_NM_OTTAWA + NADA_BODY1_4 + PLNT_CD_U ')
# # 11
# forms.append(' + ENG_MDL_CD_220026 + ENG_MDL_CD_030070 + ENG_MDL_CD_U + ENG_MDL_CD_160066 + ENG_MDL_CD_180010 + ENG_MDL_CD_140050 + ENG_MDL_CD_140199 + ENG_MDL_CD_195080 + ENG_MDL_CD_140182 + ENG_MDL_CD_010011 + ENG_MDL_CD_180012 + ENG_MDL_CD_165160 + ENG_MDL_CD_080044 + ENG_MDL_CD_165035 + ENG_MDL_CD_120030 + ENG_MDL_CD_140190 + ENG_MDL_CD_150060 + ENG_MDL_CD_180030 + ENG_MDL_CD_210122 + ENG_MDL_CD_140165 + ENG_MDL_CD_200020 + ENG_MDL_CD_140200 + ENG_MDL_CD_165220 + ENG_MDL_CD_090043 + ENG_MDL_CD_030110 + ENG_MDL_CD_110040 + ENG_MDL_CD_165020 + ENG_MDL_CD_180016 + ENG_MDL_CD_090011 + ENG_MDL_CD_195001 + ENG_MDL_CD_090051 + ENG_MDL_CD_165008 + ENG_MDL_CD_140310 + ENG_MDL_CD_090010 + ENG_MDL_CD_210040 + ENG_MDL_CD_080080 + ENG_MDL_CD_140183 + ENG_MDL_CD_140295 + ENG_MDL_CD_110020 + ENG_MDL_CD_160013 + ENG_MDL_CD_140167 + ENG_MDL_CD_070020 + ENG_MDL_CD_140135 + ENG_MDL_CD_040020 + ENG_MDL_CD_165170 + ENG_MDL_CD_140173 + ENG_MDL_CD_010050 + ENG_MDL_CD_165090 + ENG_MDL_CD_220028 + ENG_MDL_CD_195060 + ENG_MDL_CD_160022 + ENG_MDL_CD_010080 + ENG_MDL_CD_150020 + ENG_MDL_CD_210100 + ENG_MDL_CD_180020 + ENG_MDL_CD_020056 + ENG_MDL_CD_140028 + ENG_MDL_CD_210070 + ENG_MDL_CD_140105 + ENG_MDL_CD_070018 + ENG_MDL_CD_140065 + ENG_MDL_CD_180040 + ENG_MDL_CD_165215 + ENG_MDL_CD_165230 + ENG_MDL_CD_165057 + ENG_MDL_CD_140220 + ENG_MDL_CD_010045 + ENG_MDL_CD_010006 + ENG_MDL_CD_165010 + ENG_MDL_CD_165130 + ENG_MDL_CD_160060 + ENG_MDL_CD_030020 + ENG_MDL_CD_010033 + ENG_MDL_CD_245010 + ENG_MDL_CD_010005 + ENG_MDL_CD_110060 + ENG_MDL_CD_120010 + ENG_MDL_CD_110028 + ENG_MDL_CD_020100 + ENG_MDL_CD_150150 + ENG_MDL_CD_020045 + ENG_MDL_CD_030047 + ENG_MDL_CD_165200 + ENG_MDL_CD_140175 + ENG_MDL_CD_020125 + ENG_MDL_CD_020060 + ENG_MDL_CD_195050 + ENG_MDL_CD_220019 + ENG_MDL_CD_070035 + ENG_MDL_CD_020055 + ENG_MDL_CD_050020 + ENG_MDL_CD_220029 + ENG_MDL_CD_090085 + ENG_MDL_CD_160015 + ENG_MDL_CD_165195 + ENG_MDL_CD_120020 + ENG_MDL_CD_020094 + ENG_MDL_CD_165070 + ENG_MDL_CD_165225 + ENG_MDL_CD_140095 + ENG_MDL_CD_165175 + ENG_MDL_CD_140160 + ENG_MDL_CD_130030 + ENG_MDL_CD_140080 + ENG_MDL_CD_195025 + ENG_MDL_CD_110070 + ENG_MDL_CD_090087 + ENG_MDL_CD_180050 + ENG_MDL_CD_140010 + ENG_MDL_CD_010012 + ENG_MDL_CD_165180 + ENG_MDL_CD_210050 + ENG_MDL_CD_140215 + ENG_MDL_CD_165051 + ENG_MDL_CD_150005 + ENG_MDL_CD_210130 + ENG_MDL_CD_090071 + ENG_MDL_CD_140188 + ENG_MDL_CD_140234 + ENG_MDL_CD_160065 + ENG_MDL_CD_140130 + ENG_MDL_CD_140181 + ENG_MDL_CD_140030 + ENG_MDL_CD_030050 + ENG_MDL_CD_210060 + ENG_MDL_CD_080070 + ENG_MDL_CD_165185 + ENG_MDL_CD_070040 + ENG_MDL_CD_165040 + ENG_MDL_CD_210142 + ENG_MDL_CD_160010 + ENG_MDL_CD_030040 + ENG_MDL_CD_010039 + ENG_MDL_CD_030115 + ENG_MDL_CD_210143 + ENG_MDL_CD_165033 + ENG_MDL_CD_140193 + ENG_MDL_CD_140085 + ENG_MDL_CD_210080 + ENG_MDL_CD_010008 + ENG_MDL_CD_195022 + ENG_MDL_CD_140194 + ENG_MDL_CD_150010 + ENG_MDL_CD_140035 + ENG_MDL_CD_220034 + ENG_MDL_CD_020010 + ENG_MDL_CD_140027 + ENG_MDL_CD_010030 + ENG_MDL_CD_030045 + ENG_MDL_CD_140250 + ENG_MDL_CD_140325 + ENG_MDL_CD_210020 + ENG_MDL_CD_210110 + ENG_MDL_CD_165150 + ENG_MDL_CD_020090 + ENG_MDL_CD_040030 + ENG_MDL_CD_140045 + ENG_MDL_CD_110065 + ENG_MDL_CD_180042 + ENG_MDL_CD_090015 + ENG_MDL_CD_010010 + ENG_MDL_CD_020050 + ENG_MDL_CD_140191 + ENG_MDL_CD_140330 + ENG_MDL_CD_110022 + ENG_MDL_CD_080115 + ENG_MDL_CD_020020 + ENG_MDL_CD_165080 + ENG_MDL_CD_165100 + ENG_MDL_CD_140168 + ENG_MDL_CD_070015 + ENG_MDL_CD_140040 + ENG_MDL_CD_140155 + ENG_MDL_CD_110023 + ENG_MDL_CD_090075 + ENG_MDL_CD_080020 + ENG_MDL_CD_160012 + ENG_MDL_CD_010075 + ENG_MDL_CD_110067 + ENG_MDL_CD_110027 + ENG_MDL_CD_080085 + ENG_MDL_CD_195070 + ENG_MDL_CD_040010 + ENG_MDL_CD_020075 + ENG_MDL_CD_110030 + ENG_MDL_CD_090045 + ENG_MDL_CD_030060 + ENG_MDL_CD_140115 + ENG_MDL_CD_180014 + ENG_MDL_CD_010036 + ENG_MDL_CD_020091 + ENG_MDL_CD_165030 + ENG_MDL_CD_165120 + ENG_MDL_CD_080050 + ENG_MDL_CD_010038 + ENG_MDL_CD_165031 + ENG_MDL_CD_030080 + ENG_MDL_CD_010035 + ENG_MDL_CD_110050 + ENG_MDL_CD_160020 + ENG_MDL_CD_210090 + ENG_MDL_CD_165050 + ENG_MDL_CD_140300 + ENG_MDL_CD_210140 + ENG_MDL_CD_090047 + ENG_MDL_CD_020080 + ENG_MDL_CD_165110 + ENG_MDL_CD_210145 + ENG_MDL_CD_140150 + ENG_MDL_CD_160030 + ENG_MDL_CD_140055 + ENG_MDL_CD_080012 + ENG_MDL_CD_165007 + ENG_MDL_CD_140025 + ENG_MDL_CD_210030 + ENG_MDL_CD_020085 + ENG_MDL_CD_020070 + ENG_MDL_CD_165055 + ENG_MDL_CD_030105 + ENG_MDL_CD_150003 + ENG_MDL_CD_200010 + ENG_MDL_CD_140185 + ENG_MDL_CD_020072 + ENG_MDL_CD_020093 + ENG_MDL_CD_030051 + ENG_MDL_CD_080015 + ENG_MDL_CD_165140 + ENG_MDL_CD_140186 + ENG_MDL_CD_090070 + ENG_MDL_CD_180052 + ENG_MDL_CD_090050 + ENG_MDL_CD_150007 + ENG_MDL_CD_010040 + ENG_MDL_CD_030030 + ENG_MDL_CD_090040 + ENG_MDL_CD_020026 + ENG_MDL_CD_210120 + ENG_MDL_CD_090077 + ENG_MDL_CD_090020 + ENG_MDL_CD_010070 + ENG_MDL_CD_080084 + ENG_MDL_CD_160025 + ENG_MDL_CD_020105 + ENG_MDL_CD_140205 + ENG_MDL_CD_020120 + ENG_MDL_CD_020078 + ENG_MDL_CD_010020 + ENG_MDL_CD_020040 + ENG_MDL_CD_020027 + ENG_MDL_CD_070025 + ENG_MDL_CD_110075 + ENG_MDL_CD_090100 + ENG_MDL_CD_030015 + ENG_MDL_CD_160034 + ENG_MDL_CD_090053 + ENG_MDL_CD_010007 + ENG_MDL_CD_070022 + ENG_MDL_CD_080110 + ENG_MDL_CD_160040 + ENG_MDL_CD_140210 + ENG_MDL_CD_165190 + ENG_MDL_CD_140180 + ENG_MDL_CD_140170 + ENG_MDL_CD_210165 + ENG_MDL_CD_140110 + ENG_MDL_CD_010037 + ENG_MDL_CD_140140 + ENG_MDL_CD_140195 + ENG_MDL_CD_140090 + ENG_MDL_CD_165009 + ENG_MDL_CD_140145 + ENG_MDL_CD_010015 + ENG_MDL_CD_010009 + ENG_MDL_CD_220023 + ENG_MDL_CD_090005 + ENG_MDL_CD_110010 + ENG_MDL_CD_090030 + ENG_MDL_CD_030090 + ENG_MDL_CD_140020 + ENG_MDL_CD_080100 + ENG_MDL_CD_070023 + ENG_MDL_CD_140125 + ENG_MDL_CD_030035 ')
# forms.append(' + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + ENG_MDL_CD_grp4 + ENG_MDL_CD_grp5 + ENG_MDL_CD_grp6 + ENG_MDL_CD_grp7 + ENG_MDL_CD_grp8 + ENG_MDL_CD_grp9 + ENG_MDL_CD_grp10 + ENG_MDL_CD_grp11 + PLNT_CD_U + PLNT_CD_J + PLNT_CD_G + PLNT_CD_F + PLNT_CD_B + PLNT_CD_P + PLNT_CD_K + PLNT_CD_C  + PLNT_CD_Y + PLNT_CD_0 + PLNT_CD_1 + PLNT_CD_L + PLNT_CD_N + PLNT_CD_R + PLNT_CD_3 + PLNT_CD_2 + PLNT_CD_M + PLNT_CD_7 + PLNT_CD_S + PLNT_CD_T + PLNT_CD_W + PLNT_CD_4 + PLNT_CD_A + PLNT_CD_D + PLNT_CD_V + PLNT_CD_9 + PLNT_CD_E + PLNT_CD_8 + PLNT_CD_5' )
# forms.append(' + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + ENG_MDL_CD_grp4 + ENG_MDL_CD_grp5 + ENG_MDL_CD_grp6 + ENG_MDL_CD_grp7 + ENG_MDL_CD_grp8 + ENG_MDL_CD_grp9 + ENG_MDL_CD_grp10 + ENG_MDL_CD_grp11 + NADA_BODY1__ + NADA_BODY1_U  + NADA_BODY1_G  + NADA_BODY1_B + NADA_BODY1_P + NADA_BODY1_K + NADA_BODY1_I + NADA_BODY1_6 + NADA_BODY1_C + NADA_BODY1_H + NADA_BODY1_1 + NADA_BODY1_L + NADA_BODY1_N + NADA_BODY1_R + NADA_BODY1_3 + NADA_BODY1_2 + NADA_BODY1_M + NADA_BODY1_7 + NADA_BODY1_S + NADA_BODY1_T + NADA_BODY1_W + NADA_BODY1_4 + NADA_BODY1_A + NADA_BODY1_D + NADA_BODY1_V + NADA_BODY1_9 + NADA_BODY1_E + NADA_BODY1_8 + NADA_BODY1_5 + PLNT_CD_grp1  + PLNT_CD_grp2' )
# forms.append(' + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + ENG_MDL_CD_grp4 + ENG_MDL_CD_grp5 + ENG_MDL_CD_grp6 + ENG_MDL_CD_grp7 + ENG_MDL_CD_grp8 + ENG_MDL_CD_grp9 + ENG_MDL_CD_grp10 + ENG_MDL_CD_grp11 + MAK_NM_MAXIM + MAK_NM_PETERBILT + MAK_NM_DODGE + MAK_NM_DUPLEX + MAK_NM_CATERPILLAR + MAK_NM_WESTERN_STAR_AUTO_CAR + MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_ + MAK_NM_MOTOR_COACH_INDUSTRIES + MAK_NM_INDIANA_PHOENIX + MAK_NM_DIAMOND_REO + MAK_NM_LES_AUTOBUS_MCI + MAK_NM_GMC + MAK_NM_VOLVO + MAK_NM_OTTAWA + MAK_NM_HYUNDAI + MAK_NM_KALMAR + MAK_NM_CHEVROLET + MAK_NM_INTERNATIONAL + MAK_NM_CRANE_CARRIER + MAK_NM_MARMON_HERRINGTON + MAK_NM_UTILIMASTER  + MAK_NM_GIANT + MAK_NM_IVECO + MAK_NM_SEAGRAVE_FIRE_APPARATUS + MAK_NM_AUTOCAR + MAK_NM_THOMAS + MAK_NM_MITSUBISHI_FUSO_TRUCK_OF_AMERICA_INC + MAK_NM_MACK + MAK_NM_EMERGENCY_ONE + MAK_NM_VAN_HOOL + MAK_NM_HINO + MAK_NM_AUTOCAR_LLC + MAK_NM_PIERCE_MFG__INC_ + MAK_NM_LODAL + MAK_NM_WINNEBAGO + MAK_NM_IC_CORPORATION + MAK_NM_ROADMASTER_RAIL + MAK_NM_COUNTRY_COACH_MOTORHOME + MAK_NM_CHANCE_COACH_TRANSIT_BUS + MAK_NM_FREIGHTLINER + MAK_NM_ADVANCE_MIXER + MAK_NM_HENDRICKSON + MAK_NM_FWD_CORPORATION + MAK_NM_BLUE_BIRD + MAK_NM_MERCEDES_BENZ + MAK_NM_EVOBUS + MAK_NM_PREVOST + MAK_NM_AMERICAN_LA_FRANCE + MAK_NM_CAPACITY_OF_TEXAS + MAK_NM_FEDERAL_MOTORS + MAK_NM_GILLIG + MAK_NM_SAVAGE + MAK_NM_JOHN_DEERE + MAK_NM_SCANIA + MAK_NM_RAM + MAK_NM_TRANSPORTATION_MFG_CORP_ + MAK_NM_WORKHORSE_CUSTOM_CHASSIS + MAK_NM_WHITE + MAK_NM_FORETRAVEL_MOTORHOME + MAK_NM_LAFORZA + MAK_NM_SPARTAN_MOTORS + MAK_NM_ZELIGSON + MAK_NM_TEREX___TEREX_ADVANCE + MAK_NM_WHITE_GMC + MAK_NM_STERLING_TRUCK + MAK_NM_NISSAN_DIESEL + MAK_NM_UNIMOG + MAK_NM_NEW_FLYER + MAK_NM_BERING + MAK_NM_KENWORTH + MAK_NM_ISUZU  + PLNT_CD_grp1  + PLNT_CD_grp2 ' )
# forms.append(' + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + ENG_MDL_CD_grp4 + ENG_MDL_CD_grp5 + ENG_MDL_CD_grp6 + ENG_MDL_CD_grp7 + ENG_MDL_CD_grp8 + ENG_MDL_CD_grp9 + ENG_MDL_CD_grp10 + ENG_MDL_CD_grp11 + ENG_FUEL_CD_P + ENG_FUEL_CD_U  + ENG_FUEL_CD_C + ENG_FUEL_CD_Y + ENG_FUEL_CD_G + ENG_FUEL_CD_N + ENG_FUEL_CD_F  + PLNT_CD_grp1  + PLNT_CD_grp2' )
# forms.append(' + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + ENG_MDL_CD_grp4 + ENG_MDL_CD_grp5 + ENG_MDL_CD_grp6 + ENG_MDL_CD_grp7 + ENG_MDL_CD_grp8 + ENG_MDL_CD_grp9 + ENG_MDL_CD_grp10 + ENG_MDL_CD_grp11 + PLNT_CD_grp1  + PLNT_CD_grp2 + MAK_NM_grp1 + MAK_NM_grp2 + MAK_NM_grp3 + ENG_FUEL_CD_G - fuel_gas - cko_fuel_d2 - cko_mdl_yr - Liab_Ex_Veh_Age ' )
# # 17
# forms.append(' + NADA_BODY1__ + NADA_BODY1_U  + NADA_BODY1_G  + NADA_BODY1_B + NADA_BODY1_P + NADA_BODY1_K + NADA_BODY1_I + NADA_BODY1_6 + NADA_BODY1_C + NADA_BODY1_H + NADA_BODY1_1 + NADA_BODY1_L + NADA_BODY1_N + NADA_BODY1_R + NADA_BODY1_3 + NADA_BODY1_2 + NADA_BODY1_M + NADA_BODY1_7 + NADA_BODY1_S + NADA_BODY1_T + NADA_BODY1_W + NADA_BODY1_4 + NADA_BODY1_A + NADA_BODY1_D + NADA_BODY1_V + NADA_BODY1_9 + NADA_BODY1_E + NADA_BODY1_8 + NADA_BODY1_5 ' )
# forms.append(' + MAK_NM_MAXIM + MAK_NM_PETERBILT + MAK_NM_DODGE + MAK_NM_DUPLEX + MAK_NM_CATERPILLAR + MAK_NM_WESTERN_STAR_AUTO_CAR + MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_ + MAK_NM_MOTOR_COACH_INDUSTRIES + MAK_NM_INDIANA_PHOENIX + MAK_NM_DIAMOND_REO + MAK_NM_LES_AUTOBUS_MCI + MAK_NM_GMC + MAK_NM_VOLVO + MAK_NM_OTTAWA + MAK_NM_HYUNDAI + MAK_NM_KALMAR + MAK_NM_CHEVROLET + MAK_NM_INTERNATIONAL + MAK_NM_CRANE_CARRIER + MAK_NM_MARMON_HERRINGTON + MAK_NM_UTILIMASTER  + MAK_NM_GIANT + MAK_NM_IVECO + MAK_NM_SEAGRAVE_FIRE_APPARATUS + MAK_NM_AUTOCAR + MAK_NM_THOMAS + MAK_NM_MITSUBISHI_FUSO_TRUCK_OF_AMERICA_INC + MAK_NM_MACK + MAK_NM_EMERGENCY_ONE + MAK_NM_VAN_HOOL + MAK_NM_HINO + MAK_NM_AUTOCAR_LLC + MAK_NM_PIERCE_MFG__INC_ + MAK_NM_LODAL + MAK_NM_WINNEBAGO + MAK_NM_IC_CORPORATION + MAK_NM_ROADMASTER_RAIL + MAK_NM_COUNTRY_COACH_MOTORHOME + MAK_NM_CHANCE_COACH_TRANSIT_BUS + MAK_NM_FREIGHTLINER + MAK_NM_ADVANCE_MIXER + MAK_NM_HENDRICKSON + MAK_NM_FWD_CORPORATION + MAK_NM_BLUE_BIRD + MAK_NM_MERCEDES_BENZ + MAK_NM_EVOBUS + MAK_NM_PREVOST + MAK_NM_AMERICAN_LA_FRANCE + MAK_NM_CAPACITY_OF_TEXAS + MAK_NM_FEDERAL_MOTORS + MAK_NM_GILLIG + MAK_NM_SAVAGE + MAK_NM_JOHN_DEERE + MAK_NM_SCANIA + MAK_NM_RAM + MAK_NM_TRANSPORTATION_MFG_CORP_ + MAK_NM_WORKHORSE_CUSTOM_CHASSIS + MAK_NM_WHITE + MAK_NM_FORETRAVEL_MOTORHOME + MAK_NM_LAFORZA + MAK_NM_SPARTAN_MOTORS + MAK_NM_ZELIGSON + MAK_NM_TEREX___TEREX_ADVANCE + MAK_NM_WHITE_GMC + MAK_NM_STERLING_TRUCK + MAK_NM_NISSAN_DIESEL + MAK_NM_UNIMOG + MAK_NM_NEW_FLYER + MAK_NM_BERING + MAK_NM_KENWORTH + MAK_NM_ISUZU  ' )
# forms.append(' + ENG_FUEL_CD_P + ENG_FUEL_CD_U  + ENG_FUEL_CD_C + ENG_FUEL_CD_Y + ENG_FUEL_CD_G + ENG_FUEL_CD_N + ENG_FUEL_CD_F  ' )
# forms.append(' + PLNT_CD_U + PLNT_CD_J + PLNT_CD_G + PLNT_CD_F + PLNT_CD_B + PLNT_CD_P + PLNT_CD_K + PLNT_CD_C  + PLNT_CD_Y + PLNT_CD_0 + PLNT_CD_1 + PLNT_CD_L + PLNT_CD_N + PLNT_CD_R + PLNT_CD_3 + PLNT_CD_2 + PLNT_CD_M + PLNT_CD_7 + PLNT_CD_S + PLNT_CD_T + PLNT_CD_W + PLNT_CD_4 + PLNT_CD_A + PLNT_CD_D + PLNT_CD_V + PLNT_CD_9 + PLNT_CD_E + PLNT_CD_8 + PLNT_CD_5' )
# forms.append(' + ENG_MDL_CD_220026 + ENG_MDL_CD_030070 + ENG_MDL_CD_U + ENG_MDL_CD_160066 + ENG_MDL_CD_180010 + ENG_MDL_CD_140050 + ENG_MDL_CD_140199 + ENG_MDL_CD_195080 + ENG_MDL_CD_140182 + ENG_MDL_CD_010011 + ENG_MDL_CD_180012 + ENG_MDL_CD_165160 + ENG_MDL_CD_080044 + ENG_MDL_CD_165035 + ENG_MDL_CD_120030 + ENG_MDL_CD_140190 + ENG_MDL_CD_150060 + ENG_MDL_CD_180030 + ENG_MDL_CD_210122 + ENG_MDL_CD_140165 + ENG_MDL_CD_200020 + ENG_MDL_CD_140200 + ENG_MDL_CD_165220 + ENG_MDL_CD_090043 + ENG_MDL_CD_030110 + ENG_MDL_CD_110040 + ENG_MDL_CD_165020 + ENG_MDL_CD_180016 + ENG_MDL_CD_090011 + ENG_MDL_CD_195001 + ENG_MDL_CD_090051 + ENG_MDL_CD_165008 + ENG_MDL_CD_140310 + ENG_MDL_CD_090010 + ENG_MDL_CD_210040 + ENG_MDL_CD_080080 + ENG_MDL_CD_140183 + ENG_MDL_CD_140295 + ENG_MDL_CD_110020 + ENG_MDL_CD_160013 + ENG_MDL_CD_140167 + ENG_MDL_CD_070020 + ENG_MDL_CD_140135 + ENG_MDL_CD_040020 + ENG_MDL_CD_165170 + ENG_MDL_CD_140173 + ENG_MDL_CD_010050 + ENG_MDL_CD_165090 + ENG_MDL_CD_220028 + ENG_MDL_CD_195060 + ENG_MDL_CD_160022 + ENG_MDL_CD_010080 + ENG_MDL_CD_150020 + ENG_MDL_CD_210100 + ENG_MDL_CD_180020 + ENG_MDL_CD_020056 + ENG_MDL_CD_140028 + ENG_MDL_CD_210070 + ENG_MDL_CD_140105 + ENG_MDL_CD_070018 + ENG_MDL_CD_140065 + ENG_MDL_CD_180040 + ENG_MDL_CD_165215 + ENG_MDL_CD_165230 + ENG_MDL_CD_165057 + ENG_MDL_CD_140220 + ENG_MDL_CD_010045 + ENG_MDL_CD_010006 + ENG_MDL_CD_165010 + ENG_MDL_CD_165130 + ENG_MDL_CD_160060 + ENG_MDL_CD_030020 + ENG_MDL_CD_010033 + ENG_MDL_CD_245010 + ENG_MDL_CD_010005 + ENG_MDL_CD_110060 + ENG_MDL_CD_120010 + ENG_MDL_CD_110028 + ENG_MDL_CD_020100 + ENG_MDL_CD_150150 + ENG_MDL_CD_020045 + ENG_MDL_CD_030047 + ENG_MDL_CD_165200 + ENG_MDL_CD_140175 + ENG_MDL_CD_020125 + ENG_MDL_CD_020060 + ENG_MDL_CD_195050 + ENG_MDL_CD_220019 + ENG_MDL_CD_070035 + ENG_MDL_CD_020055 + ENG_MDL_CD_050020 + ENG_MDL_CD_220029 + ENG_MDL_CD_090085 + ENG_MDL_CD_160015 + ENG_MDL_CD_165195 + ENG_MDL_CD_120020 + ENG_MDL_CD_020094 + ENG_MDL_CD_165070 + ENG_MDL_CD_165225 + ENG_MDL_CD_140095 + ENG_MDL_CD_165175 + ENG_MDL_CD_140160 + ENG_MDL_CD_130030 + ENG_MDL_CD_140080 + ENG_MDL_CD_195025 + ENG_MDL_CD_110070 + ENG_MDL_CD_090087 + ENG_MDL_CD_180050 + ENG_MDL_CD_140010 + ENG_MDL_CD_010012 + ENG_MDL_CD_165180 + ENG_MDL_CD_210050 + ENG_MDL_CD_140215 + ENG_MDL_CD_165051 + ENG_MDL_CD_150005 + ENG_MDL_CD_210130 + ENG_MDL_CD_090071 + ENG_MDL_CD_140188 + ENG_MDL_CD_140234 + ENG_MDL_CD_160065 + ENG_MDL_CD_140130 + ENG_MDL_CD_140181 + ENG_MDL_CD_140030 + ENG_MDL_CD_030050 + ENG_MDL_CD_210060 + ENG_MDL_CD_080070 + ENG_MDL_CD_165185 + ENG_MDL_CD_070040 + ENG_MDL_CD_165040 + ENG_MDL_CD_210142 + ENG_MDL_CD_160010 + ENG_MDL_CD_030040 + ENG_MDL_CD_010039 + ENG_MDL_CD_030115 + ENG_MDL_CD_210143 + ENG_MDL_CD_165033 + ENG_MDL_CD_140193 + ENG_MDL_CD_140085 + ENG_MDL_CD_210080 + ENG_MDL_CD_010008 + ENG_MDL_CD_195022 + ENG_MDL_CD_140194 + ENG_MDL_CD_150010 + ENG_MDL_CD_140035 + ENG_MDL_CD_220034 + ENG_MDL_CD_020010 + ENG_MDL_CD_140027 + ENG_MDL_CD_010030 + ENG_MDL_CD_030045 + ENG_MDL_CD_140250 + ENG_MDL_CD_140325 + ENG_MDL_CD_210020 + ENG_MDL_CD_210110 + ENG_MDL_CD_165150 + ENG_MDL_CD_020090 + ENG_MDL_CD_040030 + ENG_MDL_CD_140045 + ENG_MDL_CD_110065 + ENG_MDL_CD_180042 + ENG_MDL_CD_090015 + ENG_MDL_CD_010010 + ENG_MDL_CD_020050 + ENG_MDL_CD_140191 + ENG_MDL_CD_140330 + ENG_MDL_CD_110022 + ENG_MDL_CD_080115 + ENG_MDL_CD_020020 + ENG_MDL_CD_165080 + ENG_MDL_CD_165100 + ENG_MDL_CD_140168 + ENG_MDL_CD_070015 + ENG_MDL_CD_140040 + ENG_MDL_CD_140155 + ENG_MDL_CD_110023 + ENG_MDL_CD_090075 + ENG_MDL_CD_080020 + ENG_MDL_CD_160012 + ENG_MDL_CD_010075 + ENG_MDL_CD_110067 + ENG_MDL_CD_110027 + ENG_MDL_CD_080085 + ENG_MDL_CD_195070 + ENG_MDL_CD_040010 + ENG_MDL_CD_020075 + ENG_MDL_CD_110030 + ENG_MDL_CD_090045 + ENG_MDL_CD_030060 + ENG_MDL_CD_140115 + ENG_MDL_CD_180014 + ENG_MDL_CD_010036 + ENG_MDL_CD_020091 + ENG_MDL_CD_165030 + ENG_MDL_CD_165120 + ENG_MDL_CD_080050 + ENG_MDL_CD_010038 + ENG_MDL_CD_165031 + ENG_MDL_CD_030080 + ENG_MDL_CD_010035 + ENG_MDL_CD_110050 + ENG_MDL_CD_160020 + ENG_MDL_CD_210090 + ENG_MDL_CD_165050 + ENG_MDL_CD_140300 + ENG_MDL_CD_210140 + ENG_MDL_CD_090047 + ENG_MDL_CD_020080 + ENG_MDL_CD_165110 + ENG_MDL_CD_210145 + ENG_MDL_CD_140150 + ENG_MDL_CD_160030 + ENG_MDL_CD_140055 + ENG_MDL_CD_080012 + ENG_MDL_CD_165007 + ENG_MDL_CD_140025 + ENG_MDL_CD_210030 + ENG_MDL_CD_020085 + ENG_MDL_CD_020070 + ENG_MDL_CD_165055 + ENG_MDL_CD_030105 + ENG_MDL_CD_150003 + ENG_MDL_CD_200010 + ENG_MDL_CD_140185 + ENG_MDL_CD_020072 + ENG_MDL_CD_020093 + ENG_MDL_CD_030051 + ENG_MDL_CD_080015 + ENG_MDL_CD_165140 + ENG_MDL_CD_140186 + ENG_MDL_CD_090070 + ENG_MDL_CD_180052 + ENG_MDL_CD_090050 + ENG_MDL_CD_150007 + ENG_MDL_CD_010040 + ENG_MDL_CD_030030 + ENG_MDL_CD_090040 + ENG_MDL_CD_020026 + ENG_MDL_CD_210120 + ENG_MDL_CD_090077 + ENG_MDL_CD_090020 + ENG_MDL_CD_010070 + ENG_MDL_CD_080084 + ENG_MDL_CD_160025 + ENG_MDL_CD_020105 + ENG_MDL_CD_140205 + ENG_MDL_CD_020120 + ENG_MDL_CD_020078 + ENG_MDL_CD_010020 + ENG_MDL_CD_020040 + ENG_MDL_CD_020027 + ENG_MDL_CD_070025 + ENG_MDL_CD_110075 + ENG_MDL_CD_090100 + ENG_MDL_CD_030015 + ENG_MDL_CD_160034 + ENG_MDL_CD_090053 + ENG_MDL_CD_010007 + ENG_MDL_CD_070022 + ENG_MDL_CD_080110 + ENG_MDL_CD_160040 + ENG_MDL_CD_140210 + ENG_MDL_CD_165190 + ENG_MDL_CD_140180 + ENG_MDL_CD_140170 + ENG_MDL_CD_210165 + ENG_MDL_CD_140110 + ENG_MDL_CD_010037 + ENG_MDL_CD_140140 + ENG_MDL_CD_140195 + ENG_MDL_CD_140090 + ENG_MDL_CD_165009 + ENG_MDL_CD_140145 + ENG_MDL_CD_010015 + ENG_MDL_CD_010009 + ENG_MDL_CD_220023 + ENG_MDL_CD_090005 + ENG_MDL_CD_110010 + ENG_MDL_CD_090030 + ENG_MDL_CD_030090 + ENG_MDL_CD_140020 + ENG_MDL_CD_080100 + ENG_MDL_CD_070023 + ENG_MDL_CD_140125 + ENG_MDL_CD_030035 ')
# # 22
# forms.append(' + NADA_BODY1_U + NADA_BODY1_grp1 + NADA_BODY1_grp2 + NADA_BODY1_grp3 + MAK_NM_grp1 + MAK_NM_grp2 + MAK_NM_grp3 + MAK_NM_grp4 + MAK_NM_grp5 + MAK_NM_grp6 + ENG_FUEL_CD_grp1 + ENG_FUEL_CD_grp2 + PLNT_CD_grp1 + PLNT_CD_grp2 + PLNT_CD_grp3 + PLNT_CD_grp4 + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + ENG_MDL_CD_grp4 ')
# forms.append(' + NADA_BODY1_U + NADA_BODY1_grp1 + NADA_BODY1_grp2 + NADA_BODY1_grp3 + MAK_NM_grp1 + MAK_NM_grp2 + MAK_NM_grp3 + MAK_NM_grp4 + MAK_NM_grp5 + MAK_NM_grp6 + ENG_FUEL_CD_grp1  + PLNT_CD_grp1 + PLNT_CD_grp2 + PLNT_CD_grp3 + PLNT_CD_grp4 + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + ENG_MDL_CD_grp4 + ENG_FUEL_CD_G + ENG_FUEL_CD_P - cko_fuel_d2 - fuel_gas')
# forms.append(' + NADA_BODY1_U + NADA_BODY1_grp1 + NADA_BODY1_grp2 + NADA_BODY1_grp3 + MAK_NM_grp1 + MAK_NM_grp2 + MAK_NM_grp3 + MAK_NM_grp4 + MAK_NM_grp5 + MAK_NM_grp6 + ENG_FUEL_CD_grp1  + PLNT_CD_grp1 + PLNT_CD_grp2 + PLNT_CD_grp3  + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp4 + ENG_FUEL_CD_G  - cko_fuel_d2 - fuel_gas')
# forms.append(' + NADA_BODY1_U + NADA_BODY1_grp1 + NADA_BODY1_grp2  + MAK_NM_grp1 + MAK_NM_grp2 +  MAK_NM_grp4 + MAK_NM_grp5 + MAK_NM_grp6 + ENG_FUEL_CD_grp1  + PLNT_CD_grp1 + PLNT_CD_grp2 + PLNT_CD_grp3  + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp4 + ENG_FUEL_CD_G  - cko_fuel_d2 - fuel_gas')
# forms.append(' + NADA_BODY1_U + NADA_BODY1_grp1 + NADA_BODY1_grp2  + MAK_NM_grp1 + MAK_NM_grp2 +  MAK_NM_grp4 + MAK_NM_grp5 + MAK_NM_grp6 + ENG_FUEL_CD_grp1  + PLNT_CD_grp1 + PLNT_CD_grp2 + PLNT_CD_grp3  + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp4 + ENG_FUEL_CD_G  - cko_fuel_d2 - fuel_gas - SHIP_WGHT_LBSMS - cko_max_gvwc_lt11k')
# forms.append(' + NADA_BODY1_U + NADA_BODY1_grp1 + NADA_BODY1_grp2  + MAK_NM_grp1 + MAK_NM_grp2 +  MAK_NM_grp4 + MAK_NM_grp5 + MAK_NM_grp6 + ENG_FUEL_CD_grp1  + PLNT_CD_grp1 + PLNT_CD_grp2 + PLNT_CD_grp3  + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp4 + ENG_FUEL_CD_G  - cko_fuel_d2 - fuel_gas - SHIP_WGHT_LBSMS - cko_max_gvwc_lt11k - ENG_MFG_CD_d7')
# forms.append(' + NADA_BODY1_U + NADA_BODY1_grp1 + NADA_BODY1_grp2  + MAK_NM_grp1 + MAK_NM_grp4 + MAK_NM_grp5 + MAK_NM_grp6 + ENG_FUEL_CD_grp1  + PLNT_CD_grp1 + PLNT_CD_grp2 + PLNT_CD_grp3  + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp4 + ENG_FUEL_CD_G  - cko_fuel_d2 - fuel_gas - SHIP_WGHT_LBSMS - cko_max_gvwc_lt11k - ENG_MFG_CD_d7')
# forms.append(' + NADA_BODY1_U + NADA_BODY1_grp1 + NADA_BODY1_grp2  + MAK_NM_grp1 + MAK_NM_grp2 + MAK_NM_grp3 + MAK_NM_grp4 + MAK_NM_grp5 + MAK_NM_grp6 + ENG_FUEL_CD_grp1  + PLNT_CD_grp1 + PLNT_CD_grp2 + PLNT_CD_grp3  + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp4 + ENG_FUEL_CD_G  - cko_fuel_d2 - fuel_gas - SHIP_WGHT_LBSMS - cko_max_gvwc_lt11k - ENG_MFG_CD_d7')
# # 30


# # Biggest levels + ENG_MDL_CD_165060 + PLNT_CD_H  + NADA_BODY1_F + MAK_NM_FORD + ENG_FUEL_CD_D
# for i in range(29,len(forms)):
# 	# Include only catfeatures that are needed for the attempted GLMs
# 	CATFEATURES = CATFAARRAY[i]
	
# 	DUMMYFEATURES=[]
# 	for feature in CATFEATURES:
# 		dummiesTrain=pd.get_dummies(data_train[feature],prefix=feature)
# 		dummiesTest=pd.get_dummies(data_test[feature],prefix=feature)
# 		temp=dummiesTrain.sum()	
# 		# If column appears in data_train but not data_test we need to add it to data_test as a column of zeroes
# 		for k in ml.returnNotMatches(dummiesTrain.columns.values,dummiesTest.columns.values)[0]:
# 			dummiesTest[k]=0	
# 		dummiesTrain=dummiesTrain[temp[temp<temp.max()].index.values]
# 		dummiesTest=dummiesTest[temp[temp<temp.max()].index.values]	
# 		if(len(dummiesTrain.columns)>0):
# 			dummiesTrain.columns = dummiesTrain.columns.str.replace('\s+', '_')
# 			dummiesTest.columns = dummiesTest.columns.str.replace('\s+', '_')
# 			data_train=pd.concat([data_train,dummiesTrain],axis=1)
# 			data_test=pd.concat([data_test,dummiesTest],axis=1)
# 			DUMMYFEATURES += list(dummiesTrain)

# 	print(DUMMYFEATURES)
			
	

# 	if i==1:	
# 		data_train['BODY_STYLE_DESC_GARBAGE_REFUSE']=data_train['BODY_STYLE_DESC_GARBAGE/REFUSE']
# 		data_test['BODY_STYLE_DESC_GARBAGE_REFUSE']=data_test['BODY_STYLE_DESC_GARBAGE/REFUSE']	
# 	elif i==2:
# 		data_train['cko_max_gvwc_Tr_02_le20k']=data_train['cko_max_gvwc_Tr_02.le20k']
# 		data_test['cko_max_gvwc_Tr_02_le20k']=data_test['cko_max_gvwc_Tr_02.le20k']
# 	elif i==3:
# 		data_train['ENG_CLNDR_RTR_CNT__']=data_train['ENG_CLNDR_RTR_CNT_.']
# 		data_test['ENG_CLNDR_RTR_CNT__']=data_test['ENG_CLNDR_RTR_CNT_.']
# 	elif i==5:
# 		data_train['ENTERTAIN_GP_1_2_9']=data_train['ENTERTAIN_GP_1,2,9']
# 		data_test['ENTERTAIN_GP_1_2_9']=data_test['ENTERTAIN_GP_1,2,9']
# 		data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_train['MAK_NM_WESTERN_STAR/AUTO_CAR']
# 		data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_test['MAK_NM_WESTERN_STAR/AUTO_CAR']
# 		data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_train['MAK_NM_MERCEDES_BENZ']=data_train['MAK_NM_MERCEDES-BENZ']
# 		data_test['MAK_NM_MERCEDES_BENZ']=data_test['MAK_NM_MERCEDES-BENZ']
# 	elif i==9:
# 		data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_train['MAK_NM_MERCEDES_BENZ']=data_train['MAK_NM_MERCEDES-BENZ']
# 		data_test['MAK_NM_MERCEDES_BENZ']=data_test['MAK_NM_MERCEDES-BENZ']
# 	elif i==10:
# 		data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_train['MAK_NM_MERCEDES_BENZ']=data_train['MAK_NM_MERCEDES-BENZ']
# 		data_test['MAK_NM_MERCEDES_BENZ']=data_test['MAK_NM_MERCEDES-BENZ']
# 	elif i==13:
# 		data_train['NADA_BODY1__']=data_train['NADA_BODY1_"']
# 		data_test['NADA_BODY1__']=data_test['NADA_BODY1_"']
# 		data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_Y']+data_train['PLNT_CD_5']+data_train['PLNT_CD_8'])
# 		data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_Y']+data_test['PLNT_CD_5']+data_test['PLNT_CD_8'])
# 		data_train['PLNT_CD_grp2']=(data_train['PLNT_CD_C']+data_train['PLNT_CD_0']+data_train['PLNT_CD_9'])
# 		data_test['PLNT_CD_grp2']=(data_test['PLNT_CD_C']+data_test['PLNT_CD_0']+data_test['PLNT_CD_9'])
# 	elif i==14:
# 		data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_train['MAK_NM_MERCEDES_BENZ']=data_train['MAK_NM_MERCEDES-BENZ']
# 		data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_train['MAK_NM_WESTERN_STAR/AUTO_CAR']
# 		data_train['MAK_NM_PIERCE_MFG__INC_']=data_train['MAK_NM_PIERCE_MFG._INC.']
# 		data_train['MAK_NM_WHITE_GMC']=data_train['MAK_NM_WHITE/GMC']
# 		data_train['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_train['MAK_NM_TRANSPORTATION_MFG_CORP.']
# 		data_train['MAK_NM_TEREX___TEREX_ADVANCE']=data_train['MAK_NM_TEREX_/_TEREX_ADVANCE']
# 		# data_train['MAK_NM_SPARTAN_MOTORS']=data_train['MAK_NM_SPARTAN_MOTORS']		
# 		# data_train['MAK_NM_FORETRAVEL_MOTORHOME']=data_train['MAK_NM_FORETRAVEL MOTORHOME']
# 		# data_train['MAK_NM_ADVANCE_MIXER ']=data_train['MAK_NM_ADVANCE MIXER']

# 		data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_test['MAK_NM_MERCEDES_BENZ']=data_test['MAK_NM_MERCEDES-BENZ']	
# 		data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_test['MAK_NM_WESTERN_STAR/AUTO_CAR']
# 		data_test['MAK_NM_PIERCE_MFG__INC_']=data_test['MAK_NM_PIERCE_MFG._INC.']
# 		data_test['MAK_NM_WHITE_GMC']=data_test['MAK_NM_WHITE/GMC']
# 		data_test['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_test['MAK_NM_TRANSPORTATION_MFG_CORP.']
# 		data_test['MAK_NM_TEREX___TEREX_ADVANCE']=data_test['MAK_NM_TEREX_/_TEREX_ADVANCE']
# 		# data_test['MAK_NM_SPARTAN_MOTORS']=data_test['MAK_NM_SPARTAN_MOTORS']
# 		# data_test['MAK_NM_FORETRAVEL_MOTORHOME']=data_test['MAK_NM_FORETRAVEL MOTORHOME']
# 		# data_test['MAK_NM_ADVANCE_MIXER ']=data_test['MAK_NM_ADVANCE MIXER']

# 		data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_Y']+data_train['PLNT_CD_5']+data_train['PLNT_CD_8'])
# 		data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_Y']+data_test['PLNT_CD_5']+data_test['PLNT_CD_8'])
# 		data_train['PLNT_CD_grp2']=(data_train['PLNT_CD_C']+data_train['PLNT_CD_0']+data_train['PLNT_CD_9'])
# 		data_test['PLNT_CD_grp2']=(data_test['PLNT_CD_C']+data_test['PLNT_CD_0']+data_test['PLNT_CD_9'])
# 	elif i==15:
# 		data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_Y']+data_train['PLNT_CD_5']+data_train['PLNT_CD_8'])
# 		data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_Y']+data_test['PLNT_CD_5']+data_test['PLNT_CD_8'])
# 		data_train['PLNT_CD_grp2']=(data_train['PLNT_CD_C']+data_train['PLNT_CD_0']+data_train['PLNT_CD_9'])
# 		data_test['PLNT_CD_grp2']=(data_test['PLNT_CD_C']+data_test['PLNT_CD_0']+data_test['PLNT_CD_9'])
# 	elif i==16:
# 		data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_Y']+data_train['PLNT_CD_5']+data_train['PLNT_CD_8'])
# 		data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_Y']+data_test['PLNT_CD_5']+data_test['PLNT_CD_8'])
# 		data_train['PLNT_CD_grp2']=(data_train['PLNT_CD_C']+data_train['PLNT_CD_0']+data_train['PLNT_CD_9'])
# 		data_test['PLNT_CD_grp2']=(data_test['PLNT_CD_C']+data_test['PLNT_CD_0']+data_test['PLNT_CD_9'])	
# 		data_train['MAK_NM_grp1']=(data_train['MAK_NM_NEW_FLYER']+data_train['MAK_NM_MAXIM']+data_train['MAK_NM_UNIMOG']+data_train['MAK_NM_SAVAGE']+data_train['MAK_NM_COUNTRY_COACH_MOTORHOME']+data_train['MAK_NM_OTTAWA']+data_train['MAK_NM_FEDERAL_MOTORS']+data_train['MAK_NM_PREVOST']+data_train['MAK_NM_VAN_HOOL']+data_train['MAK_NM_SPARTAN_MOTORS']+data_train['MAK_NM_DIAMOND_REO']+data_train['MAK_NM_HYUNDAI']+data_train['MAK_NM_LES_AUTOBUS_MCI']+data_train['MAK_NM_EVOBUS']+data_train['MAK_NM_FWD_CORPORATION']+data_train['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_train['MAK_NM_JOHN_DEERE']+data_train['MAK_NM_GILLIG']+data_train['MAK_NM_ZELIGSON']+data_train['MAK_NM_DUPLEX']+data_train['MAK_NM_SEAGRAVE_FIRE_APPARATUS']+data_train['MAK_NM_PIERCE_MFG._INC.'])
# 		data_train['MAK_NM_grp2']=(data_train['MAK_NM_WINNEBAGO']+data_train['MAK_NM_CAPACITY_OF_TEXAS']+data_train['MAK_NM_TRANSPORTATION_MFG_CORP.']+data_train['MAK_NM_GIANT']+data_train['MAK_NM_MERCEDES-BENZ']+data_train['MAK_NM_MOTOR_COACH_INDUSTRIES']+data_train['MAK_NM_LAFORZA']+data_train['MAK_NM_THOMAS']+data_train['MAK_NM_MARMON_HERRINGTON']+data_train['MAK_NM_TEREX_/_TEREX_ADVANCE']+data_train['MAK_NM_LODAL']+data_train['MAK_NM_AMERICAN_LA_FRANCE']+data_train['MAK_NM_KALMAR']+data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.'])
# 		data_train['MAK_NM_grp3']=(data_train['MAK_NM_AUTOCAR']+data_train['MAK_NM_WHITE/GMC'])
# 		data_test['MAK_NM_grp1']=(data_test['MAK_NM_NEW_FLYER']+data_test['MAK_NM_MAXIM']+data_test['MAK_NM_UNIMOG']+data_test['MAK_NM_SAVAGE']+data_test['MAK_NM_COUNTRY_COACH_MOTORHOME']+data_test['MAK_NM_OTTAWA']+data_test['MAK_NM_FEDERAL_MOTORS']+data_test['MAK_NM_PREVOST']+data_test['MAK_NM_VAN_HOOL']+data_test['MAK_NM_SPARTAN_MOTORS']+data_test['MAK_NM_DIAMOND_REO']+data_test['MAK_NM_HYUNDAI']+data_test['MAK_NM_LES_AUTOBUS_MCI']+data_test['MAK_NM_EVOBUS']+data_test['MAK_NM_FWD_CORPORATION']+data_test['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_test['MAK_NM_JOHN_DEERE']+data_test['MAK_NM_GILLIG']+data_test['MAK_NM_ZELIGSON']+data_test['MAK_NM_DUPLEX']+data_test['MAK_NM_SEAGRAVE_FIRE_APPARATUS']+data_test['MAK_NM_PIERCE_MFG._INC.'])
# 		data_test['MAK_NM_grp2']=(data_test['MAK_NM_WINNEBAGO']+data_test['MAK_NM_CAPACITY_OF_TEXAS']+data_test['MAK_NM_TRANSPORTATION_MFG_CORP.']+data_test['MAK_NM_GIANT']+data_test['MAK_NM_MERCEDES-BENZ']+data_test['MAK_NM_MOTOR_COACH_INDUSTRIES']+data_test['MAK_NM_LAFORZA']+data_test['MAK_NM_THOMAS']+data_test['MAK_NM_MARMON_HERRINGTON']+data_test['MAK_NM_TEREX_/_TEREX_ADVANCE']+data_test['MAK_NM_LODAL']+data_test['MAK_NM_AMERICAN_LA_FRANCE']+data_test['MAK_NM_KALMAR']+data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.'])
# 		data_test['MAK_NM_grp3']=(data_test['MAK_NM_AUTOCAR']+data_test['MAK_NM_WHITE/GMC'])
# 		data_train['ENG_FUEL_CD_grp1']=(data_train['ENG_FUEL_CD_G']+data_train['ENG_FUEL_CD_F'])
# 		data_test['ENG_FUEL_CD_grp1']=(data_test['ENG_FUEL_CD_G']+data_test['ENG_FUEL_CD_F'])
# 	elif i==17:
# 		data_train['NADA_BODY1__']=data_train['NADA_BODY1_"']
# 		data_test['NADA_BODY1__']=data_test['NADA_BODY1_"']
# 	elif i==18:
# 		data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_train['MAK_NM_MERCEDES_BENZ']=data_train['MAK_NM_MERCEDES-BENZ']
# 		data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_train['MAK_NM_WESTERN_STAR/AUTO_CAR']
# 		data_train['MAK_NM_PIERCE_MFG__INC_']=data_train['MAK_NM_PIERCE_MFG._INC.']
# 		data_train['MAK_NM_WHITE_GMC']=data_train['MAK_NM_WHITE/GMC']
# 		data_train['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_train['MAK_NM_TRANSPORTATION_MFG_CORP.']
# 		data_train['MAK_NM_TEREX___TEREX_ADVANCE']=data_train['MAK_NM_TEREX_/_TEREX_ADVANCE']
# 		data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_test['MAK_NM_MERCEDES_BENZ']=data_test['MAK_NM_MERCEDES-BENZ']	
# 		data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_test['MAK_NM_WESTERN_STAR/AUTO_CAR']
# 		data_test['MAK_NM_PIERCE_MFG__INC_']=data_test['MAK_NM_PIERCE_MFG._INC.']
# 		data_test['MAK_NM_WHITE_GMC']=data_test['MAK_NM_WHITE/GMC']
# 		data_test['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_test['MAK_NM_TRANSPORTATION_MFG_CORP.']
# 		data_test['MAK_NM_TEREX___TEREX_ADVANCE']=data_test['MAK_NM_TEREX_/_TEREX_ADVANCE']
# 	elif i==22:
# 		data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_train['MAK_NM_MERCEDES_BENZ']=data_train['MAK_NM_MERCEDES-BENZ']
# 		data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_train['MAK_NM_WESTERN_STAR/AUTO_CAR']
# 		data_train['MAK_NM_PIERCE_MFG__INC_']=data_train['MAK_NM_PIERCE_MFG._INC.']
# 		data_train['MAK_NM_WHITE_GMC']=data_train['MAK_NM_WHITE/GMC']
# 		data_train['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_train['MAK_NM_TRANSPORTATION_MFG_CORP.']
# 		data_train['MAK_NM_TEREX___TEREX_ADVANCE']=data_train['MAK_NM_TEREX_/_TEREX_ADVANCE']
# 		data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_test['MAK_NM_MERCEDES_BENZ']=data_test['MAK_NM_MERCEDES-BENZ']	
# 		data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_test['MAK_NM_WESTERN_STAR/AUTO_CAR']
# 		data_test['MAK_NM_PIERCE_MFG__INC_']=data_test['MAK_NM_PIERCE_MFG._INC.']
# 		data_test['MAK_NM_WHITE_GMC']=data_test['MAK_NM_WHITE/GMC']
# 		data_test['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_test['MAK_NM_TRANSPORTATION_MFG_CORP.']
# 		data_test['MAK_NM_TEREX___TEREX_ADVANCE']=data_test['MAK_NM_TEREX_/_TEREX_ADVANCE']
# 		data_train['NADA_BODY1_grp1']=(data_train['NADA_BODY1_K']+data_train['NADA_BODY1_H']+data_train['NADA_BODY1_B']+data_train['NADA_BODY1_D'])
# 		data_train['NADA_BODY1_grp2']=(data_train['NADA_BODY1_M']+data_train['NADA_BODY1_G']+data_train['NADA_BODY1_I'])
# 		data_train['NADA_BODY1_grp3']=(data_train['NADA_BODY1_N']+data_train['NADA_BODY1_2'])		
# 		data_train['MAK_NM_grp1']=(data_train['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_train['MAK_NM_MAXIM']+data_train['MAK_NM_VAN_HOOL']+data_train['MAK_NM_EVOBUS']+data_train['MAK_NM_SEAGRAVE_FIRE_APPARATUS']+data_train['MAK_NM_PREVOST']+data_train['MAK_NM_MERCEDES_BENZ']+data_train['MAK_NM_HYUNDAI']+data_train['MAK_NM_TRANSPORTATION_MFG_CORP_']+data_train['MAK_NM_UNIMOG']+data_train['MAK_NM_DUPLEX']+data_train['MAK_NM_COUNTRY_COACH_MOTORHOME']+data_train['MAK_NM_FEDERAL_MOTORS']+data_train['MAK_NM_MOTOR_COACH_INDUSTRIES']+data_train['MAK_NM_GILLIG']+data_train['MAK_NM_DIAMOND_REO']+data_train['MAK_NM_GIANT']+data_train['MAK_NM_LES_AUTOBUS_MCI']+data_train['MAK_NM_SAVAGE']+data_train['MAK_NM_NEW_FLYER']+data_train['MAK_NM_ZELIGSON']+data_train['MAK_NM_PIERCE_MFG__INC_']+data_train['MAK_NM_SPARTAN_MOTORS']+data_train['MAK_NM_FWD_CORPORATION']+data_train['MAK_NM_JOHN_DEERE']+data_train['MAK_NM_CAPACITY_OF_TEXAS']+data_train['MAK_NM_OTTAWA'])
# 		data_train['MAK_NM_grp2']=(data_train['MAK_NM_WINNEBAGO']+data_train['MAK_NM_LAFORZA']+data_train['MAK_NM_THOMAS']+data_train['MAK_NM_MARMON_HERRINGTON']+data_train['MAK_NM_IVECO']+data_train['MAK_NM_SCANIA']+data_train['MAK_NM_TEREX___TEREX_ADVANCE']+data_train['MAK_NM_AMERICAN_LA_FRANCE']+data_train['MAK_NM_AUTOCAR']+data_train['MAK_NM_LODAL']+data_train['MAK_NM_WHITE_GMC'])
# 		data_train['MAK_NM_grp3']=(data_train['MAK_NM_INTERNATIONAL']+data_train['MAK_NM_KENWORTH']+data_train['MAK_NM_ISUZU']+data_train['MAK_NM_MACK']+data_train['MAK_NM_NISSAN_DIESEL']+data_train['MAK_NM_FREIGHTLINER']+data_train['MAK_NM_DODGE'])
# 		data_train['MAK_NM_grp4']=(data_train['MAK_NM_HINO']+data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']+data_train['MAK_NM_EMERGENCY_ONE']+data_train['MAK_NM_CRANE_CARRIER']+data_train['MAK_NM_UTILIMASTER']+data_train['MAK_NM_RAM']+data_train['MAK_NM_AUTOCAR_LLC'])
# 		data_train['MAK_NM_grp5']=(data_train['MAK_NM_WORKHORSE_CUSTOM_CHASSIS']+data_train['MAK_NM_ADVANCE_MIXER']+data_train['MAK_NM_BERING']+data_train['MAK_NM_FORETRAVEL_MOTORHOME']+data_train['MAK_NM_CATERPILLAR'])
# 		data_train['MAK_NM_grp6']=(data_train['MAK_NM_INDIANA_PHOENIX']+data_train['MAK_NM_BLUE_BIRD']+data_train['MAK_NM_HENDRICKSON'])
# 		data_train['ENG_FUEL_CD_grp1']=(data_train['ENG_FUEL_CD_U']+data_train['ENG_FUEL_CD_N'])
# 		data_train['ENG_FUEL_CD_grp2']=(data_train['ENG_FUEL_CD_G']+data_train['ENG_FUEL_CD_P'])
# 		data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_Y']+data_train['PLNT_CD_5']+data_train['PLNT_CD_8'])
# 		data_train['PLNT_CD_grp2']=(data_train['PLNT_CD_U']+data_train['PLNT_CD_1']+data_train['PLNT_CD_V'])
# 		data_train['PLNT_CD_grp3']=(data_train['PLNT_CD_M']+data_train['PLNT_CD_W']+data_train['PLNT_CD_F']+data_train['PLNT_CD_D']+data_train['PLNT_CD_K']+data_train['PLNT_CD_7']+data_train['PLNT_CD_S']+data_train['PLNT_CD_R']+data_train['PLNT_CD_C']+data_train['PLNT_CD_L']+data_train['PLNT_CD_2']+data_train['PLNT_CD_T'])
# 		data_train['PLNT_CD_grp4']=(data_train['PLNT_CD_4']+data_train['PLNT_CD_0']+data_train['PLNT_CD_9'])
# 		data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_120010']+data_train['ENG_MDL_CD_160030']+data_train['ENG_MDL_CD_110030']+data_train['ENG_MDL_CD_180020']+data_train['ENG_MDL_CD_140220']+data_train['ENG_MDL_CD_180010']+data_train['ENG_MDL_CD_090020']+data_train['ENG_MDL_CD_140167']+data_train['ENG_MDL_CD_140173']+data_train['ENG_MDL_CD_140115']+data_train['ENG_MDL_CD_160020']+data_train['ENG_MDL_CD_020105']+data_train['ENG_MDL_CD_195001']+data_train['ENG_MDL_CD_010038']+data_train['ENG_MDL_CD_165185']+data_train['ENG_MDL_CD_020085']+data_train['ENG_MDL_CD_140168']+data_train['ENG_MDL_CD_165225']+data_train['ENG_MDL_CD_140027']+data_train['ENG_MDL_CD_090077']+data_train['ENG_MDL_CD_180052']+data_train['ENG_MDL_CD_165230']+data_train['ENG_MDL_CD_090043']+data_train['ENG_MDL_CD_090045']+data_train['ENG_MDL_CD_165160']+data_train['ENG_MDL_CD_020060']+data_train['ENG_MDL_CD_010050']+data_train['ENG_MDL_CD_020090']+data_train['ENG_MDL_CD_040030']+data_train['ENG_MDL_CD_030090']+data_train['ENG_MDL_CD_140215']+data_train['ENG_MDL_CD_210100']+data_train['ENG_MDL_CD_020100']+data_train['ENG_MDL_CD_220019']+data_train['ENG_MDL_CD_210090']+data_train['ENG_MDL_CD_140055']+data_train['ENG_MDL_CD_010070']+data_train['ENG_MDL_CD_020120']+data_train['ENG_MDL_CD_140090']+data_train['ENG_MDL_CD_140125']+data_train['ENG_MDL_CD_210110']+data_train['ENG_MDL_CD_140050']+data_train['ENG_MDL_CD_090051']+data_train['ENG_MDL_CD_140234']+data_train['ENG_MDL_CD_245010']+data_train['ENG_MDL_CD_030060']+data_train['ENG_MDL_CD_210040']+data_train['ENG_MDL_CD_210070']+data_train['ENG_MDL_CD_210050']+data_train['ENG_MDL_CD_210060']+data_train['ENG_MDL_CD_210080']+data_train['ENG_MDL_CD_210020']+data_train['ENG_MDL_CD_140025']+data_train['ENG_MDL_CD_210030']+data_train['ENG_MDL_CD_020026']+data_train['ENG_MDL_CD_110070']+data_train['ENG_MDL_CD_180040']+data_train['ENG_MDL_CD_140045']+data_train['ENG_MDL_CD_140040']+data_train['ENG_MDL_CD_140020']+data_train['ENG_MDL_CD_040020']+data_train['ENG_MDL_CD_160060']+data_train['ENG_MDL_CD_030020']+data_train['ENG_MDL_CD_090040']+data_train['ENG_MDL_CD_080012']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_050020']+data_train['ENG_MDL_CD_165007']+data_train['ENG_MDL_CD_040010'])
# 		data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_150010']+data_train['ENG_MDL_CD_150020'])
# 		data_train['ENG_MDL_CD_grp3']=(data_train['ENG_MDL_CD_150060']+data_train['ENG_MDL_CD_210165']+data_train['ENG_MDL_CD_090030']+data_train['ENG_MDL_CD_165030']+data_train['ENG_MDL_CD_180030']+data_train['ENG_MDL_CD_200010']+data_train['ENG_MDL_CD_140195']+data_train['ENG_MDL_CD_140105']+data_train['ENG_MDL_CD_165140']+data_train['ENG_MDL_CD_140210']+data_train['ENG_MDL_CD_080080']+data_train['ENG_MDL_CD_160012']+data_train['ENG_MDL_CD_140065']+data_train['ENG_MDL_CD_090011']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_140205']+data_train['ENG_MDL_CD_030050']+data_train['ENG_MDL_CD_030047']+data_train['ENG_MDL_CD_165090']+data_train['ENG_MDL_CD_140095']+data_train['ENG_MDL_CD_140150']+data_train['ENG_MDL_CD_090070']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_020072']+data_train['ENG_MDL_CD_165010'])
# 		data_train['ENG_MDL_CD_grp4']=(data_train['ENG_MDL_CD_150003']+data_train['ENG_MDL_CD_150150']+data_train['ENG_MDL_CD_150005']+data_train['ENG_MDL_CD_150007'])

# 		data_test['NADA_BODY1_grp1']=(data_test['NADA_BODY1_K']+data_test['NADA_BODY1_H']+data_test['NADA_BODY1_B']+data_test['NADA_BODY1_D'])
# 		data_test['NADA_BODY1_grp2']=(data_test['NADA_BODY1_M']+data_test['NADA_BODY1_G']+data_test['NADA_BODY1_I'])
# 		data_test['NADA_BODY1_grp3']=(data_test['NADA_BODY1_N']+data_test['NADA_BODY1_2'])		
# 		data_test['MAK_NM_grp1']=(data_test['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_test['MAK_NM_MAXIM']+data_test['MAK_NM_VAN_HOOL']+data_test['MAK_NM_EVOBUS']+data_test['MAK_NM_SEAGRAVE_FIRE_APPARATUS']+data_test['MAK_NM_PREVOST']+data_test['MAK_NM_MERCEDES_BENZ']+data_test['MAK_NM_HYUNDAI']+data_test['MAK_NM_TRANSPORTATION_MFG_CORP_']+data_test['MAK_NM_UNIMOG']+data_test['MAK_NM_DUPLEX']+data_test['MAK_NM_COUNTRY_COACH_MOTORHOME']+data_test['MAK_NM_FEDERAL_MOTORS']+data_test['MAK_NM_MOTOR_COACH_INDUSTRIES']+data_test['MAK_NM_GILLIG']+data_test['MAK_NM_DIAMOND_REO']+data_test['MAK_NM_GIANT']+data_test['MAK_NM_LES_AUTOBUS_MCI']+data_test['MAK_NM_SAVAGE']+data_test['MAK_NM_NEW_FLYER']+data_test['MAK_NM_ZELIGSON']+data_test['MAK_NM_PIERCE_MFG__INC_']+data_test['MAK_NM_SPARTAN_MOTORS']+data_test['MAK_NM_FWD_CORPORATION']+data_test['MAK_NM_JOHN_DEERE']+data_test['MAK_NM_CAPACITY_OF_TEXAS']+data_test['MAK_NM_OTTAWA'])
# 		data_test['MAK_NM_grp2']=(data_test['MAK_NM_WINNEBAGO']+data_test['MAK_NM_LAFORZA']+data_test['MAK_NM_THOMAS']+data_test['MAK_NM_MARMON_HERRINGTON']+data_test['MAK_NM_IVECO']+data_test['MAK_NM_SCANIA']+data_test['MAK_NM_TEREX___TEREX_ADVANCE']+data_test['MAK_NM_AMERICAN_LA_FRANCE']+data_test['MAK_NM_AUTOCAR']+data_test['MAK_NM_LODAL']+data_test['MAK_NM_WHITE_GMC'])
# 		data_test['MAK_NM_grp3']=(data_test['MAK_NM_INTERNATIONAL']+data_test['MAK_NM_KENWORTH']+data_test['MAK_NM_ISUZU']+data_test['MAK_NM_MACK']+data_test['MAK_NM_NISSAN_DIESEL']+data_test['MAK_NM_FREIGHTLINER']+data_test['MAK_NM_DODGE'])
# 		data_test['MAK_NM_grp4']=(data_test['MAK_NM_HINO']+data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']+data_test['MAK_NM_EMERGENCY_ONE']+data_test['MAK_NM_CRANE_CARRIER']+data_test['MAK_NM_UTILIMASTER']+data_test['MAK_NM_RAM']+data_test['MAK_NM_AUTOCAR_LLC'])
# 		data_test['MAK_NM_grp5']=(data_test['MAK_NM_WORKHORSE_CUSTOM_CHASSIS']+data_test['MAK_NM_ADVANCE_MIXER']+data_test['MAK_NM_BERING']+data_test['MAK_NM_FORETRAVEL_MOTORHOME']+data_test['MAK_NM_CATERPILLAR'])
# 		data_test['MAK_NM_grp6']=(data_test['MAK_NM_INDIANA_PHOENIX']+data_test['MAK_NM_BLUE_BIRD']+data_test['MAK_NM_HENDRICKSON'])
# 		data_test['ENG_FUEL_CD_grp1']=(data_test['ENG_FUEL_CD_U']+data_test['ENG_FUEL_CD_N'])
# 		data_test['ENG_FUEL_CD_grp2']=(data_test['ENG_FUEL_CD_G']+data_test['ENG_FUEL_CD_P'])
# 		data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_Y']+data_test['PLNT_CD_5']+data_test['PLNT_CD_8'])
# 		data_test['PLNT_CD_grp2']=(data_test['PLNT_CD_U']+data_test['PLNT_CD_1']+data_test['PLNT_CD_V'])
# 		data_test['PLNT_CD_grp3']=(data_test['PLNT_CD_M']+data_test['PLNT_CD_W']+data_test['PLNT_CD_F']+data_test['PLNT_CD_D']+data_test['PLNT_CD_K']+data_test['PLNT_CD_7']+data_test['PLNT_CD_S']+data_test['PLNT_CD_R']+data_test['PLNT_CD_C']+data_test['PLNT_CD_L']+data_test['PLNT_CD_2']+data_test['PLNT_CD_T'])
# 		data_test['PLNT_CD_grp4']=(data_test['PLNT_CD_4']+data_test['PLNT_CD_0']+data_test['PLNT_CD_9'])
# 		data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_120010']+data_test['ENG_MDL_CD_160030']+data_test['ENG_MDL_CD_110030']+data_test['ENG_MDL_CD_180020']+data_test['ENG_MDL_CD_140220']+data_test['ENG_MDL_CD_180010']+data_test['ENG_MDL_CD_090020']+data_test['ENG_MDL_CD_140167']+data_test['ENG_MDL_CD_140173']+data_test['ENG_MDL_CD_140115']+data_test['ENG_MDL_CD_160020']+data_test['ENG_MDL_CD_020105']+data_test['ENG_MDL_CD_195001']+data_test['ENG_MDL_CD_010038']+data_test['ENG_MDL_CD_165185']+data_test['ENG_MDL_CD_020085']+data_test['ENG_MDL_CD_140168']+data_test['ENG_MDL_CD_165225']+data_test['ENG_MDL_CD_140027']+data_test['ENG_MDL_CD_090077']+data_test['ENG_MDL_CD_180052']+data_test['ENG_MDL_CD_165230']+data_test['ENG_MDL_CD_090043']+data_test['ENG_MDL_CD_090045']+data_test['ENG_MDL_CD_165160']+data_test['ENG_MDL_CD_020060']+data_test['ENG_MDL_CD_010050']+data_test['ENG_MDL_CD_020090']+data_test['ENG_MDL_CD_040030']+data_test['ENG_MDL_CD_030090']+data_test['ENG_MDL_CD_140215']+data_test['ENG_MDL_CD_210100']+data_test['ENG_MDL_CD_020100']+data_test['ENG_MDL_CD_220019']+data_test['ENG_MDL_CD_210090']+data_test['ENG_MDL_CD_140055']+data_test['ENG_MDL_CD_010070']+data_test['ENG_MDL_CD_020120']+data_test['ENG_MDL_CD_140090']+data_test['ENG_MDL_CD_140125']+data_test['ENG_MDL_CD_210110']+data_test['ENG_MDL_CD_140050']+data_test['ENG_MDL_CD_090051']+data_test['ENG_MDL_CD_140234']+data_test['ENG_MDL_CD_245010']+data_test['ENG_MDL_CD_030060']+data_test['ENG_MDL_CD_210040']+data_test['ENG_MDL_CD_210070']+data_test['ENG_MDL_CD_210050']+data_test['ENG_MDL_CD_210060']+data_test['ENG_MDL_CD_210080']+data_test['ENG_MDL_CD_210020']+data_test['ENG_MDL_CD_140025']+data_test['ENG_MDL_CD_210030']+data_test['ENG_MDL_CD_020026']+data_test['ENG_MDL_CD_110070']+data_test['ENG_MDL_CD_180040']+data_test['ENG_MDL_CD_140045']+data_test['ENG_MDL_CD_140040']+data_test['ENG_MDL_CD_140020']+data_test['ENG_MDL_CD_040020']+data_test['ENG_MDL_CD_160060']+data_test['ENG_MDL_CD_030020']+data_test['ENG_MDL_CD_090040']+data_test['ENG_MDL_CD_080012']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_050020']+data_test['ENG_MDL_CD_165007']+data_test['ENG_MDL_CD_040010'])
# 		data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_150010']+data_test['ENG_MDL_CD_150020'])
# 		data_test['ENG_MDL_CD_grp3']=(data_test['ENG_MDL_CD_150060']+data_test['ENG_MDL_CD_210165']+data_test['ENG_MDL_CD_090030']+data_test['ENG_MDL_CD_165030']+data_test['ENG_MDL_CD_180030']+data_test['ENG_MDL_CD_200010']+data_test['ENG_MDL_CD_140195']+data_test['ENG_MDL_CD_140105']+data_test['ENG_MDL_CD_165140']+data_test['ENG_MDL_CD_140210']+data_test['ENG_MDL_CD_080080']+data_test['ENG_MDL_CD_160012']+data_test['ENG_MDL_CD_140065']+data_test['ENG_MDL_CD_090011']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_140205']+data_test['ENG_MDL_CD_030050']+data_test['ENG_MDL_CD_030047']+data_test['ENG_MDL_CD_165090']+data_test['ENG_MDL_CD_140095']+data_test['ENG_MDL_CD_140150']+data_test['ENG_MDL_CD_090070']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_020072']+data_test['ENG_MDL_CD_165010'])
# 		data_test['ENG_MDL_CD_grp4']=(data_test['ENG_MDL_CD_150003']+data_test['ENG_MDL_CD_150150']+data_test['ENG_MDL_CD_150005']+data_test['ENG_MDL_CD_150007'])
# 	elif i==23:
# 		data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_train['MAK_NM_MERCEDES_BENZ']=data_train['MAK_NM_MERCEDES-BENZ']
# 		data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_train['MAK_NM_WESTERN_STAR/AUTO_CAR']
# 		data_train['MAK_NM_PIERCE_MFG__INC_']=data_train['MAK_NM_PIERCE_MFG._INC.']
# 		data_train['MAK_NM_WHITE_GMC']=data_train['MAK_NM_WHITE/GMC']
# 		data_train['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_train['MAK_NM_TRANSPORTATION_MFG_CORP.']
# 		data_train['MAK_NM_TEREX___TEREX_ADVANCE']=data_train['MAK_NM_TEREX_/_TEREX_ADVANCE']
# 		data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_test['MAK_NM_MERCEDES_BENZ']=data_test['MAK_NM_MERCEDES-BENZ']	
# 		data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_test['MAK_NM_WESTERN_STAR/AUTO_CAR']
# 		data_test['MAK_NM_PIERCE_MFG__INC_']=data_test['MAK_NM_PIERCE_MFG._INC.']
# 		data_test['MAK_NM_WHITE_GMC']=data_test['MAK_NM_WHITE/GMC']
# 		data_test['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_test['MAK_NM_TRANSPORTATION_MFG_CORP.']
# 		data_test['MAK_NM_TEREX___TEREX_ADVANCE']=data_test['MAK_NM_TEREX_/_TEREX_ADVANCE']
# 		data_train['NADA_BODY1_grp1']=(data_train['NADA_BODY1_K']+data_train['NADA_BODY1_H']+data_train['NADA_BODY1_B']+data_train['NADA_BODY1_D'])
# 		data_train['NADA_BODY1_grp2']=(data_train['NADA_BODY1_M']+data_train['NADA_BODY1_G']+data_train['NADA_BODY1_I'])
# 		data_train['NADA_BODY1_grp3']=(data_train['NADA_BODY1_N']+data_train['NADA_BODY1_2'])		
# 		data_train['MAK_NM_grp1']=(data_train['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_train['MAK_NM_MAXIM']+data_train['MAK_NM_VAN_HOOL']+data_train['MAK_NM_EVOBUS']+data_train['MAK_NM_SEAGRAVE_FIRE_APPARATUS']+data_train['MAK_NM_PREVOST']+data_train['MAK_NM_MERCEDES_BENZ']+data_train['MAK_NM_HYUNDAI']+data_train['MAK_NM_TRANSPORTATION_MFG_CORP_']+data_train['MAK_NM_UNIMOG']+data_train['MAK_NM_DUPLEX']+data_train['MAK_NM_COUNTRY_COACH_MOTORHOME']+data_train['MAK_NM_FEDERAL_MOTORS']+data_train['MAK_NM_MOTOR_COACH_INDUSTRIES']+data_train['MAK_NM_GILLIG']+data_train['MAK_NM_DIAMOND_REO']+data_train['MAK_NM_GIANT']+data_train['MAK_NM_LES_AUTOBUS_MCI']+data_train['MAK_NM_SAVAGE']+data_train['MAK_NM_NEW_FLYER']+data_train['MAK_NM_ZELIGSON']+data_train['MAK_NM_PIERCE_MFG__INC_']+data_train['MAK_NM_SPARTAN_MOTORS']+data_train['MAK_NM_FWD_CORPORATION']+data_train['MAK_NM_JOHN_DEERE']+data_train['MAK_NM_CAPACITY_OF_TEXAS']+data_train['MAK_NM_OTTAWA'])
# 		data_train['MAK_NM_grp2']=(data_train['MAK_NM_WINNEBAGO']+data_train['MAK_NM_LAFORZA']+data_train['MAK_NM_THOMAS']+data_train['MAK_NM_MARMON_HERRINGTON']+data_train['MAK_NM_IVECO']+data_train['MAK_NM_SCANIA']+data_train['MAK_NM_TEREX___TEREX_ADVANCE']+data_train['MAK_NM_AMERICAN_LA_FRANCE']+data_train['MAK_NM_AUTOCAR']+data_train['MAK_NM_LODAL']+data_train['MAK_NM_WHITE_GMC'])
# 		data_train['MAK_NM_grp3']=(data_train['MAK_NM_INTERNATIONAL']+data_train['MAK_NM_KENWORTH']+data_train['MAK_NM_ISUZU']+data_train['MAK_NM_MACK']+data_train['MAK_NM_NISSAN_DIESEL']+data_train['MAK_NM_FREIGHTLINER']+data_train['MAK_NM_DODGE'])
# 		data_train['MAK_NM_grp4']=(data_train['MAK_NM_HINO']+data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']+data_train['MAK_NM_EMERGENCY_ONE']+data_train['MAK_NM_CRANE_CARRIER']+data_train['MAK_NM_UTILIMASTER']+data_train['MAK_NM_RAM']+data_train['MAK_NM_AUTOCAR_LLC'])
# 		data_train['MAK_NM_grp5']=(data_train['MAK_NM_WORKHORSE_CUSTOM_CHASSIS']+data_train['MAK_NM_ADVANCE_MIXER']+data_train['MAK_NM_BERING']+data_train['MAK_NM_FORETRAVEL_MOTORHOME']+data_train['MAK_NM_CATERPILLAR'])
# 		data_train['MAK_NM_grp6']=(data_train['MAK_NM_INDIANA_PHOENIX']+data_train['MAK_NM_BLUE_BIRD']+data_train['MAK_NM_HENDRICKSON'])
# 		data_train['ENG_FUEL_CD_grp1']=(data_train['ENG_FUEL_CD_U']+data_train['ENG_FUEL_CD_N'])
# 		data_train['ENG_FUEL_CD_grp2']=(data_train['ENG_FUEL_CD_G']+data_train['ENG_FUEL_CD_P'])
# 		data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_Y']+data_train['PLNT_CD_5']+data_train['PLNT_CD_8'])
# 		data_train['PLNT_CD_grp2']=(data_train['PLNT_CD_U']+data_train['PLNT_CD_1']+data_train['PLNT_CD_V'])
# 		data_train['PLNT_CD_grp3']=(data_train['PLNT_CD_M']+data_train['PLNT_CD_W']+data_train['PLNT_CD_F']+data_train['PLNT_CD_D']+data_train['PLNT_CD_K']+data_train['PLNT_CD_7']+data_train['PLNT_CD_S']+data_train['PLNT_CD_R']+data_train['PLNT_CD_C']+data_train['PLNT_CD_L']+data_train['PLNT_CD_2']+data_train['PLNT_CD_T'])
# 		data_train['PLNT_CD_grp4']=(data_train['PLNT_CD_4']+data_train['PLNT_CD_0']+data_train['PLNT_CD_9'])
# 		data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_120010']+data_train['ENG_MDL_CD_160030']+data_train['ENG_MDL_CD_110030']+data_train['ENG_MDL_CD_180020']+data_train['ENG_MDL_CD_140220']+data_train['ENG_MDL_CD_180010']+data_train['ENG_MDL_CD_090020']+data_train['ENG_MDL_CD_140167']+data_train['ENG_MDL_CD_140173']+data_train['ENG_MDL_CD_140115']+data_train['ENG_MDL_CD_160020']+data_train['ENG_MDL_CD_020105']+data_train['ENG_MDL_CD_195001']+data_train['ENG_MDL_CD_010038']+data_train['ENG_MDL_CD_165185']+data_train['ENG_MDL_CD_020085']+data_train['ENG_MDL_CD_140168']+data_train['ENG_MDL_CD_165225']+data_train['ENG_MDL_CD_140027']+data_train['ENG_MDL_CD_090077']+data_train['ENG_MDL_CD_180052']+data_train['ENG_MDL_CD_165230']+data_train['ENG_MDL_CD_090043']+data_train['ENG_MDL_CD_090045']+data_train['ENG_MDL_CD_165160']+data_train['ENG_MDL_CD_020060']+data_train['ENG_MDL_CD_010050']+data_train['ENG_MDL_CD_020090']+data_train['ENG_MDL_CD_040030']+data_train['ENG_MDL_CD_030090']+data_train['ENG_MDL_CD_140215']+data_train['ENG_MDL_CD_210100']+data_train['ENG_MDL_CD_020100']+data_train['ENG_MDL_CD_220019']+data_train['ENG_MDL_CD_210090']+data_train['ENG_MDL_CD_140055']+data_train['ENG_MDL_CD_010070']+data_train['ENG_MDL_CD_020120']+data_train['ENG_MDL_CD_140090']+data_train['ENG_MDL_CD_140125']+data_train['ENG_MDL_CD_210110']+data_train['ENG_MDL_CD_140050']+data_train['ENG_MDL_CD_090051']+data_train['ENG_MDL_CD_140234']+data_train['ENG_MDL_CD_245010']+data_train['ENG_MDL_CD_030060']+data_train['ENG_MDL_CD_210040']+data_train['ENG_MDL_CD_210070']+data_train['ENG_MDL_CD_210050']+data_train['ENG_MDL_CD_210060']+data_train['ENG_MDL_CD_210080']+data_train['ENG_MDL_CD_210020']+data_train['ENG_MDL_CD_140025']+data_train['ENG_MDL_CD_210030']+data_train['ENG_MDL_CD_020026']+data_train['ENG_MDL_CD_110070']+data_train['ENG_MDL_CD_180040']+data_train['ENG_MDL_CD_140045']+data_train['ENG_MDL_CD_140040']+data_train['ENG_MDL_CD_140020']+data_train['ENG_MDL_CD_040020']+data_train['ENG_MDL_CD_160060']+data_train['ENG_MDL_CD_030020']+data_train['ENG_MDL_CD_090040']+data_train['ENG_MDL_CD_080012']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_050020']+data_train['ENG_MDL_CD_165007']+data_train['ENG_MDL_CD_040010'])
# 		data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_150010']+data_train['ENG_MDL_CD_150020'])
# 		data_train['ENG_MDL_CD_grp3']=(data_train['ENG_MDL_CD_150060']+data_train['ENG_MDL_CD_210165']+data_train['ENG_MDL_CD_090030']+data_train['ENG_MDL_CD_165030']+data_train['ENG_MDL_CD_180030']+data_train['ENG_MDL_CD_200010']+data_train['ENG_MDL_CD_140195']+data_train['ENG_MDL_CD_140105']+data_train['ENG_MDL_CD_165140']+data_train['ENG_MDL_CD_140210']+data_train['ENG_MDL_CD_080080']+data_train['ENG_MDL_CD_160012']+data_train['ENG_MDL_CD_140065']+data_train['ENG_MDL_CD_090011']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_140205']+data_train['ENG_MDL_CD_030050']+data_train['ENG_MDL_CD_030047']+data_train['ENG_MDL_CD_165090']+data_train['ENG_MDL_CD_140095']+data_train['ENG_MDL_CD_140150']+data_train['ENG_MDL_CD_090070']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_020072']+data_train['ENG_MDL_CD_165010'])
# 		data_train['ENG_MDL_CD_grp4']=(data_train['ENG_MDL_CD_150003']+data_train['ENG_MDL_CD_150150']+data_train['ENG_MDL_CD_150005']+data_train['ENG_MDL_CD_150007'])

# 		data_test['NADA_BODY1_grp1']=(data_test['NADA_BODY1_K']+data_test['NADA_BODY1_H']+data_test['NADA_BODY1_B']+data_test['NADA_BODY1_D'])
# 		data_test['NADA_BODY1_grp2']=(data_test['NADA_BODY1_M']+data_test['NADA_BODY1_G']+data_test['NADA_BODY1_I'])
# 		data_test['NADA_BODY1_grp3']=(data_test['NADA_BODY1_N']+data_test['NADA_BODY1_2'])		
# 		data_test['MAK_NM_grp1']=(data_test['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_test['MAK_NM_MAXIM']+data_test['MAK_NM_VAN_HOOL']+data_test['MAK_NM_EVOBUS']+data_test['MAK_NM_SEAGRAVE_FIRE_APPARATUS']+data_test['MAK_NM_PREVOST']+data_test['MAK_NM_MERCEDES_BENZ']+data_test['MAK_NM_HYUNDAI']+data_test['MAK_NM_TRANSPORTATION_MFG_CORP_']+data_test['MAK_NM_UNIMOG']+data_test['MAK_NM_DUPLEX']+data_test['MAK_NM_COUNTRY_COACH_MOTORHOME']+data_test['MAK_NM_FEDERAL_MOTORS']+data_test['MAK_NM_MOTOR_COACH_INDUSTRIES']+data_test['MAK_NM_GILLIG']+data_test['MAK_NM_DIAMOND_REO']+data_test['MAK_NM_GIANT']+data_test['MAK_NM_LES_AUTOBUS_MCI']+data_test['MAK_NM_SAVAGE']+data_test['MAK_NM_NEW_FLYER']+data_test['MAK_NM_ZELIGSON']+data_test['MAK_NM_PIERCE_MFG__INC_']+data_test['MAK_NM_SPARTAN_MOTORS']+data_test['MAK_NM_FWD_CORPORATION']+data_test['MAK_NM_JOHN_DEERE']+data_test['MAK_NM_CAPACITY_OF_TEXAS']+data_test['MAK_NM_OTTAWA'])
# 		data_test['MAK_NM_grp2']=(data_test['MAK_NM_WINNEBAGO']+data_test['MAK_NM_LAFORZA']+data_test['MAK_NM_THOMAS']+data_test['MAK_NM_MARMON_HERRINGTON']+data_test['MAK_NM_IVECO']+data_test['MAK_NM_SCANIA']+data_test['MAK_NM_TEREX___TEREX_ADVANCE']+data_test['MAK_NM_AMERICAN_LA_FRANCE']+data_test['MAK_NM_AUTOCAR']+data_test['MAK_NM_LODAL']+data_test['MAK_NM_WHITE_GMC'])
# 		data_test['MAK_NM_grp3']=(data_test['MAK_NM_INTERNATIONAL']+data_test['MAK_NM_KENWORTH']+data_test['MAK_NM_ISUZU']+data_test['MAK_NM_MACK']+data_test['MAK_NM_NISSAN_DIESEL']+data_test['MAK_NM_FREIGHTLINER']+data_test['MAK_NM_DODGE'])
# 		data_test['MAK_NM_grp4']=(data_test['MAK_NM_HINO']+data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']+data_test['MAK_NM_EMERGENCY_ONE']+data_test['MAK_NM_CRANE_CARRIER']+data_test['MAK_NM_UTILIMASTER']+data_test['MAK_NM_RAM']+data_test['MAK_NM_AUTOCAR_LLC'])
# 		data_test['MAK_NM_grp5']=(data_test['MAK_NM_WORKHORSE_CUSTOM_CHASSIS']+data_test['MAK_NM_ADVANCE_MIXER']+data_test['MAK_NM_BERING']+data_test['MAK_NM_FORETRAVEL_MOTORHOME']+data_test['MAK_NM_CATERPILLAR'])
# 		data_test['MAK_NM_grp6']=(data_test['MAK_NM_INDIANA_PHOENIX']+data_test['MAK_NM_BLUE_BIRD']+data_test['MAK_NM_HENDRICKSON'])
# 		data_test['ENG_FUEL_CD_grp1']=(data_test['ENG_FUEL_CD_U']+data_test['ENG_FUEL_CD_N'])
# 		data_test['ENG_FUEL_CD_grp2']=(data_test['ENG_FUEL_CD_G']+data_test['ENG_FUEL_CD_P'])
# 		data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_Y']+data_test['PLNT_CD_5']+data_test['PLNT_CD_8'])
# 		data_test['PLNT_CD_grp2']=(data_test['PLNT_CD_U']+data_test['PLNT_CD_1']+data_test['PLNT_CD_V'])
# 		data_test['PLNT_CD_grp3']=(data_test['PLNT_CD_M']+data_test['PLNT_CD_W']+data_test['PLNT_CD_F']+data_test['PLNT_CD_D']+data_test['PLNT_CD_K']+data_test['PLNT_CD_7']+data_test['PLNT_CD_S']+data_test['PLNT_CD_R']+data_test['PLNT_CD_C']+data_test['PLNT_CD_L']+data_test['PLNT_CD_2']+data_test['PLNT_CD_T'])
# 		data_test['PLNT_CD_grp4']=(data_test['PLNT_CD_4']+data_test['PLNT_CD_0']+data_test['PLNT_CD_9'])
# 		data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_120010']+data_test['ENG_MDL_CD_160030']+data_test['ENG_MDL_CD_110030']+data_test['ENG_MDL_CD_180020']+data_test['ENG_MDL_CD_140220']+data_test['ENG_MDL_CD_180010']+data_test['ENG_MDL_CD_090020']+data_test['ENG_MDL_CD_140167']+data_test['ENG_MDL_CD_140173']+data_test['ENG_MDL_CD_140115']+data_test['ENG_MDL_CD_160020']+data_test['ENG_MDL_CD_020105']+data_test['ENG_MDL_CD_195001']+data_test['ENG_MDL_CD_010038']+data_test['ENG_MDL_CD_165185']+data_test['ENG_MDL_CD_020085']+data_test['ENG_MDL_CD_140168']+data_test['ENG_MDL_CD_165225']+data_test['ENG_MDL_CD_140027']+data_test['ENG_MDL_CD_090077']+data_test['ENG_MDL_CD_180052']+data_test['ENG_MDL_CD_165230']+data_test['ENG_MDL_CD_090043']+data_test['ENG_MDL_CD_090045']+data_test['ENG_MDL_CD_165160']+data_test['ENG_MDL_CD_020060']+data_test['ENG_MDL_CD_010050']+data_test['ENG_MDL_CD_020090']+data_test['ENG_MDL_CD_040030']+data_test['ENG_MDL_CD_030090']+data_test['ENG_MDL_CD_140215']+data_test['ENG_MDL_CD_210100']+data_test['ENG_MDL_CD_020100']+data_test['ENG_MDL_CD_220019']+data_test['ENG_MDL_CD_210090']+data_test['ENG_MDL_CD_140055']+data_test['ENG_MDL_CD_010070']+data_test['ENG_MDL_CD_020120']+data_test['ENG_MDL_CD_140090']+data_test['ENG_MDL_CD_140125']+data_test['ENG_MDL_CD_210110']+data_test['ENG_MDL_CD_140050']+data_test['ENG_MDL_CD_090051']+data_test['ENG_MDL_CD_140234']+data_test['ENG_MDL_CD_245010']+data_test['ENG_MDL_CD_030060']+data_test['ENG_MDL_CD_210040']+data_test['ENG_MDL_CD_210070']+data_test['ENG_MDL_CD_210050']+data_test['ENG_MDL_CD_210060']+data_test['ENG_MDL_CD_210080']+data_test['ENG_MDL_CD_210020']+data_test['ENG_MDL_CD_140025']+data_test['ENG_MDL_CD_210030']+data_test['ENG_MDL_CD_020026']+data_test['ENG_MDL_CD_110070']+data_test['ENG_MDL_CD_180040']+data_test['ENG_MDL_CD_140045']+data_test['ENG_MDL_CD_140040']+data_test['ENG_MDL_CD_140020']+data_test['ENG_MDL_CD_040020']+data_test['ENG_MDL_CD_160060']+data_test['ENG_MDL_CD_030020']+data_test['ENG_MDL_CD_090040']+data_test['ENG_MDL_CD_080012']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_050020']+data_test['ENG_MDL_CD_165007']+data_test['ENG_MDL_CD_040010'])
# 		data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_150010']+data_test['ENG_MDL_CD_150020'])
# 		data_test['ENG_MDL_CD_grp3']=(data_test['ENG_MDL_CD_150060']+data_test['ENG_MDL_CD_210165']+data_test['ENG_MDL_CD_090030']+data_test['ENG_MDL_CD_165030']+data_test['ENG_MDL_CD_180030']+data_test['ENG_MDL_CD_200010']+data_test['ENG_MDL_CD_140195']+data_test['ENG_MDL_CD_140105']+data_test['ENG_MDL_CD_165140']+data_test['ENG_MDL_CD_140210']+data_test['ENG_MDL_CD_080080']+data_test['ENG_MDL_CD_160012']+data_test['ENG_MDL_CD_140065']+data_test['ENG_MDL_CD_090011']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_140205']+data_test['ENG_MDL_CD_030050']+data_test['ENG_MDL_CD_030047']+data_test['ENG_MDL_CD_165090']+data_test['ENG_MDL_CD_140095']+data_test['ENG_MDL_CD_140150']+data_test['ENG_MDL_CD_090070']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_020072']+data_test['ENG_MDL_CD_165010'])
# 		data_test['ENG_MDL_CD_grp4']=(data_test['ENG_MDL_CD_150003']+data_test['ENG_MDL_CD_150150']+data_test['ENG_MDL_CD_150005']+data_test['ENG_MDL_CD_150007'])
# 	elif i==24:
# 		data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_train['MAK_NM_MERCEDES_BENZ']=data_train['MAK_NM_MERCEDES-BENZ']
# 		data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_train['MAK_NM_WESTERN_STAR/AUTO_CAR']
# 		data_train['MAK_NM_PIERCE_MFG__INC_']=data_train['MAK_NM_PIERCE_MFG._INC.']
# 		data_train['MAK_NM_WHITE_GMC']=data_train['MAK_NM_WHITE/GMC']
# 		data_train['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_train['MAK_NM_TRANSPORTATION_MFG_CORP.']
# 		data_train['MAK_NM_TEREX___TEREX_ADVANCE']=data_train['MAK_NM_TEREX_/_TEREX_ADVANCE']
# 		data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_test['MAK_NM_MERCEDES_BENZ']=data_test['MAK_NM_MERCEDES-BENZ']	
# 		data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_test['MAK_NM_WESTERN_STAR/AUTO_CAR']
# 		data_test['MAK_NM_PIERCE_MFG__INC_']=data_test['MAK_NM_PIERCE_MFG._INC.']
# 		data_test['MAK_NM_WHITE_GMC']=data_test['MAK_NM_WHITE/GMC']
# 		data_test['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_test['MAK_NM_TRANSPORTATION_MFG_CORP.']
# 		data_test['MAK_NM_TEREX___TEREX_ADVANCE']=data_test['MAK_NM_TEREX_/_TEREX_ADVANCE']
# 		data_train['NADA_BODY1_grp1']=(data_train['NADA_BODY1_K']+data_train['NADA_BODY1_H']+data_train['NADA_BODY1_B']+data_train['NADA_BODY1_D'])
# 		data_train['NADA_BODY1_grp2']=(data_train['NADA_BODY1_M']+data_train['NADA_BODY1_G']+data_train['NADA_BODY1_I'])
# 		data_train['NADA_BODY1_grp3']=(data_train['NADA_BODY1_N']+data_train['NADA_BODY1_2'])		
# 		data_train['MAK_NM_grp1']=(data_train['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_train['MAK_NM_MAXIM']+data_train['MAK_NM_VAN_HOOL']+data_train['MAK_NM_EVOBUS']+data_train['MAK_NM_SEAGRAVE_FIRE_APPARATUS']+data_train['MAK_NM_PREVOST']+data_train['MAK_NM_MERCEDES_BENZ']+data_train['MAK_NM_HYUNDAI']+data_train['MAK_NM_TRANSPORTATION_MFG_CORP_']+data_train['MAK_NM_UNIMOG']+data_train['MAK_NM_DUPLEX']+data_train['MAK_NM_COUNTRY_COACH_MOTORHOME']+data_train['MAK_NM_FEDERAL_MOTORS']+data_train['MAK_NM_MOTOR_COACH_INDUSTRIES']+data_train['MAK_NM_GILLIG']+data_train['MAK_NM_DIAMOND_REO']+data_train['MAK_NM_GIANT']+data_train['MAK_NM_LES_AUTOBUS_MCI']+data_train['MAK_NM_SAVAGE']+data_train['MAK_NM_NEW_FLYER']+data_train['MAK_NM_ZELIGSON']+data_train['MAK_NM_PIERCE_MFG__INC_']+data_train['MAK_NM_SPARTAN_MOTORS']+data_train['MAK_NM_FWD_CORPORATION']+data_train['MAK_NM_JOHN_DEERE']+data_train['MAK_NM_CAPACITY_OF_TEXAS']+data_train['MAK_NM_OTTAWA'])
# 		data_train['MAK_NM_grp2']=(data_train['MAK_NM_WINNEBAGO']+data_train['MAK_NM_LAFORZA']+data_train['MAK_NM_THOMAS']+data_train['MAK_NM_MARMON_HERRINGTON']+data_train['MAK_NM_IVECO']+data_train['MAK_NM_SCANIA']+data_train['MAK_NM_TEREX___TEREX_ADVANCE']+data_train['MAK_NM_AMERICAN_LA_FRANCE']+data_train['MAK_NM_AUTOCAR']+data_train['MAK_NM_LODAL']+data_train['MAK_NM_WHITE_GMC'])
# 		data_train['MAK_NM_grp3']=(data_train['MAK_NM_INTERNATIONAL']+data_train['MAK_NM_KENWORTH']+data_train['MAK_NM_ISUZU']+data_train['MAK_NM_MACK']+data_train['MAK_NM_NISSAN_DIESEL']+data_train['MAK_NM_FREIGHTLINER']+data_train['MAK_NM_DODGE'])
# 		data_train['MAK_NM_grp4']=(data_train['MAK_NM_HINO']+data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']+data_train['MAK_NM_EMERGENCY_ONE']+data_train['MAK_NM_CRANE_CARRIER']+data_train['MAK_NM_UTILIMASTER']+data_train['MAK_NM_RAM']+data_train['MAK_NM_AUTOCAR_LLC'])
# 		data_train['MAK_NM_grp5']=(data_train['MAK_NM_WORKHORSE_CUSTOM_CHASSIS']+data_train['MAK_NM_ADVANCE_MIXER']+data_train['MAK_NM_BERING']+data_train['MAK_NM_FORETRAVEL_MOTORHOME']+data_train['MAK_NM_CATERPILLAR'])
# 		data_train['MAK_NM_grp6']=(data_train['MAK_NM_INDIANA_PHOENIX']+data_train['MAK_NM_BLUE_BIRD']+data_train['MAK_NM_HENDRICKSON'])
# 		data_train['ENG_FUEL_CD_grp1']=(data_train['ENG_FUEL_CD_U']+data_train['ENG_FUEL_CD_N'])
# 		data_train['ENG_FUEL_CD_grp2']=(data_train['ENG_FUEL_CD_G']+data_train['ENG_FUEL_CD_P'])
# 		data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_Y']+data_train['PLNT_CD_5']+data_train['PLNT_CD_8'])
# 		data_train['PLNT_CD_grp2']=(data_train['PLNT_CD_U']+data_train['PLNT_CD_1']+data_train['PLNT_CD_V'])
# 		data_train['PLNT_CD_grp3']=(data_train['PLNT_CD_M']+data_train['PLNT_CD_W']+data_train['PLNT_CD_F']+data_train['PLNT_CD_D']+data_train['PLNT_CD_K']+data_train['PLNT_CD_7']+data_train['PLNT_CD_S']+data_train['PLNT_CD_R']+data_train['PLNT_CD_C']+data_train['PLNT_CD_L']+data_train['PLNT_CD_2']+data_train['PLNT_CD_T']+data_train['PLNT_CD_4']+data_train['PLNT_CD_0']+data_train['PLNT_CD_9'])
# 		data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_120010']+data_train['ENG_MDL_CD_160030']+data_train['ENG_MDL_CD_110030']+data_train['ENG_MDL_CD_180020']+data_train['ENG_MDL_CD_140220']+data_train['ENG_MDL_CD_180010']+data_train['ENG_MDL_CD_090020']+data_train['ENG_MDL_CD_140167']+data_train['ENG_MDL_CD_140173']+data_train['ENG_MDL_CD_140115']+data_train['ENG_MDL_CD_160020']+data_train['ENG_MDL_CD_020105']+data_train['ENG_MDL_CD_195001']+data_train['ENG_MDL_CD_010038']+data_train['ENG_MDL_CD_165185']+data_train['ENG_MDL_CD_020085']+data_train['ENG_MDL_CD_140168']+data_train['ENG_MDL_CD_165225']+data_train['ENG_MDL_CD_140027']+data_train['ENG_MDL_CD_090077']+data_train['ENG_MDL_CD_180052']+data_train['ENG_MDL_CD_165230']+data_train['ENG_MDL_CD_090043']+data_train['ENG_MDL_CD_090045']+data_train['ENG_MDL_CD_165160']+data_train['ENG_MDL_CD_020060']+data_train['ENG_MDL_CD_010050']+data_train['ENG_MDL_CD_020090']+data_train['ENG_MDL_CD_040030']+data_train['ENG_MDL_CD_030090']+data_train['ENG_MDL_CD_140215']+data_train['ENG_MDL_CD_210100']+data_train['ENG_MDL_CD_020100']+data_train['ENG_MDL_CD_220019']+data_train['ENG_MDL_CD_210090']+data_train['ENG_MDL_CD_140055']+data_train['ENG_MDL_CD_010070']+data_train['ENG_MDL_CD_020120']+data_train['ENG_MDL_CD_140090']+data_train['ENG_MDL_CD_140125']+data_train['ENG_MDL_CD_210110']+data_train['ENG_MDL_CD_140050']+data_train['ENG_MDL_CD_090051']+data_train['ENG_MDL_CD_140234']+data_train['ENG_MDL_CD_245010']+data_train['ENG_MDL_CD_030060']+data_train['ENG_MDL_CD_210040']+data_train['ENG_MDL_CD_210070']+data_train['ENG_MDL_CD_210050']+data_train['ENG_MDL_CD_210060']+data_train['ENG_MDL_CD_210080']+data_train['ENG_MDL_CD_210020']+data_train['ENG_MDL_CD_140025']+data_train['ENG_MDL_CD_210030']+data_train['ENG_MDL_CD_020026']+data_train['ENG_MDL_CD_110070']+data_train['ENG_MDL_CD_180040']+data_train['ENG_MDL_CD_140045']+data_train['ENG_MDL_CD_140040']+data_train['ENG_MDL_CD_140020']+data_train['ENG_MDL_CD_040020']+data_train['ENG_MDL_CD_160060']+data_train['ENG_MDL_CD_030020']+data_train['ENG_MDL_CD_090040']+data_train['ENG_MDL_CD_080012']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_050020']+data_train['ENG_MDL_CD_165007']+data_train['ENG_MDL_CD_040010'])
# 		data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_150010']+data_train['ENG_MDL_CD_150020']+data_train['ENG_MDL_CD_150060']+data_train['ENG_MDL_CD_210165']+data_train['ENG_MDL_CD_090030']+data_train['ENG_MDL_CD_165030']+data_train['ENG_MDL_CD_180030']+data_train['ENG_MDL_CD_200010']+data_train['ENG_MDL_CD_140195']+data_train['ENG_MDL_CD_140105']+data_train['ENG_MDL_CD_165140']+data_train['ENG_MDL_CD_140210']+data_train['ENG_MDL_CD_080080']+data_train['ENG_MDL_CD_160012']+data_train['ENG_MDL_CD_140065']+data_train['ENG_MDL_CD_090011']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_140205']+data_train['ENG_MDL_CD_030050']+data_train['ENG_MDL_CD_030047']+data_train['ENG_MDL_CD_165090']+data_train['ENG_MDL_CD_140095']+data_train['ENG_MDL_CD_140150']+data_train['ENG_MDL_CD_090070']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_020072']+data_train['ENG_MDL_CD_165010'])
# 		data_train['ENG_MDL_CD_grp4']=(data_train['ENG_MDL_CD_150003']+data_train['ENG_MDL_CD_150150']+data_train['ENG_MDL_CD_150005']+data_train['ENG_MDL_CD_150007'])

# 		data_test['NADA_BODY1_grp1']=(data_test['NADA_BODY1_K']+data_test['NADA_BODY1_H']+data_test['NADA_BODY1_B']+data_test['NADA_BODY1_D'])
# 		data_test['NADA_BODY1_grp2']=(data_test['NADA_BODY1_M']+data_test['NADA_BODY1_G']+data_test['NADA_BODY1_I'])
# 		data_test['NADA_BODY1_grp3']=(data_test['NADA_BODY1_N']+data_test['NADA_BODY1_2'])		
# 		data_test['MAK_NM_grp1']=(data_test['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_test['MAK_NM_MAXIM']+data_test['MAK_NM_VAN_HOOL']+data_test['MAK_NM_EVOBUS']+data_test['MAK_NM_SEAGRAVE_FIRE_APPARATUS']+data_test['MAK_NM_PREVOST']+data_test['MAK_NM_MERCEDES_BENZ']+data_test['MAK_NM_HYUNDAI']+data_test['MAK_NM_TRANSPORTATION_MFG_CORP_']+data_test['MAK_NM_UNIMOG']+data_test['MAK_NM_DUPLEX']+data_test['MAK_NM_COUNTRY_COACH_MOTORHOME']+data_test['MAK_NM_FEDERAL_MOTORS']+data_test['MAK_NM_MOTOR_COACH_INDUSTRIES']+data_test['MAK_NM_GILLIG']+data_test['MAK_NM_DIAMOND_REO']+data_test['MAK_NM_GIANT']+data_test['MAK_NM_LES_AUTOBUS_MCI']+data_test['MAK_NM_SAVAGE']+data_test['MAK_NM_NEW_FLYER']+data_test['MAK_NM_ZELIGSON']+data_test['MAK_NM_PIERCE_MFG__INC_']+data_test['MAK_NM_SPARTAN_MOTORS']+data_test['MAK_NM_FWD_CORPORATION']+data_test['MAK_NM_JOHN_DEERE']+data_test['MAK_NM_CAPACITY_OF_TEXAS']+data_test['MAK_NM_OTTAWA'])
# 		data_test['MAK_NM_grp2']=(data_test['MAK_NM_WINNEBAGO']+data_test['MAK_NM_LAFORZA']+data_test['MAK_NM_THOMAS']+data_test['MAK_NM_MARMON_HERRINGTON']+data_test['MAK_NM_IVECO']+data_test['MAK_NM_SCANIA']+data_test['MAK_NM_TEREX___TEREX_ADVANCE']+data_test['MAK_NM_AMERICAN_LA_FRANCE']+data_test['MAK_NM_AUTOCAR']+data_test['MAK_NM_LODAL']+data_test['MAK_NM_WHITE_GMC'])
# 		data_test['MAK_NM_grp3']=(data_test['MAK_NM_INTERNATIONAL']+data_test['MAK_NM_KENWORTH']+data_test['MAK_NM_ISUZU']+data_test['MAK_NM_MACK']+data_test['MAK_NM_NISSAN_DIESEL']+data_test['MAK_NM_FREIGHTLINER']+data_test['MAK_NM_DODGE'])
# 		data_test['MAK_NM_grp4']=(data_test['MAK_NM_HINO']+data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']+data_test['MAK_NM_EMERGENCY_ONE']+data_test['MAK_NM_CRANE_CARRIER']+data_test['MAK_NM_UTILIMASTER']+data_test['MAK_NM_RAM']+data_test['MAK_NM_AUTOCAR_LLC'])
# 		data_test['MAK_NM_grp5']=(data_test['MAK_NM_WORKHORSE_CUSTOM_CHASSIS']+data_test['MAK_NM_ADVANCE_MIXER']+data_test['MAK_NM_BERING']+data_test['MAK_NM_FORETRAVEL_MOTORHOME']+data_test['MAK_NM_CATERPILLAR'])
# 		data_test['MAK_NM_grp6']=(data_test['MAK_NM_INDIANA_PHOENIX']+data_test['MAK_NM_BLUE_BIRD']+data_test['MAK_NM_HENDRICKSON'])
# 		data_test['ENG_FUEL_CD_grp1']=(data_test['ENG_FUEL_CD_U']+data_test['ENG_FUEL_CD_N'])
# 		data_test['ENG_FUEL_CD_grp2']=(data_test['ENG_FUEL_CD_G']+data_test['ENG_FUEL_CD_P'])
# 		data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_Y']+data_test['PLNT_CD_5']+data_test['PLNT_CD_8'])
# 		data_test['PLNT_CD_grp2']=(data_test['PLNT_CD_U']+data_test['PLNT_CD_1']+data_test['PLNT_CD_V'])
# 		data_test['PLNT_CD_grp3']=(data_test['PLNT_CD_M']+data_test['PLNT_CD_W']+data_test['PLNT_CD_F']+data_test['PLNT_CD_D']+data_test['PLNT_CD_K']+data_test['PLNT_CD_7']+data_test['PLNT_CD_S']+data_test['PLNT_CD_R']+data_test['PLNT_CD_C']+data_test['PLNT_CD_L']+data_test['PLNT_CD_2']+data_test['PLNT_CD_T']+data_test['PLNT_CD_4']+data_test['PLNT_CD_0']+data_test['PLNT_CD_9'])
# 		data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_120010']+data_test['ENG_MDL_CD_160030']+data_test['ENG_MDL_CD_110030']+data_test['ENG_MDL_CD_180020']+data_test['ENG_MDL_CD_140220']+data_test['ENG_MDL_CD_180010']+data_test['ENG_MDL_CD_090020']+data_test['ENG_MDL_CD_140167']+data_test['ENG_MDL_CD_140173']+data_test['ENG_MDL_CD_140115']+data_test['ENG_MDL_CD_160020']+data_test['ENG_MDL_CD_020105']+data_test['ENG_MDL_CD_195001']+data_test['ENG_MDL_CD_010038']+data_test['ENG_MDL_CD_165185']+data_test['ENG_MDL_CD_020085']+data_test['ENG_MDL_CD_140168']+data_test['ENG_MDL_CD_165225']+data_test['ENG_MDL_CD_140027']+data_test['ENG_MDL_CD_090077']+data_test['ENG_MDL_CD_180052']+data_test['ENG_MDL_CD_165230']+data_test['ENG_MDL_CD_090043']+data_test['ENG_MDL_CD_090045']+data_test['ENG_MDL_CD_165160']+data_test['ENG_MDL_CD_020060']+data_test['ENG_MDL_CD_010050']+data_test['ENG_MDL_CD_020090']+data_test['ENG_MDL_CD_040030']+data_test['ENG_MDL_CD_030090']+data_test['ENG_MDL_CD_140215']+data_test['ENG_MDL_CD_210100']+data_test['ENG_MDL_CD_020100']+data_test['ENG_MDL_CD_220019']+data_test['ENG_MDL_CD_210090']+data_test['ENG_MDL_CD_140055']+data_test['ENG_MDL_CD_010070']+data_test['ENG_MDL_CD_020120']+data_test['ENG_MDL_CD_140090']+data_test['ENG_MDL_CD_140125']+data_test['ENG_MDL_CD_210110']+data_test['ENG_MDL_CD_140050']+data_test['ENG_MDL_CD_090051']+data_test['ENG_MDL_CD_140234']+data_test['ENG_MDL_CD_245010']+data_test['ENG_MDL_CD_030060']+data_test['ENG_MDL_CD_210040']+data_test['ENG_MDL_CD_210070']+data_test['ENG_MDL_CD_210050']+data_test['ENG_MDL_CD_210060']+data_test['ENG_MDL_CD_210080']+data_test['ENG_MDL_CD_210020']+data_test['ENG_MDL_CD_140025']+data_test['ENG_MDL_CD_210030']+data_test['ENG_MDL_CD_020026']+data_test['ENG_MDL_CD_110070']+data_test['ENG_MDL_CD_180040']+data_test['ENG_MDL_CD_140045']+data_test['ENG_MDL_CD_140040']+data_test['ENG_MDL_CD_140020']+data_test['ENG_MDL_CD_040020']+data_test['ENG_MDL_CD_160060']+data_test['ENG_MDL_CD_030020']+data_test['ENG_MDL_CD_090040']+data_test['ENG_MDL_CD_080012']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_050020']+data_test['ENG_MDL_CD_165007']+data_test['ENG_MDL_CD_040010'])
# 		data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_150010']+data_test['ENG_MDL_CD_150020']+data_test['ENG_MDL_CD_150060']+data_test['ENG_MDL_CD_210165']+data_test['ENG_MDL_CD_090030']+data_test['ENG_MDL_CD_165030']+data_test['ENG_MDL_CD_180030']+data_test['ENG_MDL_CD_200010']+data_test['ENG_MDL_CD_140195']+data_test['ENG_MDL_CD_140105']+data_test['ENG_MDL_CD_165140']+data_test['ENG_MDL_CD_140210']+data_test['ENG_MDL_CD_080080']+data_test['ENG_MDL_CD_160012']+data_test['ENG_MDL_CD_140065']+data_test['ENG_MDL_CD_090011']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_140205']+data_test['ENG_MDL_CD_030050']+data_test['ENG_MDL_CD_030047']+data_test['ENG_MDL_CD_165090']+data_test['ENG_MDL_CD_140095']+data_test['ENG_MDL_CD_140150']+data_test['ENG_MDL_CD_090070']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_020072']+data_test['ENG_MDL_CD_165010'])
# 		data_test['ENG_MDL_CD_grp4']=(data_test['ENG_MDL_CD_150003']+data_test['ENG_MDL_CD_150150']+data_test['ENG_MDL_CD_150005']+data_test['ENG_MDL_CD_150007'])
# 	elif i==25 | i==26 | i==27:
# 		data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_train['MAK_NM_MERCEDES_BENZ']=data_train['MAK_NM_MERCEDES-BENZ']
# 		data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_train['MAK_NM_WESTERN_STAR/AUTO_CAR']
# 		data_train['MAK_NM_PIERCE_MFG__INC_']=data_train['MAK_NM_PIERCE_MFG._INC.']
# 		data_train['MAK_NM_WHITE_GMC']=data_train['MAK_NM_WHITE/GMC']
# 		data_train['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_train['MAK_NM_TRANSPORTATION_MFG_CORP.']
# 		data_train['MAK_NM_TEREX___TEREX_ADVANCE']=data_train['MAK_NM_TEREX_/_TEREX_ADVANCE']
# 		data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_test['MAK_NM_MERCEDES_BENZ']=data_test['MAK_NM_MERCEDES-BENZ']	
# 		data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_test['MAK_NM_WESTERN_STAR/AUTO_CAR']
# 		data_test['MAK_NM_PIERCE_MFG__INC_']=data_test['MAK_NM_PIERCE_MFG._INC.']
# 		data_test['MAK_NM_WHITE_GMC']=data_test['MAK_NM_WHITE/GMC']
# 		data_test['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_test['MAK_NM_TRANSPORTATION_MFG_CORP.']
# 		data_test['MAK_NM_TEREX___TEREX_ADVANCE']=data_test['MAK_NM_TEREX_/_TEREX_ADVANCE']
		
# 		data_train['NADA_BODY1_grp1']=(data_train['NADA_BODY1_K']+data_train['NADA_BODY1_H']+data_train['NADA_BODY1_B']+data_train['NADA_BODY1_D'])
# 		data_train['NADA_BODY1_grp2']=(data_train['NADA_BODY1_M']+data_train['NADA_BODY1_G']+data_train['NADA_BODY1_I'])
# 		data_train['NADA_BODY1_grp3']=(data_train['NADA_BODY1_N']+data_train['NADA_BODY1_2'])		
# 		data_train['MAK_NM_grp1']=(data_train['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_train['MAK_NM_MAXIM']+data_train['MAK_NM_VAN_HOOL']+data_train['MAK_NM_EVOBUS']+data_train['MAK_NM_SEAGRAVE_FIRE_APPARATUS']+data_train['MAK_NM_PREVOST']+data_train['MAK_NM_MERCEDES_BENZ']+data_train['MAK_NM_HYUNDAI']+data_train['MAK_NM_TRANSPORTATION_MFG_CORP_']+data_train['MAK_NM_UNIMOG']+data_train['MAK_NM_DUPLEX']+data_train['MAK_NM_COUNTRY_COACH_MOTORHOME']+data_train['MAK_NM_FEDERAL_MOTORS']+data_train['MAK_NM_MOTOR_COACH_INDUSTRIES']+data_train['MAK_NM_GILLIG']+data_train['MAK_NM_DIAMOND_REO']+data_train['MAK_NM_GIANT']+data_train['MAK_NM_LES_AUTOBUS_MCI']+data_train['MAK_NM_SAVAGE']+data_train['MAK_NM_NEW_FLYER']+data_train['MAK_NM_ZELIGSON']+data_train['MAK_NM_PIERCE_MFG__INC_']+data_train['MAK_NM_SPARTAN_MOTORS']+data_train['MAK_NM_FWD_CORPORATION']+data_train['MAK_NM_JOHN_DEERE']+data_train['MAK_NM_CAPACITY_OF_TEXAS']+data_train['MAK_NM_OTTAWA'])
# 		data_train['MAK_NM_grp2']=(data_train['MAK_NM_WINNEBAGO']+data_train['MAK_NM_LAFORZA']+data_train['MAK_NM_THOMAS']+data_train['MAK_NM_MARMON_HERRINGTON']+data_train['MAK_NM_IVECO']+data_train['MAK_NM_SCANIA']+data_train['MAK_NM_TEREX___TEREX_ADVANCE']+data_train['MAK_NM_AMERICAN_LA_FRANCE']+data_train['MAK_NM_AUTOCAR']+data_train['MAK_NM_LODAL']+data_train['MAK_NM_WHITE_GMC'])
# 		data_train['MAK_NM_grp3']=(data_train['MAK_NM_INTERNATIONAL']+data_train['MAK_NM_KENWORTH']+data_train['MAK_NM_ISUZU']+data_train['MAK_NM_MACK']+data_train['MAK_NM_NISSAN_DIESEL']+data_train['MAK_NM_FREIGHTLINER']+data_train['MAK_NM_DODGE'])
# 		data_train['MAK_NM_grp4']=(data_train['MAK_NM_HINO']+data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']+data_train['MAK_NM_EMERGENCY_ONE']+data_train['MAK_NM_CRANE_CARRIER']+data_train['MAK_NM_UTILIMASTER']+data_train['MAK_NM_RAM']+data_train['MAK_NM_AUTOCAR_LLC'])
# 		data_train['MAK_NM_grp5']=(data_train['MAK_NM_WORKHORSE_CUSTOM_CHASSIS']+data_train['MAK_NM_ADVANCE_MIXER']+data_train['MAK_NM_BERING']+data_train['MAK_NM_FORETRAVEL_MOTORHOME']+data_train['MAK_NM_CATERPILLAR'])
# 		data_train['MAK_NM_grp6']=(data_train['MAK_NM_INDIANA_PHOENIX']+data_train['MAK_NM_BLUE_BIRD']+data_train['MAK_NM_HENDRICKSON'])
# 		data_train['ENG_FUEL_CD_grp1']=(data_train['ENG_FUEL_CD_U']+data_train['ENG_FUEL_CD_N'])
# 		data_train['ENG_FUEL_CD_grp2']=(data_train['ENG_FUEL_CD_G']+data_train['ENG_FUEL_CD_P'])
# 		data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_Y']+data_train['PLNT_CD_5']+data_train['PLNT_CD_8'])
# 		data_train['PLNT_CD_grp2']=(data_train['PLNT_CD_U']+data_train['PLNT_CD_1']+data_train['PLNT_CD_V'])
# 		data_train['PLNT_CD_grp3']=(data_train['PLNT_CD_M']+data_train['PLNT_CD_W']+data_train['PLNT_CD_F']+data_train['PLNT_CD_D']+data_train['PLNT_CD_K']+data_train['PLNT_CD_7']+data_train['PLNT_CD_S']+data_train['PLNT_CD_R']+data_train['PLNT_CD_C']+data_train['PLNT_CD_L']+data_train['PLNT_CD_2']+data_train['PLNT_CD_T']+data_train['PLNT_CD_4']+data_train['PLNT_CD_0']+data_train['PLNT_CD_9'])
# 		data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_120010']+data_train['ENG_MDL_CD_160030']+data_train['ENG_MDL_CD_110030']+data_train['ENG_MDL_CD_180020']+data_train['ENG_MDL_CD_140220']+data_train['ENG_MDL_CD_180010']+data_train['ENG_MDL_CD_090020']+data_train['ENG_MDL_CD_140167']+data_train['ENG_MDL_CD_140173']+data_train['ENG_MDL_CD_140115']+data_train['ENG_MDL_CD_160020']+data_train['ENG_MDL_CD_020105']+data_train['ENG_MDL_CD_195001']+data_train['ENG_MDL_CD_010038']+data_train['ENG_MDL_CD_165185']+data_train['ENG_MDL_CD_020085']+data_train['ENG_MDL_CD_140168']+data_train['ENG_MDL_CD_165225']+data_train['ENG_MDL_CD_140027']+data_train['ENG_MDL_CD_090077']+data_train['ENG_MDL_CD_180052']+data_train['ENG_MDL_CD_165230']+data_train['ENG_MDL_CD_090043']+data_train['ENG_MDL_CD_090045']+data_train['ENG_MDL_CD_165160']+data_train['ENG_MDL_CD_020060']+data_train['ENG_MDL_CD_010050']+data_train['ENG_MDL_CD_020090']+data_train['ENG_MDL_CD_040030']+data_train['ENG_MDL_CD_030090']+data_train['ENG_MDL_CD_140215']+data_train['ENG_MDL_CD_210100']+data_train['ENG_MDL_CD_020100']+data_train['ENG_MDL_CD_220019']+data_train['ENG_MDL_CD_210090']+data_train['ENG_MDL_CD_140055']+data_train['ENG_MDL_CD_010070']+data_train['ENG_MDL_CD_020120']+data_train['ENG_MDL_CD_140090']+data_train['ENG_MDL_CD_140125']+data_train['ENG_MDL_CD_210110']+data_train['ENG_MDL_CD_140050']+data_train['ENG_MDL_CD_090051']+data_train['ENG_MDL_CD_140234']+data_train['ENG_MDL_CD_245010']+data_train['ENG_MDL_CD_030060']+data_train['ENG_MDL_CD_210040']+data_train['ENG_MDL_CD_210070']+data_train['ENG_MDL_CD_210050']+data_train['ENG_MDL_CD_210060']+data_train['ENG_MDL_CD_210080']+data_train['ENG_MDL_CD_210020']+data_train['ENG_MDL_CD_140025']+data_train['ENG_MDL_CD_210030']+data_train['ENG_MDL_CD_020026']+data_train['ENG_MDL_CD_110070']+data_train['ENG_MDL_CD_180040']+data_train['ENG_MDL_CD_140045']+data_train['ENG_MDL_CD_140040']+data_train['ENG_MDL_CD_140020']+data_train['ENG_MDL_CD_040020']+data_train['ENG_MDL_CD_160060']+data_train['ENG_MDL_CD_030020']+data_train['ENG_MDL_CD_090040']+data_train['ENG_MDL_CD_080012']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_050020']+data_train['ENG_MDL_CD_165007']+data_train['ENG_MDL_CD_040010'])
# 		data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_150010']+data_train['ENG_MDL_CD_150020']+data_train['ENG_MDL_CD_150060']+data_train['ENG_MDL_CD_210165']+data_train['ENG_MDL_CD_090030']+data_train['ENG_MDL_CD_165030']+data_train['ENG_MDL_CD_180030']+data_train['ENG_MDL_CD_200010']+data_train['ENG_MDL_CD_140195']+data_train['ENG_MDL_CD_140105']+data_train['ENG_MDL_CD_165140']+data_train['ENG_MDL_CD_140210']+data_train['ENG_MDL_CD_080080']+data_train['ENG_MDL_CD_160012']+data_train['ENG_MDL_CD_140065']+data_train['ENG_MDL_CD_090011']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_140205']+data_train['ENG_MDL_CD_030050']+data_train['ENG_MDL_CD_030047']+data_train['ENG_MDL_CD_165090']+data_train['ENG_MDL_CD_140095']+data_train['ENG_MDL_CD_140150']+data_train['ENG_MDL_CD_090070']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_020072']+data_train['ENG_MDL_CD_165010'])
# 		data_train['ENG_MDL_CD_grp4']=(data_train['ENG_MDL_CD_150003']+data_train['ENG_MDL_CD_150150']+data_train['ENG_MDL_CD_150005']+data_train['ENG_MDL_CD_150007'])

# 		data_test['NADA_BODY1_grp1']=(data_test['NADA_BODY1_K']+data_test['NADA_BODY1_H']+data_test['NADA_BODY1_B']+data_test['NADA_BODY1_D'])
# 		data_test['NADA_BODY1_grp2']=(data_test['NADA_BODY1_M']+data_test['NADA_BODY1_G']+data_test['NADA_BODY1_I'])
# 		data_test['NADA_BODY1_grp3']=(data_test['NADA_BODY1_N']+data_test['NADA_BODY1_2'])		
# 		data_test['MAK_NM_grp1']=(data_test['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_test['MAK_NM_MAXIM']+data_test['MAK_NM_VAN_HOOL']+data_test['MAK_NM_EVOBUS']+data_test['MAK_NM_SEAGRAVE_FIRE_APPARATUS']+data_test['MAK_NM_PREVOST']+data_test['MAK_NM_MERCEDES_BENZ']+data_test['MAK_NM_HYUNDAI']+data_test['MAK_NM_TRANSPORTATION_MFG_CORP_']+data_test['MAK_NM_UNIMOG']+data_test['MAK_NM_DUPLEX']+data_test['MAK_NM_COUNTRY_COACH_MOTORHOME']+data_test['MAK_NM_FEDERAL_MOTORS']+data_test['MAK_NM_MOTOR_COACH_INDUSTRIES']+data_test['MAK_NM_GILLIG']+data_test['MAK_NM_DIAMOND_REO']+data_test['MAK_NM_GIANT']+data_test['MAK_NM_LES_AUTOBUS_MCI']+data_test['MAK_NM_SAVAGE']+data_test['MAK_NM_NEW_FLYER']+data_test['MAK_NM_ZELIGSON']+data_test['MAK_NM_PIERCE_MFG__INC_']+data_test['MAK_NM_SPARTAN_MOTORS']+data_test['MAK_NM_FWD_CORPORATION']+data_test['MAK_NM_JOHN_DEERE']+data_test['MAK_NM_CAPACITY_OF_TEXAS']+data_test['MAK_NM_OTTAWA'])
# 		data_test['MAK_NM_grp2']=(data_test['MAK_NM_WINNEBAGO']+data_test['MAK_NM_LAFORZA']+data_test['MAK_NM_THOMAS']+data_test['MAK_NM_MARMON_HERRINGTON']+data_test['MAK_NM_IVECO']+data_test['MAK_NM_SCANIA']+data_test['MAK_NM_TEREX___TEREX_ADVANCE']+data_test['MAK_NM_AMERICAN_LA_FRANCE']+data_test['MAK_NM_AUTOCAR']+data_test['MAK_NM_LODAL']+data_test['MAK_NM_WHITE_GMC'])
# 		data_test['MAK_NM_grp3']=(data_test['MAK_NM_INTERNATIONAL']+data_test['MAK_NM_KENWORTH']+data_test['MAK_NM_ISUZU']+data_test['MAK_NM_MACK']+data_test['MAK_NM_NISSAN_DIESEL']+data_test['MAK_NM_FREIGHTLINER']+data_test['MAK_NM_DODGE'])
# 		data_test['MAK_NM_grp4']=(data_test['MAK_NM_HINO']+data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']+data_test['MAK_NM_EMERGENCY_ONE']+data_test['MAK_NM_CRANE_CARRIER']+data_test['MAK_NM_UTILIMASTER']+data_test['MAK_NM_RAM']+data_test['MAK_NM_AUTOCAR_LLC'])
# 		data_test['MAK_NM_grp5']=(data_test['MAK_NM_WORKHORSE_CUSTOM_CHASSIS']+data_test['MAK_NM_ADVANCE_MIXER']+data_test['MAK_NM_BERING']+data_test['MAK_NM_FORETRAVEL_MOTORHOME']+data_test['MAK_NM_CATERPILLAR'])
# 		data_test['MAK_NM_grp6']=(data_test['MAK_NM_INDIANA_PHOENIX']+data_test['MAK_NM_BLUE_BIRD']+data_test['MAK_NM_HENDRICKSON'])
# 		data_test['ENG_FUEL_CD_grp1']=(data_test['ENG_FUEL_CD_U']+data_test['ENG_FUEL_CD_N'])
# 		data_test['ENG_FUEL_CD_grp2']=(data_test['ENG_FUEL_CD_G']+data_test['ENG_FUEL_CD_P'])
# 		data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_Y']+data_test['PLNT_CD_5']+data_test['PLNT_CD_8'])
# 		data_test['PLNT_CD_grp2']=(data_test['PLNT_CD_U']+data_test['PLNT_CD_1']+data_test['PLNT_CD_V'])
# 		data_test['PLNT_CD_grp3']=(data_test['PLNT_CD_M']+data_test['PLNT_CD_W']+data_test['PLNT_CD_F']+data_test['PLNT_CD_D']+data_test['PLNT_CD_K']+data_test['PLNT_CD_7']+data_test['PLNT_CD_S']+data_test['PLNT_CD_R']+data_test['PLNT_CD_C']+data_test['PLNT_CD_L']+data_test['PLNT_CD_2']+data_test['PLNT_CD_T']+data_test['PLNT_CD_4']+data_test['PLNT_CD_0']+data_test['PLNT_CD_9'])
# 		data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_120010']+data_test['ENG_MDL_CD_160030']+data_test['ENG_MDL_CD_110030']+data_test['ENG_MDL_CD_180020']+data_test['ENG_MDL_CD_140220']+data_test['ENG_MDL_CD_180010']+data_test['ENG_MDL_CD_090020']+data_test['ENG_MDL_CD_140167']+data_test['ENG_MDL_CD_140173']+data_test['ENG_MDL_CD_140115']+data_test['ENG_MDL_CD_160020']+data_test['ENG_MDL_CD_020105']+data_test['ENG_MDL_CD_195001']+data_test['ENG_MDL_CD_010038']+data_test['ENG_MDL_CD_165185']+data_test['ENG_MDL_CD_020085']+data_test['ENG_MDL_CD_140168']+data_test['ENG_MDL_CD_165225']+data_test['ENG_MDL_CD_140027']+data_test['ENG_MDL_CD_090077']+data_test['ENG_MDL_CD_180052']+data_test['ENG_MDL_CD_165230']+data_test['ENG_MDL_CD_090043']+data_test['ENG_MDL_CD_090045']+data_test['ENG_MDL_CD_165160']+data_test['ENG_MDL_CD_020060']+data_test['ENG_MDL_CD_010050']+data_test['ENG_MDL_CD_020090']+data_test['ENG_MDL_CD_040030']+data_test['ENG_MDL_CD_030090']+data_test['ENG_MDL_CD_140215']+data_test['ENG_MDL_CD_210100']+data_test['ENG_MDL_CD_020100']+data_test['ENG_MDL_CD_220019']+data_test['ENG_MDL_CD_210090']+data_test['ENG_MDL_CD_140055']+data_test['ENG_MDL_CD_010070']+data_test['ENG_MDL_CD_020120']+data_test['ENG_MDL_CD_140090']+data_test['ENG_MDL_CD_140125']+data_test['ENG_MDL_CD_210110']+data_test['ENG_MDL_CD_140050']+data_test['ENG_MDL_CD_090051']+data_test['ENG_MDL_CD_140234']+data_test['ENG_MDL_CD_245010']+data_test['ENG_MDL_CD_030060']+data_test['ENG_MDL_CD_210040']+data_test['ENG_MDL_CD_210070']+data_test['ENG_MDL_CD_210050']+data_test['ENG_MDL_CD_210060']+data_test['ENG_MDL_CD_210080']+data_test['ENG_MDL_CD_210020']+data_test['ENG_MDL_CD_140025']+data_test['ENG_MDL_CD_210030']+data_test['ENG_MDL_CD_020026']+data_test['ENG_MDL_CD_110070']+data_test['ENG_MDL_CD_180040']+data_test['ENG_MDL_CD_140045']+data_test['ENG_MDL_CD_140040']+data_test['ENG_MDL_CD_140020']+data_test['ENG_MDL_CD_040020']+data_test['ENG_MDL_CD_160060']+data_test['ENG_MDL_CD_030020']+data_test['ENG_MDL_CD_090040']+data_test['ENG_MDL_CD_080012']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_050020']+data_test['ENG_MDL_CD_165007']+data_test['ENG_MDL_CD_040010'])
# 		data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_150010']+data_test['ENG_MDL_CD_150020']+data_test['ENG_MDL_CD_150060']+data_test['ENG_MDL_CD_210165']+data_test['ENG_MDL_CD_090030']+data_test['ENG_MDL_CD_165030']+data_test['ENG_MDL_CD_180030']+data_test['ENG_MDL_CD_200010']+data_test['ENG_MDL_CD_140195']+data_test['ENG_MDL_CD_140105']+data_test['ENG_MDL_CD_165140']+data_test['ENG_MDL_CD_140210']+data_test['ENG_MDL_CD_080080']+data_test['ENG_MDL_CD_160012']+data_test['ENG_MDL_CD_140065']+data_test['ENG_MDL_CD_090011']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_140205']+data_test['ENG_MDL_CD_030050']+data_test['ENG_MDL_CD_030047']+data_test['ENG_MDL_CD_165090']+data_test['ENG_MDL_CD_140095']+data_test['ENG_MDL_CD_140150']+data_test['ENG_MDL_CD_090070']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_020072']+data_test['ENG_MDL_CD_165010'])
# 		data_test['ENG_MDL_CD_grp4']=(data_test['ENG_MDL_CD_150003']+data_test['ENG_MDL_CD_150150']+data_test['ENG_MDL_CD_150005']+data_test['ENG_MDL_CD_150007'])
# 	elif i==28:
# 		data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_train['MAK_NM_MERCEDES_BENZ']=data_train['MAK_NM_MERCEDES-BENZ']
# 		data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_train['MAK_NM_WESTERN_STAR/AUTO_CAR']
# 		data_train['MAK_NM_PIERCE_MFG__INC_']=data_train['MAK_NM_PIERCE_MFG._INC.']
# 		data_train['MAK_NM_WHITE_GMC']=data_train['MAK_NM_WHITE/GMC']
# 		data_train['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_train['MAK_NM_TRANSPORTATION_MFG_CORP.']
# 		data_train['MAK_NM_TEREX___TEREX_ADVANCE']=data_train['MAK_NM_TEREX_/_TEREX_ADVANCE']
# 		data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_test['MAK_NM_MERCEDES_BENZ']=data_test['MAK_NM_MERCEDES-BENZ']	
# 		data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_test['MAK_NM_WESTERN_STAR/AUTO_CAR']
# 		data_test['MAK_NM_PIERCE_MFG__INC_']=data_test['MAK_NM_PIERCE_MFG._INC.']
# 		data_test['MAK_NM_WHITE_GMC']=data_test['MAK_NM_WHITE/GMC']
# 		data_test['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_test['MAK_NM_TRANSPORTATION_MFG_CORP.']
# 		data_test['MAK_NM_TEREX___TEREX_ADVANCE']=data_test['MAK_NM_TEREX_/_TEREX_ADVANCE']
		
# 		data_train['NADA_BODY1_grp1']=(data_train['NADA_BODY1_K']+data_train['NADA_BODY1_H']+data_train['NADA_BODY1_B']+data_train['NADA_BODY1_D'])
# 		data_train['NADA_BODY1_grp2']=(data_train['NADA_BODY1_M']+data_train['NADA_BODY1_G']+data_train['NADA_BODY1_I'])
# 		data_train['NADA_BODY1_grp3']=(data_train['NADA_BODY1_N']+data_train['NADA_BODY1_2'])		
# 		data_train['MAK_NM_grp1']=(data_train['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_train['MAK_NM_MAXIM']+data_train['MAK_NM_VAN_HOOL']+data_train['MAK_NM_EVOBUS']+data_train['MAK_NM_SEAGRAVE_FIRE_APPARATUS']+data_train['MAK_NM_PREVOST']+data_train['MAK_NM_MERCEDES_BENZ']+data_train['MAK_NM_HYUNDAI']+data_train['MAK_NM_TRANSPORTATION_MFG_CORP_']+data_train['MAK_NM_UNIMOG']+data_train['MAK_NM_DUPLEX']+data_train['MAK_NM_COUNTRY_COACH_MOTORHOME']+data_train['MAK_NM_FEDERAL_MOTORS']+data_train['MAK_NM_MOTOR_COACH_INDUSTRIES']+data_train['MAK_NM_GILLIG']+data_train['MAK_NM_DIAMOND_REO']+data_train['MAK_NM_GIANT']+data_train['MAK_NM_LES_AUTOBUS_MCI']+data_train['MAK_NM_SAVAGE']+data_train['MAK_NM_NEW_FLYER']+data_train['MAK_NM_ZELIGSON']+data_train['MAK_NM_PIERCE_MFG__INC_']+data_train['MAK_NM_SPARTAN_MOTORS']+data_train['MAK_NM_FWD_CORPORATION']+data_train['MAK_NM_JOHN_DEERE']+data_train['MAK_NM_CAPACITY_OF_TEXAS']+data_train['MAK_NM_OTTAWA']+data_train['MAK_NM_WINNEBAGO']+data_train['MAK_NM_LAFORZA']+data_train['MAK_NM_THOMAS']+data_train['MAK_NM_MARMON_HERRINGTON']+data_train['MAK_NM_IVECO']+data_train['MAK_NM_SCANIA']+data_train['MAK_NM_TEREX___TEREX_ADVANCE']+data_train['MAK_NM_AMERICAN_LA_FRANCE']+data_train['MAK_NM_AUTOCAR']+data_train['MAK_NM_LODAL']+data_train['MAK_NM_WHITE_GMC'])
# 		data_train['MAK_NM_grp3']=(data_train['MAK_NM_INTERNATIONAL']+data_train['MAK_NM_KENWORTH']+data_train['MAK_NM_ISUZU']+data_train['MAK_NM_MACK']+data_train['MAK_NM_NISSAN_DIESEL']+data_train['MAK_NM_FREIGHTLINER']+data_train['MAK_NM_DODGE'])
# 		data_train['MAK_NM_grp4']=(data_train['MAK_NM_HINO']+data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']+data_train['MAK_NM_EMERGENCY_ONE']+data_train['MAK_NM_CRANE_CARRIER']+data_train['MAK_NM_UTILIMASTER']+data_train['MAK_NM_RAM']+data_train['MAK_NM_AUTOCAR_LLC'])
# 		data_train['MAK_NM_grp5']=(data_train['MAK_NM_WORKHORSE_CUSTOM_CHASSIS']+data_train['MAK_NM_ADVANCE_MIXER']+data_train['MAK_NM_BERING']+data_train['MAK_NM_FORETRAVEL_MOTORHOME']+data_train['MAK_NM_CATERPILLAR'])
# 		data_train['MAK_NM_grp6']=(data_train['MAK_NM_INDIANA_PHOENIX']+data_train['MAK_NM_BLUE_BIRD']+data_train['MAK_NM_HENDRICKSON'])
# 		data_train['ENG_FUEL_CD_grp1']=(data_train['ENG_FUEL_CD_U']+data_train['ENG_FUEL_CD_N'])
# 		data_train['ENG_FUEL_CD_grp2']=(data_train['ENG_FUEL_CD_G']+data_train['ENG_FUEL_CD_P'])
# 		data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_Y']+data_train['PLNT_CD_5']+data_train['PLNT_CD_8'])
# 		data_train['PLNT_CD_grp2']=(data_train['PLNT_CD_U']+data_train['PLNT_CD_1']+data_train['PLNT_CD_V'])
# 		data_train['PLNT_CD_grp3']=(data_train['PLNT_CD_M']+data_train['PLNT_CD_W']+data_train['PLNT_CD_F']+data_train['PLNT_CD_D']+data_train['PLNT_CD_K']+data_train['PLNT_CD_7']+data_train['PLNT_CD_S']+data_train['PLNT_CD_R']+data_train['PLNT_CD_C']+data_train['PLNT_CD_L']+data_train['PLNT_CD_2']+data_train['PLNT_CD_T']+data_train['PLNT_CD_4']+data_train['PLNT_CD_0']+data_train['PLNT_CD_9'])
# 		data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_120010']+data_train['ENG_MDL_CD_160030']+data_train['ENG_MDL_CD_110030']+data_train['ENG_MDL_CD_180020']+data_train['ENG_MDL_CD_140220']+data_train['ENG_MDL_CD_180010']+data_train['ENG_MDL_CD_090020']+data_train['ENG_MDL_CD_140167']+data_train['ENG_MDL_CD_140173']+data_train['ENG_MDL_CD_140115']+data_train['ENG_MDL_CD_160020']+data_train['ENG_MDL_CD_020105']+data_train['ENG_MDL_CD_195001']+data_train['ENG_MDL_CD_010038']+data_train['ENG_MDL_CD_165185']+data_train['ENG_MDL_CD_020085']+data_train['ENG_MDL_CD_140168']+data_train['ENG_MDL_CD_165225']+data_train['ENG_MDL_CD_140027']+data_train['ENG_MDL_CD_090077']+data_train['ENG_MDL_CD_180052']+data_train['ENG_MDL_CD_165230']+data_train['ENG_MDL_CD_090043']+data_train['ENG_MDL_CD_090045']+data_train['ENG_MDL_CD_165160']+data_train['ENG_MDL_CD_020060']+data_train['ENG_MDL_CD_010050']+data_train['ENG_MDL_CD_020090']+data_train['ENG_MDL_CD_040030']+data_train['ENG_MDL_CD_030090']+data_train['ENG_MDL_CD_140215']+data_train['ENG_MDL_CD_210100']+data_train['ENG_MDL_CD_020100']+data_train['ENG_MDL_CD_220019']+data_train['ENG_MDL_CD_210090']+data_train['ENG_MDL_CD_140055']+data_train['ENG_MDL_CD_010070']+data_train['ENG_MDL_CD_020120']+data_train['ENG_MDL_CD_140090']+data_train['ENG_MDL_CD_140125']+data_train['ENG_MDL_CD_210110']+data_train['ENG_MDL_CD_140050']+data_train['ENG_MDL_CD_090051']+data_train['ENG_MDL_CD_140234']+data_train['ENG_MDL_CD_245010']+data_train['ENG_MDL_CD_030060']+data_train['ENG_MDL_CD_210040']+data_train['ENG_MDL_CD_210070']+data_train['ENG_MDL_CD_210050']+data_train['ENG_MDL_CD_210060']+data_train['ENG_MDL_CD_210080']+data_train['ENG_MDL_CD_210020']+data_train['ENG_MDL_CD_140025']+data_train['ENG_MDL_CD_210030']+data_train['ENG_MDL_CD_020026']+data_train['ENG_MDL_CD_110070']+data_train['ENG_MDL_CD_180040']+data_train['ENG_MDL_CD_140045']+data_train['ENG_MDL_CD_140040']+data_train['ENG_MDL_CD_140020']+data_train['ENG_MDL_CD_040020']+data_train['ENG_MDL_CD_160060']+data_train['ENG_MDL_CD_030020']+data_train['ENG_MDL_CD_090040']+data_train['ENG_MDL_CD_080012']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_050020']+data_train['ENG_MDL_CD_165007']+data_train['ENG_MDL_CD_040010'])
# 		data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_150010']+data_train['ENG_MDL_CD_150020']+data_train['ENG_MDL_CD_150060']+data_train['ENG_MDL_CD_210165']+data_train['ENG_MDL_CD_090030']+data_train['ENG_MDL_CD_165030']+data_train['ENG_MDL_CD_180030']+data_train['ENG_MDL_CD_200010']+data_train['ENG_MDL_CD_140195']+data_train['ENG_MDL_CD_140105']+data_train['ENG_MDL_CD_165140']+data_train['ENG_MDL_CD_140210']+data_train['ENG_MDL_CD_080080']+data_train['ENG_MDL_CD_160012']+data_train['ENG_MDL_CD_140065']+data_train['ENG_MDL_CD_090011']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_140205']+data_train['ENG_MDL_CD_030050']+data_train['ENG_MDL_CD_030047']+data_train['ENG_MDL_CD_165090']+data_train['ENG_MDL_CD_140095']+data_train['ENG_MDL_CD_140150']+data_train['ENG_MDL_CD_090070']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_020072']+data_train['ENG_MDL_CD_165010'])
# 		data_train['ENG_MDL_CD_grp4']=(data_train['ENG_MDL_CD_150003']+data_train['ENG_MDL_CD_150150']+data_train['ENG_MDL_CD_150005']+data_train['ENG_MDL_CD_150007'])

# 		data_test['NADA_BODY1_grp1']=(data_test['NADA_BODY1_K']+data_test['NADA_BODY1_H']+data_test['NADA_BODY1_B']+data_test['NADA_BODY1_D'])
# 		data_test['NADA_BODY1_grp2']=(data_test['NADA_BODY1_M']+data_test['NADA_BODY1_G']+data_test['NADA_BODY1_I'])
# 		data_test['NADA_BODY1_grp3']=(data_test['NADA_BODY1_N']+data_test['NADA_BODY1_2'])		
# 		data_test['MAK_NM_grp1']=(data_test['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_test['MAK_NM_MAXIM']+data_test['MAK_NM_VAN_HOOL']+data_test['MAK_NM_EVOBUS']+data_test['MAK_NM_SEAGRAVE_FIRE_APPARATUS']+data_test['MAK_NM_PREVOST']+data_test['MAK_NM_MERCEDES_BENZ']+data_test['MAK_NM_HYUNDAI']+data_test['MAK_NM_TRANSPORTATION_MFG_CORP_']+data_test['MAK_NM_UNIMOG']+data_test['MAK_NM_DUPLEX']+data_test['MAK_NM_COUNTRY_COACH_MOTORHOME']+data_test['MAK_NM_FEDERAL_MOTORS']+data_test['MAK_NM_MOTOR_COACH_INDUSTRIES']+data_test['MAK_NM_GILLIG']+data_test['MAK_NM_DIAMOND_REO']+data_test['MAK_NM_GIANT']+data_test['MAK_NM_LES_AUTOBUS_MCI']+data_test['MAK_NM_SAVAGE']+data_test['MAK_NM_NEW_FLYER']+data_test['MAK_NM_ZELIGSON']+data_test['MAK_NM_PIERCE_MFG__INC_']+data_test['MAK_NM_SPARTAN_MOTORS']+data_test['MAK_NM_FWD_CORPORATION']+data_test['MAK_NM_JOHN_DEERE']+data_test['MAK_NM_CAPACITY_OF_TEXAS']+data_test['MAK_NM_OTTAWA']+data_test['MAK_NM_WINNEBAGO']+data_test['MAK_NM_LAFORZA']+data_test['MAK_NM_THOMAS']+data_test['MAK_NM_MARMON_HERRINGTON']+data_test['MAK_NM_IVECO']+data_test['MAK_NM_SCANIA']+data_test['MAK_NM_TEREX___TEREX_ADVANCE']+data_test['MAK_NM_AMERICAN_LA_FRANCE']+data_test['MAK_NM_AUTOCAR']+data_test['MAK_NM_LODAL']+data_test['MAK_NM_WHITE_GMC'])
# 		data_test['MAK_NM_grp3']=(data_test['MAK_NM_INTERNATIONAL']+data_test['MAK_NM_KENWORTH']+data_test['MAK_NM_ISUZU']+data_test['MAK_NM_MACK']+data_test['MAK_NM_NISSAN_DIESEL']+data_test['MAK_NM_FREIGHTLINER']+data_test['MAK_NM_DODGE'])
# 		data_test['MAK_NM_grp4']=(data_test['MAK_NM_HINO']+data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']+data_test['MAK_NM_EMERGENCY_ONE']+data_test['MAK_NM_CRANE_CARRIER']+data_test['MAK_NM_UTILIMASTER']+data_test['MAK_NM_RAM']+data_test['MAK_NM_AUTOCAR_LLC'])
# 		data_test['MAK_NM_grp5']=(data_test['MAK_NM_WORKHORSE_CUSTOM_CHASSIS']+data_test['MAK_NM_ADVANCE_MIXER']+data_test['MAK_NM_BERING']+data_test['MAK_NM_FORETRAVEL_MOTORHOME']+data_test['MAK_NM_CATERPILLAR'])
# 		data_test['MAK_NM_grp6']=(data_test['MAK_NM_INDIANA_PHOENIX']+data_test['MAK_NM_BLUE_BIRD']+data_test['MAK_NM_HENDRICKSON'])
# 		data_test['ENG_FUEL_CD_grp1']=(data_test['ENG_FUEL_CD_U']+data_test['ENG_FUEL_CD_N'])
# 		data_test['ENG_FUEL_CD_grp2']=(data_test['ENG_FUEL_CD_G']+data_test['ENG_FUEL_CD_P'])
# 		data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_Y']+data_test['PLNT_CD_5']+data_test['PLNT_CD_8'])
# 		data_test['PLNT_CD_grp2']=(data_test['PLNT_CD_U']+data_test['PLNT_CD_1']+data_test['PLNT_CD_V'])
# 		data_test['PLNT_CD_grp3']=(data_test['PLNT_CD_M']+data_test['PLNT_CD_W']+data_test['PLNT_CD_F']+data_test['PLNT_CD_D']+data_test['PLNT_CD_K']+data_test['PLNT_CD_7']+data_test['PLNT_CD_S']+data_test['PLNT_CD_R']+data_test['PLNT_CD_C']+data_test['PLNT_CD_L']+data_test['PLNT_CD_2']+data_test['PLNT_CD_T']+data_test['PLNT_CD_4']+data_test['PLNT_CD_0']+data_test['PLNT_CD_9'])
# 		data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_120010']+data_test['ENG_MDL_CD_160030']+data_test['ENG_MDL_CD_110030']+data_test['ENG_MDL_CD_180020']+data_test['ENG_MDL_CD_140220']+data_test['ENG_MDL_CD_180010']+data_test['ENG_MDL_CD_090020']+data_test['ENG_MDL_CD_140167']+data_test['ENG_MDL_CD_140173']+data_test['ENG_MDL_CD_140115']+data_test['ENG_MDL_CD_160020']+data_test['ENG_MDL_CD_020105']+data_test['ENG_MDL_CD_195001']+data_test['ENG_MDL_CD_010038']+data_test['ENG_MDL_CD_165185']+data_test['ENG_MDL_CD_020085']+data_test['ENG_MDL_CD_140168']+data_test['ENG_MDL_CD_165225']+data_test['ENG_MDL_CD_140027']+data_test['ENG_MDL_CD_090077']+data_test['ENG_MDL_CD_180052']+data_test['ENG_MDL_CD_165230']+data_test['ENG_MDL_CD_090043']+data_test['ENG_MDL_CD_090045']+data_test['ENG_MDL_CD_165160']+data_test['ENG_MDL_CD_020060']+data_test['ENG_MDL_CD_010050']+data_test['ENG_MDL_CD_020090']+data_test['ENG_MDL_CD_040030']+data_test['ENG_MDL_CD_030090']+data_test['ENG_MDL_CD_140215']+data_test['ENG_MDL_CD_210100']+data_test['ENG_MDL_CD_020100']+data_test['ENG_MDL_CD_220019']+data_test['ENG_MDL_CD_210090']+data_test['ENG_MDL_CD_140055']+data_test['ENG_MDL_CD_010070']+data_test['ENG_MDL_CD_020120']+data_test['ENG_MDL_CD_140090']+data_test['ENG_MDL_CD_140125']+data_test['ENG_MDL_CD_210110']+data_test['ENG_MDL_CD_140050']+data_test['ENG_MDL_CD_090051']+data_test['ENG_MDL_CD_140234']+data_test['ENG_MDL_CD_245010']+data_test['ENG_MDL_CD_030060']+data_test['ENG_MDL_CD_210040']+data_test['ENG_MDL_CD_210070']+data_test['ENG_MDL_CD_210050']+data_test['ENG_MDL_CD_210060']+data_test['ENG_MDL_CD_210080']+data_test['ENG_MDL_CD_210020']+data_test['ENG_MDL_CD_140025']+data_test['ENG_MDL_CD_210030']+data_test['ENG_MDL_CD_020026']+data_test['ENG_MDL_CD_110070']+data_test['ENG_MDL_CD_180040']+data_test['ENG_MDL_CD_140045']+data_test['ENG_MDL_CD_140040']+data_test['ENG_MDL_CD_140020']+data_test['ENG_MDL_CD_040020']+data_test['ENG_MDL_CD_160060']+data_test['ENG_MDL_CD_030020']+data_test['ENG_MDL_CD_090040']+data_test['ENG_MDL_CD_080012']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_050020']+data_test['ENG_MDL_CD_165007']+data_test['ENG_MDL_CD_040010'])
# 		data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_150010']+data_test['ENG_MDL_CD_150020']+data_test['ENG_MDL_CD_150060']+data_test['ENG_MDL_CD_210165']+data_test['ENG_MDL_CD_090030']+data_test['ENG_MDL_CD_165030']+data_test['ENG_MDL_CD_180030']+data_test['ENG_MDL_CD_200010']+data_test['ENG_MDL_CD_140195']+data_test['ENG_MDL_CD_140105']+data_test['ENG_MDL_CD_165140']+data_test['ENG_MDL_CD_140210']+data_test['ENG_MDL_CD_080080']+data_test['ENG_MDL_CD_160012']+data_test['ENG_MDL_CD_140065']+data_test['ENG_MDL_CD_090011']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_140205']+data_test['ENG_MDL_CD_030050']+data_test['ENG_MDL_CD_030047']+data_test['ENG_MDL_CD_165090']+data_test['ENG_MDL_CD_140095']+data_test['ENG_MDL_CD_140150']+data_test['ENG_MDL_CD_090070']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_020072']+data_test['ENG_MDL_CD_165010'])
# 		data_test['ENG_MDL_CD_grp4']=(data_test['ENG_MDL_CD_150003']+data_test['ENG_MDL_CD_150150']+data_test['ENG_MDL_CD_150005']+data_test['ENG_MDL_CD_150007'])
# 	elif i==29:
# 		data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_train['MAK_NM_MERCEDES_BENZ']=data_train['MAK_NM_MERCEDES-BENZ']
# 		data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_train['MAK_NM_WESTERN_STAR/AUTO_CAR']
# 		data_train['MAK_NM_PIERCE_MFG__INC_']=data_train['MAK_NM_PIERCE_MFG._INC.']
# 		data_train['MAK_NM_WHITE_GMC']=data_train['MAK_NM_WHITE/GMC']
# 		data_train['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_train['MAK_NM_TRANSPORTATION_MFG_CORP.']
# 		data_train['MAK_NM_TEREX___TEREX_ADVANCE']=data_train['MAK_NM_TEREX_/_TEREX_ADVANCE']
# 		data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# 		data_test['MAK_NM_MERCEDES_BENZ']=data_test['MAK_NM_MERCEDES-BENZ']	
# 		data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_test['MAK_NM_WESTERN_STAR/AUTO_CAR']
# 		data_test['MAK_NM_PIERCE_MFG__INC_']=data_test['MAK_NM_PIERCE_MFG._INC.']
# 		data_test['MAK_NM_WHITE_GMC']=data_test['MAK_NM_WHITE/GMC']
# 		data_test['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_test['MAK_NM_TRANSPORTATION_MFG_CORP.']
# 		data_test['MAK_NM_TEREX___TEREX_ADVANCE']=data_test['MAK_NM_TEREX_/_TEREX_ADVANCE']
		
# 		data_train['NADA_BODY1_grp1']=(data_train['NADA_BODY1_K']+data_train['NADA_BODY1_H']+data_train['NADA_BODY1_B']+data_train['NADA_BODY1_D'])
# 		data_train['NADA_BODY1_grp2']=(data_train['NADA_BODY1_M']+data_train['NADA_BODY1_G']+data_train['NADA_BODY1_I'])
# 		data_train['NADA_BODY1_grp3']=(data_train['NADA_BODY1_N']+data_train['NADA_BODY1_2'])		
# 		data_train['MAK_NM_grp1']=(data_train['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_train['MAK_NM_SEAGRAVE_FIRE_APPARATUS']+data_train['MAK_NM_PREVOST']+data_train['MAK_NM_MERCEDES_BENZ']+data_train['MAK_NM_HYUNDAI']+data_train['MAK_NM_TRANSPORTATION_MFG_CORP_']+data_train['MAK_NM_UNIMOG']+data_train['MAK_NM_DUPLEX']+data_train['MAK_NM_FEDERAL_MOTORS']+data_train['MAK_NM_MOTOR_COACH_INDUSTRIES']+data_train['MAK_NM_GILLIG']+data_train['MAK_NM_DIAMOND_REO']+data_train['MAK_NM_GIANT']+data_train['MAK_NM_LES_AUTOBUS_MCI']+data_train['MAK_NM_ZELIGSON']+data_train['MAK_NM_PIERCE_MFG__INC_']+data_train['MAK_NM_SPARTAN_MOTORS']+data_train['MAK_NM_FWD_CORPORATION']+data_train['MAK_NM_JOHN_DEERE']+data_train['MAK_NM_CAPACITY_OF_TEXAS']+data_train['MAK_NM_OTTAWA'])
# 		data_train['MAK_NM_grp2']=(data_train['MAK_NM_WINNEBAGO']+data_train['MAK_NM_LAFORZA']+data_train['MAK_NM_THOMAS']+data_train['MAK_NM_MARMON_HERRINGTON']+data_train['MAK_NM_IVECO']+data_train['MAK_NM_SCANIA']+data_train['MAK_NM_TEREX___TEREX_ADVANCE']+data_train['MAK_NM_AMERICAN_LA_FRANCE']+data_train['MAK_NM_AUTOCAR']+data_train['MAK_NM_LODAL']+data_train['MAK_NM_WHITE_GMC'])
# 		data_train['MAK_NM_grp3']=(data_train['MAK_NM_INTERNATIONAL']+data_train['MAK_NM_KENWORTH']+data_train['MAK_NM_ISUZU']+data_train['MAK_NM_MACK']+data_train['MAK_NM_NISSAN_DIESEL']+data_train['MAK_NM_FREIGHTLINER']+data_train['MAK_NM_DODGE'])
# 		data_train['MAK_NM_grp4']=(data_train['MAK_NM_HINO']+data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']+data_train['MAK_NM_EMERGENCY_ONE']+data_train['MAK_NM_CRANE_CARRIER']+data_train['MAK_NM_UTILIMASTER']+data_train['MAK_NM_RAM']+data_train['MAK_NM_AUTOCAR_LLC'])
# 		data_train['MAK_NM_grp5']=(data_train['MAK_NM_WORKHORSE_CUSTOM_CHASSIS']+data_train['MAK_NM_ADVANCE_MIXER']+data_train['MAK_NM_BERING']+data_train['MAK_NM_FORETRAVEL_MOTORHOME']+data_train['MAK_NM_CATERPILLAR'])
# 		data_train['MAK_NM_grp6']=(data_train['MAK_NM_INDIANA_PHOENIX']+data_train['MAK_NM_BLUE_BIRD']+data_train['MAK_NM_HENDRICKSON'])
# 		data_train['ENG_FUEL_CD_grp1']=(data_train['ENG_FUEL_CD_U']+data_train['ENG_FUEL_CD_N'])
# 		data_train['ENG_FUEL_CD_grp2']=(data_train['ENG_FUEL_CD_G']+data_train['ENG_FUEL_CD_P'])
# 		data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_Y']+data_train['PLNT_CD_5']+data_train['PLNT_CD_8'])
# 		data_train['PLNT_CD_grp2']=(data_train['PLNT_CD_U']+data_train['PLNT_CD_1']+data_train['PLNT_CD_V'])
# 		data_train['PLNT_CD_grp3']=(data_train['PLNT_CD_M']+data_train['PLNT_CD_W']+data_train['PLNT_CD_F']+data_train['PLNT_CD_D']+data_train['PLNT_CD_K']+data_train['PLNT_CD_7']+data_train['PLNT_CD_S']+data_train['PLNT_CD_R']+data_train['PLNT_CD_C']+data_train['PLNT_CD_L']+data_train['PLNT_CD_2']+data_train['PLNT_CD_T']+data_train['PLNT_CD_4']+data_train['PLNT_CD_0']+data_train['PLNT_CD_9'])
# 		data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_120010']+data_train['ENG_MDL_CD_160030']+data_train['ENG_MDL_CD_110030']+data_train['ENG_MDL_CD_180020']+data_train['ENG_MDL_CD_140220']+data_train['ENG_MDL_CD_180010']+data_train['ENG_MDL_CD_090020']+data_train['ENG_MDL_CD_140167']+data_train['ENG_MDL_CD_140173']+data_train['ENG_MDL_CD_140115']+data_train['ENG_MDL_CD_160020']+data_train['ENG_MDL_CD_020105']+data_train['ENG_MDL_CD_195001']+data_train['ENG_MDL_CD_010038']+data_train['ENG_MDL_CD_165185']+data_train['ENG_MDL_CD_020085']+data_train['ENG_MDL_CD_140168']+data_train['ENG_MDL_CD_165225']+data_train['ENG_MDL_CD_140027']+data_train['ENG_MDL_CD_090077']+data_train['ENG_MDL_CD_180052']+data_train['ENG_MDL_CD_165230']+data_train['ENG_MDL_CD_090043']+data_train['ENG_MDL_CD_090045']+data_train['ENG_MDL_CD_165160']+data_train['ENG_MDL_CD_020060']+data_train['ENG_MDL_CD_010050']+data_train['ENG_MDL_CD_020090']+data_train['ENG_MDL_CD_040030']+data_train['ENG_MDL_CD_030090']+data_train['ENG_MDL_CD_140215']+data_train['ENG_MDL_CD_210100']+data_train['ENG_MDL_CD_020100']+data_train['ENG_MDL_CD_220019']+data_train['ENG_MDL_CD_210090']+data_train['ENG_MDL_CD_140055']+data_train['ENG_MDL_CD_010070']+data_train['ENG_MDL_CD_020120']+data_train['ENG_MDL_CD_140090']+data_train['ENG_MDL_CD_140125']+data_train['ENG_MDL_CD_210110']+data_train['ENG_MDL_CD_140050']+data_train['ENG_MDL_CD_090051']+data_train['ENG_MDL_CD_140234']+data_train['ENG_MDL_CD_245010']+data_train['ENG_MDL_CD_030060']+data_train['ENG_MDL_CD_210040']+data_train['ENG_MDL_CD_210070']+data_train['ENG_MDL_CD_210050']+data_train['ENG_MDL_CD_210060']+data_train['ENG_MDL_CD_210080']+data_train['ENG_MDL_CD_210020']+data_train['ENG_MDL_CD_140025']+data_train['ENG_MDL_CD_210030']+data_train['ENG_MDL_CD_020026']+data_train['ENG_MDL_CD_110070']+data_train['ENG_MDL_CD_180040']+data_train['ENG_MDL_CD_140045']+data_train['ENG_MDL_CD_140040']+data_train['ENG_MDL_CD_140020']+data_train['ENG_MDL_CD_040020']+data_train['ENG_MDL_CD_160060']+data_train['ENG_MDL_CD_030020']+data_train['ENG_MDL_CD_090040']+data_train['ENG_MDL_CD_080012']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_050020']+data_train['ENG_MDL_CD_165007']+data_train['ENG_MDL_CD_040010'])
# 		data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_150010']+data_train['ENG_MDL_CD_150020']+data_train['ENG_MDL_CD_150060']+data_train['ENG_MDL_CD_210165']+data_train['ENG_MDL_CD_090030']+data_train['ENG_MDL_CD_165030']+data_train['ENG_MDL_CD_180030']+data_train['ENG_MDL_CD_200010']+data_train['ENG_MDL_CD_140195']+data_train['ENG_MDL_CD_140105']+data_train['ENG_MDL_CD_165140']+data_train['ENG_MDL_CD_140210']+data_train['ENG_MDL_CD_080080']+data_train['ENG_MDL_CD_160012']+data_train['ENG_MDL_CD_140065']+data_train['ENG_MDL_CD_090011']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_140205']+data_train['ENG_MDL_CD_030050']+data_train['ENG_MDL_CD_030047']+data_train['ENG_MDL_CD_165090']+data_train['ENG_MDL_CD_140095']+data_train['ENG_MDL_CD_140150']+data_train['ENG_MDL_CD_090070']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_020072']+data_train['ENG_MDL_CD_165010'])
# 		data_train['ENG_MDL_CD_grp4']=(data_train['ENG_MDL_CD_150003']+data_train['ENG_MDL_CD_150150']+data_train['ENG_MDL_CD_150005']+data_train['ENG_MDL_CD_150007'])

# 		data_test['NADA_BODY1_grp1']=(data_test['NADA_BODY1_K']+data_test['NADA_BODY1_H']+data_test['NADA_BODY1_B']+data_test['NADA_BODY1_D'])
# 		data_test['NADA_BODY1_grp2']=(data_test['NADA_BODY1_M']+data_test['NADA_BODY1_G']+data_test['NADA_BODY1_I'])
# 		data_test['NADA_BODY1_grp3']=(data_test['NADA_BODY1_N']+data_test['NADA_BODY1_2'])		
# 		data_test['MAK_NM_grp1']=(data_test['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_test['MAK_NM_SEAGRAVE_FIRE_APPARATUS']+data_test['MAK_NM_PREVOST']+data_test['MAK_NM_MERCEDES_BENZ']+data_test['MAK_NM_HYUNDAI']+data_test['MAK_NM_TRANSPORTATION_MFG_CORP_']+data_test['MAK_NM_UNIMOG']+data_test['MAK_NM_DUPLEX']+data_test['MAK_NM_FEDERAL_MOTORS']+data_test['MAK_NM_MOTOR_COACH_INDUSTRIES']+data_test['MAK_NM_GILLIG']+data_test['MAK_NM_DIAMOND_REO']+data_test['MAK_NM_GIANT']+data_test['MAK_NM_LES_AUTOBUS_MCI']+data_test['MAK_NM_ZELIGSON']+data_test['MAK_NM_PIERCE_MFG__INC_']+data_test['MAK_NM_SPARTAN_MOTORS']+data_test['MAK_NM_FWD_CORPORATION']+data_test['MAK_NM_JOHN_DEERE']+data_test['MAK_NM_CAPACITY_OF_TEXAS']+data_test['MAK_NM_OTTAWA'])
# 		data_test['MAK_NM_grp2']=(data_test['MAK_NM_WINNEBAGO']+data_test['MAK_NM_LAFORZA']+data_test['MAK_NM_THOMAS']+data_test['MAK_NM_MARMON_HERRINGTON']+data_test['MAK_NM_IVECO']+data_test['MAK_NM_SCANIA']+data_test['MAK_NM_TEREX___TEREX_ADVANCE']+data_test['MAK_NM_AMERICAN_LA_FRANCE']+data_test['MAK_NM_AUTOCAR']+data_test['MAK_NM_LODAL']+data_test['MAK_NM_WHITE_GMC'])
# 		data_test['MAK_NM_grp3']=(data_test['MAK_NM_INTERNATIONAL']+data_test['MAK_NM_KENWORTH']+data_test['MAK_NM_ISUZU']+data_test['MAK_NM_MACK']+data_test['MAK_NM_NISSAN_DIESEL']+data_test['MAK_NM_FREIGHTLINER']+data_test['MAK_NM_DODGE'])
# 		data_test['MAK_NM_grp4']=(data_test['MAK_NM_HINO']+data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']+data_test['MAK_NM_EMERGENCY_ONE']+data_test['MAK_NM_CRANE_CARRIER']+data_test['MAK_NM_UTILIMASTER']+data_test['MAK_NM_RAM']+data_test['MAK_NM_AUTOCAR_LLC'])
# 		data_test['MAK_NM_grp5']=(data_test['MAK_NM_WORKHORSE_CUSTOM_CHASSIS']+data_test['MAK_NM_ADVANCE_MIXER']+data_test['MAK_NM_BERING']+data_test['MAK_NM_FORETRAVEL_MOTORHOME']+data_test['MAK_NM_CATERPILLAR'])
# 		data_test['MAK_NM_grp6']=(data_test['MAK_NM_INDIANA_PHOENIX']+data_test['MAK_NM_BLUE_BIRD']+data_test['MAK_NM_HENDRICKSON'])
# 		data_test['ENG_FUEL_CD_grp1']=(data_test['ENG_FUEL_CD_U']+data_test['ENG_FUEL_CD_N'])
# 		data_test['ENG_FUEL_CD_grp2']=(data_test['ENG_FUEL_CD_G']+data_test['ENG_FUEL_CD_P'])
# 		data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_Y']+data_test['PLNT_CD_5']+data_test['PLNT_CD_8'])
# 		data_test['PLNT_CD_grp2']=(data_test['PLNT_CD_U']+data_test['PLNT_CD_1']+data_test['PLNT_CD_V'])
# 		data_test['PLNT_CD_grp3']=(data_test['PLNT_CD_M']+data_test['PLNT_CD_W']+data_test['PLNT_CD_F']+data_test['PLNT_CD_D']+data_test['PLNT_CD_K']+data_test['PLNT_CD_7']+data_test['PLNT_CD_S']+data_test['PLNT_CD_R']+data_test['PLNT_CD_C']+data_test['PLNT_CD_L']+data_test['PLNT_CD_2']+data_test['PLNT_CD_T']+data_test['PLNT_CD_4']+data_test['PLNT_CD_0']+data_test['PLNT_CD_9'])
# 		data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_120010']+data_test['ENG_MDL_CD_160030']+data_test['ENG_MDL_CD_110030']+data_test['ENG_MDL_CD_180020']+data_test['ENG_MDL_CD_140220']+data_test['ENG_MDL_CD_180010']+data_test['ENG_MDL_CD_090020']+data_test['ENG_MDL_CD_140167']+data_test['ENG_MDL_CD_140173']+data_test['ENG_MDL_CD_140115']+data_test['ENG_MDL_CD_160020']+data_test['ENG_MDL_CD_020105']+data_test['ENG_MDL_CD_195001']+data_test['ENG_MDL_CD_010038']+data_test['ENG_MDL_CD_165185']+data_test['ENG_MDL_CD_020085']+data_test['ENG_MDL_CD_140168']+data_test['ENG_MDL_CD_165225']+data_test['ENG_MDL_CD_140027']+data_test['ENG_MDL_CD_090077']+data_test['ENG_MDL_CD_180052']+data_test['ENG_MDL_CD_165230']+data_test['ENG_MDL_CD_090043']+data_test['ENG_MDL_CD_090045']+data_test['ENG_MDL_CD_165160']+data_test['ENG_MDL_CD_020060']+data_test['ENG_MDL_CD_010050']+data_test['ENG_MDL_CD_020090']+data_test['ENG_MDL_CD_040030']+data_test['ENG_MDL_CD_030090']+data_test['ENG_MDL_CD_140215']+data_test['ENG_MDL_CD_210100']+data_test['ENG_MDL_CD_020100']+data_test['ENG_MDL_CD_220019']+data_test['ENG_MDL_CD_210090']+data_test['ENG_MDL_CD_140055']+data_test['ENG_MDL_CD_010070']+data_test['ENG_MDL_CD_020120']+data_test['ENG_MDL_CD_140090']+data_test['ENG_MDL_CD_140125']+data_test['ENG_MDL_CD_210110']+data_test['ENG_MDL_CD_140050']+data_test['ENG_MDL_CD_090051']+data_test['ENG_MDL_CD_140234']+data_test['ENG_MDL_CD_245010']+data_test['ENG_MDL_CD_030060']+data_test['ENG_MDL_CD_210040']+data_test['ENG_MDL_CD_210070']+data_test['ENG_MDL_CD_210050']+data_test['ENG_MDL_CD_210060']+data_test['ENG_MDL_CD_210080']+data_test['ENG_MDL_CD_210020']+data_test['ENG_MDL_CD_140025']+data_test['ENG_MDL_CD_210030']+data_test['ENG_MDL_CD_020026']+data_test['ENG_MDL_CD_110070']+data_test['ENG_MDL_CD_180040']+data_test['ENG_MDL_CD_140045']+data_test['ENG_MDL_CD_140040']+data_test['ENG_MDL_CD_140020']+data_test['ENG_MDL_CD_040020']+data_test['ENG_MDL_CD_160060']+data_test['ENG_MDL_CD_030020']+data_test['ENG_MDL_CD_090040']+data_test['ENG_MDL_CD_080012']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_050020']+data_test['ENG_MDL_CD_165007']+data_test['ENG_MDL_CD_040010'])
# 		data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_150010']+data_test['ENG_MDL_CD_150020']+data_test['ENG_MDL_CD_150060']+data_test['ENG_MDL_CD_210165']+data_test['ENG_MDL_CD_090030']+data_test['ENG_MDL_CD_165030']+data_test['ENG_MDL_CD_180030']+data_test['ENG_MDL_CD_200010']+data_test['ENG_MDL_CD_140195']+data_test['ENG_MDL_CD_140105']+data_test['ENG_MDL_CD_165140']+data_test['ENG_MDL_CD_140210']+data_test['ENG_MDL_CD_080080']+data_test['ENG_MDL_CD_160012']+data_test['ENG_MDL_CD_140065']+data_test['ENG_MDL_CD_090011']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_140205']+data_test['ENG_MDL_CD_030050']+data_test['ENG_MDL_CD_030047']+data_test['ENG_MDL_CD_165090']+data_test['ENG_MDL_CD_140095']+data_test['ENG_MDL_CD_140150']+data_test['ENG_MDL_CD_090070']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_020072']+data_test['ENG_MDL_CD_165010'])
# 		data_test['ENG_MDL_CD_grp4']=(data_test['ENG_MDL_CD_150003']+data_test['ENG_MDL_CD_150150']+data_test['ENG_MDL_CD_150005']+data_test['ENG_MDL_CD_150007'])


# 	formula=formulabase+forms[i]
# 	if not os.path.exists('glm'+str(i)+'/'):
# 		os.makedirs('glm'+str(i)+'/')
# 	prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm'+str(i)+'/',0)


# # Step 6
# # Finalize GLM

# CONTFEATURES = []
# CATFEATURES = ['ENG_MDL_CD','TRK_CAB_CNFG_CD','NADA_BODY1','MAK_NM','ENG_FUEL_CD','PLNT_CD','ENG_TRK_DUTY_TYP_CD','ENG_ASP_SUP_CHGR_CD','ENG_MFG_CD','cko_eng_cylinders']
# # Include only catfeatures needed for GLM

# DUMMYFEATURES=[]
# for feature in CATFEATURES:
# 	dummiesTrain=pd.get_dummies(data_train[feature],prefix=feature)
# 	dummiesTest=pd.get_dummies(data_test[feature],prefix=feature)
# 	temp=dummiesTrain.sum()	
# 	# If column appears in data_train but not data_test we need to add it to data_test as a column of zeroes
# 	for k in ml.returnNotMatches(dummiesTrain.columns.values,dummiesTest.columns.values)[0]:
# 		dummiesTest[k]=0	
# 	dummiesTrain=dummiesTrain[temp[temp<temp.max()].index.values]
# 	dummiesTest=dummiesTest[temp[temp<temp.max()].index.values]
# 	if(len(dummiesTrain.columns)>0):
# 		dummiesTrain.columns = dummiesTrain.columns.str.replace('\s+', '_')
# 		dummiesTest.columns = dummiesTest.columns.str.replace('\s+', '_')
# 		data_train=pd.concat([data_train,dummiesTrain],axis=1)
# 		data_test=pd.concat([data_test,dummiesTest],axis=1)
# 		DUMMYFEATURES += list(dummiesTrain)

# # print(data_train.groupby(['TRK_CAB_CNFG_CD'], as_index=False)[WEIGHT].sum())

# # Create custom variables for GLM
# data_train['TRK_GRSS_VEH_WGHT_RATG_CD_gt6']=np.where(data_train['TRK_GRSS_VEH_WGHT_RATG_CD']>6, 1, 0)
# data_test['TRK_GRSS_VEH_WGHT_RATG_CD_gt6'] =np.where( data_test['TRK_GRSS_VEH_WGHT_RATG_CD']>6, 1, 0)
# # data_train['cko_wheelbase_mean_lt130']=np.where(data_train['cko_wheelbase_mean']<130,1,0)
# # data_train['cko_max_gvwc_lt11k']=np.where(data_train['cko_max_gvwc']<11000,1,0)
# # data_test['cko_wheelbase_mean_lt130']=np.where(data_test['cko_wheelbase_mean']<130,1,0)
# # data_test['cko_max_gvwc_lt11k']=np.where(data_test['cko_max_gvwc']<11000,1,0)
# data_train['TRK_CAB_CNFG_CD_CONSUVSPE']=(data_train['TRK_CAB_CNFG_CD_CON']+data_train['TRK_CAB_CNFG_CD_SUV']+data_train['TRK_CAB_CNFG_CD_SPE'])
# data_test['TRK_CAB_CNFG_CD_CONSUVSPE']=(data_test['TRK_CAB_CNFG_CD_CON']+data_test['TRK_CAB_CNFG_CD_SUV']+data_test['TRK_CAB_CNFG_CD_SPE'])
# # data_train['TRK_CAB_CNFG_CD_CRWTHICFWHCB']=(data_train['TRK_CAB_CNFG_CD_CRW']+data_train['TRK_CAB_CNFG_CD_THI']+data_train['TRK_CAB_CNFG_CD_CFW']+data_train['TRK_CAB_CNFG_CD_HCB'])
# # data_test['TRK_CAB_CNFG_CD_CRWTHICFWHCB']=(data_test['TRK_CAB_CNFG_CD_CRW']+data_test['TRK_CAB_CNFG_CD_THI']+data_test['TRK_CAB_CNFG_CD_CFW']+data_test['TRK_CAB_CNFG_CD_HCB'])
# # data_train['cko_max_gvwc_lt12k']=np.where((data_train['cko_max_gvwc']>0) & (data_train['cko_max_gvwc']<12000),1,0)
# # data_test['cko_max_gvwc_lt12k']=np.where((data_test['cko_max_gvwc']>0) & (data_test['cko_max_gvwc']<12000),1,0)
# data_train['NADA_GVWC2_lt14k']=np.where((data_train['NADA_GVWC2']>0) & (data_train['NADA_GVWC2']<14000),1,0)
# data_test['NADA_GVWC2_lt14k']=np.where((data_test['NADA_GVWC2']>0) & (data_test['NADA_GVWC2']<14000),1,0)
# # data_train['cko_eng_cylinders_eq6']=np.where(data_train['cko_eng_cylinders']==6, 1, 0)
# # data_test['cko_eng_cylinders_eq6']=np.where(data_test['cko_eng_cylinders']==6, 1, 0)
# # data_train['cko_eng_cylinders_numeric_eq6']=np.where(data_train['cko_eng_cylinders_numeric']==6, 1, 0)
# # data_test['cko_eng_cylinders_numeric_eq6']=np.where(data_test['cko_eng_cylinders_numeric']==6, 1, 0)

# data_train['cko_eng_cylinders_grp1']=(data_train['cko_eng_cylinders_.']+data_train['cko_eng_cylinders_4']+data_train['cko_eng_cylinders_5']+data_train['cko_eng_cylinders_8'])
# data_test['cko_eng_cylinders_grp1']=(data_test['cko_eng_cylinders_.']+data_test['cko_eng_cylinders_4']+data_test['cko_eng_cylinders_5']+data_test['cko_eng_cylinders_8'])

# # data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# # data_train['MAK_NM_MERCEDES_BENZ']=data_train['MAK_NM_MERCEDES-BENZ']
# # data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_train['MAK_NM_WESTERN_STAR/AUTO_CAR']
# # data_train['MAK_NM_PIERCE_MFG__INC_']=data_train['MAK_NM_PIERCE_MFG._INC.']
# # data_train['MAK_NM_WHITE_GMC']=data_train['MAK_NM_WHITE/GMC']
# # data_train['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_train['MAK_NM_TRANSPORTATION_MFG_CORP.']
# # data_train['MAK_NM_TEREX___TEREX_ADVANCE']=data_train['MAK_NM_TEREX_/_TEREX_ADVANCE']
# # data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
# # data_test['MAK_NM_MERCEDES_BENZ']=data_test['MAK_NM_MERCEDES-BENZ']	
# # data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_test['MAK_NM_WESTERN_STAR/AUTO_CAR']
# # data_test['MAK_NM_PIERCE_MFG__INC_']=data_test['MAK_NM_PIERCE_MFG._INC.']
# # data_test['MAK_NM_WHITE_GMC']=data_test['MAK_NM_WHITE/GMC']
# # data_test['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_test['MAK_NM_TRANSPORTATION_MFG_CORP.']
# # data_test['MAK_NM_TEREX___TEREX_ADVANCE']=data_test['MAK_NM_TEREX_/_TEREX_ADVANCE']

# data_train['NADA_BODY1_grp1']=(data_train['NADA_BODY1_K']+data_train['NADA_BODY1_H']+data_train['NADA_BODY1_B']+data_train['NADA_BODY1_D'])
# # data_train['NADA_BODY1_grp2']=(data_train['NADA_BODY1_M']+data_train['NADA_BODY1_G']+data_train['NADA_BODY1_I'])
# # data_train['NADA_BODY1_grp3']=(data_train['NADA_BODY1_N']+data_train['NADA_BODY1_2'])		
# data_train['MAK_NM_grp1']=(data_train['MAK_NM_LES_AUTOBUS_MCI']+data_train['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_train['MAK_NM_HYUNDAI']+data_train['MAK_NM_UNIMOG']+data_train['MAK_NM_SPARTAN_MOTORS']+data_train['MAK_NM_FWD_CORPORATION']+data_train['MAK_NM_ZELIGSON']+data_train['MAK_NM_JOHN_DEERE']+data_train['MAK_NM_FEDERAL_MOTORS']+data_train['MAK_NM_DIAMOND_REO']+data_train['MAK_NM_GILLIG']+data_train['MAK_NM_PREVOST']+data_train['MAK_NM_OTTAWA']+data_train['MAK_NM_DUPLEX']+data_train['MAK_NM_MERCEDES-BENZ'])
# data_train['MAK_NM_grp2']=(data_train['MAK_NM_WORKHORSE_CUSTOM_CHASSIS']+data_train['MAK_NM_CATERPILLAR']+data_train['MAK_NM_ADVANCE_MIXER']+data_train['MAK_NM_EMERGENCY_ONE']+data_train['MAK_NM_FORETRAVEL_MOTORHOME']+data_train['MAK_NM_BLUE_BIRD']+data_train['MAK_NM_INDIANA_PHOENIX']+data_train['MAK_NM_HENDRICKSON'])
# data_train['ENG_FUEL_CD_grp1']=(data_train['ENG_FUEL_CD_U']+data_train['ENG_FUEL_CD_N'])
# # data_train['ENG_FUEL_CD_grp2']=(data_train['ENG_FUEL_CD_G']+data_train['ENG_FUEL_CD_P'])
# data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_Y']+data_train['PLNT_CD_5']+data_train['PLNT_CD_8'])
# data_train['PLNT_CD_grp2']=(data_train['PLNT_CD_U']+data_train['PLNT_CD_1']+data_train['PLNT_CD_V'])
# data_train['PLNT_CD_grp3']=(data_train['PLNT_CD_M']+data_train['PLNT_CD_W']+data_train['PLNT_CD_F']+data_train['PLNT_CD_D']+data_train['PLNT_CD_K']+data_train['PLNT_CD_7']+data_train['PLNT_CD_S']+data_train['PLNT_CD_R']+data_train['PLNT_CD_C']+data_train['PLNT_CD_L']+data_train['PLNT_CD_2']+data_train['PLNT_CD_T']+data_train['PLNT_CD_4']+data_train['PLNT_CD_0']+data_train['PLNT_CD_9'])
# data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_120010']+data_train['ENG_MDL_CD_160030']+data_train['ENG_MDL_CD_110030']+data_train['ENG_MDL_CD_180020']+data_train['ENG_MDL_CD_140220']+data_train['ENG_MDL_CD_180010']+data_train['ENG_MDL_CD_090020']+data_train['ENG_MDL_CD_140167']+data_train['ENG_MDL_CD_140173']+data_train['ENG_MDL_CD_140115']+data_train['ENG_MDL_CD_160020']+data_train['ENG_MDL_CD_020105']+data_train['ENG_MDL_CD_195001']+data_train['ENG_MDL_CD_010038']+data_train['ENG_MDL_CD_165185']+data_train['ENG_MDL_CD_020085']+data_train['ENG_MDL_CD_140168']+data_train['ENG_MDL_CD_165225']+data_train['ENG_MDL_CD_140027']+data_train['ENG_MDL_CD_090077']+data_train['ENG_MDL_CD_180052']+data_train['ENG_MDL_CD_165230']+data_train['ENG_MDL_CD_090043']+data_train['ENG_MDL_CD_090045']+data_train['ENG_MDL_CD_165160']+data_train['ENG_MDL_CD_020060']+data_train['ENG_MDL_CD_010050']+data_train['ENG_MDL_CD_020090']+data_train['ENG_MDL_CD_040030']+data_train['ENG_MDL_CD_140215']+data_train['ENG_MDL_CD_210100']+data_train['ENG_MDL_CD_020100']+data_train['ENG_MDL_CD_220019']+data_train['ENG_MDL_CD_210090']+data_train['ENG_MDL_CD_140055']+data_train['ENG_MDL_CD_010070']+data_train['ENG_MDL_CD_020120']+data_train['ENG_MDL_CD_140090']+data_train['ENG_MDL_CD_140125']+data_train['ENG_MDL_CD_210110']+data_train['ENG_MDL_CD_140050']+data_train['ENG_MDL_CD_090051']+data_train['ENG_MDL_CD_140234']+data_train['ENG_MDL_CD_245010']+data_train['ENG_MDL_CD_030060']+data_train['ENG_MDL_CD_210040']+data_train['ENG_MDL_CD_210070']+data_train['ENG_MDL_CD_210050']+data_train['ENG_MDL_CD_210060']+data_train['ENG_MDL_CD_210080']+data_train['ENG_MDL_CD_210020']+data_train['ENG_MDL_CD_140025']+data_train['ENG_MDL_CD_210030']+data_train['ENG_MDL_CD_020026']+data_train['ENG_MDL_CD_110070']+data_train['ENG_MDL_CD_180040']+data_train['ENG_MDL_CD_140040']+data_train['ENG_MDL_CD_140020']+data_train['ENG_MDL_CD_160060']+data_train['ENG_MDL_CD_030020']+data_train['ENG_MDL_CD_090040']+data_train['ENG_MDL_CD_080012']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_050020']+data_train['ENG_MDL_CD_165007'])
# data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_150010']+data_train['ENG_MDL_CD_150020']+data_train['ENG_MDL_CD_150060']+data_train['ENG_MDL_CD_210165']+data_train['ENG_MDL_CD_090030']+data_train['ENG_MDL_CD_165030']+data_train['ENG_MDL_CD_180030']+data_train['ENG_MDL_CD_200010']+data_train['ENG_MDL_CD_140195']+data_train['ENG_MDL_CD_140105']+data_train['ENG_MDL_CD_165140']+data_train['ENG_MDL_CD_140210']+data_train['ENG_MDL_CD_080080']+data_train['ENG_MDL_CD_160012']+data_train['ENG_MDL_CD_140065']+data_train['ENG_MDL_CD_090011']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_140205']+data_train['ENG_MDL_CD_030050']+data_train['ENG_MDL_CD_030047']+data_train['ENG_MDL_CD_165090']+data_train['ENG_MDL_CD_140095']+data_train['ENG_MDL_CD_140150']+data_train['ENG_MDL_CD_090070']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_020072']+data_train['ENG_MDL_CD_165010'])
# data_train['ENG_MDL_CD_grp3']=(data_train['ENG_MDL_CD_150003']+data_train['ENG_MDL_CD_150150']+data_train['ENG_MDL_CD_150005']+data_train['ENG_MDL_CD_150007'])

# data_test['NADA_BODY1_grp1']=(data_test['NADA_BODY1_K']+data_test['NADA_BODY1_H']+data_test['NADA_BODY1_B']+data_test['NADA_BODY1_D'])
# # data_test['NADA_BODY1_grp2']=(data_test['NADA_BODY1_M']+data_test['NADA_BODY1_G']+data_test['NADA_BODY1_I'])
# # data_test['NADA_BODY1_grp3']=(data_test['NADA_BODY1_N']+data_test['NADA_BODY1_2'])		
# data_test['MAK_NM_grp1']=(data_test['MAK_NM_LES_AUTOBUS_MCI']+data_test['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_test['MAK_NM_HYUNDAI']+data_test['MAK_NM_UNIMOG']+data_test['MAK_NM_SPARTAN_MOTORS']+data_test['MAK_NM_FWD_CORPORATION']+data_test['MAK_NM_ZELIGSON']+data_test['MAK_NM_JOHN_DEERE']+data_test['MAK_NM_FEDERAL_MOTORS']+data_test['MAK_NM_DIAMOND_REO']+data_test['MAK_NM_GILLIG']+data_test['MAK_NM_PREVOST']+data_test['MAK_NM_OTTAWA']+data_test['MAK_NM_DUPLEX']+data_test['MAK_NM_MERCEDES-BENZ'])
# data_test['MAK_NM_grp2']=(data_test['MAK_NM_WORKHORSE_CUSTOM_CHASSIS']+data_test['MAK_NM_CATERPILLAR']+data_test['MAK_NM_ADVANCE_MIXER']+data_test['MAK_NM_EMERGENCY_ONE']+data_test['MAK_NM_FORETRAVEL_MOTORHOME']+data_test['MAK_NM_BLUE_BIRD']+data_test['MAK_NM_INDIANA_PHOENIX']+data_test['MAK_NM_HENDRICKSON'])
# data_test['ENG_FUEL_CD_grp1']=(data_test['ENG_FUEL_CD_U']+data_test['ENG_FUEL_CD_N'])
# # data_test['ENG_FUEL_CD_grp2']=(data_test['ENG_FUEL_CD_G']+data_test['ENG_FUEL_CD_P'])
# data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_Y']+data_test['PLNT_CD_5']+data_test['PLNT_CD_8'])
# data_test['PLNT_CD_grp2']=(data_test['PLNT_CD_U']+data_test['PLNT_CD_1']+data_test['PLNT_CD_V'])
# data_test['PLNT_CD_grp3']=(data_test['PLNT_CD_M']+data_test['PLNT_CD_W']+data_test['PLNT_CD_F']+data_test['PLNT_CD_D']+data_test['PLNT_CD_K']+data_test['PLNT_CD_7']+data_test['PLNT_CD_S']+data_test['PLNT_CD_R']+data_test['PLNT_CD_C']+data_test['PLNT_CD_L']+data_test['PLNT_CD_2']+data_test['PLNT_CD_T']+data_test['PLNT_CD_4']+data_test['PLNT_CD_0']+data_test['PLNT_CD_9'])
# data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_120010']+data_test['ENG_MDL_CD_160030']+data_test['ENG_MDL_CD_110030']+data_test['ENG_MDL_CD_180020']+data_test['ENG_MDL_CD_140220']+data_test['ENG_MDL_CD_180010']+data_test['ENG_MDL_CD_090020']+data_test['ENG_MDL_CD_140167']+data_test['ENG_MDL_CD_140173']+data_test['ENG_MDL_CD_140115']+data_test['ENG_MDL_CD_160020']+data_test['ENG_MDL_CD_020105']+data_test['ENG_MDL_CD_195001']+data_test['ENG_MDL_CD_010038']+data_test['ENG_MDL_CD_165185']+data_test['ENG_MDL_CD_020085']+data_test['ENG_MDL_CD_140168']+data_test['ENG_MDL_CD_165225']+data_test['ENG_MDL_CD_140027']+data_test['ENG_MDL_CD_090077']+data_test['ENG_MDL_CD_180052']+data_test['ENG_MDL_CD_165230']+data_test['ENG_MDL_CD_090043']+data_test['ENG_MDL_CD_090045']+data_test['ENG_MDL_CD_165160']+data_test['ENG_MDL_CD_020060']+data_test['ENG_MDL_CD_010050']+data_test['ENG_MDL_CD_020090']+data_test['ENG_MDL_CD_040030']+data_test['ENG_MDL_CD_140215']+data_test['ENG_MDL_CD_210100']+data_test['ENG_MDL_CD_020100']+data_test['ENG_MDL_CD_220019']+data_test['ENG_MDL_CD_210090']+data_test['ENG_MDL_CD_140055']+data_test['ENG_MDL_CD_010070']+data_test['ENG_MDL_CD_020120']+data_test['ENG_MDL_CD_140090']+data_test['ENG_MDL_CD_140125']+data_test['ENG_MDL_CD_210110']+data_test['ENG_MDL_CD_140050']+data_test['ENG_MDL_CD_090051']+data_test['ENG_MDL_CD_140234']+data_test['ENG_MDL_CD_245010']+data_test['ENG_MDL_CD_030060']+data_test['ENG_MDL_CD_210040']+data_test['ENG_MDL_CD_210070']+data_test['ENG_MDL_CD_210050']+data_test['ENG_MDL_CD_210060']+data_test['ENG_MDL_CD_210080']+data_test['ENG_MDL_CD_210020']+data_test['ENG_MDL_CD_140025']+data_test['ENG_MDL_CD_210030']+data_test['ENG_MDL_CD_020026']+data_test['ENG_MDL_CD_110070']+data_test['ENG_MDL_CD_180040']+data_test['ENG_MDL_CD_140040']+data_test['ENG_MDL_CD_140020']+data_test['ENG_MDL_CD_160060']+data_test['ENG_MDL_CD_030020']+data_test['ENG_MDL_CD_090040']+data_test['ENG_MDL_CD_080012']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_050020']+data_test['ENG_MDL_CD_165007'])
# data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_150010']+data_test['ENG_MDL_CD_150020']+data_test['ENG_MDL_CD_150060']+data_test['ENG_MDL_CD_210165']+data_test['ENG_MDL_CD_090030']+data_test['ENG_MDL_CD_165030']+data_test['ENG_MDL_CD_180030']+data_test['ENG_MDL_CD_200010']+data_test['ENG_MDL_CD_140195']+data_test['ENG_MDL_CD_140105']+data_test['ENG_MDL_CD_165140']+data_test['ENG_MDL_CD_140210']+data_test['ENG_MDL_CD_080080']+data_test['ENG_MDL_CD_160012']+data_test['ENG_MDL_CD_140065']+data_test['ENG_MDL_CD_090011']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_140205']+data_test['ENG_MDL_CD_030050']+data_test['ENG_MDL_CD_030047']+data_test['ENG_MDL_CD_165090']+data_test['ENG_MDL_CD_140095']+data_test['ENG_MDL_CD_140150']+data_test['ENG_MDL_CD_090070']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_020072']+data_test['ENG_MDL_CD_165010'])
# data_test['ENG_MDL_CD_grp3']=(data_test['ENG_MDL_CD_150003']+data_test['ENG_MDL_CD_150150']+data_test['ENG_MDL_CD_150005']+data_test['ENG_MDL_CD_150007'])

# if not os.path.exists('glm/'):
# 	os.makedirs('glm/')
# formula= 'lossratio ~  TRK_GRSS_VEH_WGHT_RATG_CD_gt6   +  TRK_CAB_CNFG_CD_CONSUVSPE + TRK_CAB_CNFG_CD_CRW  + cko_eng_cylinders_grp1 + ENG_TRK_DUTY_TYP_CD_HV  +  NADA_BODY1_U + NADA_BODY1_grp1   + ENG_FUEL_CD_grp1  + PLNT_CD_grp1 + PLNT_CD_grp2 + PLNT_CD_grp3  + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 + MAK_NM_grp1 + MAK_NM_grp2  + cko_max_msrpMS  + cko_wheelbase_maxMS + ENG_ASP_SUP_CHGR_CD_N + ENG_MFG_CD_030  + MFG_BAS_MSRPMS + NADA_GVWC3MS  + WHL_BAS_LNGST_INCHSMS + NADA_GVWC2_lt14k'
# # Trying + cko_overdrive + NADA_GVWC2_lt14k  + turbo_super + cko_max_gvwc_lt12k
# prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm/',1)

# # data_train['weight'] = data_train[WEIGHT]
# # data_train['predictedValue'] = prediction_glm_train*data_train['weight']
# # data_train['actualValue'] = data_train[LABEL]*data_train['weight']
# # data_test['weight'] = data_test[WEIGHT]
# # data_test['predictedValue'] = prediction_glm_test*data_test['weight']
# # data_test['actualValue'] = data_test[LABEL]*data_test['weight']

# # ml.actualvsfittedbyfactor(data_train,data_test,'TRK_GRSS_VEH_WGHT_RATG_CD_gt6','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'NADA_GVWC2_lt14k','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'TRK_CAB_CNFG_CD_CONSUVSPE','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'NADA_BODY1_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'ENG_FUEL_CD_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'PLNT_CD_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'PLNT_CD_grp2','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'PLNT_CD_grp3','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'ENG_MDL_CD_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'ENG_MDL_CD_grp2','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'ENG_MDL_CD_grp4','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'MAK_NM_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'MAK_NM_grp2','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'cko_eng_cylinders_grp1','glm/')




# # ALLCONTFEATURES = cl.returnCONTFEATURES()
# # ALLCATFEATURES = cl.returnCATFEATURES()

# # for feature in ALLCONTFEATURES:
# # 	print(feature)
# # 	cutoffs=BANDFEATURES[feature]
# # 	data_train[feature+'Band']=data_train[feature].apply(ml.applyband,args=(cutoffs,))
# # 	data_test[feature+'Band']=data_test[feature].apply(ml.applyband,args=(cutoffs,))
# # 	ml.actualvsfittedbyfactor(data_train,data_test,feature+'Band','glm/')

# # for feature in ALLCATFEATURES:
# # 	ml.actualvsfittedbyfactor(data_train,data_test,feature,'glm/')		


# SAS CODE

# IF TRK_GRSS_VEH_WGHT_RATG_CD>6 THEN TRK_GRSS_VEH_WGHT_RATG_CD_gt6=2; ELSE TRK_GRSS_VEH_WGHT_RATG_CD_gt6=0;
# IF TRK_CAB_CNFG_CD IN ('CON','SUV','SPE') THEN TRK_CAB_CNFG_CD_CONSUVSPE=1; ELSE TRK_CAB_CNFG_CD_CONSUVSPE=0; 
# IF TRK_CAB_CNFG_CD='CRW' THEN TRK_CAB_CNFG_CD_CRW=1; ELSE TRK_CAB_CNFG_CD_CRW=0;
# IF cko_eng_cylinders IN ('.','4','5','8') THEN cko_eng_cylinders_grp1=1; ELSE cko_eng_cylinders_grp1=0; 
# IF ENG_TRK_DUTY_TYP_CD = 'HV' THEN ENG_TRK_DUTY_TYP_CD_HV=1; ELSE ENG_TRK_DUTY_TYP_CD_HV=0;
# IF NADA_BODY1 = 'U' THEN NADA_BODY1_U=1; ELSE NADA_BODY1_U=0;
# IF NADA_BODY1 IN ('K','H','B','D') THEN NADA_BODY1_grp1=1; ELSE NADA_BODY1_grp1=0;
# IF ENG_FUEL_CD IN ('U','N') THEN ENG_FUEL_CD_grp1=1; ELSE ENG_FUEL_CD_grp1=0;
# IF PLNT_CD IN ('Y','5','8') THEN PLNT_CD_grp1=1; ELSE PLNT_CD_grp1=0; 
# IF PLNT_CD IN ('U','1','V') THEN PLNT_CD_grp2=1; ELSE PLNT_CD_grp2=0;
# IF PLNT_CD IN ('M','W','F','D','K','7','S','R','C','L','2','T','4','0','9') THEN PLNT_CD_grp3=1; ELSE PLNT_CD_grp3=0;
# IF ENG_MDL_CD IN ('120010','160030','110030','180020','140220','180010','090020','140167','140173','140115','160020','020105','195001','010038','165185','020085','140168','165225','140027','090077','180052','165230','090043','090045','165160','020060','010050','020090','040030','140215','210100','020100','220019','210090','140055','010070','020120','140090','140125','210110','140050','090051','140234','245010','030060','210040','210070','210050','210060','210080','210020','140025','210030','020026','110070','180040','140040','140020','160060','030020','090040','080012','120020','050020','165007') THEN ENG_MDL_CD_grp1=1; ELSE ENG_MDL_CD_grp1=0;
# IF ENG_MDL_CD IN ('150010','150020','150060','210165','090030','165030','180030','200010','140195','140105','165140','140210','080080','160012','140065','090011','120030','140205','030050','030047','165090','140095','140150','090070','070020','020072','165010') THEN ENG_MDL_CD_grp2=1; ELSE ENG_MDL_CD_grp2=0;
# IF ENG_MDL_CD IN ('150003','150150','150005','150007') THEN ENG_MDL_CD_grp4=1; ELSE ENG_MDL_CD_grp3=0;
# IF MAK_NM IN ('LES AUTOBUS MCI','CHANCE COACH TRANSIT BUS','HYUNDAI','UNIMOG','SPARTAN MOTORS','FWD CORPORATION','ZELIGSON','JOHN DEERE','FEDERAL MOTORS','DIAMOND REO','GILLIG','PREVOST','OTTAWA','DUPLEX','MERCEDES-BENZ') THEN MAK_NM_grp1=1; ELSE MAK_NM_grp1=0;
# IF MAK_NM IN ('WORKHORSE CUSTOM CHASSIS','CATERPILLAR','ADVANCE MIXER','EMERGENCY ONE','FORETRAVEL MOTORHOME','BLUE BIRD','INDIANA PHOENIX','HENDRICKSON') THEN MAK_NM_grp2=1; ELSE MAK_NM_grp2=0;
# IF cko_max_msrp = . THEN cko_max_msrpMS=1; ELSE cko_max_msrpMS=0;
# IF cko_wheelbase_max =. THEN cko_wheelbase_maxMS=1; ELSE cko_wheelbase_maxMS=0;
# IF ENG_ASP_SUP_CHGR_CD = 'N' THEN ENG_ASP_SUP_CHGR_CD_N=1; ELSE ENG_ASP_SUP_CHGR_CD_N=0;
# IF ENG_MFG_CD = '030' THEN ENG_MFG_CD_030=1; ELSE ENG_MFG_CD_030=0;
# IF MFG_BAS_MSRP =. THEN MFG_BAS_MSRPMS=1; ELSE MFG_BAS_MSRPMS=0;
# IF NADA_GVWC3 = . THEN NADA_GVWC3MS=1; ELSE NADA_GVWC3MS=0;
# IF WHL_BAS_LNGST_INCHS = . THEN WHL_BAS_LNGST_INCHSMS=1; ELSE WHL_BAS_LNGST_INCHSMS=0;
# IF 0<NADA_GVWC2<14000 THEN NADA_GVWC2_lt14k=1; ELSE NADA_GVWC2_lt14k=0;

# linpred = -0.3702
#  + -0.3541*TRK_GRSS_VEH_WGHT_RATG_CD_gt6
#  + -40.9586*TRK_CAB_CNFG_CD_CONSUVSPE
#  + 0.3095*TRK_CAB_CNFG_CD_CRW
#  + -0.309*cko_eng_cylinders_grp1
#  + 0.1573*ENG_TRK_DUTY_TYP_CD_HV
#  + -0.5961*NADA_BODY1_U
#  + -1.1301*NADA_BODY1_grp1
#  + 1.0216*ENG_FUEL_CD_grp1
#  + -40.6808*PLNT_CD_grp1
#  + -0.2854*PLNT_CD_grp2
#  + 0.2073*PLNT_CD_grp3
#  + -40.7521*ENG_MDL_CD_grp1
#  + -1.9303*ENG_MDL_CD_grp2
#  + 0.2928*ENG_MDL_CD_grp3
#  + -39.205*MAK_NM_grp1
#  + 1.4598*MAK_NM_grp2
#  + 0.1677*cko_max_msrpMS
#  + 0.0575*cko_wheelbase_maxMS
#  + 0.3024*ENG_ASP_SUP_CHGR_CD_N
#  + 0.195*ENG_MFG_CD_030
#  + -0.1962*MFG_BAS_MSRPMS
#  + -0.2251*NADA_GVWC3MS
#  + 0.0575*WHL_BAS_LNGST_INCHSMS
#  + -0.7057*NADA_GVWC2_lt14k
# ;




# Step 7
# Finalize GLM Again following Jim's feedback

# Drop large coefficients
# Drop levels with low exposure
# Drop PLNT_CD
# Make tire size and pressure continuous
# Drop missing unless part of a continuous variable
# Include ENG_MDL_CD and NADA_BODY1 ungrouped only
# Drop 'NC'
# Drop ENG_MFG_CD_110 due to aliasing
# Drop ENTERTAIN_CD_127
# Drop TLT_STRNG_WHL_OPT_CD_U

CONTFEATURES = []
CATFEATURES = ['TRK_CAB_CNFG_CD','cko_eng_cylinders','ENG_TRK_DUTY_TYP_CD','NADA_BODY1','ENG_ASP_SUP_CHGR_CD','ENG_MDL_CD','MAK_NM','TRK_FRNT_AXL_CD','ENG_MFG_CD']


# Include only catfeatures needed for GLM


  

tireSizeDictionary = cl.returnTireSizeDictionary()

data_train['FRNT_TIRE_SIZE_prefix'] = data_train['FRNT_TYRE_SIZE_CD'].map(tireSizeDictionary).apply(pd.Series)[0]
data_train['FRNT_TIRE_SIZE_suffix'] = data_train['FRNT_TYRE_SIZE_CD'].map(tireSizeDictionary).apply(pd.Series)[1]
data_train['REAR_TIRE_SIZE_prefix'] = data_train['REAR_TIRE_SIZE_CD'].map(tireSizeDictionary).apply(pd.Series)[0]
data_train['REAR_TIRE_SIZE_suffix'] = data_train['REAR_TIRE_SIZE_CD'].map(tireSizeDictionary).apply(pd.Series)[1]
data_test['FRNT_TIRE_SIZE_prefix'] = data_test['FRNT_TYRE_SIZE_CD'].map(tireSizeDictionary).apply(pd.Series)[0]
data_test['FRNT_TIRE_SIZE_suffix'] = data_test['FRNT_TYRE_SIZE_CD'].map(tireSizeDictionary).apply(pd.Series)[1]
data_test['REAR_TIRE_SIZE_prefix'] = data_test['REAR_TIRE_SIZE_CD'].map(tireSizeDictionary).apply(pd.Series)[0]
data_test['REAR_TIRE_SIZE_suffix'] = data_test['REAR_TIRE_SIZE_CD'].map(tireSizeDictionary).apply(pd.Series)[1]

data_train['FRNT_TIRE_SIZEMS']=np.where(data_train['FRNT_TIRE_SIZE_prefix']==-1, 1, 0)
data_train['REAR_TIRE_SIZEMS']=np.where(data_train['REAR_TIRE_SIZE_prefix']==-1, 1, 0)
data_test['FRNT_TIRE_SIZEMS']=np.where(data_test['FRNT_TIRE_SIZE_prefix']==-1, 1, 0)
data_test['REAR_TIRE_SIZEMS']=np.where(data_test['REAR_TIRE_SIZE_prefix']==-1, 1, 0)



DUMMYFEATURES=[]
for feature in CATFEATURES:
	dummiesTrain=pd.get_dummies(data_train[feature],prefix=feature)
	dummiesTest=pd.get_dummies(data_test[feature],prefix=feature)
	temp=dummiesTrain.sum()	
	# If column appears in data_train but not data_test we need to add it to data_test as a column of zeroes
	for k in ml.returnNotMatches(dummiesTrain.columns.values,dummiesTest.columns.values)[0]:
		dummiesTest[k]=0	
	dummiesTrain=dummiesTrain[temp[temp<temp.max()].index.values]
	dummiesTest=dummiesTest[temp[temp<temp.max()].index.values]
	if(len(dummiesTrain.columns)>0):
		dummiesTrain.columns = dummiesTrain.columns.str.replace('\s+', '_')
		dummiesTest.columns = dummiesTest.columns.str.replace('\s+', '_')
		data_train=pd.concat([data_train,dummiesTrain],axis=1)
		data_test=pd.concat([data_test,dummiesTest],axis=1)
		DUMMYFEATURES += list(dummiesTrain)

# print(DUMMYFEATURES)
# print(data_train.groupby(['TRK_CAB_CNFG_CD'], as_index=False)[WEIGHT].sum())

# Create custom variables for GLM
data_train['TRK_GRSS_VEH_WGHT_RATG_CD_gt6']=np.where(data_train['TRK_GRSS_VEH_WGHT_RATG_CD']>6, 1, 0)
data_test['TRK_GRSS_VEH_WGHT_RATG_CD_gt6'] =np.where( data_test['TRK_GRSS_VEH_WGHT_RATG_CD']>6, 1, 0)
data_train['TRK_GRSS_VEH_WGHT_RATG_CDMS']=np.where(data_train['TRK_GRSS_VEH_WGHT_RATG_CD']==-1, 1, 0)
data_test['TRK_GRSS_VEH_WGHT_RATG_CDMS']=np.where(data_test['TRK_GRSS_VEH_WGHT_RATG_CD']==-1, 1, 0)
# data_train['cko_wheelbase_mean_lt130']=np.where(data_train['cko_wheelbase_mean']<130,1,0)
# data_train['cko_max_gvwc_lt11k']=np.where(data_train['cko_max_gvwc']<11000,1,0)
data_train['cko_max_gvwc_lt14k']=np.where(data_train['cko_max_gvwc']<14000,1,0)
data_test['cko_max_gvwc_lt14k']=np.where(data_test['cko_max_gvwc']<14000,1,0)
# data_test['cko_wheelbase_mean_lt130']=np.where(data_test['cko_wheelbase_mean']<130,1,0)
# data_test['cko_max_gvwc_lt11k']=np.where(data_test['cko_max_gvwc']<11000,1,0)
data_train['TRK_CAB_CNFG_CD_CONSUVSPE']=(data_train['TRK_CAB_CNFG_CD_CON']+data_train['TRK_CAB_CNFG_CD_SUV']+data_train['TRK_CAB_CNFG_CD_SPE'])
data_test['TRK_CAB_CNFG_CD_CONSUVSPE']=(data_test['TRK_CAB_CNFG_CD_CON']+data_test['TRK_CAB_CNFG_CD_SUV']+data_test['TRK_CAB_CNFG_CD_SPE'])
# data_train['TRK_CAB_CNFG_CD_CRWTHICFWHCB']=(data_train['TRK_CAB_CNFG_CD_CRW']+data_train['TRK_CAB_CNFG_CD_THI']+data_train['TRK_CAB_CNFG_CD_CFW']+data_train['TRK_CAB_CNFG_CD_HCB'])
# data_test['TRK_CAB_CNFG_CD_CRWTHICFWHCB']=(data_test['TRK_CAB_CNFG_CD_CRW']+data_test['TRK_CAB_CNFG_CD_THI']+data_test['TRK_CAB_CNFG_CD_CFW']+data_test['TRK_CAB_CNFG_CD_HCB'])
# data_train['cko_max_gvwc_lt12k']=np.where((data_train['cko_max_gvwc']>0) & (data_train['cko_max_gvwc']<12000),1,0)
# data_test['cko_max_gvwc_lt12k']=np.where((data_test['cko_max_gvwc']>0) & (data_test['cko_max_gvwc']<12000),1,0)
# data_train['NADA_GVWC2_lt14k']=np.where((data_train['NADA_GVWC2']>0) & (data_train['NADA_GVWC2']<14000),1,0)
# data_test['NADA_GVWC2_lt14k']=np.where((data_test['NADA_GVWC2']>0) & (data_test['NADA_GVWC2']<14000),1,0)
# data_train['cko_eng_cylinders_eq6']=np.where(data_train['cko_eng_cylinders']==6, 1, 0)
# data_test['cko_eng_cylinders_eq6']=np.where(data_test['cko_eng_cylinders']==6, 1, 0)
# data_train['cko_eng_cylinders_numeric_eq6']=np.where(data_train['cko_eng_cylinders_numeric']==6, 1, 0)
# data_test['cko_eng_cylinders_numeric_eq6']=np.where(data_test['cko_eng_cylinders_numeric']==6, 1, 0)

data_train['cko_eng_cylinders_grp1']=(data_train['cko_eng_cylinders_.']+data_train['cko_eng_cylinders_4']+data_train['cko_eng_cylinders_5']+data_train['cko_eng_cylinders_8'])
data_test['cko_eng_cylinders_grp1']=(data_test['cko_eng_cylinders_.']+data_test['cko_eng_cylinders_4']+data_test['cko_eng_cylinders_5']+data_test['cko_eng_cylinders_8'])

data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_train['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
data_train['MAK_NM_MERCEDES_BENZ']=data_train['MAK_NM_MERCEDES-BENZ']
data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_train['MAK_NM_WESTERN_STAR/AUTO_CAR']
data_train['MAK_NM_PIERCE_MFG__INC_']=data_train['MAK_NM_PIERCE_MFG._INC.']
data_train['MAK_NM_WHITE_GMC']=data_train['MAK_NM_WHITE/GMC']
data_train['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_train['MAK_NM_TRANSPORTATION_MFG_CORP.']
data_train['MAK_NM_TEREX___TEREX_ADVANCE']=data_train['MAK_NM_TEREX_/_TEREX_ADVANCE']
data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_']=data_test['MAK_NM_OSHKOSH_MOTOR_TRUCK_CO.']
data_test['MAK_NM_MERCEDES_BENZ']=data_test['MAK_NM_MERCEDES-BENZ']	
data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']=data_test['MAK_NM_WESTERN_STAR/AUTO_CAR']
data_test['MAK_NM_PIERCE_MFG__INC_']=data_test['MAK_NM_PIERCE_MFG._INC.']
data_test['MAK_NM_WHITE_GMC']=data_test['MAK_NM_WHITE/GMC']
data_test['MAK_NM_TRANSPORTATION_MFG_CORP_']=data_test['MAK_NM_TRANSPORTATION_MFG_CORP.']
data_test['MAK_NM_TEREX___TEREX_ADVANCE']=data_test['MAK_NM_TEREX_/_TEREX_ADVANCE']

# data_train['NADA_BODY1_grp1']=(data_train['NADA_BODY1_K']+data_train['NADA_BODY1_H']+data_train['NADA_BODY1_B']+data_train['NADA_BODY1_D'])
# data_train['NADA_BODY1_grp2']=(data_train['NADA_BODY1_M']+data_train['NADA_BODY1_G']+data_train['NADA_BODY1_I'])
# data_train['NADA_BODY1_grp3']=(data_train['NADA_BODY1_N']+data_train['NADA_BODY1_2'])		
# data_train['MAK_NM_grp1']=(data_train['MAK_NM_LES_AUTOBUS_MCI']+data_train['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_train['MAK_NM_HYUNDAI']+data_train['MAK_NM_UNIMOG']+data_train['MAK_NM_SPARTAN_MOTORS']+data_train['MAK_NM_FWD_CORPORATION']+data_train['MAK_NM_ZELIGSON']+data_train['MAK_NM_JOHN_DEERE']+data_train['MAK_NM_FEDERAL_MOTORS']+data_train['MAK_NM_DIAMOND_REO']+data_train['MAK_NM_GILLIG']+data_train['MAK_NM_PREVOST']+data_train['MAK_NM_OTTAWA']+data_train['MAK_NM_DUPLEX']+data_train['MAK_NM_MERCEDES-BENZ'])
# data_train['MAK_NM_grp2']=(data_train['MAK_NM_WORKHORSE_CUSTOM_CHASSIS']+data_train['MAK_NM_CATERPILLAR']+data_train['MAK_NM_ADVANCE_MIXER']+data_train['MAK_NM_EMERGENCY_ONE']+data_train['MAK_NM_FORETRAVEL_MOTORHOME']+data_train['MAK_NM_BLUE_BIRD']+data_train['MAK_NM_INDIANA_PHOENIX']+data_train['MAK_NM_HENDRICKSON'])
# data_train['ENG_FUEL_CD_grp1']=(data_train['ENG_FUEL_CD_U']+data_train['ENG_FUEL_CD_N'])
# data_train['ENG_FUEL_CD_grp2']=(data_train['ENG_FUEL_CD_G']+data_train['ENG_FUEL_CD_P'])
# data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_Y']+data_train['PLNT_CD_5']+data_train['PLNT_CD_8'])
# data_train['PLNT_CD_grp2']=(data_train['PLNT_CD_U']+data_train['PLNT_CD_1']+data_train['PLNT_CD_V'])
# data_train['PLNT_CD_grp3']=(data_train['PLNT_CD_M']+data_train['PLNT_CD_W']+data_train['PLNT_CD_F']+data_train['PLNT_CD_D']+data_train['PLNT_CD_K']+data_train['PLNT_CD_7']+data_train['PLNT_CD_S']+data_train['PLNT_CD_R']+data_train['PLNT_CD_C']+data_train['PLNT_CD_L']+data_train['PLNT_CD_2']+data_train['PLNT_CD_T']+data_train['PLNT_CD_4']+data_train['PLNT_CD_0']+data_train['PLNT_CD_9'])
# data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_120010']+data_train['ENG_MDL_CD_160030']+data_train['ENG_MDL_CD_110030']+data_train['ENG_MDL_CD_180020']+data_train['ENG_MDL_CD_140220']+data_train['ENG_MDL_CD_180010']+data_train['ENG_MDL_CD_090020']+data_train['ENG_MDL_CD_140167']+data_train['ENG_MDL_CD_140173']+data_train['ENG_MDL_CD_140115']+data_train['ENG_MDL_CD_160020']+data_train['ENG_MDL_CD_020105']+data_train['ENG_MDL_CD_195001']+data_train['ENG_MDL_CD_010038']+data_train['ENG_MDL_CD_165185']+data_train['ENG_MDL_CD_020085']+data_train['ENG_MDL_CD_140168']+data_train['ENG_MDL_CD_165225']+data_train['ENG_MDL_CD_140027']+data_train['ENG_MDL_CD_090077']+data_train['ENG_MDL_CD_180052']+data_train['ENG_MDL_CD_165230']+data_train['ENG_MDL_CD_090043']+data_train['ENG_MDL_CD_090045']+data_train['ENG_MDL_CD_165160']+data_train['ENG_MDL_CD_020060']+data_train['ENG_MDL_CD_010050']+data_train['ENG_MDL_CD_020090']+data_train['ENG_MDL_CD_040030']+data_train['ENG_MDL_CD_140215']+data_train['ENG_MDL_CD_210100']+data_train['ENG_MDL_CD_020100']+data_train['ENG_MDL_CD_220019']+data_train['ENG_MDL_CD_210090']+data_train['ENG_MDL_CD_140055']+data_train['ENG_MDL_CD_010070']+data_train['ENG_MDL_CD_020120']+data_train['ENG_MDL_CD_140090']+data_train['ENG_MDL_CD_140125']+data_train['ENG_MDL_CD_210110']+data_train['ENG_MDL_CD_140050']+data_train['ENG_MDL_CD_090051']+data_train['ENG_MDL_CD_140234']+data_train['ENG_MDL_CD_245010']+data_train['ENG_MDL_CD_030060']+data_train['ENG_MDL_CD_210040']+data_train['ENG_MDL_CD_210070']+data_train['ENG_MDL_CD_210050']+data_train['ENG_MDL_CD_210060']+data_train['ENG_MDL_CD_210080']+data_train['ENG_MDL_CD_210020']+data_train['ENG_MDL_CD_140025']+data_train['ENG_MDL_CD_210030']+data_train['ENG_MDL_CD_020026']+data_train['ENG_MDL_CD_110070']+data_train['ENG_MDL_CD_180040']+data_train['ENG_MDL_CD_140040']+data_train['ENG_MDL_CD_140020']+data_train['ENG_MDL_CD_160060']+data_train['ENG_MDL_CD_030020']+data_train['ENG_MDL_CD_090040']+data_train['ENG_MDL_CD_080012']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_050020']+data_train['ENG_MDL_CD_165007'])
# data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_150010']+data_train['ENG_MDL_CD_150020']+data_train['ENG_MDL_CD_150060']+data_train['ENG_MDL_CD_210165']+data_train['ENG_MDL_CD_090030']+data_train['ENG_MDL_CD_165030']+data_train['ENG_MDL_CD_180030']+data_train['ENG_MDL_CD_200010']+data_train['ENG_MDL_CD_140195']+data_train['ENG_MDL_CD_140105']+data_train['ENG_MDL_CD_165140']+data_train['ENG_MDL_CD_140210']+data_train['ENG_MDL_CD_080080']+data_train['ENG_MDL_CD_160012']+data_train['ENG_MDL_CD_140065']+data_train['ENG_MDL_CD_090011']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_140205']+data_train['ENG_MDL_CD_030050']+data_train['ENG_MDL_CD_030047']+data_train['ENG_MDL_CD_165090']+data_train['ENG_MDL_CD_140095']+data_train['ENG_MDL_CD_140150']+data_train['ENG_MDL_CD_090070']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_020072']+data_train['ENG_MDL_CD_165010'])
# data_train['ENG_MDL_CD_grp3']=(data_train['ENG_MDL_CD_150003']+data_train['ENG_MDL_CD_150150']+data_train['ENG_MDL_CD_150005']+data_train['ENG_MDL_CD_150007'])

# data_test['NADA_BODY1_grp1']=(data_test['NADA_BODY1_K']+data_test['NADA_BODY1_H']+data_test['NADA_BODY1_B']+data_test['NADA_BODY1_D'])
# data_test['NADA_BODY1_grp2']=(data_test['NADA_BODY1_M']+data_test['NADA_BODY1_G']+data_test['NADA_BODY1_I'])
# data_test['NADA_BODY1_grp3']=(data_test['NADA_BODY1_N']+data_test['NADA_BODY1_2'])		
# data_test['MAK_NM_grp1']=(data_test['MAK_NM_LES_AUTOBUS_MCI']+data_test['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_test['MAK_NM_HYUNDAI']+data_test['MAK_NM_UNIMOG']+data_test['MAK_NM_SPARTAN_MOTORS']+data_test['MAK_NM_FWD_CORPORATION']+data_test['MAK_NM_ZELIGSON']+data_test['MAK_NM_JOHN_DEERE']+data_test['MAK_NM_FEDERAL_MOTORS']+data_test['MAK_NM_DIAMOND_REO']+data_test['MAK_NM_GILLIG']+data_test['MAK_NM_PREVOST']+data_test['MAK_NM_OTTAWA']+data_test['MAK_NM_DUPLEX']+data_test['MAK_NM_MERCEDES-BENZ'])
# data_test['MAK_NM_grp2']=(data_test['MAK_NM_WORKHORSE_CUSTOM_CHASSIS']+data_test['MAK_NM_CATERPILLAR']+data_test['MAK_NM_ADVANCE_MIXER']+data_test['MAK_NM_EMERGENCY_ONE']+data_test['MAK_NM_FORETRAVEL_MOTORHOME']+data_test['MAK_NM_BLUE_BIRD']+data_test['MAK_NM_INDIANA_PHOENIX']+data_test['MAK_NM_HENDRICKSON'])
# data_test['ENG_FUEL_CD_grp1']=(data_test['ENG_FUEL_CD_U']+data_test['ENG_FUEL_CD_N'])
# data_test['ENG_FUEL_CD_grp2']=(data_test['ENG_FUEL_CD_G']+data_test['ENG_FUEL_CD_P'])
# data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_Y']+data_test['PLNT_CD_5']+data_test['PLNT_CD_8'])
# data_test['PLNT_CD_grp2']=(data_test['PLNT_CD_U']+data_test['PLNT_CD_1']+data_test['PLNT_CD_V'])
# data_test['PLNT_CD_grp3']=(data_test['PLNT_CD_M']+data_test['PLNT_CD_W']+data_test['PLNT_CD_F']+data_test['PLNT_CD_D']+data_test['PLNT_CD_K']+data_test['PLNT_CD_7']+data_test['PLNT_CD_S']+data_test['PLNT_CD_R']+data_test['PLNT_CD_C']+data_test['PLNT_CD_L']+data_test['PLNT_CD_2']+data_test['PLNT_CD_T']+data_test['PLNT_CD_4']+data_test['PLNT_CD_0']+data_test['PLNT_CD_9'])
# data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_120010']+data_test['ENG_MDL_CD_160030']+data_test['ENG_MDL_CD_110030']+data_test['ENG_MDL_CD_180020']+data_test['ENG_MDL_CD_140220']+data_test['ENG_MDL_CD_180010']+data_test['ENG_MDL_CD_090020']+data_test['ENG_MDL_CD_140167']+data_test['ENG_MDL_CD_140173']+data_test['ENG_MDL_CD_140115']+data_test['ENG_MDL_CD_160020']+data_test['ENG_MDL_CD_020105']+data_test['ENG_MDL_CD_195001']+data_test['ENG_MDL_CD_010038']+data_test['ENG_MDL_CD_165185']+data_test['ENG_MDL_CD_020085']+data_test['ENG_MDL_CD_140168']+data_test['ENG_MDL_CD_165225']+data_test['ENG_MDL_CD_140027']+data_test['ENG_MDL_CD_090077']+data_test['ENG_MDL_CD_180052']+data_test['ENG_MDL_CD_165230']+data_test['ENG_MDL_CD_090043']+data_test['ENG_MDL_CD_090045']+data_test['ENG_MDL_CD_165160']+data_test['ENG_MDL_CD_020060']+data_test['ENG_MDL_CD_010050']+data_test['ENG_MDL_CD_020090']+data_test['ENG_MDL_CD_040030']+data_test['ENG_MDL_CD_140215']+data_test['ENG_MDL_CD_210100']+data_test['ENG_MDL_CD_020100']+data_test['ENG_MDL_CD_220019']+data_test['ENG_MDL_CD_210090']+data_test['ENG_MDL_CD_140055']+data_test['ENG_MDL_CD_010070']+data_test['ENG_MDL_CD_020120']+data_test['ENG_MDL_CD_140090']+data_test['ENG_MDL_CD_140125']+data_test['ENG_MDL_CD_210110']+data_test['ENG_MDL_CD_140050']+data_test['ENG_MDL_CD_090051']+data_test['ENG_MDL_CD_140234']+data_test['ENG_MDL_CD_245010']+data_test['ENG_MDL_CD_030060']+data_test['ENG_MDL_CD_210040']+data_test['ENG_MDL_CD_210070']+data_test['ENG_MDL_CD_210050']+data_test['ENG_MDL_CD_210060']+data_test['ENG_MDL_CD_210080']+data_test['ENG_MDL_CD_210020']+data_test['ENG_MDL_CD_140025']+data_test['ENG_MDL_CD_210030']+data_test['ENG_MDL_CD_020026']+data_test['ENG_MDL_CD_110070']+data_test['ENG_MDL_CD_180040']+data_test['ENG_MDL_CD_140040']+data_test['ENG_MDL_CD_140020']+data_test['ENG_MDL_CD_160060']+data_test['ENG_MDL_CD_030020']+data_test['ENG_MDL_CD_090040']+data_test['ENG_MDL_CD_080012']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_050020']+data_test['ENG_MDL_CD_165007'])
# data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_150010']+data_test['ENG_MDL_CD_150020']+data_test['ENG_MDL_CD_150060']+data_test['ENG_MDL_CD_210165']+data_test['ENG_MDL_CD_090030']+data_test['ENG_MDL_CD_165030']+data_test['ENG_MDL_CD_180030']+data_test['ENG_MDL_CD_200010']+data_test['ENG_MDL_CD_140195']+data_test['ENG_MDL_CD_140105']+data_test['ENG_MDL_CD_165140']+data_test['ENG_MDL_CD_140210']+data_test['ENG_MDL_CD_080080']+data_test['ENG_MDL_CD_160012']+data_test['ENG_MDL_CD_140065']+data_test['ENG_MDL_CD_090011']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_140205']+data_test['ENG_MDL_CD_030050']+data_test['ENG_MDL_CD_030047']+data_test['ENG_MDL_CD_165090']+data_test['ENG_MDL_CD_140095']+data_test['ENG_MDL_CD_140150']+data_test['ENG_MDL_CD_090070']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_020072']+data_test['ENG_MDL_CD_165010'])
# data_test['ENG_MDL_CD_grp3']=(data_test['ENG_MDL_CD_150003']+data_test['ENG_MDL_CD_150150']+data_test['ENG_MDL_CD_150005']+data_test['ENG_MDL_CD_150007'])

data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_120010']+data_train['ENG_MDL_CD_160030']+data_train['ENG_MDL_CD_110030']+data_train['ENG_MDL_CD_180020']+data_train['ENG_MDL_CD_140220']+data_train['ENG_MDL_CD_180010']+data_train['ENG_MDL_CD_090020']+data_train['ENG_MDL_CD_140167']+data_train['ENG_MDL_CD_140173']+data_train['ENG_MDL_CD_140115']+data_train['ENG_MDL_CD_160020']+data_train['ENG_MDL_CD_020105']+data_train['ENG_MDL_CD_195001']+data_train['ENG_MDL_CD_010038']+data_train['ENG_MDL_CD_165185']+data_train['ENG_MDL_CD_020085']+data_train['ENG_MDL_CD_140168']+data_train['ENG_MDL_CD_165225']+data_train['ENG_MDL_CD_140027']+data_train['ENG_MDL_CD_090077']+data_train['ENG_MDL_CD_180052']+data_train['ENG_MDL_CD_165230']+data_train['ENG_MDL_CD_090043']+data_train['ENG_MDL_CD_090045']+data_train['ENG_MDL_CD_165160']+data_train['ENG_MDL_CD_020060']+data_train['ENG_MDL_CD_010050']+data_train['ENG_MDL_CD_020090']+data_train['ENG_MDL_CD_040030']+data_train['ENG_MDL_CD_140215']+data_train['ENG_MDL_CD_210100']+data_train['ENG_MDL_CD_020100']+data_train['ENG_MDL_CD_220019']+data_train['ENG_MDL_CD_210090']+data_train['ENG_MDL_CD_140055']+data_train['ENG_MDL_CD_010070']+data_train['ENG_MDL_CD_020120']+data_train['ENG_MDL_CD_140090']+data_train['ENG_MDL_CD_140125']+data_train['ENG_MDL_CD_210110']+data_train['ENG_MDL_CD_140050']+data_train['ENG_MDL_CD_090051']+data_train['ENG_MDL_CD_140234']+data_train['ENG_MDL_CD_245010']+data_train['ENG_MDL_CD_030060']+data_train['ENG_MDL_CD_210040']+data_train['ENG_MDL_CD_210070']+data_train['ENG_MDL_CD_210050']+data_train['ENG_MDL_CD_210060']+data_train['ENG_MDL_CD_210080']+data_train['ENG_MDL_CD_210020']+data_train['ENG_MDL_CD_140025']+data_train['ENG_MDL_CD_210030']+data_train['ENG_MDL_CD_020026']+data_train['ENG_MDL_CD_110070']+data_train['ENG_MDL_CD_180040']+data_train['ENG_MDL_CD_140040']+data_train['ENG_MDL_CD_140020']+data_train['ENG_MDL_CD_160060']+data_train['ENG_MDL_CD_030020']+data_train['ENG_MDL_CD_090040']+data_train['ENG_MDL_CD_080012']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_050020']+data_train['ENG_MDL_CD_165007']+data_train['ENG_MDL_CD_080080']+data_train['ENG_MDL_CD_160012']+data_train['ENG_MDL_CD_030050']+data_train['ENG_MDL_CD_140095']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_165010'])
data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_120010']+data_test['ENG_MDL_CD_160030']+data_test['ENG_MDL_CD_110030']+data_test['ENG_MDL_CD_180020']+data_test['ENG_MDL_CD_140220']+data_test['ENG_MDL_CD_180010']+data_test['ENG_MDL_CD_090020']+data_test['ENG_MDL_CD_140167']+data_test['ENG_MDL_CD_140173']+data_test['ENG_MDL_CD_140115']+data_test['ENG_MDL_CD_160020']+data_test['ENG_MDL_CD_020105']+data_test['ENG_MDL_CD_195001']+data_test['ENG_MDL_CD_010038']+data_test['ENG_MDL_CD_165185']+data_test['ENG_MDL_CD_020085']+data_test['ENG_MDL_CD_140168']+data_test['ENG_MDL_CD_165225']+data_test['ENG_MDL_CD_140027']+data_test['ENG_MDL_CD_090077']+data_test['ENG_MDL_CD_180052']+data_test['ENG_MDL_CD_165230']+data_test['ENG_MDL_CD_090043']+data_test['ENG_MDL_CD_090045']+data_test['ENG_MDL_CD_165160']+data_test['ENG_MDL_CD_020060']+data_test['ENG_MDL_CD_010050']+data_test['ENG_MDL_CD_020090']+data_test['ENG_MDL_CD_040030']+data_test['ENG_MDL_CD_140215']+data_test['ENG_MDL_CD_210100']+data_test['ENG_MDL_CD_020100']+data_test['ENG_MDL_CD_220019']+data_test['ENG_MDL_CD_210090']+data_test['ENG_MDL_CD_140055']+data_test['ENG_MDL_CD_010070']+data_test['ENG_MDL_CD_020120']+data_test['ENG_MDL_CD_140090']+data_test['ENG_MDL_CD_140125']+data_test['ENG_MDL_CD_210110']+data_test['ENG_MDL_CD_140050']+data_test['ENG_MDL_CD_090051']+data_test['ENG_MDL_CD_140234']+data_test['ENG_MDL_CD_245010']+data_test['ENG_MDL_CD_030060']+data_test['ENG_MDL_CD_210040']+data_test['ENG_MDL_CD_210070']+data_test['ENG_MDL_CD_210050']+data_test['ENG_MDL_CD_210060']+data_test['ENG_MDL_CD_210080']+data_test['ENG_MDL_CD_210020']+data_test['ENG_MDL_CD_140025']+data_test['ENG_MDL_CD_210030']+data_test['ENG_MDL_CD_020026']+data_test['ENG_MDL_CD_110070']+data_test['ENG_MDL_CD_180040']+data_test['ENG_MDL_CD_140040']+data_test['ENG_MDL_CD_140020']+data_test['ENG_MDL_CD_160060']+data_test['ENG_MDL_CD_030020']+data_test['ENG_MDL_CD_090040']+data_test['ENG_MDL_CD_080012']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_050020']+data_test['ENG_MDL_CD_165007']+data_test['ENG_MDL_CD_080080']+data_test['ENG_MDL_CD_160012']+data_test['ENG_MDL_CD_030050']+data_test['ENG_MDL_CD_140095']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_165010'])
data_train['MAK_NM_grp1']=(data_train['MAK_NM_LES_AUTOBUS_MCI']+data_train['MAK_NM_DIAMOND_REO']+data_train['MAK_NM_DUPLEX']+data_train['MAK_NM_HYUNDAI']+data_train['MAK_NM_OTTAWA']+data_train['MAK_NM_GIANT']+data_train['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_train['MAK_NM_PREVOST']+data_train['MAK_NM_MERCEDES-BENZ']+data_train['MAK_NM_FWD_CORPORATION']+data_train['MAK_NM_JOHN_DEERE']+data_train['MAK_NM_FEDERAL_MOTORS']+data_train['MAK_NM_GILLIG']+data_train['MAK_NM_CAPACITY_OF_TEXAS']+data_train['MAK_NM_SPARTAN_MOTORS']+data_train['MAK_NM_UNIMOG']+data_train['MAK_NM_ZELIGSON']+data_train['MAK_NM_WINNEBAGO']+data_train['MAK_NM_WHITE_GMC'])
data_test['MAK_NM_grp1']=(data_test['MAK_NM_LES_AUTOBUS_MCI']+data_test['MAK_NM_DIAMOND_REO']+data_test['MAK_NM_DUPLEX']+data_test['MAK_NM_HYUNDAI']+data_test['MAK_NM_OTTAWA']+data_test['MAK_NM_GIANT']+data_test['MAK_NM_CHANCE_COACH_TRANSIT_BUS']+data_test['MAK_NM_PREVOST']+data_test['MAK_NM_MERCEDES-BENZ']+data_test['MAK_NM_FWD_CORPORATION']+data_test['MAK_NM_JOHN_DEERE']+data_test['MAK_NM_FEDERAL_MOTORS']+data_test['MAK_NM_GILLIG']+data_test['MAK_NM_CAPACITY_OF_TEXAS']+data_test['MAK_NM_SPARTAN_MOTORS']+data_test['MAK_NM_UNIMOG']+data_test['MAK_NM_ZELIGSON']+data_test['MAK_NM_WINNEBAGO']+data_test['MAK_NM_WHITE_GMC'])
data_train['MAK_NM_grp2']=(data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']+data_train['MAK_NM_HINO']+data_train['MAK_NM_WORKHORSE_CUSTOM_CHASSIS']+data_train['MAK_NM_INDIANA_PHOENIX']+data_train['MAK_NM_ADVANCE_MIXER']+data_train['MAK_NM_ISUZU']+data_train['MAK_NM_KENWORTH']+data_train['MAK_NM_GMC']+data_train['MAK_NM_FREIGHTLINER'])
data_test['MAK_NM_grp2']=(data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']+data_test['MAK_NM_HINO']+data_test['MAK_NM_WORKHORSE_CUSTOM_CHASSIS']+data_test['MAK_NM_INDIANA_PHOENIX']+data_test['MAK_NM_ADVANCE_MIXER']+data_test['MAK_NM_ISUZU']+data_test['MAK_NM_KENWORTH']+data_test['MAK_NM_GMC']+data_test['MAK_NM_FREIGHTLINER'])
# data_train['MAK_NM_grp2']=(data_train['MAK_NM_WESTERN_STAR_AUTO_CAR']+data_train['MAK_NM_HINO']+data_train['MAK_NM_WORKHORSE_CUSTOM_CHASSIS'])
# data_test['MAK_NM_grp2']=(data_test['MAK_NM_WESTERN_STAR_AUTO_CAR']+data_test['MAK_NM_HINO']+data_test['MAK_NM_WORKHORSE_CUSTOM_CHASSIS'])
# data_train['MAK_NM_grp3']=(data_train['MAK_NM_INDIANA_PHOENIX']+data_train['MAK_NM_ADVANCE_MIXER']+data_train['MAK_NM_HENDRICKSON'])
# data_test['MAK_NM_grp3']=(data_test['MAK_NM_INDIANA_PHOENIX']+data_test['MAK_NM_ADVANCE_MIXER']+data_test['MAK_NM_HENDRICKSON'])
# data_train['NADA_BODY1_grp1']=(data_train['NADA_BODY1_M']+data_train['NADA_BODY1_G']+data_train['NADA_BODY1_N'])
# data_test['NADA_BODY1_grp1']=(data_test['NADA_BODY1_M']+data_test['NADA_BODY1_G']+data_test['NADA_BODY1_N'])
data_train['TRK_FRNT_AXL_CD_grp1']=(data_train['TRK_FRNT_AXL_CD_SB']+data_train['TRK_FRNT_AXL_CD_SF']+data_train['TRK_FRNT_AXL_CD_U'])
data_test['TRK_FRNT_AXL_CD_grp1']=(data_test['TRK_FRNT_AXL_CD_SB']+data_test['TRK_FRNT_AXL_CD_SF']+data_test['TRK_FRNT_AXL_CD_U'])
data_train['ENG_MFG_CD_grp1']=(data_train['ENG_MFG_CD_010']+data_train['ENG_MFG_CD_020'])
data_test['ENG_MFG_CD_grp1']=(data_test['ENG_MFG_CD_010']+data_test['ENG_MFG_CD_020'])
data_train['TRK_CAB_CNFG_CD_grp1']=(data_train['TRK_CAB_CNFG_CD_CFW']+data_train['TRK_CAB_CNFG_CD_CRW'])
data_test['TRK_CAB_CNFG_CD_grp1']=(data_test['TRK_CAB_CNFG_CD_CFW']+data_test['TRK_CAB_CNFG_CD_CRW'])
data_train['TRK_CAB_CNFG_CD_grp2']=(data_train['TRK_CAB_CNFG_CD_CLN']+data_train['TRK_CAB_CNFG_CD_TLO'])
data_test['TRK_CAB_CNFG_CD_grp2']=(data_test['TRK_CAB_CNFG_CD_CLN']+data_test['TRK_CAB_CNFG_CD_TLO'])

if not os.path.exists('glm/'):
	os.makedirs('glm/')

formula= 'lossratio ~ TRK_GRSS_VEH_WGHT_RATG_CD_gt6 + cko_eng_cylinders_grp1 + ENG_TRK_DUTY_TYP_CD_HV '
formula += '  + ENG_ASP_SUP_CHGR_CD_N  + TRK_FRNT_AXL_CD_grp1'
formula += ' + cko_max_gvwc_lt14k  + ENG_MFG_CD_grp1 '
formula += ' + TRK_CAB_CNFG_CD_grp1 + TRK_CAB_CNFG_CD_grp2 '
formula += ' + MAK_NM_grp1 + MAK_NM_grp2 ' 
# Try all these below 
# formula += ' + MAK_NM_MAXIM + MAK_NM_PETERBILT + MAK_NM_DODGE + MAK_NM_DUPLEX + MAK_NM_CATERPILLAR + MAK_NM_WESTERN_STAR_AUTO_CAR + MAK_NM_OSHKOSH_MOTOR_TRUCK_CO_ + MAK_NM_MOTOR_COACH_INDUSTRIES + MAK_NM_INDIANA_PHOENIX + MAK_NM_DIAMOND_REO + MAK_NM_LES_AUTOBUS_MCI + MAK_NM_GMC + MAK_NM_VOLVO + MAK_NM_OTTAWA + MAK_NM_HYUNDAI + MAK_NM_KALMAR + MAK_NM_CHEVROLET + MAK_NM_INTERNATIONAL + MAK_NM_CRANE_CARRIER + MAK_NM_MARMON_HERRINGTON + MAK_NM_UTILIMASTER  + MAK_NM_GIANT + MAK_NM_IVECO + MAK_NM_SEAGRAVE_FIRE_APPARATUS + MAK_NM_AUTOCAR + MAK_NM_THOMAS + MAK_NM_MITSUBISHI_FUSO_TRUCK_OF_AMERICA_INC + MAK_NM_MACK + MAK_NM_EMERGENCY_ONE + MAK_NM_VAN_HOOL + MAK_NM_HINO + MAK_NM_AUTOCAR_LLC + MAK_NM_PIERCE_MFG__INC_ + MAK_NM_LODAL + MAK_NM_WINNEBAGO + MAK_NM_IC_CORPORATION + MAK_NM_ROADMASTER_RAIL + MAK_NM_COUNTRY_COACH_MOTORHOME + MAK_NM_CHANCE_COACH_TRANSIT_BUS + MAK_NM_FREIGHTLINER + MAK_NM_ADVANCE_MIXER + MAK_NM_HENDRICKSON + MAK_NM_FWD_CORPORATION + MAK_NM_BLUE_BIRD + MAK_NM_MERCEDES_BENZ + MAK_NM_EVOBUS + MAK_NM_PREVOST + MAK_NM_AMERICAN_LA_FRANCE + MAK_NM_CAPACITY_OF_TEXAS + MAK_NM_FEDERAL_MOTORS + MAK_NM_GILLIG + MAK_NM_SAVAGE + MAK_NM_JOHN_DEERE + MAK_NM_SCANIA + MAK_NM_RAM + MAK_NM_TRANSPORTATION_MFG_CORP_ + MAK_NM_WORKHORSE_CUSTOM_CHASSIS + MAK_NM_WHITE + MAK_NM_FORETRAVEL_MOTORHOME + MAK_NM_LAFORZA + MAK_NM_SPARTAN_MOTORS + MAK_NM_ZELIGSON + MAK_NM_TEREX___TEREX_ADVANCE + MAK_NM_WHITE_GMC + MAK_NM_STERLING_TRUCK + MAK_NM_NISSAN_DIESEL + MAK_NM_UNIMOG + MAK_NM_NEW_FLYER + MAK_NM_BERING + MAK_NM_KENWORTH + MAK_NM_ISUZU '
# Test MDL_CD

# formula += ' + ENG_MDL_CD_080080 + ENG_MDL_CD_070020 + ENG_MDL_CD_165170 + ENG_MDL_CD_070018 + ENG_MDL_CD_010045 + ENG_MDL_CD_165010 + ENG_MDL_CD_020045 + ENG_MDL_CD_020094 + ENG_MDL_CD_140095 + ENG_MDL_CD_165051 + ENG_MDL_CD_140085 + ENG_MDL_CD_090015 + ENG_MDL_CD_010010 + ENG_MDL_CD_160012 + ENG_MDL_CD_020080 + ENG_MDL_CD_020070 + ENG_MDL_CD_010020 + ENG_MDL_CD_070025 + ENG_MDL_CD_080100 '

# testarray=['TRK_GRSS_VEH_WGHT_RATG_CD_gt6','cko_eng_cylinders_grp1','ENG_TRK_DUTY_TYP_CD_HV','NADA_BODY1_D','NADA_BODY1_grp1','ENG_ASP_SUP_CHGR_CD_N','ENG_MDL_CD_grp1','MAK_NM_grp1','MAK_NM_grp2','MAK_NM_grp3','TRK_FRNT_AXL_CD_grp1','cko_max_gvwc_lt14k','ENG_MFG_CD_grp1','TRK_CAB_CNFG_CD_grp1','TRK_CAB_CNFG_CD_grp2']

# for testfield in testarray:
# 	print(testfield);
# 	print(data_train[testfield].sum())

prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm/',0)

# data_train['weight'] = data_train[WEIGHT]
# data_train['predictedValue'] = prediction_glm_train*data_train['weight']
# data_train['actualValue'] = data_train[LABEL]*data_train['weight']
# data_test['weight'] = data_test[WEIGHT]
# data_test['predictedValue'] = prediction_glm_test*data_test['weight']
# data_test['actualValue'] = data_test[LABEL]*data_test['weight']

# print('weight');
# print(data_train['weight'].sum())
# print('predictedValue');
# print(data_train['predictedValue'].sum())
# print('actualValue');
# print(data_train['actualValue'].sum())

# ml.actualvsfittedbyfactor(data_train,data_test,'TRK_GRSS_VEH_WGHT_RATG_CD_gt6','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'cko_eng_cylinders_grp1','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'NADA_BODY1_grp1','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'ENG_MDL_CD_grp1','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'MAK_NM_grp1','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'MAK_NM_grp2','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'MAK_NM_grp3','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'TRK_FRNT_AXL_CD_grp1','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'cko_max_gvwc_lt14k','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'ENG_MFG_CD_grp1','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'TRK_CAB_CNFG_CD_grp1','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'TRK_CAB_CNFG_CD_grp2','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'FRNT_TIRE_SIZE_prefix','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'FRNT_TIRE_SIZE_suffix','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'REAR_TIRE_SIZE_prefix','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'REAR_TIRE_SIZE_suffix','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'FRNT_TIRE_SIZEMS','glm/')


# ALLCONTFEATURES = cl.returnCONTFEATURES()
# ALLCATFEATURES = cl.returnCATFEATURES()

# for feature in ALLCONTFEATURES:
# 	print(feature)
# 	cutoffs=BANDFEATURES[feature]
# 	data_train[feature+'Band']=data_train[feature].apply(ml.applyband,args=(cutoffs,))
# 	data_test[feature+'Band']=data_test[feature].apply(ml.applyband,args=(cutoffs,))
# 	ml.actualvsfittedbyfactor(data_train,data_test,feature+'Band','glm/')

# for feature in ALLCATFEATURES:
# 	ml.actualvsfittedbyfactor(data_train,data_test,feature,'glm/')		


print(datetime.datetime.now())
print("Ending Program")
