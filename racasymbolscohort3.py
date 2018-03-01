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

filenametrain="TRAIN3.csv"
filenametest="TEST3.csv"

print("Starting Program")
print(datetime.datetime.now())

data_train = pd.read_csv(filenametrain, skipinitialspace=True, skiprows=1, names=COLUMNS, dtype=dtype)
data_test = pd.read_csv(filenametest, skipinitialspace=True, skiprows=1, names=COLUMNS, dtype=dtype)
print('train records: '+str(len(data_train)))
print('test records: '+str(len(data_test)))

avgweight = data_train[WEIGHT].mean()
# print(avgweight)

data_train[WEIGHT]=data_train[WEIGHT]/avgweight
data_test[WEIGHT]= data_test[WEIGHT]/avgweight

# CATARRAY = ['ABS_BRK_CD','bodyStyleCd','ENG_MDL_CD','NC_Airbag_P','NC_TireSize','PLNT_CD']
# for CAT in CATARRAY:
# 	text_file = open(CAT+'.txt', "w")
# 	text_file.write(str(set(data_train[CAT])))
# 	text_file.close()


# # Step 1
# # Run initital GLM

# CONTFEATURES = []
# CATFEATURES = []
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
# # data_train['WHL_BAS_SHRST_INCHS_lt115']=np.where(data_train['WHL_BAS_SHRST_INCHS']<115, 1, 0)
# # data_test['WHL_BAS_SHRST_INCHS_lt115'] =np.where( data_test['WHL_BAS_SHRST_INCHS']<115, 1, 0)
# data_train['EA_TIP_OVER_STABILITY_RATIO_ge105']=np.where(data_train['EA_TIP_OVER_STABILITY_RATIO']>=1.05, 1, 0)
# data_test['EA_TIP_OVER_STABILITY_RATIO_ge105'] =np.where( data_test['EA_TIP_OVER_STABILITY_RATIO']>=1.05, 1, 0)
# data_train['cko_hp_wheelbase_ratio_ge24']=np.where(data_train['cko_hp_wheelbase_ratio']>=2.4, 1, 0)
# data_test['cko_hp_wheelbase_ratio_ge24']=np.where(data_test['cko_hp_wheelbase_ratio']>=2.4, 1, 0)

# if not os.path.exists('glm/'):
# 	os.makedirs('glm/')
# formula= 'lossratio ~ ENTERTAIN_CD_127 + EA_TIP_OVER_STABILITY_RATIO_ge105  + ABS_BRK_AWO  + cko_hp_wheelbase_ratio_ge24  '
# prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm/',1)



# # Step 2
# # Run CONTFEATURES GBM

# ALLCONTFEATURES = cl.returnCONTFEATURES()
# CONTFEATURES = []
# CATFEATURES = []
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
# data_train['EA_TIP_OVER_STABILITY_RATIO_ge105']=np.where(data_train['EA_TIP_OVER_STABILITY_RATIO']>=1.05, 1, 0)
# data_test['EA_TIP_OVER_STABILITY_RATIO_ge105'] =np.where( data_test['EA_TIP_OVER_STABILITY_RATIO']>=1.05, 1, 0)
# data_train['cko_hp_wheelbase_ratio_ge24']=np.where(data_train['cko_hp_wheelbase_ratio']>=2.4, 1, 0)
# data_test['cko_hp_wheelbase_ratio_ge24']=np.where(data_test['cko_hp_wheelbase_ratio']>=2.4, 1, 0)

# if not os.path.exists('glm/'):
# 	os.makedirs('glm/')
# formula= 'lossratio ~  ENTERTAIN_CD_127 + EA_TIP_OVER_STABILITY_RATIO_ge105  + ABS_BRK_AWO  + cko_hp_wheelbase_ratio_ge24     '
# prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm/',2)

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


# 	X_train = np.asarray(data_train[CONTFEATURES+DUMMYFEATURES])
# 	X_test = np.asarray(data_test[CONTFEATURES+DUMMYFEATURES])

# 	if (numFEATURESATATIME>1) & (not os.path.exists('gbm'+str(i)+'/')) :
# 		os.makedirs('gbm'+str(i)+'/')
# 	print(CONTFEATURES)
# 	prediction_residuals_train, prediction_residuals_test = ml.gbm(X_train, X_test, y_train_residuals, y_test_residuals,CONTFEATURES,DUMMYFEATURES,'gbm'+str(i)+'/')


# for feature in ALLCONTFEATURES:
# 	print(feature)
# 	cutoffs=BANDFEATURES[feature]
# 	data_train[feature+'Band']=data_train[feature].apply(ml.applyband,args=(cutoffs,))
# 	data_test[feature+'Band']=data_test[feature].apply(ml.applyband,args=(cutoffs,))
# 	ml.actualvsfittedbyfactor(data_train,data_test,feature+'Band','glm/')	



# # # Step 3
# # # Run GLM from result of CONTFEATURES GBM

# CONTFEATURES = []
# CATFEATURES = []
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
# data_train['EA_TIP_OVER_STABILITY_RATIO_ge105']=np.where(data_train['EA_TIP_OVER_STABILITY_RATIO']>=1.05, 1, 0)
# data_test['EA_TIP_OVER_STABILITY_RATIO_ge105'] =np.where( data_test['EA_TIP_OVER_STABILITY_RATIO']>=1.05, 1, 0)
# data_train['cko_hp_wheelbase_ratio_ge24']=np.where(data_train['cko_hp_wheelbase_ratio']>=2.4, 1, 0)
# data_test['cko_hp_wheelbase_ratio_ge24']=np.where(data_test['cko_hp_wheelbase_ratio']>=2.4, 1, 0)

# data_train['cko_min_trans_gear_spline4']=np.minimum(data_train['cko_min_trans_gear'],4)
# data_train['cko_min_msrp_Tr_40kspline']=np.maximum(40000,data_train['cko_min_msrp_Tr'])
# data_train['cko_min_msrp_Tr0_40kspline']=np.maximum(40000,data_train['cko_min_msrp_Tr0'])
# data_train['cko_weight_min_lt2k']=np.where(data_train['cko_weight_min']<2000,1,0)
# data_train['cko_weight_min_ge5k']=np.where(data_train['cko_weight_min']>=5000,1,0)
# data_train['cko_wheelbase_mean_lt100']=np.where(data_train['cko_wheelbase_mean']<100,1,0)
# data_train['EA_CURB_WEIGHT_lt3k']=np.where(data_train['EA_CURB_WEIGHT']<3000,1,0)
# data_train['EA_height_ge84']=np.where(data_train['EA_height']>=84,1,0)
# data_train['EA_TURNING_CIRCLE_DIAMETER_ge54']=np.where(data_train['EA_TURNING_CIRCLE_DIAMETER']>=54,1,0)
# data_train['ENG_DISPLCMNT_CI_120bw360']=np.where((120<=data_train['ENG_DISPLCMNT_CI']) & (data_train['ENG_DISPLCMNT_CI']<360),1,0)
# data_train['eng_displcmnt_L_3spline']=np.maximum(3,data_train['eng_displcmnt_L'])
# data_train['enginesize_ge65']=np.where(data_train['enginesize']>=6.5,1,0)
# data_train['MFG_BAS_MSRP_ge36k']=np.where(data_train['MFG_BAS_MSRP']>=36000,1,0)
# data_train['nDriveWheels_eq2']=np.where(data_train['nDriveWheels']==2,1,0)
# data_train['TF_Thefts_ge100']=np.where(data_train['TF_Thefts']>=100,1,0)

# data_test['cko_min_trans_gear_spline4']=np.minimum(data_test['cko_min_trans_gear'],4)
# data_test['cko_min_msrp_Tr_40kspline']=np.maximum(40000,data_test['cko_min_msrp_Tr'])
# data_test['cko_min_msrp_Tr0_40kspline']=np.maximum(40000,data_test['cko_min_msrp_Tr0'])
# data_test['cko_weight_min_lt2k']=np.where(data_test['cko_weight_min']<2000,1,0)
# data_test['cko_weight_min_ge5k']=np.where(data_test['cko_weight_min']>=5000,1,0)
# data_test['cko_wheelbase_mean_lt100']=np.where(data_test['cko_wheelbase_mean']<100,1,0)
# data_test['EA_CURB_WEIGHT_lt3k']=np.where(data_test['EA_CURB_WEIGHT']<3000,1,0)
# data_test['EA_height_ge84']=np.where(data_test['EA_height']>=84,1,0)
# data_test['EA_TURNING_CIRCLE_DIAMETER_ge54']=np.where(data_test['EA_TURNING_CIRCLE_DIAMETER']>=54,1,0)
# data_test['ENG_DISPLCMNT_CI_120bw360']=np.where((120<=data_test['ENG_DISPLCMNT_CI']) & (data_test['ENG_DISPLCMNT_CI']<360),1,0)
# data_test['eng_displcmnt_L_3spline']=np.maximum(3,data_test['eng_displcmnt_L'])
# data_test['enginesize_ge65']=np.where(data_test['enginesize']>=6.5,1,0)
# data_test['MFG_BAS_MSRP_ge36k']=np.where(data_test['MFG_BAS_MSRP']>=36000,1,0)
# data_test['nDriveWheels_eq2']=np.where(data_test['nDriveWheels']==2,1,0)
# data_test['TF_Thefts_ge100']=np.where(data_test['TF_Thefts']>=100,1,0)

# formulabase= 'lossratio ~  ENTERTAIN_CD_127  + ABS_BRK_AWO  + cko_hp_wheelbase_ratio_ge24      '
# forms=[]
# forms.append(' + EA_TIP_OVER_STABILITY_RATIO_ge105 + BODY_STYLE_CD_d4  + body_cg  + BODY_STYLE_CD_d5  + ABS_BRK_AWO + cko_eng_disp_min_d2 + cko_max_to_min_wgt_ratio + cko_min_gvwcMS ')
# forms.append(' + EA_TIP_OVER_STABILITY_RATIO_ge105 + cko_over_trans_speed_MAXMS + cko_over_trans_speed_MINMS + classCd_d5 + duty_typ_me + DOOR_CNTMS ')
# forms.append(' + EA_TIP_OVER_STABILITY_RATIO_ge105 + cyl_8 + EA_BRAKING_GFORCE_60_TO_0 + EA_ACCEL_TIME_0_TO_60 + EA_ACCEL_TIME_0_TO_60MS + EA_BRAKING_RATE_60_TO_0 ')
# forms.append(' + EA_TIP_OVER_STABILITY_RATIO_ge105 + ENG_HEAD_CNFG_CD_d3 + ENG_MFG_CD_d1 + ENG_HEAD_CNFG_CD_d2 + ENG_TRK_DUTY_TYP_CD_d1 + ENG_VLVS_PER_CLNDRMS ')
# forms.append(' + EA_TIP_OVER_STABILITY_RATIO_ge105 + FRNT_TYRE_SIZE_CD_d3  + FRNT_TYRE_SIZE_CD_d4 + FRNT_TYRE_SIZE_CD_d2 + length_VMMS + hpGrossWtRatio + hpGrossWtRatioMS ')
# forms.append(' + EA_TIP_OVER_STABILITY_RATIO_ge105 + mak_cad + MAK_NM_d2 + MAK_NM_d12 + MAK_NM_d4 + MAK_NM_d3 + mfg_desc_U_FORD_GM + NADA_GVWC2MS ')
# forms.append(' + EA_TIP_OVER_STABILITY_RATIO_ge105 + priceNew + priceNewMS + RSTRNT_TYP_CD_d8 + TRK_CAB_CNFG_CD_d4 + TRANS_OVERDRV_IND_d1 + TRK_BRK_TYP_CD_d1 ')
# forms.append(' + EA_TIP_OVER_STABILITY_RATIO_ge105 + TRK_CAB_CNFG_CD_d5 + TRK_REAR_AXL_CD_TU + wdrv_awd + WHL_BAS_LNGST_INCHS + WHL_BAS_LNGST_INCHSMS + WHL_BAS_SHRST_INCHS ')
# forms.append(' + EA_TIP_OVER_STABILITY_RATIO_ge105 + cko_min_trans_gear + cko_min_trans_gearMS + cko_min_trans_gear_spline4 + cko_min_msrp_Tr_40kspline + cko_min_msrp_Tr0_40kspline ')
# forms.append(' + EA_TIP_OVER_STABILITY_RATIO_ge105 + cko_weight_min_lt2k + cko_weight_min_ge5k + cko_wheelbase_mean_lt100 + EA_CURB_WEIGHT_lt3k + EA_height_ge84 ')
# forms.append(' + EA_TIP_OVER_STABILITY_RATIO_ge105 + EA_TURNING_CIRCLE_DIAMETER_ge54 + ENG_DISPLCMNT_CI_120bw360 + eng_displcmnt_L_3spline + enginesize_ge65 ')
# forms.append(' + EA_TIP_OVER_STABILITY_RATIO_ge105 + MFG_BAS_MSRP_ge36k + nDriveWheels_eq2 + TF_Thefts_ge100 + TF_TheftsMS ')

# forms.append(' + EA_TIP_OVER_STABILITY_RATIO_ge105 + cko_max_to_min_wgt_ratio + ENG_HEAD_CNFG_CD_d2 + ENG_VLVS_PER_CLNDRMS + length_VMMS + MAK_NM_d3 + TRANS_OVERDRV_IND_d1 + cko_weight_min_lt2k + cko_weight_min_ge5k + EA_TURNING_CIRCLE_DIAMETER_ge54  ')
# forms.append(' + EA_TIP_OVER_STABILITY_RATIO_ge105 + cko_max_to_min_wgt_ratio + length_VMMS + MAK_NM_d3 + TRANS_OVERDRV_IND_d1 + cko_weight_min_lt2k + cko_weight_min_ge5k + EA_TURNING_CIRCLE_DIAMETER_ge54  ')
# forms.append(' + cko_max_to_min_wgt_ratio + length_VMMS + MAK_NM_d3 + TRANS_OVERDRV_IND_d1 + cko_weight_min_lt2k + cko_weight_min_ge5k + EA_TURNING_CIRCLE_DIAMETER_ge54  ')
# forms.append(' + cko_max_to_min_wgt_ratio + length_VMMS + TRANS_OVERDRV_IND_d1 + cko_weight_min_lt2k + cko_weight_min_ge5k + EA_TURNING_CIRCLE_DIAMETER_ge54  ')


# for i in range(15,len(forms)):
# 	formula=formulabase+forms[i]
# 	if not os.path.exists('glm'+str(i)+'/'):
# 		os.makedirs('glm'+str(i)+'/')
# 	prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm'+str(i)+'/',0)



# # Step 4
# # Run CATFEATURES GBM

# ALLCATFEATURES = cl.returnCATFEATURES()
# CONTFEATURES = []
# CATFEATURES = []
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
# data_train['cko_hp_wheelbase_ratio_ge24']=np.where(data_train['cko_hp_wheelbase_ratio']>=2.4, 1, 0)
# data_test['cko_hp_wheelbase_ratio_ge24']=np.where(data_test['cko_hp_wheelbase_ratio']>=2.4, 1, 0)
# data_train['cko_weight_min_lt2k']=np.where(data_train['cko_weight_min']<2000,1,0)
# data_test['cko_weight_min_lt2k']=np.where(data_test['cko_weight_min']<2000,1,0)
# data_train['cko_weight_min_ge5k']=np.where(data_train['cko_weight_min']>=5000,1,0)
# data_test['cko_weight_min_ge5k']=np.where(data_test['cko_weight_min']>=5000,1,0)
# data_train['EA_TURNING_CIRCLE_DIAMETER_ge54']=np.where(data_train['EA_TURNING_CIRCLE_DIAMETER']>=54,1,0)
# data_test['EA_TURNING_CIRCLE_DIAMETER_ge54']=np.where(data_test['EA_TURNING_CIRCLE_DIAMETER']>=54,1,0)

# formula = 'lossratio ~  ENTERTAIN_CD_127  + ABS_BRK_AWO  + cko_hp_wheelbase_ratio_ge24  + cko_max_to_min_wgt_ratio + length_VMMS + TRANS_OVERDRV_IND_d1 + cko_weight_min_lt2k + cko_weight_min_ge5k + EA_TURNING_CIRCLE_DIAMETER_ge54 '
# prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm/',1)

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
# CATFEATURES = []
# CATARRAY = []
# CATARRAY.append(['ABS_BRK_CD','ABS_BRK_DESC'])
# CATARRAY.append(['BODY_STYLE_CD','BODY_STYLE_DESC','BodyStyle','bodyStyleCd'])
# CATARRAY.append(['cko_4wd','cko_abs','cko_antitheft','cko_eng_cylinders','cko_eng_cylinders_Tr','cko_esc','cko_fuel','cko_turbo_super'])
# CATARRAY.append(['classCd','classCd_Tr','EA_BODY_STYLE','EA_DRIVE_WHEELS','EA_NHTSA_STAR_RATING_CALC'])
# CATARRAY.append(['ENG_ASP_SUP_CHGR_CD','ENG_ASP_SUP_CHGR_CD_Tr','ENG_ASP_TRBL_CHGR_CD','ENG_BLCK_TYP_CD','ENG_CLNDR_RTR_CNT','ENG_FUEL_CD','ENG_FUEL_INJ_TYP_CD','ENG_HEAD_CNFG_CD'])
# CATARRAY.append(['ENG_MDL_CD'])
# CATARRAY.append(['ENG_MFG_CD','ENG_MFG_CD_Tr','ENG_TRK_DUTY_TYP_CD','engineType','fourWheelDriveCd','frameType','FRNT_TYRE_SIZE_CD','FRNT_TYRE_SIZE_Desc'])
# CATARRAY.append(['MAK_NM','MAK_NM_Tr','make','MFG_DESC'])
# CATARRAY.append(['NADA_BODY1','NC_Airbag_P','NC_DayRunningLights','NC_Drive4','NC_HD_INJ_C_D','NC_HD_INJ_C_P','NC_RearSeatHeadRestraint'])
# CATARRAY.append(['NC_TireSize','NC_TireSize4','NC_TractionControl','NC_VTStabilityControl','NC_WheelsDriven','numOfCylinders'])
# CATARRAY.append(['OPT1_ENTERTAIN_CD','PLNT_CD','REAR_TIRE_SIZE_CD','REAR_TIRE_SIZE_DESC','restraintCd','RSTRNT_TYP_CD','RSTRNT_TYP_CD_Tr'])
# CATARRAY.append(['SECUR_TYP_CD','SEGMENTATION_CD','TLT_STRNG_WHL_OPT_CD','TRANS_CD','TRANS_OVERDRV_IND','TRANS_SPEED_CD','TRK_BRK_TYP_CD_Tr','TRK_BRK_TYP_CD'])
# CATARRAY.append(['TRK_CAB_CNFG_CD','TRK_CAB_CNFG_CD_Tr','TRK_FRNT_AXL_CD','TRK_REAR_AXL_CD','TRK_TNG_RAT_CD'])
# CATARRAY.append(['ABS_BRK_CD','BODY_STYLE_CD','bodyStyleCd','classCd','ENG_MDL_CD','make','NC_Airbag_P','NC_HD_INJ_C_D','NC_TireSize','NC_TractionControl','PLNT_CD'])
# CATARRAY.append(['ABS_BRK_CD','bodyStyleCd','ENG_MDL_CD','NC_Airbag_P','NC_HD_INJ_C_D','NC_TireSize','NC_TractionControl','PLNT_CD'])
# CATARRAY.append(['ABS_BRK_CD','bodyStyleCd','ENG_MDL_CD','NC_Airbag_P','NC_TireSize','PLNT_CD'])
# CATARRAY.append(['ABS_BRK_CD'])
# CATARRAY.append(['ABS_BRK_CD','bodyStyleCd'])
# CATARRAY.append(['ABS_BRK_CD','bodyStyleCd','ENG_MDL_CD'])
# CATARRAY.append(['ABS_BRK_CD','bodyStyleCd','NC_Airbag_P'])
# # 20
# CATARRAY.append(['ABS_BRK_CD','bodyStyleCd','ENG_MDL_CD','NC_Airbag_P','NC_TireSize'])
# CATARRAY.append(['ABS_BRK_CD','bodyStyleCd','ENG_MDL_CD','NC_Airbag_P','NC_TireSize','PLNT_CD'])
# CATARRAY.append(['ABS_BRK_CD','bodyStyleCd','ENG_MDL_CD','NC_Airbag_P','NC_TireSize','PLNT_CD'])

# # Include only catfeatures that are needed for the attempted GLMs

# # Create custom variables for GLM
# data_train['cko_hp_wheelbase_ratio_ge24']=np.where(data_train['cko_hp_wheelbase_ratio']>=2.4, 1, 0)
# data_test['cko_hp_wheelbase_ratio_ge24']=np.where(data_test['cko_hp_wheelbase_ratio']>=2.4, 1, 0)
# data_train['cko_weight_min_lt2k']=np.where(data_train['cko_weight_min']<2000,1,0)
# data_test['cko_weight_min_lt2k']=np.where(data_test['cko_weight_min']<2000,1,0)
# data_train['cko_weight_min_ge5k']=np.where(data_train['cko_weight_min']>=5000,1,0)
# data_test['cko_weight_min_ge5k']=np.where(data_test['cko_weight_min']>=5000,1,0)
# data_train['EA_TURNING_CIRCLE_DIAMETER_ge54']=np.where(data_train['EA_TURNING_CIRCLE_DIAMETER']>=54,1,0)
# data_test['EA_TURNING_CIRCLE_DIAMETER_ge54']=np.where(data_test['EA_TURNING_CIRCLE_DIAMETER']>=54,1,0)


# formulabase = 'lossratio ~  ENTERTAIN_CD_127  + ABS_BRK_AWO  + cko_hp_wheelbase_ratio_ge24  + cko_max_to_min_wgt_ratio + length_VMMS + TRANS_OVERDRV_IND_d1 + cko_weight_min_lt2k + cko_weight_min_ge5k + EA_TURNING_CIRCLE_DIAMETER_ge54 '

# forms=[]
# forms.append(' + ABS_BRK_CD_4 + ABS_BRK_CD_5 + ABS_BRK_DESC_ALL_WHEEL_OPT + ABS_BRK_DESC_OTHER_STD ')
# forms.append(' + BODY_STYLE_CD_CG + BODY_STYLE_CD_PV + BODY_STYLE_DESC_VAN_CARGO + BODY_STYLE_DESC_VAN_PASSENGER + BodyStyle_PICKUP + bodyStyleCd_BUS + bodyStyleCd_KING_CAB + bodyStyleCd_PKP_4X2 + bodyStyleCd_PKP4X22D + bodyStyleCd_UTL4X24D  ')
# forms.append(' + cko_4wd_AWD + cko_abs_NONE + cko_antitheft_U + cko_eng_cylinders_5 + cko_eng_cylinders_6 + cko_eng_cylinders_Tr_6 + cko_esc_S + cko_fuel_FLEX + cko_turbo_super_N   ')
# forms.append(' + classCd__ + classCd_70 + classCd_81 + classCd_Tr_Grp5 + classCd_Tr_Grp7 + EA_BODY_STYLE_4X4_PICKUP + EA_DRIVE_WHEELS_4_WHEEL_DRIVE + EA_NHTSA_STAR_RATING_CALC___ + EA_NHTSA_STAR_RATING_CALC__   ')
# forms.append(' + ENG_ASP_SUP_CHGR_CD_N + ENG_ASP_SUP_CHGR_CD_Tr_N + ENG_ASP_TRBL_CHGR_CD_N + ENG_BLCK_TYP_CD_I + ENG_CLNDR_RTR_CNT_4 + ENG_CLNDR_RTR_CNT_6 + ENG_FUEL_CD_F + ENG_FUEL_INJ_TYP_CD_M + ENG_FUEL_INJ_TYP_CD_S + ENG_HEAD_CNFG_CD_DOHC + ENG_HEAD_CNFG_CD_OHV + ENG_HEAD_CNFG_CD_SOHC   ')
# forms.append(' + ENG_MDL_CD_050011 + ENG_MDL_CD_050020 + ENG_MDL_CD_070008 + ENG_MDL_CD_070012 + ENG_MDL_CD_070014 + ENG_MDL_CD_080020 + ENG_MDL_CD_080040 + ENG_MDL_CD_080050 + ENG_MDL_CD_180053 + ENG_MDL_CD_207030 + ENG_MDL_CD_230001 + ENG_MDL_CD_235002   ')
# forms.append(' + ENG_MFG_CD_235 + ENG_MFG_CD_Tr_Unknown + ENG_MFG_CD_U + ENG_TRK_DUTY_TYP_CD_U + engineType_F + fourWheelDriveCd_4 + frameType__ + FRNT_TYRE_SIZE_CD_31 + FRNT_TYRE_SIZE_CD_38 + FRNT_TYRE_SIZE_Desc_15R215 + FRNT_TYRE_SIZE_Desc_16R205 + FRNT_TYRE_SIZE_Desc_16R215 + FRNT_TYRE_SIZE_Desc_18R275   ')
# forms.append(' + MAK_NM_CHEVROLET + MAK_NM_DATSUN + MAK_NM_GMC + MAK_NM_MAZDA + MAK_NM_Tr_CHEVROLET + MAK_NM_Tr_GMC + MAK_NM_Tr_NISSAN + make_DODG + make_FORD + make_HOND + make_ISZU + make_MAZD + make_NSSN + make_TYTA + MFG_DESC_NISSAN + MFG_DESC_TOYOTA   ')
# forms.append(' + NADA_BODY1_K + NADA_BODY1_U + NADA_BODY1_V + NC_Airbag_P_SID + NC_Airbag_P_STD + NC_DayRunningLights_S + NC_Drive4_4WD + NC_HD_INJ_C_D_1009 + NC_HD_INJ_C_D_245 + NC_HD_INJ_C_D_310 + NC_HD_INJ_C_D_425 + NC_HD_INJ_C_D_442 + NC_HD_INJ_C_D_636 + NC_HD_INJ_C_D_852 + NC_HD_INJ_C_D_990 + NC_HD_INJ_C_P_1041 + NC_HD_INJ_C_P_1211 + NC_HD_INJ_C_P_333 + NC_HD_INJ_C_P_345 + NC_HD_INJ_C_P_386 + NC_HD_INJ_C_P_518 + NC_HD_INJ_C_P_545 + NC_HD_INJ_C_P_835 + NC_RearSeatHeadRestraint_AF   ')
# forms.append(' + NC_TireSize_235_65R16 + NC_TireSize_P205_65R15 + NC_TireSize_P205_75R15 + NC_TireSize_P255_70R16 + NC_TireSize_P265_70R16 + NC_TireSize4_P265_70R16 + NC_TractionControl_A + NC_VTStabilityControl__ + NC_VTStabilityControl_YES + NC_WheelsDriven_FWD + numOfCylinders_4 + numOfCylinders_6   ')
# forms.append(' + OPT1_ENTERTAIN_CD_4 + OPT1_ENTERTAIN_CD_A + PLNT_CD_0 + PLNT_CD_8 + PLNT_CD_E + PLNT_CD_P + PLNT_CD_R + PLNT_CD_S + PLNT_CD_W + REAR_TIRE_SIZE_CD_40 + REAR_TIRE_SIZE_DESC_16R235 + restraintCd_R + restraintCd_V + RSTRNT_TYP_CD_3 + RSTRNT_TYP_CD_4 + RSTRNT_TYP_CD_Tr_U + RSTRNT_TYP_CD_U   ')
# forms.append(' + SECUR_TYP_CD_G + SECUR_TYP_CD_S + SECUR_TYP_CD_T + SECUR_TYP_CD_U + SEGMENTATION_CD_T + TLT_STRNG_WHL_OPT_CD_U + TRANS_CD_E + TRANS_CD_M + TRANS_OVERDRV_IND_U + TRANS_SPEED_CD_4 + TRANS_SPEED_CD_6 + TRK_BRK_TYP_CD_Tr_U + TRK_BRK_TYP_CD_U   ')
# forms.append(' + TRK_CAB_CNFG_CD_CRW + TRK_CAB_CNFG_CD_EXT + TRK_CAB_CNFG_CD_Tr_2_CRW + TRK_CAB_CNFG_CD_Tr_3_EXT + TRK_CAB_CNFG_CD_VAN + TRK_FRNT_AXL_CD_U + TRK_REAR_AXL_CD_U + TRK_TNG_RAT_CD_C + TRK_TNG_RAT_CD_U ')

# forms.append(' + ABS_BRK_CD_5 + BODY_STYLE_CD_PV + bodyStyleCd_BUS + bodyStyleCd_UTL4X24D + classCd_70 + ENG_MDL_CD_050020 + ENG_MDL_CD_070008 + ENG_MDL_CD_070012 + ENG_MDL_CD_080040 + make_TYTA + NC_Airbag_P_SID + NC_HD_INJ_C_D_310 + NC_TireSize_P255_70R16 + NC_TractionControl_A + PLNT_CD_R ')
# forms.append(' + ABS_BRK_CD_5 + bodyStyleCd_BUS + bodyStyleCd_UTL4X24D  + ENG_MDL_CD_050020 + ENG_MDL_CD_070008 + ENG_MDL_CD_070012 + ENG_MDL_CD_080040 + NC_Airbag_P_SID + NC_HD_INJ_C_D_310 + NC_TireSize_P255_70R16 + NC_TractionControl_A + PLNT_CD_R - cko_max_to_min_wgt_ratio')
# forms.append(' + ABS_BRK_CD_5 + bodyStyleCd_BUS + bodyStyleCd_UTL4X24D  + ENG_MDL_CD_050020 + ENG_MDL_CD_070008 + ENG_MDL_CD_070012 + ENG_MDL_CD_080040 + NC_Airbag_P_SID  + NC_TireSize_P255_70R16  + PLNT_CD_R - cko_max_to_min_wgt_ratio')

# forms.append(' + ABS_BRK_CD_6 + ABS_BRK_CD_7 + ABS_BRK_CD_5 + ABS_BRK_CD_4 + ABS_BRK_CD_U + ABS_BRK_CD_1  + ABS_BRK_CD_3 - cko_max_to_min_wgt_ratio - cko_weight_min_ge5k - ABS_BRK_AWO  ')
# forms.append(' + bodyStyleCd_VAN_4X4 + bodyStyleCd_WAG_4X4 + bodyStyleCd_ROYL_PKP + bodyStyleCd_CSTM_PKP + bodyStyleCd_WAG_4X2 + bodyStyleCd_WAG4X23D + bodyStyleCd_VAN4X42D + bodyStyleCd_PKP4X42D  + bodyStyleCd__ + bodyStyleCd_VAN + bodyStyleCd_WAG4X24D + bodyStyleCd_PKP_4X4 + bodyStyleCd_VAN4X23D + bodyStyleCd_UTL4X24D + bodyStyleCd_WAGON_5D + bodyStyleCd_VAN_4X2 + bodyStyleCd_PKP4X43D + bodyStyleCd_TRUCK + bodyStyleCd_SUT4X24D + bodyStyleCd_SPT_PKUP + bodyStyleCd_WAGON + bodyStyleCd_WAG4D4X2 + bodyStyleCd_WAG5D4X2 + bodyStyleCd_UTL4X44D + bodyStyleCd_WAG4X43D + bodyStyleCd_WAGON_4D + bodyStyleCd_BUS + bodyStyleCd_KING_CAB + bodyStyleCd_TRK_4X2 + bodyStyleCd_PKP4X22D + bodyStyleCd_MPV_4X2 + bodyStyleCd_WAGON_3D + bodyStyleCd_WAG3D4X2 + bodyStyleCd_WAG4X44D + bodyStyleCd_UTIL_4X2 + bodyStyleCd_PICKUP + bodyStyleCd_STD_PKUP + bodyStyleCd_PKP_4X2 + bodyStyleCd_PKP4X24D + bodyStyleCd_VAN4X22D + bodyStyleCd_UTIL_4X4 + bodyStyleCd_PKP4X23D + bodyStyleCd_VAN4X24D - cko_max_to_min_wgt_ratio - cko_weight_min_ge5k - ABS_BRK_AWO  + ABS_BRK_CD_5')
# forms.append(' + ENG_MDL_CD_070009 + ENG_MDL_CD_050009 + ENG_MDL_CD_207075 + ENG_MDL_CD_050006 + ENG_MDL_CD_070007 + ENG_MDL_CD_230001 + ENG_MDL_CD_070014 + ENG_MDL_CD_080110 + ENG_MDL_CD_070010 + ENG_MDL_CD_005001 + ENG_MDL_CD_207030 + ENG_MDL_CD_180057 + ENG_MDL_CD_080003 + ENG_MDL_CD_207080 + ENG_MDL_CD_070017 + ENG_MDL_CD_150085 + ENG_MDL_CD_080009 + ENG_MDL_CD_207070 + ENG_MDL_CD_070005 + ENG_MDL_CD_080040 + ENG_MDL_CD_207010 + ENG_MDL_CD_207020 + ENG_MDL_CD_235001 + ENG_MDL_CD_070012 + ENG_MDL_CD_080012 + ENG_MDL_CD_050015 + ENG_MDL_CD_070011 + ENG_MDL_CD_080044 + ENG_MDL_CD_050011 + ENG_MDL_CD_070008 + ENG_MDL_CD_080004 + ENG_MDL_CD_080020 + ENG_MDL_CD_080015 + ENG_MDL_CD_050016 + ENG_MDL_CD_050010 + ENG_MDL_CD_080050 + ENG_MDL_CD_080084 + ENG_MDL_CD_070035 + ENG_MDL_CD_070016 + ENG_MDL_CD_050003 + ENG_MDL_CD_207050 + ENG_MDL_CD_080114 + ENG_MDL_CD_095001 + ENG_MDL_CD_070028 + ENG_MDL_CD_050012 + ENG_MDL_CD_080080 + ENG_MDL_CD_080130 + ENG_MDL_CD_207065 + ENG_MDL_CD_070015 + ENG_MDL_CD_080011 + ENG_MDL_CD_235002 + ENG_MDL_CD_050020 + ENG_MDL_CD_050053 + ENG_MDL_CD_050017 + ENG_MDL_CD_050013 + ENG_MDL_CD_080002 + ENG_MDL_CD_070020 + ENG_MDL_CD_207060 + ENG_MDL_CD_239001 + ENG_MDL_CD_080013 + ENG_MDL_CD_080070 + ENG_MDL_CD_239002 + ENG_MDL_CD_070003 + ENG_MDL_CD_050005 + ENG_MDL_CD_050014 + ENG_MDL_CD_239005 + ENG_MDL_CD_050070 + ENG_MDL_CD_070004 + ENG_MDL_CD_238004 + ENG_MDL_CD_070026 + ENG_MDL_CD_150080  + ENG_MDL_CD_050060 + ENG_MDL_CD_180053 + ENG_MDL_CD_080010 + ENG_MDL_CD_070024 + ENG_MDL_CD_070013 - cko_max_to_min_wgt_ratio - cko_weight_min_ge5k - ABS_BRK_AWO  + ABS_BRK_CD_5 + bodyStyleCd_grp1 + bodyStyleCd_grp2 ' )
# forms.append(' + NC_Airbag_P_NONE + NC_Airbag_P_STD + NC_Airbag_P_SID - cko_max_to_min_wgt_ratio - cko_weight_min_ge5k - ABS_BRK_AWO  + ABS_BRK_CD_5  + bodyStyleCd_grp1 + bodyStyleCd_grp2 ')
# # 20
# forms.append(' +  NC_TireSize_P205_65R15  + NC_TireSize_P225_60R16 + NC_TireSize_P225_70R16 + NC_TireSize_P225_65R16 + NC_TireSize_P245_70R16 + NC_TireSize_P215_55R16   + NC_TireSize_P255_65R16 + NC_TireSize_P235_75R16 + NC_TireSize_P265_70R17 + NC_TireSize_P215_75R15 + NC_TireSize_P245_70R17 + NC_TireSize_P265_70R16 + NC_TireSize_P235_70R16 + NC_TireSize_P255_70R16 + NC_TireSize_P225_60R17 + NC_TireSize_P205_75R15 + NC_TireSize_P265_70R18 + NC_TireSize_P195_65R15 + NC_TireSize_P245_75R16 + NC_TireSize_P205_70R15 + NC_TireSize_P275_65R18 + NC_TireSize_P225_70R15 + NC_TireSize_P235_75R15 + NC_TireSize_P215_65R16 + NC_TireSize_P235_70R17 + NC_TireSize_P215_70R15 + NC_TireSize_P255_65R17 + NC_TireSize_P205_55R16 + NC_TireSize_235_65R16 - cko_max_to_min_wgt_ratio - cko_weight_min_ge5k - ABS_BRK_AWO  + ABS_BRK_CD_5  + bodyStyleCd_grp1 + bodyStyleCd_grp2 + NC_Airbag_P_SID + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2')
# forms.append(' + PLNT_CD_N + PLNT_CD_H + PLNT_CD_7 + PLNT_CD_E + PLNT_CD_4 + PLNT_CD_8 + PLNT_CD_1 + PLNT_CD_P + PLNT_CD_K + PLNT_CD_Y + PLNT_CD_F + PLNT_CD_L + PLNT_CD_T + PLNT_CD_X + PLNT_CD_3 + PLNT_CD_D  + PLNT_CD_6 + PLNT_CD_0 + PLNT_CD_M + PLNT_CD_R + PLNT_CD_V + PLNT_CD_5 + PLNT_CD_G + PLNT_CD_B + PLNT_CD_J + PLNT_CD_A + PLNT_CD_S + PLNT_CD_U + PLNT_CD_W + PLNT_CD_2 + PLNT_CD_9 + PLNT_CD_C - cko_max_to_min_wgt_ratio - cko_weight_min_ge5k - ABS_BRK_AWO  + ABS_BRK_CD_5  + bodyStyleCd_grp1 + bodyStyleCd_grp2 + NC_Airbag_P_SID + NC_TireSize_grp1')
# forms.append(' + PLNT_CD_grp1 + PLNT_CD_grp2  - cko_max_to_min_wgt_ratio - cko_weight_min_ge5k - ABS_BRK_AWO  + ABS_BRK_CD_5  + bodyStyleCd_grp1 + bodyStyleCd_grp2 - bodyStyleCd_grp2 + NC_Airbag_P_SID + NC_TireSize_grp1 - TRANS_OVERDRV_IND_d1 - PLNT_CD_grp2')

# # Biggest levels + ABS_BRK_CD_2 + bodyStyleCd_PKP4X44D + ENG_MDL_CD_U + NC_Airbag_P__ +  NC_TireSize__ + NC_TireSize_P265_65_R1 + PLNT_CD_Z




# for i in range(22,len(forms)):

# 	CATFEATURES=CATARRAY[i]

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

# 	# Rename problematic dummy columns
# 	if i==3:
# 		data_train['classCd__']=data_train['classCd_.']
# 		data_test['classCd__']=data_test['classCd_.']
# 		data_train['EA_NHTSA_STAR_RATING_CALC___']=data_train['EA_NHTSA_STAR_RATING_CALC_**']
# 		data_test['EA_NHTSA_STAR_RATING_CALC___']=data_test['EA_NHTSA_STAR_RATING_CALC_**']
# 		data_train['EA_NHTSA_STAR_RATING_CALC__']=data_train['EA_NHTSA_STAR_RATING_CALC_.']
# 		data_test['EA_NHTSA_STAR_RATING_CALC__']=data_test['EA_NHTSA_STAR_RATING_CALC_.']
# 	elif i==6:	
# 		data_train['frameType__']=data_train['frameType_.']
# 		data_test['frameType__']=data_test['frameType_.']
# 	elif i==9:
# 		data_train['NC_TireSize_235_65R16']=data_train['NC_TireSize_235/65R16']
# 		data_test['NC_TireSize_235_65R16']=data_test['NC_TireSize_235/65R16']
# 		data_train['NC_TireSize_P205_65R15']=data_train['NC_TireSize_P205/65R15']
# 		data_test['NC_TireSize_P205_65R15']=data_test['NC_TireSize_P205/65R15']
# 		data_train['NC_TireSize_P205_75R15']=data_train['NC_TireSize_P205/75R15']
# 		data_test['NC_TireSize_P205_75R15']=data_test['NC_TireSize_P205/75R15']
# 		data_train['NC_TireSize_P255_70R16']=data_train['NC_TireSize_P255/70R16']
# 		data_test['NC_TireSize_P255_70R16']=data_test['NC_TireSize_P255/70R16']
# 		data_train['NC_TireSize_P265_70R16']=data_train['NC_TireSize_P265/70R16']
# 		data_test['NC_TireSize_P265_70R16']=data_test['NC_TireSize_P265/70R16']
# 		data_train['NC_TireSize4_P265_70R16']=data_train['NC_TireSize4_P265/70R16']
# 		data_test['NC_TireSize4_P265_70R16']=data_test['NC_TireSize4_P265/70R16']
# 		data_train['NC_VTStabilityControl__']=data_train['NC_VTStabilityControl_.']
# 		data_test['NC_VTStabilityControl__']=data_test['NC_VTStabilityControl_.']
# 	elif i==12:	
# 		data_train['TRK_CAB_CNFG_CD_Tr_2_CRW']=data_train['TRK_CAB_CNFG_CD_Tr_2.CRW']
# 		data_test['TRK_CAB_CNFG_CD_Tr_2_CRW']=data_test['TRK_CAB_CNFG_CD_Tr_2.CRW']
# 		data_train['TRK_CAB_CNFG_CD_Tr_3_EXT']=data_train['TRK_CAB_CNFG_CD_Tr_3.EXT']
# 		data_test['TRK_CAB_CNFG_CD_Tr_3_EXT']=data_test['TRK_CAB_CNFG_CD_Tr_3.EXT']
# 	elif i==13:
# 		data_train['NC_TireSize_P255_70R16']=data_train['NC_TireSize_P255/70R16']
# 		data_test['NC_TireSize_P255_70R16']=data_test['NC_TireSize_P255/70R16']
# 		data_train['classCd__']=data_train['classCd_.']
# 		data_test['classCd__']=data_test['classCd_.']
# 	elif i==14:	
# 		data_train['NC_TireSize_P255_70R16']=data_train['NC_TireSize_P255/70R16']
# 		data_test['NC_TireSize_P255_70R16']=data_test['NC_TireSize_P255/70R16']
# 	elif i==15:	
# 		data_train['NC_TireSize_P255_70R16']=data_train['NC_TireSize_P255/70R16']
# 		data_test['NC_TireSize_P255_70R16']=data_test['NC_TireSize_P255/70R16']	
# 	elif i==17:	
# 		data_train['bodyStyleCd__']=data_train['bodyStyleCd_.']
# 		data_test['bodyStyleCd__']=data_test['bodyStyleCd_.']		
# 	elif i==18:
# 		data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_ROYL_PKP']+data_train['bodyStyleCd_WAGON_5D']+data_train['bodyStyleCd_UTIL_4X4']+data_train['bodyStyleCd_VAN']+data_train['bodyStyleCd_SPT_PKUP']+data_train['bodyStyleCd_BUS']+data_train['bodyStyleCd_CSTM_PKP']+data_train['bodyStyleCd_WAGON_3D']+data_train['bodyStyleCd_TRUCK']+data_train['bodyStyleCd_WAGON']+data_train['bodyStyleCd_STD_PKUP']+data_train['bodyStyleCd_TRK_4X2'])
# 		data_train['bodyStyleCd_grp2']=(data_train['bodyStyleCd_WAGON_4D']+data_train['bodyStyleCd_SUT4X24D']+data_train['bodyStyleCd_VAN4X23D']+data_train['bodyStyleCd_WAG4D4X2']+data_train['bodyStyleCd_WAG_4X4']+data_train['bodyStyleCd_WAG4X43D']+data_train['bodyStyleCd_UTIL_4X2']+data_train['bodyStyleCd_VAN_4X4']+data_train['bodyStyleCd_WAG4X23D']+data_train['bodyStyleCd_WAG_4X2']+data_train['bodyStyleCd_PKP_4X4']+data_train['bodyStyleCd_PKP4X43D']+data_train['bodyStyleCd_PKP4X23D'])
# 		data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_ROYL_PKP']+data_test['bodyStyleCd_WAGON_5D']+data_test['bodyStyleCd_UTIL_4X4']+data_test['bodyStyleCd_VAN']+data_test['bodyStyleCd_SPT_PKUP']+data_test['bodyStyleCd_BUS']+data_test['bodyStyleCd_CSTM_PKP']+data_test['bodyStyleCd_WAGON_3D']+data_test['bodyStyleCd_TRUCK']+data_test['bodyStyleCd_WAGON']+data_test['bodyStyleCd_STD_PKUP']+data_test['bodyStyleCd_TRK_4X2'])
# 		data_test['bodyStyleCd_grp2']=(data_test['bodyStyleCd_WAGON_4D']+data_test['bodyStyleCd_SUT4X24D']+data_test['bodyStyleCd_VAN4X23D']+data_test['bodyStyleCd_WAG4D4X2']+data_test['bodyStyleCd_WAG_4X4']+data_test['bodyStyleCd_WAG4X43D']+data_test['bodyStyleCd_UTIL_4X2']+data_test['bodyStyleCd_VAN_4X4']+data_test['bodyStyleCd_WAG4X23D']+data_test['bodyStyleCd_WAG_4X2']+data_test['bodyStyleCd_PKP_4X4']+data_test['bodyStyleCd_PKP4X43D']+data_test['bodyStyleCd_PKP4X23D'])
# 	elif i==19:	
# 		# data_train['NC_Airbag_P__']=data_train['NC_Airbag_P_.']
# 		# data_test['NC_Airbag_P__']=data_test['NC_Airbag_P_.']	
# 		data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_ROYL_PKP']+data_train['bodyStyleCd_WAGON_5D']+data_train['bodyStyleCd_UTIL_4X4']+data_train['bodyStyleCd_VAN']+data_train['bodyStyleCd_SPT_PKUP']+data_train['bodyStyleCd_BUS']+data_train['bodyStyleCd_CSTM_PKP']+data_train['bodyStyleCd_WAGON_3D']+data_train['bodyStyleCd_TRUCK']+data_train['bodyStyleCd_WAGON']+data_train['bodyStyleCd_STD_PKUP']+data_train['bodyStyleCd_TRK_4X2'])
# 		data_train['bodyStyleCd_grp2']=(data_train['bodyStyleCd_WAGON_4D']+data_train['bodyStyleCd_SUT4X24D']+data_train['bodyStyleCd_VAN4X23D']+data_train['bodyStyleCd_WAG4D4X2']+data_train['bodyStyleCd_WAG_4X4']+data_train['bodyStyleCd_WAG4X43D']+data_train['bodyStyleCd_UTIL_4X2']+data_train['bodyStyleCd_VAN_4X4']+data_train['bodyStyleCd_WAG4X23D']+data_train['bodyStyleCd_WAG_4X2']+data_train['bodyStyleCd_PKP_4X4']+data_train['bodyStyleCd_PKP4X43D']+data_train['bodyStyleCd_PKP4X23D'])
# 		data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_ROYL_PKP']+data_test['bodyStyleCd_WAGON_5D']+data_test['bodyStyleCd_UTIL_4X4']+data_test['bodyStyleCd_VAN']+data_test['bodyStyleCd_SPT_PKUP']+data_test['bodyStyleCd_BUS']+data_test['bodyStyleCd_CSTM_PKP']+data_test['bodyStyleCd_WAGON_3D']+data_test['bodyStyleCd_TRUCK']+data_test['bodyStyleCd_WAGON']+data_test['bodyStyleCd_STD_PKUP']+data_test['bodyStyleCd_TRK_4X2'])
# 		data_test['bodyStyleCd_grp2']=(data_test['bodyStyleCd_WAGON_4D']+data_test['bodyStyleCd_SUT4X24D']+data_test['bodyStyleCd_VAN4X23D']+data_test['bodyStyleCd_WAG4D4X2']+data_test['bodyStyleCd_WAG_4X4']+data_test['bodyStyleCd_WAG4X43D']+data_test['bodyStyleCd_UTIL_4X2']+data_test['bodyStyleCd_VAN_4X4']+data_test['bodyStyleCd_WAG4X23D']+data_test['bodyStyleCd_WAG_4X2']+data_test['bodyStyleCd_PKP_4X4']+data_test['bodyStyleCd_PKP4X43D']+data_test['bodyStyleCd_PKP4X23D'])
# 	elif i==20:	
# 		data_train['NC_TireSize_P205_65R15']=data_train['NC_TireSize_P205/65R15']	
# 		data_test['NC_TireSize_P205_65R15']=data_test['NC_TireSize_P205/65R15']
# 		data_train['NC_TireSize_P265_65_R17']=data_train['NC_TireSize_P265/65/R17']	
# 		data_test['NC_TireSize_P265_65_R17']=data_test['NC_TireSize_P265/65/R17']
# 		data_train['NC_TireSize_P225_60R16']=data_train['NC_TireSize_P225/60R16']	
# 		data_test['NC_TireSize_P225_60R16']=data_test['NC_TireSize_P225/60R16']
# 		data_train['NC_TireSize_P225_70R16']=data_train['NC_TireSize_P225/70R16']	
# 		data_test['NC_TireSize_P225_70R16']=data_test['NC_TireSize_P225/70R16']
# 		data_train['NC_TireSize_P225_65R16']=data_train['NC_TireSize_P225/65R16']	
# 		data_test['NC_TireSize_P225_65R16']=data_test['NC_TireSize_P225/65R16']
# 		data_train['NC_TireSize_P245_70R16']=data_train['NC_TireSize_P245/70R16']	
# 		data_test['NC_TireSize_P245_70R16']=data_test['NC_TireSize_P245/70R16']
# 		data_train['NC_TireSize_P215_55R16']=data_train['NC_TireSize_P215/55R16']	
# 		data_test['NC_TireSize_P215_55R16']=data_test['NC_TireSize_P215/55R16']
# 		# data_train['NC_TireSize__']=data_train['NC_TireSize_.']	
# 		# data_test['NC_TireSize__']=data_test['NC_TireSize_.']
# 		data_train['NC_TireSize_P255_65R16']=data_train['NC_TireSize_P255/65R16']	
# 		data_test['NC_TireSize_P255_65R16']=data_test['NC_TireSize_P255/65R16']
# 		data_train['NC_TireSize_P235_75R16']=data_train['NC_TireSize_P235/75R16']	
# 		data_test['NC_TireSize_P235_75R16']=data_test['NC_TireSize_P235/75R16']
# 		data_train['NC_TireSize_P265_70R17']=data_train['NC_TireSize_P265/70R17']	
# 		data_test['NC_TireSize_P265_70R17']=data_test['NC_TireSize_P265/70R17']
# 		data_train['NC_TireSize_P215_75R15']=data_train['NC_TireSize_P215/75R15']	
# 		data_test['NC_TireSize_P215_75R15']=data_test['NC_TireSize_P215/75R15']
# 		data_train['NC_TireSize_P245_70R17']=data_train['NC_TireSize_P245/70R17']	
# 		data_test['NC_TireSize_P245_70R17']=data_test['NC_TireSize_P245/70R17']
# 		data_train['NC_TireSize_P265_70R16']=data_train['NC_TireSize_P265/70R16']	
# 		data_test['NC_TireSize_P265_70R16']=data_test['NC_TireSize_P265/70R16']
# 		data_train['NC_TireSize_P235_70R16']=data_train['NC_TireSize_P235/70R16']	
# 		data_test['NC_TireSize_P235_70R16']=data_test['NC_TireSize_P235/70R16']
# 		data_train['NC_TireSize_P255_70R16']=data_train['NC_TireSize_P255/70R16']	
# 		data_test['NC_TireSize_P255_70R16']=data_test['NC_TireSize_P255/70R16']
# 		data_train['NC_TireSize_P225_60R17']=data_train['NC_TireSize_P225/60R17']	
# 		data_test['NC_TireSize_P225_60R17']=data_test['NC_TireSize_P225/60R17']
# 		data_train['NC_TireSize_P205_75R15']=data_train['NC_TireSize_P205/75R15']	
# 		data_test['NC_TireSize_P205_75R15']=data_test['NC_TireSize_P205/75R15']
# 		data_train['NC_TireSize_P265_70R18']=data_train['NC_TireSize_P265/70R18']	
# 		data_test['NC_TireSize_P265_70R18']=data_test['NC_TireSize_P265/70R18']
# 		data_train['NC_TireSize_P195_65R15']=data_train['NC_TireSize_P195/65R15']	
# 		data_test['NC_TireSize_P195_65R15']=data_test['NC_TireSize_P195/65R15']
# 		data_train['NC_TireSize_P245_75R16']=data_train['NC_TireSize_P245/75R16']	
# 		data_test['NC_TireSize_P245_75R16']=data_test['NC_TireSize_P245/75R16']
# 		data_train['NC_TireSize_P205_70R15']=data_train['NC_TireSize_P205/70R15']	
# 		data_test['NC_TireSize_P205_70R15']=data_test['NC_TireSize_P205/70R15']
# 		data_train['NC_TireSize_P275_65R18']=data_train['NC_TireSize_P275/65R18']	
# 		data_test['NC_TireSize_P275_65R18']=data_test['NC_TireSize_P275/65R18']
# 		data_train['NC_TireSize_P225_70R15']=data_train['NC_TireSize_P225/70R15']	
# 		data_test['NC_TireSize_P225_70R15']=data_test['NC_TireSize_P225/70R15']
# 		data_train['NC_TireSize_P235_75R15']=data_train['NC_TireSize_P235/75R15']	
# 		data_test['NC_TireSize_P235_75R15']=data_test['NC_TireSize_P235/75R15']
# 		data_train['NC_TireSize_P215_65R16']=data_train['NC_TireSize_P215/65R16']	
# 		data_test['NC_TireSize_P215_65R16']=data_test['NC_TireSize_P215/65R16']
# 		data_train['NC_TireSize_P235_70R17']=data_train['NC_TireSize_P235/70R17']	
# 		data_test['NC_TireSize_P235_70R17']=data_test['NC_TireSize_P235/70R17']
# 		data_train['NC_TireSize_P215_70R15']=data_train['NC_TireSize_P215/70R15']	
# 		data_test['NC_TireSize_P215_70R15']=data_test['NC_TireSize_P215/70R15']
# 		data_train['NC_TireSize_P255_65R17']=data_train['NC_TireSize_P255/65R17']	
# 		data_test['NC_TireSize_P255_65R17']=data_test['NC_TireSize_P255/65R17']
# 		data_train['NC_TireSize_P205_55R16']=data_train['NC_TireSize_P205/55R16']	
# 		data_test['NC_TireSize_P205_55R16']=data_test['NC_TireSize_P205/55R16']
# 		data_train['NC_TireSize_235_65R16']=data_train['NC_TireSize_235/65R16']	
# 		data_test['NC_TireSize_235_65R16']=data_test['NC_TireSize_235/65R16']
	
# 		data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_ROYL_PKP']+data_train['bodyStyleCd_WAGON_5D']+data_train['bodyStyleCd_UTIL_4X4']+data_train['bodyStyleCd_VAN']+data_train['bodyStyleCd_SPT_PKUP']+data_train['bodyStyleCd_BUS']+data_train['bodyStyleCd_CSTM_PKP']+data_train['bodyStyleCd_WAGON_3D']+data_train['bodyStyleCd_TRUCK']+data_train['bodyStyleCd_WAGON']+data_train['bodyStyleCd_STD_PKUP']+data_train['bodyStyleCd_TRK_4X2'])
# 		data_train['bodyStyleCd_grp2']=(data_train['bodyStyleCd_WAGON_4D']+data_train['bodyStyleCd_SUT4X24D']+data_train['bodyStyleCd_VAN4X23D']+data_train['bodyStyleCd_WAG4D4X2']+data_train['bodyStyleCd_WAG_4X4']+data_train['bodyStyleCd_WAG4X43D']+data_train['bodyStyleCd_UTIL_4X2']+data_train['bodyStyleCd_VAN_4X4']+data_train['bodyStyleCd_WAG4X23D']+data_train['bodyStyleCd_WAG_4X2']+data_train['bodyStyleCd_PKP_4X4']+data_train['bodyStyleCd_PKP4X43D']+data_train['bodyStyleCd_PKP4X23D'])
# 		data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_ROYL_PKP']+data_test['bodyStyleCd_WAGON_5D']+data_test['bodyStyleCd_UTIL_4X4']+data_test['bodyStyleCd_VAN']+data_test['bodyStyleCd_SPT_PKUP']+data_test['bodyStyleCd_BUS']+data_test['bodyStyleCd_CSTM_PKP']+data_test['bodyStyleCd_WAGON_3D']+data_test['bodyStyleCd_TRUCK']+data_test['bodyStyleCd_WAGON']+data_test['bodyStyleCd_STD_PKUP']+data_test['bodyStyleCd_TRK_4X2'])
# 		data_test['bodyStyleCd_grp2']=(data_test['bodyStyleCd_WAGON_4D']+data_test['bodyStyleCd_SUT4X24D']+data_test['bodyStyleCd_VAN4X23D']+data_test['bodyStyleCd_WAG4D4X2']+data_test['bodyStyleCd_WAG_4X4']+data_test['bodyStyleCd_WAG4X43D']+data_test['bodyStyleCd_UTIL_4X2']+data_test['bodyStyleCd_VAN_4X4']+data_test['bodyStyleCd_WAG4X23D']+data_test['bodyStyleCd_WAG_4X2']+data_test['bodyStyleCd_PKP_4X4']+data_test['bodyStyleCd_PKP4X43D']+data_test['bodyStyleCd_PKP4X23D'])
# 		data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_150080']+data_train['ENG_MDL_CD_070028']+data_train['ENG_MDL_CD_080002']+data_train['ENG_MDL_CD_005001']+data_train['ENG_MDL_CD_150085']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_050053']+data_train['ENG_MDL_CD_080110'])
# 		data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_070035']+data_train['ENG_MDL_CD_080009']+data_train['ENG_MDL_CD_080004']+data_train['ENG_MDL_CD_235001']+data_train['ENG_MDL_CD_070024']+data_train['ENG_MDL_CD_095001']+data_train['ENG_MDL_CD_050014']+data_train['ENG_MDL_CD_070008']+data_train['ENG_MDL_CD_070012']+data_train['ENG_MDL_CD_080040']+data_train['ENG_MDL_CD_207065']+data_train['ENG_MDL_CD_050070']+data_train['ENG_MDL_CD_050020'])
# 		data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_150080']+data_test['ENG_MDL_CD_070028']+data_test['ENG_MDL_CD_080002']+data_test['ENG_MDL_CD_005001']+data_test['ENG_MDL_CD_150085']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_050053']+data_test['ENG_MDL_CD_080110'])
# 		data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_070035']+data_test['ENG_MDL_CD_080009']+data_test['ENG_MDL_CD_080004']+data_test['ENG_MDL_CD_235001']+data_test['ENG_MDL_CD_070024']+data_test['ENG_MDL_CD_095001']+data_test['ENG_MDL_CD_050014']+data_test['ENG_MDL_CD_070008']+data_test['ENG_MDL_CD_070012']+data_test['ENG_MDL_CD_080040']+data_test['ENG_MDL_CD_207065']+data_test['ENG_MDL_CD_050070']+data_test['ENG_MDL_CD_050020'])
# 	elif i==21:		
# 		data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_ROYL_PKP']+data_train['bodyStyleCd_WAGON_5D']+data_train['bodyStyleCd_UTIL_4X4']+data_train['bodyStyleCd_VAN']+data_train['bodyStyleCd_SPT_PKUP']+data_train['bodyStyleCd_BUS']+data_train['bodyStyleCd_CSTM_PKP']+data_train['bodyStyleCd_WAGON_3D']+data_train['bodyStyleCd_TRUCK']+data_train['bodyStyleCd_WAGON']+data_train['bodyStyleCd_STD_PKUP']+data_train['bodyStyleCd_TRK_4X2'])
# 		data_train['bodyStyleCd_grp2']=(data_train['bodyStyleCd_WAGON_4D']+data_train['bodyStyleCd_SUT4X24D']+data_train['bodyStyleCd_VAN4X23D']+data_train['bodyStyleCd_WAG4D4X2']+data_train['bodyStyleCd_WAG_4X4']+data_train['bodyStyleCd_WAG4X43D']+data_train['bodyStyleCd_UTIL_4X2']+data_train['bodyStyleCd_VAN_4X4']+data_train['bodyStyleCd_WAG4X23D']+data_train['bodyStyleCd_WAG_4X2']+data_train['bodyStyleCd_PKP_4X4']+data_train['bodyStyleCd_PKP4X43D']+data_train['bodyStyleCd_PKP4X23D'])
# 		data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_ROYL_PKP']+data_test['bodyStyleCd_WAGON_5D']+data_test['bodyStyleCd_UTIL_4X4']+data_test['bodyStyleCd_VAN']+data_test['bodyStyleCd_SPT_PKUP']+data_test['bodyStyleCd_BUS']+data_test['bodyStyleCd_CSTM_PKP']+data_test['bodyStyleCd_WAGON_3D']+data_test['bodyStyleCd_TRUCK']+data_test['bodyStyleCd_WAGON']+data_test['bodyStyleCd_STD_PKUP']+data_test['bodyStyleCd_TRK_4X2'])
# 		data_test['bodyStyleCd_grp2']=(data_test['bodyStyleCd_WAGON_4D']+data_test['bodyStyleCd_SUT4X24D']+data_test['bodyStyleCd_VAN4X23D']+data_test['bodyStyleCd_WAG4D4X2']+data_test['bodyStyleCd_WAG_4X4']+data_test['bodyStyleCd_WAG4X43D']+data_test['bodyStyleCd_UTIL_4X2']+data_test['bodyStyleCd_VAN_4X4']+data_test['bodyStyleCd_WAG4X23D']+data_test['bodyStyleCd_WAG_4X2']+data_test['bodyStyleCd_PKP_4X4']+data_test['bodyStyleCd_PKP4X43D']+data_test['bodyStyleCd_PKP4X23D'])
# 		data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_150080']+data_train['ENG_MDL_CD_070028']+data_train['ENG_MDL_CD_080002']+data_train['ENG_MDL_CD_005001']+data_train['ENG_MDL_CD_150085']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_050053']+data_train['ENG_MDL_CD_080110'])
# 		data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_070035']+data_train['ENG_MDL_CD_080009']+data_train['ENG_MDL_CD_080004']+data_train['ENG_MDL_CD_235001']+data_train['ENG_MDL_CD_070024']+data_train['ENG_MDL_CD_095001']+data_train['ENG_MDL_CD_050014']+data_train['ENG_MDL_CD_070008']+data_train['ENG_MDL_CD_070012']+data_train['ENG_MDL_CD_080040']+data_train['ENG_MDL_CD_207065']+data_train['ENG_MDL_CD_050070']+data_train['ENG_MDL_CD_050020'])
# 		data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_150080']+data_test['ENG_MDL_CD_070028']+data_test['ENG_MDL_CD_080002']+data_test['ENG_MDL_CD_005001']+data_test['ENG_MDL_CD_150085']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_050053']+data_test['ENG_MDL_CD_080110'])
# 		data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_070035']+data_test['ENG_MDL_CD_080009']+data_test['ENG_MDL_CD_080004']+data_test['ENG_MDL_CD_235001']+data_test['ENG_MDL_CD_070024']+data_test['ENG_MDL_CD_095001']+data_test['ENG_MDL_CD_050014']+data_test['ENG_MDL_CD_070008']+data_test['ENG_MDL_CD_070012']+data_test['ENG_MDL_CD_080040']+data_test['ENG_MDL_CD_207065']+data_test['ENG_MDL_CD_050070']+data_test['ENG_MDL_CD_050020'])
# 		data_train['NC_TireSize_grp1']=(data_train['NC_TireSize_P255/70R16']+data_train['NC_TireSize_P265/70R17']+data_train['NC_TireSize_P225/70R16'])
# 		data_test['NC_TireSize_grp1']=(data_test['NC_TireSize_P255/70R16']+data_test['NC_TireSize_P265/70R17']+data_test['NC_TireSize_P225/70R16'])
# 	elif i==22:
# 		data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_ROYL_PKP']+data_train['bodyStyleCd_WAGON_5D']+data_train['bodyStyleCd_UTIL_4X4']+data_train['bodyStyleCd_VAN']+data_train['bodyStyleCd_SPT_PKUP']+data_train['bodyStyleCd_BUS']+data_train['bodyStyleCd_CSTM_PKP']+data_train['bodyStyleCd_WAGON_3D']+data_train['bodyStyleCd_TRUCK']+data_train['bodyStyleCd_WAGON']+data_train['bodyStyleCd_STD_PKUP']+data_train['bodyStyleCd_TRK_4X2'])
# 		data_train['bodyStyleCd_grp2']=(data_train['bodyStyleCd_WAGON_4D']+data_train['bodyStyleCd_SUT4X24D']+data_train['bodyStyleCd_VAN4X23D']+data_train['bodyStyleCd_WAG4D4X2']+data_train['bodyStyleCd_WAG_4X4']+data_train['bodyStyleCd_WAG4X43D']+data_train['bodyStyleCd_UTIL_4X2']+data_train['bodyStyleCd_VAN_4X4']+data_train['bodyStyleCd_WAG4X23D']+data_train['bodyStyleCd_WAG_4X2']+data_train['bodyStyleCd_PKP_4X4']+data_train['bodyStyleCd_PKP4X43D']+data_train['bodyStyleCd_PKP4X23D'])
# 		data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_ROYL_PKP']+data_test['bodyStyleCd_WAGON_5D']+data_test['bodyStyleCd_UTIL_4X4']+data_test['bodyStyleCd_VAN']+data_test['bodyStyleCd_SPT_PKUP']+data_test['bodyStyleCd_BUS']+data_test['bodyStyleCd_CSTM_PKP']+data_test['bodyStyleCd_WAGON_3D']+data_test['bodyStyleCd_TRUCK']+data_test['bodyStyleCd_WAGON']+data_test['bodyStyleCd_STD_PKUP']+data_test['bodyStyleCd_TRK_4X2'])
# 		data_test['bodyStyleCd_grp2']=(data_test['bodyStyleCd_WAGON_4D']+data_test['bodyStyleCd_SUT4X24D']+data_test['bodyStyleCd_VAN4X23D']+data_test['bodyStyleCd_WAG4D4X2']+data_test['bodyStyleCd_WAG_4X4']+data_test['bodyStyleCd_WAG4X43D']+data_test['bodyStyleCd_UTIL_4X2']+data_test['bodyStyleCd_VAN_4X4']+data_test['bodyStyleCd_WAG4X23D']+data_test['bodyStyleCd_WAG_4X2']+data_test['bodyStyleCd_PKP_4X4']+data_test['bodyStyleCd_PKP4X43D']+data_test['bodyStyleCd_PKP4X23D'])
# 		data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_150080']+data_train['ENG_MDL_CD_070028']+data_train['ENG_MDL_CD_080002']+data_train['ENG_MDL_CD_005001']+data_train['ENG_MDL_CD_150085']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_050053']+data_train['ENG_MDL_CD_080110'])
# 		data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_070035']+data_train['ENG_MDL_CD_080009']+data_train['ENG_MDL_CD_080004']+data_train['ENG_MDL_CD_235001']+data_train['ENG_MDL_CD_070024']+data_train['ENG_MDL_CD_095001']+data_train['ENG_MDL_CD_050014']+data_train['ENG_MDL_CD_070008']+data_train['ENG_MDL_CD_070012']+data_train['ENG_MDL_CD_080040']+data_train['ENG_MDL_CD_207065']+data_train['ENG_MDL_CD_050070']+data_train['ENG_MDL_CD_050020'])
# 		data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_150080']+data_test['ENG_MDL_CD_070028']+data_test['ENG_MDL_CD_080002']+data_test['ENG_MDL_CD_005001']+data_test['ENG_MDL_CD_150085']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_050053']+data_test['ENG_MDL_CD_080110'])
# 		data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_070035']+data_test['ENG_MDL_CD_080009']+data_test['ENG_MDL_CD_080004']+data_test['ENG_MDL_CD_235001']+data_test['ENG_MDL_CD_070024']+data_test['ENG_MDL_CD_095001']+data_test['ENG_MDL_CD_050014']+data_test['ENG_MDL_CD_070008']+data_test['ENG_MDL_CD_070012']+data_test['ENG_MDL_CD_080040']+data_test['ENG_MDL_CD_207065']+data_test['ENG_MDL_CD_050070']+data_test['ENG_MDL_CD_050020'])
# 		data_train['NC_TireSize_grp1']=(data_train['NC_TireSize_P255/70R16']+data_train['NC_TireSize_P265/70R17']+data_train['NC_TireSize_P225/70R16'])
# 		data_test['NC_TireSize_grp1']=(data_test['NC_TireSize_P255/70R16']+data_test['NC_TireSize_P265/70R17']+data_test['NC_TireSize_P225/70R16'])
# 		data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_V']+data_train['PLNT_CD_A'])
# 		data_train['PLNT_CD_grp2']=(data_train['PLNT_CD_Y']+data_train['PLNT_CD_7']+data_test['PLNT_CD_R'])
# 		data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_V']+data_test['PLNT_CD_A'])
# 		data_test['PLNT_CD_grp2']=(data_test['PLNT_CD_Y']+data_test['PLNT_CD_7']+data_test['PLNT_CD_R'])



# 	formula=formulabase+forms[i]
# 	if not os.path.exists('glm'+str(i)+'/'):
# 		os.makedirs('glm'+str(i)+'/')
# 	prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm'+str(i)+'/',0)



# Step 6
# Finalize GLM

# CONTFEATURES = []
# CATFEATURES = ['ABS_BRK_CD','bodyStyleCd','ENG_MDL_CD','NC_Airbag_P','NC_TireSize','PLNT_CD','BODY_STYLE_CD','TRANS_OVERDRV_IND']
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

# # # Create custom variables for GLM
# data_train['cko_hp_wheelbase_ratio_ge24']=np.where(data_train['cko_hp_wheelbase_ratio']>=2.4, 1, 0)
# data_test['cko_hp_wheelbase_ratio_ge24']=np.where(data_test['cko_hp_wheelbase_ratio']>=2.4, 1, 0)
# data_train['cko_weight_min_lt2k']=np.where(data_train['cko_weight_min']<2000,1,0)
# data_test['cko_weight_min_lt2k']=np.where(data_test['cko_weight_min']<2000,1,0)
# # data_train['cko_weight_min_ge5k']=np.where(data_train['cko_weight_min']>=5000,1,0)
# # data_test['cko_weight_min_ge5k']=np.where(data_test['cko_weight_min']>=5000,1,0)
# data_train['EA_TURNING_CIRCLE_DIAMETER_ge54']=np.where(data_train['EA_TURNING_CIRCLE_DIAMETER']>=54,1,0)
# data_test['EA_TURNING_CIRCLE_DIAMETER_ge54']=np.where(data_test['EA_TURNING_CIRCLE_DIAMETER']>=54,1,0)
# # data_train['priceNew_35kspline']=np.maximum(35000,data_train['priceNew'])
# # data_test['priceNew_35kspline']=np.maximum(35000,data_test['priceNew'])



# data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_ROYL_PKP']+data_train['bodyStyleCd_WAGON_5D']+data_train['bodyStyleCd_UTIL_4X4']+data_train['bodyStyleCd_VAN']+data_train['bodyStyleCd_SPT_PKUP']+data_train['bodyStyleCd_BUS']+data_train['bodyStyleCd_CSTM_PKP']+data_train['bodyStyleCd_WAGON_3D']+data_train['bodyStyleCd_TRUCK']+data_train['bodyStyleCd_WAGON']+data_train['bodyStyleCd_STD_PKUP']+data_train['bodyStyleCd_TRK_4X2'])
# data_train['bodyStyleCd_grp2']=(data_train['bodyStyleCd_WAGON_4D']+data_train['bodyStyleCd_SUT4X24D']+data_train['bodyStyleCd_VAN4X23D']+data_train['bodyStyleCd_WAG4D4X2']+data_train['bodyStyleCd_WAG_4X4']+data_train['bodyStyleCd_WAG4X43D']+data_train['bodyStyleCd_UTIL_4X2']+data_train['bodyStyleCd_VAN_4X4']+data_train['bodyStyleCd_WAG4X23D']+data_train['bodyStyleCd_WAG_4X2']+data_train['bodyStyleCd_PKP_4X4']+data_train['bodyStyleCd_PKP4X43D']+data_train['bodyStyleCd_PKP4X23D'])
# data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_ROYL_PKP']+data_test['bodyStyleCd_WAGON_5D']+data_test['bodyStyleCd_UTIL_4X4']+data_test['bodyStyleCd_VAN']+data_test['bodyStyleCd_SPT_PKUP']+data_test['bodyStyleCd_BUS']+data_test['bodyStyleCd_CSTM_PKP']+data_test['bodyStyleCd_WAGON_3D']+data_test['bodyStyleCd_TRUCK']+data_test['bodyStyleCd_WAGON']+data_test['bodyStyleCd_STD_PKUP']+data_test['bodyStyleCd_TRK_4X2'])
# data_test['bodyStyleCd_grp2']=(data_test['bodyStyleCd_WAGON_4D']+data_test['bodyStyleCd_SUT4X24D']+data_test['bodyStyleCd_VAN4X23D']+data_test['bodyStyleCd_WAG4D4X2']+data_test['bodyStyleCd_WAG_4X4']+data_test['bodyStyleCd_WAG4X43D']+data_test['bodyStyleCd_UTIL_4X2']+data_test['bodyStyleCd_VAN_4X4']+data_test['bodyStyleCd_WAG4X23D']+data_test['bodyStyleCd_WAG_4X2']+data_test['bodyStyleCd_PKP_4X4']+data_test['bodyStyleCd_PKP4X43D']+data_test['bodyStyleCd_PKP4X23D'])
# data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_150080']+data_train['ENG_MDL_CD_070028']+data_train['ENG_MDL_CD_080002']+data_train['ENG_MDL_CD_005001']+data_train['ENG_MDL_CD_150085']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_050053']+data_train['ENG_MDL_CD_080110'])
# data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_070035']+data_train['ENG_MDL_CD_080009']+data_train['ENG_MDL_CD_080004']+data_train['ENG_MDL_CD_235001']+data_train['ENG_MDL_CD_070024']+data_train['ENG_MDL_CD_095001']+data_train['ENG_MDL_CD_050014']+data_train['ENG_MDL_CD_070008']+data_train['ENG_MDL_CD_070012']+data_train['ENG_MDL_CD_080040']+data_train['ENG_MDL_CD_207065']+data_train['ENG_MDL_CD_050070']+data_train['ENG_MDL_CD_050020'])
# data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_150080']+data_test['ENG_MDL_CD_070028']+data_test['ENG_MDL_CD_080002']+data_test['ENG_MDL_CD_005001']+data_test['ENG_MDL_CD_150085']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_050053']+data_test['ENG_MDL_CD_080110'])
# data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_070035']+data_test['ENG_MDL_CD_080009']+data_test['ENG_MDL_CD_080004']+data_test['ENG_MDL_CD_235001']+data_test['ENG_MDL_CD_070024']+data_test['ENG_MDL_CD_095001']+data_test['ENG_MDL_CD_050014']+data_test['ENG_MDL_CD_070008']+data_test['ENG_MDL_CD_070012']+data_test['ENG_MDL_CD_080040']+data_test['ENG_MDL_CD_207065']+data_test['ENG_MDL_CD_050070']+data_test['ENG_MDL_CD_050020'])
# data_train['NC_TireSize_grp1']=(data_train['NC_TireSize_P255/70R16']+data_train['NC_TireSize_P265/70R17']+data_train['NC_TireSize_P225/70R16'])
# data_test['NC_TireSize_grp1']=(data_test['NC_TireSize_P255/70R16']+data_test['NC_TireSize_P265/70R17']+data_test['NC_TireSize_P225/70R16'])
# data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_V']+data_train['PLNT_CD_A'])
# # data_train['PLNT_CD_grp2']=(data_train['PLNT_CD_Y']+data_train['PLNT_CD_7']+data_test['PLNT_CD_R'])
# data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_V']+data_test['PLNT_CD_A'])
# # data_test['PLNT_CD_grp2']=(data_test['PLNT_CD_Y']+data_test['PLNT_CD_7']+data_test['PLNT_CD_R'])
# data_train['TRANS_OVERDRV_IND_grp1']=(data_train['TRANS_OVERDRV_IND_N']+data_train['TRANS_OVERDRV_IND_U'])
# data_test['TRANS_OVERDRV_IND_grp1']=(data_test['TRANS_OVERDRV_IND_N']+data_test['TRANS_OVERDRV_IND_U'])

# if not os.path.exists('glm/'):
# 	os.makedirs('glm/')
# formula = 'lossratio ~  ENTERTAIN_CD_127    + cko_hp_wheelbase_ratio_ge24  + length_VMMS + TRANS_OVERDRV_IND_grp1 + cko_weight_min_lt2k + EA_TURNING_CIRCLE_DIAMETER_ge54 + PLNT_CD_grp1   + ABS_BRK_CD_5  + bodyStyleCd_grp1 + bodyStyleCd_grp2  + NC_Airbag_P_SID + NC_TireSize_grp1 + ENG_MDL_CD_grp1 + ENG_MDL_CD_grp2 '
# # Surprisingly these three variables are not significant. + priceNew_35kspline + BODY_STYLE_CD_CG + BODY_STYLE_CD_PV  
# prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm/',0)

# # data_train['weight'] = data_train[WEIGHT]
# # data_train['predictedValue'] = prediction_glm_train*data_train['weight']
# # data_train['actualValue'] = data_train[LABEL]*data_train['weight']
# # data_test['weight'] = data_test[WEIGHT]
# # data_test['predictedValue'] = prediction_glm_test*data_test['weight']
# # data_test['actualValue'] = data_test[LABEL]*data_test['weight']


# # ml.actualvsfittedbyfactor(data_train,data_test,'bodyStyleCd_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'bodyStyleCd_grp2','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'ENG_MDL_CD_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'ENG_MDL_CD_grp2','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'NC_TireSize_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'PLNT_CD_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'cko_hp_wheelbase_ratio_ge24','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'cko_weight_min_lt2k','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'EA_TURNING_CIRCLE_DIAMETER_ge54','glm/')

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

# IF ENTERTAIN_CD IN ('1','2','7') THEN ENTERTAIN_CD_127=1; ELSE ENTERTAIN_CD_127=0;
# IF cko_hp_wheelbase_ratio>=2.4 THEN cko_hp_wheelbase_ratio_ge24=1; ELSE cko_hp_wheelbase_ratio_ge24=0;
# IF length_VM = . THEN length_VMMS=1; ELSE length_VMMS = 0;
# IF TRANS_OVERDRV_IND IN ('N','U') THEN TRANS_OVERDRV_IND_grp1=1; ELSE TRANS_OVERDRV_IND_grp1=0;
# IF cko_weight_min<2000 THEN cko_weight_min_lt2k=1; ELSE =0;
# IF EA_TURNING_CIRCLE_DIAMETER>=54 THEN EA_TURNING_CIRCLE_DIAMETER_ge54=1; ELSE EA_TURNING_CIRCLE_DIAMETER_ge54=0;
# IF PLNT_CD IN ('V','A') THEN PLNT_CD_grp1=1; ELSE PLNT_CD_grp1=0;
# IF ABS_BRK_CD ='5' THEN ABS_BRK_CD_5=1; ELSE ABS_BRK_CD_5=0;
# IF bodyStyleCd IN ('ROYL_PKP','WAGON_5D','UTIL_4X4','VAN','SPT_PKUP','BUS','CSTM_PKP','WAGON_3D','TRUCK','WAGON','STD_PKUP','TRK_4X2') THEN bodyStyleCd_grp1=1; ELSE bodyStyleCd_grp1=0;
# IF bodyStyleCd IN ('WAGON_4D','SUT4X24D','VAN4X23D','WAG4D4X2','WAG_4X4','WAG4X43D','UTIL_4X2','VAN_4X4','WAG4X23D','WAG_4X2','PKP_4X4','PKP4X43D','PKP4X23D') THEN bodyStyleCd_grp2=1; ELSE bodyStyleCd_grp2=0;
# IF NC_Airbag_P = 'SID' THEN NC_Airbag_P_SID=1; ELSE NC_Airbag_P_SID=0;
# IF NC_TireSize IN ('P255/70R16','P265/70R17','P225/70R16') THEN NC_TireSize_grp1=1; ELSE NC_TireSize_grp1=0;
# IF ENG_MDL_CD IN ('150080','070028','080002','005001','150085','070020','050053','080110') THEN ENG_MDL_CD_grp1=1; ELSE ENG_MDL_CD_grp1=0;
# IF ENG_MDL_CD IN ('070035','080009','080004','235001','070024','095001','050014','070008','070012','080040','207065','050070','050020') THEN ENG_MDL_CD_grp2=1; ELSE ENG_MDL_CD_grp2=0;

#  linpred= -0.6822
#  + -0.3391*ENTERTAIN_CD_127
#  + -0.384*cko_hp_wheelbase_ratio_ge24
#  + -0.192*length_VMMS
#  + 0.1564*TRANS_OVERDRV_IND_grp1
#  + -0.311*cko_weight_min_lt2k
#  + -41.4891*EA_TURNING_CIRCLE_DIAMETER_ge54
#  + -41.1238*PLNT_CD_grp1
#  + 0.2825*ABS_BRK_CD_5
#  + -40.8557*bodyStyleCd_grp1
#  + -0.3096*bodyStyleCd_grp2
#  + -41.4774*NC_Airbag_P_SID
#  + 0.5044*NC_TireSize_grp1
#  + -41.0118*ENG_MDL_CD_grp1
#  + -0.821*ENG_MDL_CD_grp2


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
CATFEATURES = ['ABS_BRK_CD','bodyStyleCd','ENG_MDL_CD','TRANS_OVERDRV_IND','MAK_NM','BODY_STYLE_CD','ENG_HEAD_CNFG_CD','VINA_BODY_TYPE_CD','BodyStyle','cko_eng_cylinders']


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

print(DUMMYFEATURES)

# # Create custom variables for GLM
data_train['cko_hp_wheelbase_ratio_ge24']=np.where(data_train['cko_hp_wheelbase_ratio']>=2.4, 1, 0)
data_test['cko_hp_wheelbase_ratio_ge24']=np.where(data_test['cko_hp_wheelbase_ratio']>=2.4, 1, 0)
data_train['cko_weight_min_lt2k']=np.where(data_train['cko_weight_min']<2000,1,0)
data_test['cko_weight_min_lt2k']=np.where(data_test['cko_weight_min']<2000,1,0)
data_train['cko_weight_min_ge6k']=np.where(data_train['cko_weight_min']>=6000,1,0)
data_test['cko_weight_min_ge6k']=np.where(data_test['cko_weight_min']>=6000,1,0)
data_train['cko_weight_min_3kbtwge6k']=np.where((data_train['cko_weight_min']>=3000)&(data_train['cko_weight_min']<6000),1,0)
data_test['cko_weight_min_3kbtwge6k']=np.where((data_test['cko_weight_min']>=3000)&(data_test['cko_weight_min']<6000),1,0)
data_train['EA_TURNING_CIRCLE_DIAMETER_ge54']=np.where(data_train['EA_TURNING_CIRCLE_DIAMETER']>=54,1,0)
data_test['EA_TURNING_CIRCLE_DIAMETER_ge54']=np.where(data_test['EA_TURNING_CIRCLE_DIAMETER']>=54,1,0)
# data_train['priceNew_35kspline']=np.maximum(35000,data_train['priceNew'])
# data_test['priceNew_35kspline']=np.maximum(35000,data_test['priceNew'])

data_train['cko_min_msrp_lt10k']=np.where((0<data_train['cko_min_msrp']) & (data_train['cko_min_msrp']<10000),1,0)
data_test['cko_min_msrp_lt10k']=np.where((0<data_test['cko_min_msrp']) & (data_test['cko_min_msrp']<10000),1,0)

data_train['width_VM_76bt80']=np.where((76<=data_train['width_VM']) & (data_train['width_VM']<80),1,0)
data_test['width_VM_76bt80']=np.where((76<=data_test['width_VM']) & (data_test['width_VM']<80),1,0)


# data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_ROYL_PKP']+data_train['bodyStyleCd_WAGON_5D']+data_train['bodyStyleCd_UTIL_4X4']+data_train['bodyStyleCd_VAN']+data_train['bodyStyleCd_SPT_PKUP']+data_train['bodyStyleCd_BUS']+data_train['bodyStyleCd_CSTM_PKP']+data_train['bodyStyleCd_WAGON_3D']+data_train['bodyStyleCd_TRUCK']+data_train['bodyStyleCd_WAGON']+data_train['bodyStyleCd_STD_PKUP']+data_train['bodyStyleCd_TRK_4X2'])
# data_train['bodyStyleCd_grp2']=(data_train['bodyStyleCd_WAGON_4D']+data_train['bodyStyleCd_SUT4X24D']+data_train['bodyStyleCd_VAN4X23D']+data_train['bodyStyleCd_WAG4D4X2']+data_train['bodyStyleCd_WAG_4X4']+data_train['bodyStyleCd_WAG4X43D']+data_train['bodyStyleCd_UTIL_4X2']+data_train['bodyStyleCd_VAN_4X4']+data_train['bodyStyleCd_WAG4X23D']+data_train['bodyStyleCd_WAG_4X2']+data_train['bodyStyleCd_PKP_4X4']+data_train['bodyStyleCd_PKP4X43D']+data_train['bodyStyleCd_PKP4X23D'])
# data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_ROYL_PKP']+data_test['bodyStyleCd_WAGON_5D']+data_test['bodyStyleCd_UTIL_4X4']+data_test['bodyStyleCd_VAN']+data_test['bodyStyleCd_SPT_PKUP']+data_test['bodyStyleCd_BUS']+data_test['bodyStyleCd_CSTM_PKP']+data_test['bodyStyleCd_WAGON_3D']+data_test['bodyStyleCd_TRUCK']+data_test['bodyStyleCd_WAGON']+data_test['bodyStyleCd_STD_PKUP']+data_test['bodyStyleCd_TRK_4X2'])
# data_test['bodyStyleCd_grp2']=(data_test['bodyStyleCd_WAGON_4D']+data_test['bodyStyleCd_SUT4X24D']+data_test['bodyStyleCd_VAN4X23D']+data_test['bodyStyleCd_WAG4D4X2']+data_test['bodyStyleCd_WAG_4X4']+data_test['bodyStyleCd_WAG4X43D']+data_test['bodyStyleCd_UTIL_4X2']+data_test['bodyStyleCd_VAN_4X4']+data_test['bodyStyleCd_WAG4X23D']+data_test['bodyStyleCd_WAG_4X2']+data_test['bodyStyleCd_PKP_4X4']+data_test['bodyStyleCd_PKP4X43D']+data_test['bodyStyleCd_PKP4X23D'])
data_train['ENG_MDL_CD_grp3']=(data_train['ENG_MDL_CD_150080']+data_train['ENG_MDL_CD_070028']+data_train['ENG_MDL_CD_080002']+data_train['ENG_MDL_CD_005001']+data_train['ENG_MDL_CD_150085']+data_train['ENG_MDL_CD_070020']+data_train['ENG_MDL_CD_050053']+data_train['ENG_MDL_CD_080110'])
data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_070035']+data_train['ENG_MDL_CD_080009']+data_train['ENG_MDL_CD_080004']+data_train['ENG_MDL_CD_235001']+data_train['ENG_MDL_CD_070024']+data_train['ENG_MDL_CD_095001']+data_train['ENG_MDL_CD_050014']+data_train['ENG_MDL_CD_070008']+data_train['ENG_MDL_CD_070012']+data_train['ENG_MDL_CD_080040']+data_train['ENG_MDL_CD_207065']+data_train['ENG_MDL_CD_050070']+data_train['ENG_MDL_CD_050020'])
data_test['ENG_MDL_CD_grp3']=(data_test['ENG_MDL_CD_150080']+data_test['ENG_MDL_CD_070028']+data_test['ENG_MDL_CD_080002']+data_test['ENG_MDL_CD_005001']+data_test['ENG_MDL_CD_150085']+data_test['ENG_MDL_CD_070020']+data_test['ENG_MDL_CD_050053']+data_test['ENG_MDL_CD_080110'])
data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_070035']+data_test['ENG_MDL_CD_080009']+data_test['ENG_MDL_CD_080004']+data_test['ENG_MDL_CD_235001']+data_test['ENG_MDL_CD_070024']+data_test['ENG_MDL_CD_095001']+data_test['ENG_MDL_CD_050014']+data_test['ENG_MDL_CD_070008']+data_test['ENG_MDL_CD_070012']+data_test['ENG_MDL_CD_080040']+data_test['ENG_MDL_CD_207065']+data_test['ENG_MDL_CD_050070']+data_test['ENG_MDL_CD_050020'])
# data_train['NC_TireSize_grp1']=(data_train['NC_TireSize_P255/70R16']+data_train['NC_TireSize_P265/70R17']+data_train['NC_TireSize_P225/70R16'])
# data_test['NC_TireSize_grp1']=(data_test['NC_TireSize_P255/70R16']+data_test['NC_TireSize_P265/70R17']+data_test['NC_TireSize_P225/70R16'])
# data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_V']+data_train['PLNT_CD_A'])
# data_train['PLNT_CD_grp2']=(data_train['PLNT_CD_Y']+data_train['PLNT_CD_7']+data_test['PLNT_CD_R'])
# data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_V']+data_test['PLNT_CD_A'])
# data_test['PLNT_CD_grp2']=(data_test['PLNT_CD_Y']+data_test['PLNT_CD_7']+data_test['PLNT_CD_R'])
data_train['TRANS_OVERDRV_IND_grp1']=(data_train['TRANS_OVERDRV_IND_N']+data_train['TRANS_OVERDRV_IND_U'])
data_test['TRANS_OVERDRV_IND_grp1']=(data_test['TRANS_OVERDRV_IND_N']+data_test['TRANS_OVERDRV_IND_U'])

# data_train['MAK_NM_AZURE DYNAMICS']=data_train['MAK_NM_AZURE_DYNAMICS']
# data_test['MAK_NM_AZURE DYNAMICS']=data_test['MAK_NM_AZURE_DYNAMICS']
data_train['MAK_NM_MERCEDES_BENZ']=data_train['MAK_NM_MERCEDES-BENZ']
data_test['MAK_NM_MERCEDES_BENZ']=data_test['MAK_NM_MERCEDES-BENZ']

data_train['MAK_NM_grp1']=(data_train['MAK_NM_AZURE_DYNAMICS']+data_train['MAK_NM_CHEVROLET']+data_train['MAK_NM_SUZUKI'])
data_test['MAK_NM_grp1']=(data_test['MAK_NM_AZURE_DYNAMICS']+data_test['MAK_NM_CHEVROLET']+data_test['MAK_NM_SUZUKI'])
# Negative bodyStyleCd group. Some work standalone
# data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_BUS']+data_train['bodyStyleCd_CSTM_PKP']+data_train['bodyStyleCd_PKP_4X2']+data_train['bodyStyleCd_PKP_4X4']+data_train['bodyStyleCd_TRK_4X2']+data_train['bodyStyleCd_UTIL_4X4']+data_train['bodyStyleCd_VAN']+data_train['bodyStyleCd_WAG4D4X2']+data_train['bodyStyleCd_WAGON']+data_train['bodyStyleCd_WAGON_3D']+data_train['bodyStyleCd_WAGON_5D']) 
# data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_BUS']+data_test['bodyStyleCd_CSTM_PKP']+data_test['bodyStyleCd_PKP_4X2']+data_test['bodyStyleCd_PKP_4X4']+data_test['bodyStyleCd_TRK_4X2']+data_test['bodyStyleCd_UTIL_4X4']+data_test['bodyStyleCd_VAN']+data_test['bodyStyleCd_WAG4D4X2']+data_test['bodyStyleCd_WAGON']+data_test['bodyStyleCd_WAGON_3D']+data_test['bodyStyleCd_WAGON_5D']) 
# Positive bodyStyleCd group. If not signif remove PICKUP
data_train['bodyStyleCd_grp2']=(data_train['bodyStyleCd_PICKUP']+data_train['bodyStyleCd_UTL4X24D'])
data_test['bodyStyleCd_grp2']=(data_test['bodyStyleCd_PICKUP']+data_test['bodyStyleCd_UTL4X24D'])

# Pickup group
data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_PICKUP']+data_train['bodyStyleCd_CSTM_PKP']+data_train['bodyStyleCd_PKP_4X2']+data_train['bodyStyleCd_PKP_4X4']+data_train['bodyStyleCd_PKP4X22D']+data_train['bodyStyleCd_PKP4X23D']+data_train['bodyStyleCd_PKP4X24D']+data_train['bodyStyleCd_PKP4X42D']+data_train['bodyStyleCd_PKP4X43D']+data_train['bodyStyleCd_ROYL_PKP']+data_train['bodyStyleCd_SPT_PKUP']+data_train['bodyStyleCd_STD_PKUP']) 
data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_PICKUP']+data_test['bodyStyleCd_CSTM_PKP']+data_test['bodyStyleCd_PKP_4X2']+data_test['bodyStyleCd_PKP_4X4']+data_test['bodyStyleCd_PKP4X22D']+data_test['bodyStyleCd_PKP4X23D']+data_test['bodyStyleCd_PKP4X24D']+data_test['bodyStyleCd_PKP4X42D']+data_test['bodyStyleCd_PKP4X43D']+data_test['bodyStyleCd_ROYL_PKP']+data_test['bodyStyleCd_SPT_PKUP']+data_test['bodyStyleCd_STD_PKUP']) 


data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_080110']+data_train['ENG_MDL_CD_005001']+data_train['ENG_MDL_CD_150085']+data_train['ENG_MDL_CD_080009']+data_train['ENG_MDL_CD_080040']+data_train['ENG_MDL_CD_070012']+data_train['ENG_MDL_CD_070008']+data_train['ENG_MDL_CD_070035']+data_train['ENG_MDL_CD_070028']+data_train['ENG_MDL_CD_050020']+data_train['ENG_MDL_CD_050053']+data_train['ENG_MDL_CD_080002']+data_train['ENG_MDL_CD_150080'])
data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_080110']+data_test['ENG_MDL_CD_005001']+data_test['ENG_MDL_CD_150085']+data_test['ENG_MDL_CD_080009']+data_test['ENG_MDL_CD_080040']+data_test['ENG_MDL_CD_070012']+data_test['ENG_MDL_CD_070008']+data_test['ENG_MDL_CD_070035']+data_test['ENG_MDL_CD_070028']+data_test['ENG_MDL_CD_050020']+data_test['ENG_MDL_CD_050053']+data_test['ENG_MDL_CD_080002']+data_test['ENG_MDL_CD_150080'])

if not os.path.exists('glm/'):
	os.makedirs('glm/')
formula = 'lossratio ~  cko_hp_wheelbase_ratio_ge24   '
# + cko_weight_min_3kbtwge6k + cko_weight_minMS
# formula += ' + EA_TURNING_CIRCLE_DIAMETER_ge54 + EA_TURNING_CIRCLE_DIAMETERMS  + ABS_BRK_CD_5  + ABS_BRK_CD_1   '
# formula += ' + FRNT_TIRE_SIZE_prefix + FRNT_TIRE_SIZE_suffix + REAR_TIRE_SIZE_prefix + REAR_TIRE_SIZE_suffix + REAR_TIRE_SIZEMS + REAR_TIRE_SIZEMS '
# formula += ' + MAK_NM_AZURE_DYNAMICS + MAK_NM_BUICK + MAK_NM_CADILLAC + MAK_NM_CHEVROLET + MAK_NM_CHRYSLER + MAK_NM_DATSUN + MAK_NM_DODGE + MAK_NM_GMC + MAK_NM_HONDA + MAK_NM_HUMMER + MAK_NM_HYUNDAI + MAK_NM_ISUZU + MAK_NM_JEEP + MAK_NM_KIA + MAK_NM_LINCOLN + MAK_NM_MAZDA + MAK_NM_MERCEDES_BENZ + MAK_NM_MERCURY + MAK_NM_MITSUBISHI + MAK_NM_NISSAN + MAK_NM_OLDSMOBILE + MAK_NM_PLYMOUTH + MAK_NM_PONTIAC + MAK_NM_RAM + MAK_NM_SATURN + MAK_NM_SPRINTER + MAK_NM_SUBARU + MAK_NM_SUZUKI + MAK_NM_TOYOTA + MAK_NM_VOLKSWAGEN '
formula += ' + MAK_NM_grp1 '
# formula += ' + bodyStyleCd_BUS + bodyStyleCd_CSTM_PKP + bodyStyleCd_KING_CAB + bodyStyleCd_MPV_4X2 + bodyStyleCd_PICKUP + bodyStyleCd_PKP_4X2 + bodyStyleCd_PKP_4X4 + bodyStyleCd_PKP4X22D + bodyStyleCd_PKP4X23D + bodyStyleCd_PKP4X24D '
# formula += ' + bodyStyleCd_PKP4X42D + bodyStyleCd_PKP4X43D + bodyStyleCd_ROYL_PKP + bodyStyleCd_SPT_PKUP + bodyStyleCd_STD_PKUP + bodyStyleCd_SUT4X24D + bodyStyleCd_TRK_4X2 + bodyStyleCd_UTIL_4X2 + bodyStyleCd_UTIL_4X4 '
# formula += ' + bodyStyleCd_UTL4X24D + bodyStyleCd_UTL4X44D + bodyStyleCd_VAN + bodyStyleCd_VAN_4X2 + bodyStyleCd_VAN_4X4 + bodyStyleCd_VAN4X22D + bodyStyleCd_VAN4X23D + bodyStyleCd_VAN4X24D + bodyStyleCd_VAN4X42D + bodyStyleCd_WAG_4X2 '
# formula += ' + bodyStyleCd_WAG_4X4 + bodyStyleCd_WAG3D4X2 + bodyStyleCd_WAG4D4X2 + bodyStyleCd_WAG4X23D + bodyStyleCd_WAG4X24D + bodyStyleCd_WAG4X43D + bodyStyleCd_WAG4X44D + bodyStyleCd_WAG5D4X2 + bodyStyleCd_WAGON + bodyStyleCd_WAGON_3D + bodyStyleCd_WAGON_4D + bodyStyleCd_WAGON_5D '
# formula += ' + bodyStyleCd_grp1 + bodyStyleCd_grp2   '
# formula += ' + ENG_MDL_CD_070009 + ENG_MDL_CD_050009 + ENG_MDL_CD_207075 + ENG_MDL_CD_050006 + ENG_MDL_CD_070007 + ENG_MDL_CD_230001 + ENG_MDL_CD_070014 + ENG_MDL_CD_080110 + ENG_MDL_CD_070010 + ENG_MDL_CD_005001 + ENG_MDL_CD_207030 + ENG_MDL_CD_180057 + ENG_MDL_CD_080003 + ENG_MDL_CD_207080 + ENG_MDL_CD_070017 + ENG_MDL_CD_150085 + ENG_MDL_CD_080009 + ENG_MDL_CD_207070 + ENG_MDL_CD_070005 + ENG_MDL_CD_080040 + ENG_MDL_CD_207010 + ENG_MDL_CD_207020 + ENG_MDL_CD_235001 + ENG_MDL_CD_070012 + ENG_MDL_CD_080012 + ENG_MDL_CD_050015 + ENG_MDL_CD_070011 + ENG_MDL_CD_080044 + ENG_MDL_CD_050011 + ENG_MDL_CD_070008 + ENG_MDL_CD_080004 + ENG_MDL_CD_080020 + ENG_MDL_CD_080015 + ENG_MDL_CD_050016 + ENG_MDL_CD_050010 + ENG_MDL_CD_080050 + ENG_MDL_CD_080084 + ENG_MDL_CD_070035 + ENG_MDL_CD_070016 + ENG_MDL_CD_050003 + ENG_MDL_CD_207050 + ENG_MDL_CD_080114 + ENG_MDL_CD_095001 + ENG_MDL_CD_070028 + ENG_MDL_CD_050012 + ENG_MDL_CD_080080 + ENG_MDL_CD_080130 + ENG_MDL_CD_207065 + ENG_MDL_CD_070015 + ENG_MDL_CD_080011 + ENG_MDL_CD_235002 + ENG_MDL_CD_050020 + ENG_MDL_CD_050053 + ENG_MDL_CD_050017 + ENG_MDL_CD_050013 + ENG_MDL_CD_080002 + ENG_MDL_CD_070020 + ENG_MDL_CD_207060 + ENG_MDL_CD_239001 + ENG_MDL_CD_080013 + ENG_MDL_CD_080070 + ENG_MDL_CD_239002 + ENG_MDL_CD_070003 + ENG_MDL_CD_050005 + ENG_MDL_CD_050014 + ENG_MDL_CD_239005 + ENG_MDL_CD_050070 + ENG_MDL_CD_070004 + ENG_MDL_CD_238004 + ENG_MDL_CD_070026 + ENG_MDL_CD_150080  + ENG_MDL_CD_050060 + ENG_MDL_CD_180053 + ENG_MDL_CD_080010 + ENG_MDL_CD_070024 + ENG_MDL_CD_070013 '
# bodyStyleCd_grp1 is now pickup
formula += '  + VINA_BODY_TYPE_CD_ES '
# formula += '  + TRANS_OVERDRV_IND_N '
# Delete these as you go. But re-add cylinders!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# formula += '  + classCd_d1 + ENG_ASP_SUP_CHGR_CD_d1'
# + ENG_HEAD_CNFG_CD_d1+ ENG_HEAD_CNFG_CD_d3+ ENG_MFG_CD_d2 + ENG_VLVS_PER_CLNDR_d2 + mak_ford + cko_eng_cylinders_5 + cko_eng_cylinders_6

# Try all these below 
# formula += '   + MAK_NM_KIA '


# testarray=['cko_hp_wheelbase_ratio_ge24','cko_weight_min_3kbtwge6k','cko_weight_minMS','MAK_NM_grp1','bodyStyleCd_grp1','bodyStyleCd_grp2','ENG_MDL_CD_grp1','VINA_BODY_TYPE_CD_ES']

# for testfield in testarray:
# 	print(testfield);
# 	print(data_train[testfield].sum())


# Surprisingly these three variables are not significant. + priceNew_35kspline + BODY_STYLE_CD_CG + BODY_STYLE_CD_PV  
prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm/',0)

# data_train['weight'] = data_train[WEIGHT]
# data_train['predictedValue'] = prediction_glm_train*data_train['weight']
# data_train['actualValue'] = data_train[LABEL]*data_train['weight']
# data_test['weight'] = data_test[WEIGHT]
# data_test['predictedValue'] = prediction_glm_test*data_test['weight']
# data_test['actualValue'] = data_test[LABEL]*data_test['weight']

# # print('weight');
# # print(data_train['weight'].sum())
# # print('weighted_predictedValue');
# # print(data_train['predictedValue'].sum())
# # print('weighted_actualValue');
# # print(data_train['actualValue'].sum())
# # print('avg_predictedValue');
# # print(prediction_glm_train.sum())
# # print('avg_actualValue');
# # print(data_train[LABEL].sum())

# ml.actualvsfittedbyfactor(data_train,data_test,'bodyStyleCd_grp1','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'bodyStyleCd_grp2','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'MAK_NM_grp1','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'ENG_MDL_CD_grp1','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'cko_hp_wheelbase_ratio_ge24','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'cko_weight_min_ge2k','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'FRNT_TIRE_SIZE_prefix','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'FRNT_TIRE_SIZE_suffix','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'REAR_TIRE_SIZE_prefix','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'REAR_TIRE_SIZE_suffix','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'FRNT_TIRE_SIZEMS','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'REAR_TIRE_SIZEMS','glm/')

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
