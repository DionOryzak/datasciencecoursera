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

filenametrain="TRAIN4.csv"
filenametest="TEST4.csv"

print("Starting Program")
print(datetime.datetime.now())

data_train = pd.read_csv(filenametrain, skipinitialspace=True, skiprows=1, names=COLUMNS, dtype=dtype)
data_test = pd.read_csv(filenametest, skipinitialspace=True, skiprows=1, names=COLUMNS, dtype=dtype)


avgweight = data_train[WEIGHT].mean()
# print(avgweight)

data_train[WEIGHT]=data_train[WEIGHT]/avgweight
data_test[WEIGHT]= data_test[WEIGHT]/avgweight

# print("Printing data_train['FRNT_TYRE_SIZE_CD_1']")
# print(data_train['FRNT_TYRE_SIZE_CD_1'])

# catarray=['BODY_STYLE_CD','BODY_STYLE_DESC','bodyStyleCd','cko_eng_cylinders','cko_eng_cylinders_Tr','ENG_MDL_CD','ENG_MFG_CD','FRNT_TYRE_SIZE_CD','FRNT_TYRE_SIZE_Desc','MAK_NM','MFG_DESC','NC_HD_INJ_C_D','NC_HD_INJ_C_P','NC_TireSize4','NC_TractionControl','PLNT_CD','REAR_TIRE_SIZE_CD','REAR_TIRE_SIZE_DESC','RSTRNT_TYP_CD','TLT_STRNG_WHL_OPT_CD','TRK_TNG_RAT_CD','VINA_BODY_TYPE_CD','vinmasterPerformanceCd','WHL_DRVN_CNT_Tr']


# for CAT in catarray:
# 	text_file=open(CAT+'values.txt','w')
# 	text_file.write(str(set(data_train[CAT])))
# 	text_file.close()
# 	text_file=open(CAT+'weights.txt','w')
# 	text_file.write(str(data_train.groupby([CAT], as_index=False)[WEIGHT].sum()))
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
# data_train['WHL_BAS_LNGST_INCHS_lt95']=np.where((data_train['WHL_BAS_LNGST_INCHS']>=0) & (data_train['WHL_BAS_LNGST_INCHS']<95), 1, 0)
# data_train['WHL_BAS_SHRST_INCHS_lt95']=np.where((data_train['WHL_BAS_SHRST_INCHS']>=0) & (data_train['WHL_BAS_SHRST_INCHS']<95), 1, 0)
# data_train['length_VM_bw220and250']=np.where((data_train['length_VM']>=220) & (data_train['length_VM']<250), 1, 0)
# # data_train['Polkdex_ge320k']=np.where(data_train['Polkdex']>=320000, 1, 0)
# data_train['door_cnt_bin_lt2']=np.where(data_train['door_cnt_bin']<2, 1, 0)
# data_train['cko_eng_cylinders_numeric_eq2']=np.where(data_train['cko_eng_cylinders_numeric']==2, 1, 0)
# data_test['WHL_BAS_LNGST_INCHS_lt95']=np.where((data_test['WHL_BAS_LNGST_INCHS']>=0) & (data_test['WHL_BAS_LNGST_INCHS']<95), 1, 0)
# data_test['WHL_BAS_SHRST_INCHS_lt95']=np.where((data_test['WHL_BAS_SHRST_INCHS']>=0) & (data_test['WHL_BAS_SHRST_INCHS']<95), 1, 0)
# data_test['length_VM_bw220and250']=np.where((data_test['length_VM']>=220) & (data_test['length_VM']<250), 1, 0)
# # data_test['Polkdex_ge320k']=np.where(data_test['Polkdex']>=320000, 1, 0)
# data_test['door_cnt_bin_lt2']=np.where(data_test['door_cnt_bin']<2, 1, 0)
# data_test['cko_eng_cylinders_numeric_eq2']=np.where(data_test['cko_eng_cylinders_numeric']==2, 1, 0)

# if not os.path.exists('glm/'):
# 	os.makedirs('glm/')
# formula= 'lossratio ~  cyl_8 + cko_eng_cylinders_d2 + WHL_BAS_LNGST_INCHS_lt95  + WHL_BAS_SHRST_INCHS_lt95 + length_VM_bw220and250 + door_cnt_bin_lt2 + cko_eng_cylinders_numeric_eq2  '
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
# data_train['WHL_BAS_LNGST_INCHS_lt95']=np.where((data_train['WHL_BAS_LNGST_INCHS']>=0) & (data_train['WHL_BAS_LNGST_INCHS']<95), 1, 0)
# data_train['WHL_BAS_SHRST_INCHS_lt95']=np.where((data_train['WHL_BAS_SHRST_INCHS']>=0) & (data_train['WHL_BAS_SHRST_INCHS']<95), 1, 0)
# data_train['door_cnt_bin_lt2']=np.where(data_train['door_cnt_bin']<2, 1, 0)
# data_train['cko_eng_cylinders_numeric_eq2']=np.where(data_train['cko_eng_cylinders_numeric']==2, 1, 0)
# data_test['WHL_BAS_LNGST_INCHS_lt95']=np.where((data_test['WHL_BAS_LNGST_INCHS']>=0) & (data_test['WHL_BAS_LNGST_INCHS']<95), 1, 0)
# data_test['WHL_BAS_SHRST_INCHS_lt95']=np.where((data_test['WHL_BAS_SHRST_INCHS']>=0) & (data_test['WHL_BAS_SHRST_INCHS']<95), 1, 0)
# data_test['door_cnt_bin_lt2']=np.where(data_test['door_cnt_bin']<2, 1, 0)
# data_test['cko_eng_cylinders_numeric_eq2']=np.where(data_test['cko_eng_cylinders_numeric']==2, 1, 0)
# data_train['length_VM_bw220and250']=np.where((data_train['length_VM']>=220) & (data_train['length_VM']<250), 1, 0)
# data_test['length_VM_bw220and250']=np.where((data_test['length_VM']>=220) & (data_test['length_VM']<250), 1, 0)

# if not os.path.exists('glm/'):
# 	os.makedirs('glm/')
# formula= 'lossratio ~  cyl_8 + cko_eng_cylinders_d2 + WHL_BAS_LNGST_INCHS_lt95  + WHL_BAS_SHRST_INCHS_lt95 + length_VM_bw220and250 + door_cnt_bin_lt2 + cko_eng_cylinders_numeric_eq2  '
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

# 	if (numFEATURESATATIME>1) & (not os.path.exists('gbm'+str(i)+'/')):
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
# data_train['WHL_BAS_LNGST_INCHS_lt95']=np.where((data_train['WHL_BAS_LNGST_INCHS']>=0) & (data_train['WHL_BAS_LNGST_INCHS']<95), 1, 0)
# data_train['WHL_BAS_SHRST_INCHS_lt95']=np.where((data_train['WHL_BAS_SHRST_INCHS']>=0) & (data_train['WHL_BAS_SHRST_INCHS']<95), 1, 0)
# data_train['door_cnt_bin_lt2']=np.where(data_train['door_cnt_bin']<2, 1, 0)
# data_train['cko_eng_cylinders_numeric_eq2']=np.where(data_train['cko_eng_cylinders_numeric']==2, 1, 0)
# data_test['WHL_BAS_LNGST_INCHS_lt95']=np.where((data_test['WHL_BAS_LNGST_INCHS']>=0) & (data_test['WHL_BAS_LNGST_INCHS']<95), 1, 0)
# data_test['WHL_BAS_SHRST_INCHS_lt95']=np.where((data_test['WHL_BAS_SHRST_INCHS']>=0) & (data_test['WHL_BAS_SHRST_INCHS']<95), 1, 0)
# data_test['door_cnt_bin_lt2']=np.where(data_test['door_cnt_bin']<2, 1, 0)
# data_test['cko_eng_cylinders_numeric_eq2']=np.where(data_test['cko_eng_cylinders_numeric']==2, 1, 0)
# data_train['length_VM_bw220and250']=np.where((data_train['length_VM']>=220) & (data_train['length_VM']<250), 1, 0)
# data_test['length_VM_bw220and250']=np.where((data_test['length_VM']>=220) & (data_test['length_VM']<250), 1, 0)

# data_train['EA_ACCEL_TIME_0_TO_60_6spline']=np.maximum(6,data_train['EA_ACCEL_TIME_0_TO_60'])
# data_train['cko_eng_cylinders_numeric_lt4']=np.where(data_train['cko_eng_cylinders_numeric']<4,1,0)
# data_train['cko_max_msrp_10kspline']=np.maximum(10000,data_train['cko_max_msrp'])
# data_train['cko_min_msrp_15kspline']=np.maximum(15000,data_train['cko_min_msrp'])
# data_train['cko_weight_to_disp_max_lt1100']=np.where(data_train['cko_weight_to_disp_max']<1100,1,0)
# data_train['cko_weight_to_disp_min_ge1800']=np.where(data_train['cko_weight_to_disp_min']>=1800,1,0)
# data_train['cko_wheelbase_max_spline100']=np.minimum(data_train['cko_wheelbase_max'],100)
# data_train['cko_wheelbase_min_ge135']=np.where(data_train['cko_wheelbase_min']>=135,1,0)
# data_train['EA_ACCEL_RATE_45_TO_65_ge9']=np.where(data_train['EA_ACCEL_RATE_45_TO_65']>=9,1,0)
# data_train['EA_ACCEL_RATE_0_TO_60_ge16']=np.where(data_train['EA_ACCEL_RATE_0_TO_60']>=16,1,0)
# data_train['EA_ACCEL_TIME_45_TO_65_3spline6']=np.minimum(np.maximum(3,data_train['EA_ACCEL_TIME_45_TO_65']),6)
# data_train['EA_CURB_WEIGHT_35bw45']=np.where((3500<=data_train['EA_CURB_WEIGHT']) & (data_train['EA_CURB_WEIGHT']<4500),1,0)
# data_train['EA_height_ge80']=np.where(data_train['EA_height']>=80,1,0)
# data_train['EA_NHTSA_STARS_ge4']=np.where(data_train['EA_NHTSA_STARS']>=4,1,0)
# data_train['EA_TIP_OVER_STABILITY_RATIO_ge15']=np.where(data_train['EA_TIP_OVER_STABILITY_RATIO']>=1.5,1,0)
# data_train['ENG_DISPLCMNT_CI_380spline']=np.maximum(380,data_train['ENG_DISPLCMNT_CI']) 
# data_train['enginesize_ge6']=np.where(data_train['enginesize']>=6,1,0)
# data_train['NADA_MSRP1_ge140k']=np.where(data_train['NADA_MSRP1']>=140000,1,0)
# data_train['WHL_BAS_LNGST_INCHS_90bw120']=np.where((90<=data_train['WHL_BAS_LNGST_INCHS']) & (data_train['WHL_BAS_LNGST_INCHS']<120),1,0)
# data_train['WHL_BAS_SHRST_INCHS_ge120']=np.where(data_train['WHL_BAS_SHRST_INCHS']>=120,1,0)

# data_test['EA_ACCEL_TIME_0_TO_60_6spline']=np.maximum(6,data_test['EA_ACCEL_TIME_0_TO_60'])
# data_test['cko_eng_cylinders_numeric_lt4']=np.where(data_test['cko_eng_cylinders_numeric']<4,1,0)
# data_test['cko_max_msrp_10kspline']=np.maximum(10000,data_test['cko_max_msrp'])
# data_test['cko_min_msrp_15kspline']=np.maximum(15000,data_test['cko_min_msrp'])
# data_test['cko_weight_to_disp_max_lt1100']=np.where(data_test['cko_weight_to_disp_max']<1100,1,0)
# data_test['cko_weight_to_disp_min_ge1800']=np.where(data_test['cko_weight_to_disp_min']>=1800,1,0)
# data_test['cko_wheelbase_max_spline100']=np.minimum(data_test['cko_wheelbase_max'],100)
# data_test['cko_wheelbase_min_ge135']=np.where(data_test['cko_wheelbase_min']>=135,1,0)
# data_test['EA_ACCEL_RATE_45_TO_65_ge9']=np.where(data_test['EA_ACCEL_RATE_45_TO_65']>=9,1,0)
# data_test['EA_ACCEL_RATE_0_TO_60_ge16']=np.where(data_test['EA_ACCEL_RATE_0_TO_60']>=16,1,0)
# data_test['EA_ACCEL_TIME_45_TO_65_3spline6']=np.minimum(np.maximum(3,data_test['EA_ACCEL_TIME_45_TO_65']),6)
# data_test['EA_CURB_WEIGHT_35bw45']=np.where((3500<=data_test['EA_CURB_WEIGHT']) & (data_test['EA_CURB_WEIGHT']<4500),1,0)
# data_test['EA_height_ge80']=np.where(data_test['EA_height']>=80,1,0)
# data_test['EA_NHTSA_STARS_ge4']=np.where(data_test['EA_NHTSA_STARS']>=4,1,0)
# data_test['EA_TIP_OVER_STABILITY_RATIO_ge15']=np.where(data_test['EA_TIP_OVER_STABILITY_RATIO']>=1.5,1,0)
# data_test['ENG_DISPLCMNT_CI_380spline']=np.maximum(380,data_test['ENG_DISPLCMNT_CI']) 
# data_test['enginesize_ge6']=np.where(data_test['enginesize']>=6,1,0)
# data_test['NADA_MSRP1_ge140k']=np.where(data_test['NADA_MSRP1']>=140000,1,0)
# data_test['WHL_BAS_LNGST_INCHS_90bw120']=np.where((90<=data_test['WHL_BAS_LNGST_INCHS']) & (data_test['WHL_BAS_LNGST_INCHS']<120),1,0)
# data_test['WHL_BAS_SHRST_INCHS_ge120']=np.where(data_test['WHL_BAS_SHRST_INCHS']>=120,1,0)


# formulabase= 'lossratio ~  cyl_8 + cko_eng_cylinders_d2 + WHL_BAS_LNGST_INCHS_lt95  + WHL_BAS_SHRST_INCHS_lt95 + length_VM_bw220and250 + door_cnt_bin_lt2 + cko_eng_cylinders_numeric_eq2  '
# forms=[]
# forms.append(' + BODY_STYLE_CD_d2  + BODY_STYLE_CD_d3 + cko_height + cko_hp_maxweight_ratio + cko_hp_minweight_ratio + cko_hp_minweight_ratioMS  ') 
# forms.append(' + cko_lengthMS + classCd_d4  + classCd_93 + EA_wheelbaseMS + ENG_MFG_CD_d2 + ENG_MFG_CD_d5 + FRNT_TYRE_SIZE_CD_d1  ')
# forms.append(' + enginesizeMS + FRNT_TYRE_SIZE_CD_d3 + FRNT_TYRE_SIZE_CD_d4 + mak_cad  + mak_linc + MAK_NM_d1  + MAK_NM_d2   ')
# forms.append(' + NADA_MSRP2 + NADA_MSRP2MS + RSTRNT_TYP_CD_d3 + RSTRNT_TYP_CD_d2 + RSTRNT_TYP_CD_d9 + wdrv_rwd + turbo_super  ')
# forms.append(' + WHL_DRVN_CNTMS + width_VMMS + WHL_DRVN_CNT_d2 + EA_ACCEL_TIME_0_TO_60 + EA_ACCEL_TIME_0_TO_60_6spline  ')
# forms.append(' + EA_ACCEL_TIME_0_TO_60MS + cko_eng_cylinders_numeric_lt4 + cko_max_msrp_10kspline + cko_min_msrp_15kspline  ')
# forms.append(' + cko_weight_to_disp_max_lt1100 + cko_weight_to_disp_min_ge1800 + cko_wheelbase_max_spline100 + cko_wheelbase_min_ge135  ')
# forms.append(' + EA_ACCEL_RATE_45_TO_65_ge9 + EA_ACCEL_RATE_0_TO_60_ge16 + EA_ACCEL_TIME_45_TO_65_3spline6 + EA_ACCEL_TIME_45_TO_65MS  ')
# forms.append(' + EA_CURB_WEIGHT_35bw45 + EA_height_ge80 + EA_NHTSA_STARS_ge4 + EA_TIP_OVER_STABILITY_RATIO_ge15 + ENG_DISPLCMNT_CI_380spline  ')
# forms.append(' + enginesize_ge6 + NADA_MSRP1_ge140k + WHL_BAS_LNGST_INCHS_90bw120 + WHL_BAS_SHRST_INCHS_ge120 ')

# forms.append(' + classCd_d4 + ENG_MFG_CD_d2 + mak_linc + NADA_MSRP2 + NADA_MSRP2MS + RSTRNT_TYP_CD_d2 + wdrv_rwd + turbo_super + WHL_DRVN_CNTMS + EA_ACCEL_TIME_0_TO_60 + EA_ACCEL_TIME_0_TO_60_6spline')
# forms.append(' + cko_weight_to_disp_max_lt1100 + cko_weight_to_disp_min_ge1800 + cko_wheelbase_max_spline100 + EA_CURB_WEIGHT_35bw45  + EA_NHTSA_STARS_ge4 + EA_TIP_OVER_STABILITY_RATIO_ge15 + ENG_DISPLCMNT_CI_380spline + enginesize_ge6 ')

# forms.append(' + ENG_MFG_CD_d2 + WHL_DRVN_CNTMS + cko_weight_to_disp_min_ge1800 + cko_wheelbase_max_spline100 + EA_CURB_WEIGHT_35bw45  + EA_NHTSA_STARS_ge4 + EA_TIP_OVER_STABILITY_RATIO_ge15 + ENG_DISPLCMNT_CI_380spline + enginesize_ge6 ')
# forms.append(' + ENG_MFG_CD_d2 + WHL_DRVN_CNTMS + cko_weight_to_disp_min_ge1800 + cko_wheelbase_max_spline100 + EA_CURB_WEIGHT_35bw45  + EA_TIP_OVER_STABILITY_RATIO_ge15 + ENG_DISPLCMNT_CI_380spline + enginesize_ge6 ')

# forms.append(' + ENG_MFG_CD_d2 + WHL_DRVN_CNTMS + cko_wheelbase_max_spline100 + EA_CURB_WEIGHT_35bw45  + EA_TIP_OVER_STABILITY_RATIO_ge15 + ENG_DISPLCMNT_CI_380spline + enginesize_ge6 ')

# for i in range(14,len(forms)):
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
# data_train['WHL_BAS_LNGST_INCHS_lt95']=np.where((data_train['WHL_BAS_LNGST_INCHS']>=0) & (data_train['WHL_BAS_LNGST_INCHS']<95), 1, 0)
# data_train['WHL_BAS_SHRST_INCHS_lt95']=np.where((data_train['WHL_BAS_SHRST_INCHS']>=0) & (data_train['WHL_BAS_SHRST_INCHS']<95), 1, 0)
# data_train['door_cnt_bin_lt2']=np.where(data_train['door_cnt_bin']<2, 1, 0)
# data_train['cko_eng_cylinders_numeric_eq2']=np.where(data_train['cko_eng_cylinders_numeric']==2, 1, 0)
# data_test['WHL_BAS_LNGST_INCHS_lt95']=np.where((data_test['WHL_BAS_LNGST_INCHS']>=0) & (data_test['WHL_BAS_LNGST_INCHS']<95), 1, 0)
# data_test['WHL_BAS_SHRST_INCHS_lt95']=np.where((data_test['WHL_BAS_SHRST_INCHS']>=0) & (data_test['WHL_BAS_SHRST_INCHS']<95), 1, 0)
# data_test['door_cnt_bin_lt2']=np.where(data_test['door_cnt_bin']<2, 1, 0)
# data_test['cko_eng_cylinders_numeric_eq2']=np.where(data_test['cko_eng_cylinders_numeric']==2, 1, 0)
# data_train['length_VM_bw220and250']=np.where((data_train['length_VM']>=220) & (data_train['length_VM']<250), 1, 0)
# data_test['length_VM_bw220and250']=np.where((data_test['length_VM']>=220) & (data_test['length_VM']<250), 1, 0)
# data_train['cko_wheelbase_max_spline100']=np.minimum(data_train['cko_wheelbase_max'],100)
# data_test['cko_wheelbase_max_spline100']=np.minimum(data_test['cko_wheelbase_max'],100)
# data_train['EA_CURB_WEIGHT_35bw45']=np.where((3500<=data_train['EA_CURB_WEIGHT']) & (data_train['EA_CURB_WEIGHT']<4500),1,0)
# data_test['EA_CURB_WEIGHT_35bw45']=np.where((3500<=data_test['EA_CURB_WEIGHT']) & (data_test['EA_CURB_WEIGHT']<4500),1,0)
# data_train['EA_TIP_OVER_STABILITY_RATIO_ge15']=np.where(data_train['EA_TIP_OVER_STABILITY_RATIO']>=1.5,1,0)
# data_test['EA_TIP_OVER_STABILITY_RATIO_ge15']=np.where(data_test['EA_TIP_OVER_STABILITY_RATIO']>=1.5,1,0)
# data_train['ENG_DISPLCMNT_CI_380spline']=np.maximum(380,data_train['ENG_DISPLCMNT_CI'])
# data_test['ENG_DISPLCMNT_CI_380spline']=np.maximum(380,data_test['ENG_DISPLCMNT_CI']) 
# data_train['enginesize_ge6']=np.where(data_train['enginesize']>=6,1,0)
# data_test['enginesize_ge6']=np.where(data_test['enginesize']>=6,1,0)

# formula = 'lossratio ~  cyl_8 + cko_eng_cylinders_d2 + WHL_BAS_LNGST_INCHS_lt95  + WHL_BAS_SHRST_INCHS_lt95 + length_VM_bw220and250 + door_cnt_bin_lt2 + cko_eng_cylinders_numeric_eq2 + ENG_MFG_CD_d2 + WHL_DRVN_CNTMS  + cko_wheelbase_max_spline100 + EA_CURB_WEIGHT_35bw45  + EA_TIP_OVER_STABILITY_RATIO_ge15 + ENG_DISPLCMNT_CI_380spline + enginesize_ge6   '
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

# Set numCATFEATURES to 1 initially, commenting out any CATFEATURES elements that conflict. 
# When no CATFEATURES elements conflict anymore then set to 10
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
# 			print("ERROR with this GBM")
# 			print(CATFEATURES)
# for feature in ALLCATFEATURES:
# 	ml.actualvsfittedbyfactor(data_train,data_test,feature,'glm/')	



# # Step 5
# # Run GLM from result of CATFEATURES GBM

# CONTFEATURES = []
# CATFEATURES = []
# CATARRAY=[]
# # Include only catfeatures that are needed for the attempted GLMs
# CATARRAY.append(['ABS_BRK_CD','ABS_BRK_DESC','AIR_COND_OPT_CD','antiLockCd','antiTheftCd'])
# CATARRAY.append(['BODY_STYLE_CD','BODY_STYLE_DESC','BodyStyle'])
# CATARRAY.append(['bodyStyleCd'])
# CATARRAY.append(['cko_antitheft','cko_dtrl','cko_eng_cylinders','cko_eng_cylinders_Tr','cko_esc','cko_fuel','cko_turbo_super','classCd','classCd_Tr'])
# CATARRAY.append(['dayTimeLightCd','DR_LGHT_OPT_CD','DRV_TYP_CD','EA_DRIVE_WHEELS','ENG_ASP_SUP_CHGR_CD','ENG_ASP_TRBL_CHGR_CD','ENG_ASP_VVTL_CD','ENG_BLCK_TYP_CD','ENG_CBRT_TYP_CD','ENG_CLNDR_RTR_CNT','ENG_FUEL_INJ_TYP_CD','ENG_HEAD_CNFG_CD'])
# CATARRAY.append(['ENG_MDL_CD'])
# CATARRAY.append(['ENG_MFG_CD','ENG_MFG_CD_Tr','ENG_VLVS_PER_CLNDR_Tr'])
# CATARRAY.append(['engineType','ENTERTAIN_CD','ESCcd','frameType'])
# CATARRAY.append(['FRNT_TYRE_SIZE_CD'])
# CATARRAY.append(['FRNT_TYRE_SIZE_CD_Tr','FRNT_TYRE_SIZE_Desc'])
# CATARRAY.append(['MAK_NM','MAK_NM_Tr'])
# CATARRAY.append(['make'])
# CATARRAY.append(['MFG_DESC'])
# CATARRAY.append(['NADA_BODY1','NC_ABS4w','NC_Airbag_P','NC_AntiTheft','NC_CrashDataRecorder','NC_DayRunningLights','NC_Drive'])
# CATARRAY.append(['NC_HD_INJ_C_D'])
# CATARRAY.append(['NC_HD_INJ_C_P'])
# CATARRAY.append(['NC_HeadAirbag','NC_RearCtrLapShldrBelt','NC_RearSeatHeadRestraint','NC_SeatBeltReminder_Indicators','NC_SideAirbag'])
# CATARRAY.append(['NC_TireSize','NC_TireSize4'])
# CATARRAY.append(['NC_TractionControl','NC_VTStabilityControl','NC_WheelsDriven','numOfCylinders'])
# CATARRAY.append(['OPT1_ENTERTAIN_CD','OPT1_ROOF_CD','PLNT_CD','PLNT_CNTRY_NM'])
# CATARRAY.append(['priceNewSymbl27_2'])
# CATARRAY.append(['PWR_BRK_OPT_CD','PWR_STRNG_OPT_CD'])
# CATARRAY.append(['REAR_TIRE_SIZE_CD','REAR_TIRE_SIZE_DESC'])
# CATARRAY.append(['restraintCd','ROOF_CD','RSTRNT_TYP_CD','RSTRNT_TYP_CD_Tr'])
# CATARRAY.append(['SECUR_TYP_CD','SEGMENTATION_CD'])
# CATARRAY.append(['TLT_STRNG_WHL_OPT_CD','tonCd'])
# CATARRAY.append(['TRANS_CD','TRANS_SPEED_CD'])
# CATARRAY.append(['TRK_BED_LEN_CD','TRK_BRK_TYP_CD','TRK_BRK_TYP_CD_Tr'])
# CATARRAY.append(['TRK_CAB_CNFG_CD','TRK_CAB_CNFG_CD_Tr','TRK_FRNT_AXL_CD','TRK_REAR_AXL_CD','TRK_TNG_RAT_CD'])
# CATARRAY.append(['VINA_BODY_TYPE_CD','vinmasterPerformanceCd','WHL_DRVN_CNT_Tr'])
# # 30
# CATARRAY.append(['AIR_COND_OPT_CD','BODY_STYLE_CD','BODY_STYLE_DESC','bodyStyleCd','cko_eng_cylinders','cko_eng_cylinders_Tr','classCd','ENG_MDL_CD','ENG_MFG_CD','FRNT_TYRE_SIZE_CD','FRNT_TYRE_SIZE_Desc','MAK_NM','make',
# 'MFG_DESC','NC_HD_INJ_C_D','NC_HD_INJ_C_P','NC_TireSize4','NC_TractionControl','numOfCylinders','OPT1_ENTERTAIN_CD','PLNT_CD','priceNewSymbl27_2','REAR_TIRE_SIZE_CD','REAR_TIRE_SIZE_DESC','ROOF_CD',
# 'RSTRNT_TYP_CD','TLT_STRNG_WHL_OPT_CD','TRK_TNG_RAT_CD','VINA_BODY_TYPE_CD','vinmasterPerformanceCd','WHL_DRVN_CNT_Tr'])
# CATARRAY.append(['BODY_STYLE_CD','BODY_STYLE_DESC','bodyStyleCd','cko_eng_cylinders','cko_eng_cylinders_Tr','ENG_MDL_CD','ENG_MFG_CD','FRNT_TYRE_SIZE_CD','FRNT_TYRE_SIZE_Desc','MAK_NM','MFG_DESC','NC_HD_INJ_C_D','NC_HD_INJ_C_P','NC_TireSize4','NC_TractionControl','PLNT_CD','REAR_TIRE_SIZE_CD','REAR_TIRE_SIZE_DESC','RSTRNT_TYP_CD','TLT_STRNG_WHL_OPT_CD','TRK_TNG_RAT_CD','VINA_BODY_TYPE_CD','vinmasterPerformanceCd','WHL_DRVN_CNT_Tr'])
# CATARRAY.append(['BODY_STYLE_CD'])
# CATARRAY.append(['BODY_STYLE_DESC'])
# CATARRAY.append(['bodyStyleCd'])
# CATARRAY.append(['cko_eng_cylinders'])
# CATARRAY.append(['cko_eng_cylinders_Tr'])
# CATARRAY.append(['ENG_MDL_CD'])
# CATARRAY.append(['ENG_MFG_CD'])
# CATARRAY.append(['FRNT_TYRE_SIZE_CD'])
# CATARRAY.append(['FRNT_TYRE_SIZE_Desc'])
# CATARRAY.append(['MAK_NM'])
# CATARRAY.append(['MFG_DESC'])
# CATARRAY.append(['NC_HD_INJ_C_D'])
# CATARRAY.append(['NC_HD_INJ_C_P'])
# CATARRAY.append(['NC_TireSize4'])
# CATARRAY.append(['NC_TractionControl'])
# CATARRAY.append(['PLNT_CD'])
# CATARRAY.append(['REAR_TIRE_SIZE_CD'])
# CATARRAY.append(['REAR_TIRE_SIZE_DESC'])
# CATARRAY.append(['RSTRNT_TYP_CD'])
# CATARRAY.append(['TLT_STRNG_WHL_OPT_CD'])
# CATARRAY.append(['TRK_TNG_RAT_CD'])
# CATARRAY.append(['vinmasterPerformanceCd'])
# CATARRAY.append(['WHL_DRVN_CNT_Tr'])
# # 55
# CATARRAY.append(['BODY_STYLE_CD','bodyStyleCd','cko_eng_cylinders','cko_eng_cylinders_Tr','ENG_MDL_CD','ENG_MFG_CD','FRNT_TYRE_SIZE_CD','MAK_NM','MFG_DESC','NC_HD_INJ_C_D','NC_TireSize4','NC_TractionControl','PLNT_CD','REAR_TIRE_SIZE_CD','RSTRNT_TYP_CD','TLT_STRNG_WHL_OPT_CD','TRK_TNG_RAT_CD','vinmasterPerformanceCd','WHL_DRVN_CNT_Tr'])
# CATARRAY.append(['BODY_STYLE_CD','bodyStyleCd','cko_eng_cylinders','ENG_MDL_CD','ENG_MFG_CD','FRNT_TYRE_SIZE_CD','MAK_NM','MFG_DESC','NC_HD_INJ_C_D','NC_TireSize4','NC_TractionControl','REAR_TIRE_SIZE_CD','RSTRNT_TYP_CD','TLT_STRNG_WHL_OPT_CD','TRK_TNG_RAT_CD','vinmasterPerformanceCd','WHL_DRVN_CNT_Tr'])
# CATARRAY.append(['BODY_STYLE_CD','bodyStyleCd','cko_eng_cylinders','ENG_MDL_CD','ENG_MFG_CD','FRNT_TYRE_SIZE_CD','MAK_NM','MFG_DESC','NC_HD_INJ_C_D','NC_TireSize4','NC_TractionControl','REAR_TIRE_SIZE_CD','RSTRNT_TYP_CD','TLT_STRNG_WHL_OPT_CD','TRK_TNG_RAT_CD','vinmasterPerformanceCd','WHL_DRVN_CNT_Tr'])
# CATARRAY.append(['BODY_STYLE_CD','bodyStyleCd','cko_eng_cylinders','ENG_MDL_CD','ENG_MFG_CD','FRNT_TYRE_SIZE_CD','MAK_NM','MFG_DESC','NC_HD_INJ_C_D','NC_TireSize4','NC_TractionControl','REAR_TIRE_SIZE_CD','RSTRNT_TYP_CD','TLT_STRNG_WHL_OPT_CD','TRK_TNG_RAT_CD','vinmasterPerformanceCd','WHL_DRVN_CNT_Tr'])


# formulabase = 'lossratio ~  cyl_8 + cko_eng_cylinders_d2 + WHL_BAS_LNGST_INCHS_lt95  + WHL_BAS_SHRST_INCHS_lt95 + length_VM_bw220and250 + door_cnt_bin_lt2 + cko_eng_cylinders_numeric_eq2 + ENG_MFG_CD_d2 + WHL_DRVN_CNTMS  + cko_wheelbase_max_spline100 + EA_CURB_WEIGHT_35bw45  + EA_TIP_OVER_STABILITY_RATIO_ge15 + ENG_DISPLCMNT_CI_380spline + enginesize_ge6     '

# forms=[]
# forms.append('  + ABS_BRK_CD_4 + ABS_BRK_DESC_OTHER_STD + ABS_BRK_DESC_U + AIR_COND_OPT_CD_O + AIR_COND_OPT_CD_U + antiLockCd_N + antiTheftCd__ + antiTheftCd_T + antiTheftCd_U ')
# forms.append('  + BODY_STYLE_CD_CH + BODY_STYLE_CD_CV + BODY_STYLE_CD_HB + BODY_STYLE_CD_IP + BODY_STYLE_CD_SD + BODY_STYLE_DESC_CAB_CHASSIS + BODY_STYLE_DESC_CONVERTIBLE + BODY_STYLE_DESC_HATCHBACK + BODY_STYLE_DESC_INCOMPLETE_PICKUP + BODY_STYLE_DESC_SEDAN + BodyStyle_COUPE ')
# forms.append('  + bodyStyleCd_CABRI_2D + bodyStyleCd_CABRIOLE + bodyStyleCd_CAMPER + bodyStyleCd_CNV4X42D + bodyStyleCd_CONV_2D + bodyStyleCd_CONVRTBL + bodyStyleCd_COUPE + bodyStyleCd_COUPE_2D + bodyStyleCd_COUPE_3D + bodyStyleCd_COUPE_4D + bodyStyleCd_CPE2D4X2 + bodyStyleCd_CPE2D4X4 + bodyStyleCd_CPE3D4X2 + bodyStyleCd_CPE4X44D + bodyStyleCd_EST_WGN + bodyStyleCd_FASTBACK + bodyStyleCd_HCH2D4X4 + bodyStyleCd_HCH3D4X2 + bodyStyleCd_HCH3D4X4 + bodyStyleCd_HCH5D4X4 + bodyStyleCd_HCHBK_2D + bodyStyleCd_HCHBK_3D + bodyStyleCd_HCHBK_4D + bodyStyleCd_HCHBK_CP + bodyStyleCd_SED4X44D + bodyStyleCd_SEDAN_4D + bodyStyleCd_SPTCP_GT ')
# forms.append('  + cko_antitheft_NONE + cko_antitheft_U + cko_dtrl_N + cko_eng_cylinders__ + cko_eng_cylinders_2 + cko_eng_cylinders_3 + cko_eng_cylinders_4 + cko_eng_cylinders_5 + cko_eng_cylinders_8 + cko_eng_cylinders_Tr_4 + cko_eng_cylinders_Tr_8 + cko_esc_N + cko_esc_O + cko_esc_U + cko_fuel_NGAS + cko_fuel_U + cko_turbo_super_N + cko_turbo_super_Y + classCd__ + classCd_12 + classCd_14 + classCd_22 + classCd_24 + classCd_32 + classCd_34 + classCd_52 + classCd_54 + classCd_70 + classCd_83 + classCd_91 + classCd_Tr_Grp0 + classCd_Tr_Grp1 + classCd_Tr_Grp2 ')
# forms.append('  + dayTimeLightCd__ + dayTimeLightCd_F + dayTimeLightCd_N + DR_LGHT_OPT_CD_O + DRV_TYP_CD_AWD + DRV_TYP_CD_RWD + EA_DRIVE_WHEELS__ + EA_DRIVE_WHEELS_4_WHEEL_DRIVE + EA_DRIVE_WHEELS_4X2___4X4 + EA_DRIVE_WHEELS_ALL_WHEEL_DRIVE + EA_DRIVE_WHEELS_REAR + ENG_ASP_SUP_CHGR_CD_N + ENG_ASP_TRBL_CHGR_CD_N + ENG_ASP_TRBL_CHGR_CD_Y + ENG_ASP_VVTL_CD_N + ENG_ASP_VVTL_CD_Y + ENG_BLCK_TYP_CD_I + ENG_CBRT_TYP_CD_C + ENG_CLNDR_RTR_CNT__ + ENG_CLNDR_RTR_CNT_3 + ENG_CLNDR_RTR_CNT_4 + ENG_CLNDR_RTR_CNT_8 + ENG_FUEL_INJ_TYP_CD_M + ENG_FUEL_INJ_TYP_CD_S + ENG_HEAD_CNFG_CD_SOHC ')
# forms.append('  + ENG_MDL_CD_030040 + ENG_MDL_CD_040010 + ENG_MDL_CD_050005 + ENG_MDL_CD_050006 + ENG_MDL_CD_050007 + ENG_MDL_CD_050009 + ENG_MDL_CD_050010 + ENG_MDL_CD_050012 + ENG_MDL_CD_050014 + ENG_MDL_CD_050015 + ENG_MDL_CD_050016 + ENG_MDL_CD_050060 + ENG_MDL_CD_070007 + ENG_MDL_CD_070015 + ENG_MDL_CD_070023 + ENG_MDL_CD_080003 + ENG_MDL_CD_080006 + ENG_MDL_CD_080044 + ENG_MDL_CD_080050 + ENG_MDL_CD_080070 + ENG_MDL_CD_080080 ')
# forms.append('  + ENG_MFG_CD_070 + ENG_MFG_CD_080 + ENG_MFG_CD_180 + ENG_MFG_CD_207 + ENG_MFG_CD_Tr_GENERAL_MOTORS + ENG_MFG_CD_Tr_TOYOTA + ENG_VLVS_PER_CLNDR_Tr_4 ')
# forms.append('  + engineType_T + ENTERTAIN_CD_1 + ESCcd__ + ESCcd_O + frameType__ + frameType_F ')
# forms.append('  + FRNT_TYRE_SIZE_CD_1 + FRNT_TYRE_SIZE_CD_30 + FRNT_TYRE_SIZE_CD_36 + FRNT_TYRE_SIZE_CD_45 + FRNT_TYRE_SIZE_CD_47 + FRNT_TYRE_SIZE_CD_56 + FRNT_TYRE_SIZE_CD_57 + FRNT_TYRE_SIZE_CD_7 + FRNT_TYRE_SIZE_CD_70 + FRNT_TYRE_SIZE_CD_75 + FRNT_TYRE_SIZE_CD_77 + FRNT_TYRE_SIZE_CD_78 + FRNT_TYRE_SIZE_CD_79 + FRNT_TYRE_SIZE_CD_8 + FRNT_TYRE_SIZE_CD_80 + FRNT_TYRE_SIZE_CD_81 + FRNT_TYRE_SIZE_CD_82 + FRNT_TYRE_SIZE_CD_85 + FRNT_TYRE_SIZE_CD_86 + FRNT_TYRE_SIZE_CD_87 + FRNT_TYRE_SIZE_CD_88 + FRNT_TYRE_SIZE_CD_9 + FRNT_TYRE_SIZE_CD_90 + FRNT_TYRE_SIZE_CD_92 + FRNT_TYRE_SIZE_CD_96 + FRNT_TYRE_SIZE_CD_98 ')
# forms.append('  + FRNT_TYRE_SIZE_CD_Tr_19 + FRNT_TYRE_SIZE_Desc_12R145 + FRNT_TYRE_SIZE_Desc_12R165 + FRNT_TYRE_SIZE_Desc_12R415 + FRNT_TYRE_SIZE_Desc_13R145 + FRNT_TYRE_SIZE_Desc_14R185 + FRNT_TYRE_SIZE_Desc_14R195 + FRNT_TYRE_SIZE_Desc_14R205 + FRNT_TYRE_SIZE_Desc_14R215 + FRNT_TYRE_SIZE_Desc_14R225 + FRNT_TYRE_SIZE_Desc_14RP15 + FRNT_TYRE_SIZE_Desc_15BH78 + FRNT_TYRE_SIZE_Desc_15H78 + FRNT_TYRE_SIZE_Desc_15HR78 + FRNT_TYRE_SIZE_Desc_15R145 + FRNT_TYRE_SIZE_Desc_15R155 + FRNT_TYRE_SIZE_Desc_15R175 + FRNT_TYRE_SIZE_Desc_15R185 + FRNT_TYRE_SIZE_Desc_15R190 + FRNT_TYRE_SIZE_Desc_15R195 + FRNT_TYRE_SIZE_Desc_15R205 + FRNT_TYRE_SIZE_Desc_15R215 + FRNT_TYRE_SIZE_Desc_15R225 + FRNT_TYRE_SIZE_Desc_15R235 + FRNT_TYRE_SIZE_Desc_15R245 + FRNT_TYRE_SIZE_Desc_15R255 + FRNT_TYRE_SIZE_Desc_15R265 + FRNT_TYRE_SIZE_Desc_16R175 + FRNT_TYRE_SIZE_Desc_16R185 + FRNT_TYRE_SIZE_Desc_16R195 + FRNT_TYRE_SIZE_Desc_16R205 + FRNT_TYRE_SIZE_Desc_17R245 + FRNT_TYRE_SIZE_Desc_18R255 ')
# # 10
# forms.append('  + MAK_NM_ALFA_ROMEO + MAK_NM_AMERICAN_GENERAL + MAK_NM_AMERICAN_MOTORS + MAK_NM_ASTON_MARTIN + MAK_NM_AUDI + MAK_NM_BENTLEY + MAK_NM_CHRYSLER + MAK_NM_DAEWOO + MAK_NM_DAIHATSU + MAK_NM_DATSUN + MAK_NM_EAGLE + MAK_NM_FERRARI + MAK_NM_FIAT + MAK_NM_FISKER_AUTOMOTIVE + MAK_NM_FORD + MAK_NM_GEO + MAK_NM_GLOBAL_ELECTRIC_MOTORS + MAK_NM_GMC + MAK_NM_HONDA + MAK_NM_HUMMER + MAK_NM_HYUNDAI + MAK_NM_INFINITI + MAK_NM_ISUZU + MAK_NM_IVECO + MAK_NM_JAGUAR + MAK_NM_LAND_ROVER + MAK_NM_LEXUS + MAK_NM_MCLAREN_AUTOMOTIVE + MAK_NM_MERCURY + MAK_NM_MERKUR + MAK_NM_MINI + MAK_NM_MITSUBISHI + MAK_NM_NISSAN + MAK_NM_OLDSMOBILE + MAK_NM_PEUGEOT + MAK_NM_PLYMOUTH + MAK_NM_PONTIAC + MAK_NM_PORSCHE + MAK_NM_SAAB + MAK_NM_SATURN + MAK_NM_SMART + MAK_NM_SUBARU + MAK_NM_SUZUKI + MAK_NM_TESLA + MAK_NM_TOYOTA + MAK_NM_Tr_BMW + MAK_NM_Tr_CADILLAC + MAK_NM_Tr_CHEVROLET + MAK_NM_Tr_FORD + MAK_NM_Tr_HONDA + MAK_NM_Tr_JEEP + MAK_NM_Tr_LEXUS + MAK_NM_VOLKSWAGEN + MAK_NM_VOLVO ')
# forms.append('  + make_BMW + make_CADI + make_FORD + make_GMC + make_HOND + make_JEEP + make_KIA + make_LEXS + make_MAZD + make_MBNZ + make_MINI + make_MITS + make_NSSN + make_PONT + make_SUBA + make_TYTA ')
# forms.append('  + MFG_DESC_BMW + MFG_DESC_CHRYSLER_GROUP_LLC + MFG_DESC_DAEWOO + MFG_DESC_FCA + MFG_DESC_FIAT + MFG_DESC_FISKER_AUTOMOTIVE_INC_ + MFG_DESC_FORD + MFG_DESC_HONDA + MFG_DESC_HYUNDAI + MFG_DESC_MAZDA_MOTOR_CORPORATIO + MFG_DESC_MCLAREN_AUTOMOTIVE + MFG_DESC_MITSUBISHI + MFG_DESC_NISSAN + MFG_DESC_PORSCHE + MFG_DESC_SAAB_AUTOMOBILE_USA + MFG_DESC_SUBARU + MFG_DESC_SUZUKI + MFG_DESC_TESLA_MOTORS_INC_ + MFG_DESC_TOYOTA + MFG_DESC_U + MFG_DESC_VOLKSWAGEN ')
# forms.append('  + NADA_BODY1_N + NADA_BODY1_S + NC_ABS4w_S + NC_Airbag_P_NONE + NC_Airbag_P_STD + NC_AntiTheft_TA + NC_AntiTheft_TB + NC_AntiTheft_TC + NC_CrashDataRecorder_S + NC_DayRunningLights_O + NC_DayRunningLights_S + NC_Drive_FWD + NC_Drive_RWD ')
# forms.append('  + NC_HD_INJ_C_D_281 + NC_HD_INJ_C_D_282 + NC_HD_INJ_C_D_283 + NC_HD_INJ_C_D_290 + NC_HD_INJ_C_D_293 + NC_HD_INJ_C_D_295 + NC_HD_INJ_C_D_296 + NC_HD_INJ_C_D_299 + NC_HD_INJ_C_D_303 + NC_HD_INJ_C_D_305 + NC_HD_INJ_C_D_306 + NC_HD_INJ_C_D_308 + NC_HD_INJ_C_D_312 + NC_HD_INJ_C_D_314 + NC_HD_INJ_C_D_321 + NC_HD_INJ_C_D_322 + NC_HD_INJ_C_D_324 + NC_HD_INJ_C_D_325 + NC_HD_INJ_C_D_326 + NC_HD_INJ_C_D_327 + NC_HD_INJ_C_D_329 + NC_HD_INJ_C_D_330 + NC_HD_INJ_C_D_332 + NC_HD_INJ_C_D_333 + NC_HD_INJ_C_D_335 + NC_HD_INJ_C_D_336 + NC_HD_INJ_C_D_337 + NC_HD_INJ_C_D_338 + NC_HD_INJ_C_D_339 + NC_HD_INJ_C_D_340 + NC_HD_INJ_C_D_341 + NC_HD_INJ_C_D_342 + NC_HD_INJ_C_D_343 + NC_HD_INJ_C_D_344 + NC_HD_INJ_C_D_345 + NC_HD_INJ_C_D_346 + NC_HD_INJ_C_D_350 + NC_HD_INJ_C_D_356 + NC_HD_INJ_C_D_357 + NC_HD_INJ_C_D_358 + NC_HD_INJ_C_D_359 + NC_HD_INJ_C_D_360 + NC_HD_INJ_C_D_361 + NC_HD_INJ_C_D_362 + NC_HD_INJ_C_D_363 + NC_HD_INJ_C_D_366 + NC_HD_INJ_C_D_367 + NC_HD_INJ_C_D_368 + NC_HD_INJ_C_D_370 + NC_HD_INJ_C_D_371 + NC_HD_INJ_C_D_372 + NC_HD_INJ_C_D_378 + NC_HD_INJ_C_D_379 + NC_HD_INJ_C_D_383 + NC_HD_INJ_C_D_384 + NC_HD_INJ_C_D_412 + NC_HD_INJ_C_D_415 + NC_HD_INJ_C_D_417 + NC_HD_INJ_C_D_418 + NC_HD_INJ_C_D_420 + NC_HD_INJ_C_D_423 + NC_HD_INJ_C_D_424 + NC_HD_INJ_C_D_425 + NC_HD_INJ_C_D_426 + NC_HD_INJ_C_D_427 + NC_HD_INJ_C_D_428 + NC_HD_INJ_C_D_445 + NC_HD_INJ_C_D_448 + NC_HD_INJ_C_D_450 + NC_HD_INJ_C_D_452 + NC_HD_INJ_C_D_492 + NC_HD_INJ_C_D_577 + NC_HD_INJ_C_D_595 + NC_HD_INJ_C_D_624 ')
# forms.append('  + NC_HD_INJ_C_P_1525 + NC_HD_INJ_C_P_350 + NC_HD_INJ_C_P_452 + NC_HD_INJ_C_P_454 + NC_HD_INJ_C_P_455 + NC_HD_INJ_C_P_457 + NC_HD_INJ_C_P_460 + NC_HD_INJ_C_P_462 + NC_HD_INJ_C_P_464 + NC_HD_INJ_C_P_477 + NC_HD_INJ_C_P_486 + NC_HD_INJ_C_P_512 + NC_HD_INJ_C_P_515 + NC_HD_INJ_C_P_516 + NC_HD_INJ_C_P_517 + NC_HD_INJ_C_P_518 + NC_HD_INJ_C_P_521 + NC_HD_INJ_C_P_522 + NC_HD_INJ_C_P_523 + NC_HD_INJ_C_P_524 + NC_HD_INJ_C_P_525 + NC_HD_INJ_C_P_526 + NC_HD_INJ_C_P_527 + NC_HD_INJ_C_P_528 + NC_HD_INJ_C_P_529 + NC_HD_INJ_C_P_531 + NC_HD_INJ_C_P_532 + NC_HD_INJ_C_P_533 + NC_HD_INJ_C_P_534 + NC_HD_INJ_C_P_535 + NC_HD_INJ_C_P_538 + NC_HD_INJ_C_P_622 ')
# forms.append('  + NC_HeadAirbag_SB1 + NC_HeadAirbag_SC2 + NC_RearCtrLapShldrBelt_S + NC_RearSeatHeadRestraint_S + NC_SeatBeltReminder_Indicators_S + NC_SideAirbag_OBS1 + NC_SideAirbag_SBS1 + NC_SideAirbag_STS1 ')
# forms.append('  + NC_TireSize_P205_45R17 + NC_TireSize_P245_70R16 + NC_TireSize_P255_65R16 + NC_TireSize4_225_45R17 + NC_TireSize4_P215_60R16 + NC_TireSize4_P235_50R18 + NC_TireSize4_P255_65R16 ')
# forms.append('  + NC_TractionControl_S + NC_VTStabilityControl_YES + NC_WheelsDriven_2WD + NC_WheelsDriven_4 + NC_WheelsDriven_4WD + NC_WheelsDriven_FWD + NC_WheelsDriven_FWD_AWD + NC_WheelsDriven_RWD_4WD + numOfCylinders_3 + numOfCylinders_4 + numOfCylinders_5 ')
# forms.append('  + OPT1_ENTERTAIN_CD_4 + OPT1_ROOF_CD_6 + OPT1_ROOF_CD_U + PLNT_CD_K + PLNT_CD_S + PLNT_CD_X + PLNT_CNTRY_NM_BRAZIL + PLNT_CNTRY_NM_ITALY + PLNT_CNTRY_NM_JAPAN + PLNT_CNTRY_NM_MEXICO + PLNT_CNTRY_NM_SOUTH_AFRICA + PLNT_CNTRY_NM_THAILAND ')
# # 20
# forms.append('  + priceNewSymbl27_2_02 + priceNewSymbl27_2_03 + priceNewSymbl27_2_04 + priceNewSymbl27_2_05 + priceNewSymbl27_2_06 + priceNewSymbl27_2_07 + priceNewSymbl27_2_08 + priceNewSymbl27_2_10 + priceNewSymbl27_2_11 + priceNewSymbl27_2_12 + priceNewSymbl27_2_13 + priceNewSymbl27_2_14 + priceNewSymbl27_2_15 + priceNewSymbl27_2_16 + priceNewSymbl27_2_17 + priceNewSymbl27_2_18 + priceNewSymbl27_2_19 + priceNewSymbl27_2_20 + priceNewSymbl27_2_21 + priceNewSymbl27_2_23 ')
# forms.append('  + PWR_BRK_OPT_CD_N + PWR_BRK_OPT_CD_U + PWR_STRNG_OPT_CD_N + PWR_STRNG_OPT_CD_O + PWR_STRNG_OPT_CD_U ')
# forms.append('  + REAR_TIRE_SIZE_CD_13 + REAR_TIRE_SIZE_CD_14 + REAR_TIRE_SIZE_CD_15 + REAR_TIRE_SIZE_CD_16 + REAR_TIRE_SIZE_CD_25 + REAR_TIRE_SIZE_CD_27 + REAR_TIRE_SIZE_CD_29 + REAR_TIRE_SIZE_CD_30 + REAR_TIRE_SIZE_CD_32 + REAR_TIRE_SIZE_CD_34 + REAR_TIRE_SIZE_CD_35 + REAR_TIRE_SIZE_CD_36 + REAR_TIRE_SIZE_CD_37 + REAR_TIRE_SIZE_CD_38 + REAR_TIRE_SIZE_CD_47 + REAR_TIRE_SIZE_CD_59 + REAR_TIRE_SIZE_DESC_16R195 + REAR_TIRE_SIZE_DESC_17R245 + REAR_TIRE_SIZE_DESC_18R285 ')
# forms.append('  + restraintCd_9 + restraintCd_H + restraintCd_N + restraintCd_T + ROOF_CD_3 + ROOF_CD_6 + RSTRNT_TYP_CD_A + RSTRNT_TYP_CD_G + RSTRNT_TYP_CD_M + RSTRNT_TYP_CD_P + RSTRNT_TYP_CD_R + RSTRNT_TYP_CD_S + RSTRNT_TYP_CD_Tr_E + RSTRNT_TYP_CD_Tr_K + RSTRNT_TYP_CD_Tr_O + RSTRNT_TYP_CD_Tr_R ')
# forms.append('  + SECUR_TYP_CD_E + SECUR_TYP_CD_F + SECUR_TYP_CD_K + SECUR_TYP_CD_N + SECUR_TYP_CD_P + SECUR_TYP_CD_T + SECUR_TYP_CD_U + SECUR_TYP_CD_Z + SEGMENTATION_CD_0 + SEGMENTATION_CD_2 + SEGMENTATION_CD_4 + SEGMENTATION_CD_5 + SEGMENTATION_CD_6 + SEGMENTATION_CD_A + SEGMENTATION_CD_B + SEGMENTATION_CD_C + SEGMENTATION_CD_H + SEGMENTATION_CD_I + SEGMENTATION_CD_P ')
# forms.append('  + TLT_STRNG_WHL_OPT_CD_O + TLT_STRNG_WHL_OPT_CD_U + tonCd__ + tonCd_10 ')
# forms.append('  + TRANS_CD_B + TRANS_CD_M + TRANS_CD_U + TRANS_SPEED_CD__ + TRANS_SPEED_CD_5 + TRANS_SPEED_CD_6 + TRANS_SPEED_CD_7 ')
# forms.append('  + TRK_BED_LEN_CD_E + TRK_BED_LEN_CD_L + TRK_BED_LEN_CD_R + TRK_BED_LEN_CD_S + TRK_BRK_TYP_CD_H_V + TRK_BRK_TYP_CD_HYD + TRK_BRK_TYP_CD_Tr_HYD ')
# forms.append('  + TRK_CAB_CNFG_CD_CMN + TRK_CAB_CNFG_CD_CON + TRK_CAB_CNFG_CD_CRW + TRK_CAB_CNFG_CD_CUT + TRK_CAB_CNFG_CD_EXT + TRK_CAB_CNFG_CD_TLO + TRK_CAB_CNFG_CD_Tr_0_U + TRK_CAB_CNFG_CD_Tr_3_EXT + TRK_CAB_CNFG_CD_U + TRK_CAB_CNFG_CD_VAN + TRK_FRNT_AXL_CD_N + TRK_REAR_AXL_CD_S + TRK_TNG_RAT_CD_A + TRK_TNG_RAT_CD_B + TRK_TNG_RAT_CD_BC + TRK_TNG_RAT_CD_C ')
# forms.append('  + VINA_BODY_TYPE_CD_2W + VINA_BODY_TYPE_CD_4D + VINA_BODY_TYPE_CD_C4 + VINA_BODY_TYPE_CD_RD + vinmasterPerformanceCd_1 + WHL_DRVN_CNT_Tr__ + WHL_DRVN_CNT_Tr_2')
# # 30
# forms.append(' + AIR_COND_OPT_CD_U + BODY_STYLE_CD_CV + BODY_STYLE_DESC_CONVERTIBLE + bodyStyleCd_CABRIOLE + bodyStyleCd_CAMPER + bodyStyleCd_CNV4X42D + bodyStyleCd_CONV_2D + bodyStyleCd_CONVRTBL + bodyStyleCd_COUPE + bodyStyleCd_CPE4X44D + bodyStyleCd_HCH3D4X4 + cko_eng_cylinders__ + cko_eng_cylinders_2 + cko_eng_cylinders_3 + cko_eng_cylinders_4 + cko_eng_cylinders_5 + cko_eng_cylinders_8 + cko_eng_cylinders_Tr_8 + classCd_52 + ENG_MDL_CD_050014 + ENG_MDL_CD_050016+ ENG_MDL_CD_080006 + ENG_MFG_CD_070 + FRNT_TYRE_SIZE_CD_85 + FRNT_TYRE_SIZE_Desc_12R165 + FRNT_TYRE_SIZE_Desc_12R415 + FRNT_TYRE_SIZE_Desc_13R145 + FRNT_TYRE_SIZE_Desc_14R215 + FRNT_TYRE_SIZE_Desc_14R225 + FRNT_TYRE_SIZE_Desc_15BH78 + FRNT_TYRE_SIZE_Desc_15H78 + FRNT_TYRE_SIZE_Desc_15HR78 + FRNT_TYRE_SIZE_Desc_15R145 + MAK_NM_ALFA_ROMEO + MAK_NM_DAIHATSU + MAK_NM_DATSUN + MAK_NM_FISKER_AUTOMOTIVE + MAK_NM_HONDA + MAK_NM_MCLAREN_AUTOMOTIVE + make_LEXS + make_MBNZ + MFG_DESC_FISKER_AUTOMOTIVE_INC_ + MFG_DESC_MAZDA_MOTOR_CORPORATIO + MFG_DESC_MCLAREN_AUTOMOTIVE + NC_HD_INJ_C_D_367 + NC_HD_INJ_C_D_383 + NC_HD_INJ_C_P_350 + NC_HD_INJ_C_P_457 + NC_HD_INJ_C_P_477 + NC_HD_INJ_C_P_515 + NC_HD_INJ_C_P_517 + NC_HD_INJ_C_P_528 + NC_HD_INJ_C_P_529 + NC_TireSize4_225_45R17 + NC_TireSize4_P215_60R16 + NC_TireSize4_P235_50R18 + NC_TractionControl_S + numOfCylinders_4 + OPT1_ENTERTAIN_CD_4 + PLNT_CD_X + priceNewSymbl27_2_19 + REAR_TIRE_SIZE_CD_16 + REAR_TIRE_SIZE_CD_47 + REAR_TIRE_SIZE_DESC_17R245 + ROOF_CD_6 + RSTRNT_TYP_CD_A + RSTRNT_TYP_CD_G + TLT_STRNG_WHL_OPT_CD_U + TRK_TNG_RAT_CD_BC + TRK_TNG_RAT_CD_C + VINA_BODY_TYPE_CD_C4 + VINA_BODY_TYPE_CD_RD + vinmasterPerformanceCd_1 + WHL_DRVN_CNT_Tr__ ')
# forms.append(' + BODY_STYLE_CD_CV + BODY_STYLE_DESC_CONVERTIBLE + bodyStyleCd_CABRIOLE + bodyStyleCd_CAMPER + bodyStyleCd_CNV4X42D + bodyStyleCd_COUPE + bodyStyleCd_CPE4X44D + bodyStyleCd_HCH3D4X4 + cko_eng_cylinders__ 	+ cko_eng_cylinders_2 + cko_eng_cylinders_3 + cko_eng_cylinders_4 + cko_eng_cylinders_5 + cko_eng_cylinders_8 + cko_eng_cylinders_Tr_8 + ENG_MDL_CD_050014 + ENG_MDL_CD_050016+ ENG_MDL_CD_080006 + ENG_MFG_CD_070 + FRNT_TYRE_SIZE_CD_85  + FRNT_TYRE_SIZE_Desc_12R415  + FRNT_TYRE_SIZE_Desc_14R215 + FRNT_TYRE_SIZE_Desc_14R225 + FRNT_TYRE_SIZE_Desc_15BH78 + FRNT_TYRE_SIZE_Desc_15H78 + FRNT_TYRE_SIZE_Desc_15HR78 + FRNT_TYRE_SIZE_Desc_15R145 + MAK_NM_ALFA_ROMEO + MAK_NM_DATSUN + MAK_NM_FISKER_AUTOMOTIVE + MAK_NM_MCLAREN_AUTOMOTIVE  + MFG_DESC_FISKER_AUTOMOTIVE_INC_ + MFG_DESC_MAZDA_MOTOR_CORPORATIO + MFG_DESC_MCLAREN_AUTOMOTIVE + NC_HD_INJ_C_D_367 + NC_HD_INJ_C_P_350 + NC_HD_INJ_C_P_457 + NC_HD_INJ_C_P_515 + NC_HD_INJ_C_P_517 + NC_HD_INJ_C_P_528 + NC_HD_INJ_C_P_529  + NC_TireSize4_P215_60R16 + NC_TireSize4_P235_50R18 + NC_TractionControl_S   + PLNT_CD_X  + REAR_TIRE_SIZE_CD_16 + REAR_TIRE_SIZE_CD_47 + REAR_TIRE_SIZE_DESC_17R245  + RSTRNT_TYP_CD_A + RSTRNT_TYP_CD_G + TLT_STRNG_WHL_OPT_CD_U + TRK_TNG_RAT_CD_BC + TRK_TNG_RAT_CD_C  + VINA_BODY_TYPE_CD_RD + vinmasterPerformanceCd_1 + WHL_DRVN_CNT_Tr__ ')

# forms.append(' + BODY_STYLE_CD_SD  + BODY_STYLE_CD_CP  + BODY_STYLE_CD_CC  + BODY_STYLE_CD_LM    + BODY_STYLE_CD_WG  + BODY_STYLE_CD_VC  + BODY_STYLE_CD_HB  + BODY_STYLE_CD_ST  + BODY_STYLE_CD_CV  + BODY_STYLE_CD_IP  + BODY_STYLE_CD_HR  + BODY_STYLE_CD_CH  + BODY_STYLE_CD_YY  + BODY_STYLE_CD_TU ')
# forms.append(' + BODY_STYLE_DESC_CONVERTIBLE+BODY_STYLE_DESC_CUTAWAY+BODY_STYLE_DESC_INCOMPLETE_PICKUP+BODY_STYLE_DESC_HATCHBACK+BODY_STYLE_DESC_COUPE+BODY_STYLE_DESC_STRAIGHT_TRUCK+BODY_STYLE_DESC_SPORT_UTILITY_TRUCK+BODY_STYLE_DESC_CAB_CHASSIS+BODY_STYLE_DESC_LIMOUSINE+BODY_STYLE_DESC_WAGON+BODY_STYLE_DESC_HEARSE+BODY_STYLE_DESC_VAN_CAMPER+BODY_STYLE_DESC_SEDAN+BODY_STYLE_DESC_COMMERCIAL_CHASSIS ')
# forms.append(' + bodyStyleCd_HCHBK_CP + bodyStyleCd_UTL4X24D + bodyStyleCd_SED4D4X2 + bodyStyleCd_WAG_4D + bodyStyleCd_HCH3D4X2 + bodyStyleCd_CONV_2D + bodyStyleCd_HCH3D4X4 + bodyStyleCd_WAG5D4X4 + bodyStyleCd_HCHBK_2D + bodyStyleCd_TRK_4X2 + bodyStyleCd_WAG5D4X2 + bodyStyleCd_SPORT_CP + bodyStyleCd_WAGON_2D + bodyStyleCd_PKP4X44D + bodyStyleCd_WAG4D4X4 + bodyStyleCd_CP_RDSTR + bodyStyleCd_CNV4X42D + bodyStyleCd_LFTBK_5D + bodyStyleCd_HCH2D4X4 + bodyStyleCd_UTIL_4X2 + bodyStyleCd_WAGON_5D + bodyStyleCd_SUT4X44D + bodyStyleCd_WAG_4X2 + bodyStyleCd_WAG4X44D + bodyStyleCd_SUPRA + bodyStyleCd_CABRI_2D + bodyStyleCd_CONVRTBL + bodyStyleCd_LFTBK_2D + bodyStyleCd_CAMPER + bodyStyleCd_SED4X44D + bodyStyleCd_COUPE_2D + bodyStyleCd_CPE4X44D + bodyStyleCd_CPE2D4X4 + bodyStyleCd_HRDTP_2D + bodyStyleCd_SED2D4X4 + bodyStyleCd_HCH5D4X4 + bodyStyleCd_HCHBK_4D + bodyStyleCd_SEDAN_4D + bodyStyleCd_WAGON_4D + bodyStyleCd_COUPE + bodyStyleCd_WAG4X24D + bodyStyleCd_SED4D4X4 + bodyStyleCd_TRUCK + bodyStyleCd__ + bodyStyleCd_LFBKGT3D + bodyStyleCd_WAG_3D + bodyStyleCd_SUT4X24D + bodyStyleCd_EST_WGN + bodyStyleCd_ROADSTER + bodyStyleCd_CABRIOLE + bodyStyleCd_CPE2D4X2 + bodyStyleCd_VAN + bodyStyleCd_COUPE_4D  + bodyStyleCd_LFTBK_4D + bodyStyleCd_WAG4D4X2 + bodyStyleCd_UTL4X42D + bodyStyleCd_LIMO + bodyStyleCd_SP_HCHBK + bodyStyleCd_CPE_2_2 + bodyStyleCd_HCHBK2_2 + bodyStyleCd_UTIL_4X4 + bodyStyleCd_SP_COUPE + bodyStyleCd_HCHBK_5D + bodyStyleCd_COUPE_3D + bodyStyleCd_VAN4X24D + bodyStyleCd_LFTBK_3D + bodyStyleCd_SPTCP_GT + bodyStyleCd_FASTBACK + bodyStyleCd_UTILITY + bodyStyleCd_HCHBK_3D + bodyStyleCd_SEDAN_2D + bodyStyleCd_TRK_4X4 + bodyStyleCd_PKP4X42D + bodyStyleCd_UTL4X22D + bodyStyleCd_TARGA + bodyStyleCd_VAN_4X4 + bodyStyleCd_MPV_4X4 + bodyStyleCd_WAG_4X4 + bodyStyleCd_CPE3D4X2 + bodyStyleCd_MPV_4X2 ')
# forms.append(' + cko_eng_cylinders__ + cko_eng_cylinders_2 + cko_eng_cylinders_3 + cko_eng_cylinders_4 + cko_eng_cylinders_5 + cko_eng_cylinders_8  ')
# forms.append(' + cko_eng_cylinders_Tr_4+ cko_eng_cylinders_Tr_8  ')
# forms.append(' + ENG_MDL_CD_050014 + ENG_MDL_CD_080085 + ENG_MDL_CD_050010 + ENG_MDL_CD_070005 + ENG_MDL_CD_050007 + ENG_MDL_CD_050060 + ENG_MDL_CD_070003 + ENG_MDL_CD_080050 + ENG_MDL_CD_070024 + ENG_MDL_CD_207020 + ENG_MDL_CD_050009 + ENG_MDL_CD_207080 + ENG_MDL_CD_080070 + ENG_MDL_CD_070011 + ENG_MDL_CD_165008 + ENG_MDL_CD_080084  + ENG_MDL_CD_070023 + ENG_MDL_CD_070009 + ENG_MDL_CD_070014 + ENG_MDL_CD_165051 + ENG_MDL_CD_239003 + ENG_MDL_CD_070008 + ENG_MDL_CD_080003 + ENG_MDL_CD_080015 + ENG_MDL_CD_050005 + ENG_MDL_CD_080020 + ENG_MDL_CD_050012 + ENG_MDL_CD_207030 + ENG_MDL_CD_040010 + ENG_MDL_CD_080114 + ENG_MDL_CD_050015 + ENG_MDL_CD_080080 + ENG_MDL_CD_207050 + ENG_MDL_CD_080008 + ENG_MDL_CD_070004 + ENG_MDL_CD_080040 + ENG_MDL_CD_080044 + ENG_MDL_CD_050016 + ENG_MDL_CD_080002 + ENG_MDL_CD_070018 + ENG_MDL_CD_207065 + ENG_MDL_CD_070007 + ENG_MDL_CD_080110 + ENG_MDL_CD_030040 + ENG_MDL_CD_080010 + ENG_MDL_CD_070015 + ENG_MDL_CD_080115 + ENG_MDL_CD_070010 + ENG_MDL_CD_238004 + ENG_MDL_CD_070013 + ENG_MDL_CD_080006 + ENG_MDL_CD_080100 + ENG_MDL_CD_120020 + ENG_MDL_CD_050006 + ENG_MDL_CD_207060 + ENG_MDL_CD_080130 ')
# forms.append(' + ENG_MFG_CD_080 + ENG_MFG_CD_235 + ENG_MFG_CD_040 + ENG_MFG_CD_060  + ENG_MFG_CD_204 + ENG_MFG_CD_180 + ENG_MFG_CD_050 + ENG_MFG_CD_207 + ENG_MFG_CD_070 + ENG_MFG_CD_238 + ENG_MFG_CD_030 + ENG_MFG_CD_120 + ENG_MFG_CD_165' )
# forms.append(' + FRNT_TYRE_SIZE_CD_90 + FRNT_TYRE_SIZE_CD_449 + FRNT_TYRE_SIZE_CD_34 + FRNT_TYRE_SIZE_CD_74 + FRNT_TYRE_SIZE_CD_64 + FRNT_TYRE_SIZE_CD_5 + FRNT_TYRE_SIZE_CD_25 + FRNT_TYRE_SIZE_CD_80 + FRNT_TYRE_SIZE_CD_63 + FRNT_TYRE_SIZE_CD_6 + FRNT_TYRE_SIZE_CD_77 + FRNT_TYRE_SIZE_CD_75 + FRNT_TYRE_SIZE_CD_92 + FRNT_TYRE_SIZE_CD_60 + FRNT_TYRE_SIZE_CD_44 + FRNT_TYRE_SIZE_CD_56 + FRNT_TYRE_SIZE_CD_453 + FRNT_TYRE_SIZE_CD_47 + FRNT_TYRE_SIZE_CD_7 + FRNT_TYRE_SIZE_CD_50 + FRNT_TYRE_SIZE_CD_66 + FRNT_TYRE_SIZE_CD_39 + FRNT_TYRE_SIZE_CD_59 + FRNT_TYRE_SIZE_CD_87 + FRNT_TYRE_SIZE_CD_49 + FRNT_TYRE_SIZE_CD_86 + FRNT_TYRE_SIZE_CD_459 + FRNT_TYRE_SIZE_CD_48 + FRNT_TYRE_SIZE_CD_406 + FRNT_TYRE_SIZE_CD_79 + FRNT_TYRE_SIZE_CD_57 + FRNT_TYRE_SIZE_CD_442 + FRNT_TYRE_SIZE_CD_1 + FRNT_TYRE_SIZE_CD_19 + FRNT_TYRE_SIZE_CD_21 + FRNT_TYRE_SIZE_CD_9 + FRNT_TYRE_SIZE_CD_67 + FRNT_TYRE_SIZE_CD_36 + FRNT_TYRE_SIZE_CD_51 + FRNT_TYRE_SIZE_CD_82 + FRNT_TYRE_SIZE_CD_13 + FRNT_TYRE_SIZE_CD_31 + FRNT_TYRE_SIZE_CD_85 + FRNT_TYRE_SIZE_CD_76 + FRNT_TYRE_SIZE_CD_35 + FRNT_TYRE_SIZE_CD_12 + FRNT_TYRE_SIZE_CD_72 + FRNT_TYRE_SIZE_CD_54 + FRNT_TYRE_SIZE_CD_81 + FRNT_TYRE_SIZE_CD_20 + FRNT_TYRE_SIZE_CD_53 + FRNT_TYRE_SIZE_CD_73 + FRNT_TYRE_SIZE_CD_28 + FRNT_TYRE_SIZE_CD_30 + FRNT_TYRE_SIZE_CD_70 + FRNT_TYRE_SIZE_CD_15 + FRNT_TYRE_SIZE_CD_4 + FRNT_TYRE_SIZE_CD_46 + FRNT_TYRE_SIZE_CD_41 + FRNT_TYRE_SIZE_CD_417 + FRNT_TYRE_SIZE_CD_434 + FRNT_TYRE_SIZE_CD_17 + FRNT_TYRE_SIZE_CD_61 + FRNT_TYRE_SIZE_CD_58 + FRNT_TYRE_SIZE_CD_62 + FRNT_TYRE_SIZE_CD_55 + FRNT_TYRE_SIZE_CD_98 + FRNT_TYRE_SIZE_CD_447 + FRNT_TYRE_SIZE_CD_11 + FRNT_TYRE_SIZE_CD_88 + FRNT_TYRE_SIZE_CD_3 + FRNT_TYRE_SIZE_CD_40 + FRNT_TYRE_SIZE_CD_71 + FRNT_TYRE_SIZE_CD_68 + FRNT_TYRE_SIZE_CD_65 + FRNT_TYRE_SIZE_CD_27 + FRNT_TYRE_SIZE_CD_96 + FRNT_TYRE_SIZE_CD_37 + FRNT_TYRE_SIZE_CD_45 + FRNT_TYRE_SIZE_CD_8 + FRNT_TYRE_SIZE_CD_262 + FRNT_TYRE_SIZE_CD_32 + FRNT_TYRE_SIZE_CD_465  + FRNT_TYRE_SIZE_CD_38 + FRNT_TYRE_SIZE_CD_433 + FRNT_TYRE_SIZE_CD_43 + FRNT_TYRE_SIZE_CD_14 + FRNT_TYRE_SIZE_CD_33 + FRNT_TYRE_SIZE_CD_78 + FRNT_TYRE_SIZE_CD_29 + FRNT_TYRE_SIZE_CD_461 + FRNT_TYRE_SIZE_CD_42 + FRNT_TYRE_SIZE_CD_69   ')
# # 40
# forms.append(' + FRNT_TYRE_SIZE_Desc_18R255 + FRNT_TYRE_SIZE_Desc_18R235 + FRNT_TYRE_SIZE_Desc_16R185 + FRNT_TYRE_SIZE_Desc_17R275 + FRNT_TYRE_SIZE_Desc_16R205 + FRNT_TYRE_SIZE_Desc_18R245 + FRNT_TYRE_SIZE_Desc_14R165 + FRNT_TYRE_SIZE_Desc_14RP15 + FRNT_TYRE_SIZE_Desc_21R245 + FRNT_TYRE_SIZE_Desc_20R275 + FRNT_TYRE_SIZE_Desc_14R225 + FRNT_TYRE_SIZE_Desc_20R245 + FRNT_TYRE_SIZE_Desc_18R215 + FRNT_TYRE_SIZE_Desc_15R185 + FRNT_TYRE_SIZE_Desc_15R265 + FRNT_TYRE_SIZE_Desc_19R245 + FRNT_TYRE_SIZE_Desc_90R225 + FRNT_TYRE_SIZE_Desc_18R285 + FRNT_TYRE_SIZE_Desc_18R295 + FRNT_TYRE_SIZE_Desc_19R225 + FRNT_TYRE_SIZE_Desc_17R265 + FRNT_TYRE_SIZE_Desc_21R255  + FRNT_TYRE_SIZE_Desc_13R155 + FRNT_TYRE_SIZE_Desc_18R265 + FRNT_TYRE_SIZE_Desc_14R195 + FRNT_TYRE_SIZE_Desc_19R235 + FRNT_TYRE_SIZE_Desc_21R275 + FRNT_TYRE_SIZE_Desc_19R275 + FRNT_TYRE_SIZE_Desc_16R225 + FRNT_TYRE_SIZE_Desc_14R215 + FRNT_TYRE_SIZE_Desc_13R175 + FRNT_TYRE_SIZE_Desc_16R255 + FRNT_TYRE_SIZE_Desc_15BH78 + FRNT_TYRE_SIZE_Desc_15R205 + FRNT_TYRE_SIZE_Desc_13R145 + FRNT_TYRE_SIZE_Desc_15R145 + FRNT_TYRE_SIZE_Desc_15R190 + FRNT_TYRE_SIZE_Desc_20R235 + FRNT_TYRE_SIZE_Desc_13R165 + FRNT_TYRE_SIZE_Desc_17R235 + FRNT_TYRE_SIZE_Desc_15R195 + FRNT_TYRE_SIZE_Desc_17R255 + FRNT_TYRE_SIZE_Desc_20R255 + FRNT_TYRE_SIZE_Desc_15R175 + FRNT_TYRE_SIZE_Desc_19R305 + FRNT_TYRE_SIZE_Desc_13R205 + FRNT_TYRE_SIZE_Desc_22R265 + FRNT_TYRE_SIZE_Desc_20R265 + FRNT_TYRE_SIZE_Desc_16R235 + FRNT_TYRE_SIZE_Desc_20R285 + FRNT_TYRE_SIZE_Desc_19R265 + FRNT_TYRE_SIZE_Desc_16R215 + FRNT_TYRE_SIZE_Desc_16R175 + FRNT_TYRE_SIZE_Desc_21R265 + FRNT_TYRE_SIZE_Desc_14R205 + FRNT_TYRE_SIZE_Desc_17R245 + FRNT_TYRE_SIZE_Desc_17R215 + FRNT_TYRE_SIZE_Desc_20R305 + FRNT_TYRE_SIZE_Desc_18R275 + FRNT_TYRE_SIZE_Desc_16R195 + FRNT_TYRE_SIZE_Desc_19R285 + FRNT_TYRE_SIZE_Desc_16R265 + FRNT_TYRE_SIZE_Desc_14R175 + FRNT_TYRE_SIZE_Desc_15R225 + FRNT_TYRE_SIZE_Desc_20R315 + FRNT_TYRE_SIZE_Desc_12R415 + FRNT_TYRE_SIZE_Desc_15HR78 + FRNT_TYRE_SIZE_Desc_17R225 + FRNT_TYRE_SIZE_Desc_17R285 + FRNT_TYRE_SIZE_Desc_16R275 + FRNT_TYRE_SIZE_Desc_21R295 + FRNT_TYRE_SIZE_Desc_22R285 + FRNT_TYRE_SIZE_Desc_17R205 + FRNT_TYRE_SIZE_Desc_15R215 + FRNT_TYRE_SIZE_Desc_19R255 + FRNT_TYRE_SIZE_Desc_12R165 + FRNT_TYRE_SIZE_Desc_13R185 + FRNT_TYRE_SIZE_Desc_17R315 + FRNT_TYRE_SIZE_Desc_13R195 + FRNT_TYRE_SIZE_Desc_20R295 + FRNT_TYRE_SIZE_Desc_14R185 + FRNT_TYRE_SIZE_Desc_16R245 + FRNT_TYRE_SIZE_Desc_18R225 + FRNT_TYRE_SIZE_Desc_31R265 + FRNT_TYRE_SIZE_Desc_15R255 + FRNT_TYRE_SIZE_Desc_15H78 + FRNT_TYRE_SIZE_Desc_15R235 + FRNT_TYRE_SIZE_Desc_15R245 + FRNT_TYRE_SIZE_Desc_15R155 + FRNT_TYRE_SIZE_Desc_12R145 ')
# forms.append(' + MAK_NM_MERKUR + MAK_NM_IVECO + MAK_NM_DAEWOO + MAK_NM_MERCURY + MAK_NM_NISSAN + MAK_NM_AMERICAN_GENERAL + MAK_NM_ASTON_MARTIN + MAK_NM_FERRARI + MAK_NM_CHRYSLER  + MAK_NM_DODGE + MAK_NM_PONTIAC + MAK_NM_PLYMOUTH + MAK_NM_MAYBACH + MAK_NM_LINCOLN + MAK_NM_JAGUAR + MAK_NM_BMW + MAK_NM_HUMMER + MAK_NM_ALFA_ROMEO + MAK_NM_DAIHATSU + MAK_NM_LAND_ROVER + MAK_NM_ACURA + MAK_NM_PORSCHE + MAK_NM_LANCIA + MAK_NM_MAZDA + MAK_NM_MITSUBISHI + MAK_NM_GMC + MAK_NM_BENTLEY + MAK_NM_CADILLAC + MAK_NM_LEXUS + MAK_NM_FIAT + MAK_NM_DATSUN + MAK_NM_VOLVO + MAK_NM_ISUZU + MAK_NM_MASERATI + MAK_NM_HONDA + MAK_NM_AMERICAN_MOTORS + MAK_NM_LOTUS + MAK_NM_KIA + MAK_NM_PEUGEOT + MAK_NM_SATURN + MAK_NM_GLOBAL_ELECTRIC_MOTORS + MAK_NM_JEEP + MAK_NM_OLDSMOBILE + MAK_NM_SUBARU + MAK_NM_SMART + MAK_NM_FISKER_AUTOMOTIVE + MAK_NM_SAAB + MAK_NM_SUZUKI + MAK_NM_VOLKSWAGEN + MAK_NM_TESLA + MAK_NM_FORD + MAK_NM_MERCEDES_BENZ + MAK_NM_MINI + MAK_NM_BUICK + MAK_NM_ROLLS_ROYCE + MAK_NM_INFINITI + MAK_NM_EAGLE + MAK_NM_HYUNDAI + MAK_NM_GEO + MAK_NM_TOYOTA + MAK_NM_AUDI + MAK_NM_MCLAREN_AUTOMOTIVE + MAK_NM_LAMBORGHINI ')
# forms.append(' + MFG_DESC_CHRYSLER_GROUP_LLC + MFG_DESC_DAEWOO + MFG_DESC_FORD + MFG_DESC_NISSAN + MFG_DESC_TESLA_MOTORS_INC_ + MFG_DESC_MAZDA_MOTOR_CORPORATIO + MFG_DESC_DAIMLER_CHRYSLER + MFG_DESC_MERCEDES_BENZ_USA_LLC + MFG_DESC_U + MFG_DESC_BMW + MFG_DESC_FCA  + MFG_DESC_PORSCHE + MFG_DESC_MITSUBISHI + MFG_DESC_FIAT + MFG_DESC_ISUZU + MFG_DESC_HONDA + MFG_DESC_KIA + MFG_DESC_SAAB_AUTOMOBILE_USA + MFG_DESC_SUBARU + MFG_DESC_SUZUKI + MFG_DESC_VOLKSWAGEN  + MFG_DESC_FISKER_AUTOMOTIVE_INC_ + MFG_DESC_HYUNDAI + MFG_DESC_TOYOTA + MFG_DESC_MCLAREN_AUTOMOTIVE ')
# forms.append(' + NC_HD_INJ_C_D_660 + NC_HD_INJ_C_D_873 + NC_HD_INJ_C_D_432 + NC_HD_INJ_C_D_375 + NC_HD_INJ_C_D_638 + NC_HD_INJ_C_D_1004 + NC_HD_INJ_C_D_342 + NC_HD_INJ_C_D_824 + NC_HD_INJ_C_D_683 + NC_HD_INJ_C_D_669 + NC_HD_INJ_C_D_305 + NC_HD_INJ_C_D_277 + NC_HD_INJ_C_D_734 + NC_HD_INJ_C_D_536 + NC_HD_INJ_C_D_454 + NC_HD_INJ_C_D_508 + NC_HD_INJ_C_D_469 + NC_HD_INJ_C_D_870 + NC_HD_INJ_C_D_450 + NC_HD_INJ_C_D_641 + NC_HD_INJ_C_D_708 + NC_HD_INJ_C_D_567 + NC_HD_INJ_C_D_959 + NC_HD_INJ_C_D_325 + NC_HD_INJ_C_D_460 + NC_HD_INJ_C_D_238 + NC_HD_INJ_C_D_835 + NC_HD_INJ_C_D_442 + NC_HD_INJ_C_D_557 + NC_HD_INJ_C_D_439 + NC_HD_INJ_C_D_707 + NC_HD_INJ_C_D_275 + NC_HD_INJ_C_D_1200 + NC_HD_INJ_C_D_583 + NC_HD_INJ_C_D_858 + NC_HD_INJ_C_D_366 + NC_HD_INJ_C_D_504 + NC_HD_INJ_C_D_2021 + NC_HD_INJ_C_D_472 + NC_HD_INJ_C_D_1238 + NC_HD_INJ_C_D_376 + NC_HD_INJ_C_D_724 + NC_HD_INJ_C_D_327 + NC_HD_INJ_C_D_ND + NC_HD_INJ_C_D_907 + NC_HD_INJ_C_D_786 + NC_HD_INJ_C_D_530 + NC_HD_INJ_C_D_510 + NC_HD_INJ_C_D_887 + NC_HD_INJ_C_D_673 + NC_HD_INJ_C_D_430 + NC_HD_INJ_C_D_919 + NC_HD_INJ_C_D_548 + NC_HD_INJ_C_D_608 + NC_HD_INJ_C_D_684 + NC_HD_INJ_C_D_251 + NC_HD_INJ_C_D_330 + NC_HD_INJ_C_D_434 + NC_HD_INJ_C_D_779 + NC_HD_INJ_C_D_341 + NC_HD_INJ_C_D_437 + NC_HD_INJ_C_D_407 + NC_HD_INJ_C_D_665 + NC_HD_INJ_C_D_321 + NC_HD_INJ_C_D_525 + NC_HD_INJ_C_D_269 + NC_HD_INJ_C_D_499 + NC_HD_INJ_C_D_1027 + NC_HD_INJ_C_D_930 + NC_HD_INJ_C_D_362 + NC_HD_INJ_C_D_580 + NC_HD_INJ_C_D_431 + NC_HD_INJ_C_D_346 + NC_HD_INJ_C_D_772 + NC_HD_INJ_C_D_573 + NC_HD_INJ_C_D_505 + NC_HD_INJ_C_D_247 + NC_HD_INJ_C_D_293 + NC_HD_INJ_C_D_747 + NC_HD_INJ_C_D_517 + NC_HD_INJ_C_D_560 + NC_HD_INJ_C_D_374 + NC_HD_INJ_C_D_233 + NC_HD_INJ_C_D_492 + NC_HD_INJ_C_D_872 + NC_HD_INJ_C_D_770 + NC_HD_INJ_C_D_682 + NC_HD_INJ_C_D_627 + NC_HD_INJ_C_D_381 + NC_HD_INJ_C_D_299 + NC_HD_INJ_C_D_596 + NC_HD_INJ_C_D_441  + NC_HD_INJ_C_D_818 + NC_HD_INJ_C_D_370 + NC_HD_INJ_C_D_643 + NC_HD_INJ_C_D_815 + NC_HD_INJ_C_D_273 + NC_HD_INJ_C_D_918 + NC_HD_INJ_C_D_461 + NC_HD_INJ_C_D_308 + NC_HD_INJ_C_D_271 + NC_HD_INJ_C_D_423 + NC_HD_INJ_C_D_803 + NC_HD_INJ_C_D_414 + NC_HD_INJ_C_D_345 + NC_HD_INJ_C_D_449 + NC_HD_INJ_C_D_236 + NC_HD_INJ_C_D_549 + NC_HD_INJ_C_D_516 + NC_HD_INJ_C_D_211 + NC_HD_INJ_C_D_712 + NC_HD_INJ_C_D_169 + NC_HD_INJ_C_D_575 + NC_HD_INJ_C_D_589 + NC_HD_INJ_C_D_846 + NC_HD_INJ_C_D_562 + NC_HD_INJ_C_D_1564 + NC_HD_INJ_C_D_491 + NC_HD_INJ_C_D_778 + NC_HD_INJ_C_D_674 + NC_HD_INJ_C_D_735 + NC_HD_INJ_C_D_597 + NC_HD_INJ_C_D_322 + NC_HD_INJ_C_D_564 + NC_HD_INJ_C_D_435 + NC_HD_INJ_C_D_485 + NC_HD_INJ_C_D_296 + NC_HD_INJ_C_D_1459 + NC_HD_INJ_C_D_591 + NC_HD_INJ_C_D_1219 + NC_HD_INJ_C_D_960 + NC_HD_INJ_C_D_1107 + NC_HD_INJ_C_D_500 + NC_HD_INJ_C_D_602 + NC_HD_INJ_C_D_528 + NC_HD_INJ_C_D_659 + NC_HD_INJ_C_D_617 + NC_HD_INJ_C_D_998 + NC_HD_INJ_C_D_632 + NC_HD_INJ_C_D_610 + NC_HD_INJ_C_D_722 + NC_HD_INJ_C_D_394 + NC_HD_INJ_C_D_814 + NC_HD_INJ_C_D_350 + NC_HD_INJ_C_D_223 + NC_HD_INJ_C_D_577 + NC_HD_INJ_C_D_1007 + NC_HD_INJ_C_D_448 + NC_HD_INJ_C_D_186 + NC_HD_INJ_C_D_520 + NC_HD_INJ_C_D_853 + NC_HD_INJ_C_D_719 + NC_HD_INJ_C_D_1345 + NC_HD_INJ_C_D_568 + NC_HD_INJ_C_D_493 + NC_HD_INJ_C_D_498 + NC_HD_INJ_C_D_552 + NC_HD_INJ_C_D_607 + NC_HD_INJ_C_D_808 + NC_HD_INJ_C_D_388 + NC_HD_INJ_C_D_710 + NC_HD_INJ_C_D_908 + NC_HD_INJ_C_D_595 + NC_HD_INJ_C_D_417 + NC_HD_INJ_C_D_393 + NC_HD_INJ_C_D_503 + NC_HD_INJ_C_D_914 + NC_HD_INJ_C_D_952 + NC_HD_INJ_C_D_324 + NC_HD_INJ_C_D_618 + NC_HD_INJ_C_D_544 + NC_HD_INJ_C_D_895 + NC_HD_INJ_C_D_535 + NC_HD_INJ_C_D_1279 + NC_HD_INJ_C_D_820 + NC_HD_INJ_C_D_482 + NC_HD_INJ_C_D_1260 + NC_HD_INJ_C_D_798 + NC_HD_INJ_C_D_418 + NC_HD_INJ_C_D_408 + NC_HD_INJ_C_D_314 + NC_HD_INJ_C_D_452 + NC_HD_INJ_C_D_996 + NC_HD_INJ_C_D_554 + NC_HD_INJ_C_D_398 + NC_HD_INJ_C_D_1024 + NC_HD_INJ_C_D_551 + NC_HD_INJ_C_D_274 + NC_HD_INJ_C_D_457 + NC_HD_INJ_C_D_373 + NC_HD_INJ_C_D_588 + NC_HD_INJ_C_D_258 + NC_HD_INJ_C_D_863 + NC_HD_INJ_C_D_412 + NC_HD_INJ_C_D_571 + NC_HD_INJ_C_D_335 + NC_HD_INJ_C_D_306 + NC_HD_INJ_C_D_826 + NC_HD_INJ_C_D_197 + NC_HD_INJ_C_D_401 + NC_HD_INJ_C_D_267 + NC_HD_INJ_C_D_216 + NC_HD_INJ_C_D_372 + NC_HD_INJ_C_D_921 + NC_HD_INJ_C_D_664 + NC_HD_INJ_C_D_599 + NC_HD_INJ_C_D_371 + NC_HD_INJ_C_D_1214 + NC_HD_INJ_C_D_640 + NC_HD_INJ_C_D_1030 + NC_HD_INJ_C_D_885 + NC_HD_INJ_C_D_810 + NC_HD_INJ_C_D_565 + NC_HD_INJ_C_D_367 + NC_HD_INJ_C_D_654 + NC_HD_INJ_C_D_900 + NC_HD_INJ_C_D_559 + NC_HD_INJ_C_D_661 + NC_HD_INJ_C_D_246 + NC_HD_INJ_C_D_800 + NC_HD_INJ_C_D_400 + NC_HD_INJ_C_D_692 + NC_HD_INJ_C_D_272 + NC_HD_INJ_C_D_385 + NC_HD_INJ_C_D_333 + NC_HD_INJ_C_D_1036 + NC_HD_INJ_C_D_1138 + NC_HD_INJ_C_D_676 + NC_HD_INJ_C_D_411 + NC_HD_INJ_C_D_790 + NC_HD_INJ_C_D_570 + NC_HD_INJ_C_D_917 + NC_HD_INJ_C_D_471 + NC_HD_INJ_C_D_898 + NC_HD_INJ_C_D_185 + NC_HD_INJ_C_D_726 + NC_HD_INJ_C_D_467 + NC_HD_INJ_C_D_456 + NC_HD_INJ_C_D_406 + NC_HD_INJ_C_D_425 + NC_HD_INJ_C_D_943 + NC_HD_INJ_C_D_522 + NC_HD_INJ_C_D_526 + NC_HD_INJ_C_D_948 + NC_HD_INJ_C_D_867 + NC_HD_INJ_C_D_514 + NC_HD_INJ_C_D_1068 + NC_HD_INJ_C_D_566 + NC_HD_INJ_C_D_784 + NC_HD_INJ_C_D_1105 + NC_HD_INJ_C_D_744 + NC_HD_INJ_C_D_521 + NC_HD_INJ_C_D_882 + NC_HD_INJ_C_D_631 + NC_HD_INJ_C_D_390 + NC_HD_INJ_C_D_278 + NC_HD_INJ_C_D_1006 + NC_HD_INJ_C_D_750 + NC_HD_INJ_C_D_474 + NC_HD_INJ_C_D_1306 + NC_HD_INJ_C_D_443 + NC_HD_INJ_C_D_480 + NC_HD_INJ_C_D_479 + NC_HD_INJ_C_D_713 + NC_HD_INJ_C_D_626 + NC_HD_INJ_C_D_428 + NC_HD_INJ_C_D_715 + NC_HD_INJ_C_D_693 + NC_HD_INJ_C_D_725 + NC_HD_INJ_C_D_587 + NC_HD_INJ_C_D_473 + NC_HD_INJ_C_D_553 + NC_HD_INJ_C_D_600 + NC_HD_INJ_C_D_447 + NC_HD_INJ_C_D_550 + NC_HD_INJ_C_D_518 + NC_HD_INJ_C_D_497 + NC_HD_INJ_C_D_420 + NC_HD_INJ_C_D_646 + NC_HD_INJ_C_D_427 + NC_HD_INJ_C_D_868 + NC_HD_INJ_C_D_624 + NC_HD_INJ_C_D_329 + NC_HD_INJ_C_D_523 + NC_HD_INJ_C_D_920 + NC_HD_INJ_C_D_1267 + NC_HD_INJ_C_D_415 + NC_HD_INJ_C_D_359 + NC_HD_INJ_C_D_633 + NC_HD_INJ_C_D_897 + NC_HD_INJ_C_D_844 + NC_HD_INJ_C_D_834 + NC_HD_INJ_C_D_405 + NC_HD_INJ_C_D_619 + NC_HD_INJ_C_D_303 + NC_HD_INJ_C_D_281 + NC_HD_INJ_C_D_524 + NC_HD_INJ_C_D_336 + NC_HD_INJ_C_D_502 + NC_HD_INJ_C_D_279 + NC_HD_INJ_C_D_574 + NC_HD_INJ_C_D_424 + NC_HD_INJ_C_D_696 + NC_HD_INJ_C_D_357 + NC_HD_INJ_C_D_609 + NC_HD_INJ_C_D_928 + NC_HD_INJ_C_D_546 + NC_HD_INJ_C_D_637 + NC_HD_INJ_C_D_753 + NC_HD_INJ_C_D_378 + NC_HD_INJ_C_D_581 + NC_HD_INJ_C_D_403 + NC_HD_INJ_C_D_765 + NC_HD_INJ_C_D_176 + NC_HD_INJ_C_D_515 + NC_HD_INJ_C_D_1211 + NC_HD_INJ_C_D_462 + NC_HD_INJ_C_D_625 + NC_HD_INJ_C_D_212 + NC_HD_INJ_C_D_386 + NC_HD_INJ_C_D_823 + NC_HD_INJ_C_D_379 + NC_HD_INJ_C_D_1182 + NC_HD_INJ_C_D_276 + NC_HD_INJ_C_D_266 + NC_HD_INJ_C_D_675 + NC_HD_INJ_C_D_668 + NC_HD_INJ_C_D_360 + NC_HD_INJ_C_D_250 + NC_HD_INJ_C_D_622 + NC_HD_INJ_C_D_453 + NC_HD_INJ_C_D_486 + NC_HD_INJ_C_D_777 + NC_HD_INJ_C_D_598 + NC_HD_INJ_C_D_259 + NC_HD_INJ_C_D_340 + NC_HD_INJ_C_D_343 + NC_HD_INJ_C_D_268 + NC_HD_INJ_C_D_606 + NC_HD_INJ_C_D_1314 + NC_HD_INJ_C_D_825 + NC_HD_INJ_C_D_653 + NC_HD_INJ_C_D_249 + NC_HD_INJ_C_D_740 + NC_HD_INJ_C_D_681 + NC_HD_INJ_C_D_282 + NC_HD_INJ_C_D_698 + NC_HD_INJ_C_D_605 + NC_HD_INJ_C_D_1254 + NC_HD_INJ_C_D_368 + NC_HD_INJ_C_D_337 + NC_HD_INJ_C_D_295 + NC_HD_INJ_C_D_344 + NC_HD_INJ_C_D_245 + NC_HD_INJ_C_D_466 + NC_HD_INJ_C_D_542 + NC_HD_INJ_C_D_647 + NC_HD_INJ_C_D_687 + NC_HD_INJ_C_D_358 + NC_HD_INJ_C_D_1084 + NC_HD_INJ_C_D_501 + NC_HD_INJ_C_D_458 + NC_HD_INJ_C_D_290 + NC_HD_INJ_C_D_705 + NC_HD_INJ_C_D_392 + NC_HD_INJ_C_D_691 + NC_HD_INJ_C_D_339 + NC_HD_INJ_C_D_585 + NC_HD_INJ_C_D_445 + NC_HD_INJ_C_D_363 + NC_HD_INJ_C_D_383 + NC_HD_INJ_C_D_584 + NC_HD_INJ_C_D_356 + NC_HD_INJ_C_D_264 + NC_HD_INJ_C_D_511 + NC_HD_INJ_C_D_533 + NC_HD_INJ_C_D_475 + NC_HD_INJ_C_D_652 + NC_HD_INJ_C_D_788 + NC_HD_INJ_C_D_1088 + NC_HD_INJ_C_D_531 + NC_HD_INJ_C_D_484 + NC_HD_INJ_C_D_436 + NC_HD_INJ_C_D_762 + NC_HD_INJ_C_D_793 + NC_HD_INJ_C_D_409 + NC_HD_INJ_C_D_326 + NC_HD_INJ_C_D_538 + NC_HD_INJ_C_D_283 + NC_HD_INJ_C_D_975 + NC_HD_INJ_C_D_1026 + NC_HD_INJ_C_D_590 + NC_HD_INJ_C_D_656 + NC_HD_INJ_C_D_221 + NC_HD_INJ_C_D_255 + NC_HD_INJ_C_D_679 + NC_HD_INJ_C_D_487 + NC_HD_INJ_C_D_426 + NC_HD_INJ_C_D_512 + NC_HD_INJ_C_D_384 + NC_HD_INJ_C_D_558 + NC_HD_INJ_C_D_312 + NC_HD_INJ_C_D_727 + NC_HD_INJ_C_D_519 + NC_HD_INJ_C_D_463 + NC_HD_INJ_C_D_332 + NC_HD_INJ_C_D_433 + NC_HD_INJ_C_D_541 + NC_HD_INJ_C_D_860 + NC_HD_INJ_C_D_586 + NC_HD_INJ_C_D_377 + NC_HD_INJ_C_D_655 + NC_HD_INJ_C_D_1051 + NC_HD_INJ_C_D_338 + NC_HD_INJ_C_D_685 + NC_HD_INJ_C_D_545 + NC_HD_INJ_C_D_760 + NC_HD_INJ_C_D_932 + NC_HD_INJ_C_D_361 ')
# forms.append(' + NC_HD_INJ_C_P_660 + NC_HD_INJ_C_P_298 + NC_HD_INJ_C_P_582 + NC_HD_INJ_C_P_432 + NC_HD_INJ_C_P_375 + NC_HD_INJ_C_P_667 + NC_HD_INJ_C_P_446 + NC_HD_INJ_C_P_254 + NC_HD_INJ_C_P_843 + NC_HD_INJ_C_P_342 + NC_HD_INJ_C_P_305 + NC_HD_INJ_C_P_901 + NC_HD_INJ_C_P_1127 + NC_HD_INJ_C_P_277 + NC_HD_INJ_C_P_700 + NC_HD_INJ_C_P_936 + NC_HD_INJ_C_P_454 + NC_HD_INJ_C_P_899 + NC_HD_INJ_C_P_628 + NC_HD_INJ_C_P_508 + NC_HD_INJ_C_P_864 + NC_HD_INJ_C_P_455 + NC_HD_INJ_C_P_642 + NC_HD_INJ_C_P_720 + NC_HD_INJ_C_P_469 + NC_HD_INJ_C_P_870 + NC_HD_INJ_C_P_387 + NC_HD_INJ_C_P_450 + NC_HD_INJ_C_P_782 + NC_HD_INJ_C_P_641 + NC_HD_INJ_C_P_699 + NC_HD_INJ_C_P_567 + NC_HD_INJ_C_P_716 + NC_HD_INJ_C_P_325 + NC_HD_INJ_C_P_460 + NC_HD_INJ_C_P_835 + NC_HD_INJ_C_P_442 + NC_HD_INJ_C_P_557 + NC_HD_INJ_C_P_439 + NC_HD_INJ_C_P_464 + NC_HD_INJ_C_P_366 + NC_HD_INJ_C_P_672 + NC_HD_INJ_C_P_583 + NC_HD_INJ_C_P_620 + NC_HD_INJ_C_P_472 + NC_HD_INJ_C_P_771 + NC_HD_INJ_C_P_234 + NC_HD_INJ_C_P_288 + NC_HD_INJ_C_P_416 + NC_HD_INJ_C_P_702 + NC_HD_INJ_C_P_327 + NC_HD_INJ_C_P_884 + NC_HD_INJ_C_P_1062 + NC_HD_INJ_C_P_ND + NC_HD_INJ_C_P_555 + NC_HD_INJ_C_P_786 + NC_HD_INJ_C_P_2044 + NC_HD_INJ_C_P_510 + NC_HD_INJ_C_P_757 + NC_HD_INJ_C_P_430 + NC_HD_INJ_C_P_349 + NC_HD_INJ_C_P_488 + NC_HD_INJ_C_P_684 + NC_HD_INJ_C_P_494 + NC_HD_INJ_C_P_434 + NC_HD_INJ_C_P_812 + NC_HD_INJ_C_P_437 + NC_HD_INJ_C_P_799 + NC_HD_INJ_C_P_797 + NC_HD_INJ_C_P_525 + NC_HD_INJ_C_P_499 + NC_HD_INJ_C_P_578 + NC_HD_INJ_C_P_362 + NC_HD_INJ_C_P_431 + NC_HD_INJ_C_P_772 + NC_HD_INJ_C_P_573 + NC_HD_INJ_C_P_611 + NC_HD_INJ_C_P_293 + NC_HD_INJ_C_P_1297 + NC_HD_INJ_C_P_747 + NC_HD_INJ_C_P_517 + NC_HD_INJ_C_P_247 + NC_HD_INJ_C_P_1163 + NC_HD_INJ_C_P_560 + NC_HD_INJ_C_P_374 + NC_HD_INJ_C_P_492 + NC_HD_INJ_C_P_648 + NC_HD_INJ_C_P_770 + NC_HD_INJ_C_P_682 + NC_HD_INJ_C_P_1236 + NC_HD_INJ_C_P_299 + NC_HD_INJ_C_P_596 + NC_HD_INJ_C_P_924 + NC_HD_INJ_C_P_287  + NC_HD_INJ_C_P_370 + NC_HD_INJ_C_P_1144 + NC_HD_INJ_C_P_438 + NC_HD_INJ_C_P_643 + NC_HD_INJ_C_P_273 + NC_HD_INJ_C_P_592 + NC_HD_INJ_C_P_892 + NC_HD_INJ_C_P_423 + NC_HD_INJ_C_P_509 + NC_HD_INJ_C_P_414 + NC_HD_INJ_C_P_331 + NC_HD_INJ_C_P_449 + NC_HD_INJ_C_P_532 + NC_HD_INJ_C_P_549 + NC_HD_INJ_C_P_516 + NC_HD_INJ_C_P_301 + NC_HD_INJ_C_P_589 + NC_HD_INJ_C_P_677 + NC_HD_INJ_C_P_593 + NC_HD_INJ_C_P_562 + NC_HD_INJ_C_P_1101 + NC_HD_INJ_C_P_307 + NC_HD_INJ_C_P_540 + NC_HD_INJ_C_P_778 + NC_HD_INJ_C_P_735 + NC_HD_INJ_C_P_597 + NC_HD_INJ_C_P_265 + NC_HD_INJ_C_P_576 + NC_HD_INJ_C_P_239 + NC_HD_INJ_C_P_322 + NC_HD_INJ_C_P_490 + NC_HD_INJ_C_P_485 + NC_HD_INJ_C_P_296 + NC_HD_INJ_C_P_500 + NC_HD_INJ_C_P_496 + NC_HD_INJ_C_P_468 + NC_HD_INJ_C_P_364 + NC_HD_INJ_C_P_528 + NC_HD_INJ_C_P_617 + NC_HD_INJ_C_P_350 + NC_HD_INJ_C_P_397 + NC_HD_INJ_C_P_334 + NC_HD_INJ_C_P_577 + NC_HD_INJ_C_P_448 + NC_HD_INJ_C_P_186 + NC_HD_INJ_C_P_694 + NC_HD_INJ_C_P_202 + NC_HD_INJ_C_P_568 + NC_HD_INJ_C_P_493 + NC_HD_INJ_C_P_658 + NC_HD_INJ_C_P_178 + NC_HD_INJ_C_P_252 + NC_HD_INJ_C_P_413 + NC_HD_INJ_C_P_309 + NC_HD_INJ_C_P_552 + NC_HD_INJ_C_P_388 + NC_HD_INJ_C_P_710 + NC_HD_INJ_C_P_650 + NC_HD_INJ_C_P_881 + NC_HD_INJ_C_P_503 + NC_HD_INJ_C_P_181 + NC_HD_INJ_C_P_324 + NC_HD_INJ_C_P_248 + NC_HD_INJ_C_P_280 + NC_HD_INJ_C_P_618 + NC_HD_INJ_C_P_544 + NC_HD_INJ_C_P_995 + NC_HD_INJ_C_P_535 + NC_HD_INJ_C_P_703 + NC_HD_INJ_C_P_534 + NC_HD_INJ_C_P_477 + NC_HD_INJ_C_P_418 + NC_HD_INJ_C_P_452 + NC_HD_INJ_C_P_996 + NC_HD_INJ_C_P_554 + NC_HD_INJ_C_P_398 + NC_HD_INJ_C_P_262 + NC_HD_INJ_C_P_355 + NC_HD_INJ_C_P_686 + NC_HD_INJ_C_P_300 + NC_HD_INJ_C_P_457 + NC_HD_INJ_C_P_629 + NC_HD_INJ_C_P_258 + NC_HD_INJ_C_P_506 + NC_HD_INJ_C_P_412 + NC_HD_INJ_C_P_489 + NC_HD_INJ_C_P_601 + NC_HD_INJ_C_P_421 + NC_HD_INJ_C_P_604 + NC_HD_INJ_C_P_335 + NC_HD_INJ_C_P_399 + NC_HD_INJ_C_P_2017 + NC_HD_INJ_C_P_306 + NC_HD_INJ_C_P_495 + NC_HD_INJ_C_P_401 + NC_HD_INJ_C_P_931 + NC_HD_INJ_C_P_422 + NC_HD_INJ_C_P_372 + NC_HD_INJ_C_P_807 + NC_HD_INJ_C_P_1192 + NC_HD_INJ_C_P_664 + NC_HD_INJ_C_P_599 + NC_HD_INJ_C_P_612 + NC_HD_INJ_C_P_289 + NC_HD_INJ_C_P_1826 + NC_HD_INJ_C_P_404 + NC_HD_INJ_C_P_885 + NC_HD_INJ_C_P_367 + NC_HD_INJ_C_P_654 + NC_HD_INJ_C_P_900 + NC_HD_INJ_C_P_559 + NC_HD_INJ_C_P_661 + NC_HD_INJ_C_P_623 + NC_HD_INJ_C_P_294 + NC_HD_INJ_C_P_1258 + NC_HD_INJ_C_P_941 + NC_HD_INJ_C_P_935 + NC_HD_INJ_C_P_400 + NC_HD_INJ_C_P_886 + NC_HD_INJ_C_P_385 + NC_HD_INJ_C_P_380 + NC_HD_INJ_C_P_676 + NC_HD_INJ_C_P_411 + NC_HD_INJ_C_P_570 + NC_HD_INJ_C_P_670 + NC_HD_INJ_C_P_662 + NC_HD_INJ_C_P_726 + NC_HD_INJ_C_P_467 + NC_HD_INJ_C_P_406 + NC_HD_INJ_C_P_1139 + NC_HD_INJ_C_P_579 + NC_HD_INJ_C_P_425 + NC_HD_INJ_C_P_875 + NC_HD_INJ_C_P_522 + NC_HD_INJ_C_P_902 + NC_HD_INJ_C_P_743 + NC_HD_INJ_C_P_526 + NC_HD_INJ_C_P_218 + NC_HD_INJ_C_P_566 + NC_HD_INJ_C_P_829 + NC_HD_INJ_C_P_817 + NC_HD_INJ_C_P_841 + NC_HD_INJ_C_P_521 + NC_HD_INJ_C_P_631 + NC_HD_INJ_C_P_278 + NC_HD_INJ_C_P_474 + NC_HD_INJ_C_P_729 + NC_HD_INJ_C_P_443 + NC_HD_INJ_C_P_480 + NC_HD_INJ_C_P_479 + NC_HD_INJ_C_P_704 + NC_HD_INJ_C_P_354 + NC_HD_INJ_C_P_1063 + NC_HD_INJ_C_P_848 + NC_HD_INJ_C_P_626 + NC_HD_INJ_C_P_428 + NC_HD_INJ_C_P_614 + NC_HD_INJ_C_P_725 + NC_HD_INJ_C_P_587 + NC_HD_INJ_C_P_473 + NC_HD_INJ_C_P_553 + NC_HD_INJ_C_P_1763 + NC_HD_INJ_C_P_666 + NC_HD_INJ_C_P_302 + NC_HD_INJ_C_P_550 + NC_HD_INJ_C_P_1119 + NC_HD_INJ_C_P_518 + NC_HD_INJ_C_P_243 + NC_HD_INJ_C_P_497 + NC_HD_INJ_C_P_427 + NC_HD_INJ_C_P_711 + NC_HD_INJ_C_P_624 + NC_HD_INJ_C_P_329 + NC_HD_INJ_C_P_507 + NC_HD_INJ_C_P_547 + NC_HD_INJ_C_P_678 + NC_HD_INJ_C_P_1058 + NC_HD_INJ_C_P_523 + NC_HD_INJ_C_P_352 + NC_HD_INJ_C_P_981 + NC_HD_INJ_C_P_834 + NC_HD_INJ_C_P_405 + NC_HD_INJ_C_P_369 + NC_HD_INJ_C_P_1141 + NC_HD_INJ_C_P_281 + NC_HD_INJ_C_P_524 + NC_HD_INJ_C_P_336 + NC_HD_INJ_C_P_502 + NC_HD_INJ_C_P_748 + NC_HD_INJ_C_P_444 + NC_HD_INJ_C_P_574 + NC_HD_INJ_C_P_609 + NC_HD_INJ_C_P_357 + NC_HD_INJ_C_P_546 + NC_HD_INJ_C_P_637 + NC_HD_INJ_C_P_261 + NC_HD_INJ_C_P_378 + NC_HD_INJ_C_P_903 + NC_HD_INJ_C_P_581 + NC_HD_INJ_C_P_403 + NC_HD_INJ_C_P_717 + NC_HD_INJ_C_P_323 + NC_HD_INJ_C_P_765 + NC_HD_INJ_C_P_1133 + NC_HD_INJ_C_P_318 + NC_HD_INJ_C_P_515 + NC_HD_INJ_C_P_462 + NC_HD_INJ_C_P_214 + NC_HD_INJ_C_P_386 + NC_HD_INJ_C_P_379 + NC_HD_INJ_C_P_276 + NC_HD_INJ_C_P_675 + NC_HD_INJ_C_P_854 + NC_HD_INJ_C_P_1602 + NC_HD_INJ_C_P_622 + NC_HD_INJ_C_P_284 + NC_HD_INJ_C_P_486 + NC_HD_INJ_C_P_731 + NC_HD_INJ_C_P_777 + NC_HD_INJ_C_P_751 + NC_HD_INJ_C_P_621 + NC_HD_INJ_C_P_539 + NC_HD_INJ_C_P_395 + NC_HD_INJ_C_P_259 + NC_HD_INJ_C_P_985 + NC_HD_INJ_C_P_343 + NC_HD_INJ_C_P_391 + NC_HD_INJ_C_P_723 + NC_HD_INJ_C_P_657 + NC_HD_INJ_C_P_752 + NC_HD_INJ_C_P_1018 + NC_HD_INJ_C_P_1525 + NC_HD_INJ_C_P_270 + NC_HD_INJ_C_P_1952 + NC_HD_INJ_C_P_249 + NC_HD_INJ_C_P_429 + NC_HD_INJ_C_P_681 + NC_HD_INJ_C_P_440 + NC_HD_INJ_C_P_527 + NC_HD_INJ_C_P_282 + NC_HD_INJ_C_P_833 + NC_HD_INJ_C_P_698 + NC_HD_INJ_C_P_605 + NC_HD_INJ_C_P_773 + NC_HD_INJ_C_P_368 + NC_HD_INJ_C_P_337 + NC_HD_INJ_C_P_543 + NC_HD_INJ_C_P_295 + NC_HD_INJ_C_P_344 + NC_HD_INJ_C_P_466 + NC_HD_INJ_C_P_1052 + NC_HD_INJ_C_P_647 + NC_HD_INJ_C_P_1103 + NC_HD_INJ_C_P_852 + NC_HD_INJ_C_P_687 + NC_HD_INJ_C_P_501 + NC_HD_INJ_C_P_389 + NC_HD_INJ_C_P_768 + NC_HD_INJ_C_P_736 + NC_HD_INJ_C_P_749 + NC_HD_INJ_C_P_290 + NC_HD_INJ_C_P_392 + NC_HD_INJ_C_P_396 + NC_HD_INJ_C_P_339 + NC_HD_INJ_C_P_783 + NC_HD_INJ_C_P_739 + NC_HD_INJ_C_P_690 + NC_HD_INJ_C_P_585 + NC_HD_INJ_C_P_445 + NC_HD_INJ_C_P_363 + NC_HD_INJ_C_P_419 + NC_HD_INJ_C_P_365 + NC_HD_INJ_C_P_347 + NC_HD_INJ_C_P_584 + NC_HD_INJ_C_P_356 + NC_HD_INJ_C_P_348 + NC_HD_INJ_C_P_511 + NC_HD_INJ_C_P_533 + NC_HD_INJ_C_P_475 + NC_HD_INJ_C_P_788 + NC_HD_INJ_C_P_531 + NC_HD_INJ_C_P_484 + NC_HD_INJ_C_P_328 + NC_HD_INJ_C_P_436 + NC_HD_INJ_C_P_382 + NC_HD_INJ_C_P_906 + NC_HD_INJ_C_P_561 + NC_HD_INJ_C_P_1874 + NC_HD_INJ_C_P_409 + NC_HD_INJ_C_P_663 + NC_HD_INJ_C_P_956 + NC_HD_INJ_C_P_326 + NC_HD_INJ_C_P_1093 + NC_HD_INJ_C_P_538 + NC_HD_INJ_C_P_590 + NC_HD_INJ_C_P_878 + NC_HD_INJ_C_P_529 + NC_HD_INJ_C_P_679 + NC_HD_INJ_C_P_569 + NC_HD_INJ_C_P_487 + NC_HD_INJ_C_P_426 + NC_HD_INJ_C_P_512 + NC_HD_INJ_C_P_558 + NC_HD_INJ_C_P_384 + NC_HD_INJ_C_P_789 + NC_HD_INJ_C_P_332 + NC_HD_INJ_C_P_478 + NC_HD_INJ_C_P_613 + NC_HD_INJ_C_P_433 + NC_HD_INJ_C_P_541 + NC_HD_INJ_C_P_586 + NC_HD_INJ_C_P_655 + NC_HD_INJ_C_P_338 + NC_HD_INJ_C_P_685 + NC_HD_INJ_C_P_545  ')
# forms.append(' + NC_TireSize4_235_65R17 + NC_TireSize4_P195_60R15 + NC_TireSize4_P245_60R18 + NC_TireSize4_P255_50R19 + NC_TireSize4_225_60R18 + NC_TireSize4_P275_55R20 + NC_TireSize4_P245_75R16 + NC_TireSize4_P225_70R15 + NC_TireSize4_P235_60R16 + NC_TireSize4_P275_65R18 + NC_TireSize4_P205_70R15 + NC_TireSize4_P215_60R16 + NC_TireSize4_P215_65R16 + NC_TireSize4_P205_55R16 + NC_TireSize4_P245_50R20 + NC_TireSize4_205_55R16 + NC_TireSize4_P235_75R17 + NC_TireSize4_P235_70R15 + NC_TireSize4_P245_65R17 + NC_TireSize4_P235_50R18 + NC_TireSize4_P215_65R17 + NC_TireSize4_P225_55R17 + NC_TireSize4_P225_55R19 + NC_TireSize4_P235_55R17 + NC_TireSize4_P225_65R17 + NC_TireSize4_P225_60R16 + NC_TireSize4_P255_65R16 + NC_TireSize4_P235_65R17 + NC_TireSize4_P225_55R18 + NC_TireSize4_P245_60R20 + NC_TireSize4_225_45R17 + NC_TireSize4_P205_65R15 + NC_TireSize4_235_60R18 + NC_TireSize4_P245_70R19 + NC_TireSize4_205_50R17 + NC_TireSize4_P245_50R17 + NC_TireSize4_P265_75R16 + NC_TireSize4_P235_65R16 + NC_TireSize4_P235_65_R16 + NC_TireSize4_P265_70R16 + NC_TireSize4_P235_60R18 + NC_TireSize4_P245_55R19 + NC_TireSize4_P255_70R18 + NC_TireSize4_P235_50R17 + NC_TireSize4_P215_70R16 + NC_TireSize4_P265_60R16 + NC_TireSize4_P235_75R15 + NC_TireSize4_P265_50R20 + NC_TireSize4_P205_60R16  + NC_TireSize4_P235_45R18 + NC_TireSize4_P205_50R17 + NC_TireSize4_P215_55R16 + NC_TireSize4_P225_75R16 + NC_TireSize4_P215_75R16 + NC_TireSize4_P215_60R17 + NC_TireSize4_P225_70R16 + NC_TireSize4_P235_65R18 + NC_TireSize4_P265_70R18 + NC_TireSize4_225_50R17 + NC_TireSize4_P245_70R16 + NC_TireSize4_P235_45R17 + NC_TireSize4_P255_70R16 + NC_TireSize4_P255_55R17 + NC_TireSize4_P225_65R18 + NC_TireSize4_235_50R18 + NC_TireSize4_P235_55R18 + NC_TireSize4_P245_70R17 + NC_TireSize4_P205_55_R16 + NC_TireSize4_P255_60R17 + NC_TireSize4_P235_70R16 + NC_TireSize4_P205_60R15 + NC_TireSize4_P265_70R17 + NC_TireSize4_P205_75R15 + NC_TireSize4_P235_55R20 + NC_TireSize4_P255_55R18 + NC_TireSize4_P225_60R17 + NC_TireSize4_P225_75R15 + NC_TireSize4_P265_7016 + NC_TireSize4_P215_45R18 ')
# forms.append(' + NC_TractionControl_S + NC_TractionControl_A + NC_TractionControl_O    ')
# forms.append(' + PLNT_CD_7 + PLNT_CD_C + PLNT_CD_8 + PLNT_CD_B + PLNT_CD_L + PLNT_CD_W + PLNT_CD_N + PLNT_CD_H  + PLNT_CD_U + PLNT_CD_4 + PLNT_CD_G + PLNT_CD_K + PLNT_CD_V + PLNT_CD_5 + PLNT_CD_D + PLNT_CD_6 + PLNT_CD_Y + PLNT_CD_1 + PLNT_CD_S + PLNT_CD_P + PLNT_CD_Z + PLNT_CD_2 + PLNT_CD_F + PLNT_CD_9 + PLNT_CD_0 + PLNT_CD_X + PLNT_CD_J + PLNT_CD_A + PLNT_CD_T + PLNT_CD_M + PLNT_CD_3 + PLNT_CD_E ')
# forms.append(' + REAR_TIRE_SIZE_CD_16 + REAR_TIRE_SIZE_CD_47 + REAR_TIRE_SIZE_CD_50 + REAR_TIRE_SIZE_CD_27 + REAR_TIRE_SIZE_CD_53 + REAR_TIRE_SIZE_CD_73 + REAR_TIRE_SIZE_CD_37 + REAR_TIRE_SIZE_CD_96 + REAR_TIRE_SIZE_CD_45 + REAR_TIRE_SIZE_CD_66 + REAR_TIRE_SIZE_CD_30 + REAR_TIRE_SIZE_CD_34 + REAR_TIRE_SIZE_CD_39 + REAR_TIRE_SIZE_CD_74 + REAR_TIRE_SIZE_CD_59 + REAR_TIRE_SIZE_CD_87 + REAR_TIRE_SIZE_CD_32 + REAR_TIRE_SIZE_CD_15 + REAR_TIRE_SIZE_CD_465  + REAR_TIRE_SIZE_CD_46 + REAR_TIRE_SIZE_CD_41 + REAR_TIRE_SIZE_CD_49 + REAR_TIRE_SIZE_CD_86 + REAR_TIRE_SIZE_CD_64 + REAR_TIRE_SIZE_CD_48 + REAR_TIRE_SIZE_CD_467 + REAR_TIRE_SIZE_CD_79 + REAR_TIRE_SIZE_CD_25 + REAR_TIRE_SIZE_CD_97 + REAR_TIRE_SIZE_CD_57 + REAR_TIRE_SIZE_CD_38 + REAR_TIRE_SIZE_CD_442 + REAR_TIRE_SIZE_CD_63 + REAR_TIRE_SIZE_CD_77 + REAR_TIRE_SIZE_CD_75 + REAR_TIRE_SIZE_CD_92 + REAR_TIRE_SIZE_CD_61 + REAR_TIRE_SIZE_CD_43 + REAR_TIRE_SIZE_CD_60 + REAR_TIRE_SIZE_CD_58 + REAR_TIRE_SIZE_CD_83 + REAR_TIRE_SIZE_CD_84 + REAR_TIRE_SIZE_CD_14 + REAR_TIRE_SIZE_CD_62 + REAR_TIRE_SIZE_CD_55 + REAR_TIRE_SIZE_CD_67 + REAR_TIRE_SIZE_CD_36 + REAR_TIRE_SIZE_CD_82 + REAR_TIRE_SIZE_CD_44 + REAR_TIRE_SIZE_CD_98 + REAR_TIRE_SIZE_CD_466 + REAR_TIRE_SIZE_CD_85 + REAR_TIRE_SIZE_CD_13 + REAR_TIRE_SIZE_CD_94 + REAR_TIRE_SIZE_CD_76 + REAR_TIRE_SIZE_CD_35 + REAR_TIRE_SIZE_CD_29 + REAR_TIRE_SIZE_CD_56 + REAR_TIRE_SIZE_CD_88 + REAR_TIRE_SIZE_CD_91 + REAR_TIRE_SIZE_CD_40 + REAR_TIRE_SIZE_CD_461 + REAR_TIRE_SIZE_CD_93 + REAR_TIRE_SIZE_CD_71 + REAR_TIRE_SIZE_CD_54 + REAR_TIRE_SIZE_CD_42 + REAR_TIRE_SIZE_CD_65  ')
# forms.append(' + REAR_TIRE_SIZE_DESC_16R225 + REAR_TIRE_SIZE_DESC_17R225 + REAR_TIRE_SIZE_DESC_18R255 + REAR_TIRE_SIZE_DESC_18R235 + REAR_TIRE_SIZE_DESC_19R295 + REAR_TIRE_SIZE_DESC_19R265 + REAR_TIRE_SIZE_DESC_21R295 + REAR_TIRE_SIZE_DESC_16R215 + REAR_TIRE_SIZE_DESC_16R255 + REAR_TIRE_SIZE_DESC_16R185 + REAR_TIRE_SIZE_DESC_22R285 + REAR_TIRE_SIZE_DESC_16R175 + REAR_TIRE_SIZE_DESC_15R205 + REAR_TIRE_SIZE_DESC_17R275 + REAR_TIRE_SIZE_DESC_16R205 + REAR_TIRE_SIZE_DESC_18R245  + REAR_TIRE_SIZE_DESC_17R205 + REAR_TIRE_SIZE_DESC_14R165 + REAR_TIRE_SIZE_DESC_19R325 + REAR_TIRE_SIZE_DESC_21R265 + REAR_TIRE_SIZE_DESC_14R190 + REAR_TIRE_SIZE_DESC_19R255 + REAR_TIRE_SIZE_DESC_19R275 + REAR_TIRE_SIZE_DESC_20R275 + REAR_TIRE_SIZE_DESC_20R245 + REAR_TIRE_SIZE_DESC_20R235 + REAR_TIRE_SIZE_DESC_18R215 + REAR_TIRE_SIZE_DESC_15R185 + REAR_TIRE_SIZE_DESC_19R245 + REAR_TIRE_SIZE_DESC_15R155 + REAR_TIRE_SIZE_DESC_17R315 + REAR_TIRE_SIZE_DESC_17R245 + REAR_TIRE_SIZE_DESC_17R215 + REAR_TIRE_SIZE_DESC_20R295 + REAR_TIRE_SIZE_DESC_19R225 + REAR_TIRE_SIZE_DESC_18R285 + REAR_TIRE_SIZE_DESC_17R235 + REAR_TIRE_SIZE_DESC_20R305 + REAR_TIRE_SIZE_DESC_15R195 + REAR_TIRE_SIZE_DESC_17R255 + REAR_TIRE_SIZE_DESC_20R255 + REAR_TIRE_SIZE_DESC_14R185 + REAR_TIRE_SIZE_DESC_21R315 + REAR_TIRE_SIZE_DESC_16R245 + REAR_TIRE_SIZE_DESC_18R275 + REAR_TIRE_SIZE_DESC_21R285 + REAR_TIRE_SIZE_DESC_16R195 + REAR_TIRE_SIZE_DESC_18R225 + REAR_TIRE_SIZE_DESC_19R285 + REAR_TIRE_SIZE_DESC_17R265 + REAR_TIRE_SIZE_DESC_16R265 + REAR_TIRE_SIZE_DESC_20R335 + REAR_TIRE_SIZE_DESC_22R325 + REAR_TIRE_SIZE_DESC_UNKNOW + REAR_TIRE_SIZE_DESC_19R345 + REAR_TIRE_SIZE_DESC_15R255 + REAR_TIRE_SIZE_DESC_14R175 + REAR_TIRE_SIZE_DESC_15R175 + REAR_TIRE_SIZE_DESC_18R265 + REAR_TIRE_SIZE_DESC_19R305 + REAR_TIRE_SIZE_DESC_15R225 + REAR_TIRE_SIZE_DESC_19R235 + REAR_TIRE_SIZE_DESC_21R275 + REAR_TIRE_SIZE_DESC_22R265 + REAR_TIRE_SIZE_DESC_20R265 + REAR_TIRE_SIZE_DESC_20R315 + REAR_TIRE_SIZE_DESC_16R235 + REAR_TIRE_SIZE_DESC_20R285 ')
# forms.append(' + RSTRNT_TYP_CD_C + RSTRNT_TYP_CD_B + RSTRNT_TYP_CD_L + RSTRNT_TYP_CD_W + RSTRNT_TYP_CD_R + RSTRNT_TYP_CD_U + RSTRNT_TYP_CD_4 + RSTRNT_TYP_CD_K + RSTRNT_TYP_CD_G + RSTRNT_TYP_CD_V + RSTRNT_TYP_CD_D + RSTRNT_TYP_CD_I + RSTRNT_TYP_CD_Y + RSTRNT_TYP_CD_S + RSTRNT_TYP_CD_P + RSTRNT_TYP_CD_Z + RSTRNT_TYP_CD_F + RSTRNT_TYP_CD_X + RSTRNT_TYP_CD_A + RSTRNT_TYP_CD_J + RSTRNT_TYP_CD_T + RSTRNT_TYP_CD_M + RSTRNT_TYP_CD_3 + RSTRNT_TYP_CD_E ')
# forms.append(' + TLT_STRNG_WHL_OPT_CD_N + TLT_STRNG_WHL_OPT_CD_O + TLT_STRNG_WHL_OPT_CD_U  ')
# forms.append(' + TRK_TNG_RAT_CD_BC + TRK_TNG_RAT_CD_C + TRK_TNG_RAT_CD_B  + TRK_TNG_RAT_CD_A    ')
# forms.append(' + vinmasterPerformanceCd_3 + vinmasterPerformanceCd_1  + vinmasterPerformanceCd_2+ vinmasterPerformanceCd_4   ')
# forms.append(' + WHL_DRVN_CNT_Tr__+ WHL_DRVN_CNT_Tr_2 ')
# # Biggest categories   + BODY_STYLE_CD_UT +BODY_STYLE_DESC_SPORT_UTILITY_VEHICLE + bodyStyleCd_UTL4X44D + cko_eng_cylinders_Tr_6  + ENG_MFG_CD_U + MFG_DESC_GENERAL_MOTORS + FRNT_TYRE_SIZE_CD_U + FRNT_TYRE_SIZE_Desc_Unknow + MAK_NM_CHEVROLET  + NC_HD_INJ_C_D__ + NC_HD_INJ_C_P_. + NC_TireSize4__ + NC_TractionControl__  + PLNT_CD_R + REAR_TIRE_SIZE_CD_U + REAR_TIRE_SIZE_DESC_U + RSTRNT_TYP_CD_7  + TLT_STRNG_WHL_OPT_CD_S + TRK_TNG_RAT_CD_U + vinmasterPerformanceCd__ + WHL_DRVN_CNT_Tr_4 + ENG_MDL_CD_U + FRNT_TYRE_SIZE_Desc_13_15
# # 55
# forms.append(' + BODY_STYLE_CD_HR + BODY_STYLE_CD_CV + bodyStyleCd_grp1 + cko_eng_cylinders_2 + cko_eng_cylinders_8 + cko_eng_cylinders_grp1 + cko_eng_cylinders_Tr_4 + cko_eng_cylinders_Tr_8 + ENG_MDL_CD_grp1  + ENG_MDL_CD_grp2 + ENG_MDL_CD_080100 + ENG_MDL_CD_080006 + ENG_MDL_CD_070023 + ENG_MDL_CD_165051 + ENG_MDL_CD_080110 + ENG_MDL_CD_080115 + ENG_MFG_CD_grp1 + ENG_MFG_CD_070 + FRNT_TYRE_SIZE_CD_49 + FRNT_TYRE_SIZE_CD_62 + FRNT_TYRE_SIZE_CD_67 + FRNT_TYRE_SIZE_CD_19 + FRNT_TYRE_SIZE_CD_grp1 + FRNT_TYRE_SIZE_CD_grp2 + FRNT_TYRE_SIZE_CD_grp3 + MAK_NM_grp1 + MAK_NM_grp2 + MFG_DESC_grp1 + NC_TireSize4_grp1 + NC_TireSize4_grp2 + NC_TireSize4_grp3 + NC_TireSize4_grp4 + NC_TractionControl_S + PLNT_CD_X + NC_HD_INJ_C_D_grp1 + NC_HD_INJ_C_D_grp2 + NC_HD_INJ_C_D_grp3 + NC_HD_INJ_C_D_grp4 + RSTRNT_TYP_CD_Y + RSTRNT_TYP_CD_U + RSTRNT_TYP_CD_A + RSTRNT_TYP_CD_G + TLT_STRNG_WHL_OPT_CD_U + TRK_TNG_RAT_CD_C  + TRK_TNG_RAT_CD_BC    + vinmasterPerformanceCd_3 + vinmasterPerformanceCd_1 + WHL_DRVN_CNT_Tr__ ')
# forms.append(' + BODY_STYLE_CD_HR + BODY_STYLE_CD_CV + bodyStyleCd_grp1 + cko_eng_cylinders_2 + cko_eng_cylinders_8 + cko_eng_cylinders_grp1  + ENG_MDL_CD_grp1  + ENG_MDL_CD_grp2 + ENG_MDL_CD_080100 + ENG_MDL_CD_080006   + ENG_MFG_CD_grp1 + ENG_MFG_CD_070  + FRNT_TYRE_SIZE_CD_62  + FRNT_TYRE_SIZE_CD_grp1 + FRNT_TYRE_SIZE_CD_grp2 + FRNT_TYRE_SIZE_CD_grp3 + MAK_NM_grp1 + MAK_NM_grp2 + MFG_DESC_grp1 + NC_TireSize4_grp1 + NC_TireSize4_grp2 + NC_TireSize4_grp3 + NC_TireSize4_grp4 + NC_TractionControl_S  + NC_HD_INJ_C_D_grp1 + NC_HD_INJ_C_D_grp2 + NC_HD_INJ_C_D_grp3 + NC_HD_INJ_C_D_grp4 + RSTRNT_TYP_CD_Y  + RSTRNT_TYP_CD_A + RSTRNT_TYP_CD_G + TLT_STRNG_WHL_OPT_CD_U + TRK_TNG_RAT_CD_C  + TRK_TNG_RAT_CD_BC    + vinmasterPerformanceCd_3  + WHL_DRVN_CNT_Tr__ ')
# forms.append(' + BODY_STYLE_CD_HR + BODY_STYLE_CD_CV + bodyStyleCd_grp1 + cko_eng_cylinders_2 + cko_eng_cylinders_8 + cko_eng_cylinders_grp1   + ENG_MDL_CD_080100 + ENG_MDL_CD_080006    + ENG_MFG_CD_070  + FRNT_TYRE_SIZE_CD_62  + FRNT_TYRE_SIZE_CD_grp1 + FRNT_TYRE_SIZE_CD_grp2 + FRNT_TYRE_SIZE_CD_grp3  + MAK_NM_grp2 + MFG_DESC_grp1 + NC_TireSize4_grp1  + NC_TireSize4_grp3  + NC_TractionControl_S  + NC_HD_INJ_C_D_grp1 + NC_HD_INJ_C_D_grp2 + NC_HD_INJ_C_D_grp3 + NC_HD_INJ_C_D_grp4 + RSTRNT_TYP_CD_Y  + RSTRNT_TYP_CD_A  + TLT_STRNG_WHL_OPT_CD_U + TRK_TNG_RAT_CD_C  + TRK_TNG_RAT_CD_BC    + vinmasterPerformanceCd_3  + WHL_DRVN_CNT_Tr__ + ENG_MDL_CD_050016 + ENG_MDL_CD_050014 + ENG_MDL_CD_080015 + ENG_MDL_CD_238004 + ENG_MDL_CD_120020 + ENG_MDL_CD_207065 + ENG_MDL_CD_239003 + ENG_MDL_CD_207020 + ENG_MDL_CD_070024 + ENG_MDL_CD_070009 + ENG_MDL_CD_165051 + ENG_MDL_CD_080110 + ENG_MFG_CD_238 + ENG_MFG_CD_120 + ENG_MFG_CD_204 + ENG_MFG_CD_060 + MAK_NM_FISKER_AUTOMOTIVE + MAK_NM_MAYBACH + MAK_NM_MCLAREN_AUTOMOTIVE + MAK_NM_ROLLS_ROYCE + MAK_NM_DAIHATSU + MAK_NM_ALFA_ROMEO + MAK_NM_DATSUN + MAK_NM_LOTUS + NC_TireSize4_P225_55R18 + NC_TireSize4_P235_60R18 + NC_TireSize4_225_45R17 + NC_TireSize4_P235_50R18 ')
# forms.append(' + BODY_STYLE_CD_HR + BODY_STYLE_CD_CV + bodyStyleCd_grp1 + cko_eng_cylinders_2 + cko_eng_cylinders_8 + cko_eng_cylinders_grp1   + ENG_MDL_CD_080100 + ENG_MDL_CD_080006    + ENG_MFG_CD_070  + FRNT_TYRE_SIZE_CD_62  + FRNT_TYRE_SIZE_CD_grp1 + FRNT_TYRE_SIZE_CD_grp2 + FRNT_TYRE_SIZE_CD_grp3  + MAK_NM_grp2 + MFG_DESC_grp1 + NC_TireSize4_grp1  + NC_TireSize4_grp3  + NC_TractionControl_S  + NC_HD_INJ_C_D_grp1 + NC_HD_INJ_C_D_grp2 + NC_HD_INJ_C_D_grp3 + NC_HD_INJ_C_D_grp4 + RSTRNT_TYP_CD_Y  + RSTRNT_TYP_CD_A  + TLT_STRNG_WHL_OPT_CD_U + TRK_TNG_RAT_CD_C  + TRK_TNG_RAT_CD_BC    + vinmasterPerformanceCd_3  + WHL_DRVN_CNT_Tr__ + ENG_MDL_CD_050016 + ENG_MDL_CD_050014 + ENG_MDL_CD_080015 + ENG_MDL_CD_238004 + ENG_MDL_CD_120020 + ENG_MDL_CD_207065  + ENG_MDL_CD_207020 + ENG_MDL_CD_070024 + ENG_MDL_CD_070009 + ENG_MFG_CD_238 + ENG_MFG_CD_120  + ENG_MFG_CD_060  + MAK_NM_MAYBACH  + MAK_NM_ROLLS_ROYCE + MAK_NM_DAIHATSU + MAK_NM_ALFA_ROMEO + MAK_NM_DATSUN + MAK_NM_LOTUS  ')

# # Create custom variables for GLM
# data_train['WHL_BAS_LNGST_INCHS_lt95']=np.where((data_train['WHL_BAS_LNGST_INCHS']>=0) & (data_train['WHL_BAS_LNGST_INCHS']<95), 1, 0)
# data_train['WHL_BAS_SHRST_INCHS_lt95']=np.where((data_train['WHL_BAS_SHRST_INCHS']>=0) & (data_train['WHL_BAS_SHRST_INCHS']<95), 1, 0)
# data_train['door_cnt_bin_lt2']=np.where(data_train['door_cnt_bin']<2, 1, 0)
# data_train['cko_eng_cylinders_numeric_eq2']=np.where(data_train['cko_eng_cylinders_numeric']==2, 1, 0)
# data_test['WHL_BAS_LNGST_INCHS_lt95']=np.where((data_test['WHL_BAS_LNGST_INCHS']>=0) & (data_test['WHL_BAS_LNGST_INCHS']<95), 1, 0)
# data_test['WHL_BAS_SHRST_INCHS_lt95']=np.where((data_test['WHL_BAS_SHRST_INCHS']>=0) & (data_test['WHL_BAS_SHRST_INCHS']<95), 1, 0)
# data_test['door_cnt_bin_lt2']=np.where(data_test['door_cnt_bin']<2, 1, 0)
# data_test['cko_eng_cylinders_numeric_eq2']=np.where(data_test['cko_eng_cylinders_numeric']==2, 1, 0)
# data_train['length_VM_bw220and250']=np.where((data_train['length_VM']>=220) & (data_train['length_VM']<250), 1, 0)
# data_test['length_VM_bw220and250']=np.where((data_test['length_VM']>=220) & (data_test['length_VM']<250), 1, 0)
# data_train['cko_wheelbase_max_spline100']=np.minimum(data_train['cko_wheelbase_max'],100)
# data_test['cko_wheelbase_max_spline100']=np.minimum(data_test['cko_wheelbase_max'],100)
# data_train['EA_CURB_WEIGHT_35bw45']=np.where((3500<=data_train['EA_CURB_WEIGHT']) & (data_train['EA_CURB_WEIGHT']<4500),1,0)
# data_test['EA_CURB_WEIGHT_35bw45']=np.where((3500<=data_test['EA_CURB_WEIGHT']) & (data_test['EA_CURB_WEIGHT']<4500),1,0)
# data_train['EA_TIP_OVER_STABILITY_RATIO_ge15']=np.where(data_train['EA_TIP_OVER_STABILITY_RATIO']>=1.5,1,0)
# data_test['EA_TIP_OVER_STABILITY_RATIO_ge15']=np.where(data_test['EA_TIP_OVER_STABILITY_RATIO']>=1.5,1,0)
# data_train['ENG_DISPLCMNT_CI_380spline']=np.maximum(380,data_train['ENG_DISPLCMNT_CI'])
# data_test['ENG_DISPLCMNT_CI_380spline']=np.maximum(380,data_test['ENG_DISPLCMNT_CI']) 
# data_train['enginesize_ge6']=np.where(data_train['enginesize']>=6,1,0)
# data_test['enginesize_ge6']=np.where(data_test['enginesize']>=6,1,0)

# for i in range(58,len(forms)):
	
# 	CATFEATURES=CATARRAY[i]

# 	DUMMYFEATURES=[]
# 	for feature in CATFEATURES:
# 		# print("Printing data_train['FRNT_TYRE_SIZE_CD_1']")
# 		# print(data_train['FRNT_TYRE_SIZE_CD_1'])
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
# 	if i==0:
# 		data_train['antiTheftCd__']=data_train['antiTheftCd_.']	
# 		data_test['antiTheftCd__']=data_test['antiTheftCd_.']
# 	elif i==3:	
# 		data_train['cko_eng_cylinders__']=data_train['cko_eng_cylinders_.']	
# 		data_test['cko_eng_cylinders__']=data_test['cko_eng_cylinders_.']
# 		data_train['classCd__']=data_train['classCd_.']	
# 		data_test['classCd__']=data_test['classCd_.']
# 	elif i==4:	
# 		data_train['dayTimeLightCd__']=data_train['dayTimeLightCd_.']	
# 		data_test['dayTimeLightCd__']=data_test['dayTimeLightCd_.']
# 		data_train['EA_DRIVE_WHEELS__']=data_train['EA_DRIVE_WHEELS_.']	
# 		data_test['EA_DRIVE_WHEELS__']=data_test['EA_DRIVE_WHEELS_.']
# 		data_train['EA_DRIVE_WHEELS_4X2___4X4']=data_train['EA_DRIVE_WHEELS_4X2_/_4X4']	
# 		data_test['EA_DRIVE_WHEELS_4X2___4X4']=data_test['EA_DRIVE_WHEELS_4X2_/_4X4']
# 		data_train['ENG_CLNDR_RTR_CNT__']=data_train['ENG_CLNDR_RTR_CNT_.']	
# 		data_test['ENG_CLNDR_RTR_CNT__']=data_test['ENG_CLNDR_RTR_CNT_.']
# 	elif i==7:		
# 		data_train['ESCcd__']=data_train['ESCcd_.']	
# 		data_test['ESCcd__']=data_test['ESCcd_.']
# 		data_train['frameType__']=data_train['frameType_.']	
# 		data_test['frameType__']=data_test['frameType_.']
# 	elif i==12:		
# 		data_train['MFG_DESC_FISKER_AUTOMOTIVE_INC_']=data_train['MFG_DESC_FISKER_AUTOMOTIVE_INC.']	
# 		data_test['MFG_DESC_FISKER_AUTOMOTIVE_INC_']=data_test['MFG_DESC_FISKER_AUTOMOTIVE_INC.']
# 		data_train['MFG_DESC_TESLA_MOTORS_INC_']=data_train['MFG_DESC_TESLA_MOTORS_INC.']	
# 		data_test['MFG_DESC_TESLA_MOTORS_INC_']=data_test['MFG_DESC_TESLA_MOTORS_INC.']
# 	elif i==17:		
# 		data_train['NC_TireSize_P205_45R17']=data_train['NC_TireSize_P205/45R17']	
# 		data_test['NC_TireSize_P205_45R17']=data_test['NC_TireSize_P205/45R17']
# 		data_train['NC_TireSize_P245_70R16']=data_train['NC_TireSize_P245/70R16']	
# 		data_test['NC_TireSize_P245_70R16']=data_test['NC_TireSize_P245/70R16']
# 		data_train['NC_TireSize_P255_65R16']=data_train['NC_TireSize_P255/65R16']	
# 		data_test['NC_TireSize_P255_65R16']=data_test['NC_TireSize_P255/65R16']
# 		data_train['NC_TireSize4_225_45R17']=data_train['NC_TireSize4_225/45R17']	
# 		data_test['NC_TireSize4_225_45R17']=data_test['NC_TireSize4_225/45R17']
# 		data_train['NC_TireSize4_P215_60R16']=data_train['NC_TireSize4_P215/60R16']	
# 		data_test['NC_TireSize4_P215_60R16']=data_test['NC_TireSize4_P215/60R16']
# 		data_train['NC_TireSize4_P235_50R18']=data_train['NC_TireSize4_P235/50R18']	
# 		data_test['NC_TireSize4_P235_50R18']=data_test['NC_TireSize4_P235/50R18']
# 		data_train['NC_TireSize4_P255_65R16']=data_train['NC_TireSize4_P255/65R16']	
# 		data_test['NC_TireSize4_P255_65R16']=data_test['NC_TireSize4_P255/65R16']
# 	elif i==18:	
# 		data_train['NC_WheelsDriven_FWD_AWD']=data_train['NC_WheelsDriven_FWD/AWD']	
# 		data_test['NC_WheelsDriven_FWD_AWD']=data_test['NC_WheelsDriven_FWD/AWD']
# 		data_train['NC_WheelsDriven_RWD_4WD']=data_train['NC_WheelsDriven_RWD/4WD']	
# 		data_test['NC_WheelsDriven_RWD_4WD']=data_test['NC_WheelsDriven_RWD/4WD']
# 	elif i==25:	
# 		data_train['tonCd__']=data_train['tonCd_.']	
# 		data_test['tonCd__']=data_test['tonCd_.']
# 	elif i==26:
# 		data_train['TRANS_SPEED_CD__']=data_train['TRANS_SPEED_CD_.']	
# 		data_test['TRANS_SPEED_CD__']=data_test['TRANS_SPEED_CD_.']
# 	elif i==27:
# 		data_train['TRK_BRK_TYP_CD_H_V']=data_train['TRK_BRK_TYP_CD_H/V']	
# 		data_test['TRK_BRK_TYP_CD_H_V']=data_test['TRK_BRK_TYP_CD_H/V']
# 	elif i==28:
# 		data_train['TRK_CAB_CNFG_CD_Tr_0_U']=data_train['TRK_CAB_CNFG_CD_Tr_0.U']	
# 		data_test['TRK_CAB_CNFG_CD_Tr_0_U']=data_test['TRK_CAB_CNFG_CD_Tr_0.U']
# 		data_train['TRK_CAB_CNFG_CD_Tr_3_EXT']=data_train['TRK_CAB_CNFG_CD_Tr_3.EXT']	
# 		data_test['TRK_CAB_CNFG_CD_Tr_3_EXT']=data_test['TRK_CAB_CNFG_CD_Tr_3.EXT']
# 	elif i==29:
# 		data_train['WHL_DRVN_CNT_Tr__']=data_train['WHL_DRVN_CNT_Tr_.']	
# 		data_test['WHL_DRVN_CNT_Tr__']=data_test['WHL_DRVN_CNT_Tr_.']
# 	elif i==30:		
# 		data_train['cko_eng_cylinders__']=data_train['cko_eng_cylinders_.']	
# 		data_test['cko_eng_cylinders__']=data_test['cko_eng_cylinders_.']
# 		data_train['MFG_DESC_FISKER_AUTOMOTIVE_INC_']=data_train['MFG_DESC_FISKER_AUTOMOTIVE_INC.']	
# 		data_test['MFG_DESC_FISKER_AUTOMOTIVE_INC_']=data_test['MFG_DESC_FISKER_AUTOMOTIVE_INC.']
# 		data_train['NC_TireSize4_225_45R17']=data_train['NC_TireSize4_225/45R17']	
# 		data_test['NC_TireSize4_225_45R17']=data_test['NC_TireSize4_225/45R17']
# 		data_train['NC_TireSize4_P215_60R16']=data_train['NC_TireSize4_P215/60R16']	
# 		data_test['NC_TireSize4_P215_60R16']=data_test['NC_TireSize4_P215/60R16']
# 		data_train['NC_TireSize4_P235_50R18']=data_train['NC_TireSize4_P235/50R18']	
# 		data_test['NC_TireSize4_P235_50R18']=data_test['NC_TireSize4_P235/50R18']		
# 		data_train['WHL_DRVN_CNT_Tr__']=data_train['WHL_DRVN_CNT_Tr_.']	
# 		data_test['WHL_DRVN_CNT_Tr__']=data_test['WHL_DRVN_CNT_Tr_.']	 
# 	elif i==31:		
# 		data_train['cko_eng_cylinders__']=data_train['cko_eng_cylinders_.']	
# 		data_test['cko_eng_cylinders__']=data_test['cko_eng_cylinders_.']
# 		data_train['MFG_DESC_FISKER_AUTOMOTIVE_INC_']=data_train['MFG_DESC_FISKER_AUTOMOTIVE_INC.']	
# 		data_test['MFG_DESC_FISKER_AUTOMOTIVE_INC_']=data_test['MFG_DESC_FISKER_AUTOMOTIVE_INC.']
# 		data_train['NC_TireSize4_225_45R17']=data_train['NC_TireSize4_225/45R17']	
# 		data_test['NC_TireSize4_225_45R17']=data_test['NC_TireSize4_225/45R17']
# 		data_train['NC_TireSize4_P215_60R16']=data_train['NC_TireSize4_P215/60R16']	
# 		data_test['NC_TireSize4_P215_60R16']=data_test['NC_TireSize4_P215/60R16']
# 		data_train['NC_TireSize4_P235_50R18']=data_train['NC_TireSize4_P235/50R18']	
# 		data_test['NC_TireSize4_P235_50R18']=data_test['NC_TireSize4_P235/50R18']		
# 		data_train['WHL_DRVN_CNT_Tr__']=data_train['WHL_DRVN_CNT_Tr_.']	
# 		data_test['WHL_DRVN_CNT_Tr__']=data_test['WHL_DRVN_CNT_Tr_.']	
# 	elif i==34:	
# 		data_train['bodyStyleCd_CP_RDSTR']=data_train['bodyStyleCd_CP/RDSTR']
# 		data_test['bodyStyleCd_CP_RDSTR']=data_test['bodyStyleCd_CP/RDSTR']
# 		data_train['bodyStyleCd__']=data_train['bodyStyleCd_.']
# 		data_test['bodyStyleCd__']=data_test['bodyStyleCd_.']
# 		data_train['bodyStyleCd_HCHBK2_2']=data_train['bodyStyleCd_HCHBK2+2']
# 		data_train['bodyStyleCd_CPE_2_2']=data_train['bodyStyleCd_CPE_2+2']
# 		data_test['bodyStyleCd_HCHBK2_2']=data_test['bodyStyleCd_HCHBK2+2']
# 		data_test['bodyStyleCd_CPE_2_2']=data_test['bodyStyleCd_CPE_2+2']	
# 	elif i==35:		
# 		data_train['cko_eng_cylinders__']=data_train['cko_eng_cylinders_.']	
# 		data_test['cko_eng_cylinders__']=data_test['cko_eng_cylinders_.']		 
# 	elif i==40:
# 		data_train['FRNT_TYRE_SIZE_Desc_13_615']=data_train['FRNT_TYRE_SIZE_Desc_13-615']
# 		data_test['FRNT_TYRE_SIZE_Desc_13_615']=data_test['FRNT_TYRE_SIZE_Desc_13-615']
# 	elif i==41:
# 		data_train['MAK_NM_MERCEDES_BENZ']=data_train['MAK_NM_MERCEDES-BENZ']
# 		data_train['MAK_NM_ROLLS_ROYCE']=data_train['MAK_NM_ROLLS-ROYCE']
# 		data_test['MAK_NM_MERCEDES_BENZ']=data_test['MAK_NM_MERCEDES-BENZ']
# 		data_test['MAK_NM_ROLLS_ROYCE']=data_test['MAK_NM_ROLLS-ROYCE']
# 	elif i==42:
# 		data_train['MFG_DESC_TESLA_MOTORS_INC_']=data_train['MFG_DESC_TESLA_MOTORS_INC.']
# 		data_train['MFG_DESC_DAIMLER_CHRYSLER']=data_train['MFG_DESC_DAIMLER-CHRYSLER']
# 		data_train['MFG_DESC_MERCEDES_BENZ_USA_LLC']=data_train['MFG_DESC_MERCEDES-BENZ_USA_LLC']
# 		data_train['MFG_DESC_FISKER_AUTOMOTIVE_INC_']=data_train['MFG_DESC_FISKER_AUTOMOTIVE_INC.']
# 		data_test['MFG_DESC_TESLA_MOTORS_INC_']=data_test['MFG_DESC_TESLA_MOTORS_INC.']
# 		data_test['MFG_DESC_DAIMLER_CHRYSLER']=data_test['MFG_DESC_DAIMLER-CHRYSLER']
# 		data_test['MFG_DESC_MERCEDES_BENZ_USA_LLC']=data_test['MFG_DESC_MERCEDES-BENZ_USA_LLC']
# 		data_test['MFG_DESC_FISKER_AUTOMOTIVE_INC_']=data_test['MFG_DESC_FISKER_AUTOMOTIVE_INC.']
# 	elif i==45:
# 		data_train['NC_TireSize4_235_65R17']=data_train['NC_TireSize4_235/65R17']
# 		data_train['NC_TireSize4_P195_60R15']=data_train['NC_TireSize4_P195/60R15']
# 		data_train['NC_TireSize4_P245_60R18']=data_train['NC_TireSize4_P245/60R18']
# 		data_train['NC_TireSize4_P255_50R19']=data_train['NC_TireSize4_P255/50R19']
# 		data_train['NC_TireSize4_225_60R18']=data_train['NC_TireSize4_225/60R18']
# 		data_train['NC_TireSize4_P275_55R20']=data_train['NC_TireSize4_P275/55R20']
# 		data_train['NC_TireSize4_P245_75R16']=data_train['NC_TireSize4_P245/75R16']
# 		data_train['NC_TireSize4_P225_70R15']=data_train['NC_TireSize4_P225/70R15']
# 		data_train['NC_TireSize4_P235_60R16']=data_train['NC_TireSize4_P235/60R16']
# 		data_train['NC_TireSize4_P275_65R18']=data_train['NC_TireSize4_P275/65R18']
# 		data_train['NC_TireSize4_P205_70R15']=data_train['NC_TireSize4_P205/70R15']
# 		data_train['NC_TireSize4_P215_60R16']=data_train['NC_TireSize4_P215/60R16']
# 		data_train['NC_TireSize4_P215_65R16']=data_train['NC_TireSize4_P215/65R16']
# 		data_train['NC_TireSize4_P205_55R16']=data_train['NC_TireSize4_P205/55R16']
# 		data_train['NC_TireSize4_P245_50R20']=data_train['NC_TireSize4_P245/50R20']
# 		data_train['NC_TireSize4_205_55R16']=data_train['NC_TireSize4_205/55R16']
# 		data_train['NC_TireSize4_P235_75R17']=data_train['NC_TireSize4_P235/75R17']
# 		data_train['NC_TireSize4_P235_70R15']=data_train['NC_TireSize4_P235/70R15']
# 		data_train['NC_TireSize4_P245_65R17']=data_train['NC_TireSize4_P245/65R17']
# 		data_train['NC_TireSize4_P235_50R18']=data_train['NC_TireSize4_P235/50R18']
# 		data_train['NC_TireSize4_P215_65R17']=data_train['NC_TireSize4_P215/65R17']
# 		data_train['NC_TireSize4_P225_55R17']=data_train['NC_TireSize4_P225/55R17']
# 		data_train['NC_TireSize4_P225_55R19']=data_train['NC_TireSize4_P225/55R19']
# 		data_train['NC_TireSize4_P235_55R17']=data_train['NC_TireSize4_P235/55R17']
# 		data_train['NC_TireSize4_P225_65R17']=data_train['NC_TireSize4_P225/65R17']
# 		data_train['NC_TireSize4_P225_60R16']=data_train['NC_TireSize4_P225/60R16']
# 		data_train['NC_TireSize4_P255_65R16']=data_train['NC_TireSize4_P255/65R16']
# 		data_train['NC_TireSize4_P235_65R17']=data_train['NC_TireSize4_P235/65R17']
# 		data_train['NC_TireSize4_P225_55R18']=data_train['NC_TireSize4_P225/55R18']
# 		data_train['NC_TireSize4_P245_60R20']=data_train['NC_TireSize4_P245/60R20']
# 		data_train['NC_TireSize4_225_45R17']=data_train['NC_TireSize4_225/45R17']
# 		data_train['NC_TireSize4_P205_65R15']=data_train['NC_TireSize4_P205/65R15']
# 		data_train['NC_TireSize4_235_60R18']=data_train['NC_TireSize4_235/60R18']
# 		data_train['NC_TireSize4_P245_70R19']=data_train['NC_TireSize4_P245/70R19']
# 		data_train['NC_TireSize4_205_50R17']=data_train['NC_TireSize4_205/50R17']
# 		data_train['NC_TireSize4_P245_50R17']=data_train['NC_TireSize4_P245/50R17']
# 		data_train['NC_TireSize4_P265_75R16']=data_train['NC_TireSize4_P265/75R16']
# 		data_train['NC_TireSize4_P235_65R16']=data_train['NC_TireSize4_P235/65R16']
# 		data_train['NC_TireSize4_P235_65_R16']=data_train['NC_TireSize4_P235/65/R16']
# 		data_train['NC_TireSize4_P265_70R16']=data_train['NC_TireSize4_P265/70R16']
# 		data_train['NC_TireSize4_P235_60R18']=data_train['NC_TireSize4_P235/60R18']
# 		data_train['NC_TireSize4_P245_55R19']=data_train['NC_TireSize4_P245/55R19']
# 		data_train['NC_TireSize4_P255_70R18']=data_train['NC_TireSize4_P255/70R18']
# 		data_train['NC_TireSize4_P235_50R17']=data_train['NC_TireSize4_P235/50R17']
# 		data_train['NC_TireSize4_P215_70R16']=data_train['NC_TireSize4_P215/70R16']
# 		data_train['NC_TireSize4_P265_60R16']=data_train['NC_TireSize4_P265/60R16']
# 		data_train['NC_TireSize4_P235_75R15']=data_train['NC_TireSize4_P235/75R15']
# 		data_train['NC_TireSize4_P265_50R20']=data_train['NC_TireSize4_P265/50R20']
# 		data_train['NC_TireSize4_P205_60R16']=data_train['NC_TireSize4_P205/60R16']
# 		# data_train['NC_TireSize4__']=data_train['NC_TireSize4_.']
# 		data_train['NC_TireSize4_P235_45R18']=data_train['NC_TireSize4_P235/45R18']
# 		data_train['NC_TireSize4_P205_50R17']=data_train['NC_TireSize4_P205/50R17']
# 		data_train['NC_TireSize4_P215_55R16']=data_train['NC_TireSize4_P215/55R16']
# 		data_train['NC_TireSize4_P225_75R16']=data_train['NC_TireSize4_P225/75R16']
# 		data_train['NC_TireSize4_P215_75R16']=data_train['NC_TireSize4_P215/75R16']
# 		data_train['NC_TireSize4_P215_60R17']=data_train['NC_TireSize4_P215/60R17']
# 		data_train['NC_TireSize4_P225_70R16']=data_train['NC_TireSize4_P225/70R16']
# 		data_train['NC_TireSize4_P235_65R18']=data_train['NC_TireSize4_P235/65R18']
# 		data_train['NC_TireSize4_P265_70R18']=data_train['NC_TireSize4_P265/70R18']
# 		data_train['NC_TireSize4_225_50R17']=data_train['NC_TireSize4_225/50R17']
# 		data_train['NC_TireSize4_P245_70R16']=data_train['NC_TireSize4_P245/70R16']
# 		data_train['NC_TireSize4_P235_45R17']=data_train['NC_TireSize4_P235/45R17']
# 		data_train['NC_TireSize4_P255_70R16']=data_train['NC_TireSize4_P255/70R16']
# 		data_train['NC_TireSize4_P255_55R17']=data_train['NC_TireSize4_P255/55R17']
# 		data_train['NC_TireSize4_P225_65R18']=data_train['NC_TireSize4_P225/65R18']
# 		data_train['NC_TireSize4_235_50R18']=data_train['NC_TireSize4_235/50R18']
# 		data_train['NC_TireSize4_P235_55R18']=data_train['NC_TireSize4_P235/55R18']
# 		data_train['NC_TireSize4_P245_70R17']=data_train['NC_TireSize4_P245/70R17']
# 		data_train['NC_TireSize4_P205_55_R16']=data_train['NC_TireSize4_P205/55/R16']
# 		data_train['NC_TireSize4_P255_60R17']=data_train['NC_TireSize4_P255/60R17']
# 		data_train['NC_TireSize4_P235_70R16']=data_train['NC_TireSize4_P235/70R16']
# 		data_train['NC_TireSize4_P205_60R15']=data_train['NC_TireSize4_P205/60R15']
# 		data_train['NC_TireSize4_P265_70R17']=data_train['NC_TireSize4_P265/70R17']
# 		data_train['NC_TireSize4_P205_75R15']=data_train['NC_TireSize4_P205/75R15']
# 		data_train['NC_TireSize4_P235_55R20']=data_train['NC_TireSize4_P235/55R20']
# 		data_train['NC_TireSize4_P255_55R18']=data_train['NC_TireSize4_P255/55R18']
# 		data_train['NC_TireSize4_P225_60R17']=data_train['NC_TireSize4_P225/60R17']
# 		data_train['NC_TireSize4_P225_75R15']=data_train['NC_TireSize4_P225/75R15']
# 		data_train['NC_TireSize4_P265_7016']=data_train['NC_TireSize4_P265/7016']
# 		data_train['NC_TireSize4_P215_45R18']=data_train['NC_TireSize4_P215/45R18']
# 		data_test['NC_TireSize4_235_65R17']=data_test['NC_TireSize4_235/65R17']
# 		data_test['NC_TireSize4_P195_60R15']=data_test['NC_TireSize4_P195/60R15']
# 		data_test['NC_TireSize4_P245_60R18']=data_test['NC_TireSize4_P245/60R18']
# 		data_test['NC_TireSize4_P255_50R19']=data_test['NC_TireSize4_P255/50R19']
# 		data_test['NC_TireSize4_225_60R18']=data_test['NC_TireSize4_225/60R18']
# 		data_test['NC_TireSize4_P275_55R20']=data_test['NC_TireSize4_P275/55R20']
# 		data_test['NC_TireSize4_P245_75R16']=data_test['NC_TireSize4_P245/75R16']
# 		data_test['NC_TireSize4_P225_70R15']=data_test['NC_TireSize4_P225/70R15']
# 		data_test['NC_TireSize4_P235_60R16']=data_test['NC_TireSize4_P235/60R16']
# 		data_test['NC_TireSize4_P275_65R18']=data_test['NC_TireSize4_P275/65R18']
# 		data_test['NC_TireSize4_P205_70R15']=data_test['NC_TireSize4_P205/70R15']
# 		data_test['NC_TireSize4_P215_60R16']=data_test['NC_TireSize4_P215/60R16']
# 		data_test['NC_TireSize4_P215_65R16']=data_test['NC_TireSize4_P215/65R16']
# 		data_test['NC_TireSize4_P205_55R16']=data_test['NC_TireSize4_P205/55R16']
# 		data_test['NC_TireSize4_P245_50R20']=data_test['NC_TireSize4_P245/50R20']
# 		data_test['NC_TireSize4_205_55R16']=data_test['NC_TireSize4_205/55R16']
# 		data_test['NC_TireSize4_P235_75R17']=data_test['NC_TireSize4_P235/75R17']
# 		data_test['NC_TireSize4_P235_70R15']=data_test['NC_TireSize4_P235/70R15']
# 		data_test['NC_TireSize4_P245_65R17']=data_test['NC_TireSize4_P245/65R17']
# 		data_test['NC_TireSize4_P235_50R18']=data_test['NC_TireSize4_P235/50R18']
# 		data_test['NC_TireSize4_P215_65R17']=data_test['NC_TireSize4_P215/65R17']
# 		data_test['NC_TireSize4_P225_55R17']=data_test['NC_TireSize4_P225/55R17']
# 		data_test['NC_TireSize4_P225_55R19']=data_test['NC_TireSize4_P225/55R19']
# 		data_test['NC_TireSize4_P235_55R17']=data_test['NC_TireSize4_P235/55R17']
# 		data_test['NC_TireSize4_P225_65R17']=data_test['NC_TireSize4_P225/65R17']
# 		data_test['NC_TireSize4_P225_60R16']=data_test['NC_TireSize4_P225/60R16']
# 		data_test['NC_TireSize4_P255_65R16']=data_test['NC_TireSize4_P255/65R16']
# 		data_test['NC_TireSize4_P235_65R17']=data_test['NC_TireSize4_P235/65R17']
# 		data_test['NC_TireSize4_P225_55R18']=data_test['NC_TireSize4_P225/55R18']
# 		data_test['NC_TireSize4_P245_60R20']=data_test['NC_TireSize4_P245/60R20']
# 		data_test['NC_TireSize4_225_45R17']=data_test['NC_TireSize4_225/45R17']
# 		data_test['NC_TireSize4_P205_65R15']=data_test['NC_TireSize4_P205/65R15']
# 		data_test['NC_TireSize4_235_60R18']=data_test['NC_TireSize4_235/60R18']
# 		data_test['NC_TireSize4_P245_70R19']=data_test['NC_TireSize4_P245/70R19']
# 		data_test['NC_TireSize4_205_50R17']=data_test['NC_TireSize4_205/50R17']
# 		data_test['NC_TireSize4_P245_50R17']=data_test['NC_TireSize4_P245/50R17']
# 		data_test['NC_TireSize4_P265_75R16']=data_test['NC_TireSize4_P265/75R16']
# 		data_test['NC_TireSize4_P235_65R16']=data_test['NC_TireSize4_P235/65R16']
# 		data_test['NC_TireSize4_P235_65_R16']=data_test['NC_TireSize4_P235/65/R16']
# 		data_test['NC_TireSize4_P265_70R16']=data_test['NC_TireSize4_P265/70R16']
# 		data_test['NC_TireSize4_P235_60R18']=data_test['NC_TireSize4_P235/60R18']
# 		data_test['NC_TireSize4_P245_55R19']=data_test['NC_TireSize4_P245/55R19']
# 		data_test['NC_TireSize4_P255_70R18']=data_test['NC_TireSize4_P255/70R18']
# 		data_test['NC_TireSize4_P235_50R17']=data_test['NC_TireSize4_P235/50R17']
# 		data_test['NC_TireSize4_P215_70R16']=data_test['NC_TireSize4_P215/70R16']
# 		data_test['NC_TireSize4_P265_60R16']=data_test['NC_TireSize4_P265/60R16']
# 		data_test['NC_TireSize4_P235_75R15']=data_test['NC_TireSize4_P235/75R15']
# 		data_test['NC_TireSize4_P265_50R20']=data_test['NC_TireSize4_P265/50R20']
# 		data_test['NC_TireSize4_P205_60R16']=data_test['NC_TireSize4_P205/60R16']
# 		# data_test['NC_TireSize4__']=data_test['NC_TireSize4_.']
# 		data_test['NC_TireSize4_P235_45R18']=data_test['NC_TireSize4_P235/45R18']
# 		data_test['NC_TireSize4_P205_50R17']=data_test['NC_TireSize4_P205/50R17']
# 		data_test['NC_TireSize4_P215_55R16']=data_test['NC_TireSize4_P215/55R16']
# 		data_test['NC_TireSize4_P225_75R16']=data_test['NC_TireSize4_P225/75R16']
# 		data_test['NC_TireSize4_P215_75R16']=data_test['NC_TireSize4_P215/75R16']
# 		data_test['NC_TireSize4_P215_60R17']=data_test['NC_TireSize4_P215/60R17']
# 		data_test['NC_TireSize4_P225_70R16']=data_test['NC_TireSize4_P225/70R16']
# 		data_test['NC_TireSize4_P235_65R18']=data_test['NC_TireSize4_P235/65R18']
# 		data_test['NC_TireSize4_P265_70R18']=data_test['NC_TireSize4_P265/70R18']
# 		data_test['NC_TireSize4_225_50R17']=data_test['NC_TireSize4_225/50R17']
# 		data_test['NC_TireSize4_P245_70R16']=data_test['NC_TireSize4_P245/70R16']
# 		data_test['NC_TireSize4_P235_45R17']=data_test['NC_TireSize4_P235/45R17']
# 		data_test['NC_TireSize4_P255_70R16']=data_test['NC_TireSize4_P255/70R16']
# 		data_test['NC_TireSize4_P255_55R17']=data_test['NC_TireSize4_P255/55R17']
# 		data_test['NC_TireSize4_P225_65R18']=data_test['NC_TireSize4_P225/65R18']
# 		data_test['NC_TireSize4_235_50R18']=data_test['NC_TireSize4_235/50R18']
# 		data_test['NC_TireSize4_P235_55R18']=data_test['NC_TireSize4_P235/55R18']
# 		data_test['NC_TireSize4_P245_70R17']=data_test['NC_TireSize4_P245/70R17']
# 		data_test['NC_TireSize4_P205_55_R16']=data_test['NC_TireSize4_P205/55/R16']
# 		data_test['NC_TireSize4_P255_60R17']=data_test['NC_TireSize4_P255/60R17']
# 		data_test['NC_TireSize4_P235_70R16']=data_test['NC_TireSize4_P235/70R16']
# 		data_test['NC_TireSize4_P205_60R15']=data_test['NC_TireSize4_P205/60R15']
# 		data_test['NC_TireSize4_P265_70R17']=data_test['NC_TireSize4_P265/70R17']
# 		data_test['NC_TireSize4_P205_75R15']=data_test['NC_TireSize4_P205/75R15']
# 		data_test['NC_TireSize4_P235_55R20']=data_test['NC_TireSize4_P235/55R20']
# 		data_test['NC_TireSize4_P255_55R18']=data_test['NC_TireSize4_P255/55R18']
# 		data_test['NC_TireSize4_P225_60R17']=data_test['NC_TireSize4_P225/60R17']
# 		data_test['NC_TireSize4_P225_75R15']=data_test['NC_TireSize4_P225/75R15']
# 		data_test['NC_TireSize4_P265_7016']=data_test['NC_TireSize4_P265/7016']
# 		data_test['NC_TireSize4_P215_45R18']=data_test['NC_TireSize4_P215/45R18']
# 	elif i==55:
# 		data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_CPE2D4X4']+data_train['bodyStyleCd_CAMPER']+data_train['bodyStyleCd_TRK_4X2']+data_train['bodyStyleCd_CPE4X44D']+data_train['bodyStyleCd_CNV4X42D']+data_train['bodyStyleCd_WAG5D4X4']+data_train['bodyStyleCd_WAGON_2D']+data_train['bodyStyleCd_WAG5D4X2']+data_train['bodyStyleCd_WAG4D4X2']+data_train['bodyStyleCd_VAN']+data_train['bodyStyleCd_SP_HCHBK']+data_train['bodyStyleCd_SED4D4X4']+data_train['bodyStyleCd_SED2D4X4']+data_train['bodyStyleCd_SED4X44D']+data_train['bodyStyleCd_CPE_2+2']+data_train['bodyStyleCd_CP/RDSTR']+data_train['bodyStyleCd_LIMO']+data_train['bodyStyleCd_VAN_4X4']+data_train['bodyStyleCd_ROADSTER']+data_train['bodyStyleCd_MPV_4X2']+data_train['bodyStyleCd_SPORT_CP']+data_train['bodyStyleCd_TRK_4X4']+data_train['bodyStyleCd_HRDTP_2D']+data_train['bodyStyleCd_HCH2D4X4']+data_train['bodyStyleCd_HCH5D4X4']+data_train['bodyStyleCd_CABRIOLE']+data_train['bodyStyleCd_SP_COUPE']+data_train['bodyStyleCd_TARGA']+data_train['bodyStyleCd_COUPE']+data_train['bodyStyleCd_HCH3D4X4'])
# 		data_train['cko_eng_cylinders_grp1']=(data_train['cko_eng_cylinders_4']+data_train['cko_eng_cylinders_5']+data_train['cko_eng_cylinders_.']+data_train['cko_eng_cylinders_3'])
# 		# data_train['cko_eng_cylinders_Tr_grp1']=(data_train['cko_eng_cylinders_Tr_4']+data_train['cko_eng_cylinders_Tr_8'])
# 		data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_050016']+data_train['ENG_MDL_CD_050014']+data_train['ENG_MDL_CD_080015']+data_train['ENG_MDL_CD_238004']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_207065']+data_train['ENG_MDL_CD_239003']+data_train['ENG_MDL_CD_207020']+data_train['ENG_MDL_CD_070024']+data_train['ENG_MDL_CD_070009'])
# 		data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_165051']+data_train['ENG_MDL_CD_080110'])
# 		data_train['ENG_MFG_CD_grp1']=(data_train['ENG_MFG_CD_238']+data_train['ENG_MFG_CD_120']+data_train['ENG_MFG_CD_204']+data_train['ENG_MFG_CD_060'])
# 		data_train['FRNT_TYRE_SIZE_CD_grp1']=(data_train['FRNT_TYRE_SIZE_CD_48']+data_train['FRNT_TYRE_SIZE_CD_31']+data_train['FRNT_TYRE_SIZE_CD_49']+data_train['FRNT_TYRE_SIZE_CD_42']+data_train['FRNT_TYRE_SIZE_CD_43'])
# 		data_train['FRNT_TYRE_SIZE_CD_grp2']=(data_train['FRNT_TYRE_SIZE_CD_54']+data_train['FRNT_TYRE_SIZE_CD_48'])
# 		data_train['FRNT_TYRE_SIZE_CD_grp3']=(data_train['FRNT_TYRE_SIZE_CD_433']+data_train['FRNT_TYRE_SIZE_CD_459']+data_train['FRNT_TYRE_SIZE_CD_28']+data_train['FRNT_TYRE_SIZE_CD_434']+data_train['FRNT_TYRE_SIZE_CD_449']+data_train['FRNT_TYRE_SIZE_CD_21']+data_train['FRNT_TYRE_SIZE_CD_85']+data_train['FRNT_TYRE_SIZE_CD_453']+data_train['FRNT_TYRE_SIZE_CD_406']+data_train['FRNT_TYRE_SIZE_CD_20']+data_train['FRNT_TYRE_SIZE_CD_11']+data_train['FRNT_TYRE_SIZE_CD_417']+data_train['FRNT_TYRE_SIZE_CD_5']+data_train['FRNT_TYRE_SIZE_CD_3']+data_train['FRNT_TYRE_SIZE_CD_4']+data_train['FRNT_TYRE_SIZE_CD_81'])
# 		data_train['MAK_NM_grp1']=(data_train['MAK_NM_FISKER_AUTOMOTIVE']+data_train['MAK_NM_MAYBACH']+data_train['MAK_NM_MCLAREN_AUTOMOTIVE']+data_train['MAK_NM_ROLLS-ROYCE']+data_train['MAK_NM_DAIHATSU']+data_train['MAK_NM_ALFA_ROMEO']+data_train['MAK_NM_DATSUN']+data_train['MAK_NM_LOTUS'])
# 		data_train['MAK_NM_grp2']=(data_train['MAK_NM_MERCEDES-BENZ']+data_train['MAK_NM_LEXUS']+data_train['MAK_NM_PONTIAC'])
# 		data_train['MFG_DESC_grp1']=(data_train['MFG_DESC_MAZDA_MOTOR_CORPORATIO']+data_train['MFG_DESC_FISKER_AUTOMOTIVE_INC.']+data_train['MFG_DESC_MERCEDES-BENZ_USA_LLC']+data_train['MFG_DESC_MCLAREN_AUTOMOTIVE'])
# 		data_train['NC_TireSize4_grp1']=(data_train['NC_TireSize4_P235/45R18']+data_train['NC_TireSize4_235/50R18']+data_train['NC_TireSize4_P255/50R19'])
# 		data_train['NC_TireSize4_grp2']=(data_train['NC_TireSize4_P235/60R18']+data_train['NC_TireSize4_P225/55R18'])
# 		data_train['NC_TireSize4_grp3']=(data_train['NC_TireSize4_P215/60R16']+data_train['NC_TireSize4_P265/50R20'])
# 		data_train['NC_TireSize4_grp4']=(data_train['NC_TireSize4_225/45R17']+data_train['NC_TireSize4_P235/50R18'])
# 		data_train['NC_HD_INJ_C_D_grp1']=(data_train['NC_HD_INJ_C_D_463']+data_train['NC_HD_INJ_C_D_1051']+data_train['NC_HD_INJ_C_D_1254']+data_train['NC_HD_INJ_C_D_524']+data_train['NC_HD_INJ_C_D_1564']+data_train['NC_HD_INJ_C_D_1026']+data_train['NC_HD_INJ_C_D_1238']+data_train['NC_HD_INJ_C_D_407']+data_train['NC_HD_INJ_C_D_522']+data_train['NC_HD_INJ_C_D_996']+data_train['NC_HD_INJ_C_D_707']+data_train['NC_HD_INJ_C_D_779']+data_train['NC_HD_INJ_C_D_900']+data_train['NC_HD_INJ_C_D_873']+data_train['NC_HD_INJ_C_D_908']+data_train['NC_HD_INJ_C_D_1024']+data_train['NC_HD_INJ_C_D_898']+data_train['NC_HD_INJ_C_D_867']+data_train['NC_HD_INJ_C_D_744']+data_train['NC_HD_INJ_C_D_750']+data_train['NC_HD_INJ_C_D_546']+data_train['NC_HD_INJ_C_D_1088']+data_train['NC_HD_INJ_C_D_1068']+data_train['NC_HD_INJ_C_D_1459']+data_train['NC_HD_INJ_C_D_528']+data_train['NC_HD_INJ_C_D_1036']+data_train['NC_HD_INJ_C_D_535']+data_train['NC_HD_INJ_C_D_793']+data_train['NC_HD_INJ_C_D_846']+data_train['NC_HD_INJ_C_D_626']+data_train['NC_HD_INJ_C_D_587']+data_train['NC_HD_INJ_C_D_2021']+data_train['NC_HD_INJ_C_D_853']+data_train['NC_HD_INJ_C_D_895']+data_train['NC_HD_INJ_C_D_212']+data_train['NC_HD_INJ_C_D_584']+data_train['NC_HD_INJ_C_D_654']+data_train['NC_HD_INJ_C_D_557']+data_train['NC_HD_INJ_C_D_818']+data_train['NC_HD_INJ_C_D_735']+data_train['NC_HD_INJ_C_D_719']+data_train['NC_HD_INJ_C_D_808']+data_train['NC_HD_INJ_C_D_1182']+data_train['NC_HD_INJ_C_D_433']+data_train['NC_HD_INJ_C_D_960']+data_train['NC_HD_INJ_C_D_516']+data_train['NC_HD_INJ_C_D_367']+data_train['NC_HD_INJ_C_D_622']+data_train['NC_HD_INJ_C_D_863']+data_train['NC_HD_INJ_C_D_710']+data_train['NC_HD_INJ_C_D_930']+data_train['NC_HD_INJ_C_D_1345']+data_train['NC_HD_INJ_C_D_531']+data_train['NC_HD_INJ_C_D_1138']+data_train['NC_HD_INJ_C_D_834']+data_train['NC_HD_INJ_C_D_765'])
# 		data_train['NC_HD_INJ_C_D_grp2']=(data_train['NC_HD_INJ_C_D_338']+data_train['NC_HD_INJ_C_D_519']+data_train['NC_HD_INJ_C_D_664']+data_train['NC_HD_INJ_C_D_640']+data_train['NC_HD_INJ_C_D_998']+data_train['NC_HD_INJ_C_D_236'])
# 		data_train['NC_HD_INJ_C_D_grp3']=(data_train['NC_HD_INJ_C_D_249']+data_train['NC_HD_INJ_C_D_371']+data_train['NC_HD_INJ_C_D_602']+data_train['NC_HD_INJ_C_D_306'])
# 		data_train['NC_HD_INJ_C_D_grp4']=(data_train['NC_HD_INJ_C_D_948']+data_train['NC_HD_INJ_C_D_474']+data_train['NC_HD_INJ_C_D_566']+data_train['NC_HD_INJ_C_D_383'])
# 		data_train['REAR_TIRE_SIZE_CD_grp1']=(data_train['REAR_TIRE_SIZE_CD_466']+data_train['REAR_TIRE_SIZE_CD_93']+data_train['REAR_TIRE_SIZE_CD_16']+data_train['REAR_TIRE_SIZE_CD_85']+data_train['REAR_TIRE_SIZE_CD_97'])
# 		data_train['WHL_DRVN_CNT_Tr__']=data_train['WHL_DRVN_CNT_Tr_.']	

# 		data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_CPE2D4X4']+data_test['bodyStyleCd_CAMPER']+data_test['bodyStyleCd_TRK_4X2']+data_test['bodyStyleCd_CPE4X44D']+data_test['bodyStyleCd_CNV4X42D']+data_test['bodyStyleCd_WAG5D4X4']+data_test['bodyStyleCd_WAGON_2D']+data_test['bodyStyleCd_WAG5D4X2']+data_test['bodyStyleCd_WAG4D4X2']+data_test['bodyStyleCd_VAN']+data_test['bodyStyleCd_SP_HCHBK']+data_test['bodyStyleCd_SED4D4X4']+data_test['bodyStyleCd_SED2D4X4']+data_test['bodyStyleCd_SED4X44D']+data_test['bodyStyleCd_CPE_2+2']+data_test['bodyStyleCd_CP/RDSTR']+data_test['bodyStyleCd_LIMO']+data_test['bodyStyleCd_VAN_4X4']+data_test['bodyStyleCd_ROADSTER']+data_test['bodyStyleCd_MPV_4X2']+data_test['bodyStyleCd_SPORT_CP']+data_test['bodyStyleCd_TRK_4X4']+data_test['bodyStyleCd_HRDTP_2D']+data_test['bodyStyleCd_HCH2D4X4']+data_test['bodyStyleCd_HCH5D4X4']+data_test['bodyStyleCd_CABRIOLE']+data_test['bodyStyleCd_SP_COUPE']+data_test['bodyStyleCd_TARGA']+data_test['bodyStyleCd_COUPE']+data_test['bodyStyleCd_HCH3D4X4'])
# 		data_test['cko_eng_cylinders_grp1']=(data_test['cko_eng_cylinders_4']+data_test['cko_eng_cylinders_5']+data_test['cko_eng_cylinders_.']+data_test['cko_eng_cylinders_3'])
# 		# data_test['cko_eng_cylinders_Tr_grp1']=(data_test['cko_eng_cylinders_Tr_4']+data_test['cko_eng_cylinders_Tr_8'])
# 		data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_050016']+data_test['ENG_MDL_CD_050014']+data_test['ENG_MDL_CD_080015']+data_test['ENG_MDL_CD_238004']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_207065']+data_test['ENG_MDL_CD_239003']+data_test['ENG_MDL_CD_207020']+data_test['ENG_MDL_CD_070024']+data_test['ENG_MDL_CD_070009'])
# 		data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_165051']+data_test['ENG_MDL_CD_080110'])
# 		data_test['ENG_MFG_CD_grp1']=(data_test['ENG_MFG_CD_238']+data_test['ENG_MFG_CD_120']+data_test['ENG_MFG_CD_204']+data_test['ENG_MFG_CD_060'])
# 		data_test['FRNT_TYRE_SIZE_CD_grp1']=(data_test['FRNT_TYRE_SIZE_CD_48']+data_test['FRNT_TYRE_SIZE_CD_31']+data_test['FRNT_TYRE_SIZE_CD_49']+data_test['FRNT_TYRE_SIZE_CD_42']+data_test['FRNT_TYRE_SIZE_CD_43'])
# 		data_test['FRNT_TYRE_SIZE_CD_grp2']=(data_test['FRNT_TYRE_SIZE_CD_54']+data_test['FRNT_TYRE_SIZE_CD_48'])
# 		data_test['FRNT_TYRE_SIZE_CD_grp3']=(data_test['FRNT_TYRE_SIZE_CD_433']+data_test['FRNT_TYRE_SIZE_CD_459']+data_test['FRNT_TYRE_SIZE_CD_28']+data_test['FRNT_TYRE_SIZE_CD_434']+data_test['FRNT_TYRE_SIZE_CD_449']+data_test['FRNT_TYRE_SIZE_CD_21']+data_test['FRNT_TYRE_SIZE_CD_85']+data_test['FRNT_TYRE_SIZE_CD_453']+data_test['FRNT_TYRE_SIZE_CD_406']+data_test['FRNT_TYRE_SIZE_CD_20']+data_test['FRNT_TYRE_SIZE_CD_11']+data_test['FRNT_TYRE_SIZE_CD_417']+data_test['FRNT_TYRE_SIZE_CD_5']+data_test['FRNT_TYRE_SIZE_CD_3']+data_test['FRNT_TYRE_SIZE_CD_4']+data_test['FRNT_TYRE_SIZE_CD_81'])
# 		data_test['MAK_NM_grp1']=(data_test['MAK_NM_FISKER_AUTOMOTIVE']+data_test['MAK_NM_MAYBACH']+data_test['MAK_NM_MCLAREN_AUTOMOTIVE']+data_test['MAK_NM_ROLLS-ROYCE']+data_test['MAK_NM_DAIHATSU']+data_test['MAK_NM_ALFA_ROMEO']+data_test['MAK_NM_DATSUN']+data_test['MAK_NM_LOTUS'])
# 		data_test['MAK_NM_grp2']=(data_test['MAK_NM_MERCEDES-BENZ']+data_test['MAK_NM_LEXUS']+data_test['MAK_NM_PONTIAC'])
# 		data_test['MFG_DESC_grp1']=(data_test['MFG_DESC_MAZDA_MOTOR_CORPORATIO']+data_test['MFG_DESC_FISKER_AUTOMOTIVE_INC.']+data_test['MFG_DESC_MERCEDES-BENZ_USA_LLC']+data_test['MFG_DESC_MCLAREN_AUTOMOTIVE'])
# 		data_test['NC_TireSize4_grp1']=(data_test['NC_TireSize4_P235/45R18']+data_test['NC_TireSize4_235/50R18']+data_test['NC_TireSize4_P255/50R19'])
# 		data_test['NC_TireSize4_grp2']=(data_test['NC_TireSize4_P235/60R18']+data_test['NC_TireSize4_P225/55R18'])
# 		data_test['NC_TireSize4_grp3']=(data_test['NC_TireSize4_P215/60R16']+data_test['NC_TireSize4_P265/50R20'])
# 		data_test['NC_TireSize4_grp4']=(data_test['NC_TireSize4_225/45R17']+data_test['NC_TireSize4_P235/50R18'])
# 		data_test['NC_HD_INJ_C_D_grp1']=(data_test['NC_HD_INJ_C_D_463']+data_test['NC_HD_INJ_C_D_1051']+data_test['NC_HD_INJ_C_D_1254']+data_test['NC_HD_INJ_C_D_524']+data_test['NC_HD_INJ_C_D_1564']+data_test['NC_HD_INJ_C_D_1026']+data_test['NC_HD_INJ_C_D_1238']+data_test['NC_HD_INJ_C_D_407']+data_test['NC_HD_INJ_C_D_522']+data_test['NC_HD_INJ_C_D_996']+data_test['NC_HD_INJ_C_D_707']+data_test['NC_HD_INJ_C_D_779']+data_test['NC_HD_INJ_C_D_900']+data_test['NC_HD_INJ_C_D_873']+data_test['NC_HD_INJ_C_D_908']+data_test['NC_HD_INJ_C_D_1024']+data_test['NC_HD_INJ_C_D_898']+data_test['NC_HD_INJ_C_D_867']+data_test['NC_HD_INJ_C_D_744']+data_test['NC_HD_INJ_C_D_750']+data_test['NC_HD_INJ_C_D_546']+data_test['NC_HD_INJ_C_D_1088']+data_test['NC_HD_INJ_C_D_1068']+data_test['NC_HD_INJ_C_D_1459']+data_test['NC_HD_INJ_C_D_528']+data_test['NC_HD_INJ_C_D_1036']+data_test['NC_HD_INJ_C_D_535']+data_test['NC_HD_INJ_C_D_793']+data_test['NC_HD_INJ_C_D_846']+data_test['NC_HD_INJ_C_D_626']+data_test['NC_HD_INJ_C_D_587']+data_test['NC_HD_INJ_C_D_2021']+data_test['NC_HD_INJ_C_D_853']+data_test['NC_HD_INJ_C_D_895']+data_test['NC_HD_INJ_C_D_212']+data_test['NC_HD_INJ_C_D_584']+data_test['NC_HD_INJ_C_D_654']+data_test['NC_HD_INJ_C_D_557']+data_test['NC_HD_INJ_C_D_818']+data_test['NC_HD_INJ_C_D_735']+data_test['NC_HD_INJ_C_D_719']+data_test['NC_HD_INJ_C_D_808']+data_test['NC_HD_INJ_C_D_1182']+data_test['NC_HD_INJ_C_D_433']+data_test['NC_HD_INJ_C_D_960']+data_test['NC_HD_INJ_C_D_516']+data_test['NC_HD_INJ_C_D_367']+data_test['NC_HD_INJ_C_D_622']+data_test['NC_HD_INJ_C_D_863']+data_test['NC_HD_INJ_C_D_710']+data_test['NC_HD_INJ_C_D_930']+data_test['NC_HD_INJ_C_D_1345']+data_test['NC_HD_INJ_C_D_531']+data_test['NC_HD_INJ_C_D_1138']+data_test['NC_HD_INJ_C_D_834']+data_test['NC_HD_INJ_C_D_765'])
# 		data_test['NC_HD_INJ_C_D_grp2']=(data_test['NC_HD_INJ_C_D_338']+data_test['NC_HD_INJ_C_D_519']+data_test['NC_HD_INJ_C_D_664']+data_test['NC_HD_INJ_C_D_640']+data_test['NC_HD_INJ_C_D_998']+data_test['NC_HD_INJ_C_D_236'])
# 		data_test['NC_HD_INJ_C_D_grp3']=(data_test['NC_HD_INJ_C_D_249']+data_test['NC_HD_INJ_C_D_371']+data_test['NC_HD_INJ_C_D_602']+data_test['NC_HD_INJ_C_D_306'])
# 		data_test['NC_HD_INJ_C_D_grp4']=(data_test['NC_HD_INJ_C_D_948']+data_test['NC_HD_INJ_C_D_474']+data_test['NC_HD_INJ_C_D_566']+data_test['NC_HD_INJ_C_D_383'])
# 		data_test['REAR_TIRE_SIZE_CD_grp1']=(data_test['REAR_TIRE_SIZE_CD_466']+data_test['REAR_TIRE_SIZE_CD_93']+data_test['REAR_TIRE_SIZE_CD_16']+data_test['REAR_TIRE_SIZE_CD_85']+data_test['REAR_TIRE_SIZE_CD_97'])
# 		data_test['WHL_DRVN_CNT_Tr__']=data_test['WHL_DRVN_CNT_Tr_.']	
# 	elif i==56:
# 		data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_CPE2D4X4']+data_train['bodyStyleCd_CAMPER']+data_train['bodyStyleCd_TRK_4X2']+data_train['bodyStyleCd_CPE4X44D']+data_train['bodyStyleCd_CNV4X42D']+data_train['bodyStyleCd_WAG5D4X4']+data_train['bodyStyleCd_WAGON_2D']+data_train['bodyStyleCd_WAG5D4X2']+data_train['bodyStyleCd_WAG4D4X2']+data_train['bodyStyleCd_VAN']+data_train['bodyStyleCd_SP_HCHBK']+data_train['bodyStyleCd_SED4D4X4']+data_train['bodyStyleCd_SED2D4X4']+data_train['bodyStyleCd_SED4X44D']+data_train['bodyStyleCd_CPE_2+2']+data_train['bodyStyleCd_CP/RDSTR']+data_train['bodyStyleCd_LIMO']+data_train['bodyStyleCd_VAN_4X4']+data_train['bodyStyleCd_ROADSTER']+data_train['bodyStyleCd_MPV_4X2']+data_train['bodyStyleCd_SPORT_CP']+data_train['bodyStyleCd_TRK_4X4']+data_train['bodyStyleCd_HRDTP_2D']+data_train['bodyStyleCd_HCH2D4X4']+data_train['bodyStyleCd_HCH5D4X4']+data_train['bodyStyleCd_CABRIOLE']+data_train['bodyStyleCd_SP_COUPE']+data_train['bodyStyleCd_TARGA']+data_train['bodyStyleCd_COUPE']+data_train['bodyStyleCd_HCH3D4X4'])
# 		data_train['cko_eng_cylinders_grp1']=(data_train['cko_eng_cylinders_4']+data_train['cko_eng_cylinders_5']+data_train['cko_eng_cylinders_.']+data_train['cko_eng_cylinders_3'])
# 		data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_050016']+data_train['ENG_MDL_CD_050014']+data_train['ENG_MDL_CD_080015']+data_train['ENG_MDL_CD_238004']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_207065']+data_train['ENG_MDL_CD_239003']+data_train['ENG_MDL_CD_207020']+data_train['ENG_MDL_CD_070024']+data_train['ENG_MDL_CD_070009'])
# 		data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_165051']+data_train['ENG_MDL_CD_080110'])
# 		data_train['ENG_MFG_CD_grp1']=(data_train['ENG_MFG_CD_238']+data_train['ENG_MFG_CD_120']+data_train['ENG_MFG_CD_204']+data_train['ENG_MFG_CD_060'])
# 		data_train['FRNT_TYRE_SIZE_CD_grp1']=(data_train['FRNT_TYRE_SIZE_CD_48']+data_train['FRNT_TYRE_SIZE_CD_31']+data_train['FRNT_TYRE_SIZE_CD_49']+data_train['FRNT_TYRE_SIZE_CD_42']+data_train['FRNT_TYRE_SIZE_CD_43'])
# 		data_train['FRNT_TYRE_SIZE_CD_grp2']=(data_train['FRNT_TYRE_SIZE_CD_54']+data_train['FRNT_TYRE_SIZE_CD_48'])
# 		data_train['FRNT_TYRE_SIZE_CD_grp3']=(data_train['FRNT_TYRE_SIZE_CD_433']+data_train['FRNT_TYRE_SIZE_CD_459']+data_train['FRNT_TYRE_SIZE_CD_28']+data_train['FRNT_TYRE_SIZE_CD_434']+data_train['FRNT_TYRE_SIZE_CD_449']+data_train['FRNT_TYRE_SIZE_CD_21']+data_train['FRNT_TYRE_SIZE_CD_85']+data_train['FRNT_TYRE_SIZE_CD_453']+data_train['FRNT_TYRE_SIZE_CD_406']+data_train['FRNT_TYRE_SIZE_CD_20']+data_train['FRNT_TYRE_SIZE_CD_11']+data_train['FRNT_TYRE_SIZE_CD_417']+data_train['FRNT_TYRE_SIZE_CD_5']+data_train['FRNT_TYRE_SIZE_CD_3']+data_train['FRNT_TYRE_SIZE_CD_4']+data_train['FRNT_TYRE_SIZE_CD_81'])
# 		data_train['MAK_NM_grp1']=(data_train['MAK_NM_FISKER_AUTOMOTIVE']+data_train['MAK_NM_MAYBACH']+data_train['MAK_NM_MCLAREN_AUTOMOTIVE']+data_train['MAK_NM_ROLLS-ROYCE']+data_train['MAK_NM_DAIHATSU']+data_train['MAK_NM_ALFA_ROMEO']+data_train['MAK_NM_DATSUN']+data_train['MAK_NM_LOTUS'])
# 		data_train['MAK_NM_grp2']=(data_train['MAK_NM_MERCEDES-BENZ']+data_train['MAK_NM_LEXUS']+data_train['MAK_NM_PONTIAC'])
# 		data_train['MFG_DESC_grp1']=(data_train['MFG_DESC_MAZDA_MOTOR_CORPORATIO']+data_train['MFG_DESC_FISKER_AUTOMOTIVE_INC.']+data_train['MFG_DESC_MERCEDES-BENZ_USA_LLC']+data_train['MFG_DESC_MCLAREN_AUTOMOTIVE'])
# 		data_train['NC_TireSize4_grp1']=(data_train['NC_TireSize4_P235/45R18']+data_train['NC_TireSize4_235/50R18']+data_train['NC_TireSize4_P255/50R19'])
# 		data_train['NC_TireSize4_grp2']=(data_train['NC_TireSize4_P235/60R18']+data_train['NC_TireSize4_P225/55R18'])
# 		data_train['NC_TireSize4_grp3']=(data_train['NC_TireSize4_P215/60R16']+data_train['NC_TireSize4_P265/50R20'])
# 		data_train['NC_TireSize4_grp4']=(data_train['NC_TireSize4_225/45R17']+data_train['NC_TireSize4_P235/50R18'])
# 		data_train['NC_HD_INJ_C_D_grp1']=(data_train['NC_HD_INJ_C_D_463']+data_train['NC_HD_INJ_C_D_1051']+data_train['NC_HD_INJ_C_D_1254']+data_train['NC_HD_INJ_C_D_524']+data_train['NC_HD_INJ_C_D_1564']+data_train['NC_HD_INJ_C_D_1026']+data_train['NC_HD_INJ_C_D_1238']+data_train['NC_HD_INJ_C_D_407']+data_train['NC_HD_INJ_C_D_522']+data_train['NC_HD_INJ_C_D_996']+data_train['NC_HD_INJ_C_D_707']+data_train['NC_HD_INJ_C_D_779']+data_train['NC_HD_INJ_C_D_900']+data_train['NC_HD_INJ_C_D_873']+data_train['NC_HD_INJ_C_D_908']+data_train['NC_HD_INJ_C_D_1024']+data_train['NC_HD_INJ_C_D_898']+data_train['NC_HD_INJ_C_D_867']+data_train['NC_HD_INJ_C_D_744']+data_train['NC_HD_INJ_C_D_750']+data_train['NC_HD_INJ_C_D_546']+data_train['NC_HD_INJ_C_D_1088']+data_train['NC_HD_INJ_C_D_1068']+data_train['NC_HD_INJ_C_D_1459']+data_train['NC_HD_INJ_C_D_528']+data_train['NC_HD_INJ_C_D_1036']+data_train['NC_HD_INJ_C_D_535']+data_train['NC_HD_INJ_C_D_793']+data_train['NC_HD_INJ_C_D_846']+data_train['NC_HD_INJ_C_D_626']+data_train['NC_HD_INJ_C_D_587']+data_train['NC_HD_INJ_C_D_2021']+data_train['NC_HD_INJ_C_D_853']+data_train['NC_HD_INJ_C_D_895']+data_train['NC_HD_INJ_C_D_212']+data_train['NC_HD_INJ_C_D_584']+data_train['NC_HD_INJ_C_D_654']+data_train['NC_HD_INJ_C_D_557']+data_train['NC_HD_INJ_C_D_818']+data_train['NC_HD_INJ_C_D_735']+data_train['NC_HD_INJ_C_D_719']+data_train['NC_HD_INJ_C_D_808']+data_train['NC_HD_INJ_C_D_1182']+data_train['NC_HD_INJ_C_D_433']+data_train['NC_HD_INJ_C_D_960']+data_train['NC_HD_INJ_C_D_516']+data_train['NC_HD_INJ_C_D_367']+data_train['NC_HD_INJ_C_D_622']+data_train['NC_HD_INJ_C_D_863']+data_train['NC_HD_INJ_C_D_710']+data_train['NC_HD_INJ_C_D_930']+data_train['NC_HD_INJ_C_D_1345']+data_train['NC_HD_INJ_C_D_531']+data_train['NC_HD_INJ_C_D_1138']+data_train['NC_HD_INJ_C_D_834']+data_train['NC_HD_INJ_C_D_765'])
# 		data_train['NC_HD_INJ_C_D_grp2']=(data_train['NC_HD_INJ_C_D_338']+data_train['NC_HD_INJ_C_D_519']+data_train['NC_HD_INJ_C_D_664']+data_train['NC_HD_INJ_C_D_640']+data_train['NC_HD_INJ_C_D_998']+data_train['NC_HD_INJ_C_D_236'])
# 		data_train['NC_HD_INJ_C_D_grp3']=(data_train['NC_HD_INJ_C_D_249']+data_train['NC_HD_INJ_C_D_371']+data_train['NC_HD_INJ_C_D_602']+data_train['NC_HD_INJ_C_D_306'])
# 		data_train['NC_HD_INJ_C_D_grp4']=(data_train['NC_HD_INJ_C_D_948']+data_train['NC_HD_INJ_C_D_474']+data_train['NC_HD_INJ_C_D_566']+data_train['NC_HD_INJ_C_D_383'])
# 		data_train['REAR_TIRE_SIZE_CD_grp1']=(data_train['REAR_TIRE_SIZE_CD_466']+data_train['REAR_TIRE_SIZE_CD_93']+data_train['REAR_TIRE_SIZE_CD_16']+data_train['REAR_TIRE_SIZE_CD_85']+data_train['REAR_TIRE_SIZE_CD_97'])
# 		data_train['WHL_DRVN_CNT_Tr__']=data_train['WHL_DRVN_CNT_Tr_.']	


# 		data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_CPE2D4X4']+data_test['bodyStyleCd_CAMPER']+data_test['bodyStyleCd_TRK_4X2']+data_test['bodyStyleCd_CPE4X44D']+data_test['bodyStyleCd_CNV4X42D']+data_test['bodyStyleCd_WAG5D4X4']+data_test['bodyStyleCd_WAGON_2D']+data_test['bodyStyleCd_WAG5D4X2']+data_test['bodyStyleCd_WAG4D4X2']+data_test['bodyStyleCd_VAN']+data_test['bodyStyleCd_SP_HCHBK']+data_test['bodyStyleCd_SED4D4X4']+data_test['bodyStyleCd_SED2D4X4']+data_test['bodyStyleCd_SED4X44D']+data_test['bodyStyleCd_CPE_2+2']+data_test['bodyStyleCd_CP/RDSTR']+data_test['bodyStyleCd_LIMO']+data_test['bodyStyleCd_VAN_4X4']+data_test['bodyStyleCd_ROADSTER']+data_test['bodyStyleCd_MPV_4X2']+data_test['bodyStyleCd_SPORT_CP']+data_test['bodyStyleCd_TRK_4X4']+data_test['bodyStyleCd_HRDTP_2D']+data_test['bodyStyleCd_HCH2D4X4']+data_test['bodyStyleCd_HCH5D4X4']+data_test['bodyStyleCd_CABRIOLE']+data_test['bodyStyleCd_SP_COUPE']+data_test['bodyStyleCd_TARGA']+data_test['bodyStyleCd_COUPE']+data_test['bodyStyleCd_HCH3D4X4'])
# 		data_test['cko_eng_cylinders_grp1']=(data_test['cko_eng_cylinders_4']+data_test['cko_eng_cylinders_5']+data_test['cko_eng_cylinders_.']+data_test['cko_eng_cylinders_3'])
# 		data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_050016']+data_test['ENG_MDL_CD_050014']+data_test['ENG_MDL_CD_080015']+data_test['ENG_MDL_CD_238004']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_207065']+data_test['ENG_MDL_CD_239003']+data_test['ENG_MDL_CD_207020']+data_test['ENG_MDL_CD_070024']+data_test['ENG_MDL_CD_070009'])
# 		data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_165051']+data_test['ENG_MDL_CD_080110'])
# 		data_test['ENG_MFG_CD_grp1']=(data_test['ENG_MFG_CD_238']+data_test['ENG_MFG_CD_120']+data_test['ENG_MFG_CD_204']+data_test['ENG_MFG_CD_060'])
# 		data_test['FRNT_TYRE_SIZE_CD_grp1']=(data_test['FRNT_TYRE_SIZE_CD_48']+data_test['FRNT_TYRE_SIZE_CD_31']+data_test['FRNT_TYRE_SIZE_CD_49']+data_test['FRNT_TYRE_SIZE_CD_42']+data_test['FRNT_TYRE_SIZE_CD_43'])
# 		data_test['FRNT_TYRE_SIZE_CD_grp2']=(data_test['FRNT_TYRE_SIZE_CD_54']+data_test['FRNT_TYRE_SIZE_CD_48'])
# 		data_test['FRNT_TYRE_SIZE_CD_grp3']=(data_test['FRNT_TYRE_SIZE_CD_433']+data_test['FRNT_TYRE_SIZE_CD_459']+data_test['FRNT_TYRE_SIZE_CD_28']+data_test['FRNT_TYRE_SIZE_CD_434']+data_test['FRNT_TYRE_SIZE_CD_449']+data_test['FRNT_TYRE_SIZE_CD_21']+data_test['FRNT_TYRE_SIZE_CD_85']+data_test['FRNT_TYRE_SIZE_CD_453']+data_test['FRNT_TYRE_SIZE_CD_406']+data_test['FRNT_TYRE_SIZE_CD_20']+data_test['FRNT_TYRE_SIZE_CD_11']+data_test['FRNT_TYRE_SIZE_CD_417']+data_test['FRNT_TYRE_SIZE_CD_5']+data_test['FRNT_TYRE_SIZE_CD_3']+data_test['FRNT_TYRE_SIZE_CD_4']+data_test['FRNT_TYRE_SIZE_CD_81'])
# 		data_test['MAK_NM_grp1']=(data_test['MAK_NM_FISKER_AUTOMOTIVE']+data_test['MAK_NM_MAYBACH']+data_test['MAK_NM_MCLAREN_AUTOMOTIVE']+data_test['MAK_NM_ROLLS-ROYCE']+data_test['MAK_NM_DAIHATSU']+data_test['MAK_NM_ALFA_ROMEO']+data_test['MAK_NM_DATSUN']+data_test['MAK_NM_LOTUS'])
# 		data_test['MAK_NM_grp2']=(data_test['MAK_NM_MERCEDES-BENZ']+data_test['MAK_NM_LEXUS']+data_test['MAK_NM_PONTIAC'])
# 		data_test['MFG_DESC_grp1']=(data_test['MFG_DESC_MAZDA_MOTOR_CORPORATIO']+data_test['MFG_DESC_FISKER_AUTOMOTIVE_INC.']+data_test['MFG_DESC_MERCEDES-BENZ_USA_LLC']+data_test['MFG_DESC_MCLAREN_AUTOMOTIVE'])
# 		data_test['NC_TireSize4_grp1']=(data_test['NC_TireSize4_P235/45R18']+data_test['NC_TireSize4_235/50R18']+data_test['NC_TireSize4_P255/50R19'])
# 		data_test['NC_TireSize4_grp2']=(data_test['NC_TireSize4_P235/60R18']+data_test['NC_TireSize4_P225/55R18'])
# 		data_test['NC_TireSize4_grp3']=(data_test['NC_TireSize4_P215/60R16']+data_test['NC_TireSize4_P265/50R20'])
# 		data_test['NC_TireSize4_grp4']=(data_test['NC_TireSize4_225/45R17']+data_test['NC_TireSize4_P235/50R18'])
# 		data_test['NC_HD_INJ_C_D_grp1']=(data_test['NC_HD_INJ_C_D_463']+data_test['NC_HD_INJ_C_D_1051']+data_test['NC_HD_INJ_C_D_1254']+data_test['NC_HD_INJ_C_D_524']+data_test['NC_HD_INJ_C_D_1564']+data_test['NC_HD_INJ_C_D_1026']+data_test['NC_HD_INJ_C_D_1238']+data_test['NC_HD_INJ_C_D_407']+data_test['NC_HD_INJ_C_D_522']+data_test['NC_HD_INJ_C_D_996']+data_test['NC_HD_INJ_C_D_707']+data_test['NC_HD_INJ_C_D_779']+data_test['NC_HD_INJ_C_D_900']+data_test['NC_HD_INJ_C_D_873']+data_test['NC_HD_INJ_C_D_908']+data_test['NC_HD_INJ_C_D_1024']+data_test['NC_HD_INJ_C_D_898']+data_test['NC_HD_INJ_C_D_867']+data_test['NC_HD_INJ_C_D_744']+data_test['NC_HD_INJ_C_D_750']+data_test['NC_HD_INJ_C_D_546']+data_test['NC_HD_INJ_C_D_1088']+data_test['NC_HD_INJ_C_D_1068']+data_test['NC_HD_INJ_C_D_1459']+data_test['NC_HD_INJ_C_D_528']+data_test['NC_HD_INJ_C_D_1036']+data_test['NC_HD_INJ_C_D_535']+data_test['NC_HD_INJ_C_D_793']+data_test['NC_HD_INJ_C_D_846']+data_test['NC_HD_INJ_C_D_626']+data_test['NC_HD_INJ_C_D_587']+data_test['NC_HD_INJ_C_D_2021']+data_test['NC_HD_INJ_C_D_853']+data_test['NC_HD_INJ_C_D_895']+data_test['NC_HD_INJ_C_D_212']+data_test['NC_HD_INJ_C_D_584']+data_test['NC_HD_INJ_C_D_654']+data_test['NC_HD_INJ_C_D_557']+data_test['NC_HD_INJ_C_D_818']+data_test['NC_HD_INJ_C_D_735']+data_test['NC_HD_INJ_C_D_719']+data_test['NC_HD_INJ_C_D_808']+data_test['NC_HD_INJ_C_D_1182']+data_test['NC_HD_INJ_C_D_433']+data_test['NC_HD_INJ_C_D_960']+data_test['NC_HD_INJ_C_D_516']+data_test['NC_HD_INJ_C_D_367']+data_test['NC_HD_INJ_C_D_622']+data_test['NC_HD_INJ_C_D_863']+data_test['NC_HD_INJ_C_D_710']+data_test['NC_HD_INJ_C_D_930']+data_test['NC_HD_INJ_C_D_1345']+data_test['NC_HD_INJ_C_D_531']+data_test['NC_HD_INJ_C_D_1138']+data_test['NC_HD_INJ_C_D_834']+data_test['NC_HD_INJ_C_D_765'])
# 		data_test['NC_HD_INJ_C_D_grp2']=(data_test['NC_HD_INJ_C_D_338']+data_test['NC_HD_INJ_C_D_519']+data_test['NC_HD_INJ_C_D_664']+data_test['NC_HD_INJ_C_D_640']+data_test['NC_HD_INJ_C_D_998']+data_test['NC_HD_INJ_C_D_236'])
# 		data_test['NC_HD_INJ_C_D_grp3']=(data_test['NC_HD_INJ_C_D_249']+data_test['NC_HD_INJ_C_D_371']+data_test['NC_HD_INJ_C_D_602']+data_test['NC_HD_INJ_C_D_306'])
# 		data_test['NC_HD_INJ_C_D_grp4']=(data_test['NC_HD_INJ_C_D_948']+data_test['NC_HD_INJ_C_D_474']+data_test['NC_HD_INJ_C_D_566']+data_test['NC_HD_INJ_C_D_383'])
# 		data_test['REAR_TIRE_SIZE_CD_grp1']=(data_test['REAR_TIRE_SIZE_CD_466']+data_test['REAR_TIRE_SIZE_CD_93']+data_test['REAR_TIRE_SIZE_CD_16']+data_test['REAR_TIRE_SIZE_CD_85']+data_test['REAR_TIRE_SIZE_CD_97'])
# 		data_test['WHL_DRVN_CNT_Tr__']=data_test['WHL_DRVN_CNT_Tr_.']	
# 	elif i==57:
# 		data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_CPE2D4X4']+data_train['bodyStyleCd_CAMPER']+data_train['bodyStyleCd_TRK_4X2']+data_train['bodyStyleCd_CPE4X44D']+data_train['bodyStyleCd_CNV4X42D']+data_train['bodyStyleCd_WAG5D4X4']+data_train['bodyStyleCd_WAGON_2D']+data_train['bodyStyleCd_WAG5D4X2']+data_train['bodyStyleCd_WAG4D4X2']+data_train['bodyStyleCd_VAN']+data_train['bodyStyleCd_SP_HCHBK']+data_train['bodyStyleCd_SED4D4X4']+data_train['bodyStyleCd_SED2D4X4']+data_train['bodyStyleCd_SED4X44D']+data_train['bodyStyleCd_CPE_2+2']+data_train['bodyStyleCd_CP/RDSTR']+data_train['bodyStyleCd_LIMO']+data_train['bodyStyleCd_VAN_4X4']+data_train['bodyStyleCd_ROADSTER']+data_train['bodyStyleCd_MPV_4X2']+data_train['bodyStyleCd_SPORT_CP']+data_train['bodyStyleCd_TRK_4X4']+data_train['bodyStyleCd_HRDTP_2D']+data_train['bodyStyleCd_HCH2D4X4']+data_train['bodyStyleCd_HCH5D4X4']+data_train['bodyStyleCd_CABRIOLE']+data_train['bodyStyleCd_SP_COUPE']+data_train['bodyStyleCd_TARGA']+data_train['bodyStyleCd_COUPE']+data_train['bodyStyleCd_HCH3D4X4'])
# 		data_train['cko_eng_cylinders_grp1']=(data_train['cko_eng_cylinders_4']+data_train['cko_eng_cylinders_5']+data_train['cko_eng_cylinders_.']+data_train['cko_eng_cylinders_3'])
# 		data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_050016']+data_train['ENG_MDL_CD_050014']+data_train['ENG_MDL_CD_080015']+data_train['ENG_MDL_CD_238004']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_207065']+data_train['ENG_MDL_CD_239003']+data_train['ENG_MDL_CD_207020']+data_train['ENG_MDL_CD_070024']+data_train['ENG_MDL_CD_070009'])
# 		data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_165051']+data_train['ENG_MDL_CD_080110'])
# 		data_train['ENG_MFG_CD_grp1']=(data_train['ENG_MFG_CD_238']+data_train['ENG_MFG_CD_120']+data_train['ENG_MFG_CD_204']+data_train['ENG_MFG_CD_060'])
# 		data_train['FRNT_TYRE_SIZE_CD_grp1']=(data_train['FRNT_TYRE_SIZE_CD_48']+data_train['FRNT_TYRE_SIZE_CD_31']+data_train['FRNT_TYRE_SIZE_CD_49']+data_train['FRNT_TYRE_SIZE_CD_42']+data_train['FRNT_TYRE_SIZE_CD_43'])
# 		data_train['FRNT_TYRE_SIZE_CD_grp2']=(data_train['FRNT_TYRE_SIZE_CD_54']+data_train['FRNT_TYRE_SIZE_CD_48'])
# 		data_train['FRNT_TYRE_SIZE_CD_grp3']=(data_train['FRNT_TYRE_SIZE_CD_433']+data_train['FRNT_TYRE_SIZE_CD_459']+data_train['FRNT_TYRE_SIZE_CD_28']+data_train['FRNT_TYRE_SIZE_CD_434']+data_train['FRNT_TYRE_SIZE_CD_449']+data_train['FRNT_TYRE_SIZE_CD_21']+data_train['FRNT_TYRE_SIZE_CD_85']+data_train['FRNT_TYRE_SIZE_CD_453']+data_train['FRNT_TYRE_SIZE_CD_406']+data_train['FRNT_TYRE_SIZE_CD_20']+data_train['FRNT_TYRE_SIZE_CD_11']+data_train['FRNT_TYRE_SIZE_CD_417']+data_train['FRNT_TYRE_SIZE_CD_5']+data_train['FRNT_TYRE_SIZE_CD_3']+data_train['FRNT_TYRE_SIZE_CD_4']+data_train['FRNT_TYRE_SIZE_CD_81'])
# 		data_train['MAK_NM_grp1']=(data_train['MAK_NM_FISKER_AUTOMOTIVE']+data_train['MAK_NM_MAYBACH']+data_train['MAK_NM_MCLAREN_AUTOMOTIVE']+data_train['MAK_NM_ROLLS-ROYCE']+data_train['MAK_NM_DAIHATSU']+data_train['MAK_NM_ALFA_ROMEO']+data_train['MAK_NM_DATSUN']+data_train['MAK_NM_LOTUS'])
# 		data_train['MAK_NM_grp2']=(data_train['MAK_NM_MERCEDES-BENZ']+data_train['MAK_NM_LEXUS']+data_train['MAK_NM_PONTIAC'])
# 		data_train['MFG_DESC_grp1']=(data_train['MFG_DESC_MAZDA_MOTOR_CORPORATIO']+data_train['MFG_DESC_FISKER_AUTOMOTIVE_INC.']+data_train['MFG_DESC_MERCEDES-BENZ_USA_LLC']+data_train['MFG_DESC_MCLAREN_AUTOMOTIVE'])
# 		data_train['NC_TireSize4_grp1']=(data_train['NC_TireSize4_P235/45R18']+data_train['NC_TireSize4_235/50R18']+data_train['NC_TireSize4_P255/50R19'])
# 		data_train['NC_TireSize4_grp2']=(data_train['NC_TireSize4_P235/60R18']+data_train['NC_TireSize4_P225/55R18'])
# 		data_train['NC_TireSize4_grp3']=(data_train['NC_TireSize4_P215/60R16']+data_train['NC_TireSize4_P265/50R20'])
# 		data_train['NC_TireSize4_grp4']=(data_train['NC_TireSize4_225/45R17']+data_train['NC_TireSize4_P235/50R18'])
# 		data_train['NC_HD_INJ_C_D_grp1']=(data_train['NC_HD_INJ_C_D_463']+data_train['NC_HD_INJ_C_D_1051']+data_train['NC_HD_INJ_C_D_1254']+data_train['NC_HD_INJ_C_D_524']+data_train['NC_HD_INJ_C_D_1564']+data_train['NC_HD_INJ_C_D_1026']+data_train['NC_HD_INJ_C_D_1238']+data_train['NC_HD_INJ_C_D_407']+data_train['NC_HD_INJ_C_D_522']+data_train['NC_HD_INJ_C_D_996']+data_train['NC_HD_INJ_C_D_707']+data_train['NC_HD_INJ_C_D_779']+data_train['NC_HD_INJ_C_D_900']+data_train['NC_HD_INJ_C_D_873']+data_train['NC_HD_INJ_C_D_908']+data_train['NC_HD_INJ_C_D_1024']+data_train['NC_HD_INJ_C_D_898']+data_train['NC_HD_INJ_C_D_867']+data_train['NC_HD_INJ_C_D_744']+data_train['NC_HD_INJ_C_D_750']+data_train['NC_HD_INJ_C_D_546']+data_train['NC_HD_INJ_C_D_1088']+data_train['NC_HD_INJ_C_D_1068']+data_train['NC_HD_INJ_C_D_1459']+data_train['NC_HD_INJ_C_D_528']+data_train['NC_HD_INJ_C_D_1036']+data_train['NC_HD_INJ_C_D_535']+data_train['NC_HD_INJ_C_D_793']+data_train['NC_HD_INJ_C_D_846']+data_train['NC_HD_INJ_C_D_626']+data_train['NC_HD_INJ_C_D_587']+data_train['NC_HD_INJ_C_D_2021']+data_train['NC_HD_INJ_C_D_853']+data_train['NC_HD_INJ_C_D_895']+data_train['NC_HD_INJ_C_D_212']+data_train['NC_HD_INJ_C_D_584']+data_train['NC_HD_INJ_C_D_654']+data_train['NC_HD_INJ_C_D_557']+data_train['NC_HD_INJ_C_D_818']+data_train['NC_HD_INJ_C_D_735']+data_train['NC_HD_INJ_C_D_719']+data_train['NC_HD_INJ_C_D_808']+data_train['NC_HD_INJ_C_D_1182']+data_train['NC_HD_INJ_C_D_433']+data_train['NC_HD_INJ_C_D_960']+data_train['NC_HD_INJ_C_D_516']+data_train['NC_HD_INJ_C_D_367']+data_train['NC_HD_INJ_C_D_622']+data_train['NC_HD_INJ_C_D_863']+data_train['NC_HD_INJ_C_D_710']+data_train['NC_HD_INJ_C_D_930']+data_train['NC_HD_INJ_C_D_1345']+data_train['NC_HD_INJ_C_D_531']+data_train['NC_HD_INJ_C_D_1138']+data_train['NC_HD_INJ_C_D_834']+data_train['NC_HD_INJ_C_D_765'])
# 		data_train['NC_HD_INJ_C_D_grp2']=(data_train['NC_HD_INJ_C_D_338']+data_train['NC_HD_INJ_C_D_519']+data_train['NC_HD_INJ_C_D_664']+data_train['NC_HD_INJ_C_D_640']+data_train['NC_HD_INJ_C_D_998']+data_train['NC_HD_INJ_C_D_236'])
# 		data_train['NC_HD_INJ_C_D_grp3']=(data_train['NC_HD_INJ_C_D_249']+data_train['NC_HD_INJ_C_D_371']+data_train['NC_HD_INJ_C_D_602']+data_train['NC_HD_INJ_C_D_306'])
# 		data_train['NC_HD_INJ_C_D_grp4']=(data_train['NC_HD_INJ_C_D_948']+data_train['NC_HD_INJ_C_D_474']+data_train['NC_HD_INJ_C_D_566']+data_train['NC_HD_INJ_C_D_383'])
# 		data_train['REAR_TIRE_SIZE_CD_grp1']=(data_train['REAR_TIRE_SIZE_CD_466']+data_train['REAR_TIRE_SIZE_CD_93']+data_train['REAR_TIRE_SIZE_CD_16']+data_train['REAR_TIRE_SIZE_CD_85']+data_train['REAR_TIRE_SIZE_CD_97'])
# 		data_train['WHL_DRVN_CNT_Tr__']=data_train['WHL_DRVN_CNT_Tr_.']	
# 		data_train['MAK_NM_ROLLS_ROYCE']=data_train['MAK_NM_ROLLS-ROYCE']
# 		data_train['NC_TireSize4_P225_55R18']=data_train['NC_TireSize4_P225/55R18']
# 		data_train['NC_TireSize4_P235_60R18']=data_train['NC_TireSize4_P235/60R18']
# 		data_train['NC_TireSize4_225_45R17']=data_train['NC_TireSize4_225/45R17']
# 		data_train['NC_TireSize4_P235_50R18']=data_train['NC_TireSize4_P235/50R18']

# 		data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_CPE2D4X4']+data_test['bodyStyleCd_CAMPER']+data_test['bodyStyleCd_TRK_4X2']+data_test['bodyStyleCd_CPE4X44D']+data_test['bodyStyleCd_CNV4X42D']+data_test['bodyStyleCd_WAG5D4X4']+data_test['bodyStyleCd_WAGON_2D']+data_test['bodyStyleCd_WAG5D4X2']+data_test['bodyStyleCd_WAG4D4X2']+data_test['bodyStyleCd_VAN']+data_test['bodyStyleCd_SP_HCHBK']+data_test['bodyStyleCd_SED4D4X4']+data_test['bodyStyleCd_SED2D4X4']+data_test['bodyStyleCd_SED4X44D']+data_test['bodyStyleCd_CPE_2+2']+data_test['bodyStyleCd_CP/RDSTR']+data_test['bodyStyleCd_LIMO']+data_test['bodyStyleCd_VAN_4X4']+data_test['bodyStyleCd_ROADSTER']+data_test['bodyStyleCd_MPV_4X2']+data_test['bodyStyleCd_SPORT_CP']+data_test['bodyStyleCd_TRK_4X4']+data_test['bodyStyleCd_HRDTP_2D']+data_test['bodyStyleCd_HCH2D4X4']+data_test['bodyStyleCd_HCH5D4X4']+data_test['bodyStyleCd_CABRIOLE']+data_test['bodyStyleCd_SP_COUPE']+data_test['bodyStyleCd_TARGA']+data_test['bodyStyleCd_COUPE']+data_test['bodyStyleCd_HCH3D4X4'])
# 		data_test['cko_eng_cylinders_grp1']=(data_test['cko_eng_cylinders_4']+data_test['cko_eng_cylinders_5']+data_test['cko_eng_cylinders_.']+data_test['cko_eng_cylinders_3'])
# 		data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_050016']+data_test['ENG_MDL_CD_050014']+data_test['ENG_MDL_CD_080015']+data_test['ENG_MDL_CD_238004']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_207065']+data_test['ENG_MDL_CD_239003']+data_test['ENG_MDL_CD_207020']+data_test['ENG_MDL_CD_070024']+data_test['ENG_MDL_CD_070009'])
# 		data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_165051']+data_test['ENG_MDL_CD_080110'])
# 		data_test['ENG_MFG_CD_grp1']=(data_test['ENG_MFG_CD_238']+data_test['ENG_MFG_CD_120']+data_test['ENG_MFG_CD_204']+data_test['ENG_MFG_CD_060'])
# 		data_test['FRNT_TYRE_SIZE_CD_grp1']=(data_test['FRNT_TYRE_SIZE_CD_48']+data_test['FRNT_TYRE_SIZE_CD_31']+data_test['FRNT_TYRE_SIZE_CD_49']+data_test['FRNT_TYRE_SIZE_CD_42']+data_test['FRNT_TYRE_SIZE_CD_43'])
# 		data_test['FRNT_TYRE_SIZE_CD_grp2']=(data_test['FRNT_TYRE_SIZE_CD_54']+data_test['FRNT_TYRE_SIZE_CD_48'])
# 		data_test['FRNT_TYRE_SIZE_CD_grp3']=(data_test['FRNT_TYRE_SIZE_CD_433']+data_test['FRNT_TYRE_SIZE_CD_459']+data_test['FRNT_TYRE_SIZE_CD_28']+data_test['FRNT_TYRE_SIZE_CD_434']+data_test['FRNT_TYRE_SIZE_CD_449']+data_test['FRNT_TYRE_SIZE_CD_21']+data_test['FRNT_TYRE_SIZE_CD_85']+data_test['FRNT_TYRE_SIZE_CD_453']+data_test['FRNT_TYRE_SIZE_CD_406']+data_test['FRNT_TYRE_SIZE_CD_20']+data_test['FRNT_TYRE_SIZE_CD_11']+data_test['FRNT_TYRE_SIZE_CD_417']+data_test['FRNT_TYRE_SIZE_CD_5']+data_test['FRNT_TYRE_SIZE_CD_3']+data_test['FRNT_TYRE_SIZE_CD_4']+data_test['FRNT_TYRE_SIZE_CD_81'])
# 		data_test['MAK_NM_grp1']=(data_test['MAK_NM_FISKER_AUTOMOTIVE']+data_test['MAK_NM_MAYBACH']+data_test['MAK_NM_MCLAREN_AUTOMOTIVE']+data_test['MAK_NM_ROLLS-ROYCE']+data_test['MAK_NM_DAIHATSU']+data_test['MAK_NM_ALFA_ROMEO']+data_test['MAK_NM_DATSUN']+data_test['MAK_NM_LOTUS'])
# 		data_test['MAK_NM_grp2']=(data_test['MAK_NM_MERCEDES-BENZ']+data_test['MAK_NM_LEXUS']+data_test['MAK_NM_PONTIAC'])
# 		data_test['MFG_DESC_grp1']=(data_test['MFG_DESC_MAZDA_MOTOR_CORPORATIO']+data_test['MFG_DESC_FISKER_AUTOMOTIVE_INC.']+data_test['MFG_DESC_MERCEDES-BENZ_USA_LLC']+data_test['MFG_DESC_MCLAREN_AUTOMOTIVE'])
# 		data_test['NC_TireSize4_grp1']=(data_test['NC_TireSize4_P235/45R18']+data_test['NC_TireSize4_235/50R18']+data_test['NC_TireSize4_P255/50R19'])
# 		data_test['NC_TireSize4_grp2']=(data_test['NC_TireSize4_P235/60R18']+data_test['NC_TireSize4_P225/55R18'])
# 		data_test['NC_TireSize4_grp3']=(data_test['NC_TireSize4_P215/60R16']+data_test['NC_TireSize4_P265/50R20'])
# 		data_test['NC_TireSize4_grp4']=(data_test['NC_TireSize4_225/45R17']+data_test['NC_TireSize4_P235/50R18'])
# 		data_test['NC_HD_INJ_C_D_grp1']=(data_test['NC_HD_INJ_C_D_463']+data_test['NC_HD_INJ_C_D_1051']+data_test['NC_HD_INJ_C_D_1254']+data_test['NC_HD_INJ_C_D_524']+data_test['NC_HD_INJ_C_D_1564']+data_test['NC_HD_INJ_C_D_1026']+data_test['NC_HD_INJ_C_D_1238']+data_test['NC_HD_INJ_C_D_407']+data_test['NC_HD_INJ_C_D_522']+data_test['NC_HD_INJ_C_D_996']+data_test['NC_HD_INJ_C_D_707']+data_test['NC_HD_INJ_C_D_779']+data_test['NC_HD_INJ_C_D_900']+data_test['NC_HD_INJ_C_D_873']+data_test['NC_HD_INJ_C_D_908']+data_test['NC_HD_INJ_C_D_1024']+data_test['NC_HD_INJ_C_D_898']+data_test['NC_HD_INJ_C_D_867']+data_test['NC_HD_INJ_C_D_744']+data_test['NC_HD_INJ_C_D_750']+data_test['NC_HD_INJ_C_D_546']+data_test['NC_HD_INJ_C_D_1088']+data_test['NC_HD_INJ_C_D_1068']+data_test['NC_HD_INJ_C_D_1459']+data_test['NC_HD_INJ_C_D_528']+data_test['NC_HD_INJ_C_D_1036']+data_test['NC_HD_INJ_C_D_535']+data_test['NC_HD_INJ_C_D_793']+data_test['NC_HD_INJ_C_D_846']+data_test['NC_HD_INJ_C_D_626']+data_test['NC_HD_INJ_C_D_587']+data_test['NC_HD_INJ_C_D_2021']+data_test['NC_HD_INJ_C_D_853']+data_test['NC_HD_INJ_C_D_895']+data_test['NC_HD_INJ_C_D_212']+data_test['NC_HD_INJ_C_D_584']+data_test['NC_HD_INJ_C_D_654']+data_test['NC_HD_INJ_C_D_557']+data_test['NC_HD_INJ_C_D_818']+data_test['NC_HD_INJ_C_D_735']+data_test['NC_HD_INJ_C_D_719']+data_test['NC_HD_INJ_C_D_808']+data_test['NC_HD_INJ_C_D_1182']+data_test['NC_HD_INJ_C_D_433']+data_test['NC_HD_INJ_C_D_960']+data_test['NC_HD_INJ_C_D_516']+data_test['NC_HD_INJ_C_D_367']+data_test['NC_HD_INJ_C_D_622']+data_test['NC_HD_INJ_C_D_863']+data_test['NC_HD_INJ_C_D_710']+data_test['NC_HD_INJ_C_D_930']+data_test['NC_HD_INJ_C_D_1345']+data_test['NC_HD_INJ_C_D_531']+data_test['NC_HD_INJ_C_D_1138']+data_test['NC_HD_INJ_C_D_834']+data_test['NC_HD_INJ_C_D_765'])
# 		data_test['NC_HD_INJ_C_D_grp2']=(data_test['NC_HD_INJ_C_D_338']+data_test['NC_HD_INJ_C_D_519']+data_test['NC_HD_INJ_C_D_664']+data_test['NC_HD_INJ_C_D_640']+data_test['NC_HD_INJ_C_D_998']+data_test['NC_HD_INJ_C_D_236'])
# 		data_test['NC_HD_INJ_C_D_grp3']=(data_test['NC_HD_INJ_C_D_249']+data_test['NC_HD_INJ_C_D_371']+data_test['NC_HD_INJ_C_D_602']+data_test['NC_HD_INJ_C_D_306'])
# 		data_test['NC_HD_INJ_C_D_grp4']=(data_test['NC_HD_INJ_C_D_948']+data_test['NC_HD_INJ_C_D_474']+data_test['NC_HD_INJ_C_D_566']+data_test['NC_HD_INJ_C_D_383'])
# 		data_test['REAR_TIRE_SIZE_CD_grp1']=(data_test['REAR_TIRE_SIZE_CD_466']+data_test['REAR_TIRE_SIZE_CD_93']+data_test['REAR_TIRE_SIZE_CD_16']+data_test['REAR_TIRE_SIZE_CD_85']+data_test['REAR_TIRE_SIZE_CD_97'])
# 		data_test['WHL_DRVN_CNT_Tr__']=data_test['WHL_DRVN_CNT_Tr_.']
# 		data_test['MAK_NM_ROLLS_ROYCE']=data_test['MAK_NM_ROLLS-ROYCE']
# 		data_test['NC_TireSize4_P225_55R18']=data_test['NC_TireSize4_P225/55R18']
# 		data_test['NC_TireSize4_P235_60R18']=data_test['NC_TireSize4_P235/60R18']
# 		data_test['NC_TireSize4_225_45R17']=data_test['NC_TireSize4_225/45R17']
# 		data_test['NC_TireSize4_P235_50R18']=data_test['NC_TireSize4_P235/50R18']
		
# 	elif i==58:
# 		data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_CPE2D4X4']+data_train['bodyStyleCd_CAMPER']+data_train['bodyStyleCd_TRK_4X2']+data_train['bodyStyleCd_CPE4X44D']+data_train['bodyStyleCd_CNV4X42D']+data_train['bodyStyleCd_WAG5D4X4']+data_train['bodyStyleCd_WAGON_2D']+data_train['bodyStyleCd_WAG5D4X2']+data_train['bodyStyleCd_WAG4D4X2']+data_train['bodyStyleCd_VAN']+data_train['bodyStyleCd_SP_HCHBK']+data_train['bodyStyleCd_SED4D4X4']+data_train['bodyStyleCd_SED2D4X4']+data_train['bodyStyleCd_SED4X44D']+data_train['bodyStyleCd_CPE_2+2']+data_train['bodyStyleCd_CP/RDSTR']+data_train['bodyStyleCd_LIMO']+data_train['bodyStyleCd_VAN_4X4']+data_train['bodyStyleCd_ROADSTER']+data_train['bodyStyleCd_MPV_4X2']+data_train['bodyStyleCd_SPORT_CP']+data_train['bodyStyleCd_TRK_4X4']+data_train['bodyStyleCd_HRDTP_2D']+data_train['bodyStyleCd_HCH2D4X4']+data_train['bodyStyleCd_HCH5D4X4']+data_train['bodyStyleCd_CABRIOLE']+data_train['bodyStyleCd_SP_COUPE']+data_train['bodyStyleCd_TARGA']+data_train['bodyStyleCd_COUPE']+data_train['bodyStyleCd_HCH3D4X4'])
# 		data_train['cko_eng_cylinders_grp1']=(data_train['cko_eng_cylinders_4']+data_train['cko_eng_cylinders_5']+data_train['cko_eng_cylinders_.']+data_train['cko_eng_cylinders_3'])
# 		data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_050016']+data_train['ENG_MDL_CD_050014']+data_train['ENG_MDL_CD_080015']+data_train['ENG_MDL_CD_238004']+data_train['ENG_MDL_CD_120020']+data_train['ENG_MDL_CD_207065']+data_train['ENG_MDL_CD_239003']+data_train['ENG_MDL_CD_207020']+data_train['ENG_MDL_CD_070024']+data_train['ENG_MDL_CD_070009'])
# 		data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_165051']+data_train['ENG_MDL_CD_080110'])
# 		data_train['ENG_MFG_CD_grp1']=(data_train['ENG_MFG_CD_238']+data_train['ENG_MFG_CD_120']+data_train['ENG_MFG_CD_204']+data_train['ENG_MFG_CD_060'])
# 		data_train['FRNT_TYRE_SIZE_CD_grp1']=(data_train['FRNT_TYRE_SIZE_CD_48']+data_train['FRNT_TYRE_SIZE_CD_31']+data_train['FRNT_TYRE_SIZE_CD_49']+data_train['FRNT_TYRE_SIZE_CD_42']+data_train['FRNT_TYRE_SIZE_CD_43'])
# 		data_train['FRNT_TYRE_SIZE_CD_grp2']=(data_train['FRNT_TYRE_SIZE_CD_54']+data_train['FRNT_TYRE_SIZE_CD_48'])
# 		data_train['FRNT_TYRE_SIZE_CD_grp3']=(data_train['FRNT_TYRE_SIZE_CD_433']+data_train['FRNT_TYRE_SIZE_CD_459']+data_train['FRNT_TYRE_SIZE_CD_28']+data_train['FRNT_TYRE_SIZE_CD_434']+data_train['FRNT_TYRE_SIZE_CD_449']+data_train['FRNT_TYRE_SIZE_CD_21']+data_train['FRNT_TYRE_SIZE_CD_85']+data_train['FRNT_TYRE_SIZE_CD_453']+data_train['FRNT_TYRE_SIZE_CD_406']+data_train['FRNT_TYRE_SIZE_CD_20']+data_train['FRNT_TYRE_SIZE_CD_11']+data_train['FRNT_TYRE_SIZE_CD_417']+data_train['FRNT_TYRE_SIZE_CD_5']+data_train['FRNT_TYRE_SIZE_CD_3']+data_train['FRNT_TYRE_SIZE_CD_4']+data_train['FRNT_TYRE_SIZE_CD_81'])
# 		data_train['MAK_NM_grp1']=(data_train['MAK_NM_FISKER_AUTOMOTIVE']+data_train['MAK_NM_MAYBACH']+data_train['MAK_NM_MCLAREN_AUTOMOTIVE']+data_train['MAK_NM_ROLLS-ROYCE']+data_train['MAK_NM_DAIHATSU']+data_train['MAK_NM_ALFA_ROMEO']+data_train['MAK_NM_DATSUN']+data_train['MAK_NM_LOTUS'])
# 		data_train['MAK_NM_grp2']=(data_train['MAK_NM_MERCEDES-BENZ']+data_train['MAK_NM_LEXUS']+data_train['MAK_NM_PONTIAC'])
# 		data_train['MFG_DESC_grp1']=(data_train['MFG_DESC_MAZDA_MOTOR_CORPORATIO']+data_train['MFG_DESC_FISKER_AUTOMOTIVE_INC.']+data_train['MFG_DESC_MERCEDES-BENZ_USA_LLC']+data_train['MFG_DESC_MCLAREN_AUTOMOTIVE'])
# 		data_train['NC_TireSize4_grp1']=(data_train['NC_TireSize4_P235/45R18']+data_train['NC_TireSize4_235/50R18']+data_train['NC_TireSize4_P255/50R19'])
# 		data_train['NC_TireSize4_grp2']=(data_train['NC_TireSize4_P235/60R18']+data_train['NC_TireSize4_P225/55R18'])
# 		data_train['NC_TireSize4_grp3']=(data_train['NC_TireSize4_P215/60R16']+data_train['NC_TireSize4_P265/50R20'])
# 		data_train['NC_TireSize4_grp4']=(data_train['NC_TireSize4_225/45R17']+data_train['NC_TireSize4_P235/50R18'])
# 		data_train['NC_HD_INJ_C_D_grp1']=(data_train['NC_HD_INJ_C_D_463']+data_train['NC_HD_INJ_C_D_1051']+data_train['NC_HD_INJ_C_D_1254']+data_train['NC_HD_INJ_C_D_524']+data_train['NC_HD_INJ_C_D_1564']+data_train['NC_HD_INJ_C_D_1026']+data_train['NC_HD_INJ_C_D_1238']+data_train['NC_HD_INJ_C_D_407']+data_train['NC_HD_INJ_C_D_522']+data_train['NC_HD_INJ_C_D_996']+data_train['NC_HD_INJ_C_D_707']+data_train['NC_HD_INJ_C_D_779']+data_train['NC_HD_INJ_C_D_900']+data_train['NC_HD_INJ_C_D_873']+data_train['NC_HD_INJ_C_D_908']+data_train['NC_HD_INJ_C_D_1024']+data_train['NC_HD_INJ_C_D_898']+data_train['NC_HD_INJ_C_D_867']+data_train['NC_HD_INJ_C_D_744']+data_train['NC_HD_INJ_C_D_750']+data_train['NC_HD_INJ_C_D_546']+data_train['NC_HD_INJ_C_D_1088']+data_train['NC_HD_INJ_C_D_1068']+data_train['NC_HD_INJ_C_D_1459']+data_train['NC_HD_INJ_C_D_528']+data_train['NC_HD_INJ_C_D_1036']+data_train['NC_HD_INJ_C_D_535']+data_train['NC_HD_INJ_C_D_793']+data_train['NC_HD_INJ_C_D_846']+data_train['NC_HD_INJ_C_D_626']+data_train['NC_HD_INJ_C_D_587']+data_train['NC_HD_INJ_C_D_2021']+data_train['NC_HD_INJ_C_D_853']+data_train['NC_HD_INJ_C_D_895']+data_train['NC_HD_INJ_C_D_212']+data_train['NC_HD_INJ_C_D_584']+data_train['NC_HD_INJ_C_D_654']+data_train['NC_HD_INJ_C_D_557']+data_train['NC_HD_INJ_C_D_818']+data_train['NC_HD_INJ_C_D_735']+data_train['NC_HD_INJ_C_D_719']+data_train['NC_HD_INJ_C_D_808']+data_train['NC_HD_INJ_C_D_1182']+data_train['NC_HD_INJ_C_D_433']+data_train['NC_HD_INJ_C_D_960']+data_train['NC_HD_INJ_C_D_516']+data_train['NC_HD_INJ_C_D_367']+data_train['NC_HD_INJ_C_D_622']+data_train['NC_HD_INJ_C_D_863']+data_train['NC_HD_INJ_C_D_710']+data_train['NC_HD_INJ_C_D_930']+data_train['NC_HD_INJ_C_D_1345']+data_train['NC_HD_INJ_C_D_531']+data_train['NC_HD_INJ_C_D_1138']+data_train['NC_HD_INJ_C_D_834']+data_train['NC_HD_INJ_C_D_765'])
# 		data_train['NC_HD_INJ_C_D_grp2']=(data_train['NC_HD_INJ_C_D_338']+data_train['NC_HD_INJ_C_D_519']+data_train['NC_HD_INJ_C_D_664']+data_train['NC_HD_INJ_C_D_640']+data_train['NC_HD_INJ_C_D_998']+data_train['NC_HD_INJ_C_D_236'])
# 		data_train['NC_HD_INJ_C_D_grp3']=(data_train['NC_HD_INJ_C_D_249']+data_train['NC_HD_INJ_C_D_371']+data_train['NC_HD_INJ_C_D_602']+data_train['NC_HD_INJ_C_D_306'])
# 		data_train['NC_HD_INJ_C_D_grp4']=(data_train['NC_HD_INJ_C_D_948']+data_train['NC_HD_INJ_C_D_474']+data_train['NC_HD_INJ_C_D_566']+data_train['NC_HD_INJ_C_D_383'])
# 		data_train['REAR_TIRE_SIZE_CD_grp1']=(data_train['REAR_TIRE_SIZE_CD_466']+data_train['REAR_TIRE_SIZE_CD_93']+data_train['REAR_TIRE_SIZE_CD_16']+data_train['REAR_TIRE_SIZE_CD_85']+data_train['REAR_TIRE_SIZE_CD_97'])
# 		data_train['WHL_DRVN_CNT_Tr__']=data_train['WHL_DRVN_CNT_Tr_.']	
# 		data_train['MAK_NM_ROLLS_ROYCE']=data_train['MAK_NM_ROLLS-ROYCE']


# 		data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_CPE2D4X4']+data_test['bodyStyleCd_CAMPER']+data_test['bodyStyleCd_TRK_4X2']+data_test['bodyStyleCd_CPE4X44D']+data_test['bodyStyleCd_CNV4X42D']+data_test['bodyStyleCd_WAG5D4X4']+data_test['bodyStyleCd_WAGON_2D']+data_test['bodyStyleCd_WAG5D4X2']+data_test['bodyStyleCd_WAG4D4X2']+data_test['bodyStyleCd_VAN']+data_test['bodyStyleCd_SP_HCHBK']+data_test['bodyStyleCd_SED4D4X4']+data_test['bodyStyleCd_SED2D4X4']+data_test['bodyStyleCd_SED4X44D']+data_test['bodyStyleCd_CPE_2+2']+data_test['bodyStyleCd_CP/RDSTR']+data_test['bodyStyleCd_LIMO']+data_test['bodyStyleCd_VAN_4X4']+data_test['bodyStyleCd_ROADSTER']+data_test['bodyStyleCd_MPV_4X2']+data_test['bodyStyleCd_SPORT_CP']+data_test['bodyStyleCd_TRK_4X4']+data_test['bodyStyleCd_HRDTP_2D']+data_test['bodyStyleCd_HCH2D4X4']+data_test['bodyStyleCd_HCH5D4X4']+data_test['bodyStyleCd_CABRIOLE']+data_test['bodyStyleCd_SP_COUPE']+data_test['bodyStyleCd_TARGA']+data_test['bodyStyleCd_COUPE']+data_test['bodyStyleCd_HCH3D4X4'])
# 		data_test['cko_eng_cylinders_grp1']=(data_test['cko_eng_cylinders_4']+data_test['cko_eng_cylinders_5']+data_test['cko_eng_cylinders_.']+data_test['cko_eng_cylinders_3'])
# 		data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_050016']+data_test['ENG_MDL_CD_050014']+data_test['ENG_MDL_CD_080015']+data_test['ENG_MDL_CD_238004']+data_test['ENG_MDL_CD_120020']+data_test['ENG_MDL_CD_207065']+data_test['ENG_MDL_CD_239003']+data_test['ENG_MDL_CD_207020']+data_test['ENG_MDL_CD_070024']+data_test['ENG_MDL_CD_070009'])
# 		data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_165051']+data_test['ENG_MDL_CD_080110'])
# 		data_test['ENG_MFG_CD_grp1']=(data_test['ENG_MFG_CD_238']+data_test['ENG_MFG_CD_120']+data_test['ENG_MFG_CD_204']+data_test['ENG_MFG_CD_060'])
# 		data_test['FRNT_TYRE_SIZE_CD_grp1']=(data_test['FRNT_TYRE_SIZE_CD_48']+data_test['FRNT_TYRE_SIZE_CD_31']+data_test['FRNT_TYRE_SIZE_CD_49']+data_test['FRNT_TYRE_SIZE_CD_42']+data_test['FRNT_TYRE_SIZE_CD_43'])
# 		data_test['FRNT_TYRE_SIZE_CD_grp2']=(data_test['FRNT_TYRE_SIZE_CD_54']+data_test['FRNT_TYRE_SIZE_CD_48'])
# 		data_test['FRNT_TYRE_SIZE_CD_grp3']=(data_test['FRNT_TYRE_SIZE_CD_433']+data_test['FRNT_TYRE_SIZE_CD_459']+data_test['FRNT_TYRE_SIZE_CD_28']+data_test['FRNT_TYRE_SIZE_CD_434']+data_test['FRNT_TYRE_SIZE_CD_449']+data_test['FRNT_TYRE_SIZE_CD_21']+data_test['FRNT_TYRE_SIZE_CD_85']+data_test['FRNT_TYRE_SIZE_CD_453']+data_test['FRNT_TYRE_SIZE_CD_406']+data_test['FRNT_TYRE_SIZE_CD_20']+data_test['FRNT_TYRE_SIZE_CD_11']+data_test['FRNT_TYRE_SIZE_CD_417']+data_test['FRNT_TYRE_SIZE_CD_5']+data_test['FRNT_TYRE_SIZE_CD_3']+data_test['FRNT_TYRE_SIZE_CD_4']+data_test['FRNT_TYRE_SIZE_CD_81'])
# 		data_test['MAK_NM_grp1']=(data_test['MAK_NM_FISKER_AUTOMOTIVE']+data_test['MAK_NM_MAYBACH']+data_test['MAK_NM_MCLAREN_AUTOMOTIVE']+data_test['MAK_NM_ROLLS-ROYCE']+data_test['MAK_NM_DAIHATSU']+data_test['MAK_NM_ALFA_ROMEO']+data_test['MAK_NM_DATSUN']+data_test['MAK_NM_LOTUS'])
# 		data_test['MAK_NM_grp2']=(data_test['MAK_NM_MERCEDES-BENZ']+data_test['MAK_NM_LEXUS']+data_test['MAK_NM_PONTIAC'])
# 		data_test['MFG_DESC_grp1']=(data_test['MFG_DESC_MAZDA_MOTOR_CORPORATIO']+data_test['MFG_DESC_FISKER_AUTOMOTIVE_INC.']+data_test['MFG_DESC_MERCEDES-BENZ_USA_LLC']+data_test['MFG_DESC_MCLAREN_AUTOMOTIVE'])
# 		data_test['NC_TireSize4_grp1']=(data_test['NC_TireSize4_P235/45R18']+data_test['NC_TireSize4_235/50R18']+data_test['NC_TireSize4_P255/50R19'])
# 		data_test['NC_TireSize4_grp2']=(data_test['NC_TireSize4_P235/60R18']+data_test['NC_TireSize4_P225/55R18'])
# 		data_test['NC_TireSize4_grp3']=(data_test['NC_TireSize4_P215/60R16']+data_test['NC_TireSize4_P265/50R20'])
# 		data_test['NC_TireSize4_grp4']=(data_test['NC_TireSize4_225/45R17']+data_test['NC_TireSize4_P235/50R18'])
# 		data_test['NC_HD_INJ_C_D_grp1']=(data_test['NC_HD_INJ_C_D_463']+data_test['NC_HD_INJ_C_D_1051']+data_test['NC_HD_INJ_C_D_1254']+data_test['NC_HD_INJ_C_D_524']+data_test['NC_HD_INJ_C_D_1564']+data_test['NC_HD_INJ_C_D_1026']+data_test['NC_HD_INJ_C_D_1238']+data_test['NC_HD_INJ_C_D_407']+data_test['NC_HD_INJ_C_D_522']+data_test['NC_HD_INJ_C_D_996']+data_test['NC_HD_INJ_C_D_707']+data_test['NC_HD_INJ_C_D_779']+data_test['NC_HD_INJ_C_D_900']+data_test['NC_HD_INJ_C_D_873']+data_test['NC_HD_INJ_C_D_908']+data_test['NC_HD_INJ_C_D_1024']+data_test['NC_HD_INJ_C_D_898']+data_test['NC_HD_INJ_C_D_867']+data_test['NC_HD_INJ_C_D_744']+data_test['NC_HD_INJ_C_D_750']+data_test['NC_HD_INJ_C_D_546']+data_test['NC_HD_INJ_C_D_1088']+data_test['NC_HD_INJ_C_D_1068']+data_test['NC_HD_INJ_C_D_1459']+data_test['NC_HD_INJ_C_D_528']+data_test['NC_HD_INJ_C_D_1036']+data_test['NC_HD_INJ_C_D_535']+data_test['NC_HD_INJ_C_D_793']+data_test['NC_HD_INJ_C_D_846']+data_test['NC_HD_INJ_C_D_626']+data_test['NC_HD_INJ_C_D_587']+data_test['NC_HD_INJ_C_D_2021']+data_test['NC_HD_INJ_C_D_853']+data_test['NC_HD_INJ_C_D_895']+data_test['NC_HD_INJ_C_D_212']+data_test['NC_HD_INJ_C_D_584']+data_test['NC_HD_INJ_C_D_654']+data_test['NC_HD_INJ_C_D_557']+data_test['NC_HD_INJ_C_D_818']+data_test['NC_HD_INJ_C_D_735']+data_test['NC_HD_INJ_C_D_719']+data_test['NC_HD_INJ_C_D_808']+data_test['NC_HD_INJ_C_D_1182']+data_test['NC_HD_INJ_C_D_433']+data_test['NC_HD_INJ_C_D_960']+data_test['NC_HD_INJ_C_D_516']+data_test['NC_HD_INJ_C_D_367']+data_test['NC_HD_INJ_C_D_622']+data_test['NC_HD_INJ_C_D_863']+data_test['NC_HD_INJ_C_D_710']+data_test['NC_HD_INJ_C_D_930']+data_test['NC_HD_INJ_C_D_1345']+data_test['NC_HD_INJ_C_D_531']+data_test['NC_HD_INJ_C_D_1138']+data_test['NC_HD_INJ_C_D_834']+data_test['NC_HD_INJ_C_D_765'])
# 		data_test['NC_HD_INJ_C_D_grp2']=(data_test['NC_HD_INJ_C_D_338']+data_test['NC_HD_INJ_C_D_519']+data_test['NC_HD_INJ_C_D_664']+data_test['NC_HD_INJ_C_D_640']+data_test['NC_HD_INJ_C_D_998']+data_test['NC_HD_INJ_C_D_236'])
# 		data_test['NC_HD_INJ_C_D_grp3']=(data_test['NC_HD_INJ_C_D_249']+data_test['NC_HD_INJ_C_D_371']+data_test['NC_HD_INJ_C_D_602']+data_test['NC_HD_INJ_C_D_306'])
# 		data_test['NC_HD_INJ_C_D_grp4']=(data_test['NC_HD_INJ_C_D_948']+data_test['NC_HD_INJ_C_D_474']+data_test['NC_HD_INJ_C_D_566']+data_test['NC_HD_INJ_C_D_383'])
# 		data_test['REAR_TIRE_SIZE_CD_grp1']=(data_test['REAR_TIRE_SIZE_CD_466']+data_test['REAR_TIRE_SIZE_CD_93']+data_test['REAR_TIRE_SIZE_CD_16']+data_test['REAR_TIRE_SIZE_CD_85']+data_test['REAR_TIRE_SIZE_CD_97'])
# 		data_test['WHL_DRVN_CNT_Tr__']=data_test['WHL_DRVN_CNT_Tr_.']
# 		data_test['MAK_NM_ROLLS_ROYCE']=data_test['MAK_NM_ROLLS-ROYCE']


# 	formula=formulabase+forms[i]
# 	if not os.path.exists('glm'+str(i)+'/'):
# 		os.makedirs('glm'+str(i)+'/')
# 	prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm'+str(i)+'/',0)




# Step 6
# Group CATFEATURES  and finalize model

# CONTFEATURES = []
# CATFEATURES = ['BODY_STYLE_CD','bodyStyleCd','cko_eng_cylinders','ENG_MDL_CD','ENG_MFG_CD','FRNT_TYRE_SIZE_CD','MAK_NM','MFG_DESC','NC_HD_INJ_C_D','NC_TireSize4','NC_TractionControl','REAR_TIRE_SIZE_CD','RSTRNT_TYP_CD','TLT_STRNG_WHL_OPT_CD','TRK_TNG_RAT_CD','vinmasterPerformanceCd','WHL_DRVN_CNT_Tr','cko_esc','cko_turbo_super','TRK_FRNT_AXL_CD','TRK_REAR_AXL_CD']
# # Include only catfeatures that are needed for the attempted GLMs

# # Rename problematic dummy columns

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


# data_train['WHL_BAS_LNGST_INCHS_lt95']=np.where((data_train['WHL_BAS_LNGST_INCHS']>=0) & (data_train['WHL_BAS_LNGST_INCHS']<95), 1, 0)
# data_train['WHL_BAS_SHRST_INCHS_lt95']=np.where((data_train['WHL_BAS_SHRST_INCHS']>=0) & (data_train['WHL_BAS_SHRST_INCHS']<95), 1, 0)
# data_train['door_cnt_bin_lt2']=np.where(data_train['door_cnt_bin']<2, 1, 0)
# # data_train['cko_eng_cylinders_numeric_eq2']=np.where(data_train['cko_eng_cylinders_numeric']==2, 1, 0)
# data_test['WHL_BAS_LNGST_INCHS_lt95']=np.where((data_test['WHL_BAS_LNGST_INCHS']>=0) & (data_test['WHL_BAS_LNGST_INCHS']<95), 1, 0)
# data_test['WHL_BAS_SHRST_INCHS_lt95']=np.where((data_test['WHL_BAS_SHRST_INCHS']>=0) & (data_test['WHL_BAS_SHRST_INCHS']<95), 1, 0)
# data_test['door_cnt_bin_lt2']=np.where(data_test['door_cnt_bin']<2, 1, 0)
# # data_test['cko_eng_cylinders_numeric_eq2']=np.where(data_test['cko_eng_cylinders_numeric']==2, 1, 0)
# data_train['length_VM_bw220and250']=np.where((data_train['length_VM']>=220) & (data_train['length_VM']<250), 1, 0)
# data_test['length_VM_bw220and250']=np.where((data_test['length_VM']>=220) & (data_test['length_VM']<250), 1, 0)
# data_train['cko_wheelbase_max_spline100']=np.minimum(data_train['cko_wheelbase_max'],100)
# data_test['cko_wheelbase_max_spline100']=np.minimum(data_test['cko_wheelbase_max'],100)
# data_train['EA_CURB_WEIGHT_35bw45']=np.where((3500<=data_train['EA_CURB_WEIGHT']) & (data_train['EA_CURB_WEIGHT']<4500),1,0)
# data_test['EA_CURB_WEIGHT_35bw45']=np.where((3500<=data_test['EA_CURB_WEIGHT']) & (data_test['EA_CURB_WEIGHT']<4500),1,0)
# data_train['EA_TIP_OVER_STABILITY_RATIO_ge15']=np.where(data_train['EA_TIP_OVER_STABILITY_RATIO']>=1.5,1,0)
# data_test['EA_TIP_OVER_STABILITY_RATIO_ge15']=np.where(data_test['EA_TIP_OVER_STABILITY_RATIO']>=1.5,1,0)
# data_train['ENG_DISPLCMNT_CI_380spline']=np.maximum(380,data_train['ENG_DISPLCMNT_CI'])
# data_test['ENG_DISPLCMNT_CI_380spline']=np.maximum(380,data_test['ENG_DISPLCMNT_CI']) 
# # data_train['enginesize_ge6']=np.where(data_train['enginesize']>=6,1,0)
# # data_test['enginesize_ge6']=np.where(data_test['enginesize']>=6,1,0)
# # data_train['cko_eng_cylinders_ge8']=np.where(data_train['cko_eng_cylinders']>=8, 1, 0)
# # data_test['cko_eng_cylinders_ge8']=np.where(data_test['cko_eng_cylinders']>=8, 1, 0)
# # data_train['cko_eng_cylinders_numeric_eq6']=np.where(data_train['cko_eng_cylinders_numeric']==6, 1, 0)
# # data_test['cko_eng_cylinders_numeric_eq6']=np.where(data_test['cko_eng_cylinders_numeric']==6, 1, 0)

# data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_CPE2D4X4']+data_train['bodyStyleCd_CAMPER']+data_train['bodyStyleCd_TRK_4X2']+data_train['bodyStyleCd_CPE4X44D']+data_train['bodyStyleCd_CNV4X42D']+data_train['bodyStyleCd_WAG5D4X4']+data_train['bodyStyleCd_WAGON_2D']+data_train['bodyStyleCd_WAG5D4X2']+data_train['bodyStyleCd_WAG4D4X2']+data_train['bodyStyleCd_VAN']+data_train['bodyStyleCd_SP_HCHBK']+data_train['bodyStyleCd_SED4D4X4']+data_train['bodyStyleCd_SED2D4X4']+data_train['bodyStyleCd_SED4X44D']+data_train['bodyStyleCd_CPE_2+2']+data_train['bodyStyleCd_CP/RDSTR']+data_train['bodyStyleCd_LIMO']+data_train['bodyStyleCd_VAN_4X4']+data_train['bodyStyleCd_ROADSTER']+data_train['bodyStyleCd_MPV_4X2']+data_train['bodyStyleCd_SPORT_CP']+data_train['bodyStyleCd_TRK_4X4']+data_train['bodyStyleCd_HRDTP_2D']+data_train['bodyStyleCd_HCH2D4X4']+data_train['bodyStyleCd_HCH5D4X4']+data_train['bodyStyleCd_CABRIOLE']+data_train['bodyStyleCd_SP_COUPE']+data_train['bodyStyleCd_TARGA']+data_train['bodyStyleCd_COUPE']+data_train['bodyStyleCd_HCH3D4X4'])
# data_train['cko_eng_cylinders_grp1']=(data_train['cko_eng_cylinders_4']+data_train['cko_eng_cylinders_5']+data_train['cko_eng_cylinders_.']+data_train['cko_eng_cylinders_3'])
# data_train['FRNT_TYRE_SIZE_CD_grp1']=(data_train['FRNT_TYRE_SIZE_CD_48']+data_train['FRNT_TYRE_SIZE_CD_31']+data_train['FRNT_TYRE_SIZE_CD_49']+data_train['FRNT_TYRE_SIZE_CD_42']+data_train['FRNT_TYRE_SIZE_CD_43'])
# data_train['FRNT_TYRE_SIZE_CD_grp2']=(data_train['FRNT_TYRE_SIZE_CD_54']+data_train['FRNT_TYRE_SIZE_CD_48']+data_train['FRNT_TYRE_SIZE_CD_62'])
# data_train['FRNT_TYRE_SIZE_CD_grp3']=(data_train['FRNT_TYRE_SIZE_CD_433']+data_train['FRNT_TYRE_SIZE_CD_459']+data_train['FRNT_TYRE_SIZE_CD_28']+data_train['FRNT_TYRE_SIZE_CD_434']+data_train['FRNT_TYRE_SIZE_CD_449']+data_train['FRNT_TYRE_SIZE_CD_21']+data_train['FRNT_TYRE_SIZE_CD_85']+data_train['FRNT_TYRE_SIZE_CD_453']+data_train['FRNT_TYRE_SIZE_CD_406']+data_train['FRNT_TYRE_SIZE_CD_20']+data_train['FRNT_TYRE_SIZE_CD_11']+data_train['FRNT_TYRE_SIZE_CD_417']+data_train['FRNT_TYRE_SIZE_CD_5']+data_train['FRNT_TYRE_SIZE_CD_3']+data_train['FRNT_TYRE_SIZE_CD_4']+data_train['FRNT_TYRE_SIZE_CD_81'])
# # data_train['MAK_NM_grp1']=(data_train['MAK_NM_FISKER_AUTOMOTIVE']+data_train['MAK_NM_MAYBACH']+data_train['MAK_NM_MCLAREN_AUTOMOTIVE']+data_train['MAK_NM_ROLLS-ROYCE']+data_train['MAK_NM_DAIHATSU']+data_train['MAK_NM_ALFA_ROMEO']+data_train['MAK_NM_DATSUN']+data_train['MAK_NM_LOTUS'])
# # data_train['MAK_NM_grp2']=(data_train['MAK_NM_MERCEDES-BENZ']+data_train['MAK_NM_LEXUS']+data_train['MAK_NM_PONTIAC'])
# data_train['MFG_DESC_grp1']=(data_train['MFG_DESC_MAZDA_MOTOR_CORPORATIO']+data_train['MFG_DESC_FISKER_AUTOMOTIVE_INC.']+data_train['MFG_DESC_MERCEDES-BENZ_USA_LLC']+data_train['MFG_DESC_MCLAREN_AUTOMOTIVE'])
# data_train['NC_TireSize4_grp1']=(data_train['NC_TireSize4_P235/45R18']+data_train['NC_TireSize4_235/50R18']+data_train['NC_TireSize4_P255/50R19'])
# # data_train['NC_TireSize4_grp2']=(data_train['NC_TireSize4_P235/60R18']+data_train['NC_TireSize4_P225/55R18'])
# data_train['NC_TireSize4_grp2']=(data_train['NC_TireSize4_P215/60R16']+data_train['NC_TireSize4_P265/50R20'])
# # data_train['NC_TireSize4_grp4']=(data_train['NC_TireSize4_225/45R17']+data_train['NC_TireSize4_P235/50R18'])
# data_train['NC_HD_INJ_C_D_grp1']=(data_train['NC_HD_INJ_C_D_463']+data_train['NC_HD_INJ_C_D_1051']+data_train['NC_HD_INJ_C_D_1254']+data_train['NC_HD_INJ_C_D_524']+data_train['NC_HD_INJ_C_D_1564']+data_train['NC_HD_INJ_C_D_1026']+data_train['NC_HD_INJ_C_D_1238']+data_train['NC_HD_INJ_C_D_407']+data_train['NC_HD_INJ_C_D_522']+data_train['NC_HD_INJ_C_D_996']+data_train['NC_HD_INJ_C_D_707']+data_train['NC_HD_INJ_C_D_779']+data_train['NC_HD_INJ_C_D_900']+data_train['NC_HD_INJ_C_D_873']+data_train['NC_HD_INJ_C_D_908']+data_train['NC_HD_INJ_C_D_1024']+data_train['NC_HD_INJ_C_D_898']+data_train['NC_HD_INJ_C_D_867']+data_train['NC_HD_INJ_C_D_744']+data_train['NC_HD_INJ_C_D_750']+data_train['NC_HD_INJ_C_D_546']+data_train['NC_HD_INJ_C_D_1088']+data_train['NC_HD_INJ_C_D_1068']+data_train['NC_HD_INJ_C_D_1459']+data_train['NC_HD_INJ_C_D_528']+data_train['NC_HD_INJ_C_D_1036']+data_train['NC_HD_INJ_C_D_535']+data_train['NC_HD_INJ_C_D_793']+data_train['NC_HD_INJ_C_D_846']+data_train['NC_HD_INJ_C_D_626']+data_train['NC_HD_INJ_C_D_587']+data_train['NC_HD_INJ_C_D_2021']+data_train['NC_HD_INJ_C_D_853']+data_train['NC_HD_INJ_C_D_895']+data_train['NC_HD_INJ_C_D_212']+data_train['NC_HD_INJ_C_D_584']+data_train['NC_HD_INJ_C_D_654']+data_train['NC_HD_INJ_C_D_557']+data_train['NC_HD_INJ_C_D_818']+data_train['NC_HD_INJ_C_D_735']+data_train['NC_HD_INJ_C_D_719']+data_train['NC_HD_INJ_C_D_808']+data_train['NC_HD_INJ_C_D_1182']+data_train['NC_HD_INJ_C_D_433']+data_train['NC_HD_INJ_C_D_960']+data_train['NC_HD_INJ_C_D_516']+data_train['NC_HD_INJ_C_D_367']+data_train['NC_HD_INJ_C_D_622']+data_train['NC_HD_INJ_C_D_863']+data_train['NC_HD_INJ_C_D_710']+data_train['NC_HD_INJ_C_D_930']+data_train['NC_HD_INJ_C_D_1345']+data_train['NC_HD_INJ_C_D_531']+data_train['NC_HD_INJ_C_D_1138']+data_train['NC_HD_INJ_C_D_834']+data_train['NC_HD_INJ_C_D_765'])
# data_train['NC_HD_INJ_C_D_grp2']=(data_train['NC_HD_INJ_C_D_338']+data_train['NC_HD_INJ_C_D_519']+data_train['NC_HD_INJ_C_D_664']+data_train['NC_HD_INJ_C_D_640']+data_train['NC_HD_INJ_C_D_998']+data_train['NC_HD_INJ_C_D_236'])
# data_train['NC_HD_INJ_C_D_grp3']=(data_train['NC_HD_INJ_C_D_249']+data_train['NC_HD_INJ_C_D_371']+data_train['NC_HD_INJ_C_D_602']+data_train['NC_HD_INJ_C_D_306']+data_train['NC_HD_INJ_C_D_948']+data_train['NC_HD_INJ_C_D_474']+data_train['NC_HD_INJ_C_D_566']+data_train['NC_HD_INJ_C_D_383'])
# # data_train['REAR_TIRE_SIZE_CD_grp1']=(data_train['REAR_TIRE_SIZE_CD_466']+data_train['REAR_TIRE_SIZE_CD_93']+data_train['REAR_TIRE_SIZE_CD_16']+data_train['REAR_TIRE_SIZE_CD_85']+data_train['REAR_TIRE_SIZE_CD_97'])
# # data_train['WHL_DRVN_CNT_Tr__']=data_train['WHL_DRVN_CNT_Tr_.']	
# # data_train['MAK_NM_ROLLS_ROYCE']=data_train['MAK_NM_ROLLS-ROYCE']

# data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_CPE2D4X4']+data_test['bodyStyleCd_CAMPER']+data_test['bodyStyleCd_TRK_4X2']+data_test['bodyStyleCd_CPE4X44D']+data_test['bodyStyleCd_CNV4X42D']+data_test['bodyStyleCd_WAG5D4X4']+data_test['bodyStyleCd_WAGON_2D']+data_test['bodyStyleCd_WAG5D4X2']+data_test['bodyStyleCd_WAG4D4X2']+data_test['bodyStyleCd_VAN']+data_test['bodyStyleCd_SP_HCHBK']+data_test['bodyStyleCd_SED4D4X4']+data_test['bodyStyleCd_SED2D4X4']+data_test['bodyStyleCd_SED4X44D']+data_test['bodyStyleCd_CPE_2+2']+data_test['bodyStyleCd_CP/RDSTR']+data_test['bodyStyleCd_LIMO']+data_test['bodyStyleCd_VAN_4X4']+data_test['bodyStyleCd_ROADSTER']+data_test['bodyStyleCd_MPV_4X2']+data_test['bodyStyleCd_SPORT_CP']+data_test['bodyStyleCd_TRK_4X4']+data_test['bodyStyleCd_HRDTP_2D']+data_test['bodyStyleCd_HCH2D4X4']+data_test['bodyStyleCd_HCH5D4X4']+data_test['bodyStyleCd_CABRIOLE']+data_test['bodyStyleCd_SP_COUPE']+data_test['bodyStyleCd_TARGA']+data_test['bodyStyleCd_COUPE']+data_test['bodyStyleCd_HCH3D4X4'])
# data_test['cko_eng_cylinders_grp1']=(data_test['cko_eng_cylinders_4']+data_test['cko_eng_cylinders_5']+data_test['cko_eng_cylinders_.']+data_test['cko_eng_cylinders_3'])
# data_test['FRNT_TYRE_SIZE_CD_grp1']=(data_test['FRNT_TYRE_SIZE_CD_48']+data_test['FRNT_TYRE_SIZE_CD_31']+data_test['FRNT_TYRE_SIZE_CD_49']+data_test['FRNT_TYRE_SIZE_CD_42']+data_test['FRNT_TYRE_SIZE_CD_43'])
# data_test['FRNT_TYRE_SIZE_CD_grp2']=(data_test['FRNT_TYRE_SIZE_CD_54']+data_test['FRNT_TYRE_SIZE_CD_48']+data_test['FRNT_TYRE_SIZE_CD_62'])
# data_test['FRNT_TYRE_SIZE_CD_grp3']=(data_test['FRNT_TYRE_SIZE_CD_433']+data_test['FRNT_TYRE_SIZE_CD_459']+data_test['FRNT_TYRE_SIZE_CD_28']+data_test['FRNT_TYRE_SIZE_CD_434']+data_test['FRNT_TYRE_SIZE_CD_449']+data_test['FRNT_TYRE_SIZE_CD_21']+data_test['FRNT_TYRE_SIZE_CD_85']+data_test['FRNT_TYRE_SIZE_CD_453']+data_test['FRNT_TYRE_SIZE_CD_406']+data_test['FRNT_TYRE_SIZE_CD_20']+data_test['FRNT_TYRE_SIZE_CD_11']+data_test['FRNT_TYRE_SIZE_CD_417']+data_test['FRNT_TYRE_SIZE_CD_5']+data_test['FRNT_TYRE_SIZE_CD_3']+data_test['FRNT_TYRE_SIZE_CD_4']+data_test['FRNT_TYRE_SIZE_CD_81'])
# # data_test['MAK_NM_grp1']=(data_test['MAK_NM_FISKER_AUTOMOTIVE']+data_test['MAK_NM_MAYBACH']+data_test['MAK_NM_MCLAREN_AUTOMOTIVE']+data_test['MAK_NM_ROLLS-ROYCE']+data_test['MAK_NM_DAIHATSU']+data_test['MAK_NM_ALFA_ROMEO']+data_test['MAK_NM_DATSUN']+data_test['MAK_NM_LOTUS'])
# # data_test['MAK_NM_grp2']=(data_test['MAK_NM_MERCEDES-BENZ']+data_test['MAK_NM_LEXUS']+data_test['MAK_NM_PONTIAC'])
# data_test['MFG_DESC_grp1']=(data_test['MFG_DESC_MAZDA_MOTOR_CORPORATIO']+data_test['MFG_DESC_FISKER_AUTOMOTIVE_INC.']+data_test['MFG_DESC_MERCEDES-BENZ_USA_LLC']+data_test['MFG_DESC_MCLAREN_AUTOMOTIVE'])
# data_test['NC_TireSize4_grp1']=(data_test['NC_TireSize4_P235/45R18']+data_test['NC_TireSize4_235/50R18']+data_test['NC_TireSize4_P255/50R19'])
# # data_test['NC_TireSize4_grp2']=(data_test['NC_TireSize4_P235/60R18']+data_test['NC_TireSize4_P225/55R18'])
# data_test['NC_TireSize4_grp2']=(data_test['NC_TireSize4_P215/60R16']+data_test['NC_TireSize4_P265/50R20'])
# # data_test['NC_TireSize4_grp4']=(data_test['NC_TireSize4_225/45R17']+data_test['NC_TireSize4_P235/50R18'])
# data_test['NC_HD_INJ_C_D_grp1']=(data_test['NC_HD_INJ_C_D_463']+data_test['NC_HD_INJ_C_D_1051']+data_test['NC_HD_INJ_C_D_1254']+data_test['NC_HD_INJ_C_D_524']+data_test['NC_HD_INJ_C_D_1564']+data_test['NC_HD_INJ_C_D_1026']+data_test['NC_HD_INJ_C_D_1238']+data_test['NC_HD_INJ_C_D_407']+data_test['NC_HD_INJ_C_D_522']+data_test['NC_HD_INJ_C_D_996']+data_test['NC_HD_INJ_C_D_707']+data_test['NC_HD_INJ_C_D_779']+data_test['NC_HD_INJ_C_D_900']+data_test['NC_HD_INJ_C_D_873']+data_test['NC_HD_INJ_C_D_908']+data_test['NC_HD_INJ_C_D_1024']+data_test['NC_HD_INJ_C_D_898']+data_test['NC_HD_INJ_C_D_867']+data_test['NC_HD_INJ_C_D_744']+data_test['NC_HD_INJ_C_D_750']+data_test['NC_HD_INJ_C_D_546']+data_test['NC_HD_INJ_C_D_1088']+data_test['NC_HD_INJ_C_D_1068']+data_test['NC_HD_INJ_C_D_1459']+data_test['NC_HD_INJ_C_D_528']+data_test['NC_HD_INJ_C_D_1036']+data_test['NC_HD_INJ_C_D_535']+data_test['NC_HD_INJ_C_D_793']+data_test['NC_HD_INJ_C_D_846']+data_test['NC_HD_INJ_C_D_626']+data_test['NC_HD_INJ_C_D_587']+data_test['NC_HD_INJ_C_D_2021']+data_test['NC_HD_INJ_C_D_853']+data_test['NC_HD_INJ_C_D_895']+data_test['NC_HD_INJ_C_D_212']+data_test['NC_HD_INJ_C_D_584']+data_test['NC_HD_INJ_C_D_654']+data_test['NC_HD_INJ_C_D_557']+data_test['NC_HD_INJ_C_D_818']+data_test['NC_HD_INJ_C_D_735']+data_test['NC_HD_INJ_C_D_719']+data_test['NC_HD_INJ_C_D_808']+data_test['NC_HD_INJ_C_D_1182']+data_test['NC_HD_INJ_C_D_433']+data_test['NC_HD_INJ_C_D_960']+data_test['NC_HD_INJ_C_D_516']+data_test['NC_HD_INJ_C_D_367']+data_test['NC_HD_INJ_C_D_622']+data_test['NC_HD_INJ_C_D_863']+data_test['NC_HD_INJ_C_D_710']+data_test['NC_HD_INJ_C_D_930']+data_test['NC_HD_INJ_C_D_1345']+data_test['NC_HD_INJ_C_D_531']+data_test['NC_HD_INJ_C_D_1138']+data_test['NC_HD_INJ_C_D_834']+data_test['NC_HD_INJ_C_D_765'])
# data_test['NC_HD_INJ_C_D_grp2']=(data_test['NC_HD_INJ_C_D_338']+data_test['NC_HD_INJ_C_D_519']+data_test['NC_HD_INJ_C_D_664']+data_test['NC_HD_INJ_C_D_640']+data_test['NC_HD_INJ_C_D_998']+data_test['NC_HD_INJ_C_D_236'])
# data_test['NC_HD_INJ_C_D_grp3']=(data_test['NC_HD_INJ_C_D_249']+data_test['NC_HD_INJ_C_D_371']+data_test['NC_HD_INJ_C_D_602']+data_test['NC_HD_INJ_C_D_306']+data_test['NC_HD_INJ_C_D_948']+data_test['NC_HD_INJ_C_D_474']+data_test['NC_HD_INJ_C_D_566']+data_test['NC_HD_INJ_C_D_383'])
# # data_test['REAR_TIRE_SIZE_CD_grp1']=(data_test['REAR_TIRE_SIZE_CD_466']+data_test['REAR_TIRE_SIZE_CD_93']+data_test['REAR_TIRE_SIZE_CD_16']+data_test['REAR_TIRE_SIZE_CD_85']+data_test['REAR_TIRE_SIZE_CD_97'])
# # data_test['WHL_DRVN_CNT_Tr__']=data_test['WHL_DRVN_CNT_Tr_.']
# # data_test['MAK_NM_ROLLS_ROYCE']=data_test['MAK_NM_ROLLS-ROYCE']

# data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_080100']+data_train['ENG_MDL_CD_050016']+data_train['ENG_MDL_CD_050014']+data_train['ENG_MDL_CD_080015']+data_train['ENG_MDL_CD_207065']+data_train['ENG_MDL_CD_207020']+data_train['ENG_MDL_CD_070024']+data_train['ENG_MDL_CD_070009']+data_train['ENG_MDL_CD_238004'] + data_train['ENG_MDL_CD_120020'])
# data_train['ENG_MFG_CD_grp1']=(data_train['ENG_MFG_CD_238'] + data_train['ENG_MFG_CD_120'])
# data_train['MAK_NM_grp1']=(data_train['MAK_NM_LOTUS'] + data_train['MAK_NM_MAYBACH'])
# data_train['MAK_NM_grp2']=(data_train['MAK_NM_ROLLS-ROYCE'] + data_train['MAK_NM_DAIHATSU'] + data_train['MAK_NM_ALFA_ROMEO'] + data_train['MAK_NM_DATSUN'])


# data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_080100']+data_test['ENG_MDL_CD_050016']+data_test['ENG_MDL_CD_050014']+data_test['ENG_MDL_CD_080015']+data_test['ENG_MDL_CD_207065']+data_test['ENG_MDL_CD_207020']+data_test['ENG_MDL_CD_070024']+data_test['ENG_MDL_CD_070009']+data_test['ENG_MDL_CD_238004'] + data_test['ENG_MDL_CD_120020'])
# data_test['ENG_MFG_CD_grp1']=(data_test['ENG_MFG_CD_238'] + data_test['ENG_MFG_CD_120'])
# data_test['MAK_NM_grp1']=(data_test['MAK_NM_LOTUS'] + data_test['MAK_NM_MAYBACH'])
# data_test['MAK_NM_grp2']=(data_test['MAK_NM_ROLLS-ROYCE'] + data_test['MAK_NM_DAIHATSU'] + data_test['MAK_NM_ALFA_ROMEO'] + data_test['MAK_NM_DATSUN'])


# formula = 'lossratio ~  WHL_BAS_LNGST_INCHS_lt95  + WHL_BAS_SHRST_INCHS_lt95 + length_VM_bw220and250 + door_cnt_bin_lt2  + ENG_MFG_CD_070 + WHL_DRVN_CNTMS  + cko_wheelbase_max_spline100 + EA_CURB_WEIGHT_35bw45 + EA_TIP_OVER_STABILITY_RATIO_ge15 + ENG_DISPLCMNT_CI_380spline     + BODY_STYLE_CD_HR + BODY_STYLE_CD_CV + bodyStyleCd_grp1 + cko_eng_cylinders_2  + cko_eng_cylinders_grp1   + ENG_MDL_CD_080006    + ENG_MFG_CD_070    + FRNT_TYRE_SIZE_CD_grp1 + FRNT_TYRE_SIZE_CD_grp2 + FRNT_TYRE_SIZE_CD_grp3  + MAK_NM_grp1 + MAK_NM_grp2 + MFG_DESC_grp1 + NC_TireSize4_grp1  + NC_TireSize4_grp2  + NC_TractionControl_S  + NC_HD_INJ_C_D_grp1 + NC_HD_INJ_C_D_grp2 + NC_HD_INJ_C_D_grp3  + RSTRNT_TYP_CD_Y    + TLT_STRNG_WHL_OPT_CD_U + TRK_TNG_RAT_CD_C  + TRK_TNG_RAT_CD_BC    + vinmasterPerformanceCd_3  + ENG_MDL_CD_grp1  + NADA_GVWC1MS    '
# #  Trying + ABS_BRK_AWO + cko_eng_cylindersMS + cko_ESC_N + cko_hp_enginesize_ratio_d_M + cko_hp_enginesize_ratioMS + cko_lengthMS + cko_max_gvwcMS + cko_turbo_super_Y + classCd_d3 + curbWeightMS + EA_ACCEL_RATE_0_TO_60MS + EA_ACCEL_TIME_0_TO_60 + ENG_HEAD_CNFG_CD_d2 + ENG_MFG_CD_d5 + enginesize_ge6 + FRNT_TYRE_SIZE_CD_d5 + heightMS + horsePowerMS + mak_ford + mfg_desc_U_FORD_GM + NADA_GVWC1MS + RSTRNT_TYP_CD_d2 + RSTRNT_TYP_CD_d6 + TRK_BRK_TYP_CD_d1 + TRK_FRNT_AXL_CD_N + TRK_REAR_AXL_CD_S + width_VMMS 

# if not os.path.exists('glm/'):
# 	os.makedirs('glm/')
# prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm/',0)


# # data_train['weight'] = data_train[WEIGHT]
# # data_train['predictedValue'] = prediction_glm_train*data_train['weight']
# # data_train['actualValue'] = data_train[LABEL]*data_train['weight']
# # data_test['weight'] = data_test[WEIGHT]
# # data_test['predictedValue'] = prediction_glm_test*data_test['weight']
# # data_test['actualValue'] = data_test[LABEL]*data_test['weight']

# # ml.actualvsfittedbyfactor(data_train,data_test,'WHL_BAS_LNGST_INCHS_lt95','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'WHL_BAS_SHRST_INCHS_lt95','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'door_cnt_bin_lt2','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'cko_eng_cylinders_numeric_eq2','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'length_VM_bw220and250','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'cko_wheelbase_max_spline100','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'EA_CURB_WEIGHT_35bw45','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'EA_TIP_OVER_STABILITY_RATIO_ge15','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'ENG_DISPLCMNT_CI_380spline','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'bodyStyleCd_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'cko_eng_cylinders_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'FRNT_TYRE_SIZE_CD_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'FRNT_TYRE_SIZE_CD_grp2','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'FRNT_TYRE_SIZE_CD_grp3','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'MFG_DESC_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'NC_TireSize4_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'NC_TireSize4_grp2','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'NC_HD_INJ_C_D_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'NC_HD_INJ_C_D_grp2','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'NC_HD_INJ_C_D_grp3','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'NC_HD_INJ_C_D_grp4','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'ENG_MDL_CD_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'ENG_MDL_CD_grp2','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'ENG_MFG_CD_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'MAK_NM_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'MAK_NM_grp2','glm/')


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

# IF 0<=WHL_BAS_LNGST_INCHS<95 THEN WHL_BAS_LNGST_INCHS_lt95=1; ELSE WHL_BAS_LNGST_INCHS_lt95=0;
# IF 0<=WHL_BAS_SHRST_INCHS<95 THEN WHL_BAS_SHRST_INCHS_lt95=1; ELSE WHL_BAS_SHRST_INCHS_lt95=0;
# IF 220<=length_VM<250 THEN length_VM_bw220and250=1; ELSE length_VM_bw220and250=0;
# IF door_cnt_bin<2 THEN door_cnt_bin_lt2=1; ELSE door_cnt_bin_lt2=0;
# IF ENG_MFG_CD = '070' THEN ENG_MFG_CD_070=1; ELSE ENG_MFG_CD_070=0;
# IF WHL_DRVN_CNT = . THEN WHL_DRVN_CNTMS=1; ELSE WHL_DRVN_CNTMS=0;
# IF cko_wheelbase_max=. THEN cko_wheelbase_max=-1;
# cko_wheelbase_max_spline100=MIN(cko_wheelbase_max,100);
# IF 3500<=EA_CURB_WEIGHT<4500 THEN EA_CURB_WEIGHT_35bw45=1; ELSE EA_CURB_WEIGHT_35bw45=0;
# IF EA_TIP_OVER_STABILITY_RATIO>=1.5 THEN EA_TIP_OVER_STABILITY_RATIO_ge15=1; ELSE EA_TIP_OVER_STABILITY_RATIO_ge15=0;
# ENG_DISPLCMNT_CI_380spline = MAX(380,ENG_DISPLCMNT_CI);
# IF BODY_STYLE_CD = 'HR' THEN BODY_STYLE_CD_HR=1; ELSE BODY_STYLE_CD_HR=0;
# IF BODY_STYLE_CD = 'CV' THEN BODY_STYLE_CD_CV=1; ELSE BODY_STYLE_CD_CV=0;
# IF bodyStyleCd IN ('CPE2D4X4','CAMPER','TRK_4X2','CPE4X44D','CNV4X42D','WAG5D4X4','WAGON_2D','WAG5D4X2','WAG4D4X2','VAN','SP_HCHBK','SED4D4X4','SED2D4X4','SED4X44D','CPE_2+2','CP/RDSTR','LIMO','VAN_4X4','ROADSTER','MPV_4X2','SPORT_CP','TRK_4X4','HRDTP_2D','HCH2D4X4','HCH5D4X4','CABRIOLE','SP_COUPE','TARGA','COUPE','HCH3D4X4') THEN bodyStyleCd_grp1=1; ELSE bodyStyleCd_grp1=0;
# IF cko_eng_cylinders = '2' THEN cko_eng_cylinders_2=1; ELSE cko_eng_cylinders_2=0;
# IF cko_eng_cylinders IN ('4','5','.','3') THEN cko_eng_cylinders_grp1=1; ELSE cko_eng_cylinders_grp1=0;
# IF ENG_MDL_CD = '080006' THEN ENG_MDL_CD_080006=1; ELSE ENG_MDL_CD_080006=0;
# IF FRNT_TYRE_SIZE_CD IN ('48','31','49','42','43') THEN FRNT_TYRE_SIZE_CD_grp1=1; ELSE FRNT_TYRE_SIZE_CD_grp1=0;
# IF FRNT_TYRE_SIZE_CD IN ('54','48','62') THEN FRNT_TYRE_SIZE_CD_grp2=1; ELSE FRNT_TYRE_SIZE_CD_grp2=0;
# IF FRNT_TYRE_SIZE_CD IN ('433','459','28','434','449','21','85','453','406','20','11','417','5','3','4','81') THEN FRNT_TYRE_SIZE_CD_grp3=1; ELSE FRNT_TYRE_SIZE_CD_grp3=0;
# IF MAK_NM IN ('LOTUS','MAYBACH') THEN MAK_NM_grp1=1; ELSE MAK_NM_grp1=0;
# IF MAK_NM IN ('ROLLS-ROYCE','DAIHATSU','ALFA ROMEO','DATSUN') THEN MAK_NM_grp2=1; ELSE MAK_NM_grp2=0;
# IF MFG_DESC IN ('MAZDA_MOTOR_CORPORATIO','FISKER_AUTOMOTIVE_INC.','MERCEDES-BENZ_USA_LLC','MCLAREN_AUTOMOTIVE') THEN MFG_DESC_grp1=1; ELSE MFG_DESC_grp1=0;
# IF NC_TireSize4 IN ('P235/45R18','235/50R18','P255/50R19') THEN NC_TireSize4_grp1=1; ELSE NC_TireSize4_grp1=0;
# IF NC_TireSize4 IN ('P215/60R16','P265/50R20') THEN NC_TireSize4_grp2=1; ELSE NC_TireSize4_grp2=0;
# IF NC_TractionControl ='S' THEN NC_TractionControl_S=1; ELSE NC_TractionControl_S=0;
# IF NC_HD_INJ_C_D IN ('463','1051','1254','524','1564','1026','1238','407','522','996','707','779','900','873','908','1024','898','867','744','750','546','1088','1068','1459','528','1036','535','793','846','626','587','2021','853','895','212','584','654','557','818','735','719','808','1182','433','960','516','367','622','863','710','930','1345','531','1138','834','765') THEN NC_HD_INJ_C_D_grp1=1; ELSE NC_HD_INJ_C_D_grp1=0;
# IF NC_HD_INJ_C_D IN ('338','519','664','640','998','236') THEN NC_HD_INJ_C_D_grp2=1; ELSE NC_HD_INJ_C_D_grp2=0;
# IF NC_HD_INJ_C_D IN ('249','371','602','306','948','474','566','383') THEN NC_HD_INJ_C_D_grp3=1; ELSE NC_HD_INJ_C_D_grp3=0;
# IF RSTRNT_TYP_CD = 'Y' THEN RSTRNT_TYP_CD_Y=1; ELSE RSTRNT_TYP_CD_Y=0;
# IF TLT_STRNG_WHL_OPT_CD ='U' THEN TLT_STRNG_WHL_OPT_CD_U=1; ELSE TLT_STRNG_WHL_OPT_CD_U=0;
# IF TRK_TNG_RAT_CD = 'C' THEN TRK_TNG_RAT_CD_C=1; ELSE TRK_TNG_RAT_CD_C=0;
# IF TRK_TNG_RAT_CD = 'BC' THEN TRK_TNG_RAT_CD_BC=1; ELSE TRK_TNG_RAT_CD_BC=0;
# IF vinmasterPerformanceCd ='3' THEN vinmasterPerformanceCd_3=1; ELSE vinmasterPerformanceCd_3=0;
# IF ENG_MDL_CD IN ('080100','050016','050014','080015','207065','207020','070024','070009') THEN ENG_MDL_CD_grp1=1; ELSE ENG_MDL_CD_grp1=0;	
# IF ENG_MDL_CD IN ('238004','120020') THEN ENG_MDL_CD_grp2=1; ELSE ENG_MDL_CD_grp2=0;
# IF ENG_MFG_CD IN ('238','120') THEN ENG_MFG_CD_grp1=1; ELSE ENG_MFG_CD_grp1=0;
# IF NADA_GVWC1 = . THEN NADA_GVWC1MS=1; ELSE NADA_GVWC1MS=0;


# linpred =   8.3578
 # + 39.7232*WHL_BAS_LNGST_INCHS_lt95
 # + -41.0639*WHL_BAS_SHRST_INCHS_lt95
 # + -0.4691*length_VM_bw220and250
 # + -40.2799*door_cnt_bin_lt2
 # + -0.2087*ENG_MFG_CD_070
 # + -0.1966*WHL_DRVN_CNTMS
 # + -0.0119*cko_wheelbase_max_spline100
 # + -0.2233*EA_CURB_WEIGHT_35bw45
 # + -1.9429*EA_TIP_OVER_STABILITY_RATIO_ge15
 # + -0.0206*ENG_DISPLCMNT_CI_380spline
 # + -41.1945*BODY_STYLE_CD_HR
 # + -0.6915*BODY_STYLE_CD_CV
 # + -20.4101*bodyStyleCd_grp1
 # + -41.5243*cko_eng_cylinders_2
 # + 0.1385*cko_eng_cylinders_grp1
 # + 0.9167*ENG_MDL_CD_080006
 # + -0.2659*FRNT_TYRE_SIZE_CD_grp1
 # + -0.5311*FRNT_TYRE_SIZE_CD_grp2
 # + -20.9273*FRNT_TYRE_SIZE_CD_grp3
 # + -37.3275*MAK_NM_grp1
 # + -20.8642*MAK_NM_grp2
 # + -41.4035*MFG_DESC_grp1
 # + -19.0459*NC_TireSize4_grp1
 # + 1.3372*NC_TireSize4_grp2
 # + 0.1893*NC_TractionControl_S
 # + -22.9365*NC_HD_INJ_C_D_grp1
 # + -1.6194*NC_HD_INJ_C_D_grp2
 # + 1.3241*NC_HD_INJ_C_D_grp3
 # + -40.6427*RSTRNT_TYP_CD_Y
 # + -0.5074*TLT_STRNG_WHL_OPT_CD_U
 # + 0.6982*TRK_TNG_RAT_CD_C
 # + -40.8092*TRK_TNG_RAT_CD_BC
 # + -40.5312*vinmasterPerformanceCd_3
 # + -39.0163*ENG_MDL_CD_grp1
 # + -0.1667*NADA_GVWC1MS
#  ;


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
CATFEATURES = ['BODY_STYLE_CD','bodyStyleCd','cko_eng_cylinders','ENG_MDL_CD','FRNT_TYRE_SIZE_CD','MAK_NM','MFG_DESC','RSTRNT_TYP_CD','TRK_TNG_RAT_CD','TLT_STRNG_WHL_OPT_CD','ENG_MFG_CD','cko_overdrive','cko_turbo_super','ENG_ASP_VVTL_CD','ENG_BLCK_TYP_CD','ENG_FUEL_INJ_TYP_CD','ENG_HEAD_CNFG_CD','ENG_TRK_DUTY_TYP_CD','TRANS_OVERDRV_IND','TRK_FRNT_AXL_CD','TRK_REAR_AXL_CD','ESCcd','cko_esc','frameType','cko_abs']
# Include only catfeatures that are needed for the attempted GLMs

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


data_train['WHL_BAS_LNGST_INCHS_lt95']=np.where((data_train['WHL_BAS_LNGST_INCHS']>=0) & (data_train['WHL_BAS_LNGST_INCHS']<95), 1, 0)
data_train['WHL_BAS_SHRST_INCHS_lt95']=np.where((data_train['WHL_BAS_SHRST_INCHS']>=0) & (data_train['WHL_BAS_SHRST_INCHS']<95), 1, 0)
data_train['door_cnt_bin_lt2']=np.where(data_train['door_cnt_bin']<2, 1, 0)
# data_train['cko_eng_cylinders_numeric_eq2']=np.where(data_train['cko_eng_cylinders_numeric']==2, 1, 0)
data_test['WHL_BAS_LNGST_INCHS_lt95']=np.where((data_test['WHL_BAS_LNGST_INCHS']>=0) & (data_test['WHL_BAS_LNGST_INCHS']<95), 1, 0)
data_test['WHL_BAS_SHRST_INCHS_lt95']=np.where((data_test['WHL_BAS_SHRST_INCHS']>=0) & (data_test['WHL_BAS_SHRST_INCHS']<95), 1, 0)
data_test['door_cnt_bin_lt2']=np.where(data_test['door_cnt_bin']<2, 1, 0)
# data_test['cko_eng_cylinders_numeric_eq2']=np.where(data_test['cko_eng_cylinders_numeric']==2, 1, 0)
data_train['length_VM_bw220and250']=np.where((data_train['length_VM']>=220) & (data_train['length_VM']<250), 1, 0)
data_test['length_VM_bw220and250']=np.where((data_test['length_VM']>=220) & (data_test['length_VM']<250), 1, 0)
data_train['cko_wheelbase_max_spline100']=np.minimum(data_train['cko_wheelbase_max'],100)
data_test['cko_wheelbase_max_spline100']=np.minimum(data_test['cko_wheelbase_max'],100)
data_train['EA_CURB_WEIGHT_35bw45']=np.where((3500<=data_train['EA_CURB_WEIGHT']) & (data_train['EA_CURB_WEIGHT']<4500),1,0)
data_test['EA_CURB_WEIGHT_35bw45']=np.where((3500<=data_test['EA_CURB_WEIGHT']) & (data_test['EA_CURB_WEIGHT']<4500),1,0)
data_train['EA_TIP_OVER_STABILITY_RATIO_ge15']=np.where(data_train['EA_TIP_OVER_STABILITY_RATIO']>=1.5,1,0)
data_test['EA_TIP_OVER_STABILITY_RATIO_ge15']=np.where(data_test['EA_TIP_OVER_STABILITY_RATIO']>=1.5,1,0)
data_train['ENG_DISPLCMNT_CI_380spline']=np.maximum(380,data_train['ENG_DISPLCMNT_CI'])
data_test['ENG_DISPLCMNT_CI_380spline']=np.maximum(380,data_test['ENG_DISPLCMNT_CI']) 

data_train['cko_eng_disp_max_lt2']=np.where(data_train['cko_eng_disp_max']<2,1,0)
data_test['cko_eng_disp_max_lt2']=np.where(data_test['cko_eng_disp_max']<2,1,0)
data_train['cko_eng_disp_max_gt4']=np.where(data_train['cko_eng_disp_max']>=4,1,0)
data_test['cko_eng_disp_max_gt4']=np.where(data_test['cko_eng_disp_max']>=4,1,0)
data_train['cko_max_msrp_gt120k']=np.where(data_train['cko_max_msrp']>=120000,1,0)
data_test['cko_max_msrp_gt120k']=np.where(data_test['cko_max_msrp']>=120000,1,0)
data_train['EA_ACCEL_TIME_0_TO_60MS']=np.where(data_train['EA_ACCEL_TIME_0_TO_60']<0,1,0)
data_test['EA_ACCEL_TIME_0_TO_60MS']=np.where(data_test['EA_ACCEL_TIME_0_TO_60']<0,1,0)
data_train['EA_height_ge78']=np.where(data_train['EA_height']>=78,1,0)
data_test['EA_height_ge78']=np.where(data_test['EA_height']>=78,1,0)
data_train['EA_WIDTH_ge78']=np.where(data_train['EA_WIDTH']>=78,1,0)
data_test['EA_WIDTH_ge78']=np.where(data_test['EA_WIDTH']>=78,1,0)
data_train['eng_displcmnt_L_3btw6']=np.where((data_train['eng_displcmnt_L']>=3) & (data_train['eng_displcmnt_L']<6), 1, 0)
data_test['eng_displcmnt_L_3btw6']=np.where((data_test['eng_displcmnt_L']>=3) & (data_test['eng_displcmnt_L']<6), 1, 0)
data_train['ENG_VLVS_TOTL_ge24']=np.where(data_train['ENG_VLVS_TOTL']>=24,1,0)
data_test['ENG_VLVS_TOTL_ge24']=np.where(data_test['ENG_VLVS_TOTL']>=24,1,0)
data_train['FRNT_TIRE_SIZE_prefix_ge20']=np.where(data_train['FRNT_TIRE_SIZE_prefix']>=20,1,0)
data_test['FRNT_TIRE_SIZE_prefix_ge20']=np.where(data_test['FRNT_TIRE_SIZE_prefix']>=20,1,0)
data_train['FRNT_TIRE_SIZE_suffix_ge290']=np.where(data_train['FRNT_TIRE_SIZE_suffix']>=290,1,0)
data_test['FRNT_TIRE_SIZE_suffix_ge290']=np.where(data_test['FRNT_TIRE_SIZE_suffix']>=290,1,0)
data_train['NADA_GVWC2_ge4500']=np.where(data_train['NADA_GVWC2']>=4500,1,0)
data_test['NADA_GVWC2_ge4500']=np.where(data_test['NADA_GVWC2']>=4500,1,0)
data_train['nDriveWheels_2btw4']=np.where((data_train['nDriveWheels']>=2)&(data_train['nDriveWheels']<4),1,0)
data_test['nDriveWheels_2btw4']=np.where((data_test['nDriveWheels']>=2)&(data_test['nDriveWheels']<4),1,0)
data_train['REAR_TIRE_SIZE_prefix_lt18']=np.where(data_train['REAR_TIRE_SIZE_prefix']<18,1,0)
data_test['REAR_TIRE_SIZE_prefix_lt18']=np.where(data_test['REAR_TIRE_SIZE_prefix']<18,1,0)


# data_train['enginesize_ge6']=np.where(data_train['enginesize']>=6,1,0)
# data_test['enginesize_ge6']=np.where(data_test['enginesize']>=6,1,0)
# data_train['cko_eng_cylinders_ge8']=np.where(data_train['cko_eng_cylinders']>=8, 1, 0)
# data_test['cko_eng_cylinders_ge8']=np.where(data_test['cko_eng_cylinders']>=8, 1, 0)
# data_train['cko_eng_cylinders_numeric_eq6']=np.where(data_train['cko_eng_cylinders_numeric']==6, 1, 0)
# data_test['cko_eng_cylinders_numeric_eq6']=np.where(data_test['cko_eng_cylinders_numeric']==6, 1, 0)

# data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_CPE2D4X4']+data_train['bodyStyleCd_CAMPER']+data_train['bodyStyleCd_TRK_4X2']+data_train['bodyStyleCd_CPE4X44D']+data_train['bodyStyleCd_CNV4X42D']+data_train['bodyStyleCd_WAG5D4X4']+data_train['bodyStyleCd_WAGON_2D']+data_train['bodyStyleCd_WAG5D4X2']+data_train['bodyStyleCd_WAG4D4X2']+data_train['bodyStyleCd_VAN']+data_train['bodyStyleCd_SP_HCHBK']+data_train['bodyStyleCd_SED4D4X4']+data_train['bodyStyleCd_SED2D4X4']+data_train['bodyStyleCd_SED4X44D']+data_train['bodyStyleCd_CPE_2+2']+data_train['bodyStyleCd_CP/RDSTR']+data_train['bodyStyleCd_LIMO']+data_train['bodyStyleCd_VAN_4X4']+data_train['bodyStyleCd_ROADSTER']+data_train['bodyStyleCd_MPV_4X2']+data_train['bodyStyleCd_SPORT_CP']+data_train['bodyStyleCd_TRK_4X4']+data_train['bodyStyleCd_HRDTP_2D']+data_train['bodyStyleCd_HCH2D4X4']+data_train['bodyStyleCd_HCH5D4X4']+data_train['bodyStyleCd_CABRIOLE']+data_train['bodyStyleCd_SP_COUPE']+data_train['bodyStyleCd_TARGA']+data_train['bodyStyleCd_COUPE']+data_train['bodyStyleCd_HCH3D4X4'])
# data_train['cko_eng_cylinders_grp1']=(data_train['cko_eng_cylinders_4']+data_train['cko_eng_cylinders_5']+data_train['cko_eng_cylinders_.']+data_train['cko_eng_cylinders_3'])
# data_train['FRNT_TYRE_SIZE_CD_grp1']=(data_train['FRNT_TYRE_SIZE_CD_48']+data_train['FRNT_TYRE_SIZE_CD_31']+data_train['FRNT_TYRE_SIZE_CD_49']+data_train['FRNT_TYRE_SIZE_CD_42']+data_train['FRNT_TYRE_SIZE_CD_43'])
# data_train['FRNT_TYRE_SIZE_CD_grp2']=(data_train['FRNT_TYRE_SIZE_CD_54']+data_train['FRNT_TYRE_SIZE_CD_48']+data_train['FRNT_TYRE_SIZE_CD_62'])
# data_train['FRNT_TYRE_SIZE_CD_grp3']=(data_train['FRNT_TYRE_SIZE_CD_433']+data_train['FRNT_TYRE_SIZE_CD_459']+data_train['FRNT_TYRE_SIZE_CD_28']+data_train['FRNT_TYRE_SIZE_CD_434']+data_train['FRNT_TYRE_SIZE_CD_449']+data_train['FRNT_TYRE_SIZE_CD_21']+data_train['FRNT_TYRE_SIZE_CD_85']+data_train['FRNT_TYRE_SIZE_CD_453']+data_train['FRNT_TYRE_SIZE_CD_406']+data_train['FRNT_TYRE_SIZE_CD_20']+data_train['FRNT_TYRE_SIZE_CD_11']+data_train['FRNT_TYRE_SIZE_CD_417']+data_train['FRNT_TYRE_SIZE_CD_5']+data_train['FRNT_TYRE_SIZE_CD_3']+data_train['FRNT_TYRE_SIZE_CD_4']+data_train['FRNT_TYRE_SIZE_CD_81'])
# data_train['MAK_NM_grp1']=(data_train['MAK_NM_FISKER_AUTOMOTIVE']+data_train['MAK_NM_MAYBACH']+data_train['MAK_NM_MCLAREN_AUTOMOTIVE']+data_train['MAK_NM_ROLLS-ROYCE']+data_train['MAK_NM_DAIHATSU']+data_train['MAK_NM_ALFA_ROMEO']+data_train['MAK_NM_DATSUN']+data_train['MAK_NM_LOTUS'])
# data_train['MAK_NM_grp2']=(data_train['MAK_NM_MERCEDES-BENZ']+data_train['MAK_NM_LEXUS']+data_train['MAK_NM_PONTIAC'])
# data_train['MFG_DESC_grp1']=(data_train['MFG_DESC_MAZDA_MOTOR_CORPORATIO']+data_train['MFG_DESC_FISKER_AUTOMOTIVE_INC.']+data_train['MFG_DESC_MERCEDES-BENZ_USA_LLC']+data_train['MFG_DESC_MCLAREN_AUTOMOTIVE'])
# data_train['NC_TireSize4_grp1']=(data_train['NC_TireSize4_P235/45R18']+data_train['NC_TireSize4_235/50R18']+data_train['NC_TireSize4_P255/50R19'])
# data_train['NC_TireSize4_grp2']=(data_train['NC_TireSize4_P235/60R18']+data_train['NC_TireSize4_P225/55R18'])
# data_train['NC_TireSize4_grp2']=(data_train['NC_TireSize4_P215/60R16']+data_train['NC_TireSize4_P265/50R20'])
# data_train['NC_TireSize4_grp4']=(data_train['NC_TireSize4_225/45R17']+data_train['NC_TireSize4_P235/50R18'])
# data_train['NC_HD_INJ_C_D_grp1']=(data_train['NC_HD_INJ_C_D_463']+data_train['NC_HD_INJ_C_D_1051']+data_train['NC_HD_INJ_C_D_1254']+data_train['NC_HD_INJ_C_D_524']+data_train['NC_HD_INJ_C_D_1564']+data_train['NC_HD_INJ_C_D_1026']+data_train['NC_HD_INJ_C_D_1238']+data_train['NC_HD_INJ_C_D_407']+data_train['NC_HD_INJ_C_D_522']+data_train['NC_HD_INJ_C_D_996']+data_train['NC_HD_INJ_C_D_707']+data_train['NC_HD_INJ_C_D_779']+data_train['NC_HD_INJ_C_D_900']+data_train['NC_HD_INJ_C_D_873']+data_train['NC_HD_INJ_C_D_908']+data_train['NC_HD_INJ_C_D_1024']+data_train['NC_HD_INJ_C_D_898']+data_train['NC_HD_INJ_C_D_867']+data_train['NC_HD_INJ_C_D_744']+data_train['NC_HD_INJ_C_D_750']+data_train['NC_HD_INJ_C_D_546']+data_train['NC_HD_INJ_C_D_1088']+data_train['NC_HD_INJ_C_D_1068']+data_train['NC_HD_INJ_C_D_1459']+data_train['NC_HD_INJ_C_D_528']+data_train['NC_HD_INJ_C_D_1036']+data_train['NC_HD_INJ_C_D_535']+data_train['NC_HD_INJ_C_D_793']+data_train['NC_HD_INJ_C_D_846']+data_train['NC_HD_INJ_C_D_626']+data_train['NC_HD_INJ_C_D_587']+data_train['NC_HD_INJ_C_D_2021']+data_train['NC_HD_INJ_C_D_853']+data_train['NC_HD_INJ_C_D_895']+data_train['NC_HD_INJ_C_D_212']+data_train['NC_HD_INJ_C_D_584']+data_train['NC_HD_INJ_C_D_654']+data_train['NC_HD_INJ_C_D_557']+data_train['NC_HD_INJ_C_D_818']+data_train['NC_HD_INJ_C_D_735']+data_train['NC_HD_INJ_C_D_719']+data_train['NC_HD_INJ_C_D_808']+data_train['NC_HD_INJ_C_D_1182']+data_train['NC_HD_INJ_C_D_433']+data_train['NC_HD_INJ_C_D_960']+data_train['NC_HD_INJ_C_D_516']+data_train['NC_HD_INJ_C_D_367']+data_train['NC_HD_INJ_C_D_622']+data_train['NC_HD_INJ_C_D_863']+data_train['NC_HD_INJ_C_D_710']+data_train['NC_HD_INJ_C_D_930']+data_train['NC_HD_INJ_C_D_1345']+data_train['NC_HD_INJ_C_D_531']+data_train['NC_HD_INJ_C_D_1138']+data_train['NC_HD_INJ_C_D_834']+data_train['NC_HD_INJ_C_D_765']+data_train['NC_HD_INJ_C_D_338']+data_train['NC_HD_INJ_C_D_519']+data_train['NC_HD_INJ_C_D_664']+data_train['NC_HD_INJ_C_D_640']+data_train['NC_HD_INJ_C_D_998']+data_train['NC_HD_INJ_C_D_236'])
# data_train['NC_HD_INJ_C_D_grp2']=(data_train['NC_HD_INJ_C_D_249']+data_train['NC_HD_INJ_C_D_371']+data_train['NC_HD_INJ_C_D_602']+data_train['NC_HD_INJ_C_D_306']+data_train['NC_HD_INJ_C_D_948']+data_train['NC_HD_INJ_C_D_474']+data_train['NC_HD_INJ_C_D_566']+data_train['NC_HD_INJ_C_D_383'])
# data_train['REAR_TIRE_SIZE_CD_grp1']=(data_train['REAR_TIRE_SIZE_CD_466']+data_train['REAR_TIRE_SIZE_CD_93']+data_train['REAR_TIRE_SIZE_CD_16']+data_train['REAR_TIRE_SIZE_CD_85']+data_train['REAR_TIRE_SIZE_CD_97'])
# data_train['WHL_DRVN_CNT_Tr__']=data_train['WHL_DRVN_CNT_Tr_.']	
# data_train['MAK_NM_ROLLS_ROYCE']=data_train['MAK_NM_ROLLS-ROYCE']

# data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_CPE2D4X4']+data_test['bodyStyleCd_CAMPER']+data_test['bodyStyleCd_TRK_4X2']+data_test['bodyStyleCd_CPE4X44D']+data_test['bodyStyleCd_CNV4X42D']+data_test['bodyStyleCd_WAG5D4X4']+data_test['bodyStyleCd_WAGON_2D']+data_test['bodyStyleCd_WAG5D4X2']+data_test['bodyStyleCd_WAG4D4X2']+data_test['bodyStyleCd_VAN']+data_test['bodyStyleCd_SP_HCHBK']+data_test['bodyStyleCd_SED4D4X4']+data_test['bodyStyleCd_SED2D4X4']+data_test['bodyStyleCd_SED4X44D']+data_test['bodyStyleCd_CPE_2+2']+data_test['bodyStyleCd_CP/RDSTR']+data_test['bodyStyleCd_LIMO']+data_test['bodyStyleCd_VAN_4X4']+data_test['bodyStyleCd_ROADSTER']+data_test['bodyStyleCd_MPV_4X2']+data_test['bodyStyleCd_SPORT_CP']+data_test['bodyStyleCd_TRK_4X4']+data_test['bodyStyleCd_HRDTP_2D']+data_test['bodyStyleCd_HCH2D4X4']+data_test['bodyStyleCd_HCH5D4X4']+data_test['bodyStyleCd_CABRIOLE']+data_test['bodyStyleCd_SP_COUPE']+data_test['bodyStyleCd_TARGA']+data_test['bodyStyleCd_COUPE']+data_test['bodyStyleCd_HCH3D4X4'])
# data_test['cko_eng_cylinders_grp1']=(data_test['cko_eng_cylinders_4']+data_test['cko_eng_cylinders_5']+data_test['cko_eng_cylinders_.']+data_test['cko_eng_cylinders_3'])
# data_test['FRNT_TYRE_SIZE_CD_grp1']=(data_test['FRNT_TYRE_SIZE_CD_48']+data_test['FRNT_TYRE_SIZE_CD_31']+data_test['FRNT_TYRE_SIZE_CD_49']+data_test['FRNT_TYRE_SIZE_CD_42']+data_test['FRNT_TYRE_SIZE_CD_43'])
# data_test['FRNT_TYRE_SIZE_CD_grp2']=(data_test['FRNT_TYRE_SIZE_CD_54']+data_test['FRNT_TYRE_SIZE_CD_48']+data_test['FRNT_TYRE_SIZE_CD_62'])
# data_test['FRNT_TYRE_SIZE_CD_grp3']=(data_test['FRNT_TYRE_SIZE_CD_433']+data_test['FRNT_TYRE_SIZE_CD_459']+data_test['FRNT_TYRE_SIZE_CD_28']+data_test['FRNT_TYRE_SIZE_CD_434']+data_test['FRNT_TYRE_SIZE_CD_449']+data_test['FRNT_TYRE_SIZE_CD_21']+data_test['FRNT_TYRE_SIZE_CD_85']+data_test['FRNT_TYRE_SIZE_CD_453']+data_test['FRNT_TYRE_SIZE_CD_406']+data_test['FRNT_TYRE_SIZE_CD_20']+data_test['FRNT_TYRE_SIZE_CD_11']+data_test['FRNT_TYRE_SIZE_CD_417']+data_test['FRNT_TYRE_SIZE_CD_5']+data_test['FRNT_TYRE_SIZE_CD_3']+data_test['FRNT_TYRE_SIZE_CD_4']+data_test['FRNT_TYRE_SIZE_CD_81'])
# data_test['MAK_NM_grp1']=(data_test['MAK_NM_FISKER_AUTOMOTIVE']+data_test['MAK_NM_MAYBACH']+data_test['MAK_NM_MCLAREN_AUTOMOTIVE']+data_test['MAK_NM_ROLLS-ROYCE']+data_test['MAK_NM_DAIHATSU']+data_test['MAK_NM_ALFA_ROMEO']+data_test['MAK_NM_DATSUN']+data_test['MAK_NM_LOTUS'])
# data_test['MAK_NM_grp2']=(data_test['MAK_NM_MERCEDES-BENZ']+data_test['MAK_NM_LEXUS']+data_test['MAK_NM_PONTIAC'])
# data_test['MFG_DESC_grp1']=(data_test['MFG_DESC_MAZDA_MOTOR_CORPORATIO']+data_test['MFG_DESC_FISKER_AUTOMOTIVE_INC.']+data_test['MFG_DESC_MERCEDES-BENZ_USA_LLC']+data_test['MFG_DESC_MCLAREN_AUTOMOTIVE'])
# data_test['NC_TireSize4_grp1']=(data_test['NC_TireSize4_P235/45R18']+data_test['NC_TireSize4_235/50R18']+data_test['NC_TireSize4_P255/50R19'])
# data_test['NC_TireSize4_grp2']=(data_test['NC_TireSize4_P235/60R18']+data_test['NC_TireSize4_P225/55R18'])
# data_test['NC_TireSize4_grp2']=(data_test['NC_TireSize4_P215/60R16']+data_test['NC_TireSize4_P265/50R20'])
# data_test['NC_TireSize4_grp4']=(data_test['NC_TireSize4_225/45R17']+data_test['NC_TireSize4_P235/50R18'])
# data_test['NC_HD_INJ_C_D_grp1']=(data_test['NC_HD_INJ_C_D_463']+data_test['NC_HD_INJ_C_D_1051']+data_test['NC_HD_INJ_C_D_1254']+data_test['NC_HD_INJ_C_D_524']+data_test['NC_HD_INJ_C_D_1564']+data_test['NC_HD_INJ_C_D_1026']+data_test['NC_HD_INJ_C_D_1238']+data_test['NC_HD_INJ_C_D_407']+data_test['NC_HD_INJ_C_D_522']+data_test['NC_HD_INJ_C_D_996']+data_test['NC_HD_INJ_C_D_707']+data_test['NC_HD_INJ_C_D_779']+data_test['NC_HD_INJ_C_D_900']+data_test['NC_HD_INJ_C_D_873']+data_test['NC_HD_INJ_C_D_908']+data_test['NC_HD_INJ_C_D_1024']+data_test['NC_HD_INJ_C_D_898']+data_test['NC_HD_INJ_C_D_867']+data_test['NC_HD_INJ_C_D_744']+data_test['NC_HD_INJ_C_D_750']+data_test['NC_HD_INJ_C_D_546']+data_test['NC_HD_INJ_C_D_1088']+data_test['NC_HD_INJ_C_D_1068']+data_test['NC_HD_INJ_C_D_1459']+data_test['NC_HD_INJ_C_D_528']+data_test['NC_HD_INJ_C_D_1036']+data_test['NC_HD_INJ_C_D_535']+data_test['NC_HD_INJ_C_D_793']+data_test['NC_HD_INJ_C_D_846']+data_test['NC_HD_INJ_C_D_626']+data_test['NC_HD_INJ_C_D_587']+data_test['NC_HD_INJ_C_D_2021']+data_test['NC_HD_INJ_C_D_853']+data_test['NC_HD_INJ_C_D_895']+data_test['NC_HD_INJ_C_D_212']+data_test['NC_HD_INJ_C_D_584']+data_test['NC_HD_INJ_C_D_654']+data_test['NC_HD_INJ_C_D_557']+data_test['NC_HD_INJ_C_D_818']+data_test['NC_HD_INJ_C_D_735']+data_test['NC_HD_INJ_C_D_719']+data_test['NC_HD_INJ_C_D_808']+data_test['NC_HD_INJ_C_D_1182']+data_test['NC_HD_INJ_C_D_433']+data_test['NC_HD_INJ_C_D_960']+data_test['NC_HD_INJ_C_D_516']+data_test['NC_HD_INJ_C_D_367']+data_test['NC_HD_INJ_C_D_622']+data_test['NC_HD_INJ_C_D_863']+data_test['NC_HD_INJ_C_D_710']+data_test['NC_HD_INJ_C_D_930']+data_test['NC_HD_INJ_C_D_1345']+data_test['NC_HD_INJ_C_D_531']+data_test['NC_HD_INJ_C_D_1138']+data_test['NC_HD_INJ_C_D_834']+data_test['NC_HD_INJ_C_D_765']+data_test['NC_HD_INJ_C_D_338']+data_test['NC_HD_INJ_C_D_519']+data_test['NC_HD_INJ_C_D_664']+data_test['NC_HD_INJ_C_D_640']+data_test['NC_HD_INJ_C_D_998']+data_test['NC_HD_INJ_C_D_236'])
# data_test['NC_HD_INJ_C_D_grp2']=(data_test['NC_HD_INJ_C_D_249']+data_test['NC_HD_INJ_C_D_371']+data_test['NC_HD_INJ_C_D_602']+data_test['NC_HD_INJ_C_D_306']+data_test['NC_HD_INJ_C_D_948']+data_test['NC_HD_INJ_C_D_474']+data_test['NC_HD_INJ_C_D_566']+data_test['NC_HD_INJ_C_D_383'])
# data_test['REAR_TIRE_SIZE_CD_grp1']=(data_test['REAR_TIRE_SIZE_CD_466']+data_test['REAR_TIRE_SIZE_CD_93']+data_test['REAR_TIRE_SIZE_CD_16']+data_test['REAR_TIRE_SIZE_CD_85']+data_test['REAR_TIRE_SIZE_CD_97'])
# data_test['WHL_DRVN_CNT_Tr__']=data_test['WHL_DRVN_CNT_Tr_.']
# data_test['MAK_NM_ROLLS_ROYCE']=data_test['MAK_NM_ROLLS-ROYCE']

# data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_080100']+data_train['ENG_MDL_CD_050016']+data_train['ENG_MDL_CD_050014']+data_train['ENG_MDL_CD_080015']+data_train['ENG_MDL_CD_207065']+data_train['ENG_MDL_CD_207020']+data_train['ENG_MDL_CD_070024']+data_train['ENG_MDL_CD_070009']+data_train['ENG_MDL_CD_238004'] + data_train['ENG_MDL_CD_120020'])
# data_train['ENG_MFG_CD_grp1']=(data_train['ENG_MFG_CD_238'] + data_train['ENG_MFG_CD_120'])
# data_train['MAK_NM_grp1']=(data_train['MAK_NM_LOTUS'] + data_train['MAK_NM_MAYBACH'])
# data_train['MAK_NM_grp2']=(data_train['MAK_NM_ROLLS-ROYCE'] + data_train['MAK_NM_DAIHATSU'] + data_train['MAK_NM_ALFA_ROMEO'] + data_train['MAK_NM_DATSUN'])


# data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_080100']+data_test['ENG_MDL_CD_050016']+data_test['ENG_MDL_CD_050014']+data_test['ENG_MDL_CD_080015']+data_test['ENG_MDL_CD_207065']+data_test['ENG_MDL_CD_207020']+data_test['ENG_MDL_CD_070024']+data_test['ENG_MDL_CD_070009']+data_test['ENG_MDL_CD_238004'] + data_test['ENG_MDL_CD_120020'])
# data_test['ENG_MFG_CD_grp1']=(data_test['ENG_MFG_CD_238'] + data_test['ENG_MFG_CD_120'])
# data_test['MAK_NM_grp1']=(data_test['MAK_NM_LOTUS'] + data_test['MAK_NM_MAYBACH'])
# data_test['MAK_NM_grp2']=(data_test['MAK_NM_ROLLS-ROYCE'] + data_test['MAK_NM_DAIHATSU'] + data_test['MAK_NM_ALFA_ROMEO'] + data_test['MAK_NM_DATSUN'])
data_train['ENG_MFG_CD_grp1']=(data_train['ENG_MFG_CD_204']+data_train['ENG_MFG_CD_070']+data_train['ENG_MFG_CD_238']+data_train['ENG_MFG_CD_120']+data_train['ENG_MFG_CD_165'])
data_test['ENG_MFG_CD_grp1']=(data_test['ENG_MFG_CD_204']+data_test['ENG_MFG_CD_070']+data_test['ENG_MFG_CD_238']+data_test['ENG_MFG_CD_120']+data_test['ENG_MFG_CD_165'])

data_train['ESCcd_grp1']=(data_train['ESCcd_N']+data_train['ESCcd_.'])
data_test['ESCcd_grp1']=(data_test['ESCcd_N']+data_test['ESCcd_.'])

data_train['RSTRNT_TYP_CD_grp1']=(data_train['RSTRNT_TYP_CD_G']+data_train['RSTRNT_TYP_CD_I'])
data_test['RSTRNT_TYP_CD_grp1']=(data_test['RSTRNT_TYP_CD_G']+data_test['RSTRNT_TYP_CD_I'])
# RSTRNT_TYP_CD_grp2 either stays toghether or fails togher
data_train['RSTRNT_TYP_CD_grp2']=(data_train['RSTRNT_TYP_CD_Y']+data_train['RSTRNT_TYP_CD_A']+data_train['RSTRNT_TYP_CD_M'])
data_test['RSTRNT_TYP_CD_grp2']=(data_test['RSTRNT_TYP_CD_Y']+data_test['RSTRNT_TYP_CD_A']+data_test['RSTRNT_TYP_CD_M'])

data_train['MAK_NM_ROLLS_ROYCE']=data_train['MAK_NM_ROLLS-ROYCE'] 
data_train['MFG_DESC_FISKER_AUTOMOTIVE_INC_']=data_train['MFG_DESC_FISKER_AUTOMOTIVE_INC.'] 
data_train['MFG_DESC_MERCEDES_BENZ_USA_LLC']=data_train['MFG_DESC_MERCEDES-BENZ_USA_LLC']
data_test['MAK_NM_ROLLS_ROYCE']=data_test['MAK_NM_ROLLS-ROYCE'] 
data_test['MFG_DESC_FISKER_AUTOMOTIVE_INC_']=data_test['MFG_DESC_FISKER_AUTOMOTIVE_INC.'] 
data_test['MFG_DESC_MERCEDES_BENZ_USA_LLC']=data_test['MFG_DESC_MERCEDES-BENZ_USA_LLC']
data_train['MFG_DESC_TESLA_MOTORS_INC_']=data_train['MFG_DESC_TESLA_MOTORS_INC.']	
data_test['MFG_DESC_TESLA_MOTORS_INC_']=data_test['MFG_DESC_TESLA_MOTORS_INC.']
data_train['MAK_NM_MERCEDES_BENZ']=data_train['MAK_NM_MERCEDES-BENZ'] 
data_test['MAK_NM_MERCEDES_BENZ']=data_test['MAK_NM_MERCEDES-BENZ'] 

data_train['bodyStyleCd_CP_RDSTR']=data_train['bodyStyleCd_CP/RDSTR']
data_train['bodyStyleCd_CPE_2_2']=data_train['bodyStyleCd_CPE_2+2'] 
data_train['bodyStyleCd_HCHBK2_2']=data_train['bodyStyleCd_HCHBK2+2'] 
data_test['bodyStyleCd_CP_RDSTR']=data_test['bodyStyleCd_CP/RDSTR']
data_test['bodyStyleCd_CPE_2_2']=data_test['bodyStyleCd_CPE_2+2'] 
data_test['bodyStyleCd_HCHBK2_2']=data_test['bodyStyleCd_HCHBK2+2'] 

data_train['cko_eng_cylinders__']=data_train['cko_eng_cylinders_.']	
data_test['cko_eng_cylinders__']=data_test['cko_eng_cylinders_.']
data_train['MAK_NM_grp1']=(data_train['MAK_NM_JEEP'] + data_train['MAK_NM_LEXUS'] + data_train['MAK_NM_MINI'] + data_train['MAK_NM_TOYOTA'])
data_test['MAK_NM_grp1']=(data_test['MAK_NM_JEEP'] + data_test['MAK_NM_LEXUS'] + data_test['MAK_NM_MINI'] + data_test['MAK_NM_TOYOTA'])
# If bodyStyleCd_CABRIOLE_grp1 fails delete bodyStyleCd_UTIL_4X4
# data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_CABRIOLE']+data_train['bodyStyleCd_CAMPER']+data_train['bodyStyleCd_CNV4X42D']+data_train['bodyStyleCd_CONV_2D']+data_train['bodyStyleCd_CONVRTBL']+data_train['bodyStyleCd_COUPE']+data_train['bodyStyleCd_CP_RDSTR']+data_train['bodyStyleCd_CPE_2_2']+data_train['bodyStyleCd_CPE2D4X4']+data_train['bodyStyleCd_CPE4X44D']+data_train['bodyStyleCd_HCH3D4X4']+data_train['bodyStyleCd_HCH5D4X4']+data_train['bodyStyleCd_HRDTP_2D']+data_train['bodyStyleCd_ROADSTER']+data_train['bodyStyleCd_SED2D4X4']+data_train['bodyStyleCd_SED4D4X4']+data_train['bodyStyleCd_SED4X44D']+data_train['bodyStyleCd_SP_COUPE']+data_train['bodyStyleCd_SP_HCHBK']+data_train['bodyStyleCd_SPORT_CP']+data_train['bodyStyleCd_TARGA']+data_train['bodyStyleCd_TRK_4X2']+data_train['bodyStyleCd_TRK_4X4']+data_train['bodyStyleCd_UTILITY']+data_train['bodyStyleCd_UTL4X42D']+data_train['bodyStyleCd_VAN']+data_train['bodyStyleCd_VAN_4X4']+data_train['bodyStyleCd_WAG4D4X2']+data_train['bodyStyleCd_WAG5D4X2']+data_train['bodyStyleCd_WAG5D4X4']+data_train['bodyStyleCd_WAGON_2D'])
# data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_CABRIOLE']+data_test['bodyStyleCd_CAMPER']+data_test['bodyStyleCd_CNV4X42D']+data_test['bodyStyleCd_CONV_2D']+data_test['bodyStyleCd_CONVRTBL']+data_test['bodyStyleCd_COUPE']+data_test['bodyStyleCd_CP_RDSTR']+data_test['bodyStyleCd_CPE_2_2']+data_test['bodyStyleCd_CPE2D4X4']+data_test['bodyStyleCd_CPE4X44D']+data_test['bodyStyleCd_HCH3D4X4']+data_test['bodyStyleCd_HCH5D4X4']+data_test['bodyStyleCd_HRDTP_2D']+data_test['bodyStyleCd_ROADSTER']+data_test['bodyStyleCd_SED2D4X4']+data_test['bodyStyleCd_SED4D4X4']+data_test['bodyStyleCd_SED4X44D']+data_test['bodyStyleCd_SP_COUPE']+data_test['bodyStyleCd_SP_HCHBK']+data_test['bodyStyleCd_SPORT_CP']+data_test['bodyStyleCd_TARGA']+data_test['bodyStyleCd_TRK_4X2']+data_test['bodyStyleCd_TRK_4X4']+data_test['bodyStyleCd_UTILITY']+data_test['bodyStyleCd_UTL4X42D']+data_test['bodyStyleCd_VAN']+data_test['bodyStyleCd_VAN_4X4']+data_test['bodyStyleCd_WAG4D4X2']+data_test['bodyStyleCd_WAG5D4X2']+data_test['bodyStyleCd_WAG5D4X4']+data_test['bodyStyleCd_WAGON_2D'])
# BODY_STYLE_CD_grp1 succeeds together or fails together
# data_train['BODY_STYLE_CD_grp1']=(data_train['BODY_STYLE_CD_CV']+data_train['BODY_STYLE_CD_SD']+data_train['BODY_STYLE_CD_VC']+data_train['BODY_STYLE_CD_ST']+data_train['BODY_STYLE_CD_HR']+data_train['BODY_STYLE_CD_YY'])
# data_test['BODY_STYLE_CD_grp1']=(data_test['BODY_STYLE_CD_CV']+data_test['BODY_STYLE_CD_SD']+data_test['BODY_STYLE_CD_VC']+data_test['BODY_STYLE_CD_ST']+data_test['BODY_STYLE_CD_HR']+data_test['BODY_STYLE_CD_YY'])
data_train['BODY_STYLE_CD_grp1']=(data_train['BODY_STYLE_CD_CV']+data_train['BODY_STYLE_CD_VC']+data_train['BODY_STYLE_CD_HR'])
data_test['BODY_STYLE_CD_grp1']=(data_test['BODY_STYLE_CD_CV']+data_test['BODY_STYLE_CD_VC']+data_test['BODY_STYLE_CD_HR'])
data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_050014']+data_train['ENG_MDL_CD_070024']+data_train['ENG_MDL_CD_207020']+data_train['ENG_MDL_CD_165008']+data_train['ENG_MDL_CD_070009']+data_train['ENG_MDL_CD_239003']+data_train['ENG_MDL_CD_080040']+data_train['ENG_MDL_CD_050016']+data_train['ENG_MDL_CD_207065'] + data_train['ENG_MDL_CD_238004']+data_train['ENG_MDL_CD_080100'] + data_train['ENG_MDL_CD_120020'])
data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_050014']+data_test['ENG_MDL_CD_070024']+data_test['ENG_MDL_CD_207020']+data_test['ENG_MDL_CD_165008']+data_test['ENG_MDL_CD_070009']+data_test['ENG_MDL_CD_239003']+data_test['ENG_MDL_CD_080040']+data_test['ENG_MDL_CD_050016']+data_test['ENG_MDL_CD_207065'] + data_test['ENG_MDL_CD_238004']+data_test['ENG_MDL_CD_080100'] + data_test['ENG_MDL_CD_120020'])
data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_080070']+data_train['ENG_MDL_CD_080003']+data_train['ENG_MDL_CD_080006'])
data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_080070']+data_test['ENG_MDL_CD_080003']+data_test['ENG_MDL_CD_080006'])
data_train['cko_esc_grp1']=(data_train['cko_esc_N']+data_train['cko_esc_U'])
data_test['cko_esc_grp1']=(data_test['cko_esc_N']+data_test['cko_esc_U'])
data_train['frameType_grp1']=(data_train['frameType_.']+data_train['frameType_F'])
data_test['frameType_grp1']=(data_test['frameType_.']+data_test['frameType_F'])

data_train['ENG_HEAD_CNFG_CD_grp1']=(data_train['ENG_HEAD_CNFG_CD_SOHC']+data_train['ENG_HEAD_CNFG_CD_U'])
data_test['ENG_HEAD_CNFG_CD_grp1']=(data_test['ENG_HEAD_CNFG_CD_SOHC']+data_test['ENG_HEAD_CNFG_CD_U'])

# data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_CABRIOLE']+data_train['bodyStyleCd_CAMPER']+data_train['bodyStyleCd_CNV4X42D']+data_train['bodyStyleCd_CONV_2D']+data_train['bodyStyleCd_CONVRTBL']+data_train['bodyStyleCd_COUPE']+data_train['bodyStyleCd_CP_RDSTR']+data_train['bodyStyleCd_CPE_2_2']+data_train['bodyStyleCd_CPE2D4X4']+data_train['bodyStyleCd_CPE4X44D']+data_train['bodyStyleCd_HCH3D4X4']+data_train['bodyStyleCd_HCH5D4X4']+data_train['bodyStyleCd_HRDTP_2D']+data_train['bodyStyleCd_ROADSTER']+data_train['bodyStyleCd_SED2D4X4']+data_train['bodyStyleCd_SED4D4X4']+data_train['bodyStyleCd_SED4X44D']+data_train['bodyStyleCd_SP_COUPE']+data_train['bodyStyleCd_SP_HCHBK']+data_train['bodyStyleCd_SPORT_CP']+data_train['bodyStyleCd_TARGA']+data_train['bodyStyleCd_TRK_4X2']+data_train['bodyStyleCd_TRK_4X4']+data_train['bodyStyleCd_UTILITY']+data_train['bodyStyleCd_UTL4X42D']+data_train['bodyStyleCd_VAN']+data_train['bodyStyleCd_VAN_4X4']+data_train['bodyStyleCd_WAG4D4X2']+data_train['bodyStyleCd_WAG5D4X2']+data_train['bodyStyleCd_WAG5D4X4']+data_train['bodyStyleCd_WAGON_2D'])


formula = 'lossratio ~ length_VM_bw220and250  + EA_CURB_WEIGHT_35bw45 + EA_TIP_OVER_STABILITY_RATIO_ge15  + ENG_DISPLCMNT_CI_380spline   '
# formula += ' + cko_wheelbase_max_spline100 + cko_wheelbase_maxMS ''
# formula += ' + FRNT_TIRE_SIZE_prefix + FRNT_TIRE_SIZE_suffix + REAR_TIRE_SIZE_prefix + REAR_TIRE_SIZE_suffix + REAR_TIRE_SIZEMS + REAR_TIRE_SIZEMS '
# formula += ' + cko_eng_cylinders__ + cko_eng_cylinders_2 + cko_eng_cylinders_3 + cko_eng_cylinders_4 + cko_eng_cylinders_5 + cko_eng_cylinders_8 '
# formula += ' + RSTRNT_TYP_CD_C + RSTRNT_TYP_CD_B + RSTRNT_TYP_CD_L + RSTRNT_TYP_CD_W + RSTRNT_TYP_CD_R + RSTRNT_TYP_CD_U + RSTRNT_TYP_CD_4 + RSTRNT_TYP_CD_K + RSTRNT_TYP_CD_G + RSTRNT_TYP_CD_V + RSTRNT_TYP_CD_D + RSTRNT_TYP_CD_I + RSTRNT_TYP_CD_Y + RSTRNT_TYP_CD_S + RSTRNT_TYP_CD_P + RSTRNT_TYP_CD_Z + RSTRNT_TYP_CD_F + RSTRNT_TYP_CD_X + RSTRNT_TYP_CD_A + RSTRNT_TYP_CD_J + RSTRNT_TYP_CD_T + RSTRNT_TYP_CD_M + RSTRNT_TYP_CD_3 + RSTRNT_TYP_CD_E '
formula += ' + RSTRNT_TYP_CD_grp1 + RSTRNT_TYP_CD_grp2 '
# formula += ' + TLT_STRNG_WHL_OPT_CD_N + TLT_STRNG_WHL_OPT_CD_O + TLT_STRNG_WHL_OPT_CD_U '
# formula += ' + TRK_TNG_RAT_CD_BC + TRK_TNG_RAT_CD_C + TRK_TNG_RAT_CD_B  + TRK_TNG_RAT_CD_A ' 
formula += ' + TRK_TNG_RAT_CD_C ' 
# formula += ' + MAK_NM_ACURA + MAK_NM_ALFA_ROMEO + MAK_NM_AMERICAN_GENERAL + MAK_NM_AMERICAN_MOTORS + MAK_NM_ASTON_MARTIN + MAK_NM_AUDI + MAK_NM_BENTLEY + MAK_NM_BMW + MAK_NM_BUICK + MAK_NM_CADILLAC  + MAK_NM_CHRYSLER + MAK_NM_DAEWOO + MAK_NM_DAIHATSU + MAK_NM_DATSUN  + MAK_NM_DODGE + MAK_NM_EAGLE + MAK_NM_FERRARI + MAK_NM_FIAT + MAK_NM_FISKER_AUTOMOTIVE  + MAK_NM_GEO + MAK_NM_GLOBAL_ELECTRIC_MOTORS + MAK_NM_GMC + MAK_NM_HONDA + MAK_NM_HUMMER + MAK_NM_HYUNDAI + MAK_NM_INFINITI '
# formula += ' + MAK_NM_ISUZU + MAK_NM_IVECO + MAK_NM_JAGUAR  + MAK_NM_JEEP  + MAK_NM_KIA  + MAK_NM_LAMBORGHINI + MAK_NM_LAND_ROVER + MAK_NM_LEXUS  + MAK_NM_LINCOLN  + MAK_NM_LOTUS  + MAK_NM_MASERATI + MAK_NM_MAZDA  + MAK_NM_MCLAREN_AUTOMOTIVE  + MAK_NM_MERCEDES_BENZ + MAK_NM_MERCURY + MAK_NM_MERKUR + MAK_NM_MINI + MAK_NM_MITSUBISHI + MAK_NM_NISSAN + MAK_NM_OLDSMOBILE  + MAK_NM_PLYMOUTH + MAK_NM_PONTIAC + MAK_NM_PORSCHE + MAK_NM_ROLLS_ROYCE + MAK_NM_SAAB + MAK_NM_SATURN '
# formula += ' + MAK_NM_SMART + MAK_NM_SUBARU + MAK_NM_SUZUKI + MAK_NM_TESLA + MAK_NM_TOYOTA + MAK_NM_VOLKSWAGEN + MAK_NM_VOLVO '
formula += ' + MAK_NM_grp1 '
# formula += ' + bodyStyleCd_CABRI_2D + bodyStyleCd_CABRIOLE  + bodyStyleCd_CAMPER + bodyStyleCd_CNV4X42D + bodyStyleCd_CONV_2D + bodyStyleCd_CONVRTBL + bodyStyleCd_COUPE + bodyStyleCd_COUPE_2D + bodyStyleCd_COUPE_3D '
# formula += ' + bodyStyleCd_COUPE_4D + bodyStyleCd_CP_RDSTR + bodyStyleCd_CPE_2_2 + bodyStyleCd_CPE2D4X4 + bodyStyleCd_CPE3D4X2 + bodyStyleCd_CPE4X44D + bodyStyleCd_EST_WGN + bodyStyleCd_FASTBACK + bodyStyleCd_HCH2D4X4 + bodyStyleCd_HCH3D4X2 '
# formula += ' + bodyStyleCd_HCH3D4X4 + bodyStyleCd_HCH5D4X4 + bodyStyleCd_HCHBK_2D + bodyStyleCd_HCHBK_3D + bodyStyleCd_HCHBK_4D + bodyStyleCd_HCHBK_5D + bodyStyleCd_HCHBK2_2 + bodyStyleCd_HRDTP_2D + bodyStyleCd_LFTBK_2D + bodyStyleCd_LFTBK_3D ' 
# formula += ' + bodyStyleCd_LFTBK_5D + bodyStyleCd_LIMO + bodyStyleCd_MPV_4X2 + bodyStyleCd_MPV_4X4 + bodyStyleCd_PKP4X42D + bodyStyleCd_PKP4X44D + bodyStyleCd_ROADSTER + bodyStyleCd_SED2D4X4 + bodyStyleCd_SED4D4X2 ' 
# formula += ' + bodyStyleCd_SED4D4X4 + bodyStyleCd_SED4X44D + bodyStyleCd_SEDAN_2D + bodyStyleCd_SEDAN_4D + bodyStyleCd_SP_COUPE + bodyStyleCd_SP_HCHBK + bodyStyleCd_SPORT_CP + bodyStyleCd_SUT4X24D + bodyStyleCd_SUT4X44D ' 
# formula += ' + bodyStyleCd_TARGA + bodyStyleCd_TRK_4X2 + bodyStyleCd_TRK_4X4 + bodyStyleCd_UTIL_4X2 + bodyStyleCd_UTIL_4X4 + bodyStyleCd_UTILITY + bodyStyleCd_UTL4X22D + bodyStyleCd_UTL4X24D ' 
# formula += ' + bodyStyleCd_UTL4X42D  + bodyStyleCd_VAN + bodyStyleCd_VAN_4X4 + bodyStyleCd_VAN4X24D + bodyStyleCd_WAG_3D + bodyStyleCd_WAG_4D + bodyStyleCd_WAG_4X2 + bodyStyleCd_WAG_4X4 + bodyStyleCd_WAG4D4X2 ' 
# formula += ' + bodyStyleCd_WAG4D4X4 + bodyStyleCd_WAG4X24D + bodyStyleCd_WAG4X44D + bodyStyleCd_WAG5D4X2 + bodyStyleCd_WAG5D4X4 + bodyStyleCd_WAGON_2D + bodyStyleCd_WAGON_4D + bodyStyleCd_WAGON_5D '
formula += ' + BODY_STYLE_CD_CV + BODY_STYLE_CD_SD  + BODY_STYLE_CD_CP  + BODY_STYLE_CD_CC  + BODY_STYLE_CD_LM    + BODY_STYLE_CD_WG  + BODY_STYLE_CD_VC  + BODY_STYLE_CD_HB  + BODY_STYLE_CD_ST  + BODY_STYLE_CD_CV  + BODY_STYLE_CD_IP  + BODY_STYLE_CD_HR  + BODY_STYLE_CD_CH  + BODY_STYLE_CD_YY  + BODY_STYLE_CD_TU '
# formula += ' + BODY_STYLE_CD_grp1'
# formula += ' + ENG_MFG_CD_080 + ENG_MFG_CD_235 + ENG_MFG_CD_040 + ENG_MFG_CD_060  + ENG_MFG_CD_204 + ENG_MFG_CD_180 + ENG_MFG_CD_050 + ENG_MFG_CD_207 + ENG_MFG_CD_070 + ENG_MFG_CD_238 + ENG_MFG_CD_030 + ENG_MFG_CD_120 + ENG_MFG_CD_165' 
# formula += ' + ENG_MDL_CD_050014 + ENG_MDL_CD_080085 + ENG_MDL_CD_050010 + ENG_MDL_CD_070005 + ENG_MDL_CD_050007 + ENG_MDL_CD_050060 + ENG_MDL_CD_070003 + ENG_MDL_CD_080050 + ENG_MDL_CD_070024 + ENG_MDL_CD_207020 + ENG_MDL_CD_050009 + ENG_MDL_CD_207080 + ENG_MDL_CD_080070 + ENG_MDL_CD_070011 + ENG_MDL_CD_165008 + ENG_MDL_CD_080084  + ENG_MDL_CD_070023 + ENG_MDL_CD_070009 + ENG_MDL_CD_070014 + ENG_MDL_CD_165051 + ENG_MDL_CD_239003 + ENG_MDL_CD_070008 + ENG_MDL_CD_080003 + ENG_MDL_CD_080015 + ENG_MDL_CD_050005 + ENG_MDL_CD_080020 + ENG_MDL_CD_050012 + ENG_MDL_CD_207030 + ENG_MDL_CD_040010 + ENG_MDL_CD_080114 + ENG_MDL_CD_050015 + ENG_MDL_CD_080080 + ENG_MDL_CD_207050 + ENG_MDL_CD_080008 + ENG_MDL_CD_070004 + ENG_MDL_CD_080040 + ENG_MDL_CD_080044 + ENG_MDL_CD_050016 + ENG_MDL_CD_080002 + ENG_MDL_CD_070018 + ENG_MDL_CD_207065 + ENG_MDL_CD_070007 + ENG_MDL_CD_080110 + ENG_MDL_CD_030040 + ENG_MDL_CD_080010 + ENG_MDL_CD_070015 + ENG_MDL_CD_080115 + ENG_MDL_CD_070010 + ENG_MDL_CD_238004 + ENG_MDL_CD_070013 + ENG_MDL_CD_080006 + ENG_MDL_CD_080100 + ENG_MDL_CD_120020 + ENG_MDL_CD_050006 + ENG_MDL_CD_207060 + ENG_MDL_CD_080130 '

# Delete these as you go
# + eng_displcmnt_L_3btw6


# Try all these below 




# if cylinders is significant combine ., 4,5; and 2,3,8
# If max msrp is still insignificant remove it and put cylinders_3 back in
# + cko_eng_cylinders_3 

if not os.path.exists('glm/'):
	os.makedirs('glm/')


testarray=['length_VM_bw220and250','EA_CURB_WEIGHT_35bw45','EA_TIP_OVER_STABILITY_RATIO_ge15','ENG_DISPLCMNT_CI_380spline','RSTRNT_TYP_CD_grp1','RSTRNT_TYP_CD_grp2','TLT_STRNG_WHL_OPT_CD_U','TRK_TNG_RAT_CD_C','MAK_NM_grp1','ENG_MDL_CD_grp1','ENG_MDL_CD_grp2','cko_max_msrp_gt120k','NADA_GVWC2_ge4500']

for testfield in testarray:
	print(testfield);
	print(data_train[testfield].sum())


prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm/',0)

# data_train['weight'] = data_train[WEIGHT]
# data_train['predictedValue'] = prediction_glm_train*data_train['weight']
# data_train['actualValue'] = data_train[LABEL]*data_train['weight']
# data_test['weight'] = data_test[WEIGHT]
# data_test['predictedValue'] = prediction_glm_test*data_test['weight']
# data_test['actualValue'] = data_test[LABEL]*data_test['weight']



# print('weight');
# print(data_train['weight'].sum())
# print('weighted_predictedValue');
# print(data_train['predictedValue'].sum())
# print('weighted_actualValue');
# print(data_train['actualValue'].sum())
# print('avg_predictedValue');
# print(prediction_glm_train.sum())
# print('avg_actualValue');
# print(data_train[LABEL].sum())

# ml.actualvsfittedbyfactor(data_train,data_test,'length_VM_bw220and250','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'EA_CURB_WEIGHT_35bw45','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'EA_TIP_OVER_STABILITY_RATIO_ge15','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'RSTRNT_TYP_CD_grp1','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'RSTRNT_TYP_CD_grp2','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'MAK_NM_grp1','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'bodyStyleCd_grp1','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'ENG_MDL_CD_grp1','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'ENG_MDL_CD_grp2','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'cko_max_msrp_gt120k','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'NADA_GVWC2_ge4500','glm/')
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
