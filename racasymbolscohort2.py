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

filenametrain="TRAIN2.csv"
filenametest="TEST2.csv"

print("Starting Program")
print(datetime.datetime.now())

data_train = pd.read_csv(filenametrain, skipinitialspace=True, skiprows=1, names=COLUMNS, dtype=dtype)
data_test = pd.read_csv(filenametest, skipinitialspace=True, skiprows=1, names=COLUMNS, dtype=dtype)


avgweight = data_train[WEIGHT].mean()
# print(avgweight)

data_train[WEIGHT]=data_train[WEIGHT]/avgweight
data_test[WEIGHT]= data_test[WEIGHT]/avgweight

# catarray=['antiTheftCd','bodyStyleCd','ENG_ASP_TRBL_CHGR_CD','ENG_ASP_TRBL_CHGR_CD','ENG_CBRT_TYP_CD','ENG_FUEL_INJ_TYP_CD','ENG_MDL_CD','FRNT_TYRE_SIZE_CD','NADA_BODY1','PLNT_CD','RSTRNT_TYP_CD','tonCd','TRK_CAB_CNFG_CD']

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
# data_train['cko_eng_disp_mean_lt5']=np.where(data_train['cko_eng_disp_mean']<5, 1, 0)
# data_test['cko_eng_disp_mean_lt5']=np.where(data_test['cko_eng_disp_mean']<5, 1, 0)
# data_train['EA_TURNING_CIRCLE_DIAMETER_ge64']=np.where(data_train['EA_TURNING_CIRCLE_DIAMETER']>=64, 1, 0)
# data_test['EA_TURNING_CIRCLE_DIAMETER_ge64']=np.where(data_test['EA_TURNING_CIRCLE_DIAMETER']>=64, 1, 0)
# data_train['cko_weight_min_spline5k']=np.minimum(data_train['cko_weight_min'],5000)
# data_test['cko_weight_min_spline5k']=np.minimum(data_test['cko_weight_min'],5000)

# if not os.path.exists('glm/'):
# 	os.makedirs('glm/')
# formula= 'lossratio ~   RSTRNT_TYP_CD_d2 + RSTRNT_TYP_CD_d2 + + cko_eng_disp_mean_lt5   + EA_TURNING_CIRCLE_DIAMETER_ge64  + cko_weight_min_spline5k + cko_weight_minMS '
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
# data_train['EA_TURNING_CIRCLE_DIAMETER_ge64']=np.where(data_train['EA_TURNING_CIRCLE_DIAMETER']>=64, 1, 0)
# data_test['EA_TURNING_CIRCLE_DIAMETER_ge64']=np.where(data_test['EA_TURNING_CIRCLE_DIAMETER']>=64, 1, 0)
# data_train['cko_weight_min_spline5k']=np.minimum(data_train['cko_weight_min'],5000)
# data_test['cko_weight_min_spline5k']=np.minimum(data_test['cko_weight_min'],5000)


# if not os.path.exists('glm/'):
# 	os.makedirs('glm/')
# formula= 'lossratio ~   RSTRNT_TYP_CD_d2 + RSTRNT_TYP_CD_d2  + EA_TURNING_CIRCLE_DIAMETER_ge64  + cko_weight_min_spline5k + cko_weight_minMS  '
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

# data_train['EA_TURNING_CIRCLE_DIAMETER_ge64']=np.where(data_train['EA_TURNING_CIRCLE_DIAMETER']>=64, 1, 0)
# data_test['EA_TURNING_CIRCLE_DIAMETER_ge64']=np.where(data_test['EA_TURNING_CIRCLE_DIAMETER']>=64, 1, 0)
# data_train['cko_weight_min_spline5k']=np.minimum(data_train['cko_weight_min'],5000)
# data_test['cko_weight_min_spline5k']=np.minimum(data_test['cko_weight_min'],5000)

# data_train['chenginesize2_lt5']=np.where(data_train['chenginesize2']<5, 1, 0)
# data_train['cko_height_lt80']=np.where(data_train['cko_height']<80, 1, 0)
# data_train['cko_hp_maxweight_ratio_ge45']=np.where(data_train['cko_hp_maxweight_ratio']>=0.045, 1, 0)
# data_train['cko_hp_wheelbase_ratio_12spline']=np.maximum(1.2,data_train['cko_hp_wheelbase_ratio']) 
# data_train['cko_max_msrp_ge36']=np.where(data_train['cko_max_msrp']>=36000, 1, 0)
# data_train['cko_min_msrp_spline28']=np.minimum(data_train['cko_min_msrp'],28000) 
# data_train['cko_weight_to_disp_min_lt0']=np.where(data_train['cko_weight_to_disp_min']<0, 1, 0)
# data_train['cko_weight_to_disp_max_lt0']=np.where(data_train['cko_weight_to_disp_max']<0, 1, 0)
# data_train['curbWeight_ge5']=np.where(data_train['curbWeight']>=5000, 1, 0)
# data_train['EA_ACCEL_GFORCE_45_TO_65_ge15']=np.where(data_train['EA_ACCEL_GFORCE_45_TO_65']>=0.15, 1, 0)
# data_train['EA_ACCEL_RATE_0_TO_30_ge12']=np.where(data_train['EA_ACCEL_RATE_0_TO_30']>=12, 1, 0)
# data_train['EA_ACCEL_RATE_45_TO_65_ge4']=np.where(data_train['EA_ACCEL_RATE_45_TO_65']>=4, 1, 0)
# data_train['EA_ACCEL_RATE_0_TO_60_ge8']=np.where(data_train['EA_ACCEL_RATE_0_TO_60']>=8, 1, 0)
# data_train['EA_ACCEL_TIME_45_TO_65_ge5']=np.where(data_train['EA_ACCEL_TIME_45_TO_65']>=5, 1, 0)
# data_train['EA_BRAKING_TIME_60_TO_0_ge35']=np.where(data_train['EA_BRAKING_TIME_60_TO_0']>=3.5, 1, 0)
# data_train['DOOR_CNT_ge4']=np.where(data_train['DOOR_CNT']>=4, 1, 0)
# data_train['door_cnt_bin_lt2']=np.where(data_train['door_cnt_bin']<2, 1, 0)
# data_train['EA_CURB_WEIGHT_7spline']=np.maximum(7000,data_train['EA_CURB_WEIGHT'])
# data_train['EA_CENTERLINE_LENGTH_2spline']=np.maximum(200,data_train['EA_CENTERLINE_LENGTH'])
# data_train['EA_NHTSA_STARS_ge3']=np.where(data_train['EA_NHTSA_STARS']>=3, 1, 0)
# data_train['EA_wheelbase_ge130']=np.where(data_train['EA_wheelbase']>=130, 1, 0)
# data_train['EA_YAW_MOMENT_INERTIA_575spline']=np.maximum(5750,data_train['EA_YAW_MOMENT_INERTIA'])
# data_train['ENG_DISPLCMNT_CI_lt280']=np.where(data_train['ENG_DISPLCMNT_CI']<280, 1, 0)
# data_train['MFG_BAS_MSRP_spline48']=np.minimum(data_train['MFG_BAS_MSRP'],48000)
# data_train['MFG_BAS_MSRP_48spline']=np.maximum(48000,data_train['MFG_BAS_MSRP'])
# data_train['NADA_MSRP1_lt20']=np.where(data_train['NADA_MSRP1']<20000, 1, 0)
# data_train['NADA_MSRP2_lt20']=np.where(data_train['NADA_MSRP2']<20000, 1, 0)
# data_train['NADA_MSRP3_lt20']=np.where(data_train['NADA_MSRP3']<20000, 1, 0)
# data_train['width_VM_spline98']=np.minimum(data_train['width_VM'],98)
# data_train['width_VM_98spline']=np.maximum(98,data_train['width_VM'])
# data_train['wheelBase_lt130']=np.where(data_train['wheelBase']<130, 1, 0)

# data_test['chenginesize2_lt5']=np.where(data_test['chenginesize2']<5, 1, 0)
# data_test['cko_height_lt80']=np.where(data_test['cko_height']<80, 1, 0)
# data_test['cko_hp_maxweight_ratio_ge45']=np.where(data_test['cko_hp_maxweight_ratio']>=0.045, 1, 0)
# data_test['cko_hp_wheelbase_ratio_12spline']=np.maximum(1.2,data_test['cko_hp_wheelbase_ratio']) 
# data_test['cko_max_msrp_ge36']=np.where(data_test['cko_max_msrp']>=36000, 1, 0)
# data_test['cko_min_msrp_spline28']=np.minimum(data_test['cko_min_msrp'],28000) 
# data_test['cko_weight_to_disp_min_lt0']=np.where(data_test['cko_weight_to_disp_min']<0, 1, 0)
# data_test['cko_weight_to_disp_max_lt0']=np.where(data_test['cko_weight_to_disp_max']<0, 1, 0)
# data_test['curbWeight_ge5']=np.where(data_test['curbWeight']>=5000, 1, 0)
# data_test['EA_ACCEL_GFORCE_45_TO_65_ge15']=np.where(data_test['EA_ACCEL_GFORCE_45_TO_65']>=0.15, 1, 0)
# data_test['EA_ACCEL_RATE_0_TO_30_ge12']=np.where(data_test['EA_ACCEL_RATE_0_TO_30']>=12, 1, 0)
# data_test['EA_ACCEL_RATE_45_TO_65_ge4']=np.where(data_test['EA_ACCEL_RATE_45_TO_65']>=4, 1, 0)
# data_test['EA_ACCEL_RATE_0_TO_60_ge8']=np.where(data_test['EA_ACCEL_RATE_0_TO_60']>=8, 1, 0)
# data_test['EA_ACCEL_TIME_45_TO_65_ge5']=np.where(data_test['EA_ACCEL_TIME_45_TO_65']>=5, 1, 0)
# data_test['EA_BRAKING_TIME_60_TO_0_ge35']=np.where(data_test['EA_BRAKING_TIME_60_TO_0']>=3.5, 1, 0)
# data_test['DOOR_CNT_ge4']=np.where(data_test['DOOR_CNT']>=4, 1, 0)
# data_test['door_cnt_bin_lt2']=np.where(data_test['door_cnt_bin']<2, 1, 0)
# data_test['EA_CURB_WEIGHT_7spline']=np.maximum(7000,data_test['EA_CURB_WEIGHT'])
# data_test['EA_CENTERLINE_LENGTH_2spline']=np.maximum(200,data_test['EA_CENTERLINE_LENGTH'])
# data_test['EA_NHTSA_STARS_ge3']=np.where(data_test['EA_NHTSA_STARS']>=3, 1, 0)
# data_test['EA_wheelbase_ge130']=np.where(data_test['EA_wheelbase']>=130, 1, 0)
# data_test['EA_YAW_MOMENT_INERTIA_575spline']=np.maximum(5750,data_test['EA_YAW_MOMENT_INERTIA'])
# data_test['ENG_DISPLCMNT_CI_lt280']=np.where(data_test['ENG_DISPLCMNT_CI']<280, 1, 0)
# data_test['MFG_BAS_MSRP_spline48']=np.minimum(data_test['MFG_BAS_MSRP'],48000)
# data_test['MFG_BAS_MSRP_48spline']=np.maximum(48000,data_test['MFG_BAS_MSRP'])
# data_test['NADA_MSRP1_lt20']=np.where(data_test['NADA_MSRP1']<20000, 1, 0)
# data_test['NADA_MSRP2_lt20']=np.where(data_test['NADA_MSRP2']<20000, 1, 0)
# data_test['NADA_MSRP3_lt20']=np.where(data_test['NADA_MSRP3']<20000, 1, 0)
# data_test['width_VM_spline98']=np.minimum(data_test['width_VM'],98)
# data_test['width_VM_98spline']=np.maximum(98,data_test['width_VM'])
# data_test['wheelBase_lt130']=np.where(data_test['wheelBase']<130, 1, 0)

# formulabase= 'lossratio ~   RSTRNT_TYP_CD_d2  + EA_TURNING_CIRCLE_DIAMETER_ge64  + cko_weight_min_spline5k + cko_weight_minMS  '
# forms=[]
# forms.append(' + BODY_STYLE_CD_d2 + BODY_STYLE_CD_d5 + body_cg + BODY_WAGON + BODY_STYLE_CD_d1  ')
# forms.append(' + cko_fuel_d2 + cko_max_msrp + cko_max_msrpMS  + cko_max_trans_gear  + cko_max_trans_gearMS  ')
# forms.append(' + classCd_d1  + EA_BRAKING_GFORCE_60_TO_0 + ENG_MFG_CD_d4 + ENG_MFG_CD_d5 + ENG_MFG_CD_d8  ')
# forms.append(' + height + heightMS + enginesizeMS + ENTERTAIN_CD_127  + fuel_gas + MAK_NM_d4  ')
# forms.append(' + OPT1_ENTERTAIN_CD_23 + SHIP_WGHT_LBS + SHIP_WGHT_LBSMS + TRK_CAB_CNFG_CD_d4 + TRK_CAB_CNFG_CD_d2  ')
# forms.append(' + turbo_super + TRK_CAB_CNFG_CD_d7 + WHL_BAS_LNGST_INCHS + WHL_BAS_LNGST_INCHSMS + WHL_BAS_SHRST_INCHS + WHL_BAS_SHRST_INCHS  ')
# forms.append(' + chenginesize2_lt5 + cko_height_lt80 + cko_heightMS  + cko_hp_maxweight_ratio_ge45 + cko_hp_wheelbase_ratio_12spline  ')
# forms.append(' + cko_max_msrp_ge36 + cko_min_msrp_spline28 + cko_min_msrpMS + cko_weight_to_disp_min_lt0 + cko_weight_to_disp_max_lt0 + curbWeight_ge5 + curbWeightMS  ')
# forms.append(' + EA_ACCEL_GFORCE_45_TO_65_ge15 + EA_ACCEL_RATE_0_TO_30_ge12 + EA_ACCEL_RATE_45_TO_65_ge4 + EA_ACCEL_RATE_0_TO_60_ge8 + EA_ACCEL_TIME_45_TO_65_ge5 + EA_BRAKING_TIME_60_TO_0_ge35  ')
# forms.append(' + DOOR_CNT_ge4 + DOOR_CNTMS + door_cnt_bin_lt2 + EA_CURB_WEIGHT_7spline + EA_CENTERLINE_LENGTH_2spline  ')
# forms.append(' + EA_NHTSA_STARS_ge3 + EA_wheelbase_ge130 + EA_YAW_MOMENT_INERTIA_575spline + ENG_DISPLCMNT_CI_lt280  ')
# forms.append(' + MFG_BAS_MSRP_spline48 + MFG_BAS_MSRP_48spline + NADA_MSRP1_lt20 + NADA_MSRP2_lt20 + NADA_MSRP3_lt20  ')
# forms.append(' + width_VM_spline98 + width_VM_98spline + wheelBase_lt130 ')

# forms.append(' + cko_fuel_d2 + ENG_MFG_CD_d4 + ENG_MFG_CD_d5 + fuel_gas  + SHIP_WGHT_LBS + SHIP_WGHT_LBSMS + TRK_CAB_CNFG_CD_d2  + cko_weight_to_disp_min_lt0 ')
# forms.append(' + turbo_super + WHL_BAS_LNGST_INCHS + chenginesize2_lt5 + cko_height_lt80 + cko_heightMS   + cko_hp_wheelbase_ratio_12spline + cko_min_msrp_spline28 + curbWeightMS ')
# forms.append('  + EA_ACCEL_RATE_0_TO_30_ge12 + DOOR_CNT_ge4 + DOOR_CNTMS + door_cnt_bin_lt2 + EA_NHTSA_STARS_ge3 + MFG_BAS_MSRP_spline48 + NADA_MSRP1_lt20 + NADA_MSRP2_lt20 + NADA_MSRP3_lt20 ')

# forms.append(' + cko_fuel_d2 + ENG_MFG_CD_d5 + fuel_gas + TRK_CAB_CNFG_CD_d2 + turbo_super + WHL_BAS_LNGST_INCHS + chenginesize2_lt5 + cko_height_lt80 + cko_heightMS + cko_min_msrp_spline28 + DOOR_CNTMS + door_cnt_bin_lt2 + NADA_MSRP1_lt20 + NADA_MSRP2_lt20 + NADA_MSRP3_lt20 ' )
# forms.append(' + ENG_MFG_CD_d5 + WHL_BAS_LNGST_INCHS + chenginesize2_lt5 + cko_height_lt80 + cko_heightMS + DOOR_CNTMS + door_cnt_bin_lt2 + NADA_MSRP1_lt20 + NADA_MSRP2_lt20 + NADA_MSRP3_lt20 ' )
# forms.append(' + ENG_MFG_CD_d5 + WHL_BAS_LNGST_INCHS + chenginesize2_lt5 + cko_height_lt80  + DOOR_CNTMS + door_cnt_bin_lt2 + NADA_MSRP1_lt20 + NADA_MSRP2_lt20 + NADA_MSRP3_lt20 ' )
# forms.append(' + ENG_MFG_CD_d5 + chenginesize2_lt5 + cko_height_lt80  + DOOR_CNTMS + door_cnt_bin_lt2 + NADA_MSRP1_lt20 + NADA_MSRP2_lt20 + NADA_MSRP3_lt20 ' )

# for i in range(19,len(forms)):
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
# data_train['EA_TURNING_CIRCLE_DIAMETER_ge64']=np.where(data_train['EA_TURNING_CIRCLE_DIAMETER']>=64, 1, 0)
# data_test['EA_TURNING_CIRCLE_DIAMETER_ge64']=np.where(data_test['EA_TURNING_CIRCLE_DIAMETER']>=64, 1, 0)
# data_train['cko_weight_min_spline5k']=np.minimum(data_train['cko_weight_min'],5000)
# data_test['cko_weight_min_spline5k']=np.minimum(data_test['cko_weight_min'],5000)
# data_train['chenginesize2_lt5']=np.where(data_train['chenginesize2']<5, 1, 0)
# data_test['chenginesize2_lt5']=np.where(data_test['chenginesize2']<5, 1, 0)
# data_train['cko_height_lt80']=np.where(data_train['cko_height']<80, 1, 0)
# data_test['cko_height_lt80']=np.where(data_test['cko_height']<80, 1, 0)
# data_train['door_cnt_bin_lt2']=np.where(data_train['door_cnt_bin']<2, 1, 0)
# data_test['door_cnt_bin_lt2']=np.where(data_test['door_cnt_bin']<2, 1, 0)
# data_train['NADA_MSRP1_lt20']=np.where(data_train['NADA_MSRP1']<20000, 1, 0)
# data_train['NADA_MSRP2_lt20']=np.where(data_train['NADA_MSRP2']<20000, 1, 0)
# data_train['NADA_MSRP3_lt20']=np.where(data_train['NADA_MSRP3']<20000, 1, 0)
# data_test['NADA_MSRP1_lt20']=np.where(data_test['NADA_MSRP1']<20000, 1, 0)
# data_test['NADA_MSRP2_lt20']=np.where(data_test['NADA_MSRP2']<20000, 1, 0)
# data_test['NADA_MSRP3_lt20']=np.where(data_test['NADA_MSRP3']<20000, 1, 0)

# formula = 'lossratio ~   RSTRNT_TYP_CD_d2  + EA_TURNING_CIRCLE_DIAMETER_ge64  + cko_weight_min_spline5k + cko_weight_minMS  + ENG_MFG_CD_d5 + chenginesize2_lt5 + cko_height_lt80  + DOOR_CNTMS + door_cnt_bin_lt2 + NADA_MSRP1_lt20 + NADA_MSRP2_lt20 + NADA_MSRP3_lt20 '
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
# 			print("ERROR with this GBM")
# 			print(CATFEATURES)

# for feature in ALLCATFEATURES:
# 	ml.actualvsfittedbyfactor(data_train,data_test,feature,'glm/')	



# # Step 5
# # Run GLM from result of CATFEATURES GBM

# CONTFEATURES = []
# CATFEATURES = []
# CATARRAY = []
# CATARRAY.append(['ABS_BRK_CD','ABS_BRK_DESC','antiLockCd','antiTheftCd'])
# CATARRAY.append(['bodyStyleCd'])
# CATARRAY.append(['cko_4wd','cko_abs','cko_antitheft','cko_dtrl','cko_eng_cylinders'])
# CATARRAY.append(['cko_esc','cko_fuel','cko_hp_enginesize_ratio1','cko_overdrive','cko_turbo_super','dayTimeLightCd','DR_LGHT_OPT_CD'])
# CATARRAY.append(['DRV_TYP_CD','EA_BODY_STYLE','ENG_ASP_SUP_CHGR_CD','ENG_ASP_TRBL_CHGR_CD','ENG_ASP_VVTL_CD'])
# CATARRAY.append(['ENG_CBRT_TYP_CD','ENG_CLNDR_RTR_CNT','ENG_FUEL_CD','ENG_FUEL_INJ_TYP_CD'])
# CATARRAY.append(['ENG_MDL_CD'])
# CATARRAY.append(['ENG_MFG_CD','engineType','ENTERTAIN_CD','ENTERTAIN_GP'])
# CATARRAY.append(['frameType','FRNT_TYRE_SIZE_CD'])
# CATARRAY.append(['MAK_NM','make','MFG_DESC'])
# CATARRAY.append(['NADA_BODY1','NC_ABS4w'])
# CATARRAY.append(['NC_Airbag_D','NC_Airbag_P','NC_DayRunningLights','NC_Drive','NC_Drive4'])
# CATARRAY.append(['NC_HD_INJ_C_P','NC_RearCtrLapShldrBelt','NC_RearSeatHeadRestraint','NC_SeatBeltReminder_Indicators','NC_Size_Class'])
# CATARRAY.append(['NC_TireSize','NC_TireSize4','NC_VTStabilityControl','NC_VTStabilityControl'])
# CATARRAY.append(['NC_WheelsDriven','numOfCylinders','PLNT_CD'])
# CATARRAY.append(['priceNewSymbl27_2','PWR_BRK_OPT_CD'])
# CATARRAY.append(['restraintCd','RSTRNT_TYP_CD'])
# CATARRAY.append(['TLT_STRNG_WHL_OPT_CD','tonCd','TRANS_OVERDRV_IND','TRANS_SPEED_CD'])
# CATARRAY.append(['TRK_BED_LEN_CD','TRK_BRK_TYP_CD'])
# CATARRAY.append(['TRK_CAB_CNFG_CD'])
# CATARRAY.append(['TRK_TNG_RAT_CD'])
# CATARRAY.append(['antiTheftCd','bodyStyleCd','cko_dtrl','ENG_ASP_TRBL_CHGR_CD','ENG_CBRT_TYP_CD','ENG_FUEL_CD','ENG_FUEL_INJ_TYP_CD','ENG_MDL_CD','ENG_MFG_CD','engineType','ENTERTAIN_CD','FRNT_TYRE_SIZE_CD','make','NADA_BODY1','NC_ABS4w','NC_VTStabilityControl','PLNT_CD','priceNewSymbl27_2','PWR_BRK_OPT_CD','RSTRNT_TYP_CD','TLT_STRNG_WHL_OPT_CD','tonCd','TRK_CAB_CNFG_CD'])
# CATARRAY.append(['antiTheftCd','bodyStyleCd','ENG_ASP_TRBL_CHGR_CD','ENG_CBRT_TYP_CD','ENG_FUEL_INJ_TYP_CD','ENG_MDL_CD','ENTERTAIN_CD','FRNT_TYRE_SIZE_CD','NADA_BODY1','PLNT_CD','RSTRNT_TYP_CD','tonCd','TRK_CAB_CNFG_CD'])
# CATARRAY.append(['antiTheftCd','bodyStyleCd','ENG_ASP_TRBL_CHGR_CD','ENG_CBRT_TYP_CD','ENG_FUEL_INJ_TYP_CD','ENG_MDL_CD','FRNT_TYRE_SIZE_CD','NADA_BODY1','PLNT_CD','RSTRNT_TYP_CD','tonCd','TRK_CAB_CNFG_CD'])
# CATARRAY.append(['antiTheftCd'])
# CATARRAY.append(['bodyStyleCd'])
# CATARRAY.append(['ENG_ASP_TRBL_CHGR_CD'])
# CATARRAY.append(['ENG_CBRT_TYP_CD'])
# CATARRAY.append(['ENG_FUEL_INJ_TYP_CD'])
# CATARRAY.append(['NADA_BODY1'])
# CATARRAY.append(['PLNT_CD'])
# CATARRAY.append(['RSTRNT_TYP_CD'])
# CATARRAY.append(['tonCd'])
# CATARRAY.append(['TRK_CAB_CNFG_CD'])
# CATARRAY.append(['FRNT_TYRE_SIZE_CD'])
# CATARRAY.append(['ENG_MDL_CD'])
# # 36
# CATARRAY.append(['antiTheftCd','bodyStyleCd','ENG_ASP_TRBL_CHGR_CD','ENG_CBRT_TYP_CD','ENG_FUEL_INJ_TYP_CD','NADA_BODY1','PLNT_CD','RSTRNT_TYP_CD','tonCd','TRK_CAB_CNFG_CD','FRNT_TYRE_SIZE_CD','ENG_MDL_CD'])
# CATARRAY.append(['antiTheftCd','bodyStyleCd','ENG_ASP_TRBL_CHGR_CD','ENG_CBRT_TYP_CD','ENG_FUEL_INJ_TYP_CD','NADA_BODY1','PLNT_CD','RSTRNT_TYP_CD','tonCd','TRK_CAB_CNFG_CD','FRNT_TYRE_SIZE_CD','ENG_MDL_CD'])
# CATARRAY.append(['antiTheftCd','bodyStyleCd','ENG_ASP_TRBL_CHGR_CD','ENG_CBRT_TYP_CD','ENG_FUEL_INJ_TYP_CD','NADA_BODY1','PLNT_CD','RSTRNT_TYP_CD','tonCd','TRK_CAB_CNFG_CD','FRNT_TYRE_SIZE_CD','ENG_MDL_CD'])
# CATARRAY.append(['antiTheftCd','bodyStyleCd','ENG_ASP_TRBL_CHGR_CD','ENG_CBRT_TYP_CD','ENG_FUEL_INJ_TYP_CD','NADA_BODY1','PLNT_CD','RSTRNT_TYP_CD','tonCd','TRK_CAB_CNFG_CD','FRNT_TYRE_SIZE_CD','ENG_MDL_CD'])


# # Include only catfeatures that are needed for the attempted GLMs

# # Create custom variables for GLM
# data_train['EA_TURNING_CIRCLE_DIAMETER_ge64']=np.where(data_train['EA_TURNING_CIRCLE_DIAMETER']>=64, 1, 0)
# data_test['EA_TURNING_CIRCLE_DIAMETER_ge64']=np.where(data_test['EA_TURNING_CIRCLE_DIAMETER']>=64, 1, 0)
# data_train['cko_weight_min_spline5k']=np.minimum(data_train['cko_weight_min'],5000)
# data_test['cko_weight_min_spline5k']=np.minimum(data_test['cko_weight_min'],5000)
# data_train['chenginesize2_lt5']=np.where(data_train['chenginesize2']<5, 1, 0)
# data_test['chenginesize2_lt5']=np.where(data_test['chenginesize2']<5, 1, 0)
# data_train['cko_height_lt80']=np.where(data_train['cko_height']<80, 1, 0)
# data_test['cko_height_lt80']=np.where(data_test['cko_height']<80, 1, 0)
# data_train['door_cnt_bin_lt2']=np.where(data_train['door_cnt_bin']<2, 1, 0)
# data_test['door_cnt_bin_lt2']=np.where(data_test['door_cnt_bin']<2, 1, 0)
# data_train['NADA_MSRP1_lt20']=np.where(data_train['NADA_MSRP1']<20000, 1, 0)
# data_train['NADA_MSRP2_lt20']=np.where(data_train['NADA_MSRP2']<20000, 1, 0)
# data_train['NADA_MSRP3_lt20']=np.where(data_train['NADA_MSRP3']<20000, 1, 0)
# data_test['NADA_MSRP1_lt20']=np.where(data_test['NADA_MSRP1']<20000, 1, 0)
# data_test['NADA_MSRP2_lt20']=np.where(data_test['NADA_MSRP2']<20000, 1, 0)
# data_test['NADA_MSRP3_lt20']=np.where(data_test['NADA_MSRP3']<20000, 1, 0)

# formulabase = ' lossratio ~   RSTRNT_TYP_CD_d2  + EA_TURNING_CIRCLE_DIAMETER_ge64  + cko_weight_min_spline5k + cko_weight_minMS  + ENG_MFG_CD_d5 + chenginesize2_lt5 + cko_height_lt80  + DOOR_CNTMS + door_cnt_bin_lt2 + NADA_MSRP1_lt20 + NADA_MSRP2_lt20 + NADA_MSRP3_lt20  '

# forms=[]
# forms.append(' + ABS_BRK_CD_4 + ABS_BRK_CD_U + ABS_BRK_DESC_OTHER_STD + antiLockCd_J + antiLockCd_O + antiLockCd_R + antiTheftCd_N + antiTheftCd_O + antiTheftCd_R  ')
# forms.append(' + bodyStyleCd_BUS_4X2 + bodyStyleCd_PKP_4X2 + bodyStyleCd_PKP_4X4 + bodyStyleCd_TRK_4X2 + bodyStyleCd_VAN + bodyStyleCd_VAN_4X2 + bodyStyleCd_VAN4X22D  ')
# forms.append(' + cko_4wd_RWD + cko_abs_NONE + cko_abs_S_RR + cko_abs_U + cko_antitheft_ACT_DIS + cko_antitheft_NONE + cko_antitheft_U + cko_dtrl_A + cko_eng_cylinders__ + cko_eng_cylinders_6  ')
# forms.append(' + cko_esc_S + cko_fuel_DIESL + cko_hp_enginesize_ratio1_50 + cko_overdrive_U + cko_turbo_super_Y + dayTimeLightCd_N + dayTimeLightCd_O + DR_LGHT_OPT_CD_O  ')
# forms.append(' + DRV_TYP_CD_RWD + EA_BODY_STYLE_CARGO_VAN + EA_BODY_STYLE_PASSENGER_VAN + ENG_ASP_SUP_CHGR_CD_N + ENG_ASP_TRBL_CHGR_CD_N + ENG_ASP_TRBL_CHGR_CD_Y + ENG_ASP_VVTL_CD_Y  ')
# forms.append(' + ENG_CBRT_TYP_CD_C + ENG_CLNDR_RTR_CNT__ + ENG_CLNDR_RTR_CNT_6 + ENG_FUEL_CD_D + ENG_FUEL_CD_N + ENG_FUEL_INJ_TYP_CD_D + ENG_FUEL_INJ_TYP_CD_R + ENG_FUEL_INJ_TYP_CD_S  ')
# forms.append(' + ENG_MDL_CD_010007 + ENG_MDL_CD_020020 + ENG_MDL_CD_020045 + ENG_MDL_CD_050005 + ENG_MDL_CD_050006 + ENG_MDL_CD_050010 + ENG_MDL_CD_050012 + ENG_MDL_CD_050060 + ENG_MDL_CD_070011 + ENG_MDL_CD_080040 + ENG_MDL_CD_080050 + ENG_MDL_CD_110075 + ENG_MDL_CD_165051  ')
# forms.append(' + ENG_MFG_CD_020 + ENG_MFG_CD_050 + ENG_MFG_CD_070 + engineType_N + engineType_X + ENTERTAIN_CD_1 + ENTERTAIN_CD_2 + ENTERTAIN_CD_4 + ENTERTAIN_CD_9 + ENTERTAIN_CD_U + ENTERTAIN_GP_1_2_9  ')
# forms.append(' + frameType__ + FRNT_TYRE_SIZE_CD_39 + FRNT_TYRE_SIZE_CD_41 + FRNT_TYRE_SIZE_CD_45 + FRNT_TYRE_SIZE_CD_46 + FRNT_TYRE_SIZE_CD_47  ')
# forms.append(' + MAK_NM_DODGE + MAK_NM_GMC + make_CHEV + make_DODG + make_FORD + make_RAM + MFG_DESC_CHRYSLER_GROUP_LLC + MFG_DESC_FCA + MFG_DESC_GENERAL_MOTORS + MFG_DESC_NISSAN  ')
# forms.append(' + NADA_BODY1__ + NADA_BODY1_E + NADA_BODY1_Q + NADA_BODY1_S + NADA_BODY1_U + NADA_BODY1_V + NADA_BODY1_W + NC_ABS4w_S  ')
# forms.append(' + NC_Airbag_D_NONE + NC_Airbag_D_STD + NC_Airbag_P_NONE + NC_DayRunningLights_O + NC_DayRunningLights_S + NC_Drive_2WD + NC_Drive_RWD + NC_Drive4_4WD  ')
# forms.append(' + NC_HD_INJ_C_P_1113 + NC_RearCtrLapShldrBelt_S + NC_RearSeatHeadRestraint_O + NC_RearSeatHeadRestraint_S + NC_SeatBeltReminder_Indicators_S + NC_Size_Class_PICKUP  ')
# forms.append(' + NC_TireSize_P235_75R16 + NC_TireSize_P245_75R16 + NC_TireSize_P265_70R17 + NC_TireSize4_P265_70R17 + NC_VTStabilityControl_NO + NC_VTStabilityControl_YES  ')
# forms.append(' + NC_WheelsDriven_AWD + NC_WheelsDriven_RWD_AWD + numOfCylinders__ + numOfCylinders_6 + PLNT_CD_G + PLNT_CD_H + PLNT_CD_J + PLNT_CD_Z  ')
# forms.append(' + priceNewSymbl27_2_10 + priceNewSymbl27_2_14 + priceNewSymbl27_2_20 + priceNewSymbl27_2_21 + priceNewSymbl27_2_23 + PWR_BRK_OPT_CD_O  ')
# forms.append(' + restraintCd__ + restraintCd_D + restraintCd_R + restraintCd_U + RSTRNT_TYP_CD_A + RSTRNT_TYP_CD_B + RSTRNT_TYP_CD_K + RSTRNT_TYP_CD_S + RSTRNT_TYP_CD_Z  ')
# forms.append(' + TLT_STRNG_WHL_OPT_CD_N + TLT_STRNG_WHL_OPT_CD_S + tonCd_18 + tonCd_19 + TRANS_OVERDRV_IND_U + TRANS_SPEED_CD_3 + TRANS_SPEED_CD_4 + TRANS_SPEED_CD_6  ')
# forms.append(' + TRK_BED_LEN_CD_L + TRK_BED_LEN_CD_S + TRK_BED_LEN_CD_U + TRK_BRK_TYP_CD_U  ')
# forms.append(' + TRK_CAB_CNFG_CD_CLN + TRK_CAB_CNFG_CD_CRW + TRK_CAB_CNFG_CD_CUT + TRK_CAB_CNFG_CD_EXT + TRK_CAB_CNFG_CD_STR + TRK_CAB_CNFG_CD_TLO + TRK_CAB_CNFG_CD_VAN  ')
# forms.append(' + TRK_TNG_RAT_CD_B + TRK_TNG_RAT_CD_D + TRK_TNG_RAT_CD_U ')
# # 21
# forms.append(' + antiTheftCd_O + bodyStyleCd_PKP_4X4 + bodyStyleCd_VAN + bodyStyleCd_VAN_4X2 + cko_dtrl_A  + ENG_ASP_TRBL_CHGR_CD_N + ENG_ASP_TRBL_CHGR_CD_Y + ENG_CBRT_TYP_CD_C 	+ ENG_FUEL_CD_D + ENG_FUEL_INJ_TYP_CD_S + ENG_MDL_CD_010007 + ENG_MDL_CD_050005 + ENG_MDL_CD_070011 + ENG_MDL_CD_080040 + ENG_MDL_CD_080050 + ENG_MDL_CD_110075 	+ ENG_MFG_CD_070 + engineType_X + ENTERTAIN_CD_4 + FRNT_TYRE_SIZE_CD_41 + FRNT_TYRE_SIZE_CD_45 + make_DODG + NADA_BODY1__ + NADA_BODY1_V + NC_ABS4w_S + NC_VTStabilityControl_NO + PLNT_CD_Z + priceNewSymbl27_2_23 + PWR_BRK_OPT_CD_O + RSTRNT_TYP_CD_A + RSTRNT_TYP_CD_K + TLT_STRNG_WHL_OPT_CD_S + tonCd_19 + TRK_CAB_CNFG_CD_EXT + TRK_CAB_CNFG_CD_TLO ')
# forms.append(' + antiTheftCd_O  + bodyStyleCd_VAN  + ENG_ASP_TRBL_CHGR_CD_N + ENG_CBRT_TYP_CD_C  + ENG_FUEL_INJ_TYP_CD_S  + ENG_MDL_CD_050005 + ENG_MDL_CD_070011  + ENG_MDL_CD_110075 + ENTERTAIN_CD_4 + FRNT_TYRE_SIZE_CD_41 + FRNT_TYRE_SIZE_CD_45 + NADA_BODY1_V + PLNT_CD_Z + RSTRNT_TYP_CD_A + tonCd_19 + TRK_CAB_CNFG_CD_EXT + TRK_CAB_CNFG_CD_TLO ')
# forms.append(' + antiTheftCd_O  + bodyStyleCd_VAN  + ENG_ASP_TRBL_CHGR_CD_N + ENG_CBRT_TYP_CD_C  + ENG_FUEL_INJ_TYP_CD_S  + ENG_MDL_CD_050005  + ENG_MDL_CD_110075 + FRNT_TYRE_SIZE_CD_41 + FRNT_TYRE_SIZE_CD_45 + NADA_BODY1_V + PLNT_CD_Z + RSTRNT_TYP_CD_A + tonCd_19 + TRK_CAB_CNFG_CD_EXT + TRK_CAB_CNFG_CD_TLO ')

# forms.append(' + antiTheftCd_U + antiTheftCd_N + antiTheftCd_R + antiTheftCd_P + antiTheftCd_T + antiTheftCd_O  ')
# forms.append(' + bodyStyleCd_MPV_4X2 + bodyStyleCd_VAN4X42D + bodyStyleCd_PKP4X43D + bodyStyleCd_VAN_4X2 + bodyStyleCd_SUT4X44D + bodyStyleCd_WAG_4X2 + bodyStyleCd_PKP4X44D + bodyStyleCd_WAG4X23D + bodyStyleCd_BUS_4X2 + bodyStyleCd_UTL4X44D + bodyStyleCd_WAGON + bodyStyleCd_PKP4X42D + bodyStyleCd_PKP4X24D + bodyStyleCd_PKP_4X4 + bodyStyleCd_WAGON_3D + bodyStyleCd_VAN + bodyStyleCd_PKP4X23D + bodyStyleCd_PICKUP + bodyStyleCd_PKP4X22D + bodyStyleCd_UTL4X24D + bodyStyleCd_PKP_4X2 + bodyStyleCd_VAN4X22D + bodyStyleCd_TRK_4X2 + bodyStyleCd_KING_CAB ')
# forms.append(' + ENG_ASP_TRBL_CHGR_CD_N + ENG_ASP_TRBL_CHGR_CD_I + ENG_ASP_TRBL_CHGR_CD_Y ')
# forms.append(' + ENG_CBRT_TYP_CD_C + ENG_CBRT_TYP_CD_U ')
# forms.append(' + ENG_FUEL_INJ_TYP_CD_B + ENG_FUEL_INJ_TYP_CD_D + ENG_FUEL_INJ_TYP_CD_M + ENG_FUEL_INJ_TYP_CD_P + ENG_FUEL_INJ_TYP_CD_R + ENG_FUEL_INJ_TYP_CD_S + ENG_FUEL_INJ_TYP_CD_T ')
# forms.append(' + NADA_BODY1__ + NADA_BODY1_2 + NADA_BODY1_3 + NADA_BODY1_4 + NADA_BODY1_B + NADA_BODY1_D + NADA_BODY1_E + NADA_BODY1_F + NADA_BODY1_H + NADA_BODY1_I + NADA_BODY1_M + NADA_BODY1_N + NADA_BODY1_P + NADA_BODY1_Q + NADA_BODY1_R + NADA_BODY1_S + NADA_BODY1_U + NADA_BODY1_V + NADA_BODY1_W  ')
# forms.append(' + PLNT_CD_0 + PLNT_CD_1 + PLNT_CD_3 + PLNT_CD_4 + PLNT_CD_5 + PLNT_CD_7 + PLNT_CD_9 + PLNT_CD_B + PLNT_CD_C + PLNT_CD_D + PLNT_CD_F + PLNT_CD_G + PLNT_CD_H + PLNT_CD_J + PLNT_CD_K + PLNT_CD_L + PLNT_CD_M + PLNT_CD_N + PLNT_CD_P + PLNT_CD_R + PLNT_CD_S + PLNT_CD_U + PLNT_CD_V + PLNT_CD_W + PLNT_CD_Z ')
# forms.append(' + RSTRNT_TYP_CD_3 + RSTRNT_TYP_CD_4 + RSTRNT_TYP_CD_7 + RSTRNT_TYP_CD_A + RSTRNT_TYP_CD_B + RSTRNT_TYP_CD_K + RSTRNT_TYP_CD_L + RSTRNT_TYP_CD_M + RSTRNT_TYP_CD_R + RSTRNT_TYP_CD_S + RSTRNT_TYP_CD_U + RSTRNT_TYP_CD_V + RSTRNT_TYP_CD_W + RSTRNT_TYP_CD_Y + RSTRNT_TYP_CD_Z ')
# forms.append(' + tonCd_15 + tonCd_16 + tonCd_18 + tonCd_19 + tonCd_20 + tonCd_21 + tonCd_22 + tonCd_23 + tonCd_24 + tonCd_25 + tonCd_26 + tonCd_27 + tonCd_28 ')
# forms.append(' + TRK_CAB_CNFG_CD_CLN + TRK_CAB_CNFG_CD_VAN + TRK_CAB_CNFG_CD_CRW + TRK_CAB_CNFG_CD_CUT + TRK_CAB_CNFG_CD_EXT + TRK_CAB_CNFG_CD_STR + TRK_CAB_CNFG_CD_TLO + TRK_CAB_CNFG_CD_U ')
# forms.append(' + FRNT_TYRE_SIZE_CD_17 + FRNT_TYRE_SIZE_CD_32 + FRNT_TYRE_SIZE_CD_33 + FRNT_TYRE_SIZE_CD_36 + FRNT_TYRE_SIZE_CD_38 + FRNT_TYRE_SIZE_CD_39 + FRNT_TYRE_SIZE_CD_40 + FRNT_TYRE_SIZE_CD_41 + FRNT_TYRE_SIZE_CD_45 + FRNT_TYRE_SIZE_CD_46 + FRNT_TYRE_SIZE_CD_47 + FRNT_TYRE_SIZE_CD_49 + FRNT_TYRE_SIZE_CD_58 + FRNT_TYRE_SIZE_CD_60 + FRNT_TYRE_SIZE_CD_71 + FRNT_TYRE_SIZE_CD_73 + FRNT_TYRE_SIZE_CD_74 + FRNT_TYRE_SIZE_CD_75 + FRNT_TYRE_SIZE_CD_80 ')
# forms.append(' + ENG_MDL_CD_070012 + ENG_MDL_CD_U + ENG_MDL_CD_050050 + ENG_MDL_CD_070013 + ENG_MDL_CD_050010 + ENG_MDL_CD_150085 + ENG_MDL_CD_070035 + ENG_MDL_CD_080009 + ENG_MDL_CD_050005 + ENG_MDL_CD_165040 + ENG_MDL_CD_165008 + ENG_MDL_CD_070010 + ENG_MDL_CD_020045 + ENG_MDL_CD_070017 + ENG_MDL_CD_165050 + ENG_MDL_CD_080015 + ENG_MDL_CD_050020 + ENG_MDL_CD_070015 + ENG_MDL_CD_080020 + ENG_MDL_CD_070028 + ENG_MDL_CD_020020 + ENG_MDL_CD_050006 + ENG_MDL_CD_070011 + ENG_MDL_CD_070014 + ENG_MDL_CD_080044 + ENG_MDL_CD_070022 + ENG_MDL_CD_150075 + ENG_MDL_CD_080110 + ENG_MDL_CD_080080 + ENG_MDL_CD_080050 + ENG_MDL_CD_070020 + ENG_MDL_CD_070026 + ENG_MDL_CD_070016 + ENG_MDL_CD_150001 + ENG_MDL_CD_150008 + ENG_MDL_CD_150080 + ENG_MDL_CD_050012 + ENG_MDL_CD_010007 + ENG_MDL_CD_150090 + ENG_MDL_CD_070023 + ENG_MDL_CD_110075 + ENG_MDL_CD_080040 + ENG_MDL_CD_080084 + ENG_MDL_CD_080085 + ENG_MDL_CD_080115 + ENG_MDL_CD_165035 + ENG_MDL_CD_050060 + ENG_MDL_CD_120030 + ENG_MDL_CD_165051 ')
# # 36
# forms.append(' +antiTheftCd_O  + bodyStyleCd_PKP_4X4 + bodyStyleCd_grp1 + ENG_ASP_TRBL_CHGR_CD_I + ENG_ASP_TRBL_CHGR_CD_Y + ENG_CBRT_TYP_CD_C  + ENG_FUEL_INJ_TYP_CD_P + NADA_BODY1_V + NADA_BODY1_4  +NADA_BODY1_grp1 + PLNT_CD_grp1 + PLNT_CD_grp2 + tonCd_grp1 + TRK_CAB_CNFG_CD_TLO +TRK_CAB_CNFG_CD_grp1 + FRNT_TYRE_SIZE_CD_33 + FRNT_TYRE_SIZE_CD_grp1 + ENG_MDL_CD_080040 + ENG_MDL_CD_grp1 + ENG_MDL_CD_110075 + ENG_MDL_CD_grp2 + ENG_MDL_CD_grp3 ')
# forms.append(' +antiTheftCd_O   + bodyStyleCd_PICKUP + bodyStyleCd_WAGON + bodyStyleCd_KING_CAB + bodyStyleCd_WAGON_3D + bodyStyleCd_UTL4X24D + ENG_ASP_TRBL_CHGR_CD_I  + ENG_CBRT_TYP_CD_C  + ENG_FUEL_INJ_TYP_CD_P + NADA_BODY1_V + NADA_BODY1_4  +NADA_BODY1_grp1 + PLNT_CD_grp1 + tonCd_grp1 + TRK_CAB_CNFG_CD_TLO +TRK_CAB_CNFG_CD_grp1  + FRNT_TYRE_SIZE_CD_grp1 + ENG_MDL_CD_080040 + ENG_MDL_CD_grp1 + ENG_MDL_CD_110075 + ENG_MDL_CD_grp2  ')
# forms.append(' +antiTheftCd_O   + bodyStyleCd_grp1 + ENG_ASP_TRBL_CHGR_CD_I    + ENG_FUEL_INJ_TYP_CD_P + NADA_BODY1_V + NADA_BODY1_4  +NADA_BODY1_grp1 + PLNT_CD_Z + PLNT_CD_J + PLNT_CD_3 + PLNT_CD_0 + tonCd_grp1 + TRK_CAB_CNFG_CD_TLO +TRK_CAB_CNFG_CD_grp1  + FRNT_TYRE_SIZE_CD_grp1 + ENG_MDL_CD_080040 + ENG_MDL_CD_grp1 + ENG_MDL_CD_110075 + ENG_MDL_CD_grp2  ')
# forms.append(' +antiTheftCd_O   + bodyStyleCd_grp1 + ENG_ASP_TRBL_CHGR_CD_I    + ENG_FUEL_INJ_TYP_CD_P + NADA_BODY1_V + NADA_BODY1_4  +NADA_BODY1_grp1  + tonCd_grp1 + TRK_CAB_CNFG_CD_TLO +TRK_CAB_CNFG_CD_grp1  + FRNT_TYRE_SIZE_CD_grp1 + ENG_MDL_CD_080040 + ENG_MDL_CD_grp1 + ENG_MDL_CD_110075 + ENG_MDL_CD_grp2  ')


# for i in range(39,len(forms)):

# 	CATFEATURES= CATARRAY[i]

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
# 	if i==2:
# 		data_train['cko_eng_cylinders__']=data_train['cko_eng_cylinders_.']
# 		data_test['cko_eng_cylinders__']=data_test['cko_eng_cylinders_.']
# 	elif i==5:
# 		data_train['ENG_CLNDR_RTR_CNT__']=data_train['ENG_CLNDR_RTR_CNT_.']
# 		data_test['ENG_CLNDR_RTR_CNT__']=data_test['ENG_CLNDR_RTR_CNT_.']
# 	elif i==7:
# 		data_train['ENTERTAIN_GP_1_2_9']=data_train['ENTERTAIN_GP_1,2,9']
# 		data_test['ENTERTAIN_GP_1_2_9']=data_test['ENTERTAIN_GP_1,2,9']
# 	elif i==8:
# 		data_train['frameType__']=data_train['frameType_.']
# 		data_test['frameType__']=data_test['frameType_.']
# 	elif i==10:
# 		data_train['NADA_BODY1__']=data_train['NADA_BODY1_"']
# 		data_test['NADA_BODY1__']=data_test['NADA_BODY1_"']
# 	elif i==13:
# 		data_train['NC_TireSize_P235_75R16']=data_train['NC_TireSize_P235/75R16']
# 		data_test['NC_TireSize_P235_75R16']=data_test['NC_TireSize_P235/75R16']
# 		data_train['NC_TireSize_P245_75R16']=data_train['NC_TireSize_P245/75R16']
# 		data_test['NC_TireSize_P245_75R16']=data_test['NC_TireSize_P245/75R16']
# 		data_train['NC_TireSize_P265_70R17']=data_train['NC_TireSize_P265/70R17']
# 		data_test['NC_TireSize_P265_70R17']=data_test['NC_TireSize_P265/70R17']
# 		data_train['NC_TireSize4_P265_70R17']=data_train['NC_TireSize4_P265/70R17']
# 		data_test['NC_TireSize4_P265_70R17']=data_test['NC_TireSize4_P265/70R17']
# 	elif i==14:	
# 		data_train['NC_WheelsDriven_RWD_AWD']=data_train['NC_WheelsDriven_RWD/AWD']
# 		data_test['NC_WheelsDriven_RWD_AWD']=data_test['NC_WheelsDriven_RWD/AWD']
# 		data_train['numOfCylinders__']=data_train['numOfCylinders_.']
# 		data_test['numOfCylinders__']=data_test['numOfCylinders_.']
# 	elif i==16:	
# 		data_train['restraintCd__']=data_train['restraintCd_.']
# 		data_test['restraintCd__']=data_test['restraintCd_.']
# 	elif i==21:
# 		data_train['NADA_BODY1__']=data_train['NADA_BODY1_"']
# 		data_test['NADA_BODY1__']=data_test['NADA_BODY1_"']	
# 	elif i==29:
# 		data_train['NADA_BODY1__']=data_train['NADA_BODY1_"']
# 		data_test['NADA_BODY1__']=data_test['NADA_BODY1_"']	
# 	elif i==36:
# 		data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_PICKUP']+data_train['bodyStyleCd_WAGON']+data_train['bodyStyleCd_KING_CAB']+data_train['bodyStyleCd_WAGON_3D']+data_train['bodyStyleCd_UTL4X24D'])
# 		data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_PICKUP']+data_test['bodyStyleCd_WAGON']+data_test['bodyStyleCd_KING_CAB']+data_test['bodyStyleCd_WAGON_3D']+data_test['bodyStyleCd_UTL4X24D'])	
# 		data_train['NADA_BODY1_grp1']=(data_train['NADA_BODY1_F']+data_train['NADA_BODY1_U']+data_train['NADA_BODY1_I'])
# 		data_test['NADA_BODY1_grp1']=(data_test['NADA_BODY1_F']+data_test['NADA_BODY1_U']+data_test['NADA_BODY1_I'])
# 		data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_Z']+data_train['PLNT_CD_J'])
# 		data_train['PLNT_CD_grp2']=(data_train['PLNT_CD_3']+data_train['PLNT_CD_0'])
# 		data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_Z']+data_test['PLNT_CD_J'])
# 		data_test['PLNT_CD_grp2']=(data_test['PLNT_CD_3']+data_test['PLNT_CD_0'])
# 		data_train['RSTRNT_TYP_CD_grp1']=(data_train['RSTRNT_TYP_CD_S']+data_train['RSTRNT_TYP_CD_U']+data_train['RSTRNT_TYP_CD_A']+data_train['RSTRNT_TYP_CD_M'])
# 		data_test['RSTRNT_TYP_CD_grp1']=(data_test['RSTRNT_TYP_CD_S']+data_test['RSTRNT_TYP_CD_U']+data_test['RSTRNT_TYP_CD_A']+data_test['RSTRNT_TYP_CD_M'])
# 		data_train['tonCd_grp1']=(data_train['tonCd_19']+data_train['tonCd_26'])
# 		data_test['tonCd_grp1']=(data_test['tonCd_19']+data_test['tonCd_26'])
# 		data_train['TRK_CAB_CNFG_CD_grp1']=(data_train['TRK_CAB_CNFG_CD_EXT']+data_train['TRK_CAB_CNFG_CD_CLN'])
# 		data_test['TRK_CAB_CNFG_CD_grp1']=(data_test['TRK_CAB_CNFG_CD_EXT']+data_test['TRK_CAB_CNFG_CD_CLN'])
# 		data_train['FRNT_TYRE_SIZE_CD_grp1']=(data_train['FRNT_TYRE_SIZE_CD_41']+data_train['FRNT_TYRE_SIZE_CD_45'])
# 		data_test['FRNT_TYRE_SIZE_CD_grp1']=(data_test['FRNT_TYRE_SIZE_CD_41']+data_test['FRNT_TYRE_SIZE_CD_45'])

# 		data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_150008']+data_train['ENG_MDL_CD_070014']+data_train['ENG_MDL_CD_050006']+data_train['ENG_MDL_CD_080080'])
# 		data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_070013']+data_train['ENG_MDL_CD_150085'])
# 		data_train['ENG_MDL_CD_grp3']=(data_train['ENG_MDL_CD_080009']+data_train['ENG_MDL_CD_050005']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_010007'])
# 		data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_150008']+data_test['ENG_MDL_CD_070014']+data_test['ENG_MDL_CD_050006']+data_test['ENG_MDL_CD_080080'])
# 		data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_070013']+data_test['ENG_MDL_CD_150085'])
# 		data_test['ENG_MDL_CD_grp3']=(data_test['ENG_MDL_CD_080009']+data_test['ENG_MDL_CD_050005']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_010007'])
# 	elif i==37:
# 		data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_PICKUP']+data_train['bodyStyleCd_WAGON']+data_train['bodyStyleCd_KING_CAB']+data_train['bodyStyleCd_WAGON_3D']+data_train['bodyStyleCd_UTL4X24D'])
# 		data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_PICKUP']+data_test['bodyStyleCd_WAGON']+data_test['bodyStyleCd_KING_CAB']+data_test['bodyStyleCd_WAGON_3D']+data_test['bodyStyleCd_UTL4X24D'])	
# 		data_train['NADA_BODY1_grp1']=(data_train['NADA_BODY1_F']+data_train['NADA_BODY1_U']+data_train['NADA_BODY1_I'])
# 		data_test['NADA_BODY1_grp1']=(data_test['NADA_BODY1_F']+data_test['NADA_BODY1_U']+data_test['NADA_BODY1_I'])
# 		data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_Z']+data_train['PLNT_CD_J']+data_train['PLNT_CD_3']+data_train['PLNT_CD_0'])
# 		data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_Z']+data_test['PLNT_CD_J']+data_test['PLNT_CD_3']+data_test['PLNT_CD_0'])
# 		data_train['RSTRNT_TYP_CD_grp1']=(data_train['RSTRNT_TYP_CD_S']+data_train['RSTRNT_TYP_CD_U']+data_train['RSTRNT_TYP_CD_A']+data_train['RSTRNT_TYP_CD_M'])
# 		data_test['RSTRNT_TYP_CD_grp1']=(data_test['RSTRNT_TYP_CD_S']+data_test['RSTRNT_TYP_CD_U']+data_test['RSTRNT_TYP_CD_A']+data_test['RSTRNT_TYP_CD_M'])
# 		data_train['tonCd_grp1']=(data_train['tonCd_19']+data_train['tonCd_26'])
# 		data_test['tonCd_grp1']=(data_test['tonCd_19']+data_test['tonCd_26'])
# 		data_train['TRK_CAB_CNFG_CD_grp1']=(data_train['TRK_CAB_CNFG_CD_EXT']+data_train['TRK_CAB_CNFG_CD_CLN'])
# 		data_test['TRK_CAB_CNFG_CD_grp1']=(data_test['TRK_CAB_CNFG_CD_EXT']+data_test['TRK_CAB_CNFG_CD_CLN'])
# 		data_train['FRNT_TYRE_SIZE_CD_grp1']=(data_train['FRNT_TYRE_SIZE_CD_41']+data_train['FRNT_TYRE_SIZE_CD_45'])
# 		data_test['FRNT_TYRE_SIZE_CD_grp1']=(data_test['FRNT_TYRE_SIZE_CD_41']+data_test['FRNT_TYRE_SIZE_CD_45'])

# 		data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_150008']+data_train['ENG_MDL_CD_070014']+data_train['ENG_MDL_CD_050006']+data_train['ENG_MDL_CD_080080'])
# 		data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_070013']+data_train['ENG_MDL_CD_150085'])
# 		data_train['ENG_MDL_CD_grp3']=(data_train['ENG_MDL_CD_080009']+data_train['ENG_MDL_CD_050005']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_010007'])
# 		data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_150008']+data_test['ENG_MDL_CD_070014']+data_test['ENG_MDL_CD_050006']+data_test['ENG_MDL_CD_080080'])
# 		data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_070013']+data_test['ENG_MDL_CD_150085'])
# 		data_test['ENG_MDL_CD_grp3']=(data_test['ENG_MDL_CD_080009']+data_test['ENG_MDL_CD_050005']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_010007'])	

# 	elif i==38:
# 		data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_PICKUP']+data_train['bodyStyleCd_WAGON']+data_train['bodyStyleCd_KING_CAB']+data_train['bodyStyleCd_WAGON_3D']+data_train['bodyStyleCd_UTL4X24D'])
# 		data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_PICKUP']+data_test['bodyStyleCd_WAGON']+data_test['bodyStyleCd_KING_CAB']+data_test['bodyStyleCd_WAGON_3D']+data_test['bodyStyleCd_UTL4X24D'])	
# 		data_train['NADA_BODY1_grp1']=(data_train['NADA_BODY1_F']+data_train['NADA_BODY1_U']+data_train['NADA_BODY1_I'])
# 		data_test['NADA_BODY1_grp1']=(data_test['NADA_BODY1_F']+data_test['NADA_BODY1_U']+data_test['NADA_BODY1_I'])
# 		data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_Z']+data_train['PLNT_CD_J']+data_train['PLNT_CD_3']+data_train['PLNT_CD_0'])
# 		data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_Z']+data_test['PLNT_CD_J']+data_test['PLNT_CD_3']+data_test['PLNT_CD_0'])
# 		data_train['RSTRNT_TYP_CD_grp1']=(data_train['RSTRNT_TYP_CD_S']+data_train['RSTRNT_TYP_CD_U']+data_train['RSTRNT_TYP_CD_A']+data_train['RSTRNT_TYP_CD_M'])
# 		data_test['RSTRNT_TYP_CD_grp1']=(data_test['RSTRNT_TYP_CD_S']+data_test['RSTRNT_TYP_CD_U']+data_test['RSTRNT_TYP_CD_A']+data_test['RSTRNT_TYP_CD_M'])
# 		data_train['tonCd_grp1']=(data_train['tonCd_19']+data_train['tonCd_26'])
# 		data_test['tonCd_grp1']=(data_test['tonCd_19']+data_test['tonCd_26'])
# 		data_train['TRK_CAB_CNFG_CD_grp1']=(data_train['TRK_CAB_CNFG_CD_EXT']+data_train['TRK_CAB_CNFG_CD_CLN'])
# 		data_test['TRK_CAB_CNFG_CD_grp1']=(data_test['TRK_CAB_CNFG_CD_EXT']+data_test['TRK_CAB_CNFG_CD_CLN'])
# 		data_train['FRNT_TYRE_SIZE_CD_grp1']=(data_train['FRNT_TYRE_SIZE_CD_41']+data_train['FRNT_TYRE_SIZE_CD_45'])
# 		data_test['FRNT_TYRE_SIZE_CD_grp1']=(data_test['FRNT_TYRE_SIZE_CD_41']+data_test['FRNT_TYRE_SIZE_CD_45'])

# 		data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_150008']+data_train['ENG_MDL_CD_070014']+data_train['ENG_MDL_CD_050006']+data_train['ENG_MDL_CD_080080'])
# 		data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_070013']+data_train['ENG_MDL_CD_150085'])
# 		data_train['ENG_MDL_CD_grp3']=(data_train['ENG_MDL_CD_080009']+data_train['ENG_MDL_CD_050005']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_010007'])
# 		data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_150008']+data_test['ENG_MDL_CD_070014']+data_test['ENG_MDL_CD_050006']+data_test['ENG_MDL_CD_080080'])
# 		data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_070013']+data_test['ENG_MDL_CD_150085'])
# 		data_test['ENG_MDL_CD_grp3']=(data_test['ENG_MDL_CD_080009']+data_test['ENG_MDL_CD_050005']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_010007'])		
	
# 	elif i==39:
# 		data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_PICKUP']+data_train['bodyStyleCd_WAGON']+data_train['bodyStyleCd_WAGON_3D']+data_train['bodyStyleCd_UTL4X24D'])
# 		data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_PICKUP']+data_test['bodyStyleCd_WAGON']+data_test['bodyStyleCd_WAGON_3D']+data_test['bodyStyleCd_UTL4X24D'])	
# 		data_train['NADA_BODY1_grp1']=(data_train['NADA_BODY1_F']+data_train['NADA_BODY1_U']+data_train['NADA_BODY1_I'])
# 		data_test['NADA_BODY1_grp1']=(data_test['NADA_BODY1_F']+data_test['NADA_BODY1_U']+data_test['NADA_BODY1_I'])
# 		data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_Z']+data_train['PLNT_CD_J']+data_train['PLNT_CD_3']+data_train['PLNT_CD_0'])
# 		data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_Z']+data_test['PLNT_CD_J']+data_test['PLNT_CD_3']+data_test['PLNT_CD_0'])
# 		data_train['RSTRNT_TYP_CD_grp1']=(data_train['RSTRNT_TYP_CD_S']+data_train['RSTRNT_TYP_CD_U']+data_train['RSTRNT_TYP_CD_A']+data_train['RSTRNT_TYP_CD_M'])
# 		data_test['RSTRNT_TYP_CD_grp1']=(data_test['RSTRNT_TYP_CD_S']+data_test['RSTRNT_TYP_CD_U']+data_test['RSTRNT_TYP_CD_A']+data_test['RSTRNT_TYP_CD_M'])
# 		data_train['tonCd_grp1']=(data_train['tonCd_19']+data_train['tonCd_26'])
# 		data_test['tonCd_grp1']=(data_test['tonCd_19']+data_test['tonCd_26'])
# 		data_train['TRK_CAB_CNFG_CD_grp1']=(data_train['TRK_CAB_CNFG_CD_EXT']+data_train['TRK_CAB_CNFG_CD_CLN'])
# 		data_test['TRK_CAB_CNFG_CD_grp1']=(data_test['TRK_CAB_CNFG_CD_EXT']+data_test['TRK_CAB_CNFG_CD_CLN'])
# 		data_train['FRNT_TYRE_SIZE_CD_grp1']=(data_train['FRNT_TYRE_SIZE_CD_41']+data_train['FRNT_TYRE_SIZE_CD_45'])
# 		data_test['FRNT_TYRE_SIZE_CD_grp1']=(data_test['FRNT_TYRE_SIZE_CD_41']+data_test['FRNT_TYRE_SIZE_CD_45'])

# 		data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_150008']+data_train['ENG_MDL_CD_070014']+data_train['ENG_MDL_CD_050006']+data_train['ENG_MDL_CD_080080'])
# 		data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_070013']+data_train['ENG_MDL_CD_150085'])
# 		data_train['ENG_MDL_CD_grp3']=(data_train['ENG_MDL_CD_080009']+data_train['ENG_MDL_CD_050005']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_010007'])
# 		data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_150008']+data_test['ENG_MDL_CD_070014']+data_test['ENG_MDL_CD_050006']+data_test['ENG_MDL_CD_080080'])
# 		data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_070013']+data_test['ENG_MDL_CD_150085'])
# 		data_test['ENG_MDL_CD_grp3']=(data_test['ENG_MDL_CD_080009']+data_test['ENG_MDL_CD_050005']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_010007'])		


# 	formula=formulabase+forms[i]
# 	if not os.path.exists('glm'+str(i)+'/'):
# 		os.makedirs('glm'+str(i)+'/')
# 	prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm'+str(i)+'/',0)




# # Step 6
# # Group CATFEATURES  and finalize model

# CONTFEATURES = []
# CATFEATURES = ['cko_abs','cko_dtrl','antiTheftCd','DRV_TYP_CD','NC_ABS4w','NC_SeatBeltReminder_Indicators','bodyStyleCd','ENG_ASP_TRBL_CHGR_CD','ENG_CBRT_TYP_CD','ENG_FUEL_INJ_TYP_CD','NADA_BODY1','PLNT_CD','RSTRNT_TYP_CD','tonCd','TRK_CAB_CNFG_CD','FRNT_TYRE_SIZE_CD','ENG_MDL_CD','ENG_MFG_CD']
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

# print(DUMMYFEATURES)

# data_train['EA_TURNING_CIRCLE_DIAMETER_ge64']=np.where(data_train['EA_TURNING_CIRCLE_DIAMETER']>=64, 1, 0)
# data_test['EA_TURNING_CIRCLE_DIAMETER_ge64']=np.where(data_test['EA_TURNING_CIRCLE_DIAMETER']>=64, 1, 0)
# data_train['cko_weight_min_spline5k']=np.minimum(data_train['cko_weight_min'],5000)
# data_test['cko_weight_min_spline5k']=np.minimum(data_test['cko_weight_min'],5000)
# # data_train['chenginesize2_lt5']=np.where(data_train['chenginesize2']<5, 1, 0)
# # data_test['chenginesize2_lt5']=np.where(data_test['chenginesize2']<5, 1, 0)
# data_train['cko_height_lt80']=np.where(data_train['cko_height']<80, 1, 0)
# data_test['cko_height_lt80']=np.where(data_test['cko_height']<80, 1, 0)
# # data_train['door_cnt_bin_lt2']=np.where(data_train['door_cnt_bin']<2, 1, 0)
# # data_test['door_cnt_bin_lt2']=np.where(data_test['door_cnt_bin']<2, 1, 0)
# data_train['NADA_MSRP1_lt20']=np.where(data_train['NADA_MSRP1']<20000, 1, 0)
# data_train['NADA_MSRP2_lt20']=np.where(data_train['NADA_MSRP2']<20000, 1, 0)
# data_train['NADA_MSRP3_lt20']=np.where(data_train['NADA_MSRP3']<20000, 1, 0)
# data_test['NADA_MSRP1_lt20']=np.where(data_test['NADA_MSRP1']<20000, 1, 0)
# data_test['NADA_MSRP2_lt20']=np.where(data_test['NADA_MSRP2']<20000, 1, 0)
# data_test['NADA_MSRP3_lt20']=np.where(data_test['NADA_MSRP3']<20000, 1, 0)

# data_train['cko_wheelbase_max_ge160']=np.where(data_train['cko_wheelbase_max']>=160, 1, 0)
# data_test['cko_wheelbase_max_ge160']=np.where(data_test['cko_wheelbase_max']>=160, 1, 0)
# data_train['EA_CURB_WEIGHT_ge6500']=np.where(data_train['EA_CURB_WEIGHT']>=6500, 1, 0)
# data_test['EA_CURB_WEIGHT_ge6500']=np.where(data_test['EA_CURB_WEIGHT']>=6500, 1, 0)

# data_train['cko_abs_grp1']=(data_train['cko_abs_S_UN']+data_train['cko_abs_U'])
# data_test['cko_abs_grp1']=(data_test['cko_abs_S_UN']+data_test['cko_abs_U'])

# data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_PICKUP']+data_train['bodyStyleCd_WAGON']+data_train['bodyStyleCd_WAGON_3D']+data_train['bodyStyleCd_UTL4X24D'])
# data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_PICKUP']+data_test['bodyStyleCd_WAGON']+data_test['bodyStyleCd_WAGON_3D']+data_test['bodyStyleCd_UTL4X24D'])	
# data_train['NADA_BODY1_grp1']=(data_train['NADA_BODY1_F']+data_train['NADA_BODY1_U']+data_train['NADA_BODY1_I'])
# data_test['NADA_BODY1_grp1']=(data_test['NADA_BODY1_F']+data_test['NADA_BODY1_U']+data_test['NADA_BODY1_I'])
# # data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_Z']+data_train['PLNT_CD_J']+data_train['PLNT_CD_3']+data_train['PLNT_CD_0'])
# # data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_Z']+data_test['PLNT_CD_J']+data_test['PLNT_CD_3']+data_test['PLNT_CD_0'])
# # data_train['RSTRNT_TYP_CD_grp1']=(data_train['RSTRNT_TYP_CD_S']+data_train['RSTRNT_TYP_CD_U']+data_train['RSTRNT_TYP_CD_A']+data_train['RSTRNT_TYP_CD_M'])
# # data_test['RSTRNT_TYP_CD_grp1']=(data_test['RSTRNT_TYP_CD_S']+data_test['RSTRNT_TYP_CD_U']+data_test['RSTRNT_TYP_CD_A']+data_test['RSTRNT_TYP_CD_M'])
# data_train['tonCd_grp1']=(data_train['tonCd_19']+data_train['tonCd_26'])
# data_test['tonCd_grp1']=(data_test['tonCd_19']+data_test['tonCd_26'])
# data_train['TRK_CAB_CNFG_CD_grp1']=(data_train['TRK_CAB_CNFG_CD_EXT']+data_train['TRK_CAB_CNFG_CD_CLN'])
# data_test['TRK_CAB_CNFG_CD_grp1']=(data_test['TRK_CAB_CNFG_CD_EXT']+data_test['TRK_CAB_CNFG_CD_CLN'])
# data_train['FRNT_TYRE_SIZE_CD_grp1']=(data_train['FRNT_TYRE_SIZE_CD_41']+data_train['FRNT_TYRE_SIZE_CD_45'])
# data_test['FRNT_TYRE_SIZE_CD_grp1']=(data_test['FRNT_TYRE_SIZE_CD_41']+data_test['FRNT_TYRE_SIZE_CD_45'])

# data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_150008']+data_train['ENG_MDL_CD_070014']+data_train['ENG_MDL_CD_050006']+data_train['ENG_MDL_CD_080080'])
# data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_070013']+data_train['ENG_MDL_CD_150085'])
# # data_train['ENG_MDL_CD_grp3']=(data_train['ENG_MDL_CD_080009']+data_train['ENG_MDL_CD_050005']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_010007'])
# data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_150008']+data_test['ENG_MDL_CD_070014']+data_test['ENG_MDL_CD_050006']+data_test['ENG_MDL_CD_080080'])
# data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_070013']+data_test['ENG_MDL_CD_150085'])
# # data_test['ENG_MDL_CD_grp3']=(data_test['ENG_MDL_CD_080009']+data_test['ENG_MDL_CD_050005']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_010007'])	

# formula = ' lossratio ~   RSTRNT_TYP_CD_A  + EA_TURNING_CIRCLE_DIAMETER_ge64  + cko_weight_min_spline5k + cko_weight_minMS  + ENG_MFG_CD_110  + DOOR_CNTMS   +antiTheftCd_O   + bodyStyleCd_grp1  + ENG_FUEL_INJ_TYP_CD_P + NADA_BODY1_V + NADA_BODY1_4  +NADA_BODY1_grp1  + tonCd_grp1 + TRK_CAB_CNFG_CD_TLO +TRK_CAB_CNFG_CD_grp1  + FRNT_TYRE_SIZE_CD_grp1 + ENG_MDL_CD_080040 + ENG_MDL_CD_grp1 + ENG_MDL_CD_110075 + ENG_MDL_CD_grp2   + cko_wheelbase_max_ge160 + RSTRNT_TYP_CD_K  '
# # adding  + BODY_WAGON + BODY_STYLE_CD_d5 + cko_ABS_NONE + cko_abs_grp1 + cko_abs_avalsrr + cko_dtrl_U + cko_eng_cylinders_d1
# # adding  + cko_eng_cylindersMS + cko_lengthMS + cko_min_trans_gearMS + cko_wheelbase_max_ge160 + cko_widthMS + DRV_TYP_CD_AWD + EA_BRAKING_GFORCE_60_TO_0MS
# # adding  + EA_CURB_WEIGHT_ge6500 + ENG_MFG_CD_d8 + ENG_VLVS_PER_CLNDR_d1 + enginesizeMS + MAK_NM_d4 + MFG_BAS_MSRPMS + NADA_GVWC3MS + NC_SeatBeltReminder_Indicators_S
# # adding  + NC_ABS4w_S + priceNewMS + RSTRNT_TYP_CD_d4 + TRK_CAB_CNFG_CD_d2 + TRK_CAB_CNFG_CD_d3 + wheelBaseMS

# if not os.path.exists('glm'):
# 	os.makedirs('glm')
# prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_test, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm',0)

# # data_train['weight'] = data_train[WEIGHT]
# # data_train['predictedValue'] = prediction_glm_train*data_train['weight']
# # data_train['actualValue'] = data_train[LABEL]*data_train['weight']
# # data_test['weight'] = data_test[WEIGHT]
# # data_test['predictedValue'] = prediction_glm_test*data_test['weight']
# # data_test['actualValue'] = data_test[LABEL]*data_test['weight']

# # ml.actualvsfittedbyfactor(data_train,data_test,'EA_TURNING_CIRCLE_DIAMETER_ge64','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'cko_weight_min_spline5k','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'cko_height_lt80','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'NADA_MSRP1_lt20','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'NADA_MSRP2_lt20','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'NADA_MSRP3_lt20','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'bodyStyleCd_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'NADA_BODY1_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'tonCd_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'TRK_CAB_CNFG_CD_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'FRNT_TYRE_SIZE_CD_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'ENG_MDL_CD_grp1','glm/')
# # ml.actualvsfittedbyfactor(data_train,data_test,'ENG_MDL_CD_grp2','glm/')

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

# IF RSTRNT_TYP_CD='A' THEN RSTRNT_TYP_CD_A=1; ELSE RSTRNT_TYP_CD_A=0;
# IF EA_TURNING_CIRCLE_DIAMETER>=64 THEN EA_TURNING_CIRCLE_DIAMETER_ge64=1; ELSE EA_TURNING_CIRCLE_DIAMETER_ge64=0;
# cko_weight_min_spline5k=MIN(cko_weight_min,5000);
# IF cko_weight_min = . THEN cko_weight_minMS=1; ELSE cko_weight_minMS=0;
# IF ENG_MFG_CD ='110' THEN ENG_MFG_CD_110=1; ELSE ENG_MFG_CD_110=0;
# IF DOOR_CNT= . THEN DOOR_CNTMS=1; ELSE DOOR_CNTMS=0;
# IF antiTheftCd = 'O' THEN antiTheftCd_O=1; ELSE antiTheftCd_O=0;
# IF bodyStyleCd IN ('PICKUP','WAGON','WAGON 3D','UTL4X24D') THEN bodyStyleCd_grp1=1; ELSE bodyStyleCd_grp1=0;
# IF ENG_FUEL_INJ_TYP_CD ='P' THEN ENG_FUEL_INJ_TYP_CD_P=1; ELSE ENG_FUEL_INJ_TYP_CD_P=0;
# IF NADA_BODY1 = 'V' THEN NADA_BODY1_V=1; ELSE NADA_BODY1_V=0;
# IF NADA_BODY1 = '4' THEN NADA_BODY1_4=1; ELSE NADA_BODY1_4=0;
# IF NADA_BODY1 IN ('F','U','I') THEN NADA_BODY1_grp1=1; ELSE NADA_BODY1_grp1=0;
# IF tonCd IN ('19','26') THEN tonCd_grp1=1; ELSE tonCd_grp1=0;
# IF TRK_CAB_CNFG_CD = 'TLO' THEN TRK_CAB_CNFG_CD_TLO=1; ELSE TRK_CAB_CNFG_CD_TLO=0;
# IF TRK_CAB_CNFG_CD IN ('EXT','CLN') THEN TRK_CAB_CNFG_CD_grp1=1; ELSE TRK_CAB_CNFG_CD_grp1=0;
# IF FRNT_TYRE_SIZE_CD IN ('41','45') THEN FRNT_TYRE_SIZE_CD_grp1=1; ELSe =0;
# IF ENG_MDL_CD ='080040' THEN ENG_MDL_CD_080040=1; ELSE ENG_MDL_CD_080040=0;
# IF ENG_MDL_CD = '110075' THEN ENG_MDL_CD_110075=1; ELSE ENG_MDL_CD_110075=0;
# IF ENG_MDL_CD IN('150008','070014','050006','080080') THEN ENG_MDL_CD_grp1=1; ELSE ENG_MDL_CD_grp1=0;
# IF ENG_MDL_CD IN ('070013','150085') THEN ENG_MDL_CD_grp2=1; ELSE ENG_MDL_CD_grp2=0;
# IF cko_wheelbase_max>=160 THEN cko_wheelbase_max_ge160=1; ELSE cko_wheelbase_max_ge160=0;
# IF RSTRNT_TYP_CD ='K' THEN RSTRNT_TYP_CD_K=1; ELSE RSTRNT_TYP_CD_K=0;


# linpred= -2.8658
#  + -0.5186*RSTRNT_TYP_CD_A
#  + 0.4766*EA_TURNING_CIRCLE_DIAMETER_ge64
#  + 0.0004*cko_weight_min_spline5k
#  + 1.818*cko_weight_minMS
#  + 0.146*ENG_MFG_CD_110
#  + -0.6199*DOOR_CNTMS
#  + 0.2023*antiTheftCd_O
#  + 39.9724*bodyStyleCd_grp1
#  + 41.274*ENG_FUEL_INJ_TYP_CD_P
#  + -1.5782*NADA_BODY1_V
#  + 41.4389*NADA_BODY1_4
#  + 0.2101*NADA_BODY1_grp1
#  + -0.1614*tonCd_grp1
#  + 41.3569*TRK_CAB_CNFG_CD_TLO
#  + 0.2518*TRK_CAB_CNFG_CD_grp1
#  + 0.2449*FRNT_TYRE_SIZE_CD_grp1
#  + -0.5808*ENG_MDL_CD_080040
#  + 0.146*ENG_MDL_CD_110075
#  + -1.3932*ENG_MDL_CD_grp1
#  + 0.2408*ENG_MDL_CD_grp2
#  + 0.1676*cko_wheelbase_max_ge160
#  + 0.3724*RSTRNT_TYP_CD_K


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
CATFEATURES = ['ENG_FUEL_INJ_TYP_CD','RSTRNT_TYP_CD','tonCd','TRK_CAB_CNFG_CD','FRNT_TYRE_SIZE_CD','ENG_MDL_CD','MAK_NM','SEGMENTATION_CD'] 
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

# data_train['EA_TURNING_CIRCLE_DIAMETER_ge64']=np.where(data_train['EA_TURNING_CIRCLE_DIAMETER']>=64, 1, 0)
# data_test['EA_TURNING_CIRCLE_DIAMETER_ge64']=np.where(data_test['EA_TURNING_CIRCLE_DIAMETER']>=64, 1, 0)
data_train['cko_weight_min_spline5k']=np.minimum(data_train['cko_weight_min'],5000)
data_test['cko_weight_min_spline5k']=np.minimum(data_test['cko_weight_min'],5000)
# data_train['chenginesize2_lt5']=np.where(data_train['chenginesize2']<5, 1, 0)
# data_test['chenginesize2_lt5']=np.where(data_test['chenginesize2']<5, 1, 0)
data_train['cko_height_lt80']=np.where(data_train['cko_height']<80, 1, 0)
data_test['cko_height_lt80']=np.where(data_test['cko_height']<80, 1, 0)
# data_train['door_cnt_bin_lt2']=np.where(data_train['door_cnt_bin']<2, 1, 0)
# data_test['door_cnt_bin_lt2']=np.where(data_test['door_cnt_bin']<2, 1, 0)
# data_train['NADA_MSRP1_lt20']=np.where(data_train['NADA_MSRP1']<20000, 1, 0)
# data_train['NADA_MSRP2_lt20']=np.where(data_train['NADA_MSRP2']<20000, 1, 0)
# data_train['NADA_MSRP3_lt20']=np.where(data_train['NADA_MSRP3']<20000, 1, 0)
# data_test['NADA_MSRP1_lt20']=np.where(data_test['NADA_MSRP1']<20000, 1, 0)
# data_test['NADA_MSRP2_lt20']=np.where(data_test['NADA_MSRP2']<20000, 1, 0)
# data_test['NADA_MSRP3_lt20']=np.where(data_test['NADA_MSRP3']<20000, 1, 0)

data_train['EA_ACCEL_TIME_0_TO_30_ge5']=np.where(data_train['EA_ACCEL_TIME_0_TO_30']>=5, 1, 0)
data_test['EA_ACCEL_TIME_0_TO_30_ge5']=np.where(data_test['EA_ACCEL_TIME_0_TO_30']>=5, 1, 0)
data_train['EA_ACCEL_TIME_45_TO_65_ge9']=np.where(data_train['EA_ACCEL_TIME_45_TO_65']>=9, 1, 0)
data_test['EA_ACCEL_TIME_45_TO_65_ge9']=np.where(data_test['EA_ACCEL_TIME_45_TO_65']>=9, 1, 0)
data_train['EA_BRAKING_DISTANCE_60_TO_0_ge175']=np.where(data_train['EA_BRAKING_DISTANCE_60_TO_0']>=1.75, 1, 0)
data_test['EA_BRAKING_DISTANCE_60_TO_0_ge175']=np.where(data_test['EA_BRAKING_DISTANCE_60_TO_0']>=1.75, 1, 0)

data_train['cko_wheelbase_max_ge160']=np.where(data_train['cko_wheelbase_max']>=160, 1, 0)
data_test['cko_wheelbase_max_ge160']=np.where(data_test['cko_wheelbase_max']>=160, 1, 0)
data_train['EA_CURB_WEIGHT_ge6500']=np.where(data_train['EA_CURB_WEIGHT']>=6500, 1, 0)
data_test['EA_CURB_WEIGHT_ge6500']=np.where(data_test['EA_CURB_WEIGHT']>=6500, 1, 0)

# data_train['cko_abs_grp1']=(data_train['cko_abs_S_UN']+data_train['cko_abs_U'])
# data_test['cko_abs_grp1']=(data_test['cko_abs_S_UN']+data_test['cko_abs_U'])

# data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_PICKUP']+data_train['bodyStyleCd_WAGON']+data_train['bodyStyleCd_WAGON_3D']+data_train['bodyStyleCd_UTL4X24D'])
# data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_PICKUP']+data_test['bodyStyleCd_WAGON']+data_test['bodyStyleCd_WAGON_3D']+data_test['bodyStyleCd_UTL4X24D'])	
# data_train['NADA_BODY1_grp1']=(data_train['NADA_BODY1_F']+data_train['NADA_BODY1_U']+data_train['NADA_BODY1_I'])
# data_test['NADA_BODY1_grp1']=(data_test['NADA_BODY1_F']+data_test['NADA_BODY1_U']+data_test['NADA_BODY1_I'])
# data_train['PLNT_CD_grp1']=(data_train['PLNT_CD_Z']+data_train['PLNT_CD_J']+data_train['PLNT_CD_3']+data_train['PLNT_CD_0'])
# data_test['PLNT_CD_grp1']=(data_test['PLNT_CD_Z']+data_test['PLNT_CD_J']+data_test['PLNT_CD_3']+data_test['PLNT_CD_0'])
# data_train['RSTRNT_TYP_CD_grp1']=(data_train['RSTRNT_TYP_CD_S']+data_train['RSTRNT_TYP_CD_U']+data_train['RSTRNT_TYP_CD_A']+data_train['RSTRNT_TYP_CD_M'])
# data_test['RSTRNT_TYP_CD_grp1']=(data_test['RSTRNT_TYP_CD_S']+data_test['RSTRNT_TYP_CD_U']+data_test['RSTRNT_TYP_CD_A']+data_test['RSTRNT_TYP_CD_M'])
data_train['tonCd_grp1']=(data_train['tonCd_19']+data_train['tonCd_26'])
data_test['tonCd_grp1']=(data_test['tonCd_19']+data_test['tonCd_26'])
data_train['TRK_CAB_CNFG_CD_grp1']=(data_train['TRK_CAB_CNFG_CD_EXT']+data_train['TRK_CAB_CNFG_CD_CLN'])
data_test['TRK_CAB_CNFG_CD_grp1']=(data_test['TRK_CAB_CNFG_CD_EXT']+data_test['TRK_CAB_CNFG_CD_CLN'])
# data_train['FRNT_TYRE_SIZE_CD_grp1']=(data_train['FRNT_TYRE_SIZE_CD_41']+data_train['FRNT_TYRE_SIZE_CD_45'])
# data_test['FRNT_TYRE_SIZE_CD_grp1']=(data_test['FRNT_TYRE_SIZE_CD_41']+data_test['FRNT_TYRE_SIZE_CD_45'])

data_train['RSTRNT_TYP_CD_grp1']=(data_train['RSTRNT_TYP_CD_S']+data_train['RSTRNT_TYP_CD_U']+data_train['RSTRNT_TYP_CD_4']+data_train['RSTRNT_TYP_CD_7']+data_train['RSTRNT_TYP_CD_Y'])
data_test['RSTRNT_TYP_CD_grp1']=(data_test['RSTRNT_TYP_CD_S']+data_test['RSTRNT_TYP_CD_U']+data_test['RSTRNT_TYP_CD_4']+data_test['RSTRNT_TYP_CD_7']+data_test['RSTRNT_TYP_CD_Y'])

data_train['MAK_NM_MERCEDES_BENZ']=data_train['MAK_NM_MERCEDES-BENZ']
data_test['MAK_NM_MERCEDES_BENZ']=data_test['MAK_NM_MERCEDES-BENZ']

# data_train['bodyStyleCd_BUS_4X2']=data_train['bodyStyleCd_BUS 4X2']
# data_train['bodyStyleCd_KING_CAB']=data_train['bodyStyleCd_KING CAB']
# data_train['bodyStyleCd_MPV_4X2']=data_train['bodyStyleCd_MPV 4X2']
# data_train['bodyStyleCd_PKP_4X2']=data_train['bodyStyleCd_PKP 4X2']
# data_train['bodyStyleCd_PKP_4X4']=data_train['bodyStyleCd_PKP 4X4']
# data_train['bodyStyleCd_TRK_4X2']=data_train['bodyStyleCd_TRK 4X2']
# data_train['bodyStyleCd_VAN_4X2']=data_train['bodyStyleCd_VAN 4X2']
# data_train['bodyStyleCd_WAG_4X2']=data_train['bodyStyleCd_WAG 4X2']
# data_train['bodyStyleCd_WAGON_3D']=data_train['bodyStyleCd_WAGON 3D']

# data_test['bodyStyleCd_BUS_4X2']=data_test['bodyStyleCd_BUS 4X2']
# data_test['bodyStyleCd_KING_CAB']=data_test['bodyStyleCd_KING CAB']
# data_test['bodyStyleCd_MPV_4X2']=data_test['bodyStyleCd_MPV 4X2']
# data_test['bodyStyleCd_PKP_4X2']=data_test['bodyStyleCd_PKP 4X2']
# data_test['bodyStyleCd_PKP_4X4']=data_test['bodyStyleCd_PKP 4X4']
# data_test['bodyStyleCd_TRK_4X2']=data_test['bodyStyleCd_TRK 4X2']
# data_test['bodyStyleCd_VAN_4X2']=data_test['bodyStyleCd_VAN 4X2']
# data_test['bodyStyleCd_WAG_4X2']=data_test['bodyStyleCd_WAG 4X2']
# data_test['bodyStyleCd_WAGON_3D']=data_test['bodyStyleCd_WAGON 3D']

# bodyStyleCd_grp1 succeeds together or fails together
# data_train['bodyStyleCd_grp1']=(data_train['bodyStyleCd_UTL4X24D']+data_train['bodyStyleCd_WAG_4X2']+data_train['bodyStyleCd_WAGON']+data_train['bodyStyleCd_WAGON_3D']+data_train['bodyStyleCd_PKP4X23D']+data_train['bodyStyleCd_PKP4X43D'])
# data_test['bodyStyleCd_grp1']=(data_test['bodyStyleCd_UTL4X24D']+data_test['bodyStyleCd_WAG_4X2']+data_test['bodyStyleCd_WAGON']+data_test['bodyStyleCd_WAGON_3D']+data_test['bodyStyleCd_PKP4X23D']+data_test['bodyStyleCd_PKP4X43D'])
# data_train['bodyStyleCd_grp2']=(data_train['bodyStyleCd_PKP_4X4']+data_train['bodyStyleCd_TRK_4X2'])
# data_test['bodyStyleCd_grp2']=(data_test['bodyStyleCd_PKP_4X4']+data_test['bodyStyleCd_TRK_4X2'])


# data_train['NADA_BODY1_grp1']=(data_train['NADA_BODY1_V']+data_train['NADA_BODY1_4'])
# data_test['NADA_BODY1_grp1']=(data_test['NADA_BODY1_V']+data_test['NADA_BODY1_4'])
# data_train['NADA_BODY1_grp2']=(data_train['NADA_BODY1_F']+data_train['NADA_BODY1_I'])
# data_test['NADA_BODY1_grp2']=(data_test['NADA_BODY1_F']+data_test['NADA_BODY1_I'])

# data_train['ENG_MDL_CD_grp1']=(data_train['ENG_MDL_CD_150008']+data_train['ENG_MDL_CD_070014']+data_train['ENG_MDL_CD_050006']+data_train['ENG_MDL_CD_080080'])
# data_train['ENG_MDL_CD_grp2']=(data_train['ENG_MDL_CD_070013']+data_train['ENG_MDL_CD_150085'])
# data_train['ENG_MDL_CD_grp3']=(data_train['ENG_MDL_CD_080009']+data_train['ENG_MDL_CD_050005']+data_train['ENG_MDL_CD_120030']+data_train['ENG_MDL_CD_010007'])
# data_test['ENG_MDL_CD_grp1']=(data_test['ENG_MDL_CD_150008']+data_test['ENG_MDL_CD_070014']+data_test['ENG_MDL_CD_050006']+data_test['ENG_MDL_CD_080080'])
# data_test['ENG_MDL_CD_grp2']=(data_test['ENG_MDL_CD_070013']+data_test['ENG_MDL_CD_150085'])
# data_test['ENG_MDL_CD_grp3']=(data_test['ENG_MDL_CD_080009']+data_test['ENG_MDL_CD_050005']+data_test['ENG_MDL_CD_120030']+data_test['ENG_MDL_CD_010007'])	

formula = ' lossratio ~ RSTRNT_TYP_CD_A + cko_weight_min_spline5k + cko_weight_minMS  +TRK_CAB_CNFG_CD_grp1  + cko_wheelbase_max_ge160  + RSTRNT_TYP_CD_K '
formula += '  + ENG_FUEL_INJ_TYP_CD_S + RSTRNT_TYP_CD_grp1'
# Try all these below 
# formula += ' + MAK_NM_CHEVROLET + MAK_NM_DATSUN + MAK_NM_DODGE + MAK_NM_FREIGHTLINER + MAK_NM_GMC + MAK_NM_IVECO + MAK_NM_MERCEDES_BENZ + MAK_NM_NISSAN + MAK_NM_RAM + MAK_NM_SPRINTER + MAK_NM_TOYOTA'
formula += '  + MAK_NM_MERCEDES_BENZ '
formula += '  + ENG_MDL_CD_070013 + ENG_MDL_CD_080080 + ENG_MDL_CD_110075 + ENG_MDL_CD_080040 '

# testarray=['RSTRNT_TYP_CD_A','cko_weight_min_spline5k','cko_weight_minMS','tonCd_grp1','TRK_CAB_CNFG_CD_grp1','cko_wheelbase_max_ge160','RSTRNT_TYP_CD_K','ENG_FUEL_INJ_TYP_CD_S','bodyStyleCd_grp1','bodyStyleCd_VAN','NADA_BODY1_grp1','NADA_BODY1_grp2','ENG_MDL_CD_080080','ENG_MDL_CD_070013','ENG_MDL_CD_080040','ENG_MDL_CD_110075','MAK_NM_MERCEDES_BENZ','RSTRNT_TYP_CD_grp1']

# for testfield in testarray:
# 	print(testfield);
# 	print(data_train[testfield].sum())


if not os.path.exists('glm'):
	os.makedirs('glm')
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

# ml.actualvsfittedbyfactor(data_train,data_test,'cko_weight_min_spline5k','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'cko_wheelbase_max_ge160','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'tonCd_grp1','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'TRK_CAB_CNFG_CD_grp1','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'bodyStyleCd_grp1','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'NADA_BODY1_grp1','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'NADA_BODY1_grp2','glm/')
# ml.actualvsfittedbyfactor(data_train,data_test,'RSTRNT_TYP_CD_grp1','glm/')
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
