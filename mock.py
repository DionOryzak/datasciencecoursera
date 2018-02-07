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
COLORS=cl.returnCOLORS()

def scatterchart(varname):
	fig, ax = plt.subplots()
	plt.title("Rideshare - Accidents by "+varname)
	plt.ylabel("Reported Accidents")
	plt.xlabel('Miles Driven')
	ax.grid(True)
	ax.scatter(data_train.Miles,data_train.Reported_Accidents,c=[COLORS[varname][x] for x in data_train[varname]],s=10,alpha=0.5)

def onewaychart(varname):	
	combochart_accs=data_train.groupby([varname], as_index=False)['Reported_Accidents'].sum()
	combochart_miles=data_train.groupby([varname], as_index=False)['Miles'].sum()
	combochart_miles['Miles']=combochart_miles['Miles']/1000000
	combochart_freq=combochart_accs['Reported_Accidents']/combochart_miles['Miles']
	categories=combochart_miles[varname]
	numcategories=range(len(categories))
	fig, ax = plt.subplots()
	ax2 = ax.twinx()
	ax.bar(numcategories, combochart_miles['Miles'], label="Miles Driven (Millions)",color='0.7',alpha = 0.5)
	ax2.plot(numcategories,combochart_freq,'r-', marker='o',label='Accidents Per 1 Million Miles', linewidth=1.5)
	ax2.set_ylabel('Accidents Per 1 Million Miles')
	ax.set_ylabel('Miles Driven (Millions)')
	ax.set_xlabel(varname)
	ax.set_xticks(numcategories)
	rotation=np.minimum(np.maximum((len(categories)-4)*10,0),90)	
	ax.set_xticklabels(categories, rotation=rotation)
	ax2.grid()
	lines, labels = ax2.get_legend_handles_labels()
	lines2, labels2 = ax.get_legend_handles_labels()
	ax.legend(lines + lines2, labels + labels2, loc=2)
	plt.suptitle(varname, y=1, fontsize=17)
	plt.title('Accident Rate')

filenametrain="mock_accident_data.csv"
# filenametest="acertaremediation.csv"

print("Starting Program")
print(datetime.datetime.now())

data_train = pd.read_csv(filenametrain, skipinitialspace=True, skiprows=1, names=COLUMNS, dtype=dtype, thousands=',')
# data_test = pd.read_csv(filenametest, skipinitialspace=True, skiprows=1, names=COLUMNS, dtype=dtype)
print('train records: '+str(len(data_train)))
# print('test records: '+str(len(data_test)))

# avgweight = data_train[WEIGHT].mean()
# print(avgweight)

# data_train[WEIGHT]=data_train[WEIGHT]/avgweight

CONTFEATURES = cl.returnCONTFEATURES()
CATFEATURES = cl.returnCATFEATURES()

# Keep only records where Miles > 0 AND Reported_Accidents >= 0
data_train=data_train[(data_train.Miles >0) & (data_train.Reported_Accidents >=0 )]

data_train['acc_per_1000mile']=data_train['Reported_Accidents']/data_train['Miles']*1000

# Remove outliers
data_train=data_train[data_train.acc_per_1000mile <20]
data_train=data_train[data_train.Reported_Accidents <100]

# # Rename #N/A values to "NA"
data_train.Segment= np.where(data_train.Segment.isnull(),"Segment NA",data_train.Segment)
data_train.City= np.where(data_train.City.isnull(),"City NA",data_train.City)
data_train.Product= np.where(data_train.Product.isnull(),"Product NA",data_train.Product)

# data_test=data_train[(data_train.Miles ==1) & (data_train.Segment =='Segment C') ]
# print(data_test)

# Slice and clean Month, Segment, City and Product
data_train['Month']= np.where(data_train['Month_Ending'].str.len()==9,data_train.Month_Ending.str[5:]+"_0"+data_train.Month_Ending.str[0:1],data_train.Month_Ending.str[6:]+"_"+data_train.Month_Ending.str[0:2])
data_train.Segment = data_train.Segment.str[8:]
data_train.City = data_train.City.str[5:]
data_train.Product = data_train.Product.str[8:]
data_train.City= np.where(data_train.City.isin(['0','1','2','3','4','5','6','7','8','9']),'0'+data_train.City,data_train.City)
data_train.Product= np.where(data_train.Product.isin(['0','1','2','3','4','5','6','7','8','9']),'0'+data_train.Product,data_train.Product)

print(data_train)
print('train records: '+str(len(data_train)))

# print(data_train)
DUMMYFEATURES=[]
for feature in CATFEATURES:
  dummiesTrain=pd.get_dummies(data_train[feature],prefix=feature)
  temp=dummiesTrain.sum() 
  dummiesTrain=dummiesTrain[temp[temp<temp.max()].index.values]
  if(len(dummiesTrain.columns)>0):
    dummiesTrain.columns = dummiesTrain.columns.str.replace('\s+', '_')
    data_train=pd.concat([data_train,dummiesTrain],axis=1)
    DUMMYFEATURES += list(dummiesTrain)
print(DUMMYFEATURES)

import numpy as np
import matplotlib.pyplot as plt

# # Scatterplots Miles vs Reported Accidents

# varname="Month"
# scatterchart(varname)
# onewaychart(varname)
# varname='Segment'
# scatterchart(varname)
# onewaychart(varname)
# varname='City'
# scatterchart(varname)
# onewaychart(varname)
# varname='Product'
# scatterchart(varname)
# onewaychart(varname)
# plt.show()

# x=data_train['Reported_Accidents']/data_train['Miles']*1000000
# import numpy as np
# import matplotlib.pyplot as plt
# np.random.seed(0)
# n_bins = 10
# x = np.random.randn(1000, 3)
# print(x)
# fig, ax = plt.subplots()
# ax.hist(x, n_bins, histtype='bar', stacked=True)
# plt.show()

# Do kernel histogram for rate of accs, num miles and num accs 
# https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/


# # Two way Mosaic
# varname1='Product'
# varname2='Segment'
# # accid_sum=data_train.groupby([varname1,varname2],as_index=[varname1,varname2])['Reported_Accidents'].sum()

# # miles_sum=data_train.groupby([varname1,varname2], as_index=False)['Miles'].sum()
# from statsmodels.graphics.mosaicplot import mosaic
# miles_sum=data_train.groupby([varname1,varname2],as_index=[varname1,varname2])['Miles'].sum()
# mosaic(miles_sum)
# miles_sum=data_train.groupby([varname2,varname1],as_index=[varname2,varname1])['Miles'].sum()
# mosaic(miles_sum)

# plt.show()


# Data
x=range(1,6)
y=[ [1,4,6,8,9], [2,2,7,10,12], [2,8,5,10,6] ]
 
# Plot
plt.stackplot(x,y, labels=['A','B','C'])
plt.legend(loc='upper left')
plt.show()


# GLM and Regressions tree on rate of accs


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
#   os.makedirs('glm_'+modeltype+'_'+segment+'/')
# formula = 'LABEL_'+modeltype+' ~   QTR_2 + QTR_3 + cd_coverage_04C + cd_coverage_04D + cd_coverage_17C + cd_coverage_22A + cd_coverage_24A + RiskState_grp1 + ASSETVALUE_BAND_03_1m_1_5m + ASSETVALUE_BAND_04_1_5m_2m + ASSETVALUE_BAND_05_2m_2_5m + ASSETVALUE_BAND_06_2_5m_3m + ASSETVALUE_BAND_07_3m_3_5m + ASSETVALUE_BAND_08_3_5m_5m + ASSETVALUE_BAND_09_5m_10m + ASSETVALUE_BAND_10_10m_20m + ASSETVALUE_BAND_11_20m_high + Business_Prop_Hazard_Missing '
# prediction_glm_train, prediction_glm_test = ml.GLM(data_train_segment, data_train_segment, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm_'+modeltype+'_'+segment+'/',0,'poisson')
# # ml.avecharts(data_train_segment,WEIGHT,prediction_glm_train,LABEL,CONTFEATURES,BANDFEATURES,CATFEATURES,segment,modeltype)
# prediction_glm_train.to_csv('glm_'+modeltype+'_'+segment+'/'+modeltype+'_'+segment+'.csv')


# fig, ax = plt.subplots()
# plt.title("Rideshare - Accident Rate by Segment")
# plt.ylabel("Accidents per 1000 miles")
# plt.xlabel('Miles Driven')
# ax.grid(True)
# ax.scatter(data_train.Miles,data_train.acc_per_1000mile,c=[SEGMENTCOLORS[x] for x in data_train.Segment],s=10,alpha=0.5)
# fig, ax = plt.subplots()
# plt.title("Rideshare - Accident Rate by Product")
# plt.ylabel("Accidents per 1000 miles")
# plt.xlabel('Miles Driven')
# ax.grid(True)
# ax.scatter(data_train.Miles,data_train.acc_per_1000mile,c=[PRODUCTCOLORS[x] for x in data_train.Product],s=10,alpha=0.5)
# fig, ax = plt.subplots()
# plt.title("Rideshare - Accident Rate by City")
# plt.ylabel("Accidents per 1000 miles")
# plt.xlabel('Miles Driven')
# ax.grid(True)
# ax.scatter(data_train.Miles,data_train.acc_per_1000mile,c=[CITYCOLORS[x] for x in data_train.City],s=10,alpha=0.5)


print(datetime.datetime.now())
print("Ending Program")
