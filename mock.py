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
# COLORS=cl.returnCOLORS()

SCATTERCOLORS = {
  "Month": {
    '2015_09':'#BE0032',
    '2015_10':'#222222',
    '2015_11':'#F3C300',
    '2015_12':'#875692',
    '2016_01':'#F38400',
    '2016_02':'#A1CAF1',
    '2016_03':'#F2F3F4',
    '2016_04':'#C2B280',
    '2016_05':'#848482',
    '2016_06':'#008856',
    '2016_07':'#E68FAC',
    '2016_08':'#0067A5',
    '2016_09':'#F99379',
    '2016_10':'#604E97',
    '2016_11':'#F6A600',
    '2016_12':'#B3446C'
  },
  "City": {
    '01':'#BE0032',
    '02':'#222222',
    '03':'#F3C300',
    '04':'#875692',
    '05':'#F38400',
    '06':'#A1CAF1',
    '07':'#F2F3F4',
    '08':'#C2B280',
    '09':'#848482',
    '10':'#008856',
    '11':'#E68FAC',
    '12':'#0067A5',
    '13':'#F99379',
    '14':'#604E97',
    '15':'#F6A600',
    '16':'#B3446C',
    '17':'#DCD300',
    '18':'#882D17',
    '19':'#8DB600',
    '20':'#654522',
    '21':'#E25822',
    '22':'#2B3D26',
    '23':'#F2F3F4',
    '24':'#222222',
    '25':'#F3C300',
    '26':'#875692',
    '27':'#F38400',
    '28':'#A1CAF1',
    '29':'#BE0032',
    '30':'#C2B280',
    '31':'#848482',
    '32':'#008856',
    '33':'#E68FAC',
    '34':'#0067A5',
    '35':'#F99379',
    '36':'#604E97',
    '37':'#F6A600',
    '38':'#B3446C',
    '39':'#DCD300',
    '40':'#882D17',
    '41':'#8DB600',
    '42':'#654522',
    '43':'#E25822',
    '44':'#2B3D26',
    '45':'#F2F3F4',
    '46':'#222222',
    '47':'#F3C300',
    '48':'#875692',
    '49':'#F38400',
    '50':'#A1CAF1',
    '51':'#BE0032',
    '52':'#C2B280',
    '53':'#848482',
    '54':'#008856',
    '55':'#E68FAC',
    '56':'#0067A5',
    '57':'#F99379',
    '58':'#604E97',
    '59':'#F6A600',
    '60':'#B3446C',
    '61':'#DCD300',
    '62':'#882D17',
    '63':'#8DB600',
    '64':'#654522',
    '65':'#E25822',
    '66':'#2B3D26',
    '67':'#F2F3F4',
    '68':'#222222',
    '69':'#F3C300',
    '70':'#875692',
    '71':'#F38400',
    '72':'#A1CAF1',
    '73':'#BE0032',
    '74':'#C2B280',
    '75':'#848482',
    '76':'#008856',
    '77':'#E68FAC',
    '78':'#0067A5',
    'NA':'#F99379'
  },
  "Product": {
    '01':'#BE0032',
    '02':'#222222',
    '03':'#F3C300',
    '04':'#875692',
    '05':'#F38400',
    '06':'#A1CAF1',
    '07':'#F2F3F4',
    '08':'#C2B280',
    '09':'#848482',
    '10':'#008856',
    '11':'#E68FAC',
    '12':'#0067A5',
    '13':'#F99379',
    '14':'#604E97',
    '15':'#F6A600',
    '16':'#B3446C',
    '17':'#DCD300',
    '18':'#882D17',
    '19':'#8DB600',
    '20':'#654522',
    '21':'#E25822',
    '22':'#2B3D26'
  },
  "Segment": {
    "A": "#BE0032", 
    "B": "#222222", 
    "C": "#F3C300", 
    "D": "#875692", 
    "E": "#F38400", 
    "F": "#A1CAF1", 
    "G": "#F2F3F4", 
    "H": "#C2B280", 
    "I": "#848482", 
    "J": "#008856"
  }
}

AREACOLORS = {
  "Month": (
    '#BE0032',
    '#222222',
    '#F3C300',
    '#875692',
    '#F38400',
    '#A1CAF1',
    '#F2F3F4',
    '#C2B280',
    '#848482',
    '#008856',
    '#E68FAC',
    '#0067A5',
    '#F99379',
    '#604E97',
    '#F6A600',
    '#B3446C'
  ),
  "City": (
    '#BE0032',
    '#222222',
    '#F3C300',
    '#875692',
    '#F38400',
    '#A1CAF1',
    '#F2F3F4',
    '#C2B280',
    '#848482',
    '#008856',
    '#E68FAC',
    '#0067A5',
    '#F99379',
    '#604E97',
    '#F6A600',
    '#B3446C',
    '#DCD300',
    '#882D17',
    '#8DB600',
    '#654522',
    '#E25822',
    '#2B3D26',
    '#F2F3F4',
    '#222222',
    '#F3C300',
    '#875692',
    '#F38400',
    '#A1CAF1',
    '#BE0032',
    '#C2B280',
    '#848482',
    '#008856',
    '#E68FAC',
    '#0067A5',
    '#F99379',
    '#604E97',
    '#F6A600',
    '#B3446C',
    '#DCD300',
    '#882D17',
    '#8DB600',
    '#654522',
    '#E25822',
    '#2B3D26',
    '#F2F3F4',
    '#222222',
    '#F3C300',
    '#875692',
    '#F38400',
    '#A1CAF1',
    '#BE0032',
    '#C2B280',
    '#848482',
    '#008856',
    '#E68FAC',
    '#0067A5',
    '#F99379',
    '#604E97',
    '#F6A600',
    '#B3446C',
    '#DCD300',
    '#882D17',
    '#8DB600',
    '#654522',
    '#E25822',
    '#2B3D26',
    '#F2F3F4',
    '#222222',
    '#F3C300',
    '#875692',
    '#F38400',
    '#A1CAF1',
    '#BE0032',
    '#C2B280',
    '#848482',
    '#008856',
    '#E68FAC',
    '#0067A5',
    '#F99379'
),
"Product": (
  '#BE0032',
  '#222222',
  '#F3C300',
  '#875692',
  '#F38400',
  '#A1CAF1',
  '#F2F3F4',
  '#C2B280',
  '#848482',
  '#008856',
  '#E68FAC',
  '#0067A5',
  '#F99379',
  '#604E97',
  '#F6A600',
  '#B3446C',
  '#DCD300',
  '#882D17',
  '#8DB600',
  '#654522',
  '#E25822',
  '#2B3D26'
  ),
  "Segment": (
    "#BE0032", 
    "#222222", 
    "#F3C300", 
    "#875692", 
    "#F38400", 
    "#A1CAF1", 
    "#F2F3F4", 
    "#C2B280", 
    "#848482", 
    "#008856"
  )
}

def scatterchart(varname):
  fig, ax = plt.subplots()
  plt.title("Rideshare - Accidents by "+varname)
  plt.ylabel("Reported Accidents")
  plt.xlabel('Miles Driven (Millions)')
  ax.grid(True)
  ax.scatter(data_train['Million_Miles'],data_train['Reported_Accidents'],c=[SCATTERCOLORS[varname][x] for x in data_train[varname]],s=10,alpha=0.9)

def onewaychart(varname): 
  combochart_data=data_train.groupby([varname], as_index=False).sum()
  combochart_data['Freq']=combochart_data['Reported_Accidents']/combochart_data['Million_Miles']
  categories=combochart_data[varname]
  numcategories=range(len(categories))
  fig, ax = plt.subplots()
  ax2 = ax.twinx()
  ax.bar(numcategories, combochart_data['Million_Miles'], label="Miles Driven (Millions)",color='0.7',alpha = 0.5)
  ax2.plot(numcategories,combochart_data['Freq'],'r-', marker='o',label='Accidents Per 1 Million Miles', linewidth=1.5)
  ax2.set_ylabel('Accident Rate Per 1 Million Miles')
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

def areachart(varname):
  fig, ax = plt.subplots()
  areachart_data=data_train.groupby([varname,'Accs_Per_Million_Miles'], as_index=False)['Million_Miles'].sum()
  categories=areachart_data.groupby([varname])[varname].unique()
  labels=[]
  histo_count_outer=[]
  for catlevel in categories:
    labels.append(catlevel[0])
    histo_count_inner=[]
    for i in range(60):
      rowmatch = areachart_data[(areachart_data[varname]==catlevel[0]) & (areachart_data["Accs_Per_Million_Miles"]==i)]  
      if len(rowmatch)>0:
        histo_count_inner.append(rowmatch.iloc[0]["Million_Miles"])
      else:  
        histo_count_inner.append(0)
    histo_count_outer.append(histo_count_inner)
  x=range(60)
  # ('blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold',  'darkred', 'darkblue')
  plt.stackplot(x,histo_count_outer, labels=labels,colors=AREACOLORS[varname])
  plt.legend(loc='center right',prop={'size': 6},labelspacing=-2.5)
  plt.xlabel('Accident Rate Per 1 Million Miles')
  plt.ylabel('Miles Driven (Millions)')
  plt.title('Miles Driven at Various Accident Rates By '+varname)


def trendmixchart(varname):
  fig, ax = plt.subplots()
  areachart_data=data_train.groupby([varname,'Month'], as_index=False)['Million_Miles'].sum()
  categories=areachart_data.groupby([varname])[varname].unique()
  months=areachart_data.groupby(['Month'], as_index=False)['Million_Miles'].sum()
  labels=[]
  histo_count_outer=[]
  for catlevel in categories:
    labels.append(catlevel[0])
    histo_count_inner=[]
    for i in range(len(months)):
      rowmatch = areachart_data[(areachart_data[varname]==catlevel[0]) & (areachart_data["Month"]==months['Month'][i])]  
      if len(rowmatch)>0:
        histo_count_inner.append(rowmatch.iloc[0]["Million_Miles"]/months['Million_Miles'][i])
      else:  
        histo_count_inner.append(0)  
    histo_count_outer.append(histo_count_inner)
  x=range(len(months))
  dfList = months['Month'].apply(pd.Series).stack().tolist()
  plt.xticks(x,dfList,size='small')
  ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
  ax.stackplot(x,histo_count_outer, labels=labels,colors=AREACOLORS[varname])
  plt.xlabel('Month')
  plt.ylabel('Mix of Miles Driven ')
  plt.title('Trend of Miles Driven By '+varname)  

filenametrain="mock_accident_data.csv"

print("Starting Program")
print(datetime.datetime.now())

data_train = pd.read_csv(filenametrain, skipinitialspace=True, skiprows=1, names=COLUMNS, dtype=dtype, thousands=',')
# data_test = pd.read_csv(filenametest, skipinitialspace=True, skiprows=1, names=COLUMNS, dtype=dtype)
# print('train records: '+str(len(data_train)))
# print('test records: '+str(len(data_test)))

# avgweight = data_train[WEIGHT].mean()
# print(avgweight)

# data_train[WEIGHT]=data_train[WEIGHT]/avgweight

CONTFEATURES = cl.returnCONTFEATURES()
CATFEATURES = cl.returnCATFEATURES()

data_train['Million_Miles']=data_train['Miles']/1000000
data_train['Accs_Per_Million_Miles']=np.floor(data_train['Reported_Accidents']/data_train['Million_Miles'])

# Keep only records where Miles > 0 AND Reported_Accidents >= 0
data_train=data_train[(data_train.Miles >0) & (data_train.Reported_Accidents >=0 )]

# Remove outliers
data_train=data_train[data_train['Accs_Per_Million_Miles']<20000]
data_train=data_train[data_train['Reported_Accidents'] <100]

# # Rename #N/A values to "NA"
data_train.Segment= np.where(data_train.Segment.isnull(),"Segment NA",data_train.Segment)
data_train.City= np.where(data_train.City.isnull(),"City NA",data_train.City)
data_train.Product= np.where(data_train.Product.isnull(),"Product NA",data_train.Product)

# Slice and clean Month, Segment, City and Product
data_train['Month']= np.where(data_train['Month_Ending'].str.len()==9,data_train.Month_Ending.str[5:]+"_0"+data_train.Month_Ending.str[0:1],data_train.Month_Ending.str[6:]+"_"+data_train.Month_Ending.str[0:2])
data_train['monthCONT']=data_train.Month.str[0:4].astype('float', copy=False)+(data_train.Month.str[5:7].astype('float', copy=False)-1)/12
data_train.Segment = data_train.Segment.str[8:]
data_train.City = data_train.City.str[5:]
data_train.Product = data_train.Product.str[8:]
data_train.City= np.where(data_train.City.isin(['0','1','2','3','4','5','6','7','8','9']),'0'+data_train.City,data_train.City)
data_train.Product= np.where(data_train.Product.isin(['0','1','2','3','4','5','6','7','8','9']),'0'+data_train.Product,data_train.Product)

catVariables=["Month","Segment",'City','Product']

# # One way charts
# for varname in catVariables:
#   scatterchart(varname)
#   onewaychart(varname)
#   areachart(varname)
# trendmixchart("Segment")
# trendmixchart('City')
# trendmixchart('Product')  


# # Two way Mosaic
# from statsmodels.graphics.mosaicplot import mosaic
# for i in range(len(catVariables)):
#   for j in range(len(catVariables)):
#     if i != j:
#       miles_sum=data_train.groupby([catVariables[i],catVariables[j]],as_index=[catVariables[i],catVariables[j]])['Miles'].sum()
#       accid_sum=data_train.groupby([catVariables[i],catVariables[j]],as_index=[catVariables[i],catVariables[j]])['Accs_Per_Million_Miles'].sum()
#       miles_sum=miles_sum[miles_sum>0]
#       accid_sum=accid_sum[accid_sum>0]
#       mosaic(miles_sum)
#       mosaic(accid_sum)
      # Plot axis labels, shrink text and if possible adjust color to frequency

plt.show()


# GLM and Regressions tree on rate of accs


# # print(data_train)
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


LABEL = 'Accs_Per_Million_Miles'
WEIGHT = 'Million_Miles'
if not os.path.exists('glm/'):
  os.makedirs('glm/')
# formula = 'Accs_Per_Million_Miles ~ monthCONT + Segment_B + Segment_C + Segment_D + Segment_E + Segment_F + Segment_G + Segment_H + Segment_I + Segment_J + City_01 + City_03 + City_04 + City_05 + City_06 + City_07 + City_08 + City_09 + City_10 + City_11 + City_12 + City_13 + City_14 + City_15 + City_16 + City_17 + City_18 + City_19 + City_20 + City_21 + City_22 + City_23 + City_24 + City_25 + City_26 + City_27 + City_28 + City_29 + City_30 + City_31 + City_32 + City_33 + City_34 + City_35 + City_36 + City_37 + City_38 + City_39 + City_40 + City_41 + City_42 + City_43 + City_44 + City_45 + City_46 + City_47 + City_48 + City_49 + City_50 + City_51 + City_52 + City_53 + City_54 + City_55 + City_56 + City_57 + City_58 + City_59 + City_60 + City_61 + City_62 + City_63 + City_64 + City_65 + City_66 + City_67 + City_68 + City_69 + City_70 + City_71 + City_72 + City_73 + City_74 + City_76 + City_NA + Product_01 + Product_02 + Product_03 + Product_05 + Product_06 + Product_07 + Product_08 + Product_09 + Product_10 + Product_11 + Product_12 + Product_13 + Product_14 + Product_15 + Product_17 + Product_18 + Product_19 + Product_20 + Product_21 + Product_22 '
formula = 'Accs_Per_Million_Miles ~ monthCONT + Segment_B + Segment_C + Segment_D + Segment_E + Segment_F + Segment_G + Segment_H + Segment_I + Segment_J  '
prediction_glm_train, prediction_glm_test = ml.GLM(data_train, data_train, CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,'glm/',0,'poisson')
# ml.avecharts(data_train_segment,WEIGHT,prediction_glm_train,LABEL,CONTFEATURES,BANDFEATURES,CATFEATURES,segment,modeltype)

data_train['weight'] = data_train[WEIGHT]
data_train['predictedValue'] = prediction_glm_train*data_train['weight']
data_train['actualValue'] = data_train[LABEL]*data_train['weight']

for feature in catVariables:
  ml.actualvsfittedbyfactor(data_train,data_train,feature,'glm/')  





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



print(datetime.datetime.now())
print("Ending Program")
