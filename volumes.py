import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from operator import itemgetter
from sklearn.tree import DecisionTreeClassifier, export_graphviz
# import graphviz 

def splitdata(data,CONTFEATURES,DUMMYFEATURES,LABEL):
	import numpy as np
	# Split the data into train and test
	from sklearn.model_selection import train_test_split
	predictors = np.asarray(data[CONTFEATURES+DUMMYFEATURES])
	target = np.asarray(data[LABEL])
	# np.minimum(target,40) ## Only do this for the multinomial classification model
	X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=42)
	data_train, data_test = train_test_split(data, test_size=30, random_state=42)
	return data_train, data_test, X_train, X_test, y_train, y_test


def applyband(value,cutoffs):
	if type(cutoffs)==int:
		return value	
	else:	
		# print(cutoffs)
		bandName = 'Other'
		if value<cutoffs[0]:
			bandName='0.<'+str(cutoffs[0])
		elif value>=cutoffs[len(cutoffs)-1]:
			bandName='ge '+str(cutoffs[len(cutoffs)-1])
		else:				
			for index in range(1,len(cutoffs)):
				if value>=cutoffs[index-1]:
					bandName=str(index)+'.'+str(cutoffs[index-1])+' to '+str(cutoffs[index])
					if(index*1-9<=0 & len(cutoffs)*1-10>=0):
						# print('index*1-9<=0 index*1:',index*1, ' len(cutoffs):',len(cutoffs))
						bandName='0'+str(index)+'.'+str(cutoffs[index-1])+' to '+str(cutoffs[index])
						# print(bandName)
					else:
						# print('index*1-9>0 index*1:',index*1,' len(cutoffs):',len(cutoffs))
						bandName=    str(index)+'.'+str(cutoffs[index-1])+' to '+str(cutoffs[index])
						# print(bandName)
		return bandName


filename = 'Volumes_Retention_20171113_v3.csv'

COLUMNS= ['VolumeRetentionDateKey',
'PolicyKey',
'Gross_Written_Premium',
'Book_Premium',
'Gross_Written_Premium__Annualise',
'Book_Premium__Annualised_',
'New_Business_Renewal_GWP',
'Current_Total_Insured_Value',
'Current_Asset_Value',
'Current_Coverage_Limit',
'Motor_Gross_Written_Premium',
'Motor_Gross_Written_Premium__Ann',
'Motor_Current_Asset_Value',
'Property_Gross_Written_Premium',
'Property_Gross_Written_Premium__',
'Property_Current_Asset_Value',
'Other_Gross_Written_Premium',
'Other_Gross_Written_Premium__Ann',
'Casualty_Gross_Written_Premium',
'Casualty_Gross_Written_Premium__',
'Casualty_Current_Asset_Value',
'Workers_Compensation_Gross_Writt',
'Workers_Compensation_Gross_0001',
'Workers_Compensation_Current_Ass',
'Next_Term_Gross_Written_Premium',
'Next_Term_Gross_Written_Pre_0001',
'Next_Term_Book_Premium__Annualis',
'Next_Term_Total_Insured_Value',
'Next_Term_Asset_Value',
'Next_Term_Coverage_Limit',
'Previous_Term_Gross_Written_Prem',
'Previous_Term_Total_Insured_Valu',
'Previous_Term_Asset_Value',
'Previous_Term_Coverage_Limit',
'Flat_Cancelled_Indicator',
'Midterm_Cancelled_Indicator',
'Renewed_Indicator',
'Lapsed_Not_Offered_Indicator',
'Lapsed_Offered_Indicator',
'New_Business_Indicator',
'Account_Classification_Desc',
'Account_Type_Code',
'Primary_Occupation',
'Account_Holder_Gender',
'VIP_Account__Y_N_',
'GuildInsurance_Opt_In__Y_N_',
'Newsletter_Opt_In__Y_N_',
'Other_Company_Opt_In__Y_N_',
'Account_Holder_Postal_State',
'Account_Holder_Title',
'Account_Holder_Electronic_Consen',
'Corporate_Entity_Type',
'Account_Manager_Name__Current_',
'Association_Desc',
'Association_Type',
'VolumeRetentionDateKey_0001',
'Volume_Retention_Date_Calendar_Y',
'Volume_Retention_Date_Calendar_H',
'Volume_Retention_Date_Calendar_Q',
'Volume_Retention_Date_Calendar_M',
'Distribution_Channel',
'PeriodStartKey',
'OriginalPeriodStartKey',
'Term_Number',
'New_Business__Y_N_',
'Stamp_Duty_Exempt__Y_N_',
'Auto_Renewed__Y_N_',
'Active_Policy__Y_N_',
'Employment_Category_Desc',
'Billing_Type',
'Risk_Management_Offer',
'Portal_Source_Name',
'Claim_Count_Band___2_Years',
'Claim_Count_Band___5_Years',
'Claim_Count_Band___Lifetime',
'Claim_Cost_Band___2_Years',
'Claim_Cost_Band___5_Years',
'Claim_Cost_Band___Lifetime',
'Policy_Tenure',
'Source_Product_Desc',
'Source_Offering_Desc',
'Customer_Segment',
'Product_Desc',
'Market_Desc',
'Product_Budget_Desc',
'Product_Display'
]

dtype={
	'VolumeRetentionDateKey':object,	
	'Gross_Written_Premium':object,
	'Book_Premium':object,
	'Gross_Written_Premium__Annualise':object,
	'Book_Premium__Annualised_':object,
	'New_Business_Renewal_GWP':object,
	'Current_Total_Insured_Value':object,
	'Current_Asset_Value':object,
	'Current_Coverage_Limit':object,
	'Motor_Gross_Written_Premium':object,
	'Motor_Gross_Written_Premium__Ann':object,
	'Motor_Current_Asset_Value':object,
	'Property_Gross_Written_Premium':object,
	'Property_Gross_Written_Premium__':object,
	'Property_Current_Asset_Value':object,
	'Other_Gross_Written_Premium':object,
	'Other_Gross_Written_Premium__Ann':object,
	'Casualty_Gross_Written_Premium':object,
	'Casualty_Gross_Written_Premium__':object,
	'Casualty_Current_Asset_Value':object,
	'Workers_Compensation_Gross_Writt':object,
	'Workers_Compensation_Gross_0001':object,
	'Workers_Compensation_Current_Ass':object,
	'Next_Term_Gross_Written_Premium':object,
	'Next_Term_Gross_Written_Pre_0001':object,
	'Next_Term_Book_Premium__Annualis':object,
	'Next_Term_Total_Insured_Value':object,
	'Next_Term_Asset_Value':object,
	'Next_Term_Coverage_Limit':object,
	'Previous_Term_Gross_Written_Prem':object,
	'Previous_Term_Total_Insured_Valu':object,
	'Previous_Term_Asset_Value':object,
	'Previous_Term_Coverage_Limit':object,
	'Flat_Cancelled_Indicator':object,
	'Midterm_Cancelled_Indicator':object,
	'Renewed_Indicator':object,
	'Lapsed_Not_Offered_Indicator':object,
	'Lapsed_Offered_Indicator':object,
	'New_Business_Indicator':object,
	'Account_Classification_Desc':object,
	'Account_Type_Code':object,
	'Primary_Occupation':object,
	'Account_Holder_Gender':object,
	'VIP_Account__Y_N_':object,
	'GuildInsurance_Opt_In__Y_N_':object,
	'Newsletter_Opt_In__Y_N_':object,
	'Other_Company_Opt_In__Y_N_':object,
	'Account_Holder_Postal_State':object,
	'Account_Holder_Title':object,
	'Account_Holder_Electronic_Consen':object,
	'Corporate_Entity_Type':object,
	'Account_Manager_Name__Current_':object,
	'Association_Desc':object,
	'Association_Type':object,
	'VolumeRetentionDateKey_0001':object,
	'Volume_Retention_Date_Calendar_Y':object,
	'Volume_Retention_Date_Calendar_H':object,
	'Volume_Retention_Date_Calendar_Q':object,
	'Volume_Retention_Date_Calendar_M':object,
	'Distribution_Channel':object,
	'PeriodStartKey':object,
	'OriginalPeriodStartKey':object,
	'Term_Number':object,
	'New_Business__Y_N_':object,
	'Stamp_Duty_Exempt__Y_N_':object,
	'Auto_Renewed__Y_N_':object,
	'Active_Policy__Y_N_':object,
	'Employment_Category_Desc':object,
	'Billing_Type':object,
	'Risk_Management_Offer':object,
	'Portal_Source_Name':object,
	'Claim_Count_Band___2_Years':object,
	'Claim_Count_Band___5_Years':object,
	'Claim_Count_Band___Lifetime':object,
	'Claim_Cost_Band___2_Years':object,
	'Claim_Cost_Band___5_Years':object,
	'Claim_Cost_Band___Lifetime':object,
	'Policy_Tenure':object,
	'Source_Product_Desc':object,
	'Source_Offering_Desc':object,
	'Customer_Segment':object,
	'Product_Desc':object,
	'Market_Desc':object,
	'Product_Budget_Desc':object,
	'Product_Display':object
}

outcomes=[
	'Flat_Cancelled_Indicator',
	'Midterm_Cancelled_Indicator',
	'Renewed_Indicator',
	'Lapsed_Not_Offered_Indicator',
	'Lapsed_Offered_Indicator',
	'New_Business_Indicator'
]
CATFEATURES = [
	'Account_Classification_Desc',
	'Account_Type_Code',
	'Primary_Occupation',
	'Account_Holder_Gender',
	'VIP_Account__Y_N_',
	'GuildInsurance_Opt_In__Y_N_',
	'Newsletter_Opt_In__Y_N_',
	'Other_Company_Opt_In__Y_N_',
	'Account_Holder_Postal_State',
	'Account_Holder_Title',
	'Account_Holder_Electronic_Consen',
	'Corporate_Entity_Type',
	'Account_Manager_Name__Current_',
	'Association_Desc',
	'Association_Type',
	'Volume_Retention_Date_Calendar_Y',
	'Volume_Retention_Date_Calendar_H',
	'Volume_Retention_Date_Calendar_Q',
	'Volume_Retention_Date_Calendar_M',
	'Distribution_Channel',
	'New_Business__Y_N_',
	'Stamp_Duty_Exempt__Y_N_',
	'Auto_Renewed__Y_N_',

	# 'Active_Policy__Y_N_',
	# 'Employment_Category_Desc',
	# 'Billing_Type',
	# 'Risk_Management_Offer',
	# 'Portal_Source_Name',
	
	'Claim_Count_Band___2_Years',
	'Claim_Count_Band___5_Years',
	'Claim_Count_Band___Lifetime',
	'Claim_Cost_Band___2_Years',
	'Claim_Cost_Band___5_Years',
	'Claim_Cost_Band___Lifetime',
	'Source_Product_Desc',
	'Source_Offering_Desc',
	'Customer_Segment',
	'Product_Desc',
	'Market_Desc',
	'Product_Budget_Desc',
	'Product_Display'
]

CONTFEATURES = ['VolumeRetentionDateKey',
	'PolicyKey',
	'Gross_Written_Premium',
	'Book_Premium',
	'Gross_Written_Premium__Annualise',
	'Book_Premium__Annualised_',
	'New_Business_Renewal_GWP',
	'Current_Total_Insured_Value',
	'Current_Asset_Value',
	'Current_Coverage_Limit',
	'Motor_Gross_Written_Premium',
	'Motor_Gross_Written_Premium__Ann',
	'Motor_Current_Asset_Value',
	'Property_Gross_Written_Premium',
	'Property_Gross_Written_Premium__',
	'Property_Current_Asset_Value',
	'Other_Gross_Written_Premium',
	'Other_Gross_Written_Premium__Ann',
	'Casualty_Gross_Written_Premium',
	'Casualty_Gross_Written_Premium__',
	'Casualty_Current_Asset_Value',
	'Workers_Compensation_Gross_Writt',
	'Workers_Compensation_Gross_0001',
	'Workers_Compensation_Current_Ass',
	# 'Next_Term_Gross_Written_Premium',
	# 'Next_Term_Gross_Written_Pre_0001',
	# 'Next_Term_Book_Premium__Annualis',
	# 'Next_Term_Total_Insured_Value',
	# 'Next_Term_Asset_Value',
	# 'Next_Term_Coverage_Limit',
	# 'Previous_Term_Gross_Written_Prem',
	# 'Previous_Term_Total_Insured_Valu',
	# 'Previous_Term_Asset_Value',
	# 'Previous_Term_Coverage_Limit',
	# 'VolumeRetentionDateKey_0001',
	# 'PeriodStartKey',
	# 'OriginalPeriodStartKey',
	# 'Term_Number',
	'Policy_Tenure',
]







print("Starting Program")
print(datetime.datetime.now())

data = pd.read_csv(filename, skipinitialspace=True, skiprows=1, names=COLUMNS, dtype=dtype, nrows=200000)
data=data[data['PolicyKey']>0]
data=data[outcomes+CATFEATURES+CONTFEATURES]

print("Length:")
print(len(data))
 
data['outcome'] = (0*data['Flat_Cancelled_Indicator'] + 1*data['Midterm_Cancelled_Indicator'] + 2*data['Renewed_Indicator'] + 3*data['Lapsed_Not_Offered_Indicator'] + 4*data['Lapsed_Offered_Indicator'] + 5*data['New_Business_Indicator'])


# data['outcome']=OUTCOMES.index(data[LABEL])



DUMMYFEATURES=[]
for feature in CATFEATURES:
	dummies=pd.get_dummies(data[feature],prefix=feature)	
	temp=dummies.sum()		
	dummies=dummies[temp[temp<temp.max()].index.values]	
	dummies.columns = dummies.columns.str.replace('\s+', '_')	
	data=pd.concat([data,dummies],axis=1)	
	DUMMYFEATURES += list(dummies)
# for feature in CONTFEATURES:
	# cutoffs=BANDFEATURES[feature]
	# data_train[feature+'Band']=data_train[feature].apply(ml.applyband,args=(cutoffs,))

data_train, data_test, X_train, X_test, y_train, y_test = splitdata(data,CONTFEATURES,DUMMYFEATURES,'outcome')



clf = DecisionTreeClassifier(max_depth=5, min_samples_split=5)
clf = clf.fit(X_train, y_train)
# print(clf.feature_importances_)
print(clf.tree_)
# visualize_tree(clf, CONTFEATURES)

export_graphviz(clf, feature_names= CONTFEATURES+DUMMYFEATURES, out_file='tree.dot')

import os
os.system("dot -Tpng tree.dot -o tree.png")

feature_importance = clf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())

print(feature_importance)



sorted_idx = np.argsort(feature_importance)
sorted_idx=sorted_idx[range(len(sorted_idx)-6,len(sorted_idx))]
print(sorted_idx)

pos = np.arange(sorted_idx.shape[0]) + .5
# plt.subplot(1, 2, 2)
plt.figure(figsize=(14, 6))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(CONTFEATURES+DUMMYFEATURES)[sorted_idx])
# print(np.array(CONTFEATURES+DUMMYFEATURES)[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.savefig('variableimportance.png')
plt.close()


print(datetime.datetime.now())
print("Ending Program")
