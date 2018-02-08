
def exploratorygraphs(data,CONTFEATURES,LABEL):
	# print(data)
	sns.set_style("darkgrid",{"grid.color":".9","axes.facecolor":"white"})
	sns.set_palette("YlGnBu")
	#############################################################################
	# Three scatterplots on one canvas 
	g=sns.PairGrid(data,x_vars=CONTFEATURES,y_vars=LABEL,aspect=1,size=2)
	g.map(plt.scatter,alpha=1);
	# # #############################################################################
	# Three scatter kernel plots on separate canvases
	for name in CONTFEATURES:
		sns.jointplot(LABEL,name,data=data,kind='kde',stat_func=None)
	############################################################################
	# Many pairwise scatterplots and histograms on one canvas
	g=sns.PairGrid(data,vars=CONTFEATURES,size=2)
	g=g.map_diag(plt.hist)
	g=g.map_offdiag(plt.scatter,alpha=1)
	# #############################################################################
	# Many histograms, with kernel overlay, on separate canvases
	plotcols=CONTFEATURES
	for name in plotcols:
		plt.figure(figsize=(4,2))
		plt.suptitle(name)
		sns.distplot(data[name],color='teal',norm_hist=True)
	# # #############################################################################

def visualizerandomredisuals(heading,prediction,y_test):
	import matplotlib.pyplot as plt
	import numpy as np
	split=6000
	x_as = np.arange(len(prediction))[0:split]    
	ar2=list(zip(prediction[0:split],y_test[0:split]))
	ar1=list(zip(x_as,x_as))
	fig=plt.figure(figsize=(20,10))
	fig=plt.subplot()
	for i in range(len(ar1)):
		plt.plot(ar1[i],ar2[i],'k-',lw=1, color='#2fa1bc')
	plot_pred = plt.scatter(x_as,prediction[0:split], color='#94cfb8')
	plot_test = plt.scatter(x_as,y_test[0:split], color='#a8eb7a')
	plt.title(heading)
	plt.legend([plot_pred,plot_test],["predicted","actual"])
	plt.savefig('visualizerandomredisuals.png')
	plt.close()

def visualizeorderedredisuals(heading,prediction,y_test):
	import matplotlib.pyplot as plt
	import numpy as np
	from operator import itemgetter
	split=6000
	x_as = np.arange(len(prediction))[0:split]    
	ar2=list(zip(prediction[0:split],y_test[0:split]))
	ar2.sort(key=itemgetter(0))
	ar2=np.asarray(ar2)
	# print(ar2)
	ar1=list(zip(x_as,x_as))
	fig=plt.figure(figsize=(20,10))
	fig=plt.subplot()
	for i in range(len(ar1)):
		plt.plot(ar1[i],ar2[i],'k-',lw=1, color='#2fa1bc')
	plot_pred = plt.scatter(x_as,ar2[:,0], color='#94cfb8')
	plot_test = plt.scatter(x_as,ar2[:,1], color='#a8eb7a')
	plt.title(heading)
	plt.legend([plot_pred,plot_test],["predicted","actual"])
	plt.savefig('visualizeorderedredisuals.png')
	plt.close()

def actualvsfitted(heading,prediction,y_test):
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	ax.scatter(prediction, y_test)
	ax.set_title(heading)
	ax.set_ylabel('Actual values')
	ax.set_xlabel('Fitted values')
	plt.savefig('actualvsfitted.png')
	plt.close()

def plotpearsonresiduals(heading,prediction,resid_pearson):
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	ax.scatter(prediction, resid_pearson,s=10)
	ax.hlines(0, 0,max(prediction))
	# ax.set_xlim(0, 1)
	ax.set_title(heading)
	ax.set_ylabel('Pearson Residuals')
	ax.set_xlabel('Fitted values')	
	plt.savefig('plotpearsonresiduals.png')		
	plt.close()

def histogramstandardizeddevianceresiduals(heading,prediction,resid_deviance):
	import matplotlib.pyplot as plt
	from scipy import stats
	fig, ax = plt.subplots()
	resid = resid_deviance.copy()
	resid_std = stats.zscore(resid)
	ax.hist(resid_std, bins=25)
	ax.set_title(heading)
	plt.savefig('histogramstandardizeddevianceresiduals.png')	
	plt.close()

def qqplot(resid_deviance):
	import matplotlib.pyplot as plt
	from statsmodels import graphics
	graphics.gofplots.qqplot(resid_deviance, line='r')
	plt.savefig('qqplot.png')	
	plt.close()

# actualvsfittedbyfactor(temp['weight'],temp['actualValue'],temp['predictedValue'],temp[groupByVariable],groupByVariable,folder)
# actualvsfittedbyfactor(exposure,actual,fitted,categories,feature,folder)
def actualvsfittedbyfactor(data_train,data_test,groupByVariableName,folder):
	import matplotlib.pyplot as plt
	import matplotlib.gridspec as gridspec
	import numpy as np

	print(groupByVariableName)
	print(data_train.groupby([groupByVariableName], as_index=False)['weight'].sum())

	temp=data_train.groupby([groupByVariableName], as_index=False)['weight'].sum()
	predictedValues=data_train.groupby([groupByVariableName], as_index=False)['predictedValue'].sum()
	actualValues=data_train.groupby([groupByVariableName], as_index=False)['actualValue'].sum() 
	fitted=predictedValues['predictedValue']/temp['weight']	
	actual=actualValues['actualValue']/temp['weight']	
	exposure=temp['weight']
	categories=temp[groupByVariableName]
	numcategories=range(len(categories))

	fig=plt.figure(figsize=(15, 6))
	# plt.figure(figsize=(6, 6))
	# plt.subplot(1, 2, 1)	
	gs = gridspec.GridSpec(1, 3, width_ratios=[3,1,3])
	# gs = gridspec.GridSpec(1, 2,width_ratios=[2,1])	

	# ax = fig.add_subplot(121)	
	ax = plt.subplot(gs[0])
	ax2 = ax.twinx()
	ax.bar(numcategories, exposure, label='Exposure',color='0.7',alpha = 0.5)
	ax2.plot(numcategories,actual,'r-', marker='o',label='Actual', linewidth=1.5)
	ax2.plot(numcategories,fitted, color='#FF9133', marker='o',label='Predicted' ,linewidth=1.5)
	ax2.set_ylabel('Actual,Fitted')
	ax.set_ylabel('Exposure')
	ax.set_xlabel(groupByVariableName)
	ax.set_xticks(numcategories)
	# print(len(categories))
	rotation=np.minimum(np.maximum((len(categories)-4)*10,0),90)	
	# print(rotation)
	ax.set_xticklabels(categories, rotation=rotation)
	ax2.grid()
	lines, labels = ax2.get_legend_handles_labels()
	lines2, labels2 = ax.get_legend_handles_labels()
	ax.legend(lines + lines2, labels + labels2, loc=2)
	plt.suptitle(groupByVariableName, y=1, fontsize=17)
	plt.title('Actual vs Fitted: Train')





	temp=data_test.groupby([groupByVariableName], as_index=False)['weight'].sum()
	predictedValues=data_test.groupby([groupByVariableName], as_index=False)['predictedValue'].sum()
	actualValues=data_test.groupby([groupByVariableName], as_index=False)['actualValue'].sum() 
	fitted=predictedValues['predictedValue']/temp['weight']
	actual=actualValues['actualValue']/temp['weight']
	exposure=temp['weight']
	categories=temp[groupByVariableName]
	numcategories=range(len(categories))

	# plt.subplot(1, 2, 2)
	# ax = fig.add_subplot(122)
	ax = plt.subplot(gs[2])
	ax2 = ax.twinx()
	ax.bar(numcategories, exposure, label='Exposure',color='0.7',alpha = 0.5)
	ax2.plot(numcategories,actual,'r-', marker='o',label='Actual', linewidth=1.5)
	ax2.plot(numcategories,fitted, color='#FF9133', marker='o',label='Predicted' ,linewidth=1.5)
	ax2.set_ylabel('Actual,Fitted')
	ax.set_ylabel('Exposure')
	ax.set_xlabel(groupByVariableName)
	ax.set_xticks(numcategories)
	# print(len(categories))
	rotation=np.minimum(np.maximum((len(categories)-4)*10,0),90)	
	# print(rotation)
	ax.set_xticklabels(categories, rotation=rotation)
	ax2.grid()
	lines, labels = ax2.get_legend_handles_labels()
	lines2, labels2 = ax.get_legend_handles_labels()
	ax.legend(lines + lines2, labels + labels2, loc=2)
	plt.suptitle(groupByVariableName, y=1, fontsize=17)
	plt.title('Actual vs Fitted: Test')
	plt.savefig(folder+groupByVariableName+'.png')
	plt.close()

def splitdata(data,CONTFEATURES,DUMMYFEATURES,LABEL):
	import numpy as np
	# Split the data into train and test
	from sklearn.model_selection import train_test_split
	# predictors = np.asarray(data[CONTFEATURES+DUMMYFEATURES])
	# target = np.asarray(data[LABEL])
	# np.minimum(target,40) ## Only do this for the multinomial classification model
	# X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=42)
	data_train, data_test = train_test_split(data, test_size=30, random_state=42)
	X_train = np.asarray(data_train[CONTFEATURES+DUMMYFEATURES])
	y_train = np.asarray(data_train[LABEL])
	X_test = np.asarray(data_test[CONTFEATURES+DUMMYFEATURES])
	y_test = np.asarray(data_test[LABEL])
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

def linearregression(X_train, X_test, y_train, y_test):
	from sklearn import linear_model
	from sklearn import metrics
	glm=linear_model.LinearRegression()
	glm.fit(X_train, y_train)
	print(glm.intercept_, "linear regression intercept")
	print(glm.coef_, "linear regression coefficients")
	prediction_lr_train = glm.predict(X_train)
	prediction_lr_test = glm.predict(X_test)
	print(metrics.mean_squared_error(prediction_lr_test, y_test), "linear regression mean squared error")
	print(metrics.r2_score(prediction_lr_test, y_test), "linear regression r2 score")
	confidence = glm.score(X_test, y_test)
	print('confidence',confidence)
	visualizerandomredisuals(heading="Linear Regression Random Residuals",prediction=prediction_lr_test,y_test=y_test)
	visualizeorderedredisuals(heading="Linear Regression Ordered Residuals",prediction=prediction_lr_test,y_test=y_test)
	return prediction_lr_train, prediction_lr_test

def logisticregression(X_train, X_test, y_train, y_test):
	from sklearn import linear_model
	from sklearn import metrics
	import pickle
	glm=linear_model.LogisticRegression()
	glm.fit(X_train, y_train)
	prediction_lr_train = glm.predict_proba(X_train)
	prediction_lr_test = glm.predict_proba(X_test)
	prediction_lr_train=prediction_lr_train[:,1]
	prediction_lr_test=prediction_lr_test[:,1]
	# print(prediction_lr_train)
	print(metrics.mean_squared_error(prediction_lr_test, y_test), "linear regression mean squared error")
	print(metrics.r2_score(prediction_lr_test, y_test), "linear regression r2 score")
	confidence = glm.score(X_test, y_test)
	print('confidence',confidence)
	temptest=glm.predict(X_test)
	precision = metrics.precision_score(temptest, y_test)
	recall = metrics.recall_score(temptest, y_test)	
	print('precision',precision)
	print('recall',recall)
	print(glm.intercept_, "linear regression intercept")
	print(glm.coef_, "linear regression coefficients")
	visualizerandomredisuals(heading="Linear Regression Random Residuals",prediction=prediction_lr_test,y_test=y_test)
	visualizeorderedredisuals(heading="Linear Regression Ordered Residuals",prediction=prediction_lr_test,y_test=y_test)
	return prediction_lr_train, prediction_lr_test	

def GLM(data_train,data_test,CONTFEATURES,CATFEATURES,LABEL,WEIGHT,formula,filepath,picklefile,disttype):

	# pickle: 0=none; 1=save; 2=load

	# Generalized Linear Model
	import statsmodels.api as sm
	import statsmodels.formula.api as smf
	import pickle

	# print(data_train)
	if picklefile<2 :
		if disttype== 'poisson':
			glm = smf.glm(formula=formula, data=data_train, freq_weights=data_train[WEIGHT], family=sm.families.Poisson(sm.families.links.log)).fit(maxiter=500)
		elif disttype== 'gamma':	
			glm = smf.glm(formula=formula, data=data_train, freq_weights=data_train[WEIGHT], family=sm.families.Gamma(sm.families.links.log)).fit(maxiter=500)
		elif disttype== 'tweedie':
			glm = smf.glm(formula=formula, data=data_train, freq_weights=data_train[WEIGHT], family=sm.families.Tweedie(sm.families.links.log,1.67)).fit(maxiter=500)
		if picklefile==1 :
				pickle.dump(glm,open(filepath+'glm_model.sav','wb'))

	if picklefile==2 :
		glm=pickle.load(open(filepath+'glm_model.sav','rb'))

	print(glm.summary())
	text_file = open(filepath+'glmparams.txt', "w")
	text_file.write(str(glm.summary()))
	text_file.close()   

	prediction_glm_train = glm.predict(data_train)
	prediction_glm_test = glm.predict(data_test)

	# print('Parameters: ', glm.params)
	# print('T-values: ', glm.tvalues)
	# Models: Gaussian, Poisson, Gamma, Binomial
	# Links: Identity, log, logit, inverse 
	# #############################################################################

	actualvsfitted('Actual vs Fitted: GLM',prediction_glm_test,data_test[LABEL])
	plotpearsonresiduals('Pearson Residuals: GLM',prediction_glm_train,glm.resid_pearson)
	histogramstandardizeddevianceresiduals('Histogram of standardized deviance residuals',prediction_glm_train,glm.resid_deviance)
	qqplot(glm.resid_deviance)
	visualizerandomredisuals(heading="GLM Random Residuals",prediction=prediction_glm_test,y_test=data_test[LABEL])
	visualizeorderedredisuals(heading="GLM Ordered Residuals",prediction=prediction_glm_test,y_test=data_test[LABEL])
	# #############################################################################
	

	# data_train['weight'] = data_train[WEIGHT]
	# data_train['predictedValue'] = prediction_glm_train*data_train['weight']
	# data_train['actualValue'] = data_train[LABEL]*data_train['weight']

	
	# data_test['weight'] = data_test[WEIGHT]
	# data_test['predictedValue'] = prediction_glm_test*data_test['weight']
	# data_test['actualValue'] = data_test[LABEL]*data_test['weight']

	# for groupByVariable in (CATFEATURES):
	# 	groupByVariableName=groupByVariable
	# 	actualvsfittedbyfactor(data_train,data_test,groupByVariableName,folder)
	# for groupByVariable in (CONTFEATURES):
	# 	groupByVariableName=groupByVariable+'Band'		
	# 	actualvsfittedbyfactor(data_train,data_test,groupByVariableName,folder)	
	# #############################################################################	
	return prediction_glm_train, prediction_glm_test

def randomforest(X_train, X_test, y_train, y_test):
	from sklearn import ensemble
	from sklearn import metrics
	rf=ensemble.RandomForestRegressor()
	rf.fit(X_train,y_train)
	prediction_rf_train = rf.predict(X_train)
	prediction_rf_test = rf.predict(X_test)
	print(metrics.mean_squared_error(prediction_rf_test, y_test), "random forest mean squared error")
	print(metrics.r2_score(prediction_rf_test, y_test), "random forest r2 score")
	confidence = rf.score(X_test, y_test)
	print('confidence',confidence)
	visualizerandomredisuals(heading="Random Forest Random Residuals",prediction=prediction_rf_test,y_test=y_test)
	visualizeorderedredisuals(heading="Random Forest Ordered Residuals",prediction=prediction_rf_test,y_test=y_test)
	return prediction_rf_train, prediction_rf_test


def gbm(X_train, X_test, y_train, y_test,CONTFEATURES,DUMMYFEATURES,folder):
	import matplotlib.pyplot as plt
	import numpy as np
	from sklearn import ensemble
	from sklearn import metrics
	import os

	if ( os.path.exists(folder)) :	
		n_estimators=100
	else:
		n_estimators=2

	params = {'n_estimators': n_estimators, 'max_depth': 4, 'min_samples_split': 3,'learning_rate': 0.01, 'loss': 'ls'}
	gbm=ensemble.GradientBoostingRegressor(**params)
	gbm.fit(X_train,y_train)
	prediction_gbm_train = gbm.predict(X_train)
	prediction_gbm_test = gbm.predict(X_test)
	confidence_train = gbm.score(X_train, y_train)
	confidence_test = gbm.score(X_test, y_test)
	print("GBM mean squared error on train set: ",metrics.mean_squared_error(prediction_gbm_train, y_train))
	if ( os.path.exists(folder)) :
		print("GBM R2 score on train set: ", metrics.r2_score(prediction_gbm_train, y_train))
		print('Confidence on train set: ',confidence_train)
		print("GBM mean squared error on test set: ",metrics.mean_squared_error(prediction_gbm_test, y_test))
		print("GBM R2 score on test set: ", metrics.r2_score(prediction_gbm_test, y_test))
		print('Confidence on test set: ',confidence_test)
		# #############################################################################
		fig=plt.figure()
		# Plot feature importance
		feature_importance = gbm.feature_importances_
		feature_importance = 100.0 * (feature_importance / feature_importance.max())
		sorted_idx = np.argsort(feature_importance)

		print(np.array(CONTFEATURES+DUMMYFEATURES)[sorted_idx])
		text_file = open(folder+'gbmfactors.txt', "w")
		text_file.write(str(np.array(CONTFEATURES+DUMMYFEATURES)[sorted_idx]))
		text_file.close()  

		sorted_idx=sorted_idx[range(len(sorted_idx)-8,len(sorted_idx))]

		pos = np.arange(sorted_idx.shape[0]) + .5
		# plt.subplot(1, 2, 2)
		plt.figure(figsize=(14, 6))
		plt.barh(pos, feature_importance[sorted_idx], align='center')
		plt.yticks(pos, np.array(CONTFEATURES+DUMMYFEATURES)[sorted_idx])

		plt.xlabel('Relative Importance')
		plt.title('Variable Importance')
		plt.savefig(folder+'variableimportance.png')
		plt.close()
		# #############################################################################
		# Plot training deviance
		test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
		for i, y_pred in enumerate(gbm.staged_predict(X_test)):
		    test_score[i] = gbm.loss_(y_test, y_pred)
		# plt.figure(figsize=(6, 6))
		# plt.subplot(1, 2, 1)
		plt.title('Deviance')
		plt.plot(np.arange(params['n_estimators']) + 1, gbm.train_score_, 'b-', label='Training Set Deviance')
		plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-', label='Test Set Deviance')
		plt.legend(loc='upper right')
		plt.xlabel('Boosting Iterations')
		plt.ylabel('Deviance')
		plt.savefig(folder+'deviance.png')
		plt.close()
		#############################################################################
		graph_features=CONTFEATURES+DUMMYFEATURES	
		fig,axs=ensemble.partial_dependence.plot_partial_dependence(gbm,X_train,graph_features,feature_names=graph_features)
		fig.subplots_adjust(top=0.99, bottom=0.09)
		# plt.figure(figsize=(6, 16))
		plt.savefig(folder+'partialdependence.png')
		plt.close()
		# i=0
		# for graph_feature_element in graph_features:
		# 	graph_feature=[]
		# 	graph_feature.append(graph_feature_element)
		# 	fig,axs=ensemble.partial_dependence.plot_partial_dependence(gbm,X_train,graph_feature,feature_names=graph_feature)
		# 	# fig.subplots_adjust(top=1.0, bottom=0.1)		
		# 	plt.savefig(folder+graph_feature_element+'_partialdependence.png')		
		# 	plt.close()	
		# 	i += 1
		#############################################################################
		# visualizerandomredisuals(heading="GBM Random Residuals",prediction=prediction_gbm_test,y_test=y_test)
		# visualizeorderedredisuals(heading="GBM Ordered Residuals",prediction=prediction_gbm_test,y_test=y_test)
		############################################################################
		# plt.show()	
	
	return prediction_gbm_train, prediction_gbm_test


def gbmgridsearch(X_train, y_train):
	from sklearn import ensemble
	from sklearn import model_selection
	param_grid = {	'max_depth': [4,6], 
					'min_samples_split': [2,3],
	          		'learning_rate': [0.2,0.1,0.01], 
	          	 }
	gbm=ensemble.GradientBoostingRegressor(n_estimators=30)
	gs_cv = model_selection.GridSearchCV(gbm,param_grid).fit(X_train,y_train)
	print(gs_cv.best_params_)
	print(gs_cv.best_score_)
	print(gs_cv.cv_results_)

def gbmmultiomial(X_train, X_test, y_train, y_test):
	# GBM multinomial classification
	from sklearn import ensemble
	from sklearn import metrics
	params = {'n_estimators': 270, 'max_depth': 6, 'min_samples_split': 3,
	          'learning_rate': 0.2, 'loss': 'deviance'}
	gbm=ensemble.GradientBoostingClassifier()
	print('about to start fit')
	gbm.fit(X_train,y_train)
	print('about to start prediction')
	prediction_gbm_train = gbm.predict_proba(X_train)
	prediction_gbm_test = gbm.predict_proba(X_test)
	print('confidence',gbm.score(X_test, y_test))
	print('accuracy_score',metrics.accuracy_score(y_test, prediction_gbm_test, normalize=False))
	print(metrics.classification_report(y_test, prediction_gbm_test))
	# #############################################################################
	return prediction_gbm_train, prediction_gbm_test

def svm(X_train, X_test, y_train, y_test):
	# Support Vector Machine
	from sklearn.svm import SVR
	highestconfidence=0
	# for k in ['linear','poly','rbf','sigmoid']:	
	for k in ['linear','rbf','sigmoid']:
		clf = SVR(kernel=k)
		clf.fit(X_train, y_train) 
		confidence = clf.score(X_test, y_test)
		print(k,confidence)
		if confidence>highestconfidence:
			prediction_svm_train = clf.predict(X_train)
			prediction_svm_test = clf.predict(X_test)		
			highestconfidence=confidence
			highestk=k
	visualizerandomredisuals(heading="SVM: "+highestk ,prediction=prediction_svm_test,y_test=y_test)
	visualizeorderedredisuals(heading="SVM: "+highestk ,prediction=prediction_svm_test,y_test=y_test)	
	return 	prediction_svm_train,prediction_svm_test


# Deep neural network with Tensorflow
def nn(data_train, data_test,CONTFEATURES,DUMMYFEATURES):

	def input_fn(data_set):
		feature_cols = { k: tf.constant(data_set[k].values, shape=[data_set[k].size,1]) for k in (CONTFEATURES+DUMMYFEATURES) }
		labels = tf.constant(data_set[LABEL].values)
		return feature_cols, labels

	def batched_input_fn(data_set):
		def _input_fn(data_set):
			feature_cols = { k: tf.constant(data_set[k].values) for k in (CONTFEATURES+DUMMYFEATURES) }
			labels = tf.constant(data_set[LABEL].values)
			feature_cols, labels= tf.train.slice_input_producer([feature_cols, labels])
			return tf.train.batch([feature_cols, labels], batch_size=80)		
		return _input_fn

	import os
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	import tensorflow as tf
	from tensorflow.contrib import learn
	from sklearn import datasets, metrics, preprocessing
	# tf.logging.set_verbosity(tf.logging.INFO)
	tf.logging.set_verbosity(tf.logging.ERROR)#
	# np.random.seed(0)
	# msk = np.random.rand(len(data)) < 0.7
	# data_train=data[msk]
	# data_test=data[~msk]
	feature_cols = [tf.contrib.layers.real_valued_column(k) for k in (CONTFEATURES+DUMMYFEATURES)]
	regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, hidden_units=[128,128],dropout=0.05)
	regressor.fit(input_fn=lambda: input_fn(data_train), steps=50000)
	ev=regressor.evaluate(input_fn=lambda: input_fn(data_test), steps=1)
	loss_score = ev["loss"]
	print("Loss: {0:f}".format(loss_score))
	prediction_nn_train = regressor.predict_scores(input_fn=lambda: input_fn(data_train))
	prediction_nn_test = regressor.predict_scores(input_fn=lambda: input_fn(data_test))
	prediction_nn_test=list(prediction_nn_test)
	actuals=list(data_test[LABEL])
	print(metrics.mean_squared_error(prediction_nn_test,actuals), "deep learning mean squared error")
	print(metrics.r2_score(prediction_nn_test, actuals),"deep learning r2")
	# confidence = regressor.score((tf.constant(data_test[k].values) for k in (CONTFEATURES+DUMMYFEATURES)), data_test[LABEL].values)
	# print('confidence',confidence)
	visualizerandomredisuals(heading="Deep neural network",prediction=prediction_nn_test,y_test=actuals)
	visualizeorderedredisuals(heading="Deep neural network",prediction=prediction_nn_test,y_test=actuals)
	return prediction_nn_train,prediction_nn_test

def converttofloatarray(data):
  return np.array(data, dtype=np.float32)

def tflearn(X_train, X_test, y_train, y_test):
	# Deep neural network with tfLearn
	import tflearn
	from sklearn import metrics
	import os
	os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
	X_train = converttofloatarray(X_train)
	y_train = converttofloatarray(np.reshape(y_train, (-1, 1)))
	X_test = converttofloatarray(X_test)
	y_test = converttofloatarray(np.reshape(y_test, (-1, 1)))
	net = tflearn.input_data(shape=[None,len(X_train[0])])
	net = tflearn.fully_connected(net,128)
	net = tflearn.fully_connected(net,128)
	net = tflearn.fully_connected(net,len(y_train[0]))
	net = tflearn.regression(net, optimizer='sgd', loss='mean_square', metric='R2', learning_rate=0.005)
	model = tflearn.DNN(net)
	model.fit(X_train, y_train, n_epoch=50000, batch_size=160, show_metric=False)
	prediction_tflearn_train = model.predict(X_train)
	prediction_tflearn_test = model.predict(X_test)
	# print("y_test: ",y_test)
	# print("prediction_tflearn: ",prediction_tflearn)
	print(metrics.mean_squared_error(prediction_tflearn_test,y_test), "tflearn deep learning mean squared error")
	print(metrics.r2_score(prediction_tflearn_test, y_test),"tflearn deep learning r2")
	# confidence = model.score(X_test, y_test)
	# print('confidence',confidence)
	visualizerandomredisuals(heading="tflearn Deep neural network",prediction=prediction_tflearn_test,y_test=y_test)
	visualizeorderedredisuals(heading="tflearn Deep neural network",prediction=prediction_tflearn_test,y_test=y_test)
	return prediction_tflearn_train,prediction_tflearn_test

def returnNotMatches(a, b):
    return [[x for x in a if x not in b], [x for x in b if x not in a]]	

