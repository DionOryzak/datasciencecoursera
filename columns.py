
def returnBandFeatures():
	BANDFEATURES={
		'EARNEDQUARTER_cont':(2010.5,2010.75)
	}   
	return  BANDFEATURES


def returndtype():
	dtype={
		"Segment": object, 
		"City": object, 
		"Product": object
	}
	return 	dtype

def returnCOLUMNS():
	COLUMNS=['Month_Ending',
	'Segment',
	'City',
	'Product',
	'Miles',
	'Reported_Accidents'
]
	return COLUMNS

def returnCATFEATURES():
	CATFEATURES=[
		'Segment',
		'City',
		'Product'
	]
	return CATFEATURES

def returnCONTFEATURES():
	CONTFEATURES=[
		'Month_Ending'
	]
	return CONTFEATURES


