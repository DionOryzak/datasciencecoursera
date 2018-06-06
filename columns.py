
def returnBandFeatures():
	BANDFEATURES={
		'NumberPhysicians':(0,1,2,5,10,20,50,100,200,500),
		'Revenue':(0,1,100000,20000,500000,1000000,2000000,5000000,10000000,20000000,50000000,100000000),
		'CyberLimit':(0,10000,25000,50000,100000,150000,250000,500000,750000,1000000,2000000,2500000,3000000,4000000,5000000,6000000,7500000,8000000,10000000,15000000,20000000),
		'MedefenseLimit':(0,10000,25000,50000,100000,250000,500000,1000000,2000000),
		'BothCoverages':(1)
	}   
	return  BANDFEATURES


def returndtype():
	dtype={
		'ProgramType':object
	}
	return 	dtype

def returnCOLUMNS():
	COLUMNS=[
	'ProgramType',
	'NumberPhysicians',
	'Revenue',
	'MedefenseLimit',
	'CyberLimit',
	'MedefenseLimit_Cat',
	'CyberLimit_Cat',
	'BothCoverages',
	'CYBExposure',
	'CYBClaimCount',
	'CYBClaimCost',
	'MEDExposure',
	'MEDClaimCount',
	'MEDClaimCost'
	]
	return COLUMNS

def returnCATFEATURES():
	CATFEATURES=[
	]
	return CATFEATURES

def returnCONTFEATURES():
	CONTFEATURES=[
	'MedefenseLimit',
	'CyberLimit',
	'NumberPhysicians',
	'Revenue',
	'BothCoverages'
	]
	return CONTFEATURES

