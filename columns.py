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
  return  dtype


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

def returnSEGMENTCOLORS():
  SEGMENTCOLORS = {
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
  return SEGMENTCOLORS

def returnPRODUCTCOLORS():
  PRODUCTCOLORS = {
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
    }
  return PRODUCTCOLORS

def returnCITYCOLORS():
  CITYCOLORS = {
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
    }
  return CITYCOLORS
