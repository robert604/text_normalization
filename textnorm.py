

import pandas as pd
import numpy as np
from myutils import mapSeqInd,mapSeqsInd,mapSeqInd1,mapSeqsInd1,padLists,prependLists
from myutils import appendToIndex,makeStepSliceInds,makeIndsFromSliceInds,makeFoldSliceInds
from myutils import pmap,pmapSeries,pstarmap,makeIndex,findIndexes,padWithLast
from myutils import makeEmbeddingsDict_w2v,makeEmbeddingsMatrix
from mypytorchutils import makePretrainedEmbeddings
import re
from collections import Counter,OrderedDict
import itertools
import math

import torch
from torch import FloatTensor as FT,LongTensor as LT,IntTensor as IT,ByteTensor as BT
from torch.autograd import Variable as V
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import pickle
import inspect

from tn_funcs import addSubTokData,removeLongSentences
from tn_funcs import makeSentdf,makeSentLimOutInds,saveModelState,makeSavedModel
from tn_funcs import asYear,makeReplaceDict,makeReplaceSameDict,makeReplaceSingleDict,makeReplaceMultiDict
from tn_funcs import getChangeTransformed2,getChangeTransformed,transformRemainingUnknown
from tn_funcs import replaceSame,replaceSingle,replaceUni,replaceMulti








torch.manual_seed(1234)
np.random.seed(1234)

MAX_SENTENCE_LENGTH = 135

BATCH_SIZE = 64
ENC_HIDDEN_SIZE = 99 + 100
DEC_HIDDEN_SIZE = 101 + 100
ENC_TOK_EMBED_SIZE = 97 + 40
DEC_EMBED_SIZE = 103 + 40
ENC_SUBTOKTYPE_EMBED_SIZE = 5 + 5
ENC_SUBTOKLEN_EMBED_SIZE = 5 + 5
SUBTOK_SEQ_LEN = 10
NUM_LAYERS = 1

eosStr = '=eos='
defaultStr = '=default='

useGPU = True 

def gpu(obj):
  if useGPU:
    return obj.cuda()
  else:
    return obj




  

   
class TwoDicts:
  def __init__(self,firstDict,second=True):
    self.firstDict = firstDict
    self.secondDict = self.makeLowerCaseDict(firstDict) if second else None

  def makeLowerCaseDict(self,thedict):
    lc = {k.lower():v.lower() for k,v in thedict.items()}
    return lc

  def has(self,item):
    if item in self.firstDict:
      return True
    elif self.secondDict is not None and item in self.secondDict:
      return True
    else:
      return False
 
  def lookup(self,item):
    if item in self.firstDict:
      return self.firstDict[item]
    elif self.secondDict is not None and item in self.secondDict:
      return self.secondDict[item]
    else:
      raise Exception('Item not in TwoDicts',item)


class Encoder(nn.Module):
  def __init__(self,tokVocabSize,tokEmbedSize,tokSeqLen
               ,subTokTypeVocabSize,subTokTypeEmbedSize,subTokLengthVocabSize,subTokLengthEmbedSize,subTokSeqLen
               ,hiddenSize):
    super(Encoder,self).__init__()
    frame = inspect.currentframe()
    args,_,_,argvals = inspect.getargvalues(frame)
    self.initVals = [argvals[a] for a in args[1:]]
    
    self.hiddenSize = hiddenSize
    self.numLayers = NUM_LAYERS
    bidir = True
    self.numDir = 2 if bidir else 1
    self.outputSize = self.numDir*self.hiddenSize
    self.wordEmbeddings = preTrainedWordEmbs    
    wordEmbedSize = self.wordEmbeddings.embedding_dim   
    self.lstmInSize = tokEmbedSize + wordEmbedSize + ((subTokTypeEmbedSize+subTokLengthEmbedSize)*subTokSeqLen)
    self.tokEmbeddings = nn.Embedding(tokVocabSize,tokEmbedSize)

    self.subTokTypeEmbedsList = nn.ModuleList([nn.Embedding(subTokTypeVocabSize,subTokTypeEmbedSize) for i in range(subTokSeqLen)])    
    self.subTokLengthEmbedsList = nn.ModuleList([nn.Embedding(subTokLengthVocabSize,subTokLengthEmbedSize) for i in range(subTokSeqLen)])    
    self.lstm = nn.LSTM(self.lstmInSize,hiddenSize,num_layers=self.numLayers,batch_first=True,bidirectional=bidir)
    self.hiddenSizeZerosTensor = gpu(torch.zeros(self.numLayers*self.numDir,BATCH_SIZE,self.hiddenSize))
    
  def forward(self,tokInds,wordEmbInds,subTokTypes,subTokLengths):
    x1 = self.tokEmbeddings(tokInds)
    x2 = [self.subTokTypeEmbedsList[i](subTokTypes[:,:,i]) for i in range(subTokTypes.size()[-1])]
    x3 = [self.subTokLengthEmbedsList[i](subTokLengths[:,:,i]) for i in range(subTokLengths.size()[-1])]
    x4 = self.wordEmbeddings(wordEmbInds)
    x = x2 + x3    
    x.append(x1)
    x.append(x4)
    x = torch.cat(x,dim=-1)    
    x,self.hidden = self.lstm(x,self.hidden)
    return x    
      
  def initHidden(self):
    self.hidden = (V(self.hiddenSizeZerosTensor),V(self.hiddenSizeZerosTensor))

  def getState(self):
    s = {
        'initVals':self.initVals,
        'state_dict':self.state_dict
        }
    return s



class Decoder(nn.Module):
  def __init__(self,tokVocabSize,tokEmbedSize,tokSeqLen
               ,subTokTypeVocabSize,subTokTypeEmbedSize,subTokLengthVocabSize,subTokLengthEmbedSize,subTokSeqLen               
               ,hiddenSize,outVocabSize,encodedSize):
    super(Decoder,self).__init__() 
    frame = inspect.currentframe()
    args,_,_,argvals = inspect.getargvalues(frame)
    self.initVals = [argvals[a] for a in args[1:]]    
    
    self.tokSeqLen = tokSeqLen
    self.hiddenSize = hiddenSize
    self.numLayers = NUM_LAYERS
    bidir = True
    self.numDir = 2 if bidir else 1
    self.workingHiddenSize = self.hiddenSize*self.numLayers*self.numDir
    self.wordEmbeddings = preTrainedWordEmbs    
    wordEmbedSize = self.wordEmbeddings.embedding_dim     
    self.tokEmbeddings = nn.Embedding(tokVocabSize,tokEmbedSize)
    self.lstm = nn.LSTM(tokEmbedSize+wordEmbedSize+encodedSize+(subTokSeqLen*(subTokTypeEmbedSize+subTokLengthEmbedSize)),self.hiddenSize,num_layers=self.numLayers,batch_first=True,bidirectional=bidir)

    self.output = nn.Linear(self.hiddenSize*self.numDir,outVocabSize)
    self.hiddenSizeZerosTensor = gpu(torch.zeros(self.numLayers*self.numDir,BATCH_SIZE,self.hiddenSize))

    self.encOutLinear = nn.Linear(encodedSize,self.workingHiddenSize)
    self.hiddenLinear = nn.Linear(self.workingHiddenSize,self.workingHiddenSize)
    #self.V = nn.Parameter(torch.randn(self.workingHiddenSize))

    self.subTokTypeEmbedsList = nn.ModuleList([nn.Embedding(subTokTypeVocabSize,subTokTypeEmbedSize) for i in range(subTokSeqLen)])    
    self.subTokLengthEmbedsList = nn.ModuleList([nn.Embedding(subTokLengthVocabSize,subTokLengthEmbedSize) for i in range(subTokSeqLen)])    
   


  def forward(self,tokInd,wordInd,encoderOutputs,subTokTypes,subTokLengths):
    encOutL = self.encOutLinear(encoderOutputs)
    h = self.hidden[-1]
    hiddenL = torch.cat([h[i] for i in range(h.size()[0])],dim=-1)
    hiddenL = hiddenL.unsqueeze(1)
    hiddenL = self.hiddenLinear(hiddenL)
    enc_hid = F.tanh(encOutL + hiddenL)
    
    
    #enc_hid_v = (enc_hid * self.V).sum(-1) 
    enc_hid_v = enc_hid.sum(-1)  
    
    attenWeights = F.softmax(enc_hid_v)
    tokEmbeds = self.tokEmbeddings(tokInd)
    wordEmbeds = self.wordEmbeddings(wordInd)
    
    x2 = [self.subTokTypeEmbedsList[i](subTokTypes[:,i]) for i in range(subTokTypes.size()[-1])]
    x3 = [self.subTokLengthEmbedsList[i](subTokLengths[:,i]) for i in range(subTokLengths.size()[-1])]
    x = x2 + x3 + [tokEmbeds,wordEmbeds]   
    x = torch.cat(x,dim=-1)
    selectedOutput = torch.bmm(attenWeights.unsqueeze(1),encoderOutputs)
    x = torch.cat((x.unsqueeze(1),selectedOutput),dim=-1)     
     
       
    x,self.hidden = self.lstm(x,self.hidden)
    x = x.squeeze(1)
    x = self.output(x)        
    return x
      
  def initHidden(self):
    self.hidden = (V(self.hiddenSizeZerosTensor),V(self.hiddenSizeZerosTensor))

  def getState(self):
    s = {
        'initVals':self.initVals,
        'state_dict':self.state_dict
        }
    return s


    


  


def makeDicts3(nameInfo):
  symboldict = {}
  pluraldict = {}
  for i in range(0,len(nameInfo),3):
    name,pluralName,symbols = nameInfo[i:i+3]
    if type(symbols)!=list: raise Exception('makeDicts3: Not a symbol list',symbols)
    for symbol in symbols:
      if symbol in symboldict:
        raise Exception('makeDicts3: Symbol already in dict:',symbol)
      symboldict[symbol] = name
      if symbol in pluraldict:
        raise Exception('makeDicts3: symbol already in plural dict:',symbol)
      pluraldict[symbol] = pluralName if pluralName is not None else name
  return symboldict,pluraldict

def makeDicts2(nameInfo):
  symboldict = {}
  for i in range(0,len(nameInfo),2):
    name,symbols = nameInfo[i:i+2]
    if type(symbols)!=list: raise Exception('makeDicts2: Not a symbol list',symbols)
    for symbol in symbols:
      if symbol in symboldict:
        raise Exception('makeDicts2: Symbol already in dict:',symbol)
      symboldict[symbol] = name
  return symboldict



singleDigits = ["o", "one", "two", "three", "four",  "five", 
    "six", "seven", "eight", "nine"]

u20 = ["zero", "one", "two", "three", "four",  "five", 
    "six", "seven", "eight", "nine","ten", "eleven", "twelve",
    "thirteen",  "fourteen","fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen"]
tens = ["-", "=", "twenty", "thirty", "forty",
    "fifty", "sixty", "seventy", "eighty", "ninety"]
thousands = ["thousand", "million",  "billion",  "trillion", 
    "quadrillion",  "quintillion",  "sextillion",  "septillion", "octillion", 
    "nonillion",  "decillion",  "undecillion",  "duodecillion",  "tredecillion", 
    "quattuordecillion","quindecillion",  "sexdecillion",  "septendecillion",  "octodecillion", 
    "novemdecillion",  "vigintillion"]

ordinalDict = {"zero":"zeroth","one":"first", "two":"second", "three":"third", "four":"fourth",  "five":"fifth", 
    "six":"sixth", "seven":"seventh", "eight":"eighth", "nine":"ninth","ten":"tenth", "eleven":"eleventh", "twelve":"twelfth",
    "thirteen":"thirteenth",  "fourteen":"fourteenth","fifteen":"fifteenth", "sixteen":"sixteenth",
    "seventeen":"seventeenth", "eighteen":"eighteenth", "nineteen":"nineteenth","twenty":"twentieth", "thirty":"thirtieth", "forty":"fortieth",
    "fifty":"fiftieth", "sixty":"sixtieth", "seventy":"seventieth", "eighty":"eightieth", "ninety":"ninetieth","hundred":"hundredth","thousand":"thousandth",
    "million":"millionth","billion":"billionth","trillion":"trillionth","quadrillion":"quadrillionth",

    "quintillion":"quintillionth","sextillion":"sextillionth","septillion":"septillionth",
    "octillion":"octillionth", 
    "nonillion":"nonillionth","decillion":"decillionth","undecillion":"undecillionth",
    "duodecillion":"duodecillionth","tredecillion":"tredecillionth", 
    "quattuordecillion":"quattuordecillionth","quindecillion":"quindecillionth",
    "sexdecillion":"sexdecillionth","septendecillion":"septendecillionth",
    "octodecillion":"octodecillionth", 
    "novemdecillion":"novemdecillionth","vigintillion":"vigintillionth"
    }


monthInfo = ['january',['january','jan']
             ,'february',['february','feb']
             ,'march',['march','mar']
             ,'april',['april','apr']
             ,'may',['may']
             ,'june',['june','jun']
             ,'july',['july','jul']
             ,'august',['august','aug']
             ,'september',['september','sep','sept']
             ,'october',['october','oct']
             ,'november',['november','nov']
             ,'december',['december','dec']    
    ]
monthDict = makeDicts2(monthInfo)

intMonthDict = {1:'january',2:'february',3:'march',
             4:'april',5:'may',6:'june',7:'july',
             8:'august',9:'september',10:'october',
             11:'november',12:'december'
    }

weekdayDict = {'monday':'monday','tuesday':'tuesday','wednesday':'wednesday',
               'thursday':'thursday','friday':'friday','saturday':'saturday',
               'sunday':'sunday','mon':'monday','tue':'tuesday','wed':'wednesday',
               'thu':'thursday','fri':'friday','sat':'saturday','sun':'sunday'
    }

sDict = {'one':'ones','two':'twos','three':'threes','four':'fours','five':'fives',
         'six':'sixes','seven':'sevens','eight':'eights','nine':'nines',
    'ten':'tens','eleven':'elevens','twelve':'twelves','thirteen':'thirteens',
    'fourteen':'fourteens','fifteen':'fifteens','sixteen':'sixteens',
    'seventeen':'seventeens','eighteen':'eighteens','nineteen':'nineteens',
    'twenty':'twenties','thirty':'thirties','forty':'forties',
         'fifty':'fifties','sixty':'sixties','seventy':'seventies',
         'eighty':'eighties','ninety':'nineties','hundred':'hundreds',
         'thousand':'thousands'
    }



def under1000(num):
  string = []
  h = num//100
  if h>=1:
    string.append(u20[h])
    string.append('hundred')
    num %=100
  t = num//10
  if t>=2:
    string.append(tens[t])
    u = num%10
    if u>=1:
      string.append(u20[u])
  elif num>=1:
    string.append(u20[num])
  if len(string)==0: string.append('zero')
  return string



def ntow(numstr,ordinal=False):
  if numstr[0]=='-':
    negative = True
    numstr = numstr[1:]
  else:
    negative = False
  ts = math.ceil(len(numstr)/3)
  x = numstr.zfill((ts)*3)
  string = []  
  for t in range(ts-1,-1,-1):
    numint = int(x[:3])
    if numint>0:
      string += under1000(numint)
      if t>=1: string.append(thousands[t-1])
    x = x[3:]
  if len(string)==0: string.append('zero')
  if negative: string.insert(0,'minus')
  if ordinal: string[-1] = ordinalDict[string[-1]]    
  string = ' '.join(string)
  return string

oPattern = re.compile(r'[\d,]*(?:1st|2nd|3rd|4th|5th|6th|7th|8th|9th|0th|11th|12th|13th|14th|15th|16th|17th|18th|19th)$')
onumPattern = re.compile(r'(\d+)[^\d]+$')
def otow(ordinalstring):
  lc = ordinalstring.lower()
  m = oPattern.match(lc)
  if m is None:
    out = ordinalstring
  else:
    lc = commaPat.sub('',lc)
    m = onumPattern.match(lc)
    out = ntow(m.group(1),ordinal=True)  
  return out

digitsWithDashPat = re.compile(r'(\d+)-?$')
def digits(digitString):
  m = digitsWithDashPat.match(digitString)
  if m is None:
    out = digitString
  else:
    dstr = m.group(1)
    out = []
    for i in range(len(dstr)):
      out.append(singleDigits[int(dstr[i])])
    out = ' '.join(out)
  return out

telephonePat = re.compile(r'([\d(][\dA-Z ()-]*)$')
nonAlphanumericPat = re.compile(r'[\W_]')
dashPat = re.compile(r'-')
def telephone(thestring,spell):
  m = telephonePat.match(thestring)
  if m is None:
    out = thestring
  else:
    thestr = m.group(1)
    words = [w for w in nonAlphanumericPat.split(thestr) if w!='']    
    if len(words)<=1:
      out = thestring
    else:    
      outwords = []
      for wi in range(len(words)):
        w = words[wi]
        if w.isdigit():
          if len(w)==5 and w[0] in '01' and w[1]!='0' and (int(w[1:])%1000)==0:
            outwords.append(singleDigits[int(w[0])])
            outwords.append(ntow(w[1:]))
          elif len(w)==4 and w[0] in '01' and w[1]!='0' and (int(w[1:])%100)==0:
            outwords.append(singleDigits[int(w[0])])
            outwords.append(ntow(w[1:]))
          elif len(w)==4 and w[0]!='0' and (int(w)%1000)==0:
            outwords.append(ntow(w))
          elif len(w)==3 and w[0]!='0' and (int(w)%100)==0:
            outwords.append(ntow(w))
          else:
            for d in w: outwords.append(singleDigits[int(d)])
        elif w.isalpha():
          if spell:
            outwords += [c.lower() for c in w]
          else:
            outwords.append(w.lower())
        else:
          for i in range(len(w)):
            c = w[i]
            if c.isalpha(): outwords.append(c.lower())
            else: outwords.append(singleDigits[int(c)])
        if wi+1<len(words): outwords.append('sil')
      out = ' '.join(outwords)
  return out

def telephone_spell(thestring):
  return telephone(thestring,True)

def telephone_dontSpell(thestring):
  return telephone(thestring,False)          

time1Pat = re.compile(r'(\d{1,2}) ?([apm.]+)(?: ?([a-z]+))?$')
def time1(thestring): 
  lc = thestring.lower()
  m = time1Pat.match(lc)
  if m is None:
    out = thestring
  else:
    hours,ampm,more = m.groups()
    ampm = dotPat.sub('',ampm)
    if ampm!='am' and ampm!='pm':
      out = thestring
    else:  
      if 0<=int(hours)<=12:
        out = ntow(hours)
        out += ' ' + (' '.join([c for c in ampm]))
#      elif 13<=int(hours)<=24 and not has_ampm :
#        out = ntow(hours)
#        out += ' ' + (' '.join([c for c in ampm if c.isalpha()]))
        if more is not None: out += ' ' + (' '.join(list(more)))
      else:
        out = thestring

  return out 

time2Pat = re.compile(r'(\d{1,2})([:.]) ?(\d{2})(?: ?([a-zA-Z][a-zA-Z .]*))?$')
def time2(thestring):
  m = time2Pat.match(thestring)
  if m is None:
    out = thestring
  else:
    hours,sep,minutes,ampmZones = m.groups()
    if ampmZones is not None:
      ampmZones = dotPat.sub('',ampmZones).lower()
      ampmZonesSplit = ampmZones.split()
      has_ampm = 'am' in ampmZonesSplit or 'pm' in ampmZonesSplit
    else:
      has_ampm = False

    ihours,iminutes = int(hours),int(minutes)
    
    if sep=='.' and not has_ampm:
      out = thestring    
    elif 1<=ihours<=12:
      out = ntow(hours)

      if int(minutes)==0:
        if not has_ampm: out += " o'clock"
      else:
        if int(minutes)<10: out += ' ' + digits(minutes)
        else: out  += ' ' + ntow(minutes)
      if ampmZones is not None:
        out += ' ' + (' '.join([c for c in ampmZones if c.isalpha()]))
    elif (ihours==0 or 13<=ihours<=24) and not has_ampm:
      out = ntow(hours)
      if int(minutes)!=0:
        if int(minutes)<10: out += ' ' + digits(minutes)
        else: out  += ' ' + ntow(minutes)
      else:
        out += ' hundred'
      if ampmZones is not None:
        out += ' ' + (' '.join([c for c in ampmZones if c.isalpha()]))        
    elif (ihours==0 or 13<=ihours<=24) and has_ampm: 
      out = ntow(str(ihours-12))
      if int(minutes)!=0:
        if int(minutes)<10: out += ' ' + digits(minutes)
        else: out  += ' ' + ntow(minutes)
      else:
        pass
      if ampmZones is not None:
        out += ' ' + (' '.join([c for c in ampmZones if c.isalpha()]))        
    else:
      out = thestring

  return out     

time3Pat = re.compile(r'(\d+):(\d{2})([:.])(\d+)(?: ?([a-zA-Z][a-zA-Z .]*))?$')
def time3(thestring):
  m = time3Pat.match(thestring)
  if m is None:
    out = thestring
  else:
    d1,d2,sep,d3,alpha = m.groups()
    if sep==':':
      hours,minutes,seconds = d1,d2,d3
      out = ntow(hours) + (' hour' if int(hours)==1 else ' hours')
      out += ' ' + ntow(minutes) + (' minute' if int(minutes)==1 else ' minutes')      
      out += ' and ' + ntow(seconds) + (' second' if int(seconds)==1 else ' seconds')       
    else:
      minutes,seconds,milliseconds = d1,d2,d3
      out = ntow(minutes) + (' minute' if int(minutes)==1 else ' minutes')     
      out += ' ' + ntow(seconds) + (' second' if int(seconds)==1 else ' seconds')       
      out += ' and ' + ntow(milliseconds) + (' millisecond' if int(milliseconds)==1 else ' milliseconds')  
    if alpha is not None:
      out += ' ' + (' '.join([c for c in alpha.lower() if c.isalpha()]))
  return out

time4Pat = re.compile(r'(\d{2})\.(\d{2})$')
def time4(thestring):
  m = time4Pat.match(thestring)
  if m is None:
    out = thestring
  else:
    hours,minutes = m.groups()
    ihours,iminutes = int(hours),int(minutes)
    if not 0<=ihours<=9 or not 0<=iminutes<=59:
      out = thestring
    else:
      out = ntow(hours)
      if iminutes<10: out += ' ' + digits(minutes)
      else: out  += ' ' + ntow(minutes)
  return out

yearDigitsPat = re.compile(r'\d+$')
def ytow(yearstring):
  m = yearDigitsPat.match(yearstring)
  if m is None:
    out = yearstring
  elif int(yearstring)>=10000: #44
    out = ntow(yearstring)
  else:
    hundredsStr = yearstring[:-2]
    remainingStr = yearstring[-2:]
    if hundredsStr=='':
      if remainingStr[0]=='0':
        out = digits(remainingStr)
      else:
        out = ntow(remainingStr)
    else:
      hundredsVal,remainingVal,yearVal =int(hundredsStr),int(remainingStr),int(yearstring)
      if hundredsVal==0 and remainingVal==0:
        out = 'zero'
      elif yearVal%1000==0: out = ntow(yearstring) #2000
      elif yearVal<1000:
        if remainingStr[0]=='0':
          if yearVal<10:
            out = 'o' + ' ' + ntow(yearstring)
          else:
            out = ntow(yearstring) #700 704
        else:
          out = ntow(hundredsStr) + ' ' + ntow(remainingStr) #313
      else: #elif yearVal<2000:
        if hundredsVal%10==0:
          if remainingVal<10: out = ntow(yearstring) #1008
          else: out = ntow(hundredsStr) + ' ' + ntow(remainingStr) #1056
        elif remainingVal==0: out = ntow(hundredsStr) + ' hundred' # 1900
        elif remainingVal<10: out = ntow(hundredsStr) + ' ' + digits(remainingStr)  #1505      
        else: out = ntow(hundredsStr) + ' ' + ntow(remainingStr)
               
  return out


  

def dtow(digitstring):
  v = int(digitstring)
  if v==0: out = 'o'
  else: out = u20[v]
  return out

# none incorrect
wdayMonthDateYearPat = re.compile('(?:(' + ('|'.join(weekdayDict.keys())) + r')(?:,| |, |\. ))?(' + ('|'.join(monthDict.keys())) + ')' + r'\.? (\d+)(?:st|nd|rd|th)?(?:,|, | )?(?:(\d+),?(?: ?(bc|ad)\.?)?)?$')
def wdayMonthDateYear(mdystring):
  lc = mdystring.lower()
  m = wdayMonthDateYearPat.match(lc)
  if m is None:
    out = mdystring
  else:
    wday,month,date,year,era = m.groups()    
    if date is not None and month is not None and year is None and int(date)>31:
      out = mdystring
    else:      
      text = []
      if wday is not None:
        wday = weekdayDict[wday]
        text.append(wday)
      if month is not None:
        month = monthDict[month]
        text.append(month)      
      if date is not None:
        date = ntow(date,ordinal=True)
        text.append(date)
      if year is not None:
        year = ytow(year)
        text.append(year)
      if era is not None:
        era = list(era)
        text.extend(era)
  
      if len(text)==0:
        out = mdystring
      else:
        out = ' '.join(text)
      
  return out

# some incorrect
wdayDateMonthYearPat = re.compile('(?:(' + ('|'.join(weekdayDict.keys())) + r')(?:,| |, |. ))?' +  r'(?:(?:the )?(\d+)(?:st|nd|rd|th)?(?: |, |,|-))?('+ '|'.join(monthDict.keys()) +r')(?:,|, | |\.|\. |-|\., )?(\d+)?(?: ?(bc|ad|ce)\.?)?,?$')
def wdayDateMonthYear(dmystring): 
  lc = dmystring.lower()
  m = wdayDateMonthYearPat.match(lc)
  if m is None:
    out = dmystring
  else:
    wday,date,month,year,era = m.groups()
    if month is None:
      out = dmystring
    elif date is None and month is not None and year is not None and int(year)<=31:
      out = dmystring
    elif wday is None and date is None and month is not None and year is None:
      out = dmystring
    elif date is not None and (len(date)>2 or int(date)<1 or int(date)>31):
      out = dmystring
    else:
      text = []
      if wday is not None:
        wday = weekdayDict[wday]
        text.append(wday)
      if date is not None:
        date = ntow(date,ordinal=True)
        text.append('the')
        text.append(date)
      if month is not None:
        month = monthDict[month]
        if len(text)>0: text.append('of')
        text.append(month)
      if year is not None:
        year = ytow(year)
        text.append(year)
      if era is not None:
        era = list(era)
        text.extend(era)
  
      if len(text)==0:
        out = dmystring
      else:
        out = ' '.join(text)
  return out

yearMonthDatePat = re.compile(r'(\d{4})(?: |, |,|-)('+ '|'.join(monthDict.keys()) +r')(?:,|, | |\.|\. |-)(\d+)(?: ?(bc|ad|ce)\.?)?$')
def yearMonthDate(dmystring): 
  lc = dmystring.lower()
  m = yearMonthDatePat.match(lc)
  if m is None:
    out = dmystring
  else:
    year,month,date,era = m.groups()
    if month is None:
      out = dmystring
    elif date is None and month is not None and year is not None and int(year)<=31:
      out = dmystring
    elif date is not None and (len(date)>2 or int(date)<1 or int(date)>31):
      out = dmystring
    else:
      text = []
      if date is not None:
        date = ntow(date,ordinal=True)
        text.append('the')
        text.append(date)
      if month is not None:
        month = monthDict[month]
        if len(text)>0: text.append('of')
        text.append(month)
      if year is not None:
        year = ytow(year)
        text.append(year)
      if era is not None:
        era = list(era)
        text.extend(era)
  
      if len(text)==0:
        out = dmystring
      else:
        out = ' '.join(text)
  return out


yearEraPat = re.compile(r'(\d+) ?(?:(a[.]?d|b[.]?c|bce|c[.]?e)[.,]?|(b))$')
dotPat = re.compile(r'\.')
def yearEra(string): 
  lc = string.lower()
  m = yearEraPat.match(lc)
  if m is None:
    out = string
  else:
    year,era,other = m.groups()
    if len(year)==1: year = '0' + year
    out = ytow(year)
    if era is not None:
      era = dotPat.sub('',era)
      out += ' ' + (' '.join(list(era)))
  return out

yearOnlyPat = re.compile(r'\d{1,4}$')
def yearOnly(thestring):
  m = yearOnlyPat.match(thestring)
  if m is None:
    out = thestring
  else:
    val = int(thestring)
    if val<1010 or val>2099:
      out = thestring
    else:
      out = ytow(thestring)
  return out


yearPunctPat = re.compile(r'(\d{1,4})[,/]$')
def yearPunct(string):
  m = yearPunctPat.match(string)
  if m is None:
    out = string
  else:
    year = m.group(1)
    out = ytow(year)
  return out

 
numberPunctPat = re.compile(r'(\d+)[,/]$')
def numberPunct(string):
  m = numberPunctPat.match(string)
  if m is None:
    out = string
  else:
    number = m.group(1)
    if len(number)<4:
      out = ntow(number)
    else:
      out = string
  return out

numberAndLettersPat = re.compile(r'^([1-9][,\d]*) (?:U\.S\.|US|[AM])$')
def numberAndLetters(thestring): 
  m = numberAndLettersPat.search(thestring)
  if m is None:
    out = thestring
  else:
    number = m.group(1)
    number = commaPat.sub('',number)
    if len(number)>=10 or asYear(number):
      out = thestring
    else:
      out = ntow(number)
  return out  
  

dmPattern = re.compile(r'(\d+) ('+ '|'.join(monthDict.keys()) +r')$')
def dmtow(dmstring):
  lc = dmstring.lower()
  m = dmPattern.match(lc)
  if m is None:
    out = dmstring
  else:
    out = 'the ' + ntow(m.group(1),ordinal=True) + ' of ' + monthDict[m.group(2)]
  return out  

mdPattern = re.compile('(' + ('|'.join(monthDict.keys())) + ')' + r'\.? (\d+)$')
def mdtow(mdstring):
  lc = mdstring.lower()
  m = mdPattern.match(lc)
  if m is None:
    out = mdstring
  else:
    out = monthDict[m.group(1)] + ' ' + ntow(m.group(2),ordinal=True)
  return out
  

myPattern = re.compile('(' + ('|'.join(monthDict.keys())) + ')' + r'(?:,|, | )(\d+)$')
def mytow(mystring):
  lc = mystring.lower()
  m = myPattern.match(lc)
  if m is None:
    out = mystring
  else:
    out = monthDict[m.group(1)] + ' ' + ytow(m.group(2))
  return out  

indPermutations = [(1, 2, 3),(2, 1, 3),(3, 2, 1)]
dateDashPat = re.compile('(?:('+('|'.join(weekdayDict.keys())+r')[.,]? ')+')?' + r'(\d+)[-/.](\d+)[-/.](\d+)$')
def dateDash(string):
  lc = string.lower()
  mat = dateDashPat.match(lc)
  if mat is None:
    out = string
  else:
    groups = mat.groups()
    poss = []
    for p in indPermutations:
      dateInd,monthInd,yearInd = p
      dateStr,monthStr,yearStr = groups[dateInd],groups[monthInd],groups[yearInd]
      idate,imonth,iyear = int(dateStr),int(monthStr),int(yearStr)
      if idate<1 or idate>31: continue
      if imonth<1 or imonth>12: continue
      if len(dateStr)>2 or len(monthStr)>2 or len(yearStr)<2 or len(yearStr)>4: continue
      poss.append([dateInd,monthInd,yearInd])
    if len(poss)>1:

      if [1,2,3] in poss and [3,2,1] in poss:
        if int(groups[1])>12: poss.remove([1,2,3])
        else: poss.remove([3,2,1])
      if [1,2,3] in poss and [2,1,3] in poss: poss.remove([1,2,3])
      
    if len(poss)!=1:
      out = string
    else:        
      dateInd,monthInd,yearInd = poss[0]
      wdayStr,dateStr,monthStr,yearStr = groups[0],groups[dateInd],groups[monthInd],groups[yearInd]
      idate,imonth,iyear = int(dateStr),int(monthStr),int(yearStr)
      if monthInd==1:
        if idate<=12 and wdayStr is None:
          out = 'the ' + ntow(dateStr,ordinal=True) + ' of ' + intMonthDict[imonth] + ' ' + ytow(yearStr)
        else:
          if yearStr[0]=='0': out = intMonthDict[imonth] + ' ' + ntow(dateStr,ordinal=True) + ' ' + digits(yearStr)
          else: out = intMonthDict[imonth] + ' ' + ntow(dateStr,ordinal=True) + ' ' + ytow(yearStr)
      else:
        out = 'the ' + ntow(dateStr,ordinal=True) + ' of ' + intMonthDict[imonth] + ' ' + ytow(yearStr)
      if wdayStr is not None: out = weekdayDict[wdayStr] + ' ' + out      
  return out     




ysPat = re.compile(r'(\d{2,4})\'?s$')
def ys(string):
  m = ysPat.match(string)
  if m is None:
    out = string
  else:
    out = ytow(m.group(1))
    outsplitted = out.split()
    if outsplitted[-1] in sDict:
      outsplitted[-1] = sDict[outsplitted[-1]]
      out = ' '.join(outsplitted)
  return out
    

commaPat = re.compile(r',')
digitsPat = re.compile(r'\d+$')
def ntowc(numstring):
  ns = commaPat.sub('',numstring)
  m = digitsPat.match(ns)
  if m is None:
    out = numstring
  else:
    out = ntow(ns)
  return out


ndPat = re.compile(r'(-?\d+)\.(\d+)$')
def numDecComma(numdstring):
  nc = commaPat.sub('',numdstring) 
  m = ndPat.match(nc)
  if m is None:
    out = numdstring
  else:
    leftd = ntow(m.group(1))
    rightd = ' '.join([dtow(digit) for digit in m.group(2)])
    out = leftd + ' point ' + rightd
  return out


numDecPat1 = re.compile(r'(?:(-?\d+)(\.\d+)|(-?\d+)|(-?\.\d+))$')
def numDec1(numdstring):
  nc = numdstring
  m = numDecPat1.match(nc)
  if m is None:
    out,leftout,rightout = numdstring,None,None
  else:
    if m.group(1) is not None: left = m.group(1)
    elif m.group(3) is not None: left = m.group(3)
    else: left = None
    if m.group(2) is not None: right = m.group(2)
    elif m.group(4) is not None: right = m.group(4)    
    else: right = None        
    
    if left is None: leftout = ''
    else: leftout = ntow(left)
    
    minusWithoutLeft = False
    
    if right is None:
      rightout = ''
    else:
      if right[0]=='-':
        right = right[1:]
        minusWithoutLeft = True
      right = right[1:]
      if right=='0':
        if left is None: rightout = 'o'
        else: rightout = 'zero'
      else: rightout = ' '.join([dtow(digit) for digit in right])
      
    if leftout!='':
      out = leftout
      if rightout!='':
        out += ' point ' + rightout
    elif rightout!='':
      if minusWithoutLeft: out = 'minus point ' + rightout
      else: out = 'point ' + rightout

  return out,leftout,rightout



amountDict = {'m':'million','million':'million',
             'b':'billion','bn':'billion','billion':'billion',
             'trillion':'trillion','t':'trillion',
             'crore':'crore','cr':'crore','crores':'crore',
             'lacs':'lakh','lakh':'lakh','lakhs':'lakh',
             'k':'thousand','thousand':'thousand'
    }

currencyInfo = ['dollar','dollars',['$','us$','u$s','a$','au$','nz$','ca$','hk$','s$','nt$','dollars','dollar'],'cent','cents',
                 'euro','euros',['€','eur'],'cent','cents',
                'pound','pounds',['£','e£'],'pence','pence',
                'british pound','british pounds',['gbp'],'pence','pence',
                'yen','yen',['¥','yen'],'cent','cents',
                'danish krone','danish kroner',['dkk'],'ore','ore',
              'czech koruna','czech korunas',['czk'],'cent','cents',
               'rupee','rupees',['rs','rs.'],'paisa','paisas',
              'serbian dinar','serbian dinars',['rsd'],'cent','cents',
               'hungarian forint','hungarian forints',['huf'],'cent','cents',
              'norwegian krone','norwegian kroner',['nok'],'ore','ore',
              'united arab emirates dirham','united arab emirates dirhams',['aed'],'cent','cents',
               'belgian franc','belgian francs',['bef'],'cent','cents',
                'israeli new sheqel','israeli new sheqels',['ils'],'cent','cents',
              'real','reals',['r$'],'cent','cents',
               'german mark','german marks',['dm'],'pfennig','pfennigs',
                'seychelles rupee','seychelles rupees',['scr'],'cent','cents',
              'polish zloty','polish zlotys',['pln'],'cent','cents',
               'philippine peso','philippine pesos',['php'],'centavo','centavos',
               'indian rupee','indian rupees',['inr'],'paisa','paisas',
              'united states dollar','united states dollars',['usd'],'cent','cents',
               'south african rand','south african rands',['zar'],'cent','cents',
              'cypriot pound','cypriot pounds',['cyp'],'cent','cents',
               'indonesian rupiah','indonesian rupiahs',['idr'],'cent','cents',
              'botswana pula','botswana pulas',['bwp'],'cent','cents',
               'swiss franc','swiss francs',['chf'],'cent','cents',
              'bermudian dollar','bermudian dollars',['bmd'],'cent','cents',
               'gambian dalasi','gambian dalasis',['gmd'],'cent','cents',
              'argentine peso','argentine pesos',['ars'],'cent','cents',
               'taka','takas',['tk'],'cent','cents',
                'bangladeshi taka','bangladeshi takas',['bdt'],'cent','cents',
              'ukrainian hryvnia','ukrainian hryvnias',['uah'],'cent','cents',
               'papua new guinean kina','papua new guinean kinas',['pgk'],'cent','cents',
              'lithuanian litas','lithuanian litass',['ltl'],'cent','cents',
               'costa rican colon','costa rican colons',['crc'],'cent','cents',
              'yuan','yuan',['yuan'],'cent','cents',
               'finnish markka','finnish markkas',['fim'],'cent','cents',
                'thai baht','thai bahts',['thb'],'cent','cents',
              'won','won',['won'],'cent','cents',
               'canadian dollar','canadian dollars',['cad'],'cent','cents',
                'east caribbean dollar','east caribbean dollars',['xcd'],'cent','cents',
              'pakistani rupee','pakistani rupees',['pkr'],'paisa','paisas',
               'swedish kronor','swedish kronor',['sek'],'cent','cents',
              'korean won','korean won',['krw'],'cent','cents',
                'vietnamese dong','vietnamese dongs',['vnd'],'cent','cents',
              'bahamian dollar','bahamian dollars',['bsd'],'cent','cents',
               'zloty','zlotys',['zl'],'cent','cents',
                'saudi riyal','saudi riyals',['sar'],'cent','cents',
              'saint helena pound','saint helena pounds',['shp'],'cent','cents',
               'solomon islands dollar','solomon islands dollars',['sbd'],'cent','cents',
              'south sudanese pound','south sudanese pounds',['ssp'],'cent','cents'   
    ]

def makeCurrencyDicts(nameInfo):
  currDict = {}
  currPluralDict = {}
  centDict = {}
  centPluralDict = {}
  for i in range(0,len(nameInfo),5):
    currName,currPluralName,symbols,centName,centPluralName = nameInfo[i:i+5]
    if type(symbols)!=list: raise Exception('makeCurrencyDicts: Not a symbol list',symbols)
    for symbol in symbols:
      if symbol in currDict: raise Exception('makeCurrencyDicts: Symbol already in dict:',symbol)
      currDict[symbol] = currName
      currPluralDict[symbol] = currPluralName
      centDict[symbol] = centName
      centPluralDict[symbol] = centPluralName
  return currDict,currPluralDict,centDict,centPluralDict


   

currDict,currPluralDict,centDict,centPluralDict = makeCurrencyDicts(currencyInfo)
amountRegex = '|'.join(amountDict.keys())
currFirstRegex = re.sub('([$.])',r'\\\1','|'.join(currDict.keys()))
currencyFirstPat = re.compile(r'(' + currFirstRegex + r') ?(-?\d[\d, ]*)(\.\d+)? ?(' + amountRegex + ')?$')
currLastRegex = re.sub('([$.])',r'\\\1','|'.join(currDict.keys()))
currencyLastPat = re.compile(r'(-?\d[\d, ]*)(\.\d+)? ?(' + amountRegex + ')?' + ' ?' + r'(' + currLastRegex + ')$')
commaSpacePat = re.compile('[, ]')



def currencyFirst(moneyStr): 
  ms = moneyStr.lower()
  m = currencyFirstPat.match(ms)

  if m is None:
    out = moneyStr
  else:
    curr,qLeft,qPointRight,scale = m.groups()
    curr = dotPat.sub('',curr)
    qLeft = commaSpacePat.sub('',qLeft)    
    
    currName = currDict[curr]
    centName = centDict[curr]
    total = qLeft
    if qPointRight is not None:
      total += qPointRight
      qPointRight = qPointRight[1:]
    if int(qLeft)!=1 or scale is not None: currName = currPluralDict[curr]
    if scale is None:
      amountInWords = ntow(qLeft)
      if amountInWords=='zero':
        amountPortion = ''
      else:
        amountPortion = amountInWords + ' ' + currName
      if qPointRight is None:
        centPortion = '';
      else:
        qPointRight = qPointRight.ljust(2,'0')
        centsInWords = ntow(qPointRight)
        if centsInWords=='zero':
          centPortion = ''
        else:
          if int(qPointRight)>1: centName = centPluralDict[curr]
          centPortion = ntow(qPointRight) + ' ' + centName
      if amountPortion!='':
        out = amountPortion
        if centPortion!='': out += ' and ' + centPortion
      else:
        out = centPortion
      if out=='': out = 'zero ' + currName
          
    else:
      if qPointRight is None:
        out = ntow(qLeft) + ' ' + amountDict[scale] + ' ' + currName        
      else:
        if qPointRight=='00' or qPointRight=='0':
          out = ntow(qLeft) + ' ' + amountDict[scale] + ' ' + currName
        else:
          out = numDecComma(total) + ' ' + amountDict[scale] + ' ' + currName
  return out 

def currencyLast(moneyStr): 
  ms = moneyStr.lower()
  m = currencyLastPat.match(ms)
  if m is None:
    out = moneyStr
  else:
    qLeft,qPointRight,scale,curr = m.groups()
    qLeft = commaSpacePat.sub('',qLeft)    
    
    currName = currDict[curr]
    centName = centDict[curr]
    total = qLeft
    if qPointRight is not None:
      total += qPointRight
      qPointRight = qPointRight[1:]
    if int(qLeft)!=1 or scale is not None: currName = currPluralDict[curr]
    if scale is None:
      amountInWords = ntow(qLeft)
      if amountInWords=='zero':
        amountPortion = ''
      else:
        amountPortion = amountInWords + ' ' + currName
      if qPointRight is None:
        centPortion = '';
      else:
        qPointRight = qPointRight.ljust(2,'0')
        centsInWords = ntow(qPointRight)
        if centsInWords=='zero':
          centPortion = ''
        else:
          if int(qPointRight)>1: centName = centPluralDict[curr]
          centPortion = ntow(qPointRight) + ' ' + centName
      if amountPortion!='':
        out = amountPortion
        if centPortion!='': out += ' and ' + centPortion
      else:
        out = centPortion
      if out=='': out = 'zero ' + currName
          
    else:
      if qPointRight is None:
        out = ntow(qLeft) + ' ' + amountDict[scale] + ' ' + currName        
      else:
        if qPointRight=='00' or qPointRight=='0':
          out = ntow(qLeft) + ' ' + amountDict[scale] + ' ' + currName
        else:
          out = numDecComma(total) + ' ' + amountDict[scale] + ' ' + currName
  return out 

measureInfo = ['cal','cal',['cal']
    #volume
    ,'milliliter','milliliters',['ml','mL']
    ,'giga liter','giga liters',['GL']
    ,'cubic meter','cubic meters',['m3','m³']
    ,'cubic kilometer','cubic kilometers',['km3','km³']
    ,'c c',None,['cc']
    ,'tera liter','tera liters',['TL']
    ,'mega liter','mega liters',['ML']
    ,'nano liter','nano liters',['nl']
    ,'kilo liter','kilo liters',['kl']
    ,'hecto liter','hecto liters',['hl']
    ,'gallon','gallons',['gal']
    ,'barrel','barrels',['bbl','barrel','barrels']
    ,'million barrel','million barrels',['million barrels']    
    ,'cubic','cubic',['cubic']
    
    ,'gigabyte','gigabytes',['GB']
    ,'kilobyte','kilobytes',['kB','KB','kilobyte','kilobytes']
    ,'terabyte','terabytes',['TB']
    ,'megabyte','megabytes',['MB','megabyte','megabytes']  
    ,'megabit','megabits',['Mb']
    ,'gigabit','gigabits',['Gb']
    ,'kilobit','kilobits',['kb','Kb']
    ,'peta bit','peta bits',['Pb']
    
    ,'kilogram force','kilograms force',['kgf']
    
    ,'cubic meter per second','cubic meters per second',['m³/s','m3/s']
    ,'liter per second','liters per second',['liters/second']
    
    ,'per hour',None,['/h']
    
    #power
    ,'watt','watts',['watt','watts']    
    ,'megawatt','megawatts',['MW']
    ,'peta watt','peta watts',['PW']
    ,'horsepower','horsepower',['hp']
    ,'kilowatt','kilowatts',['kW','KW']
    ,'gigawatt','gigawatts',['GW']
    ,'tera watt','tera watts',['TW']
    ,'mega joule','mega joules',['MJ']
    
    ,'megabit per second','megabits per second',['Mbps']
    ,'megabyte per second','megabytes per second',['MB/s']
    ,'gigabit per second','gigabits per second',['Gb/s']

    ,'kilogram per meter','kilograms per meter',['kg/m']
    
    ,'electron volt','electron volts',['eV']
    
    ,'milligram per kilogram','milligrams per kilogram',['mg/kg','mg/Kg']
    ,'gram per kilogram','grams per kilogram',['g/kg']

    #density
    ,'kilogram per cubic meter','kilograms per cubic meter',['kg/m3']
    ,'milligram per cubic meter','milligrams per cubic meter',['mg/m3']
    ,'gram per c c','grams per c c',['g/cm3']    
    ,'nano gram per deci liter','nano grams per deci liter',['ng/dL']
    ,'milligram per milliliter','milligrams per milliliters',['mg/ml']
    ,'microgram per milliliter','micrograms per milliliters',['μg/ml']                                                             
    ,'milligram per liter','milligrams per liters',['mg/L']
    ,'gram per liter','grams per liters',['g/liter']
                                                             
    ,'per year','per years',['/year']

    ,'kilogram per square meter','kilograms per square meter',['kg/m²']
    ,'kilogram per hectare','kilograms per hectare',['kg/ha']    

    
    ,'tera watt hour','tera watt hours',['TWh']
    ,'kilo watt hour','kilo watt hours',['kWh']
    ,'watt hour','watt hours',['Wh']
    ,'giga watt hour','giga watt hours',['GWh']
    ,'mega watt hour','mega watt hours',['MWh']

    ,'deci farad','deci farads',['dF']
    
    ,'milli ampere','milli amperes',['mA']
    ,'kilo ampere','kilo amperes',['KA','kA']
    ,'deci ampere','deci amperes',['dA']
    ,'mega ampere','mega amperes',['MA']
    
    ,'pico henry','pico henrys',['pH']
    ,'megahertz',None,['MHz','Mhz']
    ,'gigahertz',None,['GHz']
    ,'kilohertz',None,['kHz','KHz','kilohertz']
    ,'hertz',None,['hertz','Hz','hz']
    ,'revolution per minute','revolutions per minute',['rpm']
    ,'milli hertz','milli hertz',['mHz']

    ,'milli amp hour','milli amp hours',['mAh']
    
    ,'becquerel per cubic meter','becquerels per cubic meter',['Bq/m3']
    
    ,'kilo volt','kilo volts',['kV','KV']
    ,'milli volt','milli volts',['mV']
    ,'volt','volts',['V']
    ,'mega volt','mega volts',['MV']
    
    #time
    ,'million years','million years',['million years']     
    ,'year','years',['yr','yrs','year','years']
    ,'month','months',['month','months'] 
    ,'week','weeks',['week','weeks']     
    ,'millisecond','milliseconds',['ms']
    ,'giga second','giga seconds',['Gs']
    ,'minute','minutes',['min','mins','minute','minutes']
    ,'second','seconds',['second','seconds']
    ,'peta second','peta seconds',['Ps']
    ,'kilo second','kilo seconds',['Ks','ks']
    ,'hour','hours',['hr','hrs','h','hour','hours']      
    ,'microsecond','microseconds',['µs','μs']
    ,'mega second','mega seconds',['Ms']
    ,'tera second','tera seconds',['Ts']
    
    ,'megapascal','megapascals',['MPa']
    ,'bar','bars',['bar']
    ,'pascal','pascals',['Pa']
    ,'millibar','millibars',['mbar']
    ,'atmosphere','atmospheres',['atm']
    ,'kilopascal','kilopascals',['kPa']    
    ,'giga pascal','giga pascals',['GPa']  
    
    #weight
    ,'kilogram','kilograms',['kg','kilogram','kilograms']
    ,'gram','grams',['g','gram','grams']
    ,'milligram','milligrams',['mg']
    ,'microgram','micrograms',['µg','μg'] #these look the same but are not same
    ,'peta gram','peta grams',['Pg']
    ,'pound','pounds',['lb','lbs','pound','pounds']
    ,'hundredweight',None,['cwt']
    ,'ounce','ounces',['oz','ounce','ounces']
    ,'stone','stone',['st','stone']
    ,'mega gram','mega grams',['Mg']
    ,'ton','tons',['ton','tons','tonne','tonnes']
    
    ,'dalton','daltons',['Da']
    ,'sievert','sieverts',['Sv']
    
    #distance
    ,'meter','meters',['m','meter','meters','metre','metres']
    ,'kilometer','kilometers',['km','Km','kilometer','kilometers','kilometre','kilometres']
    ,'mile','miles',['mi','mile','miles']
    ,'foot','feet',['foot','feet',"'",'ft']
    ,'inch','inches',['inch','inches','"','in']
    ,'nanometer','nanometers',['nm']
    ,'millimeter','millimeters',['mm','millimeter','millimeters','millimetre','millimetres']
    ,'chain','chains',['ch']
    ,'yard','yards',['yd','yard','yards']
    ,'centimeter','centimeters',['cm','centimeter','centimeters','centimetre','centimetres']
    ,'micrometer','micrometers',['µm','μm']
    ,'astronomical unit','astronomical units',['AU']
    ,'deci meter','deci meters',['dm']
    ,'giga meter','giga meters',['Gm']
    ,'light year','light years',['ly']
    ,'million light years','million light years',['million light years']
    ,'nautical mile','nautical miles',['nautical miles']
       
    ,'per square kilometer','per square kilometers',['/km²','/km2','per km²']
    ,'per square mile','per square miles',['/mi²','/mi2','/sq mi']
    
    #speed
    ,'mile per hour','miles per hour',['mph','mile per hour','miles per hour']
    ,'kilometer per hour','kilometers per hour',['km/h','kph','km/hr']
    ,'kilometer per second','kilometers per second',['km/s']    
    ,'meter per hour','meters per hour',['m/h']
    ,'meter per second','meters per second',['meters/second','m/s']
    ,'foot per second','feet per second',['ft/s']
    ,'knot','knots',['kt','kts','knot','knots']
    
    ,'per second','per seconds',['/s']
    
    #area
    ,'square kilometer','square kilometers',['km²','km2']
    ,'square mile','square miles',['sq mi','mi²','mi2','square mile','square miles']
    ,'hectare','hectares',['ha','hectare','hectares']
    ,'square meter','square meters',['m2','m²']
    ,'million square meter','million square meters',['million m²']    
    ,'square foot','square feet',['sq ft']
    ,'square centimeter','square centimeters',['cm2']
    ,'square millimeter','square millimeters',['mm2','mm²']
    ,'acre','acres',['acre','acres']
    
    ,'ton per hour','tons per hour',['tons/hour']
    
    ,'kilometer per square kilometer','kilometers per square kilometer',['km/km2']
    
    ,'square','square',['sq','square']
    
    ,'mole','moles',['mol','mols','mole','moles']

    ,'calory','calories',['calory','calories']
    
    ,'kilo calory per mole','kilo calories per mole',['kcal/mol']
    
    ,'kilo joule per mole','kilo joules per mole',['kJ/mol']
    
    ,'kilo joule per kilogram','kilo joules per kilogram',['kJ/kg']
    
    ,'kilo joule per cubic meter','kilo joules per cubic meter',['kJ/m³']
    
    ,'kilo watt hour per kilogram','kilo watt hours per kilogram',['kWh/kg']

    ,'millisievert per year','millisieverts per year',['mSv/yr']
    
    ,'newton meter','newton meters',['Nm']

    
    ,'candela','candelas',['cd']
    ,'kilo lumen','kilo lumens',['Klm']
    
    ,'kilonewton','kilonewtons',['kN']
    ,'mega newton','mega newtons',['MN']
    
    ,'kilo coulomb','kilo coulombs',['KC']
    ,'mega coulomb','mega coulombs',['MC']
    
    ,'decibel','decibels',['dB','decibel','decibels']
    ,'degree celsius','degrees celsius',['degrees C']

    ,'degree','degrees',['degree','degrees']
    
    ,'percent',None,['percent','%','pc']
    
    ,'nanobarn','nanobarns',['nanobarn','nanobarns']    
    
    ,'east','east',['east']
    ,'south','south',['south'] 

    ]
#
measDict,measPluralDict = makeDicts3(measureInfo)
measureDecPat = re.compile(r'((?:-?[\d,]+)(?:\.\d+)|(?:-?[\d,]+)|(?:-?\.\d+))( )?(' + ('|'.join(measDict.keys())) + ')$')
def measurementDec(string):
  m = measureDecPat.match(string)
  if m is None:
    out = string
  else:
    val,space,measUnit = m.groups()
    if measUnit=='st' and space is None:
      out = string
    else:
      val = commaPat.sub('',val)     
      measureName = measDict[measUnit]
      out,leftout,rightout = numDec1(val)
      if abs(float(val))!=1 or rightout!='':
        measureName = measPluralDict[measUnit]    
      out += ' ' + measureName
  return out


fractionPat = re.compile(r'(-)?(?:([\d,]+) )?([\d,]+) ?/ ?([\d,]+)$')
def fraction(string):
  m = fractionPat.match(string)
  if m is None:
    out = string
  else:
    minus,whole,numer,denom = m.groups()
    numer = commaPat.sub('',numer)
    denom = commaPat.sub('',denom)
    if numer[0]=='0' and whole is not None:
      numer = whole + numer
      whole = None
    numerVal = int(numer)
    denomVal = int(denom)    
    out = ''
    if minus is not None:
      out += 'minus '
    if whole is not None:
      whole = commaPat.sub('',whole)      
      out += ntow(whole) + ' and '
      
    if numerVal==1:
      if whole is not None:
        if denomVal==8: numerPart = 'an'
        else: numerPart = 'a'
      else:
        numerPart = 'one'
    else:
        numerPart = ntow(numer)      

    if denomVal==1: denomPart = 'over one'
    elif denomVal==2:
      if numerVal==1: denomPart = 'half'
      else: denomPart = 'halves'
    elif denomVal==3:
      if numerVal==1: denomPart = 'third'
      else: denomPart = 'thirds'
    elif denomVal==4:
      if numerVal==1: denomPart = 'quarter'
      else: denomPart = 'quarters'
    elif denomVal==5:
      if numerVal==1: denomPart = 'fifth'
      else: denomPart = 'fifths'
    elif denomVal==6:
      if numerVal==1: denomPart = 'sixth'
      else: denomPart = 'sixths'
    elif denomVal==7:
      if numerVal==1: denomPart = 'seventh'
      else: denomPart = 'sevenths'
    elif denomVal==8:
      if numerVal==1: denomPart = 'eighth'
      else: denomPart = 'eighths'
    elif denomVal==9:
      if numerVal==1: denomPart = 'ninth'
      else: denomPart = 'ninths'
    else:
      if numerVal==1: denomPart = ntow(denom,ordinal=True)
      else: denomPart = ntow(denom,ordinal=True) + 's'

    out += numerPart + ' ' + denomPart
  
  return out

fractionCharInfo = [
    'one half','a half',['½']
    ,'one third','a third',['⅓']
    ,'one quarter','a quarter',['¼']
    ,'one eighth','an eighth',['⅛']
    ,'three quarters','three quarters',['¾']
    ,'two thirds','two thirds',['⅔']
    ,'seven eighths','seven eighths',['⅞']
    ,'five eighths','five eighths',['⅝']    
    ]

def makeFractionCharDicts(info):
  woWholeDict = {}
  wWholeDict = {}
  for i in range(0,len(info),3):
    woWholeName,wWholeName,symbols = info[i:i+3]
    if type(symbols)!=list: raise Exception('makeFractionCharDicts: Not a symbol list',symbols)
    for symbol in symbols:
      if symbol in woWholeDict: raise Exception('makeFractionCharDicts: Symbol already in dict:',symbol)
      woWholeDict[symbol] = woWholeName
      wWholeDict[symbol] = wWholeName
  return woWholeDict,wWholeDict

woWholeFracDict,wWholeFracDict = makeFractionCharDicts(fractionCharInfo)

fractionCharPat = re.compile(r'(-)?(?:(\d[\d,]*) ?)?(' +  ('|'.join(woWholeFracDict.keys())) + ')$')
def fractionChar(string):
  m = fractionCharPat.match(string)
  if m is None:
    out = string
  else:
    minus,whole,frac = m.groups()
    if minus is None: minusPart = ''
    else: minusPart = 'minus '
    if whole is None:
      wholePart = ''
    else:
      wholenc = commaPat.sub('',whole)
      wholePart = ntow(wholenc) + ' and '
      
    if whole is None: fracPart = woWholeFracDict[frac]
    else: fracPart = wWholeFracDict[frac]
    
    out = minusPart + wholePart + fracPart
  return out




measureFracPat = re.compile(r'(?:(-?[\d,]+ )?((\d+)/(\d+))) ?(' + ('|'.join(measDict.keys())) + ')$')
def measurementFrac(string):
  m = measureFracPat.match(string)
  if m is None:
    out = string
  else:
    whole,frac,numer,denom,meas = m.groups()
    if whole is None: val = frac
    else: val = whole + frac
    val = commaPat.sub('',val)     
    measureName = measDict[meas]
    valPart = fraction(val)
    if int(numer)==1 and int(denom)>1:
      if measureName=='percent': joinStr = ' '
      else:
        if valPart.endswith('one half'):
          valPart = re.sub('one half','half',valPart)
          joinStr = ' a '
        else: joinStr = ' of a '
    else:
      joinStr = ' of a '      
    
    out = valPart + joinStr + measureName
  return out

measureFracCharPat = re.compile(r'(-?(?:[\d,]+)?(?:' +  ('|'.join(woWholeFracDict.keys())) + ')) ?(' + ('|'.join(measDict.keys())) + ')$')
def measurementFracChar(string):
  m = measureFracCharPat.match(string)
  if m is None:
    out = string
  else:
    val,meas = m.groups()
    val = commaPat.sub('',val)
    measureName = measPluralDict[meas]
    valPart = fractionChar(val)
    out = valPart + ' ' + measureName
  return out

romanNumeralValues = [('M',1000),('CM',900),('D',500),('CD',400),('C',100),('XC',90),('L',50),('XL',40),('X',10),('IX',9),('V',5),('IV',4),('I',1)]
romanNumeralPat = re.compile(r'^M{0,3}(CM|D?C{0,4}|CD)?(XC|L?X{0,4}|XL)?(IX|V?I{0,4}|IV)?$')
def romanToInt(remaining):
  if romanNumeralPat.search(remaining) is None:
    out = None
  else:
    total = 0  
    for rn,val in romanNumeralValues:
      while remaining.startswith(rn):
        remaining = remaining[len(rn):]
        total += val
  
    if len(remaining)>0: out = None
    else: out = total
  return out 
   

romanToCardinalPat = re.compile('([' + (''.join(set(''.join(list(zip(*romanNumeralValues))[0])))) + r"]+)(\.|'s)?$")
def romanToCardinal(romanStr):
  m = romanToCardinalPat.match(romanStr)
  if m is None:
    out = romanStr
  else:
    roman,extra = m.groups()    
    rint = romanToInt(roman)
    if rint is None: out = romanStr
    else: out = ntow(str(rint))
    if extra=="'s": out += "'s"
  return out 

romanToOrdinalPat = re.compile('([' + (''.join(set(''.join(list(zip(*romanNumeralValues))[0])))) + r"]+)(\.|'s|st|nd|rd|th)?$")
def romanToOrd(romanStr,add_the_prefix):
  m = romanToOrdinalPat.match(romanStr)
  if m is None:
    out = romanStr
  else:
    roman,ending = m.groups()    
    rint = romanToInt(roman)
    if rint is None: out = romanStr
    else:
      out = 'the ' if add_the_prefix else ''
      out += ntow(str(rint),ordinal=True)
      if ending=="'s": out += ending
  return out

def romanToOrdinal_wthe(romanStr):
  return romanToOrd(romanStr,add_the_prefix=True)

def romanToOrdinal_wothe(romanStr):
  return romanToOrd(romanStr,add_the_prefix=False)

def namesForThe(thedf):
  thedf = thedf.sort_values(['sentence_id','token_id'])
  mask = thedf.before.str.contains(r"^[CMDXLIV]+(?:\.|'s)?$") & thedf.after.str.contains(r'^the ') & (thedf.token_id!=0)
  mask.iloc[0] = False
  names = thedf.before.shift(periods=1).loc[mask]
  names = set(names.str.lower())
  return names

def makeThe(thedf,names):
  thedf = thedf.sort_values(['sentence_id','token_id'])
  sh = thedf.before.shift(periods=1)
  sh.iloc[0] = ''
  sh = sh.str.lower()
  mask = thedf.before.str.contains(r"^[CMDXLIV]+(?:\.|'s)?$") & sh.isin(names) & (thedf.token_id!=0)
  mask.iloc[0] = False
  thedf['the'] = mask
  return thedf

def romanToOrdinal_wthe_g(romanStr,candidateForThePrefix):
  if candidateForThePrefix: ret = romanToOrd(romanStr,add_the_prefix=True)
  else: ret = romanStr
  return ret
 

letterJoinedToNumbersPat = re.compile(r'([a-zA-Z][a-zA-Z.]*)[ -]?(\d+)$')
def letterJoinedToNumbers_digits(thestring):
  m = letterJoinedToNumbersPat.match(thestring)
  if m is None:
    out = thestring
  else:
    letter,numbers = m.groups()
    letter = dotPat.sub('',letter)     
    out = ' '.join(list(letter.lower())) + ' ' + digits(numbers)
  return out

def letterJoinedToNumbers_numbers(thestring):
  m = letterJoinedToNumbersPat.match(thestring)
  if m is None:
    out = thestring
  else:
    letter,numbers = m.groups()
    letter = dotPat.sub('',letter)    
    if letter.isupper():
      out = ' '.join(list(letter.lower())) + ' ' + ytow(numbers)
    else:
      out = letter.lower() + ' ' + ytow(numbers)      
  return out

lettersAndNumbersPat = re.compile(r'([a-zA-Z.]+)[ -](\d+)$')
def lettersAndNumbers(thestring):
  m = lettersAndNumbersPat.match(thestring)
  if m is None:
    out = thestring
  else:
    letters,numbers = m.groups()
    letters = dotPat.sub('',letters)
    if letters.isupper():
      out = ' '.join(list(letters.lower())) + ' ' + ytow(numbers)
    else:
      out = letters.lower() + ' ' + ytow(numbers)
  return out

numberDecQuantityPat = re.compile(r'((?:-?\d[\d,]*)(?:\.\d+)|(?:-?\d[\d,]*)|(?:-?\.\d+)) ?(' + ('|'.join(thousands)) + ')$')
def numberDecQuantity(thestring):
  m = numberDecQuantityPat.match(thestring) 
  if m is None:
    out = thestring
  else:
    number,quantity = m.groups()
    number = commaPat.sub('',number)
    combined,leftout,rightout = numDec1(number)
    if quantity is None:
      out = combined
    else:
      if rightout=='zero': rightout = 'o'
      if leftout!='':
        out = leftout
        if rightout!='':
          out += ' point ' + rightout
      elif rightout!='':
        out = 'point ' + rightout
      out = out + ' ' + quantity
  return out



numberWithDecimalPat = re.compile(r'(-?(?:\d[\d,]*)?\.\d+)$')
def numberWithDecimal(thestring):
  m = numberWithDecimalPat.match(thestring) 
  if m is None:
    out = thestring
  else:
    number = m.group(1)
    number = commaPat.sub('',number)
    combined,leftout,rightout = numDec1(number)
    out = combined
  return out

def timeOrNumberWithDecimal(thestring):
  out = time4(thestring)
  if out==thestring:
    out = numberWithDecimal(thestring)
  return out

numberOnlyPat = re.compile(r'(-?\d[\d,]*)$')
def numberOnly(thestring):
  m = numberOnlyPat.match(thestring) 
  if m is None:
    out = thestring
  else:
    number = m.group(1)
    number = commaPat.sub('',number)
    combined,leftout,rightout = numDec1(number)
    out = combined
  return out



numberWithSpacesPat = re.compile(r'([1-9]\d{0,2}(?: \d{3})+)$')
spacePat = re.compile(r' ')
def numberWithSpaces(thestring):
  m = numberWithSpacesPat.match(thestring) 
  if m is None:
    out = thestring
  else:
    number = m.group(1)
    number = spacePat.sub('',number)
    combined,leftout,rightout = numDec1(number)
    out = combined
  return out

punctDict = {'-':'dash','.':'dot'}
dotNumPunctPat = re.compile(r'(\.\d+[.-].+)$') 
def dotNumPunct(thestring):
  lc = thestring.lower() #check
  m = dotNumPunctPat.match(lc)
  if m is None:
    out = thestring
  else:
    text = m.group(1)
    outchars = []
    for c in text:
      if c=='.':
        outchars.append('dot')
      else:
        if c in punctDict:
          word = punctDict[c]
          outchars += list(word)
        elif c.isdigit():
          word = digits(c)
          outchars += list(word)
        else:
          outchars.append(c)
    out = ' '.join(outchars)
  return out
      
    
 

lettersPat = re.compile(r"([a-zA-Z&é'. ,-]+)$")
letters_charsToDropPat = re.compile(r"[^a-zA-Z&é]")
def letters(thestring):
  m = lettersPat.match(thestring)
  if m is None:
    out = thestring
  else:
    lets = m.group(1)
    lets = letters_charsToDropPat.sub('',lets)
    if len(lets)==0:
      out = thestring
    else:
      outwords = []
      for l in lets:
        if l=='&': outwords.append('and')
        elif l=='é': outwords.append('e acute')
        else: outwords.append(l)
      if len(outwords)>=2 and outwords[-1]=='s' and outwords[-2]!='e acute':
        outwords = outwords[0:-1]
        outwords[-1] += "'s"
      outwords = [ow.lower() for ow in outwords]
      out = ' '.join(outwords)
  return out
    
lettersOnlyPat = re.compile(r"([a-zA-Zé]+)$")
def lettersOnly(thestring):
  m = lettersOnlyPat.match(thestring)
  if m is None:
    out = thestring
  else:
    lets = m.group(1)
    outwords = []
    for l in lets:
      if l=='é': outwords.append('e acute')
      else: outwords.append(l)
    outwords = [ow.lower() for ow in outwords]
    out = ' '.join(outwords)
  return out

lettersWithPunctPat = re.compile(r"([a-zA-Z&é'. ,-]+)$")
lettersWithPunct_charsToDropPat = re.compile(r"[^a-zA-Z&é]")
def lettersWithPunct(thestring):
  m = lettersWithPunctPat.match(thestring)
  if m is None:
    out = thestring
  else:
    original = m.group(1)
    lets = lettersWithPunct_charsToDropPat.sub('',original)
    if lets==original or lets=='':
      out = thestring
    else:
      outwords = []
      for l in lets:
        if l=='&': outwords.append('and')
        elif l=='é': outwords.append('e acute')
        else: outwords.append(l)
      if len(outwords)>=2 and outwords[-1]=='s' and outwords[-2]!='e acute':
        outwords = outwords[0:-1]
        outwords[-1] += "'s"
      outwords = [ow.lower() for ow in outwords]
      out = ' '.join(outwords)
  return out

separateLettersWithPunctPat = re.compile(r"([a-zA-Z&é. ,-]+)$")
twoLettersPat = re.compile(r'[a-zA-Zé]{2,}')
def separateLettersWithPunct(thestring):
  m1 = separateLettersWithPunctPat.match(thestring)
  m2 = twoLettersPat.search(thestring)
  if m1 is None or m2 is not None:
    out = thestring
  else:
    out = lettersWithPunct(thestring)
  return out

def multipleLettersWithPunct(thestring):
  m2 = twoLettersPat.search(thestring)
  if m2 is None:
    out = thestring
  else:
    out = lettersWithPunct(thestring)
  return out  
  




subTokTypePat = re.compile(r'(?:'+r"(')"+r'(~)|(I)|(X)|(V)|(i)|(v)|(s)|(")|(\()|(\))|(;)|(\?)|(—)|(!)|(»)|(«)|(¿)|(¡)|([)|(])|()|( )|( +)|(,)|(\.)|(-)|(/)|(:)|(#)|(%)|(00\d+)|(0\d+)|([12]\d{3})|(\d+)|([MCDXLIV]+)|([^\W\d_a-z]+)|([A-Z][a-z]+[A-Z][a-z]+)|([A-Z][^\W\d_A-Z]+)|([^\W\d_A-Z]+)|([^\W\d_]+)|(\d+[^\W\d_]+)|([^\W\d_]+\d+)|(.)|(.*))$')
def makeSubTokAttributes(subToks):
  subTokTypes = [subTokTypePat.match(st).lastindex for st in subToks]
  subTokLengths = [len(st) for st in subToks]
  subTokLengths = [min(stl,8) for stl in subTokLengths]
  return subTokTypes,subTokLengths 


ipFindallPat = re.compile(r'\d+|.')
ipPat = re.compile(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})$')
def ip(thestring):
  m = ipPat.match(thestring)
  if m is None:
    out = thestring
  else:
    text = m.group(1)
    words = ipFindallPat.findall(text)
    for i in range(len(words)):
      if words[i]=='.':
        words[i] = 'dot'
      else:
        if len(words[i])==2:
          words[i] = ntow(words[i])
        else:
          words[i] = digits(words[i])
        words[i] = spacePat.sub('',words[i])        
        words[i] = ' '.join(list(''.join(words[i])))
    out = ' '.join(words)
  return out

webDict = {'/':'slash',':':'colon','-':'dash'
  ,'#':'hash',';':'s e m i colon'
  }

digitPat = re.compile(r'\d')
web1FindallPat = re.compile(r'\d+|[^\W\d_]+|.')
hashtagPat = re.compile(r'(#[a-zA-Z][\w.-]+)$')
def hashtag(thestring):
  m = hashtagPat.match(thestring)
  if m is None:
    out = thestring
  else:
    text = m.group(1)
    words = web1FindallPat.findall(text)
    if len(words)>1 and digitPat.search(text) is not None: words[0] = 'hash'
    else: words[0] = 'hash tag'
    for i in range(1,len(words)):
      if words[i].isdigit():
        if len(words[i])==2: words[i] = ntow(words[i])
        else: words[i] = digits(words[i])
      elif words[i] in webDict:
        words[i] = webDict[words[i]]
    
    out = ' '.join(words).lower()
  return out



webExpandDict = {'_':'underscore',',':'comma','(':'openingparenthesis'
  ,')':'closingparenthesis','~':'tilde','%':'percent'
  ,'vol':'volume',"'":"singlequote"
  }

web1Pat = re.compile(r'((?:[a-zA-Z/][\w/-]*|^[\w/-]+)\.[a-zA-Z][\w-][\w-]*)')
httpPat = re.compile(r'https?:')
web2FindallPat = re.compile(r'(?<=\.)com|\d+|[^\W\d_]+|.')

def web1(thestring):
  m = web1Pat.search(thestring)
  if m is None:
    out = thestring
  else:
    w = thestring
    w = w.lower()
    words = web2FindallPat.findall(w)
    if httpPat.match(w) is not None: hasHttp = True
    else: hasHttp = False
    
    for i in range(len(words)):
      w = words[i]
      if digitsPat.match(w) is not None:
        iw = int(w)
        if w[0]=='0': words[i] = digits(w)
        elif len(w)>4:
#          wleft4,wrightrem = w[:4],w[4:]
#          iwleft4 = int(wleft4)          
          
          wleftrem,wright4 = w[:-4],w[-4:]
          iwright4 = int(wright4)

           
          if not wright4.startswith('200') and 1900<=iwright4<=2100:
            if len(w)==6: wordsleft = ntow(wleftrem)
            else: wordsleft = digits(wleftrem)            
            wordsright = ytow(wright4)
#          elif not wleft4.startswith('200') and 1900<=iwleft4<=2100:
#            wordsleft = ytow(wleft4)
#            if len(w)==6: wordsright = ntow(wrightrem)
#            else: wordsright = digits(wrightrem)             
          else:
            wordsleft = digits(wleftrem)            
            wordsright = digits(wright4)            



          words[i] = wordsleft + ' ' + wordsright
        elif len(w)==4:
          if w.startswith('200') or iw<1900 or iw>2100: words[i] = digits(w)
          else: words[i] = ytow(w)
          
        elif len(w)<=2: words[i] = ntow(w)
        else: words[i] = digits(w)
        words[i] = ' '.join(list(spacePat.sub('',words[i])))
#        elif w in monthDict:
#          w = monthDict[w]
#          words[i] = ' '.join(list(w))           
      elif w in webDict:
        if hasHttp:
          words[i] = webDict[w]
        else:        
          words[i] = webDict[w]
          words[i] = ' '.join(list(words[i]))         
      elif w in webExpandDict:
        w = webExpandDict[w]
        words[i] = ' '.join(list(w))          
      else:        
        if w in exsingleRep_lc:
          wlct = exsingleRep_lc[w]
          if len(w)>=len(wlct): w = wlct # 4769 19 #newone
                   
        if hasHttp:
          if w not in ['com','slash','dash']:
            words[i] = ' '.join('e a c u t e' if l=='é' else l for l in list(w))
        else:
          words[i] = ' '.join('e a c u t e' if l=='é' else l for l in list(w))          

#    chars = list(''.join(words))
    words = ['dot' if c=='.' else c for c in words]
    out = ' '.join(words)
  return out

web2Pat = re.compile(r'^\.([a-z]{3})$')
def web2(thestring):
  m = web2Pat.search(thestring)
  if m is None:
    out = thestring  
  else:
    w = m.group(1)
    out = 'dot ' + ' '.join(list(w))
  return out
    
    
chemicalSymbolsDict = {
    'PbSO4':'lead two sulphate'
    ,'Al2O3':'aluminium oxide'
    ,'NO':'nitrogen monoxide'
    ,'CCl4':'carbon tetra chloride'
    ,'CO)16':'carbon hexadecoxide'
    ,'SO':'sulphur monoxide'
    ,'CO)5':'carbon pentoxide'
    ,'CO)9':'carbon nonoxide'
    ,'CO)':'carbon monoxide'
    ,'NO)':'nitrogen monoxide'
    ,'NO':'nitrogen monoxide'
    ,'MgO)':'magnesium oxide'
    ,'PO)':'phosphorus monoxide'
    ,'ClO)2':'chlorine di oxide'
    ,'ClO4':'chlorine tetroxide'  
    ,'Al2O6':'aluminium oxide'  
    ,'NaCl':'sodium chloride'  
    ,'Al2Cl3':'aluminium chloride'  
    ,'AlCl3':'aluminium chloride'  
    ,'CO)6':'carbon hexoxide'  
    ,'OO':'oxygen monoxide'  
    ,'Cl2O3':'di chlorine tri oxide'  
    ,'SnCl':'tin one chloride'  
    ,'CO)12':'carbon dodecoxide' 
  
    #,'CB₁':''  
    #,'CB₂':''  
    ,'ZnI₂':'zinc iodide'  
    ,'NH₄':'ammonium'  
    ,'SiO₂':'silicon dioxide' 
    #,'OH)₂':''  
    #,'CH₃':''  
    ,'CaCO₃':'calcium carbonate'  
    ,'NH₃':'ammonia'  
    ,'NO₃':'nitrate' 
    ,'H₂O':'water'  
    #,'Cu₃O₇':''  
    ,'CCl₄':'carbon tetrachloride'   
    ,'PI₃':'phosphorus triiodide'  
    ,'CH₂':'methylene'  
    ,'Zn(OH)₂':'zinc hydroxide'  
    ,'TiO₂':'titanium dioxide'     
    #,'C₁₀H₆':''  
    ,'SO₃':'sulfur trioxide'  
    #,'NH₂':''  
    ,'MnS₂':'manganese disulphide'  
    #,'O₂N)₃':''     
    ,'C₆H₂':'hexatriyne'  
    #,'SN₂':''  
    #,'C₁₉H₁₆':''  
    ,'AlCl₃':'aluminium chloride'  
    ,'SO₂':'sulfur dioxide'     
    #,'C(S':''  
    ,'Ca₃(AsO₄)₂':'calcium arsenate'  
    #,'SiO₅':''  
    #,'P₄S₄':''  
    ,'CO)₃':'carbon trioxide'     
    #,'F₁F₂':''  
    ,'C₃H':'propynylidyne'  
    #,'Ca5(PO4)3':''  
    #,'Mg₃B₇':''  
    #,'O₁₃Cl':''
    ,'Li2O2':'lithium peroxide'  
    ,'Li2O':'lithium oxide'  
    ,'HNO₃':'nitric acid'  
    #,'AlSi₃':''  
    ,'BeF₂':'beryllium fluoride'
    ,'PbCl₂':'lead dichloride'  
    ,'C₄H₄':'vinylacetylene'  
    ,'CO)₆':'carbon hexoxide'  
    #,'Pb₃As₄':''  
    ,'C₆H₄':'terephthalic acid'
    #,'C₂₅H₂₁':''  
    #,'N₅O':''  
    ,'XeF₂':'xenon difluoride'  

    
    }    

def chemicalSymbols(thestring):
  if thestring in chemicalSymbolsDict:
    ret = chemicalSymbolsDict[thestring]
  else:
    ret = thestring
  return ret

measurementPerDict = {
    '/MA':'per mega ampere'
    ,'/hp':'per horsepower'
    ,'/hr':'per hour'
    ,'/mL':'per milliliter'
    ,'/L':'per liter'
    ,'/lb':'per pound'
    ,'/bar':'per bar'
    ,'/ft':'per foot'
    ,'/Gs':'per giga second'
    ,'/mi':'per mile'
  
    }

def measurementPer(thestring):
  if thestring in measurementPerDict:
    ret = measurementPerDict[thestring]
  else:
    ret = thestring
  return ret

uncertainTransformTypes = [digits,'digits'
                           ,telephone_spell,'telephone_spell'
                           ,telephone_dontSpell,'telephone_dontSpell'
                           ,yearOnly,'yearOnly'
                           ,otow,'otow'
                           ,yearPunct,'yearPunct'
                           #,numberPunct,'numberPunct'
                           ,currencyLast,'currencyLast'
                           #,measurementDec,'measurementDec'
                           ,romanToCardinal,'romanToCardinal'
                           ,letterJoinedToNumbers_digits,'letterJoinedToNumbers_digits'
                           ,letterJoinedToNumbers_numbers,'letterJoinedToNumbers_numbers'
                           ,numberOnly,'numberOnly'
                           ,romanToOrdinal_wthe,'romanToOrdinal_wthe'
                           ,romanToOrdinal_wothe,'romanToOrdinal_wothe'
                           ,letters,'letters'
                           ,web1,'web1'
                           ,hashtag,'hashtag'
                           ]

limitOutputPat = re.compile(r"('|[^\W\d_])+$")
#allAlphabeticPat = re.compile(r'[^\W\d_]+$')
numberPat = re.compile(r'^[\d ,.]+$')


def limitedOutputInds(thestring):
  validTypeInds = []
  isNumber = numberPat.search(thestring) is not None
  if not isNumber and singleRep_loi.has(thestring): # single      
    validTypeInds.append(changeTypeIndex.get_loc('replaceSingle'))
  elif not isNumber and sameRep_loi.has(thestring): # same 
    validTypeInds.append(changeTypeIndex.get_loc('replaceSame'))    
  else:
    canDoLetters = False
    for i in range(0,len(uncertainTransformTypes),2):
      func,typestring = uncertainTransformTypes[i],uncertainTransformTypes[i+1]
      out = func(thestring)
      if out!=thestring:
        validTypeInds.append(changeTypeIndex.get_loc(typestring))
        if typestring=='letters': canDoLetters = True
        
    if thestring in multiRepl: # for multi replacements
      for i in range(len(multiRepl[thestring])):
        changeTypeStr = 'replaceMulti' + str(i)
        validTypeInds.append(changeTypeIndex.get_loc(changeTypeStr))        
  
    if remainingSingleRep.has(thestring): # remaining single      
      validTypeInds.append(changeTypeIndex.get_loc('replaceSingle')) 

    if remainingSameRep.has(thestring) or (canDoLetters and len(validTypeInds)==1) or len(validTypeInds)==0:
               
    #Alphabetic words unseen in train can have same replacement.  
      validTypeInds.append(changeTypeIndex.get_loc('replaceSame'))    
  return validTypeInds


def addLimitedOutputInds(thedf):
  valIndsColumn = pmap(limitedOutputInds,thedf.before,chunksize=100000)
  thedf['limOutInds'] = valIndsColumn  

def checkTransform(thedf,thefunc):
  thedf = thedf.copy()
  thedf['transformed'] = pmap(thefunc,thedf.before,chunksize=100000)
  df_incor = thedf[(thedf.transformed!=thedf.before) & (thedf.transformed!=thedf.after)]
  df_cor = thedf[(thedf.transformed!=thedf.before) & (thedf.transformed==thedf.after)]
  return df_cor,df_incor


def transformCertain2(df,ctdf,forceType,check):
  #roman to ordinal with the, guaranteed
  ctdf,df = getChangeTransformed2(df,ctdf,romanToOrdinal_wthe_g,'romanToOrdinal_wthe',['before','the'],forceType,check) 
  return ctdf,df  

def transformCertain1a(df,ctdf,forceType,check):
  # dot followed by 3 lower case letters
  ctdf,df = getChangeTransformed(df,ctdf,web2,'web2',forceType,check) 
  
  return ctdf,df  

def transformCertain1b(df,ctdf,forceType,check):
  #date dash. none incorrect
  ctdf,df = getChangeTransformed(df,ctdf,dateDash,'dateDash',forceType,check) 
  #date year s. none incorrect
  ctdf,df = getChangeTransformed(df,ctdf,ys,'ys',forceType,check) 
  #year era. none incorrect
  ctdf,df = getChangeTransformed(df,ctdf,yearEra,'yearEra',forceType,check) 
  #time with 1 number. none incorrect
  ctdf,df = getChangeTransformed(df,ctdf,time1,'time1',forceType,check) 
  #time with 2 numbers. two incorrect but from time class
  ctdf,df = getChangeTransformed(df,ctdf,time2,'time2',forceType,check) 
  #time with 3 numbers. none incorrect
  ctdf,df = getChangeTransformed(df,ctdf,time3,'time3',forceType,check) 
  #fraction. one incorrect in fraction class
  ctdf,df = getChangeTransformed(df,ctdf,fraction,'fraction',forceType,check) 
  #fraction chars. none incorrect
  ctdf,df = getChangeTransformed(df,ctdf,fractionChar,'fractionChar',forceType,check) 
  #number and quantity. none incorrect
  ctdf,df = getChangeTransformed(df,ctdf,numberDecQuantity,'numberDecQuantity',forceType,check) 
  #month date year to words. none incorrect
  ctdf,df = getChangeTransformed(df,ctdf,wdayMonthDateYear,'wdayMonthDateYear',forceType,check) 
  #weekday date month year to words. none incorrect
  ctdf,df = getChangeTransformed(df,ctdf,wdayDateMonthYear,'wdayDateMonthYear',forceType,check) 

  #year month date era
  ctdf,df = getChangeTransformed(df,ctdf,yearMonthDate,'yearMonthDate',forceType,check) 

  #currency with symbol first. none incorrect
  ctdf,df = getChangeTransformed(df,ctdf,currencyFirst,'currencyFirst',forceType,check) 
  #measurements with fraction characters. none incorrect
  ctdf,df = getChangeTransformed(df,ctdf,measurementFracChar,'measurementFracChar',forceType,check) 
  #measurements with fractions. two incorrect but they are in measure class.
  ctdf,df = getChangeTransformed(df,ctdf,measurementFrac,'measurementFrac',forceType,check) 
  #ip, none incorrect
  ctdf,df = getChangeTransformed(df,ctdf,ip,'ip',forceType,check)
  #time or number with decimal
  ctdf,df = getChangeTransformed(df,ctdf,timeOrNumberWithDecimal,'timeOrNumberWithDecimal',forceType,check)  
  #number with space
  ctdf,df = getChangeTransformed(df,ctdf,numberWithSpaces,'numberWithSpaces',forceType,check)  
  # dot number punc char
  ctdf,df = getChangeTransformed(df,ctdf,dotNumPunct,'dotNumPunct',forceType,check)  
  # number followed with space and some letters
  ctdf,df = getChangeTransformed(df,ctdf,numberAndLetters,'numberAndLetters',forceType,check)    
  #number punctuation. needs nn
  ctdf,df = getChangeTransformed(df,ctdf,numberPunct,'numberPunct',forceType,check)
  #latest --------- measurements with decimal point. some incorrect
  ctdf,df = getChangeTransformed(df,ctdf,measurementDec,'measurementDec',forceType,check)
  #chemical symbols
  ctdf,df = getChangeTransformed(df,ctdf,chemicalSymbols,'chemicalSymbols',forceType,check)
  #slash and measurement unit
  ctdf,df = getChangeTransformed(df,ctdf,measurementPer,'measurementPer',forceType,check)
   
  
  return ctdf,df



def transformUncertain(df,ctdf,forceType,check):
  #digits. needs nn
  ctdf,df = getChangeTransformed(df,ctdf,digits,'digits',forceType,check)
  #telephone spell words. needs nn
  ctdf,df = getChangeTransformed(df,ctdf,telephone_spell,'telephone_spell',forceType,check)
  #telephone don't spell words. needs nn
  ctdf,df = getChangeTransformed(df,ctdf,telephone_dontSpell,'telephone_dontSpell',forceType,check)
  #year only. needs nn
  ctdf,df = getChangeTransformed(df,ctdf,yearOnly,'yearOnly',forceType,check)
  #ordinal. 3 incorrect, needs nn.
  ctdf,df = getChangeTransformed(df,ctdf,otow,'otow',forceType,check)
  #year punctuation. needs nn.
  ctdf,df = getChangeTransformed(df,ctdf,yearPunct,'yearPunct',forceType,check)


  #currency with symbol last. one incorrect
  ctdf,df = getChangeTransformed(df,ctdf,currencyLast,'currencyLast',forceType,check)
  #roman numerals as numbers. needs nn
  ctdf,df = getChangeTransformed(df,ctdf,romanToCardinal,'romanToCardinal',forceType,check)
  #letter joined to numbers, convert to digits. needs nn
  ctdf,df = getChangeTransformed(df,ctdf,letterJoinedToNumbers_digits,'letterJoinedToNumbers_digits',forceType,check)
  #letter joined to numbers, convert to numbers. needs nn
  ctdf,df = getChangeTransformed(df,ctdf,letterJoinedToNumbers_numbers,'letterJoinedToNumbers_numbers',forceType,check)
  #number. needs nn
  ctdf,df = getChangeTransformed(df,ctdf,numberOnly,'numberOnly',forceType,check)
  #roman numerals to ordinal with "the". needs nn
  ctdf,df = getChangeTransformed(df,ctdf,romanToOrdinal_wthe,'romanToOrdinal_wthe',forceType,check)
  #roman numerals to ordinal without "the".needs nn
  ctdf,df = getChangeTransformed(df,ctdf,romanToOrdinal_wothe,'romanToOrdinal_wothe',forceType,check)
  #letters spelled out. needs nn
  ctdf,df = getChangeTransformed(df,ctdf,letters,'letters',forceType,check)
  #web1
  ctdf,df = getChangeTransformed(df,ctdf,web1,'web1',forceType,check)
  #hashtag
  ctdf,df = getChangeTransformed(df,ctdf,hashtag,'hashtag',forceType,check)

  return ctdf,df



loadCode = 'train test'

if loadCode=='extra train':
  extradf = pd.read_pickle('../temp/extra.pickle')
  trn_sid = extradf.sentence_id.unique()
  targ_sentid_offset = max(trn_sid) + 1   
  traindf = pd.read_csv('../data/en_train.csv',keep_default_na=False) 
  train_sid = traindf.sentence_id.unique()
  np.random.shuffle(train_sid)
  foldSliceInds = list(makeFoldSliceInds(len(train_sid),nFolds=5))
  tar_sid = train_sid[makeIndsFromSliceInds(len(train_sid),foldSliceInds[0],False)]
  tardf = traindf[traindf.sentence_id.isin(tar_sid)].copy()
  tardf['sentence_id'] += targ_sentid_offset 
  tar_sid = tardf.sentence_id.unique()
  alldf = extradf.append(tardf)
  extrarepdf = pd.read_pickle('../temp/extrarep_wot.pickle')
  with open('../temp/nameset_wot.pickle','rb') as f:
    nameSetForThe = pickle.load(f)   
  usingTestSet = False
  del extradf,traindf,tardf,train_sid  
elif loadCode=='extra test':
  extradf = pd.read_pickle('../temp/extra.pickle')
  trn_sid = extradf.sentence_id.unique()
  targ_sentid_offset = max(trn_sid) + 1   
  testdf = pd.read_csv('../data/en_test_2.csv',keep_default_na=False)
  testdf['sentence_id'] += targ_sentid_offset  
  tar_sid = testdf.sentence_id.unique()
  alldf = extradf.append(testdf)
  extrarepdf = pd.read_pickle('../temp/extrarep_wt.pickle')
  with open('../temp/nameset_wt.pickle','rb') as f:
    nameSetForThe = pickle.load(f)   
  usingTestSet = True  
  del extradf,testdf  
elif loadCode=='train':
  traindf = pd.read_csv('../data/en_train.csv',keep_default_na=False)
  #traindf = pd.read_csv('../temp/extradatadf.csv',keep_default_na=False)   
  train_sid = traindf.sentence_id.unique()
  np.random.shuffle(train_sid)
  foldSliceInds = list(makeFoldSliceInds(len(train_sid),nFolds=5))
  tar_sid = train_sid[makeIndsFromSliceInds(len(train_sid),foldSliceInds[0],False)]
  trn_sid = train_sid[makeIndsFromSliceInds(len(train_sid),foldSliceInds[0],True)]
  targ_sentid_offset = 0
  alldf = traindf
  extrarepdf = pd.read_pickle('../temp/extrarep_wot.pickle')
  with open('../temp/nameset_wot.pickle','rb') as f:
    nameSetForThe = pickle.load(f)
  usingTestSet = False  
  del train_sid
elif loadCode=='train test':
  traindf = pd.read_csv('../data/en_train.csv',keep_default_na=False)  
  trn_sid = traindf.sentence_id.unique()
  targ_sentid_offset = max(trn_sid) + 1   
  testdf = pd.read_csv('../data/en_test_2.csv',keep_default_na=False)
  testdf['sentence_id'] += targ_sentid_offset  
  tar_sid = testdf.sentence_id.unique()
  alldf = traindf.append(testdf)
  extrarepdf = pd.read_pickle('../temp/extrarep_wt.pickle')
  with open('../temp/nameset_wt.pickle','rb') as f:
    nameSetForThe = pickle.load(f)   
  usingTestSet = True
  del traindf,testdf 
else:
  raise Exception('Bad data load code "',loadCode,'"')
  
#trim spaces
alldf['before'] = alldf.before.str.strip()  
######################################


################ dicts from extra data
exsingleRep = TwoDicts(makeReplaceSingleDict(None,extrarepdf),False)
exsameRep = TwoDicts(makeReplaceSameDict(None,extrarepdf),False)
exanyRep = TwoDicts(makeReplaceDict(None,extrarepdf),False)
#exanyRep2 = TwoDicts(makeReplaceDict(None,extrarepdf),True)

exsingleRep_lc = {k.lower():v.lower() for k,v in exsingleRep.firstDict.items()}
############################3



########### word embeddings
#wordEmbDict = makeEmbeddingsDict_glove(alldf.before,'../data/glove.6B.100d.txt')
wordEmbDict = makeEmbeddingsDict_w2v(alldf.before,'../data/GoogleNews-vectors-negative300.bin')
wordEmbMat,wordEmbIndsDict = makeEmbeddingsMatrix(wordEmbDict)
alldf['wordEmbInd'] = [wordEmbIndsDict[b] if b in wordEmbIndsDict else 0 for b in alldf.before]
preTrainedWordEmbs=makePretrainedEmbeddings(wordEmbMat)

tardf = alldf[alldf.sentence_id.isin(tar_sid)].copy()
trndf = alldf[alldf.sentence_id.isin(trn_sid)].copy()
del alldf


tokIndex = makeIndex(trndf.before.str.lower(),minCount=1)
tokEosInd,tokIndex = appendToIndex(tokIndex,eosStr,True)
tokDefInd,tokIndex = appendToIndex(tokIndex,defaultStr,True)

if not usingTestSet:
  singleRep = TwoDicts(makeReplaceSingleDict(trndf,None),False)
  sameRep = TwoDicts(makeReplaceSameDict(trndf,None),False)


trndf = removeLongSentences(trndf,MAX_SENTENCE_LENGTH)

alldf = trndf.append(tardf)
del trndf,tardf

if usingTestSet:
  singleRep_loi,sameRep_loi = exsingleRep,exsameRep
else:
  singleRep_loi,sameRep_loi = singleRep,sameRep

##############################
alldf = makeThe(alldf,nameSetForThe)


#certain
temp,alldf = transformCertain1a(alldf,None,forceType=True,check=True)
temp['assignedType'] = temp.changeType
temp['changeType'] = defaultStr
allctdf = temp
temp,alldf = transformCertain1b(alldf,None,forceType=True,check=True)
temp['assignedType'] = temp.changeType
temp['changeType'] = defaultStr
allctdf = allctdf.append(temp)
#certain2
temp,alldf = transformCertain2(alldf,None,forceType=True,check=True)
temp['assignedType'] = temp.changeType
allctdf = allctdf.append(temp)
#uncertain
temp,alldf = transformUncertain(alldf,None,forceType=True,check=True)
temp['assignedType'] = temp.changeType
allctdf = allctdf.append(temp)
#single transform tokens
temp,alldf = replaceSingle(alldf,None,exsingleRep,forceType=True,check=True)
temp['assignedType'] = temp.changeType
allctdf = allctdf.append(temp)
#same transform tokens
temp,alldf = replaceSame(alldf,None,exsameRep,forceType=True,check=True)
temp['assignedType'] = temp.changeType
allctdf = allctdf.append(temp)

tardf = alldf[alldf.sentence_id.isin(tar_sid)].copy()
trndf = alldf[alldf.sentence_id.isin(trn_sid)].copy()
del alldf

#latest
multiRepl = makeReplaceMultiDict(None,extrarepdf) #(trndf,None)
remainingSingleRep = exsingleRep #TwoDicts(makeReplaceSingleDict(trndf,None),False)
remainingSameRep = exsameRep #TwoDicts(makeReplaceSameDict(trndf,None),False)
alldf = trndf.append(tardf)
del trndf,tardf

#multi replace
temp,alldf = replaceMulti(alldf,None,multiRepl,forceType=True)
temp['assignedType'] = temp.changeType
allctdf = allctdf.append(temp)
#remaining single transform tokens
temp,alldf = replaceSingle(alldf,None,remainingSingleRep,forceType=True,check=True)
temp['assignedType'] = temp.changeType
allctdf = allctdf.append(temp)
#remaining same transform tokens
temp,alldf = replaceSame(alldf,None,remainingSameRep,forceType=True,check=True)
temp['assignedType'] = temp.changeType
allctdf = allctdf.append(temp)


allctdf_wrong=allctdf[allctdf.changed!=allctdf.after]
print('wrong',len(allctdf_wrong))
print('remaining',len(alldf),'different',sum(alldf.before!=alldf.after))
#raise Exception('=============')

temp,alldf = transformRemainingUnknown(alldf,None,forceType=True)
temp['assignedType'] = temp.changeType
temp['changeType'] = defaultStr 
allctdf = allctdf.append(temp)


alldf = allctdf
del allctdf

#############################################



changeTypeIndex = makeIndex(alldf.changeType)
for typeName in uncertainTransformTypes[1::2]:
  ind,changeTypeIndex = appendToIndex(changeTypeIndex,typeName,False)
changeTypeEosInd,changeTypeIndex = appendToIndex(changeTypeIndex,eosStr,True)
changeTypeDefInd = changeTypeIndex.get_loc(defaultStr)
alldf['changeTypeInd'] = findIndexes(changeTypeIndex,alldf.changeType,changeTypeDefInd)




alldf.sort_values(['sentence_id','token_id'],inplace=True)

alldf['tokInd'] = findIndexes(tokIndex,alldf.before.str.lower(),tokDefInd)

addSubTokData(alldf)


tokEosInd = tokIndex.get_loc(eosStr)
changeTypeEosInd = changeTypeIndex.get_loc(eosStr)


alldf['subTokTypes'],alldf['subTokLengths'] = zip(*alldf.subToks.map(makeSubTokAttributes))

subTokLengthIndex = makeIndex(itertools.chain.from_iterable(alldf.subTokLengths))
subTokLengthEosInd,subTokLengthIndex = appendToIndex(subTokLengthIndex,eosStr,True)
alldf['subTokLengths'] = alldf.subTokLengths.map(lambda xlist: [subTokLengthIndex.get_loc(x) for x in xlist])
alldf['subTokLengths'] = padLists(alldf.subTokLengths,SUBTOK_SEQ_LEN,subTokLengthEosInd)

subTokTypeIndex = makeIndex(itertools.chain.from_iterable(alldf.subTokTypes))
subTokTypeEosInd,subTokTypeIndex = appendToIndex(subTokTypeIndex,eosStr,True)
alldf['subTokTypes'] = alldf.subTokTypes.map(lambda xlist: [subTokTypeIndex.get_loc(x) for x in xlist])
alldf['subTokTypes'] = padLists(alldf.subTokTypes,SUBTOK_SEQ_LEN,subTokTypeEosInd)





tardf = alldf[alldf.sentence_id.isin(tar_sid)].copy()
trndf = alldf[alldf.sentence_id.isin(trn_sid)].copy()
del alldf
trnsentdf = makeSentdf(trndf)
tarsentdf = makeSentdf(tardf)

ctiCounts = Counter(trndf.changeTypeInd)

ctiOffset_normal = {k:0.020-0 for k,v in ctiCounts.items()}
ctiOffset_scaled = {k:0.020-min(0.95,(v/len(trndf))) for k,v in ctiCounts.items()}
ctiOffset_alt = {k:0.020-min(0.73,(v/len(trndf))) for k,v in ctiCounts.items()}
ctiOffsets = [ctiOffset_normal,ctiOffset_scaled,ctiOffset_alt,ctiOffset_normal]

ctiOffsets = [{k:0.020-0 for k,v in ctiCounts.items()}
              ,{k:0.020-min(0.95,(v/len(trndf))) for k,v in ctiCounts.items()}
              ,{k:0.020-min(0.85,(v/len(trndf))) for k,v in ctiCounts.items()}
              ,{k:0.020-min(0.80,(v/len(trndf))) for k,v in ctiCounts.items()}
              ,{k:0.020-min(0.75,(v/len(trndf))) for k,v in ctiCounts.items()}
    ]




maxTokSeqLen = int(trnsentdf.tokInd.append(tarsentdf.tokInd).map(len).max())
  




excludedChangeTypes = [defaultStr,eosStr]
excludedChangeTypeInds = [changeTypeIndex.get_loc(e) for e in excludedChangeTypes]

modelNumbers = [0,1]


subTokTypeEosIndList = [subTokTypeEosInd]*SUBTOK_SEQ_LEN
subTokLengthEosIndList = [subTokLengthEosInd]*SUBTOK_SEQ_LEN
for modelNumber in modelNumbers:
  modStr = str(modelNumber)
  print('train model',modStr)
  #### set random seeds
  torch.manual_seed(1234)
  np.random.seed(1234)
  
  dataInds = np.arange(len(trnsentdf))  

  enc = gpu(Encoder(len(tokIndex),ENC_TOK_EMBED_SIZE,maxTokSeqLen
                    ,len(subTokTypeIndex),ENC_SUBTOKTYPE_EMBED_SIZE,len(subTokLengthIndex),ENC_SUBTOKLEN_EMBED_SIZE,SUBTOK_SEQ_LEN
                    ,ENC_HIDDEN_SIZE))
  dec = gpu(Decoder(len(tokIndex),DEC_EMBED_SIZE,maxTokSeqLen
                    ,len(subTokTypeIndex),ENC_SUBTOKTYPE_EMBED_SIZE,len(subTokLengthIndex),ENC_SUBTOKLEN_EMBED_SIZE,SUBTOK_SEQ_LEN                                        
                    ,DEC_HIDDEN_SIZE,len(changeTypeIndex),enc.outputSize))
  
  encOpt = optim.Adam(filter(lambda p: p.requires_grad,enc.parameters()),lr=0.0009)
  decOpt = optim.Adam(filter(lambda p: p.requires_grad,dec.parameters()),lr=0.0009)  
 

  #crit = nn.CrossEntropyLoss(weight=gpu(FT(ctiWeights.weight)))
  crit = nn.CrossEntropyLoss()
  
  ctiOffset = ctiOffsets[modelNumber]
  #tokMaskFuncs[modelNumber](trnsentdf)  
  startTime = time.time()
  for epoch in range(0,10):
    np.random.shuffle(dataInds)
    batchSliceIndsGen = makeStepSliceInds(len(dataInds),BATCH_SIZE)
    
    
    totalLoss = 0
    totalInstances = 0
    batchCountdown = 1000
    updateCounter = 0
    for batchSliceStart,batchSliceEnd in batchSliceIndsGen:
      batchInds = dataInds[batchSliceStart:batchSliceEnd]
      batchdf = trnsentdf.iloc[batchInds]
  
      tokData = gpu(V(LT(np.array(padLists(batchdf.tokInd,maxTokSeqLen,tokEosInd)))))
      subTokTypeData = gpu(V(LT(padLists(batchdf.subTokTypes,maxTokSeqLen,subTokTypeEosIndList))))
      subTokLengthData = gpu(V(LT(padLists(batchdf.subTokLengths,maxTokSeqLen,subTokLengthEosIndList))))
  
      changeTypeData = gpu(V(torch.from_numpy(np.array(padLists(batchdf.changeTypeInd,maxTokSeqLen,changeTypeEosInd)))))
      wordEmbData = gpu(V(torch.from_numpy(np.array(padLists(batchdf.wordEmbInd,maxTokSeqLen,0)))))
      
      if updateCounter==0:
        encOpt.zero_grad()
        decOpt.zero_grad()
      
      enc.initHidden()
      
   
      encoded = enc(tokData,wordEmbData,subTokTypeData,subTokLengthData)
      dec.initHidden()
      batchLoss = 0
      for seq in range(tokData.size()[1]):
        
  
        ctd = changeTypeData[:,seq]
   

        if (ctd.data!=changeTypeEosInd).sum()==0: break
      
        ctdMask = gpu(BT([c not in excludedChangeTypeInds for c in ctd.data]))
       
        if ctdMask.sum()==0: continue 
  
        # subsampling for majority class, by excluding it from loss calc.
        ctirand = np.random.uniform(size=len(ctdMask))
        subsampMask = gpu(BT([1 if (i in ctiOffset and (r+ctiOffset[i])>=0) else 0 for r,i in zip(ctirand,ctd.data.cpu().numpy())]))
        ctdMask = ctdMask & subsampMask
        if ctdMask.sum()==0: continue 
     
        ctdInds = torch.nonzero(ctdMask).squeeze(-1)
        out = dec(tokData[:,seq],wordEmbData[:,seq],encoded,subTokTypeData[:,seq,:],subTokLengthData[:,seq,:])
        batchLoss += crit(out[ctdInds,:],ctd[ctdInds])
  
      totalInstances += len(batchInds)
  
      if type(batchLoss)==torch.autograd.variable.Variable: 
        batchLoss.backward()
        totalLoss += batchLoss.data[0]  
        updateCounter += 1
        if updateCounter>=1:
          updateCounter = 0
          encOpt.step()
          decOpt.step()
  
      del batchLoss 
     
      batchCountdown -= 1
      #print('train batch',batchCount,totalLoss/totalInstances)
      if batchCountdown<=0: break
    epochLoss = totalLoss/totalInstances
    print('epoch',epoch,epochLoss)
  print('training time',time.time()-startTime)  
  saveModelState(enc,'../temp/enc'+modStr)
  saveModelState(dec,'../temp/dec'+modStr)

  





################################

################### predict on target data
addLimitedOutputInds(tardf)
tarsentdf = makeSentLimOutInds(tardf,tarsentdf)

for modelNumber in modelNumbers:
  modStr = str(modelNumber) 
  print('predict model',modStr)  
  enc = gpu(makeSavedModel(Encoder,'../temp/enc'+modStr))
  dec = gpu(makeSavedModel(Decoder,'../temp/dec'+modStr))

  enc.eval()
  dec.eval()

  batchSliceIndsGen = makeStepSliceInds(len(tarsentdf),BATCH_SIZE)
  totalLoss = 0
  totalInstances = 0
  outputs = []
  for batchSliceStart,batchSliceEnd in batchSliceIndsGen:
    batchInds = np.arange(batchSliceStart,batchSliceEnd)
    batchdf = tarsentdf.iloc[batchInds]
    tokInd = padWithLast(list(batchdf.tokInd),BATCH_SIZE)
    subTokTypes = padWithLast(list(batchdf.subTokTypes),BATCH_SIZE)
    subTokLengths = padWithLast(list(batchdf.subTokLengths),BATCH_SIZE)  
    changeTypeInd = padWithLast(list(batchdf.changeTypeInd),BATCH_SIZE)
    limOutInds = padWithLast(list(batchdf.limOutInds),BATCH_SIZE)
    wordEmbInd = padWithLast(list(batchdf.wordEmbInd),BATCH_SIZE)
    
    tokData = gpu(V(LT(np.array(padLists(tokInd,maxTokSeqLen,tokEosInd)))))
    subTokTypeData = gpu(V(LT(padLists(subTokTypes,maxTokSeqLen,subTokTypeEosIndList))))
    subTokLengthData = gpu(V(LT(padLists(subTokLengths,maxTokSeqLen,subTokLengthEosIndList))))
  
    changeTypeData = gpu(V(LT(np.array(padLists(changeTypeInd,maxTokSeqLen,changeTypeEosInd)))))  
    limOutData = np.array(padLists(limOutInds,maxTokSeqLen,[]))
    wordEmbData = gpu(V(torch.from_numpy(np.array(padLists(wordEmbInd,maxTokSeqLen,0)))))
  
    enc.initHidden()
    
    encoded = enc(tokData,wordEmbData,subTokTypeData,subTokLengthData)
    dec.initHidden()
    batchLoss = 0
    outInds = []
    for seq in range(tokData.size()[1]):
      tokd = tokData[:,seq]
      ctd = changeTypeData[:,seq]
      loi = limOutData[:,seq]
      if (tokd==tokEosInd).data.all():
        cteosindarray = np.empty((len(tokd),1),dtype=int)
        cteosindarray.fill(changeTypeEosInd)
        outInds.append(cteosindarray)
      else:        
        out = dec(tokData[:,seq],wordEmbData[:,seq],encoded,subTokTypeData[:,seq,:],subTokLengthData[:,seq,:])        
        outprobs = out.data.cpu().numpy()
        for i in range(len(outprobs)):
          outprobs[i,loi[i]] += 1e6  
        out.data = gpu(FT(outprobs))
        outInds.append(out.data.topk(1)[1].cpu().numpy())
          
      ctdMask = gpu(BT([c not in excludedChangeTypeInds for c in ctd.data]))
      if ctdMask.sum()==0: continue
      ctdInds = torch.nonzero(ctdMask).squeeze(-1)    
      batchLoss += crit(out[ctdInds,:],ctd[ctdInds]).data[0]       
  
  
    outInds = list(np.hstack(outInds))
    outputs.append(pd.Series(outInds[:len(batchdf)],index=batchdf.index))
    totalInstances += len(batchInds)
    
    totalLoss += batchLoss
    
  
    
  lossPerInstance = totalLoss/totalInstances
  print('lossPerInstance',lossPerInstance)
  
  tarsentout = tarsentdf.copy()
  tarsentout['out'] = pd.concat(outputs)
  tarsentout.reset_index(inplace=True)
  
  sidList = []
  tidList = []
  oList = []
  ctiList = []
  for sid,o,cti in zip(tarsentout.sentence_id,tarsentout.out,tarsentout.changeTypeInd):  
    l = len(cti)
    sidList += [sid]*l
    tidList += list(range(l))
    oList += list(o)[:l]
    ctiList += cti

  
  
  outdf = pd.DataFrame({'sentence_id':sidList,'token_id':tidList,'out':oList,'changeTypeInd':ctiList})#,'tokMask':tmList})
  outdf = tardf.drop('changeTypeInd',axis=1).merge(outdf,how='right',on=['sentence_id','token_id'])
  outdf['actualType'] = changeTypeIndex[outdf.changeTypeInd]
  outdf['outType'] = changeTypeIndex[outdf.out]
  outdf['correctType'] = outdf.out==outdf.changeTypeInd

  outdf.to_pickle('../temp/outdf'+modStr+'.pickle')
  
##################### outputs  
for modelNumber in modelNumbers:
  modStr = str(modelNumber) 
  print('make output, model',modStr)
  outdf = pd.read_pickle('../temp/outdf'+modStr+'.pickle')  

  if modelNumber==1:
    print('doing replacements from other outputs')
    outdfOther = pd.read_pickle('../temp/outdf0.pickle')
    outdfOtherMask = outdfOther.outType=='replaceSame'
    #outdfOtherMask = (outdfOther.outType=='replaceSame') | (outdfOther.outType=='letters')
    outdf.loc[outdfOtherMask,'outType'] = outdfOther.loc[outdfOtherMask,'outType']


  #########################
  
  outdf['changeType'] = outdf['outType']
  
  outcdf,outdf = transformCertain1a(outdf,None,forceType=True,check=False)
  outcdf,outdf = replaceUni(outdf,outcdf,exsingleRep,'exsingleRep',forceType=True,check=False)
  outcdf,outdf = replaceUni(outdf,outcdf,exsameRep,'exsameRep',forceType=True,check=False)
  outcdf,outdf = transformCertain1b(outdf,outcdf,forceType=True,check=False)
  outcdf,outdf = transformCertain2(outdf,outcdf,forceType=True,check=False)
  outcdf,outdf = transformUncertain(outdf,outcdf,forceType=False,check=False)
  outcdf,outdf = replaceMulti(outdf,outcdf,multiRepl,forceType=False)
  
  outcdf,outdf = replaceUni(outdf,outcdf,exanyRep,'exanyRep',forceType=True,check=False)
  
  outcdf,outdf = transformRemainingUnknown(outdf,outcdf,forceType=True)
  outdf = outcdf
  del outcdf
  outdf = outdf.sort_values(['sentence_id','token_id'])
  outdf['limOutInds']=[','.join(changeTypeIndex[i] for i in loi) for loi in outdf.limOutInds]

  outdf.to_pickle('../temp/finaldf'+modStr+'.pickle')
  
  if not usingTestSet:
    outdf['corrAfter'] = outdf.after==outdf.changed
    print('corrAfter score',outdf.corrAfter.sum(),len(outdf),'ratio',outdf.corrAfter.sum()/len(outdf))
    
    g = outdf.groupby('assignedType')
    temp = g.changeTypeInd.apply(len).rename('total').to_frame()
    temp['corrAfter'] = g.corrAfter.apply(sum)
    temp['incAfter'] = temp.total-temp.corrAfter  
    temp['corrRatio'] = temp.corrAfter/temp.total
    scores = temp.sort_index()
    print(scores)
    print('total corrAfter',scores.corrAfter.sum(),'/',scores.total.sum(),'ratio',scores.corrAfter.sum()/scores.total.sum()) 
      
    g = outdf.groupby('sentence_id')
    temp = g.corrAfter.all()
    print('sentence score',temp.sum(),len(temp),'ratio',temp.sum()/len(temp))
    
    dferr = outdf[(outdf.assignedType=='replaceMulti1') & (outdf.corrAfter==False)]
    dfcor = outdf[(outdf.assignedType=='replaceMulti1') & (outdf.corrAfter==True)]
    temp = outdf.copy()
    temp.index = temp.sentence_id
    temperr = temp.loc[dferr.sentence_id.drop_duplicates()]
    tempcor = temp.loc[dfcor.sentence_id.drop_duplicates()]  
  
    letdf=outdf[outdf.changeType=='replaceMulti1'].copy()
    samelc = pd.Index(x.lower() for x in exsameRep.firstDict)
    letdf['inlower'] = [x.lower() in samelc for x in letdf.before]

  if usingTestSet:
    g = outdf.groupby('changeType')
    temp = g.changeType.apply(len).rename('total').to_frame()
    distribution = temp.sort_index()
    print(distribution)
    
    dfa = outdf[(outdf.changeType=='unknown')]
    dfa.to_pickle('../temp/dfa'+modStr+'.pickle')

  if usingTestSet and modelNumber==1:    
    subdf = pd.DataFrame(index=outdf.index)
    subdf['id'] = (outdf.sentence_id-targ_sentid_offset).astype(str) + '_' + outdf.token_id.astype(str)
    subdf['after'] = outdf.changed
    subdf.to_csv('../temp/submission.csv.gz',index=False,compression='gzip')


 


 
  
