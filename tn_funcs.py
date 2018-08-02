import pandas as pd
import numpy as np
import re
import torch
import pickle
from myutils import pmapSeries,pstarmap


def addSubTokData(df):
  splitPat = r'(\W+|\s+)'
  df['subToks'] = df.before.map(lambda x: re.split(splitPat,x))
  df['subToks'] = [[w for w in sb if w!=''] for sb in df.subToks]




def removeLongSentences(thedf,limit):
  sids = thedf.sentence_id[thedf.token_id>limit]
  sids = sids.drop_duplicates()
  mask = ~thedf.sentence_id.isin(sids)
  thedf = thedf.loc[mask]
  return thedf

def getAttribute(word,pattern):
  m = pattern.match(word)
  if m is None:
    return 0
  else:
    return m.lastindex
  

def makeSentdf(df):
  g = df.sort_values(['sentence_id','token_id']).groupby('sentence_id',sort=False)  
  tempdf = g.tokInd.apply(list).to_frame()
  tempdf['subTokTypes'] = g.subTokTypes.apply(list)
  tempdf['subTokLengths'] = g.subTokLengths.apply(list)  
  tempdf['changeTypeInd'] = g.changeTypeInd.apply(list)
  tempdf['wordEmbInd'] = g.wordEmbInd.apply(list)  
  tempdf['before'] = g.before.apply(list)    
  tempdf.reset_index(drop=False,inplace=True)
  return tempdf

def makeSentLimOutInds(tokdf,sentdf):
  g = tokdf.sort_values(['sentence_id','token_id']).groupby('sentence_id',sort=False)  
  tempdf = g.limOutInds.apply(list)
  sentdf = sentdf.sort_values('sentence_id')
  sentdf['limOutInds'] = tempdf.values
  return sentdf

def saveModelState(model,pathName):
  with open(pathName+'_initvals','wb') as f:
    pickle.dump(model.initVals,f)
  torch.save(model.state_dict(),pathName+'_statedict')
  
def loadModelState(pathName):
  with open(pathName+'_initvals','rb') as f:
    initVals = pickle.load(f)
  state = torch.load(pathName+'_statedict')
  return initVals,state
  
def makeSavedModel(cls,pathName):
  initVals,state = loadModelState(pathName)
  model = cls(*initVals)
  model.load_state_dict(state)
  return model

def asYear(thestr):
  return len(thestr)==4 and not thestr.startswith('200') and 1900<=int(thestr)<=2100  

def makeReplaceDict(thedf,repdf):
  if repdf is None:
    udf = thedf[['before','after']].copy()
    udf['occurrences'] = 1  
    g = udf.groupby(['before','after'],as_index=False)
    repdf = g.occurrences.sum()
#  else: 
#    repdf = repdf[repdf.before.isin(thedf.before.drop_duplicates())]
  repdf = repdf.sort_values('occurrences',ascending=False)

  g = repdf.groupby('before')  
  temp = g.after.agg(['first']).reset_index(drop=False).rename(columns={'first':'fPoss'})
  replacementDict = {b:fp for b,fp in zip(temp.before,temp.fPoss)}  
  return replacementDict


def makeReplaceSameDict(thedf,repdf):
  if repdf is None:
    udf = thedf[['before','after']].copy()
    udf['occurrences'] = 1  
    g = udf.groupby(['before','after'],as_index=False)
    repdf = g.occurrences.sum()
#  else:
#    repdf = repdf[repdf.before.isin(thedf.before.drop_duplicates())] 

  g = repdf.groupby('before')  
  temp = g.after.agg(['size','first']).reset_index(drop=False).rename(columns={'size':'nPoss','first':'fPoss'})
  sameReplacement = {b:fp for b,np,fp in zip(temp.before,temp.nPoss,temp.fPoss) if np==1 and b==fp}  
  return sameReplacement



def makeReplaceSingleDict(thedf,repdf):
  if repdf is None:
    udf = thedf[['before','after']].copy()
    udf['occurrences'] = 1  
    g = udf.groupby(['before','after'],as_index=False)
    repdf = g.occurrences.sum()
#  else:   
#    repdf = repdf[repdf.before.isin(thedf.before.drop_duplicates())]

  g = repdf.groupby('before')  
  temp = g.after.agg(['size','first']).reset_index(drop=False).rename(columns={'size':'nPoss','first':'fPoss'})
  singleReplacement = {b:fp for b,np,fp in zip(temp.before,temp.nPoss,temp.fPoss) if np==1 and b!=fp}  
  return singleReplacement

def makeReplaceMultiDict(thedf,repdf):
  if repdf is None:
    udf = thedf[['before','after']].copy()
    udf['occurrences'] = 1  
    g = udf.groupby(['before','after'],as_index=False)
    repdf = g.occurrences.sum()
#  else: 
#    repdf = repdf[repdf.before.isin(thedf.before.drop_duplicates())]
  repdf = repdf.sort_values('occurrences',ascending=False)
  
  g = repdf.groupby('before')  
  temp = g.after.apply(lambda x: list(x))
  multiReplacement = {}
  for token,replacements in temp.items():
    if len(replacements)>1:
      multiReplacement[token] = replacements[0:2] #latest
  return multiReplacement


def getChangeTransformed2(thedf,cdf,transformFunc,changeTypeStr,colNames,forceType,check):
  if not forceType:
    changedMask = thedf.changeType==changeTypeStr
    changed = thedf.loc[changedMask].copy()
    cols = [changed[cn] for cn in colNames]
    changed['changed'] = pstarmap(transformFunc,zip(*cols),chunksize=100000)
  else:
    cols = [thedf[cn] for cn in colNames]    
    transformed = pstarmap(transformFunc,zip(*cols),chunksize=100000)
    transformed = pd.Series(transformed,index=thedf.index)
    changedMask = (transformed!=thedf.before)
    if check: changedMask = changedMask & (transformed==thedf.after)
    changed = thedf.loc[changedMask].copy()
    changed['changed'] = transformed[changedMask]
    changed['changeType'] = changeTypeStr
  
  if cdf is None: cdf = changed
  else: cdf = pd.concat([cdf,changed],axis=0)
  thedf = thedf.loc[~changedMask]  
  return cdf,thedf

def getChangeTransformed(thedf,cdf,transformFunc,changeTypeStr,forceType,check):
  if not forceType:
    changedMask = thedf.changeType==changeTypeStr
    changed = thedf.loc[changedMask].copy()  
    changed['changed'] = pmapSeries(transformFunc,changed.before,chunksize=100000)
  else:
    #transformed = thedf.before.map(transformFunc)
    transformed = pmapSeries(transformFunc,thedf.before,chunksize=100000)
    
    changedMask = (transformed!=thedf.before)
    if check: changedMask = changedMask & (transformed==thedf.after)
    changed = thedf.loc[changedMask].copy()
    changed['changed'] = transformed[changedMask]
    changed['changeType'] = changeTypeStr
  
  if cdf is None: cdf = changed
  else: cdf = pd.concat([cdf,changed],axis=0)
  thedf = thedf.loc[~changedMask]  
  return cdf,thedf

def transformRemainingUnknown(thedf,cdf,forceType):
  changeTypeStr = 'unknown'   
  if not forceType:
    mdf = thedf.copy()
    mdf['changed'] = mdf.before 
  else:   
    mdf = thedf.copy()
    mdf['changed'] = mdf.before
    mdf['changeType'] = changeTypeStr
  if cdf is None: cdf = mdf
  else: cdf = pd.concat([cdf,mdf],axis=0)
  thedf = None  
  return cdf,thedf

def replaceSame(thedf,cdf,sameReplDict,forceType,check):
  changeTypeStr = 'replaceSame'     
  if not forceType:
    mask = thedf.changeType==changeTypeStr
    mdf = thedf.loc[mask].copy()
    mdf['changed'] = mdf.before.map(lambda x: sameReplDict.lookup(x) if sameReplDict.has(x) else x)    
  else:   
    mask = thedf.before.map(lambda x: sameReplDict.has(x))
    transformed = thedf.before.map(lambda x: sameReplDict.lookup(x) if sameReplDict.has(x) else None)
    if check: mask = mask & (transformed==thedf.after)
    mdf = thedf.loc[mask].copy()
    mdf['changed'] = transformed.loc[mask]
    mdf['changeType'] = changeTypeStr
  if cdf is None: cdf = mdf
  else: cdf = pd.concat([cdf,mdf],axis=0)
  thedf = thedf.loc[~mask]  
  return cdf,thedf



def replaceSingle(thedf,cdf,singleTransDict,forceType,check):
  changeTypeStr = 'replaceSingle'  
  if not forceType:
    mask = thedf.changeType==changeTypeStr
    mdf = thedf.loc[mask].copy()
    mdf['changed'] = mdf.before.map(lambda x: singleTransDict.lookup(x) if singleTransDict.has(x) else x)    
  else:  
    mask = thedf.before.map(lambda x: singleTransDict.has(x))
    transformed = thedf.before.map(lambda x: singleTransDict.lookup(x) if singleTransDict.has(x) else None)
    if check: mask = mask & (transformed==thedf.after)
    mdf = thedf.loc[mask].copy()
    mdf['changed'] = transformed.loc[mask]
    mdf['changeType'] = changeTypeStr
  if cdf is None: cdf = mdf
  else: cdf = pd.concat([cdf,mdf],axis=0)
  thedf = thedf.loc[~mask]  
  return cdf,thedf 



def replaceUni(thedf,cdf,uniTransDict,changeTypeStr,forceType,check):
  if not forceType:
    mask = thedf.changeType==changeTypeStr
    mdf = thedf.loc[mask].copy()
    mdf['changed'] = mdf.before.map(lambda x: uniTransDict.lookup(x) if uniTransDict.has(x) else x)    
  else:  
    mask = thedf.before.map(lambda x: uniTransDict.has(x))
    mask = mask & ~thedf.before.str.contains(r'^[\d ,.]+$')
    transformed = thedf.before.map(lambda x: uniTransDict.lookup(x) if uniTransDict.has(x) else None)
    if check: mask = mask & (transformed==thedf.after)    
    mdf = thedf.loc[mask].copy()
    mdf['changed'] = transformed.loc[mask]
    mdf['changeType'] = changeTypeStr
  if cdf is None: cdf = mdf
  else: cdf = pd.concat([cdf,mdf],axis=0)
  thedf = thedf.loc[~mask]  
  return cdf,thedf 



def replaceMulti(thedf,cdf,multiTransDict,forceType):
  changeTypeStr = 'replaceMulti'
  if not forceType:
    mask = thedf.changeType.str.startswith(changeTypeStr)
    mdf = thedf.loc[mask].copy()
    repInd = mdf.changeType.str.replace(changeTypeStr,'').astype(int)
    mdf['changed'] = [multiTransDict[b][ri] if b in multiTransDict and ri<len(multiTransDict[b]) else b for b,ri in zip(mdf.before,repInd)]
  else:
    mask = [b in multiTransDict and a in multiTransDict[b] for b,a in zip(thedf.before,thedf.after)]
    #mask = thedf.before.map(lambda x: x in multiTransDict)
    mdf = thedf.loc[mask].copy()
  
    mdf['changeType'] = mdf[['before','after']].apply(lambda x: multiTransDict[x[0]].index(x[1]),axis=1)
    mdf['changed'] = mdf[['before','changeType']].apply(lambda x: multiTransDict[x[0]][x[1]],axis=1)
    replacementIndexDict = {ind:changeTypeStr+str(ind) for ind in mdf.changeType.unique()}
    mdf['changeType'] = mdf.changeType.map(lambda x: replacementIndexDict[x])
  if cdf is None: cdf = mdf
  else: cdf = pd.concat([cdf,mdf],axis=0)  
  thedf = thedf.loc[np.logical_not(mask)]
  return cdf,thedf  













