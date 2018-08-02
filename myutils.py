#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 07:12:47 2017

@author: robert
"""

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from scipy import sparse
import pandas as pd
import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed
import itertools
from collections import Counter,OrderedDict
from gensim.models import KeyedVectors


def makeRandomizedFoldIndexes(total,nfolds,randomSeed):
  inds = np.arange(total)
  np.random.seed(randomSeed)
  np.random.shuffle(inds)
  foldInds = np.array_split(inds,nfolds)
  return foldInds

def makeStratifiedFoldIndexes(series,nfolds,randomSeed):
  df = series.rename('class').to_frame()
  df['inds'] = np.arange(len(df))
  pinds = np.arange(len(df))  
  np.random.seed(randomSeed)
  np.random.shuffle(pinds)
  df = df.iloc[pinds]  
  g = df.groupby('class')
  foldInds = [np.array([])]*nfolds
  for i,gdf in g:
    gFoldInds = np.array_split(gdf['inds'].values,nfolds)
    for i,gfi in enumerate(gFoldInds):
      foldInds[i] = np.concatenate((foldInds[i],gfi),axis=0)
      
  return foldInds
    

  

def trainValFoldInds(foldInds,foldNum):
  inds = foldInds.copy()
  valInds = inds.pop(foldNum)
  trnInds = np.hstack(inds)
  return trnInds,valInds

def validationPosInds(thedf,valFoldNum):
  trnpos = np.flatnonzero(thedf['valid']!=valFoldNum)
  valpos = np.flatnonzero(thedf['valid']==valFoldNum)
  return trnpos,valpos

def columnToOneHot(thecolumn):
  le = LabelEncoder()
  ohe = OneHotEncoder()
  encoded = le.fit_transform(thecolumn)
  oneHot = ohe.fit_transform(encoded.reshape(-1,1))
  return oneHot

def columnsToOneHot(thedf):
  with Parallel(n_jobs=mp.cpu_count()) as par:
    res = par(delayed(columnToOneHot)(thedf[cname]) for cname in thedf)
  return res

def dfToOneHot(thedf):
  colnames = []
  res = columnsToOneHot(thedf)
  for cname,spar in zip(thedf.keys(),res):
    colnames = colnames + [cname+'_'+str(i) for i in range(spar.shape[1])]
  sparseArray = sparse.hstack(res)
  return sparseArray,colnames

#def dfToOneHot(thedf):
#  colnames = []
#  sparseArray = None
#  for cname in thedf:
#    oh = columnToOneHot(thedf[cname])
#    if sparseArray is None: sparseArray = oh
#    else: sparseArray = sparse.hstack((sparseArray,oh))
#    colnames = colnames + [cname+'_'+str(i) for i in range(oh.shape[1])]
#  #sparseArray = sparseArray.tocsr()
#  return sparseArray,colnames

def dfCatToNum(thedf):
  newdf = pd.DataFrame(index=thedf.index)
  le = LabelEncoder()
  for cname in thedf:
    newdf[cname] = le.fit_transform(thedf[cname].values)
  return newdf
  

def convertToSparse_ohe(allfeats):
  oneHotTypes = ['object','category']
  #numeric and non numeric types
  allDataNonNum = allfeats.select_dtypes(include=oneHotTypes)
  allDataNum = allfeats.select_dtypes(exclude=oneHotTypes)
  ######### convert categorical to onehot and concatenate with numeric
  allsparse,sparseNames = dfToOneHot(allDataNonNum)
  allsparse = sparse.hstack((allsparse,allDataNum.values))
  allsparse = allsparse.tocsr()
  sparseNames = sparseNames + allDataNum.columns.tolist()
  return allsparse,sparseNames

def convertToSparse_numeric(allfeats):
  oneHotTypes = ['object','category']  
  #numeric and non numeric types
  allDataNonNum = allfeats.select_dtypes(include=oneHotTypes)
  allDataNum = allfeats.select_dtypes(exclude=oneHotTypes)
  ######### convert categorical to numeric and concatenate with numeric
  allDataNonNum = dfCatToNum(allDataNonNum)

  allfeats = pd.concat([allDataNonNum,allDataNum],axis=1)
  allsparse = sparse.csr_matrix(allfeats.values)
  sparseNames = allfeats.columns.tolist()
  return allsparse,sparseNames

def reduce_dtypes_32(df):
  for cname in df:
    if df[cname].dtype=='int64':
      df[cname] = df[cname].astype('int32')
    if df[cname].dtype=='float64':
      df[cname] = df[cname].astype('float32')
  return df      
      
def checkBad(df):
  print('Checking for nan and inf')
  for cname in df:
    #print('checking column',cname)
    if df[cname].isnull().any():
      print(cname,'nan ***************')
  for cname in df.select_dtypes(include=[np.number]):
    #print('checking column',cname)
    if np.isinf(df[cname].values).any():
      print(cname,'inf ===============')       

#def paramGen(paramLists):
#  eta = [0.1]
#  max_depth = [8,10,12]
#  subsample = [0.7]
#  colsample_bytree = [0.7]
#  
#  perms = itertools.product(eta,max_depth,subsample,colsample_bytree)
#  for p in perms:
#    d = dict(zip(['eta','max_depth','subsample','colsample_bytree'],p))
#    d['objective'] = 'binary:logistic'
#    d['eval_metric'] = 'auc'
#    yield d
    
def paramGen(paramListsDict):
  vals = [v if type(v)==list else [v] for v in paramListsDict.values()]
  keys = paramListsDict.keys()
  
  perms = itertools.product(*vals)
  for p in perms:
    d = dict(zip(keys,p))
    yield d
    
def makeSplitPosIndexes(series,nSections):
  sections = np.array_split(series.unique(),nSections)
  indsList = []
  for s in sections:
    inds = np.where(series.isin(s))
    indsList.append(inds)
  return indsList
    
def doSplitPar(thedf,seriesForSplit,thefunc):
  secinds = makeSplitPosIndexes(seriesForSplit,mp.cpu_count())    
  with Parallel(n_jobs=mp.cpu_count()) as par:
    res = par(delayed(thefunc)(thedf.iloc[si]) for si in secinds)
  res = pd.concat(res,axis=0)
  return res

def makeStepSliceInds(total,step):
  for start in range(0,total,step):
    yield (start,min(start+step,total))

def makeFoldSliceInds(total,nFolds):
  return makeStepSliceInds(total,int(total/nFolds))

def makeFractionSliceInds(total,fraction):
  return makeStepSliceInds(total,int(total*fraction))
    
def makeIndsFromSliceInds(total,sliceInds,excludeSlice=False):
  sliceStart,sliceEnd = sliceInds
  a = np.arange(total)
  if excludeSlice:
    ret = np.concatenate((a[:sliceStart],a[sliceEnd:]))
  else:
    ret = a[sliceStart:sliceEnd]
  return ret

def sliceSeries(series,sliceInds):
  frac = series.iloc[makeIndsFromSliceInds(len(series),sliceInds,excludeSlice=False)]
  rem = series.iloc[makeIndsFromSliceInds(len(series),sliceInds,excludeSlice=True)]
  return rem,frac
  
def splitSeriesByFraction(series,fraction,shuffle):
  if shuffle:
    series = series.copy()
    np.random.shuffle(series)
  sliceInds = next(makeFractionSliceInds(len(series),fraction))
  rem,frac = sliceSeries(series,sliceInds)
  return rem,frac

def trainValSplit(xcols,ycol,valFraction):
  isdf = isinstance(xcols,pd.DataFrame)
  isseries = isinstance(ycol,pd.Series)
  
  total = len(xcols)
  step = int(total*valFraction)
  sliceInds = next(makeStepSliceInds(total,step))
  inds = np.arange(total)
  np.random.shuffle(inds)
  traininds = inds[makeIndsFromSliceInds(total,sliceInds,excludeSlice=True)]
  valinds = inds[makeIndsFromSliceInds(total,sliceInds,excludeSlice=False)]
  if isdf: xtrain,xval = xcols.iloc[traininds],xcols.iloc[valinds]
  else: xtrain,xval = xcols[traininds],xcols[valinds]
  if isseries: ytrain,yval = ycol.iloc[traininds],ycol.iloc[valinds]
  else: ytrain,yval = ycol[traininds],ycol[valinds]
  
  return (xtrain,xval,ytrain,yval)

def mapSeqDict(inseq,mapDict):
  outlist = [mapDict[i] for i in inseq]
  return outlist

def mapSeqsDict(inseqs,mapDict):
  outlists = [mapSeqDict(i,mapDict) for i in inseqs]
  return outlists

def mapSeqInd(inseq,mapInd):
  outlist = [mapInd.get_loc(i) for i in inseq]
  return outlist

def mapSeqsInd(inseqs,mapInd):
  outlists = [mapSeqInd(inseq,mapInd) for inseq in inseqs]
  return outlists

#def mapSeqsInd(inseqs,mapInd):
#  with Parallel(n_jobs=mp.cpu_count()) as par:
#    outlists = par(delayed(mapSeqInd)(inseq,mapInd) for inseq in inseqs)
#  return outlists
  
def mapSeqInd1(inseq,mapInd,defaultVal):
  outlist = [mapInd.get_loc(i) if i in mapInd else defaultVal for i in inseq]
  return outlist

def mapSeqsInd1(inseqs,mapInd,defaultVal):
  outlists = [mapSeqInd1(inseq,mapInd,defaultVal) for inseq in inseqs]
  return outlists

def padList(thelist,padToLength,padWith):
  amountToPad = padToLength-len(thelist)
  if amountToPad>=1: return thelist + [padWith]*amountToPad
  elif amountToPad<0: return thelist[:padToLength]
  return thelist

def padLists(lists,padToLength,padWith):
  padded = [padList(l,padToLength,padWith) for l in lists]
  return padded

def padSeq(theseq,padToLength,padWith):
  amountToPad = padToLength-len(theseq)
  if amountToPad>=1:
    if isinstance(theseq,list):
      return theseq + [padWith]*amountToPad
    elif isinstance(theseq,np.ndarray):
      return np.concatenate([theseq,np.repeat(padWith,amountToPad)])
    else:
      raise Exception('Invalid object type for padding sequence',type(theseq))
  elif amountToPad<0:
    return theseq[:padToLength]
  else: return theseq

def padSeqs(seqs,padToLength,padWith):
  padded = [padSeq(s,padToLength,padWith) for s in seqs]
  return padded
  
def padWithLast(thelist,padToLength):
  amount = padToLength-len(thelist)
  if amount>0:
    thelist = thelist + (thelist[-1:]*amount)
  return thelist

def prependLists(lists,toPrepend):
  outlists = [[toPrepend]+l for l in lists]
  return outlists

#def getGloveEmbeddings(filepath):
#  gloveWordToEmbeddings = {}
#  with open(filepath) as f:
#    for line in f:
#      splitline = line.split()
#      word = splitline[0]
#      gloveEmbeddings = np.array(splitline[1:],dtype='float32')
#      gloveWordToEmbeddings[word] = gloveEmbeddings    
#  gloveEmbeddingsLength = len(gloveEmbeddings)
#  return gloveWordToEmbeddings
#
#def getVocabEmbeddings(words,embeddingsDict,embeddingSize):
#  embeds = []
#  found = 0
#  for i in range(len(words)):
#    word = words[i]
#    if word in embeddingsDict:
#      embeds.append(embeddingsDict[word])
#      found += 1
#    else:
#      embeds.append(np.random.normal(scale=0.6, size=(embeddingSize,))) 
#  embeds = np.array(embeds)
#  return embeds,found

def makeEmbeddingsDict_glove(theseries,filepath):
  gloveWordToEmbeddings = {}
  with open(filepath) as f:
    for line in f:
      splitline = line.split()
      word = splitline[0]
      gloveEmbeddings = np.array(splitline[1:],dtype='float32')
      gloveWordToEmbeddings[word] = gloveEmbeddings
      
  words = theseries.unique()  
  eDict = {}
  for w in words:
    if w in gloveWordToEmbeddings:
      eDict[w] = gloveWordToEmbeddings[w]
    else:
      wl = w.lower()
      if wl in gloveWordToEmbeddings:
        eDict[wl] = gloveWordToEmbeddings[wl]
  return eDict
      
def makeEmbeddingsDict_w2v(theseries,filepath):
  #https://code.google.com/archive/p/word2vec/
  w2vWordVectors = KeyedVectors.load_word2vec_format(filepath,binary=True)
  
  words = theseries.unique()  
  eDict = {}
  for w in words:
    if w in w2vWordVectors.vocab:
      eDict[w] = w2vWordVectors.word_vec(w)
    else:
      wl = w.lower()
      if wl in w2vWordVectors.vocab:
        eDict[wl] = w2vWordVectors.word_vec(wl)
  return eDict

def makeEmbeddingsMatrix(eDict):
  indDict = {}
  embeddingsMatrix = None
  for i,w in enumerate(eDict,1):
    if embeddingsMatrix is None:
      embeddingsMatrix = np.zeros((len(eDict)+1,len(eDict[w])))
    embeddingsMatrix[i] = eDict[w]
    indDict[w] = i  
  return embeddingsMatrix,indDict

def makeIndex(tokens,minCount=1):
  c = Counter(tokens)
  t = [token for token,count in c.items() if count>=minCount]
  theIndex = pd.Index(t)
  theIndex = theIndex.sort_values()
  return theIndex

def appendToIndex(theIndex,theitem,newOnly):
  if theitem in theIndex:
    if newOnly:
      raise Exception('Item already in vocab',theitem)
  else:
    theIndex = theIndex.append(pd.Index([theitem]))
  loc = theIndex.get_loc(theitem)    
  return loc,theIndex

def findIndexes(theIndex,theTokens,defaultInd):
  out = theTokens.map(lambda x: theIndex.get_loc(x) if x in theIndex else defaultInd)
  return out

def pmap(func,data,chunksize):
  with mp.Pool(processes=mp.cpu_count()) as pool:
    res = pool.map(func,data,chunksize=chunksize)
  return res

def pmapSeries(func,data,chunksize):
  with mp.Pool(processes=mp.cpu_count()) as pool:
    res = pool.map(func,data,chunksize=chunksize)
  res = pd.Series(res,index=data.index)
  return res

def pstarmap(func,data,chunksize):
  with mp.Pool(processes=mp.cpu_count()) as pool:
    res = pool.starmap(func,data,chunksize=chunksize)
  return res
    