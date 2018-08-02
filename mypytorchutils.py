import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable as V
from torch import FloatTensor as FT
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from myimageutils import calcPadding,calcOutputSize,splitPadding

def calcCoverageOneDim(inSize,kernel,stride):
  missed = (inSize-kernel)%stride
  return missed

def calcCoverage(inputSize,kernel,stride):
  inputSizeRows,inputSizeCols = inputSize
  kernelRows,kernelCols = kernel
  strideRows,strideCols = stride
  
  rowsMissed = calcCoverageOneDim(inputSizeRows,kernelRows,strideRows)
  colsMissed = calcCoverageOneDim(inputSizeCols,kernelCols,strideCols)
  return (rowsMissed,colsMissed)  


  
#def padSame(conv):
#  rowSize,colSize = conv.inputSize
#  kernelRows,kernelCols = conv.kernel_size
#  strideRows,strideCols = conv.stride
#  vert = calcPadding(rowSize,rowSize,kernelRows,strideRows) 
#  horiz = calcPadding(colSize,colSize,kernelCols,strideCols)
#  padder = nn.ZeroPad2d((0,horiz,0,vert))
##  padder = nn.ReflectionPad2d((0,horiz,0,vert))
#  return padder
#
#  
#
#
#def padCover(conv):
#  rowSize,colSize = conv.inputSize
#  outrows,outcols = coverOutputSize(conv)
#  
#  kernelRows,kernelCols = conv.kernel_size
#  strideRows,strideCols = conv.stride
#  
#  vert = calcPadding(rowSize,outrows,kernelRows,strideRows) 
#  horiz = calcPadding(colSize,outcols,kernelCols,strideCols)
#  padder = nn.ZeroPad2d((0,horiz,0,vert))
##  padder = nn.ReflectionPad2d((0,horiz,0,vert))
#  return padder



class conv2d(nn.Conv2d):
  def __init__(self,inputObj,in_channels,out_channels,kernel_size,stride,activation,padMode):
    super(conv2d,self).__init__(in_channels,out_channels,kernel_size,stride,0)
    self.inputObj = inputObj
    self.activation = activation
    self.padMode = padMode
    
    self.makeInputSize()    
    self.makeOutputSize()

    self.makePadder()
    missed = calcCoverage(self.inputSize,self.kernel_size,self.stride)
    print(self)
    print(self.padMode,'in',self.inputSize,'out',self.outputSize,'missed',missed)
    
  def forward(self,x):
    if self.padder is not None:
      x = self.padder(x)
    
    x = super(conv2d,self).forward(x)
    if self.activation is not None:
      x = self.activation(x)
    #print('testing',x.size()[2:]==self.outputSize,x.size(),self.outputSize)
    return x

  def makeInputSize(self):
    if type(self.inputObj)==tuple:
      self.inputSize = self.inputObj
      #self.receptiveField = self.kernel_size
    else:
      self.inputSize = self.inputObj.outputSize
      #self.receptiveField = tuple(np.array(self.inputObj.receptiveField)*np.array(self.kernel_size))

  def makeOutputSize(self):
    self.outputSize,self.oneSide = calcOutputSize(self.padMode,self.inputSize,self.kernel_size,self.stride)

    

  def makePadder(self):
    rowSize,colSize = self.inputSize
    outrows,outcols = self.outputSize
    
    kernelRows,kernelCols = self.kernel_size
    strideRows,strideCols = self.stride
    
    vert = calcPadding(rowSize,outrows,kernelRows,strideRows) 
    horiz = calcPadding(colSize,outcols,kernelCols,strideCols)
    if self.oneSide:
      padder = nn.ZeroPad2d((0,horiz,0,vert))
    else:
      hSplit1,hSplit2 = splitPadding(horiz)
      vSplit1,vSplit2 = splitPadding(vert)      
      padder = nn.ZeroPad2d((hSplit1,hSplit2,vSplit1,vSplit2)) 
    self.padder = padder
    
#class upsample(nn.Upsample):
#  def __init__(self,inputObj,size=None,scale_factor=None,mode='nearest'):
#    super(upsample,self).__init__(size,scale_factor,mode)
#    self.inputObj = inputObj  
#    self.makeInputSize()    
#    self.makeOutputSize(size,scale_factor)
#    print('upsamp','inputSize',self.inputSize,'outputSize',self.outputSize)
#
#  def makeInputSize(self):
#    if type(self.inputObj)==tuple:
#      self.inputSize = self.inputObj
#    else:
#      self.inputSize = self.inputObj.outputSize
#
#  def makeOutputSize(self,size,scale_factor):
#    if size is not None:
#      self.outputSize = size
#    elif scale_factor is not None:
#      self.outputSize = tuple((np.array(self.inputSize)*scale_factor).astype(int))

def padBatch(tensor,batchSize):
  lentopad = batchSize-len(tensor)
  if lentopad>0:
    toadd = tensor[-1:]
    s = list(toadd.size())
    s[0] = lentopad
    toadd = toadd.expand(s)
    tensor = torch.cat((tensor,toadd),dim=0)
  return tensor

def makePretrainedEmbeddings(embMat,trainable=False):  
  vocabSize,embSize = embMat.shape
  embs = nn.Embedding(vocabSize,embSize)
  embs.load_state_dict({'weight':FT(embMat)})
  for p in embs.parameters(): p.requires_grad = trainable
  return embs



    
    
    