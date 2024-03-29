'''
Author: Mina Rafla (Orange Innovation Lannion) 
'''


from math import log, pi, pow, exp, lgamma, sqrt
from UMODL_Discretizer import UMODL_Discretizer

import warnings
# warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# np.random.seed(10)
import math
from matplotlib import rcParams

from operator import itemgetter
import bisect
from math import log, pi, pow, exp, lgamma, sqrt
import numpy as np
from typing import Callable
from math import ceil, floor
from operator import itemgetter
from sortedcontainers import SortedKeyList
from operator import add
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import logging
import time
import os
import pickle
import sys
import json
from equalFreq import generate_EqualFreqSteps
import equalFreq

from stats_rissanen import universal_code_natural_numbers
from stats_rissanen import log_2_star
# import MDL_Criteria

logger = logging.getLogger('causalml')
logging.basicConfig(level=logging.INFO)

from scipy.special import comb
import scipy.special as sc

from operator import itemgetter
import operator
import bisect
import stats_rissanen
from sklift.metrics import (
    uplift_at_k, uplift_auc_score, qini_auc_score, weighted_average_uplift
)

# from NodeClass import Node
# from DLLClass import DLL

from helperFunctions import log_fact
from helperFunctions import log_binomial_coefficient

import helperFunctions


from stats_rissanen import BoundedNaturalNumbersUniversalCodeLength
import scipy.special as sc
import random
# random.seed(10)

'''
Helper attributes
'''



_Log_Fact_Table = []


def log_fact(n):
    """
    Compute log(fact(n))
    :param n:
    :return: value of log(fact(n))
    """
    # print("\t\t asked for log_fact(n=%d)"%n)
    # Use approximation for large n
    if n > 1e6:
        # print('\t\t Using approximation : res=%.f' %log_fact_approx(n))
        return log_fact_approx(n)
    # computation of values, tabulation in private array
    else:
        s = len(_Log_Fact_Table)
        if n >= s:
            if s == 0:
                _Log_Fact_Table.append(0)
            size = len(_Log_Fact_Table)
            while size <= n:
                # print('%d<=%d' %(size,n))
                _Log_Fact_Table.append(log(size) + _Log_Fact_Table[size - 1])
                size = size + 1
        return _Log_Fact_Table[n]
def log_binomial_coefficient(n: int, k: int):
    """
    Computes the log of the binomial coefficient  (n
                                                   k)
    (log of the total number of combinations of k elements from n)
    :param n: Total number of elements
    :param k: Number of selected elements
    :return:
    """
    nf = log_fact(n)
    kf = log_fact(k)
    nkf = log_fact(n - k)
    return (nf - nkf) - kf








'''
Attributes:

N = Number of individuals in the node
nj=number of individuals of class j ==> an array [Ni0,Ni1]
Nt=number of individuals for each class ==> ann array [Ni0,Ni1]
isLeaf: boolean
Ntj[n00,n01,n10,n11]
NodeID:int
Children=[Child1,Child2]
X: segmentation variable for this node that will result in its children
CriterionCa
Note: We can consider each node as a dataset, so when creating a new node we create it using a dataset.

data may consist of [X,y,t]
X = [num_samples,num_features]
t : array-like, shape = [num_samples]
y : array-like, shape = [num_samples]
'''
class Node:
    def __init__(self, data,treatmentName,outcomeName,ID=None):
        self.id=ID
        self.data=data.copy() #ordered data
        self.treatment=treatmentName
        self.output=outcomeName
#         print("self.data is ",self.data.head())
        self.N=data.shape[0]
        self.Nj = data[data[self.output]==1].shape[0] 
        self.Ntj=[data[(data[self.treatment]==0)&(data[self.output]==0)].shape[0],
                 data[(data[self.treatment]==0)&(data[self.output]==1)].shape[0],
                 data[(data[self.treatment]==1)&(data[self.output]==0)].shape[0],
                 data[(data[self.treatment]==1)&(data[self.output]==1)].shape[0]]
        
        self.X=data.iloc[:,:-2].copy()
        self.T=data.iloc[:,-2].copy()
        self.Y=data.iloc[:,-1].copy()
        
        
        
        try:
            if (self.Ntj[2]+self.Ntj[3])==0:
                denum=0.00001
            else:
                denum=(self.Ntj[2]+self.Ntj[3])
            self.outcomeProbInTrt=(self.Ntj[3]/denum)
        except:
            self.outcomeProbInTrt=0
        try:
            if (self.Ntj[0]+self.Ntj[1])==0:
                denum=0.00001
            else:
                denum=self.Ntj[0]+self.Ntj[1]
            self.outcomeProbInCtrl=(self.Ntj[1]/denum)
        except:
            self.outcomeProbInCtrl=0
        self.averageUplift=self.outcomeProbInTrt-self.outcomeProbInCtrl
        
        self.Attribute=None
        self.SplitThreshold=None
        self.isLeaf=True
        
        self.CandidateSplitsVsDataLeftDataRight=None
        self.CandidateSplitsVsCriterion=None
        
        self.leftNode=None
        self.rightNode=None
        
        self.PriorOfInternalNode=self.calcPriorOfInternalNode()
        self.PriorLeaf,self.LikelihoodLeaf,self.W=self.calcPriorAndLikelihoodLeaf()
    
    def calcPriorOfInternalNode(self):
        return log_binomial_coefficient(sum(self.Ntj)+1,1)
    
    def calcPriorAndLikelihoodLeaf(self):
        NumberOfTreatment=self.Ntj[2]+self.Ntj[3]
        NumberOfControl=self.Ntj[0]+self.Ntj[1]
        NumberOfPosOutcome=self.Ntj[1]+self.Ntj[3]
        NumberOfZeroOutcome=self.Ntj[0]+self.Ntj[2]

        #W=0
        LeafPrior_W0=log_binomial_coefficient(sum(self.Ntj)+1, 1)
        TreeLikelihood_W0=log_fact(sum(self.Ntj))-log_fact(NumberOfPosOutcome)-log_fact(NumberOfZeroOutcome)
        #W=1
        LeafPrior_W1=log_binomial_coefficient(NumberOfTreatment+1, 1)+log_binomial_coefficient(NumberOfControl+1, 1)
        TreeLikelihood_W1=(log_fact(NumberOfTreatment)-log_fact(self.Ntj[2])-log_fact(self.Ntj[3]))+(log_fact(NumberOfControl)-log_fact(self.Ntj[0])-log_fact(self.Ntj[1]))

        if (LeafPrior_W0+TreeLikelihood_W0)<(LeafPrior_W1+TreeLikelihood_W1):
            W=0
            LeafPrior=LeafPrior_W0
            TreeLikelihood=TreeLikelihood_W0
        else:
            W=1
            LeafPrior=LeafPrior_W1
            TreeLikelihood=TreeLikelihood_W1
        return LeafPrior,TreeLikelihood,W
    
    def DiscretizeVarsAndGetAttributesSplitsCosts(self):
        '''
        For this node loop on all attributes and get the optimal split for each one
        
        return a dictionary of lists
        {age: Cost, sex: Cost}
        The cost here corresponds to 
        1- the cost of this node to internal instead of leaf (CriterionToBeInternal-PriorLeaf)
        2- The combinatorial terms of the leaf prior and likelihood
        
        NOTE: Maybe I should save the AttributeToSplitVsLeftAndRightData in this node.
        '''
        features=list(self.X.columns)
        AttributeToSplitVsLeftAndRightData={}
#         print("features are ",features)
        for attribute in features:
            if len(self.X[attribute].value_counts())==1 or len(self.X[attribute].value_counts())==0:
                continue
            DiscRes=UMODL_Discretizer(self.X,self.T,self.Y,attribute)
            if DiscRes==-1:
                continue
            dataLeft,dataRight,threshold=DiscRes[0],DiscRes[1],DiscRes[2]
            AttributeToSplitVsLeftAndRightData[attribute]=[dataLeft,dataRight,threshold]
        
        self.CandidateSplitsVsDataLeftDataRight=AttributeToSplitVsLeftAndRightData.copy()
        CandidateSplitsVsCriterion=self.GetAttributesSplitsCosts(AttributeToSplitVsLeftAndRightData)
        self.CandidateSplitsVsCriterion=CandidateSplitsVsCriterion.copy()
        return CandidateSplitsVsCriterion.copy()
    
    def GetAttributesSplitsCosts(self,DictOfEachAttVsEffectifs):
        #Prior of Internal node is only the combinatorial calculations
        CriterionToBeInternal=self.calcPriorOfInternalNode() #In case we split this node, it will be no more a leaf but an internal node
        NewPriorVals=CriterionToBeInternal-self.PriorLeaf-self.LikelihoodLeaf
        
        CandidateSplitsVsCriterion={}
        for key in DictOfEachAttVsEffectifs:
            LeavesVal=self.updateTreeCriterion(DictOfEachAttVsEffectifs[key][:2])#,K_subset,subsetFeatures)
            CandidateSplitsVsCriterion[key]=NewPriorVals+LeavesVal
        return CandidateSplitsVsCriterion.copy()
            

    def updateTreeCriterion(self,LeftAndRightData,simulate=True):
#         NewNodeEffectifs=[T0J0,T0J1,T1J0,T1J1]
        LeavesVals=0
        for NewNodeEffectifs in LeftAndRightData:#Loop on Left and Right candidate nodes
            L=Node(NewNodeEffectifs,self.treatment,self.output)
            LeavesVals+=(L.PriorLeaf+L.LikelihoodLeaf)
            del L
        return LeavesVals
    
    def performSplit(self,Attribute):
        if self.CandidateSplitsVsDataLeftDataRight==None:
            raise
        else:
            self.isLeaf=False
            self.leftNode = Node(self.CandidateSplitsVsDataLeftDataRight[Attribute][0],self.treatment,self.output,ID=self.id*2)
            self.rightNode = Node(self.CandidateSplitsVsDataLeftDataRight[Attribute][1],self.treatment,self.output,ID=self.id*2+1)
            self.Attribute = Attribute
            self.SplitThreshold=self.CandidateSplitsVsDataLeftDataRight[Attribute][2]
            return self.leftNode,self.rightNode
        return -1
            
            

# Uplift Tree Classifier
class UpliftTreeClassifier:
    
    def __init__(self, data,treatmentName,outcomeName):#ordered data as argument
        self.nodesIds=0
        self.rootNode=Node(data,treatmentName,outcomeName,ID=self.nodesIds+1) 
        self.terminalNodes=[self.rootNode]
        self.internalNodes=[]
        
        self.K=len(list(data.columns))
        self.K_t=1
        self.features=list(data.columns)
        self.feature_subset=[]

        self.Prob_Kt=None
        self.EncodingOfBeingAnInternalNode=None
        self.ProbAttributeSelection=None
        self.PriorOfInternalNodes=None
        self.EncodingOfBeingALeafNodeAndContainingTE=len(self.terminalNodes)*log(2)*2 #TE=TreatmentEffect
        self.LeafPrior=None
        self.TreeLikelihood=None
        
        
        self.calcCriterion()
        
        self.TreeCriterion=self.Prob_Kt+self.EncodingOfBeingAnInternalNode+self.ProbAttributeSelection+self.PriorOfInternalNodes+self.EncodingOfBeingALeafNodeAndContainingTE+self.LeafPrior+self.TreeLikelihood
        
#================================================================================================================================================
#================================================================================================================================================
    def calcCriterion(self):
        self.calcProb_kt()
        self.calcPriorOfInternalNodes()
        self.calcEncoding()
        self.calcLeafPrior()
        self.calcTreeLikelihood()
#---------------------------------------------------------------------------------------------------------------------------------------  
    def calcProb_kt(self):
        self.Prob_Kt=universal_code_natural_numbers(self.K_t)-log_fact(self.K_t)+self.K_t*log(self.K)
#---------------------------------------------------------------------------------------------------------------------------------------    
    def calcPriorOfInternalNodes(self):
        if len(self.internalNodes)==0:
            self.PriorOfInternalNodes=0
            self.ProbAttributeSelection=0
        else:
            PriorOfInternalNodes=0
            for internalNode in self.internalNodes:
                PriorOfInternalNodes+=internalNode.PriorOfInternalNode
            self.PriorOfInternalNodes=PriorOfInternalNodes
            self.ProbAttributeSelection=log(self.K_t)*len(self.internalNodes)
#---------------------------------------------------------------------------------------------------------------------------------------    
    def calcEncoding(self):
        self.EncodingOfBeingALeafNodeAndContainingTE=len(self.terminalNodes)*log(2)*2
        self.EncodingOfBeingAnInternalNode=len(self.internalNodes)*log(2)
#---------------------------------------------------------------------------------------------------------------------------------------
    def calcTreeLikelihood(self):
        LeafLikelihoods=0
        for leafNode in self.terminalNodes:
            LeafLikelihoods+=leafNode.LikelihoodLeaf
        self.TreeLikelihood=LeafLikelihoods
#---------------------------------------------------------------------------------------------------------------------------------------    
    def calcLeafPrior(self):
        leafPriors=0
        for leafNode in self.terminalNodes:
            leafPriors+=leafNode.PriorLeaf
        self.LeafPrior=leafPriors
#---------------------------------------------------------------------------------------------------------------------------------------
#================================================================================================================================================
#================================================================================================================================================
    
    def growTree(self):
        #In case if we have a new attribute for splitting
        Prob_KtPlusOne=universal_code_natural_numbers(self.K_t+1)-log_fact(self.K_t+1)+(self.K_t+1)*log(self.K)
        ProbOfAttributeSelectionAmongSubsetAttributesPlusOne=log(self.K_t+1)*(len(self.internalNodes)+1)
        
        EncodingOfBeingAnInternalNodePlusOne=self.EncodingOfBeingAnInternalNode+log(2)
        
        #When splitting a node to 2 nodes, the number of leaf nodes is incremented only by one, since the parent node was leaf and is now internal.
        #2 for two extra leaf nodes multiplied by 2 for W. Total = 4.
        EncodingOfBeingALeafNodeAndContainingTEPlusTWO=self.EncodingOfBeingALeafNodeAndContainingTE+(2*log(2)) 
        
        EncodingOfInternalAndLeavesAndWWithExtraNodes=EncodingOfBeingAnInternalNodePlusOne+EncodingOfBeingALeafNodeAndContainingTEPlusTWO
        
        
        i=0
        while(True):
            NodeVsBestAttributeCorrespondingToTheBestCost={}
            NodeVsBestCost={}
            NodeVsCandidateSplitsCosts={}#Dictionary containing Nodes as key and their values are another dictionary each with attribute:CostSplit
            NodeVsCandidateSplitsCostsInTheNode={}

            for terminalNode in self.terminalNodes:

                #This if condition is here to not to repeat calculations of candidate splits
                if terminalNode.CandidateSplitsVsCriterion==None:
                    NodeVsCandidateSplitsCosts[terminalNode]=terminalNode.DiscretizeVarsAndGetAttributesSplitsCosts()
                else:
                    NodeVsCandidateSplitsCosts[terminalNode]=terminalNode.CandidateSplitsVsCriterion.copy()
                
                if len(NodeVsCandidateSplitsCosts[terminalNode])==0:
                    continue

                #Update Costs
                for attribute in NodeVsCandidateSplitsCosts[terminalNode]:
                    if attribute in self.feature_subset:
                        NodeVsCandidateSplitsCosts[terminalNode][attribute]+=(self.Prob_Kt
                                                                              +self.ProbAttributeSelection
                                                                              +EncodingOfInternalAndLeavesAndWWithExtraNodes
                                                                              +self.LeafPrior+self.TreeLikelihood+self.PriorOfInternalNodes)
                    else:
                        NodeVsCandidateSplitsCosts[terminalNode][attribute]+=(Prob_KtPlusOne
                                                                              +EncodingOfInternalAndLeavesAndWWithExtraNodes
                                                                              +ProbOfAttributeSelectionAmongSubsetAttributesPlusOne
                                                                              +self.LeafPrior+self.TreeLikelihood+self.PriorOfInternalNodes)
               
                #Once costs are updated, I get the key of the minimal value split for terminalNode
                KeyOfTheMinimalVal=min(NodeVsCandidateSplitsCosts[terminalNode], key=NodeVsCandidateSplitsCosts[terminalNode].get)
                
                NodeVsBestAttributeCorrespondingToTheBestCost[terminalNode]=KeyOfTheMinimalVal
                NodeVsBestCost[terminalNode]=NodeVsCandidateSplitsCosts[terminalNode][KeyOfTheMinimalVal]
            
            if len(list(NodeVsBestCost))==0:
                break
            
            OptimalNodeAttributeToSplitUp=min(NodeVsBestCost, key=NodeVsBestCost.get)
            OptimalVal=NodeVsBestCost[OptimalNodeAttributeToSplitUp]
            OptimalNode=OptimalNodeAttributeToSplitUp
            OptimalAttribute=NodeVsBestAttributeCorrespondingToTheBestCost[OptimalNodeAttributeToSplitUp]
            
            if OptimalVal<self.TreeCriterion:
                self.TreeCriterion=OptimalVal
                if OptimalAttribute not in self.feature_subset:
                    self.feature_subset.append(OptimalAttribute)
                    self.K_t+=1
                NewLeftLeaf,NewRightLeaf=OptimalNode.performSplit(OptimalAttribute)
                self.terminalNodes.append(NewLeftLeaf)
                self.terminalNodes.append(NewRightLeaf)
                self.internalNodes.append(OptimalNode)
                self.terminalNodes.remove(OptimalNode)
                
                self.calcCriterion()
            else:
                break
        print("Learning Finished")
        for node in self.terminalNodes:
            print("Node id ",node.id)
            print("Node outcomeProbInTrt ",node.outcomeProbInTrt)
            print("Node outcomeProbInCtrl ",node.outcomeProbInCtrl)
            print("self ntj ",node.Ntj)
        print("===============")
    
    
    def _traverse_tree(self, x, node):
        if node.isLeaf==True:
            return node.averageUplift
        
        if x[node.Attribute] <= node.SplitThreshold:
            return self._traverse_tree(x, node.leftNode)
        return self._traverse_tree(x, node.rightNode)
    
    def predict(self, X):
        predictions = [self._traverse_tree(X.iloc[x], self.rootNode) for x in range(len(X))]
        return np.array(predictions)
            