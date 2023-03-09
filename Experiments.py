import Tree
import Tree_NoTreeCriterion
import pandas as pd
import logging
from scipy import stats
import sys
from sklift.datasets import fetch_lenta
from sklift.models import ClassTransformation
from sklift.metrics import uplift_at_k
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklift.models import TwoModels
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import warnings
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
from causalml.feature_selection.filters import FilterSelect
# from UMODL_featureImportance import getImportantVariables_UMODL_ForMultiProcessing
from sklearn.model_selection import StratifiedKFold
from causalml.inference.tree import UpliftRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklift.models import ClassTransformation
import multiprocessing as mp
import pickle
import os
import time
from causalml.metrics import plot_gain, auuc_score,plot_qini,qini_score
import math
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseSRegressor, BaseTRegressor, BaseXLearner, BaseDRLearner
from random import randrange, randint
from sklift.metrics import (
    uplift_at_k, uplift_auc_score, qini_auc_score, weighted_average_uplift
)
import random

random.seed(8)
np.random.seed(8)
# %matplotlib inline
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from xgboost import XGBRegressor
import warnings

from causalml.inference.meta import LRSRegressor
from causalml.inference.meta import XGBTRegressor, MLPTRegressor
from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseSRegressor, BaseTRegressor
from causalml.match import NearestNeighborMatch, MatchOptimizer, create_table_one
from causalml.propensity import ElasticNetPropensityModel
from causalml.dataset import *
from causalml.metrics import *
from UMODL_Forest import UMODL_RandomForest
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

import causalml
from causalml.dataset import make_uplift_classification
from CTL.causal_tree_learn import CausalTree

from ExtraFunctionForExpProtocol import createPredictionDF_2M
from ExtraFunctionForExpProtocol import createPredictionDF
from ExtraFunctionForExpProtocol import Average
from ExtraFunctionForExpProtocol import learnWithUpliftRandomForests
from ExtraFunctionForExpProtocol import learnWithUpliftTrees
from ExtraFunctionForExpProtocol import learnWith2MApproach
from ExtraFunctionForExpProtocol import dataChoice
from ExtraFunctionForExpProtocol import learnWithMetaLearners
from ExtraFunctionForExpProtocol import learnWithCausalTrees
from ExtraFunctionForExpProtocol import learnWithCausalForests

ExpMode=sys.argv[1] #MODES: DiffSize, OneSize
dataName=sys.argv[2]
OptionNumber=sys.argv[3]
TrainingSampleSize=''
try:
    TrainingSampleSize = int(sys.argv[4])
except:
    if ExpMode=='DiffSize':
        raise
    else:
        print("ExpMode ",ExpMode)

#ConstantsForResSavings
LogFileDirec='Log_files_10Trees'
PKLFileDirec='10Trees'
BenchmarkStatsDirec='BenchmarkTrees_10Trees'
NumberOfTreesInForests=10
'''
For Hillstrom and bank datasets we have three options||
For criteo we have two options 
For Megafon, Information_R and dataOrange13_v1 it's only one option||
'''
    
datasetNames=['criteo-uplift-v2.1','megafon_dataset','bank-full','hillstorm_no_indices','gerber','cmpOrange2014_v1','cmpOrange201501-03_v1','cmpOrange2013_v1']

url='/data/userstorage/mrafla/Datasets/'

if dataName=='criteo-uplift-v2.1':
    df=pd.read_csv("~/../../data/userstorage/mrafla/Datasets/"+dataName+str(OptionNumber)+"_100000.csv")
else:
    df=dataChoice(dataName,OptionNumber,url)
#     df.dropna(axis=0,inplace=True)
#     df.fillna(df.mean(),inplace=True)
    print("Does nan values exist here ",df.isnull().values.any())
    print("df head is in the experimental Protocol\n",df.head())
    print("df shape is \n",df.shape)
    print("==================================================")

    cols = df.columns
    num_cols = list(df._get_numeric_data().columns)

    num_cols.remove('T')
    num_cols.remove('Y')
    for num_col in num_cols:
        df[num_col] = df[num_col].fillna(df[num_col].mean())

    categoricalCols=list(set(cols) - set(num_cols))
    if 'T' in categoricalCols:
        categoricalCols.remove('T')
    if 'Y' in categoricalCols:
        categoricalCols.remove('Y')
    print("Categorical variables are  ",categoricalCols)
    for catCol in categoricalCols:
        print("Encoding ",catCol)
        df[catCol] = df[catCol].fillna(df[catCol].mode()[0])
        DictValVsUplift={}
        for val in df[catCol].value_counts().index:
            dataset_slice=df[df[catCol]==val]
            t0j0=dataset_slice[(dataset_slice['T']==0)&(dataset_slice['Y']==0)].shape[0]
            t0j1=dataset_slice[(dataset_slice['T']==0)&(dataset_slice['Y']==1)].shape[0]
            t1j0=dataset_slice[(dataset_slice['T']==1)&(dataset_slice['Y']==0)].shape[0]
            t1j1=dataset_slice[(dataset_slice['T']==1)&(dataset_slice['Y']==1)].shape[0]

            if (t1j1+t1j0)==0:
                UpliftInThisSlice=-1
            elif (t0j1+t0j1)==0:
                 UpliftInThisSlice=0
            else:
                UpliftInThisSlice=(t1j1/(t1j1+t1j0))-(t0j1/(t0j1+t0j1))
            DictValVsUplift[val]=UpliftInThisSlice
        # print("DictValVsUplift")
        # print(DictValVsUplift)
        OrderedDict={k: v for k, v in sorted(DictValVsUplift.items(), key=lambda item: item[1])}
        encoded_i=0
        for k,v in OrderedDict.items():
            df[catCol] = df[catCol].replace([k],encoded_i)
            encoded_i+=1
    try:
        df.to_csv("~/../../data/userstorage/mrafla/Datasets/"+dataName+str(OptionNumber)+"_"+str(TrainingSampleSize)+".csv",index=False)
        print("saved")
    except:
        print("File cannot be saved")

print("df head is ",df.head())
print("df shape is ",df.shape)
print("treatment ratio ",df['T'].value_counts(normalize=True))
print("treatment ratio ",df['Y'].value_counts(normalize=True))
log_file_name=LogFileDirec+'/'+dataName+str(OptionNumber)+"_"+str(TrainingSampleSize)+'.log'
log_file = open(log_file_name,"w")

sys.stdout = log_file

print("df shape before learning is ",df.shape)
print("df head ",df.head())
print("Does nan values exist ",df.isnull().values.any())

df['TY']=df['T'].astype(str)+df['Y'].astype(str)
skf = StratifiedKFold(n_splits=10,random_state=60,shuffle=True)

s = skf.split(df,df['TY'])

df.drop(['TY'],inplace=True,axis=1)
resQini_2M_rfc=[]
resQini_2M_LR=[]
resQini_2M_DT=[]
resQini_2M_Xgboost=[]
resQini_XLearnerLR=[]
resQini_RLearnerLR=[]
resQini_RLearnerRF=[]
resQini_XLearnerXgboost=[]
resQini_XLearnerRF=[]
resQini_RLearnerXgboost=[]
resQini_DR_Xgboost=[]
resQini_DR_RF=[]
resQini_DR_LR=[]
resQini_KL_RF=[]
resQini_Chi_RF=[]
resQini_ED_RF=[]
resQini_CTS_RF=[]
resQini_UMODL_RF=[]
resQini_UMODL_RF_NotAllVars=[]
resQini_CausalForests=[]
resQini_CausalTree=[]
resQini_UMODL_DT=[]
resQini_KL_DT=[]
resQini_Chi_DT=[]
resQini_ED_DT=[]
resQini_CTS_DT=[]
resQini_UMODL_DT_NoGlobal=[]

cols=list(df.columns[:-2])
foldnum=0
dataFrameInformationAboutTrainAndTestFolds=pd.DataFrame(columns=["Uplift Moyen Train",'T0Y0 Train','T0Y1 Train','T1Y0 Train','T1Y1 Train',"Uplift Moyen Test",'T0Y0 Test','T0Y1 Test','T1Y0 Test','T1Y1 Test'])
for train_index, test_index in s:
    df_train, df_test = df.iloc[train_index], df.iloc[test_index]
    if ExpMode=='DiffSize':
        if dataName=='criteo-uplift-v2.1':
            df_trainY1=df_train[df_train['Y']==1]
            if df_trainY1.shape[0]>(int(TrainingSampleSize/2)):
                df_trainY1=df_trainY1.sample(int(TrainingSampleSize/2),random_state=80)
            df_trainY0=df_train[df_train['Y']==0]
            df_trainY0=df_trainY0.sample(TrainingSampleSize-df_trainY1.shape[0],random_state=80)
            df_train=pd.concat([df_trainY0,df_trainY1])
        else:
            df_train=df_train.sample(TrainingSampleSize,random_state=80)
    '''
    Distributions of Train and test folds
    '''
    T0Y0_shape_train=df_train[(df_train['T']==0)&(df_train['Y']==0)].shape[0]
    T0Y1_shape_train=df_train[(df_train['T']==0)&(df_train['Y']==1)].shape[0]
    T1Y0_shape_train=df_train[(df_train['T']==1)&(df_train['Y']==0)].shape[0]
    T1Y1_shape_train=df_train[(df_train['T']==1)&(df_train['Y']==1)].shape[0]
    
    T0Y0_shape_test=df_test[(df_test['T']==0)&(df_test['Y']==0)].shape[0]
    T0Y1_shape_test=df_test[(df_test['T']==0)&(df_test['Y']==1)].shape[0]
    T1Y0_shape_test=df_test[(df_test['T']==1)&(df_test['Y']==0)].shape[0]
    T1Y1_shape_test=df_test[(df_test['T']==1)&(df_test['Y']==1)].shape[0]
    
    dataFrameInformationAboutTrainAndTestFolds.loc["Foldnum "+str(foldnum),'Uplift Moyen Train']=(T1Y1_shape_train/(T1Y0_shape_train+T1Y1_shape_train))-(T0Y1_shape_train/(T0Y0_shape_train+T0Y1_shape_train))
    dataFrameInformationAboutTrainAndTestFolds.loc["Foldnum "+str(foldnum),'T0Y0 Train']=T0Y0_shape_train
    dataFrameInformationAboutTrainAndTestFolds.loc["Foldnum "+str(foldnum),'T0Y1 Train']=T0Y1_shape_train
    dataFrameInformationAboutTrainAndTestFolds.loc["Foldnum "+str(foldnum),'T1Y0 Train']=T1Y0_shape_train
    dataFrameInformationAboutTrainAndTestFolds.loc["Foldnum "+str(foldnum),'T1Y1 Train']=T1Y1_shape_train
    
    dataFrameInformationAboutTrainAndTestFolds.loc["Foldnum "+str(foldnum),'Uplift Moyen Test']=(T1Y1_shape_test/(T1Y0_shape_test+T1Y1_shape_test))-(T0Y1_shape_test/(T0Y0_shape_test+T0Y1_shape_test))
    dataFrameInformationAboutTrainAndTestFolds.loc["Foldnum "+str(foldnum),'T0Y0 Test']=T0Y0_shape_test
    dataFrameInformationAboutTrainAndTestFolds.loc["Foldnum "+str(foldnum),'T0Y1 Test']=T0Y1_shape_test
    dataFrameInformationAboutTrainAndTestFolds.loc["Foldnum "+str(foldnum),'T1Y0 Test']=T1Y0_shape_test
    dataFrameInformationAboutTrainAndTestFolds.loc["Foldnum "+str(foldnum),'T1Y1 Test']=T1Y1_shape_test
    
    
#     #Model training and testing
#     #===========================================================================
#     #===========================================================================
#     print("UMODL Tree WITHOUT GLOBAL CRITERION")
#     #UMODL Tree no global criterion
#     T=Tree_NoTreeCriterion.UpliftTreeClassifier(df_train)
# #     T=Tree_withPossibleBug.UpliftTreeClassifier(df_train)
#     num_leaves=T.growTree()
#     print("num leaves is ",str(num_leaves),"in fold ",str(foldnum))
#     preds=T.predict(df_test[cols])
#     df_preds=createPredictionDF(df_test.copy(),preds,AlgoName="UMODL_DT_NO_GLOBAL",foldNum=str(foldnum),dataName=dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
#     resQini_UMODL_DT_NoGlobal.append(qini_auc_score( y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
#     #=====================================================================================================
#     #=====================================================================================================
    print("UMODL Tree")
    #UMODL Tree
    T=Tree.UpliftTreeClassifier(df_train)
    T.growTree()
    preds=T.predict(df_test[cols])
    df_preds=createPredictionDF(df_test.copy(),preds,AlgoName="UMODL_DT",foldNum=str(foldnum),dataName=dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
    resQini_UMODL_DT.append(qini_auc_score( y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
#     #=====================================================================================================
#     #=====================================================================================================
#     print("Causal Trees")
#     df_preds=learnWithCausalTrees('CausalTree',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
#     resQini_CausalTree.append(qini_auc_score( y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
# #     #=====================================================================================================
# #     #=====================================================================================================
    print("Causal Forests")
    df_preds=learnWithCausalForests('CausalForest',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
    resQini_CausalForests.append(qini_auc_score( y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
# #     #=====================================================================================================
# #     #=====================================================================================================
    
#     print("XLearnerLR")
#     df_preds=learnWithMetaLearners('XLearnerLR',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
#     resQini_XLearnerLR.append(qini_auc_score(y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
# #     #==================================================================================================================
# #     #=====================================================================================================
    print("XLearnerXgboost")
    df_preds=learnWithMetaLearners('XLearnerXgboost',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
    resQini_XLearnerXgboost.append(qini_auc_score(y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
# #     #==================================================================================================================
# #     #=====================================================================================================
#     print("XLearnerRF")
#     df_preds=learnWithMetaLearners('XLearnerRF',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
#     resQini_XLearnerRF.append(qini_auc_score(y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
    
# #     #==================================================================================================================
#     print("RLearnerLR")
#     df_preds=learnWithMetaLearners('RLearnerLR',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
#     resQini_RLearnerLR.append(qini_auc_score(y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
    
# #     #==================================================================================================================
    print("RLearnerXgboost")
    df_preds=learnWithMetaLearners('RLearnerXgboost',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
    resQini_RLearnerXgboost.append(qini_auc_score(y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
#     #==================================================================================================================
#     print("RLearnerRF")
#     df_preds=learnWithMetaLearners('RLearnerRF',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
#     resQini_RLearnerRF.append(qini_auc_score(y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))

# #     #==================================================================================================================
#     #=====================================================================================================,NotAllVars=False
    print("UMODL Forest")
    T=UMODL_RandomForest(df_train,numberOfTrees=NumberOfTreesInForests)
    T.fit()
    preds=T.predict(df_test[cols])
    df_preds=createPredictionDF(df_test.copy(),preds,AlgoName="UMODL_RF",foldNum=str(foldnum),dataName=dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
    resQini_UMODL_RF.append(qini_auc_score( y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
    # #     #==================================================================================================================
#     #=====================================================================================================,NotAllVars=False
#     print("UMODL Forest Not all vars")
#     T=UMODL_RandomForest(df_train,numberOfTrees=NumberOfTreesInForests,NotAllVars=False)
#     T.fit()
#     preds=T.predict(df_test[cols])
#     df_preds=createPredictionDF(df_test.copy(),preds,AlgoName="UMODL_RF_NotAllVars",foldNum=str(foldnum),dataName=dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
#     resQini_UMODL_RF_NotAllVars.append(qini_auc_score( y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
# #     #=====================================================================================================
# #     #=====================================================================================================
#     print("Two Model Approach DT")
#     df_preds=learnWithMetaLearners('2MDT',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
#     resQini_2M_DT.append(qini_auc_score( y_true =df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))

# # #     #=====================================================================================================
# # #     #=====================================================================================================
# #     print("Two Model Approach LR")
#     print("Two Model Approach LR")
#     df_preds=learnWithMetaLearners('2MLR',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
#     resQini_2M_LR.append(qini_auc_score( y_true =df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
#     lr_trmnt=LogisticRegression(random_state=60)#,max_features=3,max_depth=10,min_samples_leaf=100)
#     lr_ctrl=LogisticRegression(random_state=60)#,max_features=3,max_depth=10,min_samples_leaf=100)
#     df_preds=learnWith2MApproach(lr_trmnt,lr_ctrl,df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize),'LR')
#     resQini_2M_LR.append(qini_auc_score( y_true =df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
#     #=====================================================================================================
# #     #=====================================================================================================
    print("Two Model Approach Xgboost")
    df_preds=learnWithMetaLearners('2MXgboost',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
    resQini_2M_Xgboost.append(qini_auc_score( y_true =df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
#     #=====================================================================================================
#     #=====================================================================================================
#     print("Two Model Approach RF")
#     df_preds=learnWithMetaLearners('2MRF',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
#     resQini_2M_rfc.append(qini_auc_score( y_true =df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
#     #=====================================================================================================
#     #=====================================================================================================

    print("DR_Xgboost")
    df_preds=learnWithMetaLearners('DR_Xgboost',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
    resQini_DR_Xgboost.append(qini_auc_score( y_true =df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
#     #=====================================================================================================
#     #=====================================================================================================
#     print("DR_RF")
#     df_preds=learnWithMetaLearners('DR_RF',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
#     resQini_DR_RF.append(qini_auc_score( y_true =df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
# #    #=====================================================================================================
# #     #=====================================================================================================
#     print("DR_LR")
#     df_preds=learnWithMetaLearners('DR_LR',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
#     resQini_DR_LR.append(qini_auc_score( y_true =df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
    
# #     #=====================================================================================================
# #     #=====================================================================================================
    print("ED RF")
    df_preds=learnWithUpliftRandomForests('ED',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
    resQini_ED_RF.append(qini_auc_score(y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
    #=====================================================================================================
    #=====================================================================================================
    print("Chi RF")
    df_preds=learnWithUpliftRandomForests('Chi',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
    resQini_Chi_RF.append(qini_auc_score(y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
    #=====================================================================================================
    #=====================================================================================================
    print("KL RF")
    df_preds=learnWithUpliftRandomForests('KL',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
    resQini_KL_RF.append(qini_auc_score(y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
    #=====================================================================================================
    #=====================================================================================================
    print("CTS RF")
    df_preds=learnWithUpliftRandomForests('CTS',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
    resQini_CTS_RF.append(qini_auc_score(y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
    #=====================================================================================================
    #=====================================================================================================
    print("ED DT")
    df_preds=learnWithUpliftTrees('ED',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
    resQini_ED_DT.append(qini_auc_score(y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
    #=====================================================================================================
    #=====================================================================================================
    print("Chi DT")
    df_preds=learnWithUpliftTrees('Chi',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
    resQini_Chi_DT.append(qini_auc_score(y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
    #=====================================================================================================
    #=====================================================================================================
    print("KL DT")
    df_preds=learnWithUpliftTrees('KL',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
    resQini_KL_DT.append(qini_auc_score(y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
    #=====================================================================================================
    #=====================================================================================================
    print("CTS DT")
    df_preds=learnWithUpliftTrees('CTS',df_train.copy(),df_test.copy(),cols,foldnum,dataName+str(OptionNumber)+"_"+str(TrainingSampleSize))
    resQini_CTS_DT.append(qini_auc_score(y_true=df_preds['y'], uplift=df_preds['Predictions'],treatment=df_preds['Treatment']))
    #=====================================================================================================
    #=====================================================================================================
    
    df_train.to_csv("/data/userstorage/mrafla/Datasets/"+BenchmarkStatsDirec+"/"+dataName+str(OptionNumber)+"_"+str(TrainingSampleSize)+"/df_train"+str(foldnum)+".csv",index=False)
    df_test.to_csv("/data/userstorage/mrafla/Datasets/"+BenchmarkStatsDirec+"/"+dataName+str(OptionNumber)+"_"+str(TrainingSampleSize)+"/df_test"+str(foldnum)+".csv",index=False)
    
    foldnum+=1
    print("finished a fold")
    
    
DictMethodVsListOfQinis={'2M_rfc':resQini_2M_rfc,
                         '2M_LR':resQini_2M_LR,
                         '2M_DT':resQini_2M_DT,
                         '2M_Xgboost':resQini_2M_Xgboost,
                         'XLearnerLR':resQini_XLearnerLR,
                         'XLearnerXgboost':resQini_XLearnerXgboost,
                         'XLearnerRF':resQini_XLearnerRF, ###############
                         'RLearnerLR':resQini_RLearnerLR,
                         'RLearnerXgboost':resQini_RLearnerXgboost,
                         'RLearnerRF':resQini_RLearnerRF,######################
                         'DR_Xgboost':resQini_DR_Xgboost,
                         'DR_RF':resQini_DR_RF,
                         'DR_LR':resQini_DR_LR,######################
                        'KL_RF':resQini_KL_RF,
                        'Chi_RF':resQini_Chi_RF,
                        'ED_RF':resQini_ED_RF,
                        'CTS_RF':resQini_CTS_RF,
                        'UMODL_RF':resQini_UMODL_RF,
                         'UMODL_RF_NotAllVars':resQini_UMODL_RF_NotAllVars,
                        'UMODL_DT':resQini_UMODL_DT,
                        'KL_DT':resQini_KL_DT,
                        'Chi_DT':resQini_Chi_DT,
                        'ED_DT':resQini_ED_DT,
                        'CTS_DT':resQini_CTS_DT,
                         'CausalTree':resQini_CausalTree,
                         'CausalForest':resQini_CausalForests,
                        'UMODL_NOGlobal':resQini_UMODL_DT_NoGlobal}

for key in DictMethodVsListOfQinis:
    if len(DictMethodVsListOfQinis[key])!=0:
        pkl_file_name='Pkl_files/'+PKLFileDirec+'/'+dataName+str(OptionNumber)+"_"+str(TrainingSampleSize)
        isExist = os.path.exists(pkl_file_name)
        if not isExist:
           # Create a new directory because it does not exist
            os.makedirs(pkl_file_name)
            print("The new directory is created!")
        pkl_file_name+="/"+key+'.pkl'
        with open(pkl_file_name, 'wb') as f:
            pickle.dump(DictMethodVsListOfQinis[key], f)
    
    
for k,v in DictMethodVsListOfQinis.items():
    try:
        print("Method ",k,' Qini ',Average(v))
    except:
        print("Couldn't calc average for ",k)
    
dataFrameInformationAboutTrainAndTestFolds.to_csv("/data/userstorage/mrafla/Datasets/"+BenchmarkStatsDirec+"/"+dataName+str(OptionNumber)+"_"+str(TrainingSampleSize)+"/TrainTestDistributions_"+str(OptionNumber)+"_"+str(TrainingSampleSize)+".csv")
