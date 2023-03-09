import Tree
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
import warnings
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseSRegressor, BaseTRegressor
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
from causalml.feature_selection.filters import FilterSelect
# from UMODL_featureImportance import getImportantVariables_UMODL_ForMultiProcessing
from sklearn.model_selection import StratifiedKFold
from causalml.inference.tree import UpliftRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
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
from xgboost import XGBClassifier
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
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV
from econml.metalearners import XLearner
from econml.dr import DRLearner
from econml.dml import DML
from econml.metalearners import TLearner
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from sklearn import tree


NumberOfTreesInForests=10
Direc='BenchmarkTrees_10Trees'


def createPredictionDF_2M(df_test,treatmentPreds,ControlPreds,preds,dataName,AlgoName="",foldNum=""):
    df_preds=pd.DataFrame()
    df_preds['YProbInTrt']=treatmentPreds
    df_preds['YProbInCtrl']=ControlPreds
    df_preds['Predictions']=preds
    # df_preds['Predictions']=predsNoisty
    df_preds['Treatment']=df_test['T'].reset_index()['T'].astype(int)
    df_preds['y']=df_test['Y'].reset_index()['Y'].astype(int)
    
    path="/data/userstorage/mrafla/Datasets/"+Direc+"/"+dataName
    isExist = os.path.exists(path)
    if not isExist:
       # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")


    df_preds.to_csv("/data/userstorage/mrafla/Datasets/"+Direc+"/"+dataName+"/df_preds"+str(AlgoName)+"fold"+foldNum+".csv")
    return df_preds.copy()
def createPredictionDF(df_test,preds,dataName,AlgoName="",foldNum=""):
    try:
        df_preds=pd.DataFrame()
        df_preds['Predictions']=preds
        df_preds['Treatment']=df_test['T'].reset_index()['T'].astype(int)
        df_preds['y']=df_test['Y'].reset_index()['Y'].astype(int)
        
        path="/data/userstorage/mrafla/Datasets/"+Direc+"/"+dataName
        isExist = os.path.exists(path)
        if not isExist:
           # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")
        
        df_preds.to_csv("/data/userstorage/mrafla/Datasets/"+Direc+"/"+dataName+"/df_preds"+str(AlgoName)+"fold"+foldNum+".csv")
        return df_preds.copy()
    except:
        print("EXCEPTION  ",preds)

def Average(lst):
    return sum(lst) / len(lst)


def learnWithUpliftRandomForests(evalFunction,df_train,df_test,cols,foldnum,dataName):
    TreeModel = UpliftRandomForestClassifier(n_estimators=NumberOfTreesInForests,control_name='0', evaluationFunction=evalFunction)
    TreeModel.fit(X = df_train[cols].values,treatment=df_train['T'].astype(str).values,y=df_train['Y'].values)

    y_preds=TreeModel.predict(df_test[cols].values)
    
    result = pd.DataFrame(y_preds,columns=TreeModel.classes_[1:])
    df_preds=createPredictionDF(df_test,result['1'].tolist(),AlgoName=evalFunction+"_RF",foldNum=str(foldnum),dataName=dataName)

    return df_preds

def learnWithCausalTrees(evalFunction,df_train,df_test,cols,foldnum,dataName):
    ctl = CausalTree(magnitude=False)
    ctl.fit(df_train[cols].values, df_train['Y'].values, df_train['T'].values)
    try:
        ctl.prune()
    except:
        print("Couldn't prune a CAUSAL Tree")
    ctl_predict = ctl.predict(df_test[cols].values)
    
    print("ctl_predict in causal trees ",ctl_predict)

    result = pd.DataFrame(ctl_predict,columns=['1'])
    df_preds=createPredictionDF(df_test,result['1'].tolist(),AlgoName=evalFunction+"_DT",foldNum=str(foldnum),dataName=dataName)
    return df_preds
def learnWithCausalForests(evalFunction,df_train,df_test,cols,foldnum,dataName):
    est = CausalForestDML(n_estimators=NumberOfTreesInForests,subforest_size=2)
    est.fit(df_train['Y'].values,  df_train['T'].values, X=df_train[cols].values, W=None)
    
    ctl_predict = est.effect(df_test[cols].values)
    
#     print("ctl_predict in causal Forests ",ctl_predict)

    result = pd.DataFrame(ctl_predict,columns=['1'])
    df_preds=createPredictionDF(df_test,result['1'].tolist(),AlgoName=evalFunction+"_DT",foldNum=str(foldnum),dataName=dataName)
    return df_preds

def learnWithUpliftTrees(evalFunction,df_train,df_test,cols,foldnum,dataName):
    TreeModel = UpliftTreeClassifier(control_name='0', evaluationFunction=evalFunction)
    TreeModel.fit(X = df_train[cols].values,treatment=df_train['T'].astype(str).values,y=df_train['Y'].values)
    try:
        TreeModel.prune(X = df_train[cols].values,treatment=df_train['T'].astype(str).values,y=df_train['Y'].values)
    except:
        print("couldn't do PRUNING")

    y_preds=TreeModel.predict(df_test[cols].values)
    
    result = pd.DataFrame(y_preds,
                      columns=['control','treatment'])
    df_preds=createPredictionDF(df_test,result['treatment'].tolist(),AlgoName=evalFunction+"_DT",foldNum=str(foldnum),dataName=dataName)
    return df_preds
def learnWithMetaLearners(evalFunction,df_train,df_test,cols,foldnum,dataName):
    if evalFunction=="XLearnerLR":
        #causalml
#         learner = BaseXRegressor(learner=LinearRegression())
#         learner.fit(X=df_train[cols].values, treatment=df_train['T'].values, y=df_train['Y'].values)

#         y_preds=learner.predict(df_test[cols].values)
        
        #EconML
        est = XLearner(models=LinearRegression())
        est.fit(df_train['Y'].values, df_train['T'].values, X=df_train[cols].values)
        y_preds=est.effect(df_test[cols].values)
    elif evalFunction=="XLearnerXgboost":
#         learner = BaseXRegressor(learner=XGBRegressor(n_estimators=NumberOfTreesInForests))
#         learner.fit(X=df_train[cols].values, treatment=df_train['T'].values, y=df_train['Y'].values)

#         y_preds=learner.predict(df_test[cols].values)
        #EconML
        est = XLearner(models=XGBRegressor(n_estimators=NumberOfTreesInForests))
        est.fit(df_train['Y'].values, df_train['T'].values, X=df_train[cols].values)
        y_preds=est.effect(df_test[cols].values)
    elif evalFunction=="XLearnerRF":
        #EconML
        est = XLearner(models=RandomForestRegressor(n_estimators=NumberOfTreesInForests))
        est.fit(df_train['Y'].values, df_train['T'].values, X=df_train[cols].values)
        y_preds=est.effect(df_test[cols].values)
    elif evalFunction=="RLearnerLR":
#         learner = BaseRRegressor(learner=LinearRegression())
#         learner.fit(X=df_train[cols].values, treatment=df_train['T'].values, y=df_train['Y'].values)

#         y_preds=learner.predict(df_test[cols].values)
        est = DML(
            model_y=LinearRegression(),
            model_t=LogisticRegression(),
            model_final=StatsModelsLinearRegression(fit_intercept=False),
            linear_first_stages=False,
            discrete_treatment=True
        )
        est.fit(df_train['Y'].values, df_train['T'].values, X=df_train[cols].values)

        y_preds=est.effect(df_test[cols].values)

        
    elif evalFunction=="RLearnerXgboost":
#         learner = BaseRRegressor(learner=XGBRegressor(n_estimators=NumberOfTreesInForests))
#         learner.fit(X=df_train[cols].values, treatment=df_train['T'].values, y=df_train['Y'].values)

#         y_preds=learner.predict(df_test[cols].values)
        
        est = DML(
            model_y=XGBRegressor(n_estimators=NumberOfTreesInForests,random_state=60),
            model_t=XGBClassifier(n_estimators=NumberOfTreesInForests,random_state=60),
            model_final=StatsModelsLinearRegression(fit_intercept=False),
            linear_first_stages=False,
            discrete_treatment=True
        )
        est.fit(df_train['Y'].values, df_train['T'].values, X=df_train[cols].values)

        y_preds=est.effect(df_test[cols].values)

    elif evalFunction=="RLearnerRF":
        #CausalML
#         learner = BaseRRegressor(learner=RandomForestRegressor(n_estimators=NumberOfTreesInForests))
#         learner.fit(X=df_train[cols].values, treatment=df_train['T'].values, y=df_train['Y'].values)

#         y_preds=learner.predict(df_test[cols].values)
        #EconML
        est = DML(
            model_y=RandomForestRegressor(n_estimators=NumberOfTreesInForests),
            model_t=RandomForestClassifier(n_estimators=NumberOfTreesInForests),
            model_final=StatsModelsLinearRegression(fit_intercept=False),
            linear_first_stages=False,
            discrete_treatment=True
        )
        est.fit(df_train['Y'].values, df_train['T'].values, X=df_train[cols].values)

        y_preds=est.effect(df_test[cols].values)
    elif evalFunction=="DR_LR":
#         learner=BaseDRLearner(learner=XGBClassifier(n_estimators=NumberOfTreesInForests))
#         learner.fit(X=df_train[cols].values, treatment=df_train['T'].values, y=df_train['Y'].values)

#         y_preds=learner.predict(df_test[cols].values)
        est = DRLearner(model_propensity=LogisticRegression(),
                model_regression=LinearRegression(),
                model_final=LassoCV(cv=3),
                featurizer=None)
        est.fit(df_train['Y'].values, df_train['T'].values, X=df_train[cols].values)
        y_preds=est.effect(df_test[cols].values)

    elif evalFunction=="DR_Xgboost":
#         learner=BaseDRLearner(learner=XGBClassifier(n_estimators=NumberOfTreesInForests))
#         learner.fit(X=df_train[cols].values, treatment=df_train['T'].values, y=df_train['Y'].values)

#         y_preds=learner.predict(df_test[cols].values)
        est = DRLearner(model_propensity=XGBClassifier(n_estimators=NumberOfTreesInForests),
                model_regression=XGBRegressor(n_estimators=NumberOfTreesInForests),
                model_final=LassoCV(cv=3),
                featurizer=None)
        est.fit(df_train['Y'].values, df_train['T'].values, X=df_train[cols].values)
        y_preds=est.effect(df_test[cols].values)

    elif evalFunction=="DR_RF":
        #CausalML
#         learner=BaseDRLearner(learner=RandomForestClassifier(n_estimators=NumberOfTreesInForests))
#         learner.fit(X=df_train[cols].values, treatment=df_train['T'].values, y=df_train['Y'].values)

#         y_preds=learner.predict(df_test[cols].values)
        #EconML
        est = DRLearner(model_propensity=RandomForestClassifier(n_estimators=NumberOfTreesInForests),
                model_regression=RandomForestRegressor(n_estimators=NumberOfTreesInForests),
                model_final=LassoCV(cv=3),
                featurizer=None)
        est.fit(df_train['Y'].values, df_train['T'].values, X=df_train[cols].values)
        y_preds=est.effect(df_test[cols].values)
    elif evalFunction=="2MLR":
        trmntClassifier=LogisticRegression()
        ctrlClassifier=LogisticRegression()
        trmntClassifier.fit(df_train[df_train['T']==1][cols].values, df_train[df_train['T']==1]['Y'])
        trmntPreds=trmntClassifier.predict_proba(df_test[cols].values)

        ctrlClassifier.fit(df_train[df_train['T']==0][cols].values, df_train[df_train['T']==0]['Y'].values)
        CtrlPreds=ctrlClassifier.predict_proba(df_test[cols].values)

        treatmentPreds=[]
        for index in range(len(trmntPreds)):
            treatmentPreds.append(trmntPreds[index][1])

        ControlPreds=[]
        for index in range(len(CtrlPreds)):
            ControlPreds.append(CtrlPreds[index][1])

        y_preds=[a_i - b_i for a_i, b_i in zip(treatmentPreds, ControlPreds)]
        #CausalML
#         learner=BaseDRLearner(learner=RandomForestClassifier(n_estimators=NumberOfTreesInForests))
#         learner.fit(X=df_train[cols].values, treatment=df_train['T'].values, y=df_train['Y'].values)

#         y_preds=learner.predict(df_test[cols].values)
        #EconML
#         est = TLearner(models=LinearRegression())
#         est.fit(df_train['Y'].values, df_train['T'].values, X=df_train[cols].values)
#         y_preds=est.effect(df_test[cols].values)
    elif evalFunction=="2MDT":
        trmntClassifier=tree.DecisionTreeClassifier()
        ctrlClassifier=tree.DecisionTreeClassifier()
        trmntClassifier.fit(df_train[df_train['T']==1][cols].values, df_train[df_train['T']==1]['Y'])
        trmntPreds=trmntClassifier.predict_proba(df_test[cols].values)

        ctrlClassifier.fit(df_train[df_train['T']==0][cols].values, df_train[df_train['T']==0]['Y'].values)
        CtrlPreds=ctrlClassifier.predict_proba(df_test[cols].values)

        treatmentPreds=[]
        for index in range(len(trmntPreds)):
            treatmentPreds.append(trmntPreds[index][1])

        ControlPreds=[]
        for index in range(len(CtrlPreds)):
            ControlPreds.append(CtrlPreds[index][1])

        y_preds=[a_i - b_i for a_i, b_i in zip(treatmentPreds, ControlPreds)]
        #EconML
#         est = TLearner(models=tree.DecisionTreeClassifier())
#         est.fit(df_train['Y'].values, df_train['T'].values, X=df_train[cols].values)
#         y_preds=est.effect(df_test[cols].values)

    elif evalFunction=="2MXgboost":
        #CausalML
        trmntClassifier=XGBClassifier(n_estimators=NumberOfTreesInForests)
        ctrlClassifier=XGBClassifier(n_estimators=NumberOfTreesInForests)
        trmntClassifier.fit(df_train[df_train['T']==1][cols].values, df_train[df_train['T']==1]['Y'])
        trmntPreds=trmntClassifier.predict_proba(df_test[cols].values)

        ctrlClassifier.fit(df_train[df_train['T']==0][cols].values, df_train[df_train['T']==0]['Y'].values)
        CtrlPreds=ctrlClassifier.predict_proba(df_test[cols].values)

        treatmentPreds=[]
        for index in range(len(trmntPreds)):
            treatmentPreds.append(trmntPreds[index][1])

        ControlPreds=[]
        for index in range(len(CtrlPreds)):
            ControlPreds.append(CtrlPreds[index][1])

        y_preds=[a_i - b_i for a_i, b_i in zip(treatmentPreds, ControlPreds)]

        
        
        #EconML  LogisticRegression()
#         est = TLearner(models=XGBClassifier(n_estimators=NumberOfTreesInForests))
#         est.fit(df_train['Y'].values, df_train['T'].values, X=df_train[cols].values)
#         y_preds=est.effect(df_test[cols].values)
    elif evalFunction=="2MRF":
        trmntClassifier=RandomForestClassifier(n_estimators=NumberOfTreesInForests)
        ctrlClassifier=RandomForestClassifier(n_estimators=NumberOfTreesInForests)
        trmntClassifier.fit(df_train[df_train['T']==1][cols].values, df_train[df_train['T']==1]['Y'])
        trmntPreds=trmntClassifier.predict_proba(df_test[cols].values)

        ctrlClassifier.fit(df_train[df_train['T']==0][cols].values, df_train[df_train['T']==0]['Y'].values)
        CtrlPreds=ctrlClassifier.predict_proba(df_test[cols].values)

        treatmentPreds=[]
        for index in range(len(trmntPreds)):
            treatmentPreds.append(trmntPreds[index][1])

        ControlPreds=[]
        for index in range(len(CtrlPreds)):
            ControlPreds.append(CtrlPreds[index][1])

        y_preds=[a_i - b_i for a_i, b_i in zip(treatmentPreds, ControlPreds)]


        #CausalML
#         learner=BaseDRLearner(learner=RandomForestClassifier(n_estimators=NumberOfTreesInForests))
#         learner.fit(X=df_train[cols].values, treatment=df_train['T'].values, y=df_train['Y'].values)

#         y_preds=learner.predict(df_test[cols].values)
        #EconML
#         est = TLearner(models=RandomForestClassifier(n_estimators=NumberOfTreesInForests))
#         est.fit(df_train['Y'].values, df_train['T'].values, X=df_train[cols].values)
#         y_preds=est.effect(df_test[cols].values)
        
    else:
        print("No meta learner was specified")
        raise
    
#     print("ypreds ",y_preds)
    result = pd.DataFrame(y_preds,
                      columns=['treatment'])
    df_preds=createPredictionDF(df_test,result['treatment'].tolist(),AlgoName=evalFunction+"_DT",foldNum=str(foldnum),dataName=dataName)
    return df_preds
def learnWith2MApproach(trmntClassifier,ctrlClassifier,df_train,df_test,cols,foldnum,dataName,classifier):
    trmntClassifier.fit(df_train[df_train['T']==1][cols].values, df_train[df_train['T']==1]['Y'])
    trmntPreds=trmntClassifier.predict_proba(df_test[cols].values)

    ctrlClassifier.fit(df_train[df_train['T']==0][cols].values, df_train[df_train['T']==0]['Y'].values)
    CtrlPreds=ctrlClassifier.predict_proba(df_test[cols].values)
    
    treatmentPreds=[]
    for index in range(len(trmntPreds)):
        treatmentPreds.append(trmntPreds[index][1])

    ControlPreds=[]
    for index in range(len(CtrlPreds)):
        ControlPreds.append(CtrlPreds[index][1])

    preds=[a_i - b_i for a_i, b_i in zip(treatmentPreds, ControlPreds)]
    
    df_preds=createPredictionDF_2M(df_test,treatmentPreds,ControlPreds,preds,AlgoName="2M_"+classifier,foldNum=str(foldnum),dataName=dataName)
    
    return df_preds
    
def dataChoice(dataName,OptionNumber,url):
    if dataName=="syntheticDataPattern_VerySimplePattern" or dataName=='syntheticDataPattern_VeryCont_100x100_120K' or dataName=='syntheticDataPattern_gridCont_4x4_120K':
        df=pd.read_csv(dataName+".csv")
        df['T']=df['T'].astype(int)
        df['Y']=df['Y'].astype(int)

        cols=list(df.columns).copy()
        cols.remove('T')
        cols.remove('Y')
        df=df[cols+['T','Y']]
    if dataName=="rhc":
        df=pd.read_csv(url+dataName+'.csv',index_col=0)
        df.rename(columns={'swang1':'T','death':'Y'},inplace=True)
        df['T'].replace({'RHC':1,'No RHC':0},inplace=True)
        df['Y'].replace({'Yes':0,'No':1},inplace=True)
        df.drop(['dthdte','t3d30','lstctdte','surv2md1','dth30'],inplace=True,axis=1)
        cols=list(df.columns).copy()
        cols.remove('T')
        cols.remove('Y')
        df=df[cols+['T','Y']]
        
        df['T']=df['T'].astype(int)
        df['Y']=df['Y'].astype(int)
        
        for col in cols:
            if len(df[col].value_counts())<1000:
                df[col]=df[col].astype(str)
    if dataName=="Gerber"or dataName=="Gerber_Self"or dataName=="Gerber_NeighborVsCivic"or dataName=="Gerber_SelfVsCivic" or dataName=="gerber_selfAndNeighbours" or dataName=="Gerber_Self_EncodingCat" or dataName=='Gerber_Neighbours_EncodingCat':
        df=pd.read_csv(url+dataName+'.csv')
        cols=list(df.columns).copy()
        cols.remove('T')
        cols.remove('Y')
        df=df[cols+['T','Y']]
        df['T']=df['T'].astype(int)
        df['Y']=df['Y'].astype(int)
    if dataName=="Starbucks":
        df=pd.read_csv(url+dataName+'.csv')
        cols=list(df.columns).copy()
        cols.remove('T')
        cols.remove('Y')
        df=df[cols+['T','Y']]
        df['T']=df['T'].astype(int)
        df['Y']=df['Y'].astype(int)
        
#         df.reset_index(drop=True,inplace=True)
    if dataName=="syntheticDataPatternContByCell_112K" or dataName=="syntheticDataPattern_VeryCont_24K":
        df=pd.read_csv("./SynthData/"+dataName+".csv")
        cols=list(df.columns).copy()
        cols.remove('T')
        cols.remove('Y')
        df=df[cols+['T','Y']]
        df['T']=df['T'].astype(int)
        df['Y']=df['Y'].astype(int)
        df=df.sample(frac=1,random_state=80)
        df.reset_index(drop=True,inplace=True)
    if dataName=="lenta_dataset":
        df=pd.read_csv(url+dataName+'.csv')
        df['T']=df['T'].replace({'test':1,'control':0})
        df['T']=df['T'].astype(int)
        df['Y']=df['Y'].astype(int)
        
        cols=list(df.columns).copy()
        cols.remove('T')
        cols.remove('Y')
        
        df=df.sample(50000,random_state=80)
        df=df[cols+['T','Y']]
        df.reset_index(drop=True,inplace=True)
        print("df is \n",df)
    if dataName=="colon_death" or dataName=='colon_recurrence':
        print("dataName is ",dataName)
        df=pd.read_csv(dataName+'.csv')
        cols=list(df.columns).copy()
        cols.remove('T')
        cols.remove('Y')
        
        df['T']=df['T'].astype(int)
        df['Y']=df['Y'].astype(int)


        df=df[cols+['T','Y']]
        
    if dataName=='hillstorm_no_indices':
        print("dataName is ",dataName)
        df=pd.read_csv(url+dataName+'.csv')
        df = df.rename(columns={'segment': 'T', 'visit': 'Y'})

        if OptionNumber=='1':
            df['T'] = df['T'].replace(['Mens E-Mail'],1)
            df['T'] = df['T'].replace(['No E-Mail'],0)
            df=df[df['T']!='Womens E-Mail']
        elif OptionNumber=='2':
            df['T'] = df['T'].replace(['Womens E-Mail'],1)
            df['T'] = df['T'].replace(['No E-Mail'],0)
            df=df[df['T']!='Mens E-Mail']
        elif OptionNumber=='3':
            df['T'] = df['T'].replace(['Womens E-Mail'],1)
            df['T'] = df['T'].replace(['Mens E-Mail'],1)
            df['T'] = df['T'].replace(['No E-Mail'],0)
    #     df=df[df['T']!='Mens E-Mail']
        cols=list(df.columns).copy()
        cols.remove('T')
        cols.remove('Y')
        cols.remove('spend')
        cols.remove('conversion')
        
        df['T']=df['T'].astype(int)
        df['Y']=df['Y'].astype(int)


        df=df[cols+['T','Y']]
    if dataName=="RetailX5":
        dataName='retail_hero_final_model_train_data'
        df=pd.read_csv(url+dataName+'.csv')
#         df = df.sample(20000,random_state=80)
#         df['age']="str"+df['age'].astype(str)
        df=df.reset_index(drop=True)
    if dataName=='megafon_dataset':
        print("dataName is ",dataName)
        df=pd.read_csv(url+dataName+'.csv')
#         df = df.sample(20000,random_state=80)
        print("df after reading the csv file is ",df.head())
        print("df columns are \n",df.columns)
        df = df.rename(columns={'treatment_group': 'T', 'conversion': 'Y'})
        print("df after renaming treatment and output cols \n",df.head())
        df['T'] = df['T'].replace(['control'],0)
        df['T'] = df['T'].replace(['treatment'],1)
        cols=list(df.columns).copy()
        cols.remove('T')
        cols.remove('Y')
        df=df[cols+['T','Y']]
        df=df.reset_index(drop=True)
    if dataName=='bank-full':
        print("dataName is ",dataName)
        df=pd.read_csv(url+dataName+'.csv',sep=';')
        df = df.rename(columns={'contact': 'T', 'y': 'Y'})
        if OptionNumber=='1':
            df['T'] = df['T'].replace(['unknown'],0)
            df['T'] = df['T'].replace(['telephone'],1)
            df=df[df['T']!='cellular']
        elif OptionNumber=='2':
            df['T'] = df['T'].replace(['unknown'],0)
            df['T'] = df['T'].replace(['cellular'],1)
            df=df[df['T']!='telephone']
        elif OptionNumber=='3':
            df['T'] = df['T'].replace(['unknown'],0)
            df['T'] = df['T'].replace(['cellular'],1)
            df['T'] = df['T'].replace(['telephone'],1)

        cols=list(df.columns).copy()
        cols.remove('T')
        cols.remove('Y')
        


        df['Y'] = df['Y'].replace(['no'],0)
        df['Y'] = df['Y'].replace(['yes'],1)
        
        df['T']=df['T'].astype(int)
        df['Y']=df['Y'].astype(int)


        df=df[cols+['T','Y']]
    if dataName=='df_information':
#     /data/userstorage/mrafla/Datasets/Information_R_data/df_information.csv
        print("dataName is ",dataName)
        df=pd.read_csv(url+"Information_R_data/"+dataName+'.csv')

        cols=list(df.columns).copy()
        cols.remove('T')
        cols.remove('Y')

        df=df[cols+['T','Y']]
    if dataName=="cmpOrange2013_v1":
        df = pd.read_csv(url+dataName+".txt",sep='\t')
        df = df.rename(columns={'traitement': 'T', 'target_prise': 'Y'})
        df.drop(['source','traitementNR','target_churn'],inplace=True,axis=1)
        df.drop(['VAR85','VAR87'],axis=1,inplace=True)
        df=df.sample(20000,random_state=80)
        df['T']=df['T'].astype(int)
        df['Y']=df['Y'].astype(int)

        cols=list(df.columns).copy()
        cols.remove('T')
        cols.remove('Y')
        df=df[cols+['T','Y']]

        df=df.reset_index(drop=True)

    if dataName=='criteo-uplift-v2.1':
        print("dataName is ",dataName)
        df=pd.read_csv(url+dataName+'.csv')
        
        if OptionNumber=='5':#Option number 5 means with rebalancing
            df = df.rename(columns={'treatment': 'T', 'conversion': 'Y'})  # old method
            df.drop(['visit','exposure'],axis=1,inplace=True)
        elif OptionNumber=='6':#Option number 6 means with rebalancing
            df = df.rename(columns={'treatment': 'T', 'visit': 'Y'})  # old method
            df.drop(['conversion','exposure'],axis=1,inplace=True)

        df['T']=df['T'].astype(int)
        df['Y']=df['Y'].astype(int)
        
        CriteoCatCols=['f1','f3','f4','f5','f6','f8','f9','f11']
        for col in CriteoCatCols:
            print("col ",col," value counts is ",len(df[col].value_counts()))
            df[col]='cat'+df[col].astype(str)
        '''
        Resampling
        '''
        cols=list(df.columns).copy()
        cols.remove('T')
        cols.remove('Y')
        df=df[cols+['T','Y']]

        df=df.reset_index(drop=True)
    return df

