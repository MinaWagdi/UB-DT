# UPDATE: PLEASE REFER TO THE KUPLIFT LIBRARY for our official implementation: https://github.com/UData-Orange/kuplift

# UB-DT
Uplift Bayesian Decision tree: Bayesian decision tree algorithm designed specifically for uplift modeling.

This repository contains the code and supplementary results of an accepted paper in the PAKDD conference.

UPDATE: PLEASE REFER TO THE KUPLIFT LIBRARY for our official implementation: https://github.com/UData-Orange/kuplift

## How to test a UB-DT ?

Treatment and output variable should respetively be name 'T' and 'Y' and should be boolean variables.

<pre><code>
import Tree
import pandas as pd

T=Tree.UpliftTreeClassifier(df_train,treatmentName, outcomeName)
T.growTree()
preds=T.predict(df_test[cols])
</code></pre>

## How to test a UB-RF ?

Treatment and output variable should respetively be name 'T' and 'Y' and should be boolean variables.

<pre><code>
from UMODL_Forest import UMODL_RandomForest

T=UMODL_RandomForest(df_train,treatmentName, outcomeName,numberOfTrees=NumberOfTreesInForests)
T.fit()
preds=T.predict(df_test[cols])
</code></pre>

## Supplementary experimental results

### Real Datasets

#### UB-DT compared with uplift tree-based methods:
![image](https://user-images.githubusercontent.com/103153876/207080808-d96339d5-6e9a-4ea6-8869-d54a0a269cf0.png)

| {}            | 2M_DT        | KL_DT      | Chi_DT     | ED_DT         | CTS_DT     | UMODL_DT      |
|---------------|--------------|------------|------------|---------------|------------|---------------|
| Method        |              |            |            |               |            |               |
| Hillstrom-m   | 0.1(1.4)     | 1.1(1.9)   | 1.0(1.9)   | 0.0(1.4)      | 0.2(1.0)   | **1.6(1.6)**  |
| Hillstrom-w   | 0.9(2.0)     | 5.2(2.5)   | 5.2(2.6)   | **6.4(1.2)**  | -0.4(2.0)  | 4.8(2.3)      |
| Hillstrom-mw  | -0.5(0.8)    | -0.1(1.2)  | -0.8(1.1)  | **4.4(2.7)**  | -0.0(1.0)  | -0.4(1.4)     |
| Gerber-n      | **5.5(0.9)** | 1.3(0.8)   | 1.2(0.8)   | 1.1(0.6)      | 1.3(0.8)   | 1.9(0.6)      |
| Gerber-s      | **5.6(0.8)** | 0.4(0.5)   | 0.4(0.6)   | 0.5(0.3)      | 0.4(0.4)   | 0.8(0.6)      |
| Criteo-c      | 7.5(1.1)     | 4.1(1.4)   | 4.8(1.5)   | **15.2(0.3)** | 1.7(0.3)   | 13.7(3.2)     |
| Criteo-v      | 0.4(0.3)     | -1.2(0.2)  | -1.1(0.3)  | -1.3(0.3)     | 0.4(1.1)   | **3.6(1.2)**  |
| Megafon       | 5.2(0.5)     | 4.5(0.9)   | 4.7(0.9)   | 4.7(0.9)      | 4.9(0.8)   | **7.8(0.8)**  |
| Bank-tel      | 5.9(3.6)     | -12.5(2.8) | -10.8(7.0) | -10.2(7.8)    | -12.8(2.9) | **12.8(8.0)** |
| Bank-cell     | 10.5(3.3)    | -2.0(1.5)  | -1.4(2.5)  | -2.2(1.5)     | -3.7(1.5)  | **38.4(3.4)** |
| Bank-tel-cell | 10.7(2.3)    | -1.9(1.2)  | -1.2(2.1)  | -1.8(1.2)     | -3.4(1.4)  | **37.1(2.6)** |
| Information   | 4.4(3.2)     | -6.3(2.8)  | -6.3(2.8)  | -2.8(1.5)     | -5.4(1.5)  | **11.8(2.4)** |
| Starbucks     | 1.2(1.7)     | 20.1(3.0)  | 18.3(3.4)  | 19.9(3.2)     | 13.9(3.9)  | **20.2(3.5)** |
| RHC           | 12.8(1.9)    | 18.4(3.8)  | 19.9(4.2)  | 18.4(3.8)     | 16.7(2.5)  | **20.7(5.0)** |



#### UB-RF compared with meta learners and forest-based methods:

![image](https://user-images.githubusercontent.com/103153876/207080249-ffc2e052-dbd7-4096-a615-c0670a42a356.png)

| {}            | XLearnerLR    | XLearnerXgboost | XLearnerRF   | RLearnerLR | RLearnerXgboost | RLearnerRF | DR_LR      | DR_Xgboost | DR_RF      | 2M_LR         | 2M_Xgboost | 2M_rfc    | KL_RF      | Chi_RF       | ED_RF         | CTS_RF     | UB_RF      | CausalForest |
|---------------|---------------|-----------------|--------------|------------|-----------------|------------|------------|------------|------------|---------------|------------|-----------|------------|--------------|---------------|------------|---------------|--------------|
| Method        |               |                 |              |            |                 |            |            |            |            |               |            |           |            |              |               |            |               |              |
| Hillstrom-m   | 0.2(2.0)      | 0.3(2.3)        | -0.3(1.9)    | 0.2(2.1)   | 0.3(1.8)        | 0.9(2.3)   | 1.3(1.8)   | 1.2(1.6)   | -0.9(2.0)  | 0.2(2.0)      | 0.7(2.3)   | -0.7(1.5) | -0.0(2.1)  | -0.9(1.5)    | 0.7(1.5)      | 1.1(1.9)   | **1.8(1.6)**  | -0.2(1.6)    |
| Hillstrom-w   | 6.2(1.4)      | 6.2(1.7)        | 2.6(2.7)     | 6.3(1.5)   | 6.2(1.4)        | 5.5(1.6)   | 6.0(1.4)   | 6.0(1.4)   | -0.2(1.6)  | 6.2(1.4)      | 4.9(1.1)   | 0.5(0.9)  | 6.2(1.1)   | **7.0(1.0)** | 6.2(1.1)      | 5.7(1.3)   | 6.7(1.1)      | 2.1(1.9)     |
| Hillstrom-mw  | 3.8(2.7)      | 3.7(2.3)        | 0.5(1.3)     | 3.8(2.7)   | **3.9(2.7)**    | 3.8(2.5)   | 3.8(2.7)   | 3.8(2.8)   | -0.3(1.7)  | 3.8(2.7)      | 3.0(2.0)   | 0.1(1.4)  | 3.0(1.3)   | 2.8(1.5)     | 3.6(2.5)      | 2.3(2.4)   | 3.1(1.7)      | 0.1(1.7)     |
| Gerber-n      | 1.9(0.6)      | 3.7(0.6)        | **8.5(1.2)** | 1.9(0.6)   | 1.9(0.7)        | 1.9(0.7)   | 0.2(0.6)   | 0.5(0.9)   | 0.3(0.8)   | 1.9(0.6)      | 3.1(0.6)   | 2.4(1.0)  | 1.8(1.0)   | 2.1(1.1)     | 1.9(0.5)      | 1.4(1.0)   | 2.7(0.7)      | 2.9(1.0)     |
| Gerber-s      | 1.7(0.7)      | 2.4(0.9)        | **8.4(1.7)** | 1.7(0.7)   | 1.7(0.7)        | 1.6(0.7)   | 0.5(0.7)   | 0.6(0.9)   | -0.0(0.3)  | 1.7(0.7)      | 2.2(0.8)   | 2.8(0.8)  | 1.3(1.0)   | 1.4(0.6)     | 1.6(0.8)      | 1.4(0.7)   | 1.8(0.8)      | 3.1(0.5)     |
| Criteo-c      | -1.0(2.1)     | **22.3(1.8)**   | 15.6(1.4)    | 14.8(2.2)  | 19.4(1.0)       | 19.4(1.1)  | 2.5(9.9)   | 20.0(0.6)  | 11.1(7.8)  | -1.0(2.1)     | 19.5(1.6)  | 8.4(1.3)  | 14.6(3.5)  | 12.4(4.3)    | 21.1(2.3)     | 7.3(3.9)   | 18.7(1.5)     | 10.9(2.4)    |
| Criteo-v      | 2.4(0.7)      | 0.3(0.8)        | 1.3(0.6)     | 5.5(0.5)   | 5.3(0.5)        | 5.7(0.6)   | 2.7(3.0)   | 4.8(1.5)   | -4.7(4.1)  | 2.4(0.7)      | 3.9(0.5)   | 0.5(0.2)  | 5.4(1.2)   | 4.8(1.7)     | **6.1(1.0)**  | 2.4(0.8)   | 5.7(0.7)      | 0.4(0.4)     |
| Megafon       | 2.6(0.5)      | **18.2(0.6)**   | 14.3(0.4)    | 2.7(0.6)   | 2.6(0.5)        | 2.5(0.6)   | 1.6(1.2)   | 2.2(0.9)   | 0.2(0.4)   | 2.6(0.6)      | 16.6(0.9)  | 3.4(0.3)  | 11.2(0.7)  | 11.0(1.2)    | 10.8(0.8)     | 9.2(1.1)   | 12.8(1.0)     | 9.7(0.7)     |
| Bank-tel      | **35.5(6.6)** | 14.5(7.6)       | 5.3(9.2)     | 23.0(5.8)  | 2.8(8.8)        | 2.6(11.8)  | -20.1(8.6) | 16.0(9.0)  | 14.5(14.4) | **35.5(6.6)** | 21.1(11.6) | 9.1(6.0)  | -15.5(6.3) | -6.1(12.6)   | -15.8(5.6)    | -18.7(2.9) | 26.7(7.2)     | 25.4(5.3)    |
| Bank-cell     | 33.0(3.2)     | 18.8(4.7)       | 16.9(4.7)    | 22.2(2.0)  | 23.3(3.6)       | 15.9(5.0)  | 11.3(2.4)  | 17.4(6.5)  | 7.3(7.7)   | 33.0(3.2)     | 31.0(3.9)  | 15.2(2.9) | 0.4(2.3)   | 1.5(2.5)     | -2.5(2.6)     | -1.0(1.9)  | **45.5(2.7)** | 20.8(2.6)    |
| Bank-tel-cell | 32.2(3.6)     | 16.2(5.6)       | 16.2(3.3)    | 23.7(2.8)  | 23.8(2.5)       | 20.0(9.1)  | 11.9(3.0)  | 17.0(3.4)  | 8.8(10.3)  | 32.2(3.6)     | 30.5(2.7)  | 14.5(2.9) | 1.4(3.4)   | -0.4(5.7)    | -1.7(3.1)     | -0.5(2.3)  | **46.1(2.1)** | 23.5(2.9)    |
| Information   | 10.0(2.6)     | **14.9(3.3)**   | 12.6(2.9)    | 10.1(2.9)  | 10.0(3.1)       | 9.3(2.3)   | 1.8(3.0)   | 4.1(2.3)   | -0.2(3.2)  | 10.0(2.6)     | 13.7(4.1)  | 4.3(2.1)  | 9.6(2.0)   | 9.7(3.1)     | 11.2(2.9)     | 10.6(2.9)  | 12.0(3.1)     | 10.5(3.2)    |
| Starbucks     | 22.6(3.8)     | 22.3(4.5)       | 9.6(5.5)     | 22.6(3.9)  | 22.4(3.9)       | 22.0(3.9)  | 22.5(3.8)  | 22.4(3.7)  | -2.1(5.7)  | 22.6(3.8)     | 22.7(4.1)  | 0.1(0.3)  | 22.4(2.1)  | 21.4(3.4)    | **23.4(3.2)** | 20.8(3.1)  | 20.2(3.3)     | 8.1(3.7)     |
| RHC           | **37.9(4.6)** | 32.4(3.5)       | 30.2(4.2)    | 36.6(4.3)  | 31.3(4.3)       | 31.7(4.3)  | 26.6(5.7)  | 30.3(5.0)  | 1.5(5.8)   | 29.6(5.1)     | 34.6(4.3)  | 27.1(4.8) | 29.6(4.2)  | 29.7(5.0)    | 30.0(4.1)     | 29.1(3.7)  | 27.2(5.0)     | 27.6(4.5)    |
