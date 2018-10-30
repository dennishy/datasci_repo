import pandas as pd 

'''
1- Start by pulling down Python 3, SKLearn, Pandas, and any dependencies. 
Pull down either the titanic dataset or the telco churn dataset. 
Load the dataset into Python, split off a validation dataset of 20% of your data, 
chosen randomly. With the remaining 80%, train a decision tree to classify the outcome 
(churn or survive). Use 10-fold cross-validation to get a true positive rate and 
false positive rate. Visualize a ROC curve (and understand what it is) and then
use your 20% validation set to double check your metrics. 
How close were the metrics created by 10-fold CV?
'''

data = '~/workspace/datasci/data/telco_churn.csv'

df = pd.read_csv(data)

print(df)
print('this')


