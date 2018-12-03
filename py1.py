import pandas as pd 
import numpy as np
from sklearn import tree, model_selection, preprocessing, metrics
from collections import defaultdict
import matplotlib.pyplot as plt
import graphviz
#Change df default column display

pd.set_option('display.max_columns', 500)


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

#Datasets created onetime for convenience

#Complete dataset
source = '~/workspace/datasci/datasci_repo/data/telco_churn.csv'

df = pd.read_csv(source)

#print(dfSource.head(n=5))

#dfSource.shape = 7043,21

#Count nulls
#Note - no nulls? 

'''
missing = pd.concat([dfSource.isnull().sum()], axis = 1, keys = ['DataID'])
missing[missing.sum(axis=1)>0]

print(missing)
'''


#Preprocessing of categorical features 

#Create a defaultdict to facilitate mass preprocessing while retaining inverse 

mapDict = defaultdict(preprocessing.LabelEncoder)
df = df.apply(lambda x: mapDict[x.name].fit_transform(x))

#Separate dependent variable
#labels = df.pop('Churn')

#features = list(df.columns.values)

#split into test/train sets
train, test = model_selection.train_test_split(df, test_size=0.2)

y_train = train['Churn']
y_test = test['Churn']

x_train = train
x_train.pop('Churn')
x_test = test
x_test.pop('Churn')

#Instantiate decisiontree object
dTree = tree.DecisionTreeClassifier()

dTree.fit(x_train,y_train)

# Visualize data

churnTree = tree.export_graphviz(dTree, out_file=None, 
                         feature_names = list(x_train.columns.values),  
                         class_names = ['No churn', 'Churn'],
                         filled=True, rounded=True,  
                         special_characters=True)  


#Uncomment for decision tree viz
#graph = graphviz.Source(churnTree)
#graph.render('decision_tree.gv', view=True)


#Create predictions
y_pred = dTree.predict(x_test)



#evaluate
false_pos_rate, true_pos_rate, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(false_pos_rate, true_pos_rate)

print("AUC Score:")
print(roc_auc)

#Uncomment for roc viz
plt.plot(false_pos_rate, true_pos_rate, label = "auc = "+str(roc_auc))
plt.show()


'''
#Cross-validation
depth = []
for i in range(1,20):
    clf = tree.DecisionTreeClassifier(max_depth=i)
 
    scores = model_selection.cross_val_score(estimator=clf, X=dfTrain, y=labels, cv=10, n_jobs=4)

    depth.append([i,scores.mean()])

print("Depth cross validation scores:")
print(depth)

best_depth = max(depth, key=lambda x:x[1])

print('Best Depth:')
print(best_depth) 

'''