# Cancer ALL | AML Classifier - Bioinformatics Assignment (Gene Expression)
# Chirath Hettiarachchi

import numpy as np
from numpy.random import seed
seed(7)
import pandas as pd
import graphviz 
from sklearn.utils import shuffle
from sklearn.feature_selection import f_classif
from sklearn import tree
from sklearn.model_selection import cross_validate, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from scipy.stats import uniform as sp_rand
from scipy.stats import randint as sp_randint
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def modelEvaluator(clf,x,y,folds):
    
    skf = StratifiedKFold(folds)
    crossValResults = cross_validate(clf, x, y, cv=skf ,scoring = ['roc_auc','f1','precision','recall'])
    f1              = crossValResults['test_f1']
    roc_auc         = crossValResults['test_roc_auc']
    precision       = crossValResults['test_precision']
    recall          = crossValResults['test_recall']

    print("Mean ROC_AUC  :%f   SD: %f " %(np.mean(roc_auc),np.std(roc_auc)))
    print("Mean F1       :%f   SD: %f " %(np.mean(f1),np.std(f1)))
    print("Mean Precision:%f   SD: %f " %(np.mean(precision),np.std(precision)))
    print("Mean Recall   :%f   SD: %f " %(np.mean(recall),np.std(recall)))

def decisionTree(x,y,folds):
    print("Decision Tree Model Evaluated Through Cross Validation: ",folds)    
    clf = tree.DecisionTreeClassifier(random_state=7)
    skf = StratifiedKFold(folds)
    n_iter_search = 50

    #subset for the hyper param tune
    x, trainX, y, trainY = train_test_split(x, y, stratify=y, test_size=0.50)

    pipe = make_pipeline(preprocessing.StandardScaler(), clf)
    # Hyper parameters
    criterion = ['gini','entropy']
    max_depth = sp_randint(1, 5)
    min_samples_split = sp_randint(2, 10)
    min_samples_leaf = sp_randint(1, 10)

    param_dist = dict(decisiontreeclassifier__criterion=criterion, decisiontreeclassifier__max_depth=max_depth,
                decisiontreeclassifier__min_samples_split=min_samples_split,
                decisiontreeclassifier__min_samples_leaf=min_samples_leaf)
    random_search = RandomizedSearchCV(pipe, param_distributions=param_dist, 
                                n_iter=n_iter_search,cv=skf, scoring='roc_auc')
    
    random_search.fit(x,y) #hyper param tuning subset
    tuned_params= random_search.best_params_
    print tuned_params
    
    #Tuned Hyper parameters 
    criterion = tuned_params['decisiontreeclassifier__criterion']
    max_depth = tuned_params['decisiontreeclassifier__max_depth']
    min_samples_split = tuned_params['decisiontreeclassifier__min_samples_split'] 
    min_samples_leaf = tuned_params['decisiontreeclassifier__min_samples_leaf']
    tuned_clf = tree.DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,
            min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,random_state=7)
    
    #modelEvaluator(tuned_clf,trainX,trainY,folds) #training subset

    return tuned_clf

# Training Data
df = pd.read_csv('train.csv', sep=',', header = 0)
df.set_index('Subject',inplace=True)
df = df.transpose()
df = shuffle(df, random_state=7)
x  = df.drop(['Label','Class'], axis = 1)
x  = x.loc[:,~x.columns.duplicated()] # Remove the duplicated genes. 
y  = df['Label']
y  = y.astype(int)

### Shortlisting the genes based on statitical significant measures.
print("\n Conducting an ANOVA Test.... ")
anova  = f_classif(x,y)
result = pd.DataFrame(list(anova))
head   = list(x.columns) 
result.columns = head
# Row 1 - F values, 
# Row 2 - P values. 
result.index   = ['F','P'] 
sorted_result  = result.sort_values(by ='P', axis=1) 

# Select the 50 best genes. (Feature Reduction)
selected_column = []
genes = 0
for column in sorted_result:
    genes = genes + 1
    if genes > 50:
        break
    #print column
    selected_column.append(column)
reduced_x = x[selected_column]
print("Selected Features")
print(selected_column)

# Testing Data
test = pd.read_csv('test.csv', sep=',', header = 0)
test.set_index('Subject',inplace=True)
test   = test.transpose()
test   = test.loc[:,~test.columns.duplicated()]
test_x = test.drop(['Label','Class'], axis = 1)
test_x = test_x[selected_column]
test_y = test['Label']
test_y = test_y.astype(int)

# Question 01 & 02
# Develop a Decision Tree. features: reduced_x, label: y 
print("\n Developing a Decision Tree")
clf = decisionTree(reduced_x,y,5)
clf.fit(reduced_x,y)
dot_data = tree.export_graphviz(clf, out_file=None, feature_names= selected_column,  
            class_names=['AML','ALL'], filled=True, rounded=True, special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render("cancer_decisionTree") 
predicted = clf.predict(test_x)
test_y    = test_y.values
output = confusion_matrix(test_y, predicted)
print(pd.DataFrame(output, index=['true:AML', 'true:ALL'], columns=['pred:AML', 'pred:ALL']))

# Unsupervised Clustering. (Question 03)
# ALL - 1, AML - 0 ; original labels
print("\nRunning Unsupervised Clustering....")
reduced_x = reduced_x.append(test_x)
test_y    = pd.DataFrame(test_y)
y         = y.append(test_y) 
y.columns = ['orig_label']

kmeans = KMeans(n_clusters=2, random_state=7).fit(reduced_x)
labels =  kmeans.labels_
count_class_0 = 0
count_class_1 = 0
for label in labels:
    if label == 0:
        count_class_0 = count_class_0 + 1
    else:
        count_class_1 = count_class_1 + 1
orig_labels = y['orig_label'].tolist()

#match the kmeans labels with the original label
match     = 0
new_label = []
for index in range(0,len(labels)):
    if labels[index] == orig_labels[index]:
        match = match + 1
if match < (len(labels)/2):
    for label in labels:
        if label == 0:
            label = 1
        else:
            label = 0
        new_label.append(label)
else:
    new_label = labels
output = confusion_matrix(y, new_label) # AML is 0 & ALL is 1
print(pd.DataFrame(output, index=['true:AML', 'true:ALL'], columns=['pred:AML', 'pred:ALL']))
