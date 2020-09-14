import pandas as pd
import numpy as np

#load the data set
train = pd.read_csv(r'C:\Users\user-pc\Documents\Python\MLtrain.csv')

train = train.replace(' ', np.nan)
print(train.isnull().sum())

#fill the missing value using mean value
train = train.fillna(train.mean())
print(train.isnull().sum())

#formatted the label in numbers
train.label = train.label.map({'correct':1,'incorrect':0 })

label=train.label
train.drop('label',axis=1,inplace=True)

#import the test data set
test=pd.read_csv(r'C:\Users\user-pc\Documents\Python\MLtest.csv')
tripid=test.tripid
data=train.append(test)

print(data.head())

print(data.describe())

print(data.isnull().sum())

print(data.head())

data.drop('tripid',inplace=True,axis=1)

print(data.isnull().sum())

#Stats of the data
print('No of data in Original dataset = ', data.shape)


train_X=data.iloc[:17176,]
train_y=label
X_test=data.iloc[17176:,]
seed=8

from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(train_X,train_y,test_size=0.40, random_state=seed)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=8)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn_lvq import GlvqModel

import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

models=[]
models.append(("logreg",LogisticRegression()))
models.append(("tree",DecisionTreeClassifier()))
models.append(("lda",LinearDiscriminantAnalysis()))
models.append(("svc",SVC()))
models.append(("knn",KNeighborsClassifier()))
models.append(("nb",GaussianNB()))
models.append(("rm",RandomForestClassifier()))

scoring='accuracy'
seed=8

from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
result=[]
names=[]

for name,model in models:
    #print(model)
    kfold=KFold(n_splits=10)
    cv_result=cross_val_score(model,train_X,train_y,cv=kfold,scoring=scoring)
    result.append(cv_result)
    names.append(name)
    print("%s %f %f" % (name,cv_result.mean(),cv_result.std()))


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

random.seed(100)
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(train_X,train_y)
pred = rf.predict(test_X)

print('Accuracy of RM model=', accuracy_score(test_y,pred))
print(confusion_matrix(test_y,pred))
print(classification_report(test_y,pred))

df_output=pd.DataFrame()

outp=rf.predict(X_test).astype(int)
print(outp)

df_output['tripid']=tripid
df_output['prediction']=outp

#df_output['label'].replace(1, 'correct', inplace=True)
#df_output['label'].replace(0, 'incorrect', inplace=True)

print(df_output.head())

df_output[['tripid','prediction']].to_csv(r'C:\Users\user-pc\Documents\Python\result.csv',index=False)

