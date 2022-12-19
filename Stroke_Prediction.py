
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datae = pd.read_csv(r'D:\Github\Stroke_Prediction\stroke_data.csv')
datae.columns
datae.isnull().sum()
d = datae.dropna(axis = 'index')
d =d.drop(columns = ['id',])
 #removing bmi outliers  
for i in d.index:
    if d.bmi[i] < 15:
        d = d.drop(index = i)
    elif d.bmi[i] >50:
        d =d.drop(index = i)
#One hot encoding 
d = pd.get_dummies(d)
X = d.drop(columns = 'stroke')
y = d.stroke
#Mean values of all features for stroke suffered and no stroke suffered
import seaborn as sn
X_stroke = d[d['stroke'] == 1].describe().T
X_no_stroke = d[d['stroke'] == 0].describe().T
fig, ax= plt.subplots(nrows=1, ncols=2, figsize = (15,7))
plt.subplot(1,2,1)
sn.heatmap(X_stroke[['mean']], annot=True,cmap = 'coolwarm', linecolor='black',linewidths=0.4)
plt.title('Stroke Suffered')
plt.subplot(1,2,2)
sn.heatmap(X_no_stroke[['mean']], annot=True,cmap = 'coolwarm',linecolor='black',linewidths=0.4)
plt.title('No Stroke Suffered')
#Build model for data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=500)
model.fit(X_train,y_train)
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv = ShuffleSplit(n_splits=5, test_size=0.2)
cross_val_score(LogisticRegression(), X, y, cv=cv)
#Maximum score by Logistic Regression is 96.3%
from sklearn import tree
modeld = tree.DecisionTreeClassifier()
modeld.fit(X_train,y_train)
model.score(X_test,y_test)
cross_val_score(tree.DecisionTreeClassifier(), X, y, cv =cv)
#Maximum score by Decision Tree is 92.5%
from sklearn.ensemble import RandomForestClassifier
modelr = RandomForestClassifier(n_estimators=50)
modelr.fit(X_train,y_train)
modelr.score(X_test,y_test)
cross_val_score(RandomForestClassifier(n_estimators=50), X, y, cv = cv)
#Maximum score by Random Forest is 95.9%
y_predicted = modelr.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
plt.figure(figsize=(10,8))
sn.heatmap(cm, annot=True)







    
