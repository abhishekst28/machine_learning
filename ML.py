# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:22:19 2019

@author: toshn
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
df =pd.read_csv('weatherAUS.csv')
df.dropna(inplace=True)
df.describe()
lb_make = LabelEncoder()
df["Area_code"] = lb_make.fit_transform(df["Location"])
df = df[['Date', 'Location', 'Area_code', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RISK_MM', 'RainTomorrow']]
df.head()
#print(list(df.columns.values))
print(df.describe())
df=df.drop(columns=['RISK_MM','WindDir9am','WindDir3pm','WindGustDir'])
#df=pd.get_dummies(df, columns=["WindGustDir", "WindDir9am","WindDir3pm"], prefix=["GustDir", "Dir9am","Dir3pm"])

x=df.iloc[:,3:19].values
#x=df.iloc[:,21:68].values
y=df.iloc[:,[20]].values
sc=StandardScaler()
x=sc.fit_transform(x)
print(x)
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)
#clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)
c = DecisionTreeClassifier(criterion = "entropy")
c.fit(xTrain, yTrain)
y_pred = c.predict(xTest) 
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(yTest, y_pred))  
print(classification_report(yTest, y_pred))  
from sklearn.metrics import accuracy_score
print(accuracy_score(yTest,y_pred))

dot_data = StringIO()
export_graphviz(c, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
a=Image(graph.create_png())
display(a)


print(clf_entropy)
accu_train=np.sum(clf_entropy.predict(xTrain) == yTrain)/float(yTrain.size())
accu_test=np.sum(clf_entropy.predict(xTset) == yTest)/float(yTest.size())
print (accu_train)
print (accu_test)