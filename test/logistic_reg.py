#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score,classification_report
#from sklearn.datasets import load_iris
#from sklearn.model_selection import train_test_split

#iris=load_iris()
#X=iris.data
#y=(iris.target==0).astype(int)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model=LogisticRegression()
#model.fit(X_train,y_train)

#y_pred=model.predict(X_test)

#print(accuracy_score(y_test,y_pred))
#print(classification_report(y_test,y_pred))

print("decision tree regression 7.2")
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
iris=load_iris()
X=iris.data
y=iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model=DecisionTreeClassifier()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
plt.figure(figsize=(10,6))
tree.plot_tree(model,feature_names=iris.feature_names,class_names=iris.target_names,filled=True)
plt.show()