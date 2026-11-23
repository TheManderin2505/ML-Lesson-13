#clasification

import pandas as pd

data = pd.read_csv(r"C:\Users\alexd\Desktop\Machine Learning\Lesson 13\WineQT.csv")

print(data.head())
print(data.info())
y = data["quality"]

data = data.drop("quality",axis = 1)
x = data

print(data.head())

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=5)

from sklearn import svm
model = svm.SVC(kernel="linear")

model.fit(xtrain,ytrain)

ypredict = model.predict(xtest)

from sklearn.metrics import accuracy_score,confusion_matrix

acuracy = accuracy_score(ytest,ypredict)
acuracy = round(acuracy*100,2)

print("Accuracy : ",acuracy,"%")

import matplotlib.pyplot as plt
import seaborn as sb

matrix = confusion_matrix(ytest,ypredict)
sb.heatmap(matrix,annot=True,fmt="d")

plt.title("Confusion matrix")
plt.xlabel("Prediction")
plt.ylabel("Accuracy")
plt.show()







print("END")