import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
data = pd.read_csv("heart.csv")

data = data[["age", "cigsPerDay", "currentSmoker", "diabetes", "BMI", "heartRate"]]
data = data.apply(pd.to_numeric, errors='coerce')
data = data.fillna(data.mean())

predict = "cigsPerDay"

X = np.array(data.drop([predict], 1)) # Features
y = np.array(data[predict]) # Labels
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)

best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
    with open("heartrate.pickle", "wb") as f:
        pickle.dump(linear, f)

pickle_in = open("heartrate.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_) # These are each slope value
print('Intercept: \n', linear.intercept_) # This is the intercept

predictions = linear.predict(x_test) # Gets a list of all predictions

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'heartRate'
style.use("ggplot")
pyplot.scatter(data[p],data["cigsPerDay"])
pyplot.xlabel(p)
pyplot.ylabel("Cigs")
pyplot.show()