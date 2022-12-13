import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = np.loadtxt('house_price.txt', delimiter=',')

X = data[:, 0]
y = data[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train.reshape(-1, 1), y_train)
y_pred = model.predict(X_test.reshape(-1, 1))
plt.scatter(X_train, y_train,color='y')
plt.plot(X_test, y_pred,color='r')
plt.xlabel('size')
plt.ylabel('price')
plt.title('House Dataset')
plt.show()


