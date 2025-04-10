import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
data = fetch_california_housing()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd = SGDRegressor(learning_rate=0.01, n_iter=1000, random_state=42)
sgd.fit(X_train, y_train)
y_pred = sgd.predict(X_test)
mse = np.mean((y_pred - y_test)**2)
print(f'MSE: {mse}')