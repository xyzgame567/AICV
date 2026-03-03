# Prac1 a,b,c,d
# linear , losgistic , decision tree , kneighbours

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#features
x= np.array([1000,1500,2000,2500,3000]).reshape(-1,1)

y= np.array([150000,200000,250000,300000,350000])
model= LinearRegression()
model.fit(x,y)

print("intercept",model.intercept_)
print("slope",model.coef_)

predicted_price = model.predict([[2200]])
print(predicted_price)

plt.scatter(x,y, color='blue', label='Actual Data')
plt.plot(x, model.predict(x), color='red', label='Regression Line')
plt.scatter(2200, predicted_price, color='green', label='Predicted Price', marker='X')
plt.xlabel('Size of the house (sq ft)')
plt.ylabel('Price of the house')    
plt.legend()
plt.show()
