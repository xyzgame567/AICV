# Prac2 : WAP working with supervise learning

import numpy as np
from sklearn.linear_model import LogisticRegression

# Features 
x= np.array([[2],[4],[6],[8],[10]])
y= np.array([0,0,1,1,1])

#train supervise model
model=LogisticRegression()
model.fit(x,y)

# Prediction for the students who studies for 5 hrs
prediction = model.predict([[5]])
probability= model.predict_proba([[5]])

print("Prediction",prediction)
print("Probability",probability)