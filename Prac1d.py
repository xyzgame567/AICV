import numpy as np
from sklearn.neighbors import KNeighborsClassifier

x=np.array([[2,5],[4,6],[5,7],[1,4],[3,5],[6,8]])
y=np.array([0,1,1,0,0,1])

model=KNeighborsClassifier(n_neighbors=3)
model.fit(x,y)

prediction=model.predict([[4,5]])
probability=model.predict_proba([[4,5]])

print("Prediction",prediction)
print("Probability",probability)