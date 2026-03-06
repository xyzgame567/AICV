import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import matplotlib.pyplot as plt

x=np.array([[25,0],[30,1],[45,2],[35,1],[50,0]])
y=np.array([1,0,0,0,1])

model=tree.DecisionTreeClassifier(max_depth=3,random_state=42)
model.fit(x,y)

prediction=model.predict([[30,1]])
print("Prediction",prediction)

plt.figure(figsize=(5,7))
tree.plot_tree(model,feature_names=["age","income"],class_names=["yes","no"],filled=True)

plt.show()