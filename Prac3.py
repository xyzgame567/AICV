# WAp working with Unsupervise
#we have customer data wirth anual income and spending score we want to group customer into cluster kmean

import numpy as np 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#Features ; Anual Income and spending Score

x= np.array([[15,39],[16,81],[17,6],[18,77],[19,14],[20,80]])

kmean = KMeans(n_clusters=2)
kmean.fit(x)

label = kmean.labels_
print(label)

plt.scatter(x[:,0],x[:,1], c=label, cmap='rainbow')
plt.scatter(kmean.cluster_centers_[:,0], kmean.cluster_centers_[:,1], color='black', marker='X', label="centroid")

plt.xlabel("Anual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
