from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
y = pd.DataFrame(iris.target, columns=['Target'])

plt.figure(figsize=(12,5))
colors = np.array(['red', 'green', 'blue'])
plt.subplot(1, 2, 1)
plt.scatter(x['Sepal Length'], x['Sepal Width'], c=colors[y['Target']], s=40)
plt.title('Sepal Length vs Sepal Width')
plt.subplot(1,2,2)
plt.scatter(x['Petal Length'], x['Petal Width'], c=colors[y['Target']], s=40)
plt.title('Petal Length vs Petal Width')

# Step1. Search for optimal K

SSE = {}
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(x)
    SSE[i] = kmeans.inertia_
print (SSE)


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(list(SSE.keys()), list(SSE.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
ax.annotate('elbow', xy=(2, 150), xytext=(3, 170), arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('elbow', xy=(3, 90), xytext=(3, 170), arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()

# Step2 Fit with K=3 given SSE "optimal"

model = KMeans(n_clusters=3)
model.fit(x)

print(model.labels_)
print(iris.target)
colors = np.array(['red', 'green', 'blue'])

# Step3: Compare Actual vs Predicted 

plt.figure(figsize=(12,5))
predictedY = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)

plt.subplot(1, 2, 1)
plt.scatter(x['Petal Length'], x['Petal Width'], c=colors[y['Target']], s=40)
plt.title('Before classification')

plt.subplot(1, 2, 2)
plt.scatter(x['Petal Length'], x['Petal Width'], c=colors[predictedY], s=40)
plt.title("Model's classification")




