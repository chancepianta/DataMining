import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

congress = np.genfromtxt("congress.csv", delimiter=",", skip_header=1, usecols=range(1,1001), dtype=int)

meanVector = np.mean(congress, axis=0)

centeredCongress = np.subtract(congress, meanVector)

congressTarget = np.genfromtxt("congress.csv", delimiter=",", skip_header=1, usecols=(1001), dtype=int)

components = [1, 2, 5, 10, 20, 50, 100, 200]

totalvariances = []
explvariances = []
for i in components:
	pcaobject = TruncatedSVD(n_components=i)
	Xt = pcaobject.fit(centeredCongress, congressTarget).transform(centeredCongress)
	variance = np.var(Xt, axis=0).sum()
	explvariances.append(np.var(Xt, axis=0).sum() / np.var(centeredCongress, axis=0).sum())
	totalvariances.append(variance)

plt.plot(components, explvariances, 'b', label='Explained')
plt.plot(components, totalvariances, 'r', label='Total')
plt.legend()
plt.show()