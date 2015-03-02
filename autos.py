from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.cross_validation import LeaveOneLabelOut
import matplotlib.pyplot as plt
import numpy as np

x = np.genfromtxt("autos.csv", delimiter=",", skip_header=1, usecols=range(1,7))
y = np.genfromtxt("autos.csv", delimiter=",", skip_header=1, usecols=(0))
folds = np.genfromtxt("auto.folds")

ridgeAlphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
lassoAlphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

cv = LeaveOneLabelOut(folds)
ridgeModel = RidgeCV(cv=cv, alphas=ridgeAlphas)
ridgeFit = ridgeModel.fit(x,y)
lassoModel = LassoCV(cv=cv, alphas=lassoAlphas)
lassoFit = lassoModel.fit(x,y)

print("Ridge Alpha")
print(ridgeFit.alpha_)
print("Ridge Coefs")
print(ridgeFit.coef_)

print("Lasso Alpha")
print(lassoFit.alpha_)
print("Lasso Coefs")
print(lassoFit.coef_)
print("Lasso MSE")
print(lassoFit.mse_path_)