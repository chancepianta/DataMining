library(glmnet)
data(mtcars)

X = data.frame(disp = mtcars$disp, hp = mtcars$hp, wt = mtcars$wt, drat = mtcars$drat)
X2 = X
X2$alsohp = X$hp
Y = mtcars$mpg

scaledX = scale(X)
scaledX2 = scale(X2)
scaledY = scale(Y)

modelX = model.matrix(scaledY ~ scaledX)
modelX2 = model.matrix(scaledY ~ scaledX2)

# Ridge Plots
ridgeX = glmnet(modelX, scaledY, alpha=0)
ridgeX2 = glmnet(modelX2, scaledY, alpha=0)

plot(ridgeX, xvar="lambda", label=TRUE)
plot(ridgeX2, xvar="lambda", label=TRUE)

cvRidgeX = cv.glmnet(modelX, scaledY, alpha=0)
cvRidgeX2 = cv.glmnet(modelX2, scaledY, alpha=0)

plot(cvRidgeX, xvar="lambda", label=TRUE)
plot(cvRidgeX2, xvar="lambda", label=TRUE)

coef(cvRidgeX)
coef(cvRidgeX2)

#Lasso Plots
lassoX = glmnet(modelX, scaledY, alpha=1)
lassoX2 = glmnet(modelX2, scaledY, alpha=1)

plot(lassoX, xvar="lambda", label=TRUE)
plot(lassoX2, xvar="lambda", label=TRUE)
plot(lassoX, xvar="dev", label=TRUE)
plot(lassoX2, xvar="dev", label=TRUE)

cvLassoX = cv.glmnet(modelX, scaledY)
cvLassoX2 = cv.glmnet(modelX2, scaledY)

plot(cvLassoX)
plot(cvLassoX2)

coef(cvLassoX)
coef(cvLassoX2)