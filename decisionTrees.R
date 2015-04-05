library(rpart)
library(rpart.plot)

breastCancerData <- read.csv("/home/chance/Downloads/BreastCancer.csv", header=T, sep=",")

giniParms = list(split="gini")
entropyParms = list(split="information")

control = rpart.control(maxdepth=2)

formula = (diagnosis ~ mean.radius + mean.texture + mean.perimeter + mean.area + mean.smoothness + mean.compactness + mean.concavity + mean.concave.points + mean.symmetry + mean.fractal.dim + se.radius + se.texture + se.perimeter + se.area + se.smoothness + se.compactness + se.concavity + se.concave.points + se.symmetry + se.fractal.dim + worst.radius + worst.texture + worst.perimeter + worst.area + worst.smoothness + worst.compactness + worst.concavity + worst.concave.points + worst.symmetry + worst.fractal.dim)

giniFit <- rpart(formula, method="class", data=breastCancerData, parms=giniParms, control=control)
entropyFit <- rpart(formula, method="class", data=breastCancerData, parms=entropyParms, control=control)

summary(giniFit)
summary(entropyFit)


rpart.plot(giniFit, type=4, extra=1, main="Gini Decision Tree")
rpart.plot(entropyFit, type=4, extra=1, main="Entropy Decision Tree")