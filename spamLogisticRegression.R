library(glmnet)
library(ROCR)
library(ggplot2)

load("/Users/chance/Documents/Maestría/UT/DataMining/Assignment3/spam.Rdata")

scaled.spam <- scale(spam[,1:57])

scaledFit.spam <- cv.glmnet(x=scaled.spam, y=spam$spam, alpha=0, family="binomial")
summary(scaledFit)

logTransform <- function(x) {
	result <- log(x + 0.2)
	return(result)
}

transform.spam <- apply(spam[, 1:57], 2, logTransform)

transformFit.spam <- cv.glmnet(x=transform.spam, y=spam$spam, alpha=0, family="binomial")
summary(transformFit)

binaryTransform <- function(x) {
	if (x > 0) {
		return(1)
	} else {
		return(0)
	}
}

binary.spam <- apply(spam[, 1:57], 2, binaryTransform)

binaryFit.spam <- cv.glmnet(binary.spam, spam$spam, alpha=0, family="binomial")

scaled.newx = data.matrix(scaled.spam)
transform.newx = data.matrix(transform.spam)
binary.newx = data.matrix(binary.spam)

scaled.pre = predict(scaledFit.spam, newx=scaled.newx, type="response")
transform.pre = predict(transformFit.spam, newx=transform.newx, type="response")
binary.pre = predict(binaryFit.spam, newx=binary.newx)

scaled.pred <- prediction(scaled.pre, spam$spam)
transform.pred <- prediction(transform.pre, spam$spam)
binary.pred <- prediction(binary.pre, spam$spam)

scaled.roc.perf <- performance(scaled.pred, "tpr", "fpr")
scaled.tpr.points <- attr(scaled.roc.perf, "x.values")[[1]]
scaled.fpr.points <- attr(scaled.roc.perf, "y.values")[[1]]
plot(scaled.roc.perf, main="Scaled ROC Performance")
plot(scaled.tpr.points, main="Scaled TPR Points")
plot(scaled.fpr.points, main="Scaled FPR Points")

scaled.auc <- attr(performance(scaled.pred, "auc"), "y.values")[[1]]
plot(scaled.auc, main="Scaled AUC")

scaled.lift.perf <- performance(scaled.pred, "lift", "rpp")
plot(scaled.lift.perf, main="Scaled Lift Performance")

transform.roc.perf <- performance(transform.pred, "tpr", "fpr")
transform.tpr.points <- attr(transform.roc.perf, "x.values")[[1]]
transform.fpr.points <- attr(transform.roc.perf, "y.values")[[1]]
plot(transform.roc.perf, main="Log Transform ROC Performance")
plot(transform.tpr.points, main="Log Transform TPR Points")
plot(transform.fpr.points, main="Log Transform FPR Points")

transform.auc <- attr(performance(transform.pred, "auc"), "y.values")[[1]]
plot(transform.auc, main="Log Transform AUC")

transform.lift.perf <- performance(transform.pred, "lift", "rpp")
plot(transform.lift.perf, main="Log Transform Lift Performance")