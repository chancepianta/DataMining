library(e1071)

load("/home/vmadmin/Descargas/spam.Rdata")

train = subset(spam, test == FALSE)
test = subset(spam, test == TRUE)

train$test <- NULL
test$test <- NULL

# Naive Bayes Classifier - Binarized Spam Data
binaryClassifier <- naiveBayes(spam ~ ., train)
binaryPred <- predict(binaryClassifier, test)
binaryPredTable = table(binaryPred, test$spam)

binaryClassifier
binaryPredTable

CrossTable(binaryPred, test$spam, prop.chisq = FALSE, prop.t = FALSE, dnn = c('predicted', 'actual'))

# Naive Bayes Classifier - Real Valued Attributes
scaled.test <- scale(test[,1:56])
scaled.train <- scale(train[,1:56]) 