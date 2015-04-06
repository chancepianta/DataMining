library(e1071)

load("/home/vmadmin/Descargas/spam.Rdata")

train = subset(spam, test == FALSE)$spam
test = subset(spam, test == TRUE)$spam

# Naive Bayes Classifier - Binarized Spam Data
binaryClassifier <- naiveBayes(train, train)
binaryPred <- predict(binaryClassifier, test)
binaryPredTable = table(binaryPred, test)

binaryClassifier
binaryPredTable

CrossTable(binaryPred, test, prop.chisq = FALSE, prop.t = FALSE, dnn = c('predicted', 'actual'))

# Naive Bayes Classifier - Real Valued Attributes
