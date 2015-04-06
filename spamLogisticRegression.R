library(glmnet)

load("/home/chance/Documents/spam.Rdata")

train = subset(spam, test == FALSE)
test = subset(spam, test == TRUE)

train$test <- NULL
test$test <- NULL

scaled.train <- scale(train[,1:57])
scaled.test <- scale(test[,1:57])

fit.train <- cv.glmnet(scaled.train, train$spam, family="multinomial")
