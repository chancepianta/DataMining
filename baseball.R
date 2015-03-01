baseball <- read.csv(file="RunsPerGame.txt", header=TRUE, sep="\t")
summary(baseball)

lmAVG <- lm(baseball$R ~ baseball$AVG)
lmSLG <- lm(baseball$R ~ baseball$SLG)
lmOBP <- lm(baseball$R ~ baseball$OBP)

summary(lmAVG)
plot(baseball$AVG, baseball$R)
abline(lmAVG)

layout(matrix(1:4, 2, 2))
plot(lmAVG)

summary(lmSLG)
plot(baseball$SLG, baseball$R)
abline(lmSLG)

layout(matrix(1:4, 2, 2))
plot(lmSLG)

summary(lmOBP)
plot(baseball$OBP, baseball$R)
abline(lmOBP)

layout(matrix(1:4, 2, 2))
plot(lmOBP)