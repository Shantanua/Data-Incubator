
# Load required libraries #

library(data.table)
library(xgboost)
library(lubridate)
library(caret)
library(Ckmeans.1d.dp)
library(DiagrammeR)

# Read the data and perform manipulations #

dataNew <- read.csv("C:/Users/shant/Downloads/NYPD_Motor_Vehicle_Collisions.csv", na.strings=c(""," ","NA"))
dataNew2 <- dataNew[,c(-1,-7:-18,-24)]
dataNew2$TIME <- period_to_seconds(hm(dataNew$TIME))
dataNew2$casualties <- 0
dataNew2$casualties <- rowSums(dataNew[,c(11,12)])
data_cas <- dataNew2[dataNew2$casualties != 0,]
data_nocas <- dataNew2[dataNew2$casualties == 0,]

# Oversampling for better predicitve power #

smp_size <- floor(0.4 * nrow(data_nocas)) 
set.seed(123)
ind <- sample(seq_len(nrow(data_nocas)), size = smp_size)
data_nocas1 <- data_nocas[ind, ]
data_comp <- rbind(data_cas, data_nocas1)
data_comp2 <- data_comp[sample(nrow(data_comp)),]

# Divide data into train & test set #

smp_size <- floor(0.6 * nrow(data_comp2)) 
set.seed(123)
ind <- sample(seq_len(nrow(data_comp2)), size = smp_size) 
train <- data_comp2[ind, ]
test <- data_comp2[-ind, ]

# Prepare data for training #

dmy <- dummyVars(" ~ .", data = train[,-16])
trainD <- data.frame(predict(dmy, newdata = train[,-16]))
dmy2 <- dummyVars(" ~ .", data = test[,-16])
testD <- data.frame(predict(dmy2, newdata = test[,-16]))
trainM <- data.matrix(trainD)
testM <- data.matrix(testD)
label <- train[,16]
obs <- test[,16]

# Making labels for classification #

label2 = label
label2[label == 0] = 0
label2[label > 0] = 1

obs2 = obs
obs2[obs == 0] = 0
obs2[obs > 0] = 1

# Perform cross validation and training #

bst <- xgb.cv(data = trainM, booster = 'gbtree', num_class = 2, silent = 0, min_child_weight = 4, subsample = 0.5, colsample_bytree = 0.5, eval_metric = 'mlogloss', label = label2, max.depth = 6, eta = 0.1, nthread = 2, nround = 100, objective = "multi:softprob", nfold = 5, missing = NaN)
bst <- xgboost(data = trainM, booster = 'gbtree', num_class = 2, silent = 0, min_child_weight = 4, subsample = 0.5, colsample_bytree = 0.5, eval_metric = 'mlogloss', label = label2, max.depth = 6, eta = 0.1, nthread = 2, nround = 100, objective = "multi:softprob", missing = NaN)

# Get predictions on training Data #

pred <- predict(bst, testM, missing = NaN)
pred2 <- matrix(pred, nrow = dim(testD)[1], byrow = TRUE)
pred3 <- data.frame(obs2, pred2)
names(pred3) <- c('Observed', 'Predicted_0', 'Predicted_1')
write.csv2(pred3, file = "C:/Users/shant/Desktop/Data Incubator/Predictions.csv")

# Determine important features #

names <- dimnames(trainM)[[2]]
importance_matrix <- xgb.importance(names, model = bst)
xgb.plot.importance(importance_matrix = importance_matrix[1:15,])


# Calculating error #

err <- mean(as.numeric(pred3[,3] > 0.5) != pred3[,1])
print(paste("test-error=", err))


# EXTRA #
# For regression #

bst <- xgb.cv(data = trainM, silent = 0, min_child_weight = 4, subsample = 0.5,colsample_bytree = 0.5, eval_metric = 'rmse', label = label, max.depth = 6, eta = 0.1, nthread = 2, nround = 100, objective = "reg:linear", nfold = 5, missing = NaN)

pred <- predict(bst, testM)

RMSE = function(m, o){
  sqrt(mean((m - o)^2))
}




