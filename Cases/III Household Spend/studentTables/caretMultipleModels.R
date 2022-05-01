library(caret)

tgrid <- expand.grid(
  .mtry = 2:10,
  .min.node.size = c(10, 20),
  .splitrule = "variance"
)

model_caret <- train(yHat  ~ ., data = train,
                     method = "ranger",
                     trControl = trainControl(method="cv", number = 5, verboseIter = T, classProbs = F),
                     tuneGrid = tgrid,
                     num.trees = 100,
                     importance = "permutation")


varImp(model_caret)

plot(model_caret)

print(model_caret)



#print(predictorrf)
trainPreds      <- predict(model_caret, train)
validationPreds <- predict(model_caret, validation)
testingPreds    <- predict(model_caret, testing)
#Evaluating performance on testing data
rmse <- MLmetrics::RMSE(testingPreds,testing$yHat)
mae <- MLmetrics::MAE(testingPreds,testing$yHat)
r2 <- MLmetrics::R2_Score(testingPreds,testing$yHat)


#Decision Tree

tgrid <- expand.grid(
 .cp = c(0.001, 0.002, 0.003, 0.004, 0.05, 0.06, 0.7)
)

model_caret <- train(yHat  ~ ., data = train,
                     method = "rpart",
                     trControl = trainControl(method="cv", number = 5, verboseIter = T, classProbs = F),
                     tuneGrid = tgrid)


varImp(model_caret)

plot(model_caret)

print(model_caret)



#print(predictorrf)
trainPreds      <- predict(model_caret, train)
validationPreds <- predict(model_caret, validation)
testingPreds    <- predict(model_caret, testing)
#Evaluating performance on training data
rmse <- MLmetrics::RMSE(testingPreds,testing$yHat)
mae <- MLmetrics::MAE(testingPreds,testing$yHat)
r2 <- MLmetrics::R2_Score(testingPreds,testing$yHat)


#knn
tgrid <- expand.grid(k = c(5,7,10,15,19,21))

#grid = expand.grid(k = c(5,7,10,15,19,21))

model_caret <- train(yHat  ~ ., data = train,
                     method = "knn",
                     trControl = trainControl(method="cv", number = 5, verboseIter = T, classProbs = F),
                     tuneGrid = tgrid)


varImp(model_caret)

plot(model_caret)

print(model_caret)



#print(predictorrf)
trainPreds      <- predict(model_caret, train)
validationPreds <- predict(model_caret, validation)
testingPreds    <- predict(model_caret, testing)
#Evaluating performance on testing data
rmse <- MLmetrics::RMSE(testingPreds,testing$yHat)
mae <- MLmetrics::MAE(testingPreds,testing$yHat)
r2 <- MLmetrics::R2_Score(testingPreds,testing$yHat)
