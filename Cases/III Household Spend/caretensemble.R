library(caretEnsemble)
trainControl <- trainControl(method = "repeatedcv",
                             number = 5,
                             index = createFolds(train$yHat, 5),
                             repeats = 3,
                             savePredictions = "all",
                             #classProbs=TRUE,
                             search = "random")
tgrid <- expand.grid(
  .mtry = c(seq(from = 33, to = 39, by =2)),
  .min.node.size = c(7, 11),
  .splitrule = "variance"
)
models <- caretList(
  yHat ~ .,data  = train,
  trControl=trainControl,
  metric="RMSE",
  methodList=c('ranger', 'gbm'),
  tuneList=list(
    rf1=caretModelSpec(method="rpart", 
                       tuneGrid = expand.grid(cp = c(0.0001, 0.0005)) ),
    rf2=caretModelSpec(method="ranger",
                       tuneGrid = tgrid,
                       num.trees = 300,
                       importance = "permutation"),
    gbm=caretModelSpec(method="gbm",
                       tuneGrid = expand.grid(interaction.depth = c(1, 3, 6, 9, 10),
                                              n.trees = (0:50)*50, 
                                              shrinkage = seq(.0005, .05,.0005),
                                              n.minobsinnode = 10),
                       num.trees = 300,
                       importance = "permutation"),
    nn=caretModelSpec(method="ctree2", 
                      tuneGrid = expand.grid(maxdepth = c(20:30), mincriterion = c(seq(from = 0.5, to = .99, by =0.1)))
    )
  )
)

#models <- caretList(yHat ~ .,data  = train,methodList=c('rpart','ranger', 'ctree2'),trControl=trainControl)
model <- caretEnsemble(models, metric="RMSE", trControl=trainControl)
summary(model)

trainPreds      <- predict(model, train)
validationPreds <- predict(model, validation)
testingPreds    <- predict(model, testing)

#checking the results for Testing data
rmse <- MLmetrics::RMSE(testingPreds,test_y)
mae <- MLmetrics::MAE(testingPreds,test_y)
r2 <- MLmetrics::R2_Score(testingPreds,test_y)
message("Random Forest - Ranger - Testing (mtry=", mtry, ", ntree=", ntrees,    "):")
message("  RMSE: ", rmse)
message("  MAE: ", mae)
message("  R2: ", r2)

caretModelList<-modelLookup()

varImp(model)

plot(model)

print(model)

caretModelList<-modelLookup()

print(caretModelList[caretModelList$forReg==TRUE,])

