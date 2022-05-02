library(caret)
library(skimr)
skimmed <- skim(trainingTables)
skimmed[, c(1:5, 9:11, 13, 15:16)]

subsets <- c(1:5, 10, 15, 18)

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

lmProfile <- rfe(x=train_x, y=train_y,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile


modelnames <- paste(names(getModelInfo()), collapse=',  ')
modelnames

set.seed(100)

# Train the model using randomForest and predict on the training data itself.
model_mars = train(train_y ~ ., data=train, method='earth')
fitted <- predict(model_mars)
model_mars

plot(model_mars, main="Model Accuracies with MARS")

varimp_mars <- varImp(model_mars)
plot(varimp_mars, main="Variable Importance with MARS")


library(caretEnsemble)

# Stacking Algorithms - Run multiple algos in one call.
trainControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE)

algorithmList <- c('rf', 'adaboost', 'earth', 'xgbDART', 'svmRadial','gbm','glmboost')

set.seed(100)
models <- caretList(train_y ~ ., data=train, trControl=trainControl, methodList=algorithmList) 
results <- resamples(models)
summary(results)

scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)




# Create the trainControl
set.seed(101)
stackControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE, 
                             classProbs=TRUE)

# Ensemble the predictions of `models` to form a new combined prediction based on glm
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)


