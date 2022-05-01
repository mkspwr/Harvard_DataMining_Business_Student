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
model_mars = train(train_y ~ ., data=train_x, method='earth')
fitted <- predict(model_mars)
