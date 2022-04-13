

Sys.setenv(MLFLOW_PYTHON_BIN="C:\\Users\\mksharma\\AppData\\Local\\Microsoft\\WindowsApps\\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\\python.exe")

library(mlflow)

treatedTrain <- training
treatedTest <- testing

control <- trainControl(method="adaptive_cv", number=9, verboseIter = TRUE,allowParallel = TRUE)
metric <- "Accuracy"

with(mlflow_start_run(), {
  mlflow_set_tag("mlflow.runName", "randomForest")
  fit <- train(as.factor(Y_AcceptedOffer) ~ ., data=treatedTrain, method="rf", metric=metric, trControl=control) 
  
  jpeg(filename="1.jpeg")
  print(plot(fit))
  dev.off()
  # Save the plot and log it as an artifact
  mlflow_log_artifact("1.jpeg") 
  accu_train <- get_accuracy(fit, treatedTrain)
  accu_test <- get_accuracy(fit, treatedTest)
  
  message("  accu_train: ", accu_train)
  message("  accu_test: ", accu_test)
  message("  splitPercent: ", splitPercent)
  message("  seed: ", seed)
  message("  device: ", getOption("device"))
  mlflow_log_metric("accu_train", accu_train)
  mlflow_log_metric("accu_test", accu_test)
  mlflow_log_param("splitPercent", splitPercent)
  mlflow_log_param("seed", seed)
  mlflow_log_param("method", "adaptive_cv")
  mlflow_log_param("number", "9")
  
})
