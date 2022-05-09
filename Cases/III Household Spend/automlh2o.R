library(h2o)
h2o.init()
#importing files

training<-h2o.importFile(path="C:/Harvard/Harvard_DataMining_Business_Student/Cases/III Household Spend/studentTables/training.csv")
testing<-h2o.importFile(path="C:/Harvard/Harvard_DataMining_Business_Student/Cases/III Household Spend/studentTables/testing.csv")
h2o.head(training)
h2o.describe(training)

# Plot a histogram for the Regression problem target column
h2o.hist(training[, c("yHat")])

# Split your data into 3 and save into variable "splits"
splits <- h2o.splitFrame(training, c(0.8), seed = 42)

# Extract all 3 splits into train, valid and test
train <- splits[[1]]
validation  <- splits[[2]]
h2o.head(train)
h2o.head(validation)

# Identify predictors and response for the regression use case
ignore_reg <- c("lat","lon","yHat","FirstName","LastName","TelephonesFullPhone","tmpID", "stateFips",
                "fips")

y_reg <- "yHat"

x_reg <- setdiff(colnames(train), ignore_reg)
x_reg

#Run AutoML for UPTO 30 minutes
aml_reg <- h2o.automl(x = x_reg,
                      y = y_reg,
                      training_frame = train,
                      max_runtime_secs = 1800,
                      stopping_metric = 'RMSE',
                      seed = 42,
                      project_name = "regression",
                      sort_metric = 'RMSE')

# View the AutoML Leaderboard
lb <- h2o.get_leaderboard(aml_reg)
h2o.head(lb, n = -1)

# Get model ids for all models in the AutoML Leaderboard
model_ids <- as.data.frame(aml_reg@leaderboard$model_id)[,1]
# Get top GBM from leaderboard
gbm <- h2o.getModel(grep("GBM", model_ids, value = TRUE)[1])

# Retrieve specific parameters from the obtained model
gbm@allparameters[["ntrees"]]
gbm@allparameters[["max_depth"]]
gbm@allparameters[["learn_rate"]]
gbm@allparameters[["sample_rate"]]

# Print the results for the model
gbm

# Save the validation performance of the model
gbm_validation_perf <- h2o.performance(gbm, validation)
# Print the validation RMSE
h2o.rmse(gbm_validation_perf)

# Save the test performance of the model
gbm_test_perf <- h2o.performance(gbm, testing)
# Print the test RMSE
h2o.rmse(gbm_test_perf)


#deep learning model

dl <- h2o.deeplearning(x = x_reg, 
                       y = y_reg, 
                       training_frame = train, 
                       model_id = "default_dl",
                       validation_frame = validation, 
                       seed = 42)

dl
# Plot the scoring history for the DL model
plot(dl)

# Retrieve the number of epochs and hidden layers of the model
dl@allparameters[["epochs"]]

dl.varimp_plot()

# Save the validation model performance
valid_def_dl_perf <- h2o.performance(dl, validation)
test_def_dl_perf<-h2o.performance(dl,testing)
print(test_def_dl_perf)

#tune the DL model with gridsearch
# Do a Grid Search to tune the hidden layers and the droput ratio
dl_hidden_grid <- h2o.grid(algorithm = "deeplearning",
                           grid_id = "dl_hidden_grid",
                           activation = "RectifierWithDropout",
                           epochs = 10,
                           seed = 42,
                           stopping_rounds = 3,
                           stopping_metric ="RMSE",
                           stopping_tolerance = 1e-3,
                           x = x_reg,
                           y = y_reg,
                           training_frame = train,
                           validation_frame = validation,
                           hyper_params = list(
                             hidden = list(c(100, 100), c(165, 165), c(200, 200), c(330, 330),
                                           c(165, 200)),
                             hidden_dropout_ratios = list(c(0,0), c(0.01,0.01), c(0.15,0.15),
                                                          c(0.30, 0.30), c(0.5,0.5))),
                           search_criteria = list(
                             strategy = "RandomDiscrete",
                             max_runtime_secs = 1800,
                             seed = 42))

# Retrieve the Grid Search
dl_hidden_grid_rmse <- h2o.getGrid(grid_id = "dl_hidden_grid", sort_by = "rmse", decreasing = FALSE)
as.data.frame(dl_hidden_grid_rmse@summary_table)

top_dl <- h2o.getModel(dl_hidden_grid_rmse@model_ids[[1]]) #getting the best model

print(top_dl)
# Grid Search to tune the Max W2 and L2
dl_random_grid <- h2o.grid(algorithm = "deeplearning", 
                                grid_id = "dl_random_grid",
                                activation = "RectifierWithDropout",
                                hidden = top_dl@allparameters[["hidden"]],
                                epochs = 10,
                                seed = 42,
                                hidden_dropout_ratios = top_dl@allparameters[["hidden_dropout_ratios"]],
                                stopping_rounds = 3,
                                stopping_metric = "RMSE",
                                stopping_tolerance = 1e-3,
                                x = x_reg,
                                y = y_reg,
                                training_frame = train,
                                validation_frame = validation,
                                hyper_params = list(
                                  max_w2 = c(1e38, 1e35, 1e36, 1e37, 1e34, 5e35),
                                  l2 = c(1e-7, 1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 0)),
                                search_criteria = list(
                                  strategy = "RandomDiscrete",
                                  max_runtime_secs = 900,
                                  seed = 42))

# Retrieve the Grid Search
dl_random_grid_rmse <- h2o.getGrid(grid_id = "dl_random_grid", sort_by = "rmse", decreasing = FALSE)
as.data.frame(dl_random_grid_rmse@summary_table)

# Retrieve the best model from the Grid Search
tuned_dl <- h2o.getModel(dl_random_grid_rmse@model_ids[[1]]) #getting the best model
print(tuned_dl)
# Checkpointing for DL model to increase the number of epochs
dl_checkpoint <- h2o.deeplearning(x = x_reg,
                                  y = y_reg,
                                  training_frame = train,
                                  model_id = "dl_checkpoint",
                                  validation_frame = validation,
                                  checkpoint = dl_random_grid_rmse@model_ids[[1]],
                                  activation = "RectifierWithDropout",
                                  hidden = tuned_dl@allparameters[["hidden"]],
                                  epochs = 400,
                                  seed = 42,
                                  hidden_dropout_ratios = tuned_dl@allparameters[["hidden_dropout_ratios"]],
                                  l2 = tuned_dl@parameters$l2,
                                  max_w2 = tuned_dl@parameters$max_w2,
                                  reproducible = TRUE,
                                  stopping_rounds = 5,
                                  stopping_metric = "RMSE",
                                  stopping_tolerance= 1e-5)



# Save the validation performance of the best DL model
valid_tuned_dl_perf <- h2o.performance(dl_checkpoint, validation)
# Compare the RMSE of the default and tuned DL model
h2o.rmse(valid_tuned_dl_perf)

test_tuned_dl_perf<-h2o.performance(dl_checkpoint,testing)
h2o.rmse(test_tuned_dl_perf)

