library(mlflow)


mlflow_set_experiment(
  experiment_name = "Run 9"
)
features = "asdfasdfcmks"
mlflow_set_experiment_tag(key="features",value=features)

features <- mlflow_param("features", 0.15, "string")
alpha <- mlflow_param("alpha", 0.15, "numeric")
lambda <- mlflow_param("lambda", 0.45, "numeric")
run_name<-mlflow_param("name","model_name","string")
mlflow_start_run() 
for (mtry in 1:10) {

  #mlflow_start_run() 
  mlflow_log_param("alpha", alpha)
  mlflow_log_param("lambda", lambda)
  mlflow_log_param("run_name", "glmnet")
  mlflow_log_param("features","sdfasdf|asdfasdf |asdfaff |asdfsadf |asdfsadfa|asdfsafd")
  
  mlflow_log_metric("rmse-testing", 0.33)
  mlflow_log_metric("r2-testing", 0.33)
  mlflow_log_metric("mae-testing", 0.33)
  mlflow_log_param("features","sdfasdf|asdfasdf |asdfaff |asdfsadf |asdfsadfa|asdfsafd")
  
 # mlflow_end_run()
  
}
mlflow_end_run()


