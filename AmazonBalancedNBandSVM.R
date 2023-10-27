library(tidyverse)
library(vroom)
library(tidymodels)
library(embed)


#Read in data
amazonTrain <- vroom("./train.csv")
amazonTest <- vroom("./test.csv")

amazonTrain$ACTION <- as.factor(amazonTrain$ACTION)

library(themis)

#Recipe
my_recipe <- recipe(ACTION~., amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors5
  #step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_nominal_predictors()) %>%
  step_smote(all_outcomes(), neighbors=5)
prep <- prep(my_recipe)

##Naive Bayes
library(tidymodels)
library(discrim)
library(naivebayes)

## nb model
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") 

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model) 

## Tune smoothness and Laplace here
tuning_grid <- grid_regular(Laplace(),smoothness(),levels = 4)
## Split data for CV
folds <- vfold_cv(amazonTrain, v = 5, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,grid=tuning_grid,metrics=metric_set(roc_auc)) 

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

final_wf <-nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazonTrain)

## Predict
nb_preds <- predict(final_wf, new_data=amazonTest, type="prob") %>%
  bind_cols(., amazonTest) %>%
  select(id, .pred_1) %>%
  rename(Action=.pred_1) %>%
  rename(Id=id)

vroom_write(x=nb_preds, file="./NB_Preds.csv", delim=",")

##KNN
library(kknn)
knn_model <- nearest_neighbor(neighbors=tune()) %>% 
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

## Tune smoothness and Laplace here
tuning_grid <- grid_regular(neighbors(),levels = 3)
## Split data for CV
folds <- vfold_cv(amazonTrain, v = 5, repeats=1)

## Run the CV
CV_results <- knn_wf %>%
  tune_grid(resamples=folds,grid=tuning_grid,metrics=metric_set(roc_auc)) 

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

final_wf <-knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazonTrain)


## Predict
knn_preds <- predict(final_wf, new_data=amazonTest, type="prob") %>%
  bind_cols(., amazonTest) %>%
  select(id, .pred_1) %>%
  rename(Action=.pred_1) %>%
  rename(Id=id)

vroom_write(x=knn_preds, file="./KNN_Preds.csv", delim=",")

##SVM
library(kernlab)

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)

## Tune parameters
tuning_grid <- grid_regular(rbf_sigma(), cost(),levels = 3)
## Split data for CV
folds <- vfold_cv(amazonTrain, v = 5, repeats=1)

## Run the CV
CV_results <- svm_wf %>%
  tune_grid(resamples=folds,grid=tuning_grid,metrics=metric_set(roc_auc)) 

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("roc_auc")

final_wf <-svm_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=amazonTrain)

svm_preds <- predict(final_wf, new_data=amazonTest, type="prob") %>%
  bind_cols(., amazonTest) %>%
  select(id, .pred_1) %>%
  rename(Action=.pred_1) %>%
  rename(Id=id)

vroom_write(x=svm_preds, file="./SVM_Preds.csv", delim=",")
