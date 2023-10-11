library(tidyverse)
library(vroom)
library(tidymodels)
library(embed)
library(ggmosaic)

#Read in data
amazonTrain <- vroom("./train.csv")
amazonTest <- vroom("./test.csv")

##EDA
ggplot(amazonTrain, aes(RESOURCE, ACTION))+geom_point()
ggplot(amazonTrain, aes(MGR_ID, ACTION))+geom_point()

amazonTrain$ACTION <- as.factor(amazonTrain$ACTION)

my_recipe <- recipe(ACTION~., data=amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%# turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" value
  step_dummy(all_nominal_predictors())
prep <- prep(my_recipe)
baked <- bake


##Logistic Regression
logit_mod <- logistic_reg() %>% #Type of model
  set_engine("glm")

logit_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(logit_mod) %>%
fit(data = amazonTrain)

logit_preds <- predict(logit_wf,new_data=amazonTest,type="prob")%>%
  bind_cols(., amazonTest) %>%
  select(id, .pred_1) %>%
  rename(Action=.pred_1) %>%
  rename(Id=id)

## Write prediction file to CSV
vroom_write(x=logit_preds, file="./LogitPreds.csv", delim=",") 


##Penalized Logistic Regression

my_recipe <- recipe(ACTION~., amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors5
  step_other(all_nominal_predictors(), threshold = .001) %>% # combines categorical values that occur <5% into an "other" value
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) 

plogit_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

plogit_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(plogit_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),mixture(),levels = 4) 

## Split data for CV
folds <- vfold_cv(amazonTrain, v =5, repeats=1)

## Run the CV
CV_results <- plogit_wf %>%
tune_grid(resamples=folds,grid=tuning_grid,
          metrics=metric_set(roc_auc)) #Or leave metrics NULL

bestTune <- CV_results %>%
select_best("roc_auc")

## Finalize the Workflow & fit it
plogit_mod <- logistic_reg(mixture=0, penalty=.0000000001) %>% #Type of model
  set_engine("glmnet")

plogit_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(plogit_mod) %>%
  fit(amazonTrain)

## Predict
plogit_preds <- predict(plogit_wf,new_data=amazonTest,type="prob")%>%
  bind_cols(., amazonTest) %>%
  select(id, .pred_1) %>%
  rename(Action=.pred_1) %>%
  rename(Id=id)

## Write prediction file to CSV
vroom_write(x=plogit_preds, file="./PLogitPreds.csv", delim=",") 
