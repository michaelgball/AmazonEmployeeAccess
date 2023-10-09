library(tidyverse)
library(vroom)
library(tidymodels)
library(embed)
library(ggmosaic)

#Read in data
amazonTrain <- vroom("./train.csv")
amazonTest <- vroom("./test.csv")

amazonTrain$RESOURCE <- as.character(amazonTrain$RESOURCE)
ggplot(amazonTrain, aes(RESOURCE, ACTION))+geom_bar(stat="identity")

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
