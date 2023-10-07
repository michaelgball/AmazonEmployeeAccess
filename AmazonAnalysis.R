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

 my_recipe <- recipe(ACTION~., data=amazonTrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_nominal_predictors(), threshold = .01) %>% # combines categorical values that occur <5% into an "other" value
  step_dummy(all_nominal_predictors()) # dummy variable encoding
prep <- prep(my_recipe)
baked <- bake(prep, new_data = NULL)
