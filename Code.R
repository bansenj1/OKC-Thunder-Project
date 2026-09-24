# OKC Analyst Intern Project — NBA Shot Make Probability Model


# Setup 


library(readr)
library(dplyr)
library(tidyr)
library(xgboost)
library(ggplot2)
set.seed(42)


# Loading data 


train_raw <- read_csv("analyst-intern-2027-bansenj1-1ab10b6e/training.csv.gz", show_col_types = FALSE)
test_raw  <- read_csv("analyst-intern-2027-bansenj1-1ab10b6e/testing.csv.gz",  show_col_types = FALSE)
# Check Data
glimpse(train_raw)       
# 29 columns, contester2,3 & distcont2,3 have notable NA, contester4 and distcont4 all NA
summary(train_raw)        
colSums(is.na(train_raw)) # Checks for na in raw data and puts into columns
table(train_raw$shottype)
rate <- mean(train_raw$outcome)   
# Rate of shots made (.458)
log_make <- -log(rate)
log_miss <- -log(1 - rate)
(rate * log_make) + ((1 - rate) * log_miss)
# Benchmark log-loss


#Feature engineering

num_approach <- function(df) { 
  clean <- gsub("[{}]", "", df$closestdefapproach) # gets rid of {} from closestdefapproach
  split <- strsplit(clean, ",") # splits at each ,  
  matrx <- do.call(rbind, lapply(split, function(x) as.numeric(x))) # converting into numeric and putting into a matrix
  colnames(matrx) <- c("def_dist_t1", "def_dist_t075", "def_dist_t05", "def_dist_t025") # give column names based off defender distance
  as.data.frame(matrx)
}
features <- function(df) {
  col_approach <- num_approach(df) 
  df %>%
    bind_cols(col_approach) %>% # adding four columns to original
    mutate(
      outcome_num = if ("outcome" %in% names(df)) as.integer(outcome) else NA_integer_, # handles difference of trains and test sets 
      contested_num = as.integer(contested),
      three_num = as.integer(three),
      closeout_speed = (def_dist_t1 - closestdefdist) / 1.0, # estimate of how fast the defender closed gap before shot
      min_contester_dist = pmin(distcont1, distcont2, distcont3, na.rm = TRUE), # finds closest contender
      # contester4 & distcont4 entirely empty
      contesters_within_4ft = rowSums(across(c(distcont1, distcont2, distcont3), ~ !is.na(.x) & .x <= 4),na.rm = TRUE), # checks value and 4 ft or less
      shotclock_bucket = factor(case_when(shotclock <= 4 ~ "late", shotclock <= 10 ~ "mid", TRUE ~ "early")), # create categories for shotclock
      # make categorical factor 
      shottype  = factor(shottype),
      gamestate = factor(gamestate),
      month     = factor(month)
    )
}
train_features <- features(train_raw)
test_features  <- features(test_raw)
# min_contester_dist is NA when num_contesters == 0 — not missing data, means "no listed contester", back to closestdefdist as the nearest available signal
train_features <- train_features %>%
  mutate(min_contester_dist = ifelse(is.na(min_contester_dist), closestdefdist, min_contester_dist)) # replace missing vals with closestdefdist
test_features <- test_features %>%
  mutate(min_contester_dist = ifelse(is.na(min_contester_dist), closestdefdist, min_contester_dist))
# Check
glimpse(train_features)
# Make sure no NA's remain in the engineered columns
colSums(is.na(train_features[, c("def_dist_t1","def_dist_t075","def_dist_t05", "def_dist_t025","closeout_speed","min_contester_dist")]))
# No NA's so all good


# DROPPED FEATURES:
# Earlier iterations I tried adding shooter_fgpct and def_fgpct_allowed with leave-one-out FG% for the shooter and closest defender, 
# using empirical Bayes shrinkage toward the league mean (to try and handle long tail of low-attempt players: 10th percentile of
# shooter attempts was only 14). Even with heavy shrinkage (k up to 200), these features made validation log-loss Worse 
# (0.630-0.636 vs. 0.6295 without them) and caused the model to stop earlier which is a clear overfitting signal.
# A head-to-head comparison on identical train/val splits confirmed removing them gave a better score. Left out of the final model; 
# see writeup for more.


# Train/Test splitting (not the real test set)

n <- nrow(train_features)
test <- sample(seq_len(n), size = floor(0.2 * n)) # randomly select 20% of rows
train_split <- train_features[-test, ]
test_split  <- train_features[test, ]
# Train and tests splits are obatained
logloss <- function(actual, pred) {
  eps <- 1e-15 # prevents problems wtih log(0)
  pred <- pmin(pmax(pred, eps), 1 - eps)
  -mean(actual * log(pred) + (1 - actual) * log(1 - pred)) # log-loss formula
}


# Logistic regression with GLM (Generalized linear model)

glm_model <- glm( #getting outcome given predictors 
  outcome_num ~ distance + three_num + dribblesbefore + shotclock +
    closestdefdist + shooterspeed + shottype + gamestate + contested_num +
    closeout_speed + min_contester_dist + contesters_within_4ft,
  data = train_split, family = binomial() # outcome is binary
)
summary(glm_model) 
car::vif(glm_model) # calculates variance inflation factors
# flags distance/closestdefdist/min_contester_dist collinearity — discuss in writeup
glm_pred <- predict(glm_model, newdata = test_split, type = "response") # use model to predict
cat("GLM validation log-loss:", logloss(test_split$outcome_num, glm_pred), "\n") # calculate and output log-loss


# XGBoost

feature_cols <- c(
  "distance", "three_num", "dribblesbefore", "shotclock",
  "closestdefdist", "shooterspeed", "shottype", "gamestate",
  "contested_num", "closeout_speed", "min_contester_dist",
  "contesters_within_4ft"
)
make_matrix <- function(df, cols) {
  model.matrix(~ . - 1, data = df[, cols]) # converts data into numeric matrix (need for xgboost)
}
x_train <- make_matrix(train_split, feature_cols)
x_test  <- make_matrix(test_split, feature_cols)
dtrain <- xgb.DMatrix(data = x_train, label = train_split$outcome_num) # xgboost has own optimization "DMatrix"
dtest  <- xgb.DMatrix(data = x_test,  label = test_split$outcome_num)
params <- list(
  objective = "binary:logistic", # output between 0 & 1
  eval_metric = "logloss", # uses log-loss to evaulate to xgboost
  eta = 0.05, # learning rate
  max_depth = 5, # decision tree depth
  subsample = 0.8, # tree uses .8 of training data
  colsample_bytree = 0.8 # tree uses .8 of features
  # subsample and colsample_bytree used to reduce overfitting with randomness
)
xgb_fit <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,
  evals = list(train = dtrain, val = dtest), # what to evaluate performance on
  early_stopping_rounds = 25, # prevents overfitting if model isn;t improving
  print_every_n = 25
)
# best_iteration/best_ntreelimit weren't populated in this version — read the round number directly from the printed "Best iteration: [N]" line
best_nrounds <- 401 # achieved from output for best performance
# Variable importance
importance <- xgb.importance(model = xgb_fit) # stores most important variables
xgb.plot.importance(importance, top_n = 15)


# Final model: retrain on ALL training data, predict on real test set

x_train_full <- make_matrix(train_features, feature_cols)
x_test_full  <- make_matrix(test_features, feature_cols)
dtrain_full <- xgb.DMatrix(data = x_train_full, label = train_features$outcome_num)
dtest_final <- xgb.DMatrix(data = x_test_full)
# repeat xgboost but on real/full data
xgb_final <- xgb.train(
  params = params,
  data = dtrain_full,
  nrounds = best_nrounds   # round count chosen from above
)
test_pred_final <- predict(xgb_final, dtest_final) # final model with predicted vals
submission <- tibble(
  shot_id = test_features$shot_id,
  make_prob = test_pred_final
)
# Checks both are TRUE
nrow(submission) == nrow(test_raw)
all(submission$shot_id == test_raw$shot_id)