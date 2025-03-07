#Installing the required Libraries
if (!requireNamespace("data.table", quietly = TRUE)) {
  install.packages("data.table", repos='http://cran.us.r-project.org')
}
if (!requireNamespace("caret", quietly = TRUE)) {
  install.packages("caret", repos='http://cran.us.r-project.org')
}
if (!requireNamespace("randomForest", quietly = TRUE)) {
  install.packages("randomForest", repos='http://cran.us.r-project.org')
}
if (!requireNamespace("nnet", quietly = TRUE)) {
  install.packages("nnet", repos='http://cran.us.r-project.org')
}
if (!requireNamespace("gbm", quietly = TRUE)) {
  install.packages("gbm", repos='http://cran.us.r-project.org')
}

if (!requireNamespace("xgboost", quietly = TRUE)) {
  install.packages("xgboost", repos='http://cran.us.r-project.org')
}


#Loading the Libraries
library(data.table)
library(caret)
library(randomForest)
library(gbm)
library(xgboost)
library(nnet)

# Loading the Data
train_file_path <- 'regression_train.csv'
test_file_path <- 'regression_test.csv'

# Reading the Data
train <- fread(train_file_path)
test <- fread(test_file_path)


# Function for Cleaning the Datasets and replacing the null values 
clean_data <- function(data) {
  # Handle missing values
  data[is.na(data)] <- -999
  
  return(data)
}

# Function for Feature Engineering handling categorical and numerical Columns  
feature_engineering <- function(data, drop_target = FALSE) {
  nominal_features <- c('gender', 'income', 'doYouFeelASenseOfPurposeAndMeaningInYourLife104',
                        'howDoYouReconcileSpiritualBeliefsWithScientificOrRationalThinki',
                        'howOftenDoYouFeelSociallyConnectedWithYourPeersAndFriends',
                        'doYouHaveASupportSystemOfFriendsAndFamilyToTurnToWhenNeeded')
  
  ordinal_features <- c('whatIsYourHeightExpressItAsANumberInMetresM')
  updated_ordinal_categories <- list(c('140 - 150', '150 - 155', '155 - 160', '160 - 165', '165 - 170',
                                       '170 - 175', '175 - 180', '180 - 185', '185 - 190', '190 above'))
  
  #Performing One-hot encoding
  onehot_encoded <- model.matrix(~ . - 1, data = data[, ..nominal_features])
  data <- cbind(data, onehot_encoded)
  data[, (nominal_features) := NULL]
  
  for (i in seq_along(ordinal_features)) {
    data[[ordinal_features[i]]] <- factor(data[[ordinal_features[i]]], levels = updated_ordinal_categories[[i]])
    data[[ordinal_features[i]]] <- as.numeric(data[[ordinal_features[i]]])
  }
  
  additional_nominal_features <- c('howOftenDoYouParticipateInSocialActivitiesIncludingClubsSportsV',
                                   'doYouFeelComfortableEngagingInConversationsWithPeopleFromDiffer',
                                   'doYouFeelASenseOfPurposeAndMeaningInYourLife105')
  
  onehot_encoded_additional <- model.matrix(~ . - 1, data = data[, ..additional_nominal_features])
  data <- cbind(data, onehot_encoded_additional)
  data[, (additional_nominal_features) := NULL]
  
  if (drop_target && "happiness" %in% colnames(data)) {
    data <- data[, !names(data) %in% c("happiness"), with = FALSE]
  }
  
  return(data)
}

# Apply data cleaning and feature engineering
train <- clean_data(train)
test <- clean_data(test)

# Applying the Feature Engineering function on the train data and test data 
train_data_encoded <- feature_engineering(train)
test_data_encoded <- feature_engineering(test, drop_target = TRUE) # Excluding the target column happiness

X_train <- train_data_encoded[, !names(train_data_encoded) %in% c("happiness"), with = FALSE]
y_train <- train_data_encoded$happiness

# Applying the Normalization on the numerical features
scaler <- preProcess(X_train, method = "range")
X_train_scaled <- predict(scaler, X_train)
X_test_scaled <- predict(scaler, test_data_encoded)

# Splitting the data into training and validation sets
set.seed(42) #setting seed
trainIndex <- createDataPartition(y_train, p = .8, list = FALSE, times = 1)
X_train_full <- X_train_scaled[trainIndex, ]
X_val <- X_train_scaled[-trainIndex, ]
y_train_full <- y_train[trainIndex]
y_val <- y_train[-trainIndex]

# Training individual models on the training data with hyper parameter tuning
train_model <- function(method, X_train, y_train) {
  train(X_train, y_train, method = method, trControl = trainControl(method = "cv", number = 10), tuneLength = 5)
}

models <- list(
  lm = train_model("lm", X_train_full, y_train_full), # Linear Regression Model
  rf = train_model("rf", X_train_full, y_train_full), # Random Forest Model
  gbm = train_model("gbm", X_train_full, y_train_full), # Gradient Boosting Model
  xgb = train_model("xgbTree", X_train_full, y_train_full), # Xg Boost Model
  nn = train(
    x = X_train_full, 
    y = y_train_full, 
    method = "nnet", 
    trControl = trainControl(method = "cv", number = 10),
    tuneLength = 5,
    linout = TRUE,
    trace = FALSE
  ) # Neural Network Model
)

# Predicting and calculating accuracy score for each model
accuracy_scores <- list()
for (model_name in names(models)) {
  predictions <- predict(models[[model_name]], newdata = X_val)
  rmse <- RMSE(predictions, y_val)
  accuracy_scores[[model_name]] <- rmse
}

# Printing accuracy scores for all the trained model 
print("Accuracy scores for individual models (RMSE):")
print(accuracy_scores)

predict_model <- function(model, X) {
  predict(model, newdata = X)
}

# Stacking predictions to form a new data set for the meta-model
stacked_predictions <- function(models, X) {
  data.table(
    lr_predictions = predict_model(models$lm, X),
    rf_predictions = predict_model(models$rf, X),
    gb_predictions = predict_model(models$gbm, X),
    xgb_predictions = predict_model(models$xgb, X),
    nn_predictions = predict_model(models$nn, X)
  )
}

stacked_train_data <- stacked_predictions(models, X_train_full)
stacked_val_data <- stacked_predictions(models, X_val)

# Training the meta-model using Neural Network
meta_model <- train(
  x = stacked_train_data, 
  y = y_train_full, 
  method = "nnet", 
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 5,
  linout = TRUE,
  trace = FALSE
)

meta_predictions_val <- predict(meta_model, newdata = stacked_val_data)

# Calculate RMSE for the stacked model
rmse_stacked <- RMSE(meta_predictions_val, y_val)

print(paste("Stacked model RMSE:", rmse_stacked))

fin.mod <- meta_model

# Make predictions on the test data using the stacked model
stacked_test_data <- stacked_predictions(models, X_test_scaled)
meta_predictions_test <- predict(fin.mod, newdata = stacked_test_data)


# Put these predicted labels in a CSV file that you can use to commit to the Kaggle Leaderboard
write.csv(
  data.frame("RowIndex" = seq(1, length(meta_predictions_test)), "Prediction" = meta_predictions_test),  
  "RegressionPredictLabel.csv", 
  row.names = FALSE
)

print(paste("Submission file saved to: RegressionPredictLabel.csv"))




