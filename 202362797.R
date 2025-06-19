############ Introduction ############ 

# The following code is intended as a supplement to a dissertation for student 202362797 
# at Strathclyde University. Created to answer "What machine learning classification model works 
# best at predicting the outcome (make/miss) of a 5 foot putt in golf?"


# This code can act as a blueprint for novice coders and golfer to predict the outcome of putts.
# If you are viewing this code from Github and have suggestions to improve or update this code
# then please submit a pull request. 

# Install Libraries
# Libraries
# Dataset
# Generalized Linear Model Metric Reduction
# Outcome is a factor
# train/test split
# Function to evaluate predictions

# Models
# XGBoost Model
# Random Forest
# Support Vector Machine
# Logistic Regression

# Output & Model Evaluation
# Confusion Matrix 
# Performance Matrix
# Saving the files

##################### Start #####################

###### Install Libraries #######
# This section is only required if the libraries have not been installed before, 
# this is a crucial step and the code will not work without library installation.

packages <- c("tidyverse", "caret", "randomForest", "e1071", "xgboost", "rpart",
  "flextable", "ggplot2", "dplyr", "tidyr", "patchwork", "fromhere")

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

invisible(lapply(packages, install_if_missing))

###### Libraries ###### 
library(tidyverse)
library(caret)
library(randomForest)
library(e1071)
library(xgboost)
library(rpart)
library(flextable)
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(fromhere)
library(here)

###### Dataset ###### 
data <- read.csv(here("GitHub/MScModelPerformance/Data.csv"))

###### End of data upload ###### 

# Now that the dat is uploaded we need to check for missing fields

###### Checking for missing data ###### 

Empty <- is.na(data)
head(Empty)

cat("Total missing data:","\n", sum(is.na(Empty)|Empty ==""),"\n")


###### End of checking for missing data ###### 

# No missing data. Looking at significance testing and reducing the variables

####### Generalized Linear Model Metric Reduction ###### 

model <- glm(Outcome ~ ., data = data, family = binomial)
model1 <- glm(Outcome ~ ISS + FAI + LoftCh + LieCh + BSL + ForRot, data = data, family = binomial)
model2 <- glm(Outcome ~ ISS + FAI + LoftCh + LieCh, data = data, family = binomial)

summary(model)
summary(model1)
summary(model2)

####### End of Metric Reduction ###### 

# Model 2 provides the best results, now change the outcome to factor
# Splitting the dataset to train and test

###### Change Outcome as a factor ###### 
data$Outcome <- as.factor(data$Outcome)

###### train/test split ###### 
set.seed(123) # this line of code creates the same split for purposes of the marking of this dissertation.
train_indices <- sample(1:nrow(data), size = 0.7 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

###### End of train/test split ###### 

# Now the data is split we can proceed with the both sides of the prediction process.
# This approach has used a function to reduce code length and ensure consistency. 

######  Function to evaluate predictions ###### 
evaluate_model <- function(actual, predicted, model_name) {
  cat("\n========================\n")
  cat("Model:", model_name, "\n")
  cat("========================\n")
  
  cm <- confusionMatrix(factor(predicted), factor(actual))
  print(cm$table)
  
  TP <- cm$table["1", "1"]
  TN <- cm$table["0", "0"]
  FP <- cm$table["1", "0"]
  FN <- cm$table["0", "1"]
  
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  cat("Accuracy:", round(cm$overall['Accuracy'], 4), "\n")
  cat("Precision:", round(precision, 4), "\n")
  cat("Recall:", round(recall, 4), "\n")
  cat("F1 Score:", round(f1_score, 4), "\n")
}

####################### End of Function code ########################

# First model to be used is XGBoost

##################  XGBoost Model ###################  
# Define the features you want to use
selected_features <- c("ISS", "FAI", "LoftCh", "LieCh")

# Create DMatrix objects with only selected features
dtrain <- xgb.DMatrix(data = as.matrix(train_data[, selected_features]),
                      label = as.numeric(train_data$Outcome) - 1)

dtest <- xgb.DMatrix(data = as.matrix(test_data[, selected_features]),
                     label = as.numeric(test_data$Outcome) - 1)


params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.3,
  max_depth = 6
)

xgb_model <- xgb.train(params = params, 
                       data = dtrain, 
                       nrounds = 100, 
                       verbose = 0)

xgb_preds <- ifelse(predict(xgb_model, dtest) > 0.5, 1, 0)

evaluate_model(as.numeric(test_data$Outcome) - 1, xgb_preds, "XGBoost")

######################## End of XGBoost Model Code#############################

# Moving onto Random Forest 

##################  Random Forest ################### 
rf_model <- randomForest(Outcome ~ ISS + FAI + LoftCh + LieCh,
                         data = train_data, 
                         ntree = 100, 
                         mtry = 4)

rf_preds <- predict(rf_model, test_data)

evaluate_model(test_data$Outcome, rf_preds, "Random Forest")

##################  End of Random Forest ################### 

# Onto Support Vector Machines

###############  Support Vector Machine #############  
svm_model <- svm(Outcome ~ ISS + FAI + LoftCh + LieCh,
                 data = train_data, 
                 kernel = "radial")

svm_preds <- predict(svm_model, test_data)

evaluate_model(test_data$Outcome, svm_preds, "SVM")

############ End of Support Vector Machine Code ##############

# Final Model used is Logistic Regression

###############  Logistic Regression ################  
log_model <- glm(Outcome ~ ISS + FAI + LoftCh + LieCh,
                 data = train_data, 
                 family = "binomial")

log_probs <- predict(log_model, test_data, type = "response")

log_preds <- ifelse(log_probs > 0.5, 1, 0)

evaluate_model(as.numeric(test_data$Outcome) - 1, log_preds, "Logistic Regression")

###################### End of Logistic Regression Code ###############################

# The models have now learned from the training data and made predictions on the test data

#####################  Output #######################

confusion_metrics <- list()

performance_metrics <- list()

# Modified evaluation function to return two separate lists
evaluate_model <- function(actual, predicted, model_name) {
  cm <- confusionMatrix(factor(predicted), factor(actual))
  
  TP <- cm$table["1", "1"]
  TN <- cm$table["0", "0"]
  FP <- cm$table["1", "0"]
  FN <- cm$table["0", "1"]
  
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  accuracy <- cm$overall['Accuracy']
  
  confusion_metrics[[model_name]] <<- c(TP = TP, 
                                        TN = TN, 
                                        FP = FP, 
                                        FN = FN)
  
  performance_metrics[[model_name]] <<- c(
    Accuracy = unname(round(accuracy, 4)),
    Precision = round(precision, 4),
    Recall = round(recall, 4),
    F1_Score = round(f1_score, 4)
  )
}

###################### End of Output Code  ##########################

# Creating the confusion and performance matrix

###################### Creating Confusion and Performance matrix  ##########################

# Run all evaluations again and store in the lists
evaluate_model(as.numeric(test_data$Outcome) - 1, xgb_preds, "XGBoost")
evaluate_model(test_data$Outcome, rf_preds, "Random Forest")
evaluate_model(test_data$Outcome, svm_preds, "SVM")
evaluate_model(as.numeric(test_data$Outcome) - 1, log_preds, "Logistic Regression")

# Create and transpose confusion matrix data frame
confusion_df <- as.data.frame(confusion_metrics)
confusion_df <- cbind(Metric = rownames(confusion_df), confusion_df)
rownames(confusion_df) <- NULL
colnames(confusion_df) <- gsub("\\.", " ", colnames(confusion_df))
confusion_df$Mean <- round(rowMeans(confusion_df[ , 2:5]), 2)
confusion_df$SD <- round(apply(confusion_df[ , 2:5], 1, sd, na.rm = TRUE), 2)


# Create and transpose performance metrics data frame
performance_df <- as.data.frame(performance_metrics)
performance_df <- cbind(Metric = rownames(performance_df), performance_df)
performance_df[ , -1] <- lapply(performance_df[ , -1], function(x) {
  paste0(round(as.numeric(x) * 100, 2), "%")
})
rownames(performance_df) <- NULL
colnames(performance_df) <- gsub("\\.", " ", colnames(performance_df))


# Print both data frames
cat("\nConfusion Matrix Components:\n")
print(confusion_df)

cat("\nPerformance Metrics:\n")
print(performance_df)

############ Creating table for Results ###########

Table0 <- flextable(confusion_df)
Table0 <- set_table_properties(Table0, layout = "autofit")
Table0 <- theme_vanilla(Table0)
Table0 <- align(Table0, align = "center", part = "all")
Table0

Table1 <- flextable(performance_df)
Table1 <- set_table_properties(Table1, layout = "autofit")
Table1 <- theme_vanilla(Table1)
Table1 <- align(Table1, align = "center", part = "all")
Table1


################ End Creating table for Results ########################

################ Creating plots for Results ########################

long_data <- pivot_longer(confusion_df, cols = 2:5,
                          names_to = "Model", values_to = "Value")

# Function to plot one metric
plot_one_metric <- function(metric_name) {
  
  # Filter data for this metric
  df <- long_data %>% filter(Metric == metric_name)
  
  # Calculate mean of actual values
  mean_val <- mean(df$Value, na.rm = TRUE)
  
  # Plot
  ggplot(data = df, aes(x = Model, y = Value, fill = Model)) +
    geom_bar(stat = "identity", width = 0.6, color = "black") +
    geom_hline(yintercept = mean_val, linetype = "dashed", color = "black", size = 0.8) +
    annotate("text", x = -Inf, y = mean_val, 
             label = paste0("Mean = ", round(mean_val, 2)), 
             hjust = -0.1, vjust = -0.5, color = "blue") +
    labs(title = paste(metric_name, "by Model"),
         x = "Model",
         y = "Count") +
    theme_minimal() +
    theme(legend.position = "none",
          plot.title = element_text(face = "bold"))
}


# Create separate plots for each metric
plot_tp <- plot_one_metric("TP")
plot_tn <- plot_one_metric("TN")


print(plot_tp)
print(plot_tn)

# Creating the plots side by side
(plot_tp | plot_tn)

######   ######

plot_one_metric <- function(metric_name) {
  
  # Filter data for this metric
  df <- long_data %>% filter(Metric == metric_name)
  
  # Calculate mean of actual values
  mean_val <- mean(df$Value, na.rm = TRUE)
  
  # Plot
  ggplot(data = df, aes(x = Model, y = Value, fill = Model)) +
    geom_bar(stat = "identity", width = 0.6, color = "black") +
    geom_hline(yintercept = mean_val, linetype = "dashed", color = "black", size = 0.8) +
    annotate("text", x = Inf, y = mean_val, 
             label = paste0("Mean = ", round(mean_val, 2)), 
             hjust = 1.1, vjust = -0.5, color = "blue") +
    labs(title = paste(metric_name, "by Model"),
         x = "Model",
         y = "Count") +
    theme_minimal() +
    theme(legend.position = "none",
          plot.title = element_text(face = "bold"))
}

plot_fp <- plot_one_metric("FP")
plot_fn <- plot_one_metric("FN")

print(plot_fp)
print(plot_fn)

# Creating the plots side by side
(plot_fp | plot_fn)

################ End Creating plots for Results ########################

# The following code saves the datasets (train/test) and both matrix's (confusion/performance)
# These are optional steps, the file locations will need to be changed 

################ Saving the Confusion and Performance Matrix ###########
# Please note this part of the code has been labelled as a comment
# this is to avoid overwriting the files saved for the dissertation. 
# Simply remove the # and enter your file location to save the csv files. 


# write.csv(confusion_df,"C:/Users/scott/OneDrive/Documents/University/B1706 Dissertation/1st Attempt/ConfMat.csv")

# write.csv(performance_df,"C:/Users/scott/OneDrive/Documents/University/B1706 Dissertation/1st Attempt/PerfMat.csv")

# write.csv(test_data,"C:/Users/scott/OneDrive/Documents/University/B1706 Dissertation/1st Attempt/testdata.csv")

# write.csv(train_data,"C:/Users/scott/OneDrive/Documents/University/B1706 Dissertation/1st Attempt/traindata.csv")

####################### End of Saving CSV Files ########################

####################### End of Script ########################
