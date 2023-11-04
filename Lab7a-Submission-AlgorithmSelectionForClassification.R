# STEP 1. Install and Load the Required Packages ----
## stats ----
if (require("stats")) {
  require("stats")
} else {
  install.packages("stats", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## mlbench ----
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## MASS ----
if (require("MASS")) {
  require("MASS")
} else {
  install.packages("MASS", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## glmnet ----
if (require("glmnet")) {
  require("glmnet")
} else {
  install.packages("glmnet", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## e1071 ----
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## kernlab ----
if (require("kernlab")) {
  require("kernlab")
} else {
  install.packages("kernlab", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## rpart ----
if (require("rpart")) {
  require("rpart")
} else {
  install.packages("rpart", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# Load the datasets ----
data(ToothGrowth)
View(ToothGrowth)

# Algorithm Selection for Classification ----
# A. Linear Algorithms ----
## 1. Linear Discriminant Analysis----
### 1.a. Linear Discriminant Analysis without caret ----
# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(ToothGrowth$supp,
                                   p = 0.7,
                                   list = FALSE)
ToothGrowth_train <- ToothGrowth[train_index, ]
ToothGrowth_test <- ToothGrowth[-train_index, ]

#### Train the model ----
ToothGrowth_model_lda <- lda(supp ~ ., data = ToothGrowth_train)

#### Display the model's details ----
print(ToothGrowth_model_lda)


#### Make predictions ----
predictions <- predict(ToothGrowth_model_lda,
                       ToothGrowth_test[, 1:3])$class

#### Display the model's evaluation metrics ----
table(predictions, ToothGrowth_test$supp)

### 1.b. Linear Discriminant Analysis with caret ----
# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(ToothGrowth$supp,
                                   p = 0.7,
                                   list = FALSE)
ToothGrowth_train <- ToothGrowth[train_index, ]
ToothGrowth_test <- ToothGrowth[-train_index, ]

#### Train the model ----
set.seed(7)

# We apply a Leave One Out Cross Validation resampling method
train_control <- trainControl(method = "LOOCV")
# We also apply a standardize data transform to make the mean = 0 and
# standard deviation = 1
ToothGrowth_caret_model_lda <- train(supp ~ .,
                                     data = ToothGrowth_train,
                                     method = "lda", metric = "Accuracy",
                                     preProcess = c("center", "scale"),
                                     trControl = train_control)

#### Display the model's details ----
print(ToothGrowth_caret_model_lda)

#### Make predictions ----
predictions <- predict(ToothGrowth_caret_model_lda,
                       ToothGrowth_test[, 1:3])

#### Display the model's evaluation metrics ----
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         ToothGrowth_test[, 1:2]$supp)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

##2. Regularized Linear Regression ----
### 2.a. Regularized Linear Regression Classification Problem with CARET ----
#### Load and split the dataset ----
data(ToothGrowth)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(ToothGrowth$supp,
                                   p = 0.7,
                                   list = FALSE)
ToothGrowth_regression_train <- ToothGrowth[train_index, ]
ToothGrowth_regression_test <- ToothGrowth[-train_index, ]

#### Train the model ----
# We apply the 5-fold cross validation resampling method
set.seed(7)
train_control <- trainControl(method = "cv", number = 5)
ToothGrowth_caret_model_glmnet <-
  train(supp ~ ., data = ToothGrowth_regression_train,
        method = "glmnet", metric = "Accuracy",
        preProcess = c("center", "scale"), trControl = train_control)

#### Display the model's details ----
print(ToothGrowth_caret_model_glmnet)

#### Make predictions ----
predictions <- predict(ToothGrowth_caret_model_glmnet,
                       ToothGrowth_regression_test[, 1:3])
predictions <- factor(predictions, levels = levels(ToothGrowth_regression_test$supp))


#### Display the model's evaluation metrics ----
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         ToothGrowth_regression_test$supp)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

## 3. Logistic Regression ----
### 3.a. Logistic Regression without caret ----
#### Load and split the dataset ----
data(ToothGrowth)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(ToothGrowth$supp,
                                   p = 0.7,
                                   list = FALSE)
ToothGrowth_logistic_train <- ToothGrowth[train_index, ]
ToothGrowth_logistic_test <- ToothGrowth[-train_index, ]

#### Train the model ----
ToothGrowth_logistic_model_glm <- glm(supp ~ ., data = ToothGrowth_logistic_train,
                                      family = binomial(link = "logit"))

#### Display the model's details ----
print(ToothGrowth_logistic_model_glm)

#### Make predictions ----
probabilities <- predict(ToothGrowth_logistic_model_glm, ToothGrowth_logistic_test[, 1:3],
                         type = "response")
print(probabilities)
predictions <- ifelse(probabilities > 0.5, "OJ" ,"VC")
print(predictions)

#### Display the model's evaluation metrics ----
table(predictions, ToothGrowth_logistic_test$supp)

### 3.b. Logistic Regression with caret ----
#### Load and split the dataset ----
data(ToothGrowth)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(ToothGrowth$supp,
                                   p = 0.7,
                                   list = FALSE)
ToothGrowth_logisticb_train <- ToothGrowth[train_index, ]
ToothGrowth_logisticb_test <- ToothGrowth[-train_index, ]

#### Train the model ----
# We apply the 5-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 5)
# We can use "regLogistic" instead of "glm"
# Notice the data transformation applied when we call the train function
# in caret, i.e., a standardize data transform (centre + scale)
set.seed(7)
ToothGrowth_caret_model_logistic <-
  train(supp ~ ., data = ToothGrowth_logisticb_train,
        method = "regLogistic", metric = "Accuracy",
        preProcess = c("center", "scale"), trControl = train_control)

#### Display the model's details ----
print(ToothGrowth_caret_model_logistic)

#### Make predictions ----

predictions <- predict(ToothGrowth_caret_model_logistic,
                       ToothGrowth_logisticb_test[, 1:3])

#### Display the model's evaluation metrics ----

confusion_matrix <-
  caret::confusionMatrix(predictions,
                         ToothGrowth_logisticb_test[, 1:2]$supp)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")


# B. Non-Linear Algorithms ----
## 1.  Classification and Regression Trees ----
### 1.a. Decision tree for a classification problem without caret ----
#### Load and split the dataset ----
data(ToothGrowth)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(ToothGrowth$supp,
                                   p = 0.7,
                                   list = FALSE)
ToothGrowth_train <- ToothGrowth[train_index, ]
ToothGrowth_test <- ToothGrowth[-train_index, ]

#### Train the model ----
ToothGrowth_model_rpart <- rpart(supp ~ ., data = ToothGrowth_train)

#### Display the model's details ----
print(ToothGrowth_model_rpart)

#### Make predictions ----
predictions <- predict(ToothGrowth_model_rpart,
                       ToothGrowth_test[, 1:3],
                       type = "class")

#### Display the model's evaluation metrics ----
table(predictions, ToothGrowth_test$supp)

confusion_matrix <-
  caret::confusionMatrix(predictions,
                         ToothGrowth_test[, 1:3]$supp)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")


### 1.b. Decision tree for a classification problem with caret ----
#### Load and split the dataset ----
data(ToothGrowth)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(ToothGrowth$supp,
                                   p = 0.7,
                                   list = FALSE)
ToothGrowth_train <- ToothGrowth[train_index, ]
ToothGrowth_test <- ToothGrowth[-train_index, ]

#### Train the model ----
set.seed(7)
# We apply the 5-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 5)
ToothGrowth_caret_model_rpart <- train(supp ~ ., data = ToothGrowth,
                                       method = "rpart", metric = "Accuracy",
                                       trControl = train_control)

#### Display the model's details ----
print(ToothGrowth_caret_model_rpart)

#### Make predictions ----
predictions <- predict(ToothGrowth_model_rpart,
                       ToothGrowth_test[, 1:3],
                       type = "class")

#### Display the model's evaluation metrics ----
table(predictions, ToothGrowth_test$supp)

confusion_matrix <-
  caret::confusionMatrix(predictions,
                         ToothGrowth_test[, 1:3]$supp)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

## 2.  Naïve Bayes ----
### 2.a. Naïve Bayes Classifier for a Classification Problem without CARET ----
#### Load and split the dataset ----
data(ToothGrowth)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(ToothGrowth$supp,
                                   p = 0.7,
                                   list = FALSE)
ToothGrowth_train <- ToothGrowth[train_index, ]
ToothGrowth_test <- ToothGrowth[-train_index, ]

#### Train the model ----
ToothGrowth_model_nb <- naiveBayes(supp ~ .,
                                   data = ToothGrowth_train)

#### Display the model's details ----
print(ToothGrowth_model_nb)

#### Make predictions ----
predictions <- predict(ToothGrowth_model_nb,
                       ToothGrowth_test[, 1:3])

#### Display the model's evaluation metrics ----
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         ToothGrowth_test[, 1:3]$supp)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

### 2.b. Naïve Bayes Classifier for a Classification Problem with CARET ----
#### Load and split the dataset ----
data(ToothGrowth)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(ToothGrowth$supp,
                                   p = 0.7,
                                   list = FALSE)
ToothGrowth_train <- ToothGrowth[train_index, ]
ToothGrowth_test <- ToothGrowth[-train_index, ]

#### Train the model ----
# We apply the 5-fold cross validation resampling method
set.seed(7)
train_control <- trainControl(method = "cv", number = 5)
ToothGrowth_caret_model_nb <- train(supp ~ .,
                                    data = ToothGrowth_train,
                                    method = "nb", metric = "Accuracy",
                                    trControl = train_control)

#### Display the model's details ----
print(ToothGrowth_caret_model_nb)

#### Make predictions ----
predictions <- predict(ToothGrowth_caret_model_nb,
                       ToothGrowth_test[, 1:3])

#### Display the model's evaluation metrics ----
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         ToothGrowth_test[, 1:3]$supp)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

## 3.  k-Nearest Neighbours ----
### 3.a. kNN for a classification problem without CARET's train function ----
#### Load and split the dataset ----
data(ToothGrowth)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(ToothGrowth$supp,
                                   p = 0.7,
                                   list = FALSE)
ToothGrowth_train <- ToothGrowth[train_index, ]
ToothGrowth_test <- ToothGrowth[-train_index, ]

#### Train the model ----
ToothGrowth_caret_model_knn <- knn3(supp ~ ., data = ToothGrowth_train, k=3)

#### Display the model's details ----
print(ToothGrowth_caret_model_knn)

#### Make predictions ----
predictions <- predict(ToothGrowth_caret_model_knn,
                       ToothGrowth_test[, 1:3],
                       type = "class")

#### Display the model's evaluation metrics ----
table(predictions, ToothGrowth_test$supp)


### 3.b. kNN for a classification problem with CARET's train function ----
#### Load and split the dataset ----
data(ToothGrowth)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(ToothGrowth$supp,
                                   p = 0.7,
                                   list = FALSE)
ToothGrowth_train <- ToothGrowth[train_index, ]
ToothGrowth_test <- ToothGrowth[-train_index, ]

#### Train the model ----
# We apply the 10-fold cross validation resampling method
# We also apply the standardize data transform
set.seed(7)
train_control <- trainControl(method = "cv", number = 10)
ToothGrowth_caret_model_knn <- train(supp ~ ., data = ToothGrowth,
                                     method = "knn", metric = "Accuracy",
                                     preProcess = c("center", "scale"),
                                     trControl = train_control)

#### Display the model's details ----
print(ToothGrowth_caret_model_knn)

#### Make predictions ----
predictions <- predict(ToothGrowth_caret_model_knn,
                       ToothGrowth_test[, 1:3])

#### Display the model's evaluation metrics ----
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         ToothGrowth_test[, 1:2]$supp)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

## 4.  Support Vector Machine ----
### 4.a. SVM Classifier for a classification problem without CARET ----
#### Load and split the dataset ----
data(ToothGrowth)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(ToothGrowth$supp,
                                   p = 0.7,
                                   list = FALSE)
ToothGrowth_train <- ToothGrowth[train_index, ]
ToothGrowth_test <- ToothGrowth[-train_index, ]

#### Train the model ----
ToothGrowth_model_svm <- ksvm(supp ~ ., data = ToothGrowth_train,
                              kernel = "rbfdot")

#### Display the model's details ----
print(ToothGrowth_model_svm)

#### Make predictions ----
predictions <- predict(ToothGrowth_model_svm, ToothGrowth_test[, 1:3],
                       type = "response")

#### Display the model's evaluation metrics ----
table(predictions, ToothGrowth_test$supp)

confusion_matrix <-
  caret::confusionMatrix(predictions,
                         ToothGrowth_test[, 1:3]$supp)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")


### 4.b. SVM Classifier for a classification problem with CARET ----
#### Load and split the dataset ----
data(ToothGrowth)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(ToothGrowth$supp,
                                   p = 0.7,
                                   list = FALSE)
ToothGrowth_train <- ToothGrowth[train_index, ]
ToothGrowth_test <- ToothGrowth[-train_index, ]

#### Train the model ----
set.seed(7)
train_control <- trainControl(method = "cv", number = 5)
ToothGrowth_caret_model_svm_radial <- # nolint: object_length_linter.
  train(supp ~ ., data = ToothGrowth_train, method = "svmRadial",
        metric = "Accuracy", trControl = train_control)

#### Display the model's details ----
print(ToothGrowth_caret_model_svm_radial)

#### Make predictions ----
predictions <- predict(ToothGrowth_caret_model_svm_radial,
                       ToothGrowth_test[, 1:3])

#### Display the model's evaluation metrics ----
table(predictions, ToothGrowth_test$supp)

confusion_matrix <-
  caret::confusionMatrix(predictions,
                         ToothGrowth_test[, 1:3]$supp)
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")
