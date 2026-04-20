library(caret)
library(pROC)
library(ggplot2)
library(tidyr)
library(splines)

set.seed(123)

ger <- read.csv("GermanCredit.csv", header = TRUE)

# Convert categorical variables to factors
ger[c("CHK_ACCT", "HISTORY", "NEW_CAR", "USED_CAR", "FURNITURE",
      "RADIO.TV", "EDUCATION", "RETRAINING", "SAV_ACCT", "EMPLOYMENT",
      "MALE_DIV", "MALE_SINGLE", "MALE_MAR_or_WID", "CO.APPLICANT", "GUARANTOR",
      "PRESENT_RESIDENT", "REAL_ESTATE", "PROP_UNKN_NONE", "OTHER_INSTALL", "RENT",
      "OWN_RES", "JOB", "TELEPHONE", "FOREIGN", "RESPONSE")] <-
  lapply(ger[c("CHK_ACCT", "HISTORY", "NEW_CAR", "USED_CAR", "FURNITURE",
               "RADIO.TV", "EDUCATION", "RETRAINING", "SAV_ACCT", "EMPLOYMENT",
               "MALE_DIV", "MALE_SINGLE", "MALE_MAR_or_WID", "CO.APPLICANT",
               "GUARANTOR", "PRESENT_RESIDENT", "REAL_ESTATE", "PROP_UNKN_NONE",
               "OTHER_INSTALL", "RENT", "OWN_RES", "JOB", "TELEPHONE", "FOREIGN",
               "RESPONSE")], factor)

# Recode outcome
ger$RESPONSE <- factor(ger$RESPONSE, levels = c("0", "1"), labels = c("No", "Yes"))

# Train/test split
idx <- createDataPartition(ger$RESPONSE, p = 0.7, list = FALSE)
train <- ger[idx, ]
valid <- ger[-idx, ]

# Repeated CV with class balancing
ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  sampling = "smote"
)

# Logistic regression with linear predictor, plus transformations/interactions
glm_formula <- RESPONSE ~ 
  bs(DURATION, df = 4) +
  bs(AMOUNT, df = 4) +
  bs(AGE, df = 4) +
  CHK_ACCT + HISTORY + SAV_ACCT + EMPLOYMENT +
  INSTALL_RATE + PRESENT_RESIDENT + NUM_CREDITS +
  NEW_CAR + USED_CAR + FURNITURE + RADIO.TV + EDUCATION + RETRAINING +
  MALE_DIV + MALE_SINGLE + MALE_MAR_or_WID + CO.APPLICANT + GUARANTOR +
  REAL_ESTATE + PROP_UNKN_NONE + OTHER_INSTALL + RENT + OWN_RES +
  JOB + TELEPHONE + FOREIGN +
  CHK_ACCT:SAV_ACCT +
  HISTORY:EMPLOYMENT +
  AMOUNT:CHK_ACCT +
  DURATION:AMOUNT

# Fit logistic regression
cv_mod <- train(
  glm_formula,
  data = train,
  method = "glm",
  family = binomial(link = "logit"),
  trControl = ctrl,
  metric = "ROC"
)

print(cv_mod)
summary(cv_mod$finalModel)

# -----------------------------
# Threshold tuning on cross-validated predictions
# Maximize balanced score: Accuracy + Sensitivity + Specificity
# -----------------------------
cv_preds <- cv_mod$pred

# Keep only predictions from the final tuning setting if needed
if ("parameter" %in% names(cv_preds)) {
  cv_preds <- cv_preds[cv_preds$parameter == unique(cv_preds$parameter)[1], ]
}

cv_probs <- cv_preds$Yes
cv_obs <- factor(cv_preds$obs, levels = c("No", "Yes"))

threshold_grid <- seq(0.05, 0.95, by = 0.01)

metric_results <- lapply(threshold_grid, function(th) {
  pred_class <- factor(ifelse(cv_probs >= th, "Yes", "No"), levels = c("No", "Yes"))
  cm <- confusionMatrix(pred_class, cv_obs, positive = "Yes")
  
  data.frame(
    threshold = th,
    Accuracy = unname(cm$overall["Accuracy"]),
    Sensitivity = unname(cm$byClass["Sensitivity"]),
    Specificity = unname(cm$byClass["Specificity"])
  )
})

metric_results <- do.call(rbind, metric_results)
metric_results$Score <- with(metric_results, Accuracy + Sensitivity + Specificity)

best_row <- metric_results[which.max(metric_results$Score), ]
best_thresh <- best_row$threshold

print(best_row)
cat("Optimal threshold =", best_thresh, "\n")

# -----------------------------
# Training set evaluation
# -----------------------------
prob_train <- predict(cv_mod, newdata = train, type = "prob")[, "Yes"]
pred_train <- factor(ifelse(prob_train >= best_thresh, "Yes", "No"),
                     levels = c("No", "Yes"))

cm_train <- confusionMatrix(pred_train, train$RESPONSE, positive = "Yes")
print(cm_train)

# -----------------------------
# Validation set evaluation
# -----------------------------
prob_valid <- predict(cv_mod, newdata = valid, type = "prob")[, "Yes"]
pred_valid <- factor(ifelse(prob_valid >= best_thresh, "Yes", "No"),
                     levels = c("No", "Yes"))

cm_valid <- confusionMatrix(pred_valid, valid$RESPONSE, positive = "Yes")
print(cm_valid)

# -----------------------------
# Validation ROC and AUC
# -----------------------------
roc_obj <- roc(valid$RESPONSE, prob_valid, levels = c("No", "Yes"), direction = "<")
print(auc(roc_obj))
plot(roc_obj, main = "Validation ROC Curve")

# -----------------------------
# Final value as R code
# -----------------------------
best_thresh

