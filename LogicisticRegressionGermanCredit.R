############################################################
# A. Setup
############################################################
set.seed(123)

library(caret)

############################################################
# B. Load
############################################################
ger <- read.csv("GermanCredit.csv", header = TRUE)
str(ger)
summary(ger)

############################################################
# C. Prepare variable types
############################################################
ger[c("CHK_ACCT", "HISTORY", "NEW_CAR", "USED_CAR", "FURNITURE", "RADIO.TV",
         "EDUCATION", "RETRAINING", "SAV_ACCT", "EMPLOYMENT", "MALE_DIV",
         "MALE_SINGLE", "MALE_MAR_or_WID", "CO.APPLICANT", "GUARANTOR", "PRESENT_RESIDENT",
         "REAL_ESTATE", "PROP_UNKN_NONE", "OTHER_INSTALL", "RENT", "OWN_RES", "JOB",
         "TELEPHONE", "FOREIGN", "RESPONSE")] <-
  lapply(ger[c("CHK_ACCT", "HISTORY", "NEW_CAR", "USED_CAR", "FURNITURE",
                  "RADIO.TV", "EDUCATION", "RETRAINING", "SAV_ACCT", "EMPLOYMENT",
                  "MALE_DIV", "MALE_SINGLE", "MALE_MAR_or_WID", "CO.APPLICANT", "GUARANTOR",
                  "PRESENT_RESIDENT", "REAL_ESTATE", "PROP_UNKN_NONE", "OTHER_INSTALL", "RENT",
                  "OWN_RES", "JOB", "TELEPHONE", "FOREIGN", "RESPONSE")], factor )

str(ger)
############################################################
# D. Train / Validation split
############################################################
n <- nrow(ger)
idx <- sample(1:n, size = round(0.7 * n))

train <- ger[idx, ]
valid <- ger[-idx, ]

############################################################
# E. Logistic regression (full model)
############################################################
log_mod <- glm(RESPONSE ~ . - OBS., data = train, family = binomial())
summary(log_mod)

# Training
prob_train <- predict(log_mod, newdata = train, type = "response")
pred_train <- ifelse(prob_train >= 0.405, levels(train$RESPONSE)[2], levels(train$RESPONSE)[1])
pred_train <- factor(pred_train, levels = levels(train$RESPONSE))
cm_train <- confusionMatrix(pred_train, train$RESPONSE)
cm_train

# Validation
prob_valid <- predict(log_mod, newdata = valid, type = "response")
pred_valid <- ifelse(prob_valid >= 0.405, levels(valid$RESPONSE)[2], levels(valid$RESPONSE)[1])
pred_valid <- factor(pred_valid, levels = levels(valid$RESPONSE))
cm_valid <- confusionMatrix(pred_valid, valid$RESPONSE)
cm_valid
# cm_valid has highest Accurarcy

############################################################
# F. Stepwise models
############################################################
null_mod <- glm(RESPONSE ~ 1, data = train, family = binomial())
full_mod <- glm(RESPONSE ~ ., data = train, family = binomial())
# 
# step_back <- stats::step(full_mod, direction = "backward", trace = 0)
# step_forw <- stats::step(null_mod,
#                          scope = formula(full_mod),
#                          direction = "forward",
#                          trace = 0)
# step_both <- stats::step(full_mod, direction = "both", trace = 0)
# 
# # Backward
# prob_back <- predict(step_back, newdata = valid, type = "response")
# pred_back <- ifelse(prob_back >= 0.5, levels(valid$RESPONSE)[2], levels(valid$RESPONSE)[1])
# pred_back <- factor(pred_back, levels = levels(valid$RESPONSE))
# confusionMatrix(pred_back, valid$RESPONSE)
# 
# # Forward
# prob_forw <- predict(step_forw, newdata = valid, type = "response")
# pred_forw <- ifelse(prob_forw >= 0.5, levels(valid$RESPONSE)[2], levels(valid$RESPONSE)[1])
# pred_forw <- factor(pred_forw, levels = levels(valid$RESPONSE))
# confusionMatrix(pred_forw, valid$RESPONSE)
# 
# # Both
# prob_both <- predict(step_both, newdata = valid, type = "response")
# pred_both <- ifelse(prob_both >= 0.5, levels(valid$RESPONSE)[2], levels(valid$RESPONSE)[1])
# pred_both <- factor(pred_both, levels = levels(valid$RESPONSE))
# confusionMatrix(pred_both, valid$RESPONSE)

############################################################
# G. Linear predictor and probability
############################################################
z_valid <- predict(full_mod, newdata = valid, type = "link")
prob_from_z <- plogis(z_valid)

head(cbind(Z = z_valid, Prob = prob_from_z), 10)
coefs <- coef(full_mod)
coefs
