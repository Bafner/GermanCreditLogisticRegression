############################################################
# A. Setup: Packages & Reproducibility                   #
############################################################
# Set a seed for any random process (like sampling) to make our results reproducible.
set.seed(123)

###########################################################
# B. Load and Inspect Data                               #
############################################################

# Load the dataset. 
library(caret) 
ger <- read.csv("GermanCredit.csv", header = TRUE)

# --- Initial Inspection ---
str(ger)                   # Check data types (numeric, character) and preview values.
summary(ger)               # Get a statistical summary for each column.

# --- Variable Conversion ---

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
                  "RESPONSE")], factor )

str(ger)
summary(ger)

table(ger$RESPONSE) 
table(ger$EMPLOYMENT)

# --- Train/Validation ---

n <- nrow(ger)                                             # get number of rows
idx <- sample(1:n, size = round(0.7 * n), replace = FALSE)   # sample 70% of rows for training
train <- ger[idx,]                                        # create training set
valid <- ger[-idx,]                                        # create validation set


# --- Logistic Regression ---
# We use all columns except the target as predictors; adjust if you want a subset.
logistic_mod <- glm(RESPONSE ~ . - RESPONSE, 
                    data = train, 
                    family = binomial())     # fit logistic regression 
summary(logistic_mod)                                            # show coefficient signs and significance (for teaching)


#  Training error 
prob_train <- predict(logistic_mod, newdata = train , type = 'response')    # predicted probabilities on TRAINING set (type="response" gives p = P(Y=1); we need p to apply a 0.5 cutoff for the confusion matrix)
pred_train <- ifelse(prob_train >= .5, "1", "0")           # convert probabilities to classes at 0.5
pred_train <- factor(pred_train, levels = levels(train$RESPONSE))    # make predicted labels use the SAME factor levels/order as the target so confusionMatrix() is correct and the positive class maps properly
cm_train   <- confusionMatrix(pred_train, train$RESPONSE, positive = '1')  # confusion matrix on TRAINING set
cm_train                                   # print full training metrics table


#You Try....Validation error (already in your code)              
prob_valid <- predict(logistic_mod, newdata = valid, type = 'response')      # predicted probabilities on VALIDATION set
pred_valid <- ifelse(prob_valid >= .5, "1", "0")       # convert probabilities to classes at 0.5
pred_valid <- factor(pred_valid, levels = levels(valid$RESPONSE))      # align factor levels with the validation target
cm_valid   <- confusionMatrix(pred_valid, valid$RESPONSE, positive = '1')      # confusion matrix on VALIDATION set
cm_valid      # print full validation metrics table



#  --- Stepwise + Confusion Matrix. Use a cutoff value of 0.1 for significance --- 

null_mod <- glm(RESPONSE ~ 1, data = train, family = binomial())     # intercept-only null model
full_mod <- glm(RESPONSE ~ ., data = train, family = binomial())        # scope equals full model’s formula

scope_all <- formula(full_mod)

step_back <- step(full_mod, direction = "backward", trace = 0);step_back # backward selection from full
step_forw <- step(full_mod, scope = full_mod, direction = "forward", trace = 0);step_forw  # forward selection from null

step_both <- step(full_mod, scope= full_mod, direction = "both", trace = 0);step_both
summary(step_both)

prob_back <- predict(step_back, newdata = valid, type = "response")      # probabilities from backward model
pred_back <- ifelse(prob_back >= 0.5, "1", "0")                          # threshold at 0.5
pred_back <- factor(pred_back, levels = levels(valid$RESPONSE))       # match factor levels
confusionMatrix(pred_back, valid$RESPONSE, positive = '1')            # confusion matrix for backward

