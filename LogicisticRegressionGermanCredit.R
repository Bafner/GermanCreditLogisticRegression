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

