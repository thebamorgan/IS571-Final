# 1. Load the dataset
# Ensure the CSV file is in your R working directory, or provide the full path.
loan_data <- read.csv("LoanData.csv", stringsAsFactors = TRUE)

# 2. Recode the 'Status' variable
# The original levels are "Current", "Default", "Late".
# We will map "Current" to 0 and both "Default" and "Late" to 1.
levels(loan_data$Status) <- c(0, 1, 1)
# Convert the factor to a numeric variable for the model
loan_data$Status <- as.numeric(as.character(loan_data$Status))

# 3. Recode 'Debt.To.Income.Ratio' into a new 'Ratio' factor
# Define the break points and labels for the new categories.
loan_data$Ratio <- cut(loan_data$Debt.To.Income.Ratio, 
                       breaks = c(-Inf, 0.1, 0.3, Inf), 
                       labels = c("low", "medium", "high"), 
                       right = FALSE) # right=FALSE means intervals are [min, max)

# 4. Fit the logistic regression model
# We use glm() with family=binomial for logistic regression.
# The formula specifies 'Status' as the dependent variable.
log_model <- glm(Status ~ Credit.Grade + Amount + Age + Borrower.Rate + Ratio, 
                 family = binomial, 
                 data = loan_data)

# 5. Display the model summary
summary(log_model)

# 6. In-sample fitting evaluation with a 0.5 cutoff probability
# Predict probabilities on the training data. 'type = "response"' gives us probabilities.
in_sample_pred <- predict(log_model, newdata = loan_data, type = "response")

# Create a confusion matrix using a 0.5 cutoff.
# floor(in_sample_pred + 0.5) converts probabilities >= 0.5 to 1 and < 0.5 to 0.
confusion_matrix <- table(Actual = loan_data$Status, Predicted = floor(in_sample_pred + 0.5))

# 7. Display the confusion matrix
print(confusion_matrix)

# 8. Calculate the optimal cutoff probability
# With symmetric costs of misclassification, the optimal cutoff is the proportion of the minority class (Status = 1).
# This is because we want to classify an observation as the majority class (Status = 0) only if its predicted probability of being in the majority class is higher than the proportion of the majority class.
# This is equivalent to using the proportion of the minority class as the threshold for predicting the minority class.
class_proportions <- table(loan_data$Status)
cutoff_prob <- class_proportions["1"] / sum(class_proportions)

# The helper file uses an equivalent but less direct calculation: floor(pred + (1 - cutoff_prob)).
# This is the same as classifying as 1 if pred >= cutoff_prob.

# 9. Explain the new cutoff probability
cat("\nThe proportion of loans that are 'Default' or 'Late' (Status = 1) is:", round(cutoff_prob, 4), "\n")
cat("With symmetric misclassification costs, this proportion is used as the new cutoff probability.\n")
cat("This is because if we predict every loan to be 'Current' (Status = 0), our accuracy would be the proportion of 'Current' loans. We use the model to do better than that baseline.\n")

# 10. Create a new confusion matrix with the updated cutoff
confusion_matrix_updated <- table(Actual = loan_data$Status, Predicted = ifelse(in_sample_pred >= cutoff_prob, 1, 0))

# 11. Display the new confusion matrix
print(confusion_matrix_updated)

# 12. Calculate and display the overall in-sample misclassification rate
misclassification_rate <- (confusion_matrix_updated[1, 2] + confusion_matrix_updated[2, 1]) / sum(confusion_matrix_updated)

cat("\nThe overall in-sample misclassification rate with the updated cutoff is:", round(misclassification_rate, 4), "\n")
