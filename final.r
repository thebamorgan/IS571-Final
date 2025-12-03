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

# --- Out-of-Sample Validation ---

# 13. Set a seed for reproducible random sampling
set.seed(123) # Using a seed ensures you get the same random split every time.

# 14. Randomly select indices for the training set
train_indices <- sample(1:nrow(loan_data), 4611)
train_data <- loan_data[train_indices, ]
test_data <- loan_data[-train_indices, ]

# 15. Fit the logistic model on the training set
log_model_oos <- glm(Status ~ Credit.Grade + Amount + Age + Borrower.Rate + Ratio, 
                     family = binomial, 
                     data = train_data)

# 16. Calculate the optimal cutoff using the TRAINING data proportion
oos_class_proportions <- table(train_data$Status)
oos_cutoff_prob <- oos_class_proportions["1"] / sum(oos_class_proportions)

cat("\n--- Out-of-Sample Validation ---\n")
cat("Optimal cutoff probability based on training data:", round(oos_cutoff_prob, 4), "\n")

# 17. Predict probabilities on the test set
out_of_sample_pred <- predict(log_model_oos, newdata = test_data, type = "response")

# 18. Create the out-of-sample confusion matrix
confusion_matrix_oos <- table(Actual = test_data$Status, Predicted = ifelse(out_of_sample_pred >= oos_cutoff_prob, 1, 0))
print(confusion_matrix_oos)

# 19. Calculate and display the out-of-sample prediction accuracy rate
accuracy_rate_oos <- (confusion_matrix_oos[1, 1] + confusion_matrix_oos[2, 2]) / sum(confusion_matrix_oos)
cat("\nOut-of-sample prediction accuracy rate:", round(accuracy_rate_oos, 4), "\n")

# --- Lift Chart Generation ---

# 20. Prepare data for the lift chart
# Create a data frame with predicted probabilities and actual outcomes for the test set.
lift_data <- data.frame(Prediction = out_of_sample_pred, Actual = test_data$Status)

# Sort the data frame by prediction probability in descending order.
lift_data_sorted <- lift_data[order(lift_data$Prediction, decreasing = TRUE), ]

# 21. Calculate the baseline (random) performance
# This is the overall proportion of bad loans in the test set.
baseline_prob <- mean(test_data$Status)

# 22. Use a FOR loop to calculate the cumulative lift
num_loans <- nrow(test_data)
cumulative_bad_loans <- numeric(num_loans)
random_baseline <- numeric(num_loans)

for (i in 1:num_loans) {
  # Cumulative bad loans found by the model
  cumulative_bad_loans[i] <- sum(lift_data_sorted$Actual[1:i])
  # Expected bad loans if selected randomly
  random_baseline[i] <- i * baseline_prob
}

# 23. Plot the lift chart
plot(1:num_loans, cumulative_bad_loans, type = 'l', col = 'black', lwd = 2,
     xlab = "Number of Loans (Sorted by Risk)", ylab = "Cumulative Bad Loans Found",
     main = "Lift Chart for Test Set")
lines(1:num_loans, random_baseline, col = 'green', lwd = 2, lty = 2)
legend("bottomright", legend = c("Model", "Random Baseline"), col = c("black", "green"), lwd = 2, lty = c(1, 2))

# --- 20-Fold Random Sample Validation ---

# 24. Initialize a vector to store accuracy rates from each fold
accuracy_rates_20_fold <- numeric(20)

cat("\n--- 20-Fold Random Sample Validation ---\n")

# 25. Loop 20 times to get a robust measure of out-of-sample accuracy
for (j in 1:20) {
  # Create a new random split for each iteration
  train_indices_j <- sample(1:nrow(loan_data), 4611)
  train_data_j <- loan_data[train_indices_j, ]
  test_data_j <- loan_data[-train_indices_j, ]
  
  # Fit the model on the current training fold
  log_model_j <- glm(Status ~ Credit.Grade + Amount + Age + Borrower.Rate + Ratio, 
                     family = binomial, 
                     data = train_data_j)
  
  # Calculate the cutoff probability from the current training fold's proportions
  cutoff_prob_j <- table(train_data_j$Status)["1"] / sum(table(train_data_j$Status))
  
  # Predict on the current test fold
  pred_j <- predict(log_model_j, newdata = test_data_j, type = "response")
  
  # Calculate accuracy for the current fold
  cm_j <- table(Actual = test_data_j$Status, Predicted = ifelse(pred_j >= cutoff_prob_j, 1, 0))
  accuracy_rates_20_fold[j] <- (cm_j[1, 1] + cm_j[2, 2]) / sum(cm_j)
}

# 26. Display the results
cat("Accuracy rates for 20 random test samples:\n")
print(round(accuracy_rates_20_fold, 4))
cat("\nMean out-of-sample prediction accuracy:", round(mean(accuracy_rates_20_fold), 4), "\n")
