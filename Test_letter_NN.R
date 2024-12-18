# Load the data
library(microbenchmark)

# Training data
letter_train <- read.table("Data/letter-train.txt", header = F, colClasses = "numeric")
Y <- letter_train[, 1]
X <- as.matrix(letter_train[, -1])

# Update training to set last part as validation
id_val = 1801:2000
Yval = Y[id_val]
Xval = X[id_val, ]
Ytrain = Y[-id_val]
Xtrain = X[-id_val, ]

# Testing data
letter_test <- read.table("Data/letter-test.txt", header = F, colClasses = "numeric")
Yt <- letter_test[, 1]
Xt <- as.matrix(letter_test[, -1])

# Source the NN function
source("FunctionsNN.R")

# [ToDo] Source the functions from HW3 (replace FunctionsLR.R with your working code)
source("FunctionsLR.R")

# Recall the results of linear classifier from HW3
# Add intercept column
Xinter <- cbind(rep(1, nrow(Xtrain)), Xtrain)
Xtinter <- cbind(rep(1, nrow(Xt)), Xt)

#  Apply LR (note that here lambda is not on the same scale as in NN due to scaling by training size)
out <- LRMultiClass(Xinter, Ytrain, Xtinter, Yt, lambda = 1, numIter = 150, eta = 0.1)
plot(out$objective, type = 'o')
plot(out$error_train, type = 'o') # around 19.5 if keep training
plot(out$error_test, type = 'o') # around 25 if keep training


# Apply neural network training with default given parameters
out2 = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
                rate = 0.1, mbatch = 50, nEpoch = 150,
                hidden_p = 100, scale = 1e-3, seed = 12345)
plot(1:length(out2$error), out2$error, ylim = c(0, 70))
lines(1:length(out2$error_val), out2$error_val, col = "red")

# Evaluate error on testing data
test_error = evaluate_error(Xt, Yt, out2$params$W1, out2$params$b1, out2$params$W2, out2$params$b2)
test_error # 16.1

# A different set of parameters for minimizing test error
out4 = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
                rate = 0.1, mbatch = 25, nEpoch = 100,
                hidden_p = 200, scale = 1e-2, seed = 12345)
plot(1:length(out4$error), out4$error, ylim = c(0, 70))
lines(1:length(out4$error_val), out4$error_val, col = "red")

# Evaluate error on testing data
test_error2 = evaluate_error(Xt, Yt, out4$params$W1, out4$params$b1, out4$params$W2, out4$params$b2)
test_error2 # 16.1
# test_error2 = 15.75


microbenchmark(
  LRMultiClass(Xinter, Ytrain, Xtinter, Yt, lambda = 1, numIter = 150, eta = 0.1),
  NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
                  rate = 0.1, mbatch = 50, nEpoch = 150,
                  hidden_p = 100, scale = 1e-3, seed = 12345),
  NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
           rate = 0.1, mbatch = 25, nEpoch = 100,
           hidden_p = 200, scale = 1e-2, seed = 12345),
  times = 25
)
# Median time (HW3) = 3.134627 seconds
# Median time (HW7 - default input) = 2.463481 seconds
# Median time (HW7 - input with minimum test error) = 3.348792 seconds


# [ToDo] Try changing the parameters above to obtain a better performance,
# this will likely take several trials
lambda <- c(0.1, 0.01, 0.001)
rate <- c(0.1, 0.01, 0.001)
mbatch <- c(25, 50, 100, 250)
nEpoch <- c(50, 100)
hidden_p <- c(50, 100, 200)
scale <- c(0.1, 0.01, 0.001)

param_grid <- expand.grid(lambda = lambda,
                          rate = rate,
                          mbatch = mbatch,
                          nEpoch = nEpoch,
                          hidden_p = hidden_p,
                          scale = scale)

train_model <- function(params) {
  out3 <- NN_train(Xtrain, Ytrain, Xval, Yval,
           lambda = as.numeric(params["lambda"]),
           rate = as.numeric(params["rate"]),
           mbatch = as.numeric(params["mbatch"]),
           nEpoch = as.numeric(params["nEpoch"]),
           hidden_p = as.numeric(params["hidden_p"]),
           scale = as.numeric(params["scale"]),
           seed = 12345)
  
  train_error_grid = evaluate_error(Xval = Xtrain, 
                                    yval = Ytrain, 
                                    W1 = out3$params$W1, 
                                    b1 = out3$params$b1, 
                                    W2 = out3$params$W2, 
                                    b2 = out3$params$b2)
  val_error_grid = evaluate_error(Xval = Xval, 
                                  yval = Yval, 
                                  W1 = out3$params$W1, 
                                  b1 = out3$params$b1, 
                                  W2 = out3$params$W2, 
                                  b2 = out3$params$b2)
  test_error_grid = evaluate_error(Xval = Xt, 
                                   yval = Yt, 
                                   W1 = out3$params$W1, 
                                   b1 = out3$params$b1, 
                                   W2 = out3$params$W2, 
                                   b2 = out3$params$b2)
  return(list(train_error_grid = train_error_grid,
              val_error_grid = val_error_grid,
              test_error_grid = test_error_grid))
}

results <- apply(param_grid, 1, function(params){
  params <- as.list(params)
  train_model(params)
})

results_df <- do.call(rbind, lapply(results, as.data.frame))

min_train_error <- min(results_df$train_error_grid)
min_val_error <- min(results_df$val_error_grid)
min_test_error <- min(results_df$test_error_grid)

best_train_error <- results_df[which.min(results_df$train_error_grid), ]
best_val_error <- results_df[which.min(results_df$val_error_grid), ]
best_test_error <- results_df[which.min(results_df$test_error_grid), ]

print(paste("Minimum Train Error:", min_train_error))
# "Minimum Train Error: 5.16666666666667"
print("All Errors when Train error is minimum: ")
print(best_train_error)
# train_error_grid val_error_grid test_error_grid
# 399         5.166667           16.5           15.75
print("Best Parameter for Minimum Train Error: ")
print(param_grid[which.min(results_df$train_error_grid),])
# lambda rate mbatch nEpoch hidden_p scale
# 399  0.001  0.1     25    100      200  0.01

print(paste("Minimum val Error:", min_val_error))
# "Minimum val Error: 15"
print("All Errors when Val error is minimum: ")
print(best_val_error)
# train_error_grid val_error_grid test_error_grid
# 408              5.5             15        16.18333
print("Best Parameter for Minimum Val Error: ")
print(param_grid[which.min(results_df$val_error_grid),])
# lambda rate mbatch nEpoch hidden_p scale
# 408  0.001  0.1     50    100      200  0.01

print(paste("Minimum Test Error:", min_test_error))
# "Minimum Test Error: 15.75"
print("All Errors when Test error is minimum: ")
print(best_test_error)
# train_error_grid val_error_grid test_error_grid
# 399         5.166667           16.5           15.75
print("Best Parameter for Minimum Test Error: ")
print(param_grid[which.min(results_df$test_error_grid),])
# lambda rate mbatch nEpoch hidden_p scale
# 399  0.001  0.1     25    100      200  0.01


# Note: The first number in the above results is the row number of the results_df dataframe.

# Plot for variability in train error
plot(1:length(results_df$train_error_grid), results_df$train_error_grid)
lines(1:length(results_df$train_error_grid), results_df$train_error_grid, col = "red")

# Plot for variability in val error
plot(1:length(results_df$val_error_grid), results_df$val_error_grid)
lines(1:length(results_df$val_error_grid), results_df$val_error_grid, col = "red")

# Plot for variability in test error
plot(1:length(results_df$test_error_grid), results_df$test_error_grid)
lines(1:length(results_df$test_error_grid), results_df$test_error_grid, col = "red")

# Plot for variability comparison for different errors
plot(results_df$train_error_grid, type = 'p', col = 'red', pch = 19,
     xlab = "Index", ylab = "% Error")
points(results_df$val_error_grid, col = 'blue', pch = 17)
points(results_df$test_error_grid, col = 'green', pch = 15)
