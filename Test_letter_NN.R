# Load the data

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

# [ToDo] Try changing the parameters above to obtain a better performance,
# this will likely take several trials
lambda <- c(0.1, 0.01, 0.001)
rate <- c(0.1, 0.01, 0.001)
mbatch <- 50
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
  
  train_error_grid = evaluate_error(Xtrain, Ytrain, out3$params$W1, out3$params$b1, out3$params$W2, out3$params$b2)
  val_error_grid = evaluate_error(Xval, Yval, out3$params$W1, out3$params$b1, out3$params$W2, out3$params$b2)
  test_error_grid = evaluate_error(Xt, Yt, out3$params$W1, out3$params$b1, out3$params$W2, out3$params$b2)
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

best_train_params <- results_df[which.min(results_df$train_error_grid), ]
best_val_params <- results_df[which.min(results_df$val_error_grid), ]
best_test_params <- results_df[which.min(results_df$test_error_grid), ]


