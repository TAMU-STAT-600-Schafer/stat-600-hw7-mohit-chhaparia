# Source the NN function
source("FunctionsNN.R")

library(testthat)

##########################
# Tests on initialize_bw #
##########################

test_that("initialize_bw works correctly", {
  expect_error(initialize_bw(p = 0, hidden_p = 2, K = 2))
  expect_error(initialize_bw(p = 3, hidden_p = 0, K = 2))
  expect_error(initialize_bw(p = 3, hidden_p = 2, K = 0))
})

test_that("initialize_bw works correctly with default and custom scale values", {
  out <- initialize_bw(p = 3, hidden_p = 2, K = 2)
  W1_sd <- sd(as.vector(out$W1))
  W2_sd <- sd(as.vector(out$W2))
  expect_true(abs(W1_sd - 1e-3) < 1e-3)
  expect_true(abs(W2_sd - 1e-3) < 1e-3)
  
  out <- initialize_bw(p = 3, hidden_p = 2, K = 2, scale = 0.05)
  W1_sd <- sd(as.vector(out$W1))
  W2_sd <- sd(as.vector(out$W2))
  expect_true(abs(W1_sd - 0.05) < 5e-2)
  expect_true(abs(W2_sd - 0.05) < 5e-2)
})

test_that("initialize_bw with incorrect inputs", {
  expect_error(initialize_bw(p = "a", hidden_p = 2, K = 2))
  expect_error(initialize_bw(p = -3, hidden_p = 2, K = 2))
  expect_error(initialize_bw(p = 3, hidden_p = 2, K = 2, scale = "a"))
  expect_error(initialize_bw(p = 3.5, hidden_p = 2, K = 2))
})

#############################
# Tests on loss_grad_scores #
#############################

test_that("loss_grad_scores works correctly", {
  scores <- matrix(1:3, nrow = 1)
  out <- loss_grad_scores(y = c(2), scores = scores, K = 3)
  expect_equal(dim(out$grad), dim(scores))
  expect_equal(out$error, 0)
  
  scores <- matrix(c(0.1, 0.5, 0.4, 0.2, 0.3, 0.5),
                   nrow = 2, byrow = TRUE)
  out <- loss_grad_scores(y = c(0, 2), scores = scores, K = 3)
  expect_equal(dim(out$grad), dim(scores))
})

test_that("Verify output type and length from loss_grad_scores", {
  scores <- matrix(c(2, 1, 0.5, 0.2, 0.3, 0.1),
                   nrow = 2, byrow = TRUE)
  out <- loss_grad_scores(y = c(0, 1), scores = scores, K = 3)
  expect_type(out, "list")
  expect_named(out, c("loss", "grad", "error"))
  expect_type(out$loss, "double")
  expect_length(out$loss, 1)
  expect_equal(dim(out$grad), dim(scores))
  expect_type(out$error, "double")
  expect_length(out$error, 1)
})

test_that("loss_grad_scores with incorrect inputs", {
  scores <- matrix(c(0.1, 0.5, 0.4, 0.2, 0.3, 0.5),
                   nrow = 2, byrow = TRUE)
  y <- c(0, 2)
  K <- 3
  
  expect_error(loss_grad_scores(y = y,
                                scores = matrix(c("a", "b", "c", "d"), nrow = 2),
                                K = K))
  expect_error(loss_grad_scores(y = c(0),
                                scores = scores,
                                K = K))
  expect_error(loss_grad_scores(y = y,
                                scores = scores,
                                K = 2))
})

#####################
# Tests on one_pass #
#####################

test_that("Checking output length, dimensions, name, and data type for one_pass", {
  X <- matrix(c(1, 0, -1, 2, -2, 0.5),
              nrow = 3, byrow = TRUE)
  y <- c(0, 1, 2)
  K <- 3
  W1 <- matrix(rnorm(6, mean = 0, sd = 1e-3), nrow = 2)
  b1 <- rep(0, 3)
  W2 <- matrix(rnorm(9, mean = 0, sd = 1e-3), nrow = 3)
  b2 <- rep(0, 3)
  lambda <- 0.01
  
  out <- one_pass(X = X,
                  y = y,
                  W1 = W1,
                  K = K,
                  b1 = b1,
                  W2 = W2,
                  b2 = b2,
                  lambda = lambda)
  
  expect_type(out, "list")
  expect_named(out, c("loss", "error", "grads"))
  expect_type(out$loss, "double")
  expect_type(out$error, "double")
  expect_named(out$grads, c("dW1", "db1", "dW2", "db2"))
  expect_length(out$grads$db1, length(b1))
  expect_length(out$grads$db2, length(b2))
  expect_equal(dim(out$grads$dW1), dim(W1))
  expect_equal(dim(out$grads$dW2), dim(W2))
})

test_that("Testing one_pass with incorrect inputs", {
  X <- matrix(1, nrow = 3, ncol = 2)
  W1 <- matrix(1, nrow = 3, ncol = 3)
  b1 <- rep(0, 3)
  W2 <- matrix(1, nrow = 3, ncol = 3)
  b2 <- rep(0, 3)
  y <- c(0:2)
  lambda <- 0.01
  K <- 3
  
  expect_error(one_pass(X = X,
                        y = y,
                        K = K,
                        W1 = W1,
                        b1 = b1,
                        W2 = W2,
                        b2 = b2,
                        lambda = lambda))
  expect_error(one_pass(X = X,
                        y = y,
                        K = K,
                        W1 = matrix(1, nrow = 2, ncol = 3),
                        b1 = b1,
                        W2 = W2,
                        b2 = b2,
                        lambda = -0.01))
  expect_error(one_pass(X = X,
                        y = c(0, 1, 3),
                        K = K,
                        W1 = matrix(1, nrow = 2, ncol = 3),
                        b1 = b1,
                        W2 = W2,
                        b2 = b2,
                        lambda = lambda))
})

###########################
# Tests on evaluate_error #
###########################

test_that("Testing evalutate_error with incorrect inputs", {
  Xval <- matrix(1:4, nrow = 2)
  W1 <- matrix(1:6, nrow = 3)
  b1 <- rep(0, 3)
  W2 <- matrix(rnorm(9, mean = 0, sd = 1e-3), nrow = 3)
  b2 <- rep(0, 3)
  yval <- 0:1
  
  expect_error(evaluate_error(Xval = Xval,
                              yval = yval,
                              W1 = W1,
                              b1 = b1,
                              W2 = W2,
                              b2 = b2))
  expect_error(evaluate_error(Xval = Xval,
                              yval = 0,
                              W1 = matrix(1:6, nrow = 2),
                              b1 = b1,
                              W2 = W2,
                              b2 = b2))
})

test_that("output type and dimension of evaluate_error with valid inputs", {
  Xval <- matrix(1:4, nrow = 2)
  W1 <- matrix(1:6, nrow = 2)
  b1 <- rep(0, 3)
  W2 <- matrix(rnorm(9, mean = 0, sd = 1e-3), nrow = 3)
  b2 <- rep(0, 3)
  yval <- 0:1
  
  out <- evaluate_error(Xval = Xval,
                        yval = yval,
                        W1 = W1,
                        b1 = b1,
                        W2 = W2,
                        b2 = b2)
  
  expect_type(out, "double")
  expect_length(out, 1)
})

#####################
# Tests on NN_train #
#####################

test_that("Training Error decreases over Epochs", {
  set.seed(0928)
  X <- matrix(rnorm(60), nrow = 20, ncol = 3)
  y <- sample(0:2, 20, replace = TRUE)
  
  out <- NN_train(X = X,
                  y = y,
                  Xval = X,
                  yval = y,
                  lambda = 0.01,
                  rate = 0.1,
                  mbatch = 5,
                  nEpoch = 20,
                  hidden_p = 5,
                  scale = 1e-3,
                  seed = 0928)
  
  expect_true(all(diff(out$error) <= 0))
})

test_that("NN_train works with a single batch", {
  set.seed(0928)
  X <- matrix(rnorm(60), nrow = 20, ncol = 3)
  y <- sample(0:2, 20, replace = TRUE)
  Xval <- matrix(rnorm(15), nrow = 5, ncol = 3)
  yval <- sample(0:2, 5, replace = TRUE)
  
  out <- NN_train(X = X,
                  y = y,
                  Xval = Xval,
                  yval = yval,
                  lambda = 0.01,
                  rate = 0.1,
                  mbatch = 20,
                  nEpoch = 20,
                  hidden_p = 5,
                  scale = 1e-3,
                  seed = 0928)
  
  expect_length(out$error, 20)
  expect_length(out$error_val, 20)
})

test_that("NN_train with invalid inputs", {
  set.seed(0928)
  X <- matrix(rnorm(60), nrow = 20, ncol = 3)
  y <- sample(0:2, 20, replace = TRUE)
  Xval <- matrix(rnorm(15), nrow = 5, ncol = 3)
  yval <- sample(0:2, 5, replace = TRUE)
  
  expect_error(NN_train(X = X,
                        y = y,
                        Xval = Xval,
                        yval = yval,
                        lambda = -0.01,
                        rate = 0.1,
                        mbatch = 20,
                        nEpoch = 20,
                        hidden_p = 5,
                        scale = 1e-3,
                        seed = 0928))
  expect_error(NN_train(X = matrix(rnorm(20), nrow = 5, ncol = 4),
                        y = y,
                        Xval = Xval,
                        yval = yval,
                        lambda = 0.01,
                        rate = 0.1,
                        mbatch = 20,
                        nEpoch = 20,
                        hidden_p = 5,
                        scale = 1e-3,
                        seed = 0928))
  expect_error(NN_train(X = X,
                        y = y,
                        Xval = Xval,
                        yval = yval,
                        lambda = 0.01,
                        rate = 0.1,
                        mbatch = 30,
                        nEpoch = 20,
                        hidden_p = 5,
                        scale = 1e-3,
                        seed = 0928))
  expect_error(NN_train(X = X,
                        y = y,
                        Xval = Xval,
                        yval = yval,
                        lambda = 0.01,
                        rate = 0.1,
                        mbatch = 20,
                        nEpoch = -20,
                        hidden_p = 5,
                        scale = 1e-3,
                        seed = 0928))
})

