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

