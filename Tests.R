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

