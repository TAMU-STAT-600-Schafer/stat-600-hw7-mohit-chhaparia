# Source the NN function
source("FunctionsNN.R")

library(testthat)

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