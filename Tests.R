# Source the NN function
source("FunctionsNN.R")

library(testthat)

test_that("initialize_bw works correctly", {
  
  expect_error(initialize_bw(p = 0, hidden_p = 2, K = 2))
  expect_error(initialize_bw(p = 3, hidden_p = 0, K = 2))
  expect_error(initialize_bw(p = 3, hidden_p = 2, K = 0))
  
})