# Function that implements multi-class logistic regression.
#############################################################
# Description of supplied parameters:
# X - n x p training data, 1st column should be 1s to account for intercept
# y - a vector of size n of class labels, from 0 to K-1
# Xt - ntest x p testing data, 1st column should be 1s to account for intercept
# yt - a vector of size ntest of test class labels, from 0 to K-1
# numIter - number of FIXED iterations of the algorithm, default value is 50
# eta - learning rate, default value is 0.1
# lambda - ridge parameter, default value is 1
# beta_init - (optional) initial starting values of beta for the algorithm, should be p x K matrix 

## Return output
##########################################################################
# beta - p x K matrix of estimated beta values after numIter iterations
# error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
# error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
# objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
LRMultiClass <- function(X, y, Xt, yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL){
  ## Check the supplied parameters as described. You can assume that X, Xt are matrices; y, yt are vectors; and numIter, eta, lambda are scalars. You can assume that beta_init is either NULL (default) or a matrix.
  ###################################
  
  ###############
  # Checks on X #
  ###############
  
  # Check to ensure X is a matrix or a dataframe
  if(!is.matrix(X)){
    if(!is.data.frame(X)){
      stop("X should be a matrix or a data frame.")
    }else{
      X <- as.matrix(X)
    }
  }
  # Check to ensure X isn't empty
  if(nrow(X) == 0 | ncol(X) == 0) stop("X must be a non-empty matrix.")
  # Check to ensure X does not have any NAs or non-numeric values.
  if(any(is.na(X)) | !all(is.numeric(X))) stop("All elements of X should be numeric.")
  # Check that the first column of X are 1s, if not - display appropriate message and stop execution.
  if(any(X[ , 1] != 1)) stop("First column of X must all be 1s to account for intercept.")
  
  ################
  # Checks on Xt #
  ################
  
  # Check to ensure Xt is a matrix or a dataframe
  if(!is.matrix(Xt)){
    if(!is.data.frame(Xt)){
      stop("Xt should be a matrix or a data frame.")
    }else{
      Xt <- as.matrix(Xt)
    }
  }
  # Check to ensure Xt isn't empty
  if(nrow(Xt) == 0 | ncol(Xt) == 0) stop("Xt must be a non-empty matrix.")
  # Check to ensure Xt does not have any NAs or non-numeric values.
  if(any(is.na(Xt)) | !all(is.numeric(Xt))) stop("All elements of Xt should be numeric.")
  # Check that the first column of Xt are 1s, if not - display appropriate message and stop execution.
  if(any(Xt[ , 1] != 1)) stop("First column of Xt must all be 1s to account for intercept.")
  
  ###############
  # Checks on y #
  ###############
  
  # Check to ensure y is a vector or a matrix with 1 column
  if(!is.vector(y)){
    if(is.matrix(y) & ncol(y) == 1){
      y <- as.vector(y)
    }else{
      stop("y should be a vector or a matrix with 1 column.")
    }
  }
  # Check to ensure y does not have any NAs or non-numeric values.
  if(any(is.na(y)) | !all(is.numeric(y))) stop("All elements of y should be numeric.")
  # Check to ensure y is not an empty vector
  if(length(y) == 0) stop("y must be a non-empty vector.")
  # Check to ensure y contains values from 0 to K-1
  if(any(y < 0) | any(y >= length(unique(y)))) stop("y must contain class labels from 0 to K - 1.")
  
  ################
  # Checks on yt #
  ################
  
  # Check to ensure yt is a vector or a matrix with 1 column
  if(!is.vector(yt)){
    if(is.matrix(yt) & ncol(yt) == 1){
      yt <- as.vector(yt)
    }else{
      stop("yt should be a vector or a matrix with 1 column.")
    }
  }
  # Check to ensure yt does not have any NAs or non-numeric values.
  if(any(is.na(yt)) | !all(is.numeric(yt))) stop("All elements of yt should be numeric.")
  # Check to ensure yt is not an empty vector
  if(length(yt) == 0) stop("yt must be a non-empty vector.")
  # Check to ensure yt contains values from 0 to K-1
  if(any(yt < 0) | any(yt >= length(unique(yt)))) stop("yt must contain class labels from 0 to K - 1.")
  
  #####################
  # Checks on numIter #
  #####################
  # Check to ensure numIter is not NA or non-numeric
  if(is.na(numIter) | !is.numeric(numIter)) stop("numIter must be a single positive integer.")
  # Check if numIter is a single positive integer
  if(length(numIter) != 1 | numIter <= 0 | numIter != as.integer(numIter)) stop(" numIter must be a positive integer.")
  
  #################
  # Checks on eta #
  #################
  # Check to ensure eta is not NA or non-numeric
  if(is.na(eta) | !is.numeric(eta) | length(eta) != 1 | is.infinite(eta)) stop("eta must be a single positive number.")
  # Check eta is positive
  if(eta <= 0) stop("Learning rate (eta) should strictly be positive.")
  
  ####################
  # Checks on lambda #
  ####################
  # Check to ensure lambda is not NA or non-numeric
  if(is.na(lambda) | !is.numeric(lambda) | length(lambda) != 1 | is.infinite(lambda)) stop("lambda must be a single non-negative number.")
  # Check lambda is non-negative
  if(lambda < 0) stop("Ridge parameter (lambda) should strictly be non-negative.")
  
  ###########################################
  # Dimension check within X, Xt, y, and yt #
  ###########################################
  # Check for compatibility of dimensions between X and Y
  if(length(y) != nrow(X)) stop("Length of y and number of rows of X should be equal.")
  # Check for compatibility of dimensions between Xt and Yt
  if(length(yt) != nrow(Xt)) stop("Length of yt and number of rows of Xt should be equal.")
  # Check for compatibility of dimensions between X and Xt
  if(ncol(X) != ncol(Xt)) stop("Number of columns of X and Xt should be equal.")
  
  #######################
  # Checks on beta_init #
  #######################
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes. If not NULL, check for compatibility of dimensions with what has been already supplied.
  p <- ncol(X)
  K <- length(unique(y))
  if(is.null(beta_init)){
    beta <- matrix(rep(0, p * K), p, K)
  } else{
    if(!all(dim(beta_init) == c(p, K))) stop("Number of rows of beta_init should be equal to the number of columns of X and number of columns of beta_init should be equal to the number of classes.")
    if(!is.matrix(beta_init)){
      if(!is.data.frame(beta_init)){
        stop("beta_init must be a matrix or data frame of dimension (p x K).")
      }else{
        beta <- as.matrix(beta_init)
      }
    }
    beta <- beta_init
  }
  # Check if any element of beta is NA or non-numeric
  if(any(is.na(beta)) | any(!is.numeric(beta))) stop("No values of beta_init can be NA or non-numeric.")
  
  
  ## Calculate corresponding pk, objective value f(beta_init), training error and testing error given the starting point beta_init
  ##########################################################################
  # Declaring the structures of objective, error_train, and error_test
  objective <- vector(mode = 'numeric', length = (numIter + 1))
  error_train <- vector(mode = 'numeric', length = (numIter + 1))
  error_test <- vector(mode = 'numeric', length = (numIter + 1))
  
  # Calculating initial values of pk, objective, and train and test errors
  pk <- cal_pk(X, beta)
  objective[1] <- cal_obj(X, y, beta, lambda, pk)
  error_train[1] <- cal_err(X, y, beta, pk)
  error_test[1] <- cal_err(Xt, yt, beta)
  
  # Calculating indicating values. Added it outside of for loop because it does not change with iterations
  indicator <- sapply(0:(K - 1), function(j) as.numeric(y == j))
  
  # Added the following line outside for loop because it does not change with iterations
  lambda_diag <- lambda * diag(p)
  
  ## Newton's method cycle - implement the update EXACTLY numIter iterations
  ##########################################################################
  for(i in 1:numIter){
    
    # Gradient, Hessian, and beta calculations
    gradient <- - crossprod(X, (indicator - pk)) + lambda * beta
    wt <- pk * (1 - pk)
    # Profvis helped me to target crossprod to reduce time.
    # The output of crossprod(X_wt) is the same as the output of crossprod(X, wt[ , j] * X). To verify
    # this compare the two outputs after rounding them as the values are not exactly equal because of 
    # difference in precision in calculation of square root and that of direct multiplication. I rounded the outputs
    # to 4 digits after the decimal for my comparison.
    for(j in 1:K){
      X_wt <- X * sqrt(wt[ , j])
      hessian <- solve(crossprod(X_wt) + lambda_diag)
      beta[ , j] <- beta[ , j] - eta * hessian %*% gradient[ , j]
    }
    
    # Within one iteration: perform the update, calculate updated objective function and training/testing errors in %
    pk <- cal_pk(X, beta)
    error_train[i + 1] <- cal_err(X, y, beta, pk)
    error_test[i + 1] <- cal_err(Xt, yt, beta)
    objective[i + 1] <- cal_obj(X, y, beta, lambda, pk)
    
    # Terminate the loop if we've a convergence on the objective value
    if(abs(objective[i + 1] - objective[i]) < 1e-05){
      cat("Converged after", i, "iterations.\n")
      break
    }
    
  }
  ## Return output
  ##########################################################################
  # beta - p x K matrix of estimated beta values after numIter iterations
  # error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
  # error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
  # objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
  return(list(beta = beta, error_train = error_train, error_test = error_test, objective =  objective))
}

# Calculation of pk as a separate function since it is a repetitive calculation.
# This function is another point that consumes a significant amount of time in this code.
cal_pk <- function(X, beta){
  pk <- exp(X %*% beta)
  return(pk / rowSums(pk))
}

# Calculation of objective as a separate function since it is a repetitive calculation.
cal_obj <- function(X, Y, beta, lambda, pk){
  return( - (sum(log(pk[cbind(1:nrow(X), Y + 1)]))) + ((lambda / 2) * sum(beta ^ 2)))
}

# Calculation of error as a separate function since it is a repetitive calculation and it 
# is used to calculate both error_train and error_test. Here we've give pk a default value of null
# as we're calculating pk in the for loop which saves time but in case of error_test pk is calculated
# on Xt and y instead of X and y and hence, this calculation is done separately from this point.
cal_err <- function(X, Y, beta, pk = NULL){
  if(is.null(pk)) pk <- cal_pk(X, beta)
  pred <- max.col(pk) - 1
  return(mean(pred != Y) * 100)
}