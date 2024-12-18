# Initialization
#####################################################
# p - dimension of input layer
# hidden_p - dimension of hidden layer
# K - number of classes, dimension of output layer
# scale - magnitude for initialization of W_k (standard deviation of normal)
# seed - specified seed to use before random normal draws
initialize_bw <- function(p, hidden_p, K, scale = 1e-3, seed = 12345){
  
  ###############
  # Checks on p #
  ###############
  if(!is.numeric(p)) stop("p should be a positive integer")
  if(!is.vector(p)){
    if(!is.matrix(p)) stop("p should contain a single element")
    if(nrow(p) == 1 & ncol(p) == 1){
      p <- as.vector(p)
    } else{
      stop("p should contain a single element")
    }
  }
  if(p != round(p) | p <= 0) stop("Dimension of input layer should be an integer greater than or equal to 1.")
  
  ######################
  # Checks on hidden_p #
  ######################
  if(!is.numeric(hidden_p)) stop("hidden_p should be a positive integer")
  if(!is.vector(hidden_p)){
    if(!is.matrix(hidden_p)) stop("hidden_p should contain a single element")
    if(nrow(hidden_p) == 1 & ncol(hidden_p) == 1){
      hidden_p <- as.vector(hidden_p)
    } else{
      stop("hidden_p should contain a single element")
    }
  }
  if(hidden_p != round(hidden_p) | hidden_p <= 0) stop("Dimension of hidden layer should be an integer greater than or equal to 1.")
  
  ###############
  # Checks on K #
  ###############
  if(!is.numeric(K)) stop("K should be a positive integer")
  if(!is.vector(K)){
    if(!is.matrix(K)) stop("K should contain a single element")
    if(nrow(K) == 1 & ncol(K) == 1){
      K <- as.vector(K)
    } else{
      stop("K should contain a single element")
    }
  }
  if(K != round(K) | K <= 0) stop("Dimension of input layer should be an integer greater than or equal to 1.")
  
  ###################
  # Checks on scale #
  ###################
  if(!is.numeric(scale)) stop("scale should be a positive number")
  if(!is.vector(scale)){
    if(!is.matrix(scale)) stop("scale should contain a single element")
    if(nrow(scale) == 1 & ncol(scale) == 1){
      scale <- as.vector(scale)
    } else{
      stop("scale should contain a single element")
    }
  }
  if(scale <= 0) stop("scale should be a positive number.")
  
  ##################
  # Checks on seed #
  ##################
  if(!is.numeric(seed)) stop("seed should be a number")
  if(!is.vector(seed)) if(!is.matrix(seed)) stop("seed should be a vector or a matrix")
  
  
  set.seed(seed)
  
  # [ToDo] Initialize intercepts as zeros
  b1 <- rep(0, hidden_p)
  b2 <- rep(0, K)
  
  # [ToDo] Initialize weights by drawing them iid from Normal
  # with mean zero and scale as sd
  W1 <- matrix(rnorm(p * hidden_p, mean = 0, sd = scale), nrow = p, ncol = hidden_p)
  W2 <- matrix(rnorm(hidden_p * K, mean = 0, sd = scale), nrow = hidden_p, ncol = K)
  
  # Return
  return(list(b1 = b1, b2 = b2, W1 = W1, W2 = W2))
}

# Function to calculate loss, error, and gradient strictly based on scores
# with lambda = 0
#############################################################
# scores - a matrix of size n by K of scores (output layer)
# y - a vector of size n of class labels, from 0 to K-1
# K - number of classes
loss_grad_scores <- function(y, scores, K){
  
  ###############
  # Checks on y #
  ###############
  if(!all(is.numeric(y))) stop("All elements of y should be numeric.")
  if(length(y) != nrow(scores)) stop("Length of y should be equal to the number of rows in scores")
  # if(min(y) != 0 & max(y) != (K - 1)) stop("Classes should range from 0 to K-1")
  # if(length(unique(y)) != K) stop("All classes from 0 to K-1 should be present in y")
  # Note: Commenting these tests as there are batches being passed to one_pass
  # and this to loss_grad_scores which do not contain all the classes
  if(!(min(y) >= 0) | !(max(y) <= (K - 1))) stop("Classes should range from 0 to K-1")
  if(!is.vector(y)){
    if(!is.matrix(y)) stop("y should be a vector or a column matrix")
    if(ncol(y) == 1){
      y <- as.vector(y)
    } else{
      stop("y should be a vector or a column matrix")
    }
  }
  
  ###############
  # Checks on K #
  ###############
  if(!is.numeric(K)) stop("K should be a positive integer")
  if(!is.vector(K)){
    if(!is.matrix(K)) stop("K should contain a single element")
    if(nrow(K) == 1 & ncol(K) == 1){
      K <- as.vector(K)
    } else{
      stop("K should contain a single element")
    }
  }
  if(K != round(K) | K <= 0) stop("K should be an integer greater than or equal to 1.")
  
  ####################
  # Checks on scores #
  ####################
  if(!all(is.numeric(scores))) stop("All elements of scores should be numeric")
  if(!is.matrix(scores)) stop("scores should be a matrix")
  if(ncol(scores) != K) stop("Number of columns of scores should be equal to the number of classes")
  
  
  n <- length(y) # Length of y
  
  temp_mat <- matrix(rep(0, n * K), nrow = n, ncol = K)
  for(i in 1:n) temp_mat[i, y[i] + 1] <- 1 # We add 1 to the column index as R indexes start from 1 where elements of y start from 0.
  probs <- 1.0 * exp(scores) / rowSums(exp(scores))
  
  # [ToDo] Calculate loss when lambda = 0
  loss = - sum(temp_mat * log(probs)) / n
  
  # [ToDo] Calculate misclassification error rate (%)
  # when predicting class labels using scores versus true y
  pred_classes <- max.col(probs) - 1 # (-1) to adjust for R indexes and class labels.
  error = mean(pred_classes != y) * 100
  
  # [ToDo] Calculate gradient of loss with respect to scores (output)
  # when lambda = 0
  grad = (probs - temp_mat) * 1.0 / n
  
  # Return loss, gradient and misclassification error on training (in %)
  return(list(loss = loss, grad = grad, error = error))
}

# One pass function
################################################
# X - a matrix of size n by p (input)
# y - a vector of size n of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
# lambda - a non-negative scalar, ridge parameter for gradient calculations
one_pass <- function(X, y, K, W1, b1, W2, b2, lambda){

  if(!all(is.numeric(X))) stop("All elements of X should be numeric")
  if(!all(is.numeric(y))) stop("All elements of y should be numeric")
  if(!all(is.numeric(W1))) stop("All elements of W1 should be numeric")
  if(!all(is.numeric(b1))) stop("All elements of b1 should be numeric")
  if(!all(is.numeric(W2))) stop("All elements of W2 should be numeric")
  if(!all(is.numeric(b2))) stop("All elements of b2 should be numeric")
  if(!is.numeric(lambda)) stop("Lambda should be numeric")
  if(!is.numeric(K)) stop("K should be a positive integer")
  
  if(!(min(y) >= 0) | !(max(y) <= (K - 1))) stop("Classes should range from 0 to K-1")
  if(!is.vector(y)){
    if(!is.matrix(y)) stop("y should be a vector or a column matrix")
    if(ncol(y) == 1){
      y <- as.vector(y)
    } else{
      stop("y should be a vector or a column matrix")
    }
  }
  
  if(!is.vector(K)){
    if(!is.matrix(K)) stop("K should contain a single element")
    if(nrow(K) == 1 & ncol(K) == 1){
      K <- as.vector(K)
    } else{
      stop("K should contain a single element")
    }
  }
  if(K != round(K) | K <= 0) stop("K should be an integer greater than or equal to 1.")
  
  if(!is.vector(lambda)){
    if(!is.matrix(lambda)) stop("lambda should contain a single element")
    if(nrow(lambda) == 1 & ncol(lambda) == 1){
      lambda <- as.vector(lambda)
    } else{
      stop("lambda should contain a single element")
    }
  }
  if(lambda < 0) stop("lambda should be non-negative")
  
  if(!(length(K) == length(lambda)) | !(length(K) == 1)) stop("K and lambda should contain single numeric elements.")
  if(nrow(X) != length(y)) stop("Number of rows of X should be equal to the number of elements in y.")
  if(ncol(X) != nrow(W1)) stop("Number of columns of X should be equal to the number of rows of W1.")
  if(ncol(W1) != length(b1)) stop("Number of columns of W1 should be equal to the number of elements in b1.")
  if(length(b1) != nrow(W2)) stop("Number of rows of W2 should be equal to the number of elements in b1.")
  if(ncol(W2) != K) stop("Number of columns of W2 should be equal to K.")
  if(length(b2) != K) stop("Number of elements in b2 should be equal to K.")
  
  # [To Do] Forward pass
  # From input to hidden 
  hidden <- matrix(pmax(0, X %*% W1 + matrix(b1, nrow = nrow(X), ncol = length(b1), byrow = TRUE)), nrow = nrow(X))
  
  # ReLU
  
  # From hidden to output scores
  scores <- hidden %*% W2 + matrix(b2, nrow =  nrow(hidden), ncol = length(b2), byrow = TRUE)
  
  # [ToDo] Backward pass
  # Get loss, error, gradient at current scores using loss_grad_scores function
  out <- loss_grad_scores(y = y, scores = scores, K = K)
  grad_scores <- out$grad
  
  dW2 <- t(hidden) %*% grad_scores + lambda * W2
  db2 <- colSums(grad_scores)
  
  hidden_grad <- grad_scores %*% t(W2)
  hidden_grad[hidden <= 0] <- 0 # Zero for non-positive inputs
  
  dW1 <- t(X) %*% hidden_grad + lambda * W1
  db1 <- colSums(hidden_grad)

  # Get gradient for 2nd layer W2, b2 (use lambda as needed)
  
  # Get gradient for hidden, and 1st layer W1, b1 (use lambda as needed)
  
  # Return output (loss and error from forward pass,
  # list of gradients from backward pass)
  return(list(loss = out$loss, error = out$error, grads = list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)))
}

# Function to evaluate validation set error
####################################################
# Xval - a matrix of size nval by p (input)
# yval - a vector of size nval of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
evaluate_error <- function(Xval, yval, W1, b1, W2, b2){
  
  if(!all(is.numeric(Xval))) stop("All elements of Xval should be numeric.")
  if(!all(is.numeric(yval))) stop("All elements of yval should be numeric.")
  if(!all(is.numeric(W1))) stop("All elements of W1 should be numeric.")
  if(!all(is.numeric(b1))) stop("All elements of b1 should be numeric.")
  if(!all(is.numeric(W2))) stop("All elements of W2 should be numeric.")
  if(!all(is.numeric(b2))) stop("All elements of b2 should be numeric.")
  
  if(!is.matrix(Xval)) stop("Xval should be a matrix.")
  if(!is.matrix(W1)) stop("W1 should be a matrix.")
  if(!is.matrix(W2)) stop("W2 should be a matrix.")
  if(!is.vector(yval)) if(!is.matrix(yval)){
    stop("yval should be a vector")
  } else{
    if(ncol(yval) != 1){
      stop("yval should be a vector")
    } else yval <- as.vector(yval)
  }
  if(!is.vector(b1)) if(!is.matrix(b1)){
    stop("b1 should be a vector")
  } else{
    if(ncol(b1) != 1){
      stop("b1 should be a vector")
    } else b1 <- as.vector(b1)
  }
  if(!is.vector(b2)) if(!is.matrix(b2)){
    stop("b2 should be a vector")
  } else{
    if(ncol(b2) != 1){
      stop("b2 should be a vector")
    } else b2 <- as.vector(b2)
  }

  if(nrow(Xval) != length(yval)) stop("Number of rows of Xval should be equal to the number of elements in yval.")
  if(ncol(Xval) != nrow(W1)) stop("Number of columns of Xval should be equal to the number of rows of W1.")
  if(ncol(W1) != length(b1)) stop("Number of columns of W1 should be equal to the number of elements in b1.")
  if(ncol(W1) != nrow(W2)) stop("Number of columns of W1 should be equal to the number of rows of W2.")
  if(ncol(W2) != length(b2)) stop("Number of columns of W2 should be equal to the number of elements of b2.")
  if(!(min(yval) >= 0) | !(max(yval) <= (ncol(W2) - 1))) stop("Elements of yval should range from 0 to K-1.")
  if(!all(yval == round(yval))) stop("All elements of yval should be integers.")
  
  # [ToDo] Forward pass to get scores on validation data
  hidden <- matrix(pmax(0, Xval %*% W1 + matrix(b1, nrow = nrow(Xval), ncol = length(b1), byrow = TRUE)), nrow = nrow(Xval))

  scores <- hidden %*% W2 + matrix(b2, nrow = nrow(hidden), ncol = length(b2), byrow = TRUE)
  
  probs <- exp(scores) / rowSums(exp(scores)) # Compute probabilities
  pred_classes <- max.col(probs) - 1 # Predict classes
  
  # [ToDo] Evaluate error rate (in %) when 
  # comparing scores-based predictions with true yval
  error <- mean(pred_classes != yval) * 100 # Calculate classification error in percentage
  
  return(error)
}


# Full training
################################################
# X - n by p training data
# y - a vector of size n of class labels, from 0 to K-1
# Xval - nval by p validation data
# yval - a vector of size nval of of class labels, from 0 to K-1, for validation data
# lambda - a non-negative scalar corresponding to ridge parameter
# rate - learning rate for gradient descent
# mbatch - size of the batch for SGD
# nEpoch - total number of epochs for training
# hidden_p - size of hidden layer
# scale - a scalar for weights initialization
# seed - for reproducibility of SGD and initialization
NN_train <- function(X, y, Xval, yval, lambda = 0.01,
                     rate = 0.01, mbatch = 20, nEpoch = 100,
                     hidden_p = 20, scale = 1e-3, seed = 12345){
  
  if(!all(is.numeric(X))) stop("All elements of X should be numeric.")
  if(!all(is.numeric(y))) stop("All elements of y should be numeric.")
  if(!all(is.numeric(Xval))) stop("All elements of Xval should be numeric.")
  if(!all(is.numeric(yval))) stop("All elements of Yval should be numeric.")
  if(!is.numeric(lambda)) stop("Lambda should be numeric.")
  if(!is.numeric(rate)) stop("rate should be numeric.")
  if(!is.numeric(mbatch)) stop("mbatch should be numeric.")
  if(!is.numeric(nEpoch)) stop("nEpoch should be numeric.")
  if(!is.numeric(hidden_p)) stop("hidden_p should be numeric.")
  if(!is.numeric(scale)) stop("scale should be numeric.")
  if(!is.numeric(seed)) stop("seed should be numeric.")
  
  K <- length(unique(y))
  
  if(!(min(y) == 0) | !(max(y) == (K - 1))) stop("Classes in y should range from 0 to K-1")
  if(!is.vector(y)){
    if(!is.matrix(y)) stop("y should be a vector or a column matrix")
    if(ncol(y) == 1){
      y <- as.vector(y)
    } else{
      stop("y should be a vector or a column matrix")
    }
  }
  
  if(!(min(yval) == 0) | !(max(yval) == (K - 1))) stop("Classes in yval should range from 0 to K-1")
  if(!is.vector(yval)){
    if(!is.matrix(yval)) stop("yval should be a vector or a column matrix")
    if(ncol(yval) == 1){
      y <- as.vector(yval)
    } else{
      stop("yval should be a vector or a column matrix")
    }
  }
  
  if(!is.matrix(X)) stop("X should be matrix")
  if(!is.matrix(Xval)) stop("Xval should be matrix")
  if(nrow(X) != length(y)) stop("Number of rows of X should be equal to the length of y.")
  if(nrow(Xval) != length(yval)) stop("Number of rows of Xval should be equal to the length of yval.")
  if(ncol(X) != ncol(Xval)) stop("X and Xval should have the same number of columns.")
  
  if(!is.vector(lambda)){
    if(!is.matrix(lambda)) stop("lambda should contain a single element")
    if(nrow(lambda) == 1 & ncol(lambda) == 1){
      lambda <- as.vector(lambda)
    } else{
      stop("lambda should contain a single element")
    }
  }
  if(lambda < 0) stop("lambda should be non-negative")
  
  if(!is.vector(rate)){
    if(!is.matrix(rate)) stop("rate should contain a single element")
    if(nrow(rate) == 1 & ncol(rate) == 1){
      rate <- as.vector(rate)
    } else{
      stop("rate should contain a single element")
    }
  }
  
  if(!is.vector(mbatch)){
    if(!is.matrix(mbatch)) stop("mbatch should contain a single element")
    if(nrow(mbatch) == 1 & ncol(mbatch) == 1){
      mbatch <- as.vector(mbatch)
    } else{
      stop("mbatch should contain a single element")
    }
  }
  if(!(mbatch <= nrow(X))) stop("mbatch should be less than or equal to the number of rows in X.")
  
  if(!is.vector(nEpoch)){
    if(!is.matrix(nEpoch)) stop("nEpoch should contain a single element")
    if(nrow(nEpoch) == 1 & ncol(nEpoch) == 1){
      nEpoch <- as.vector(nEpoch)
    } else{
      stop("nEpoch should contain a single element")
    }
  }
  if(nEpoch != round(nEpoch) | nEpoch <= 0) stop("nEpoch should be a positive integer.")
  
  if(!is.vector(hidden_p)){
    if(!is.matrix(hidden_p)) stop("hidden_p should contain a single element")
    if(nrow(hidden_p) == 1 & ncol(hidden_p) == 1){
      hidden_p <- as.vector(hidden_p)
    } else{
      stop("hidden_p should contain a single element")
    }
  }
  
  if(!is.vector(scale)){
    if(!is.matrix(scale)) stop("scale should contain a single element")
    if(nrow(scale) == 1 & ncol(scale) == 1){
      scale <- as.vector(scale)
    } else{
      stop("scale should contain a single element")
    }
  }
  
  if(!is.vector(seed)) if(!is.matrix(seed)) stop("seed should be a vector or a matrix")
  
  # Get sample size and total number of batches
  n = length(y)
  nBatch = floor(n / mbatch)

  # [ToDo] Initialize b1, b2, W1, W2 using initialize_bw with seed as seed,
  # and determine any necessary inputs from supplied ones
  with(initialize_bw(p = ncol(X), 
                     hidden_p = hidden_p, 
                     K = max(y) + 1, 
                     scale = scale, 
                     seed = seed), {
    b1 <<- b1
    b2 <<- b2
    W1 <<- W1
    W2 <<- W2
  })
  
  # Initialize storage for error to monitor convergence
  error = rep(0, nEpoch)
  error_val = rep(0, nEpoch)
  
  # Set seed for reproducibility
  set.seed(seed)
  # Start iterations
  for (i in 1:nEpoch){
    # Allocate bathes
    batchids = sample(rep(1:nBatch, length.out = n), size = n)
    
    # Variables to store loss and error for each epoch
    epoch_loss <- 0
    epoch_error <- 0
    
    # [ToDo] For each batch
    #  - do one_pass to determine current error and gradients
    #  - perform SGD step to update the weights and intercepts
    for (batch in 1:nBatch){
      # Extract batch data
      batch_idx <- which(batchids == batch)
      X_batch <- X[batch_idx, , drop = FALSE]
      y_batch <- y[batch_idx]
      
      # Perform 1 forward-backward pass
      out <- one_pass(X = X_batch, 
                      y = y_batch, 
                      K = max(y) + 1, 
                      W1 = W1, 
                      b1 = b1, 
                      W2 = W2, 
                      b2 = b2, 
                      lambda = lambda)
      epoch_loss <- epoch_loss + out$loss
      epoch_error <- epoch_error + out$error
      
      # Update parameters using gradients
      grads <- out$grads
      with(grads, {
        W1 <<- W1 - rate * dW1
        W2 <<- W2 - rate * dW2
        b1 <<- b1 - rate * db1
        b2 <<- b2 - rate * db2
      })
    }

    # [ToDo] In the end of epoch, evaluate
    # - average training error across batches
    # - validation error using evaluate_error function
    error[i] <- epoch_error / nBatch # Average error for the epoch
    error_val[i] <- evaluate_error(Xval = Xval, 
                                   yval = yval, 
                                   W1 = W1, 
                                   b1 = b1, 
                                   W2 = W2, 
                                   b2 = b2) # Validation error
  }
  # Return end result
  return(list(error = error, error_val = error_val, params =  list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)))
}