## Access test file and preprocess its data

library(mxnet)
library(tidyr)
library(dplyr)
library(tibble)

#mx.set.seed(42)

#' Generates the model.
#' The model takes as input successive bytes,
#' and predicts the following byte.
#'
#' @return
#' @export
#'
#' @examples
getModel <- function() {
  data <- mx.symbol.Variable("data")
  fc1 <- mx.symbol.FullyConnected(data, num.hidden = 100)
  act1 <- mx.symbol.Activation(fc1, act.type = "relu")
  fc2 <- mx.symbol.FullyConnected(act1, num.hidden = 255)
  out <- mx.symbol.LogisticRegressionOutput(fc2)
  return (out)
}


#' Open input file connection
#'
#' @return a connection
#' @export
#'
#' @examples
getConnection <- function() {
  return (file("dataset/hugo.txt", "rb"))
}


#' Title
#'
#' @param con Connection
#' @param debug = FALSE : Do we print debug info ?
#' @param size = 8000 : fragment size
#'
#' @return A fragment (string), or a zero length string if eof
#' @export
#'
#' @examples
getFragment <- function(con, debug = FALSE, size = 8000) {
  c <- readChar(con, size)
  if (length(c) == 0)
    return (c)
  c <- gsub("[\n\r ]+", " ", c)
  if (debug)
    print(c)
  return (c)
}



#' Convert a fragment to an array of substrings, 
#' each encoded a a matrix per byte value.
#' The final array dim is :
#'  255  x LengthOfSubstring x nbOfSubstrings
#'
#' @param fragment : the input text
#' @param window : the substring length (default 5)
#'
#' @return
#' @export
#'
#' @examples
fragmentToArray <- function(fragment, window=5) {
  r <- as.integer(charToRaw(fragment))
  nbs <- length(r) - window
  X <- array(0, dim = c(255,window, nbs))  
  for(i in 1:nbs) {
    for(j in 1:window) {
      X[r[i+j],j,i] <- 1
    }
  }
  return(X)
}

#' Train the model. If it already exists (as a saved file), use the existing model. 
#' If it does not exist, acll getodel to generate a new one.
#'
#' @param fragment - the fragment of the text to use for training
#' @param window - the number of bytes used for prediction, including the byte to predict 
#' (ie : 5 means use the preceeding 4 bytes to predict the 5th).
#'
#' @return nothing
#' @export
#'
#' @examples
trainModel <- function(fragment, window=5) {
  
  arr <- fragmentToArray(frag, window)
  X = matrix(c(arr[,-(window),]),nrow = 255*(window - 1))
  y = matrix(c(arr[,window,]),nrow = 255)
  it <- mx.io.arrayiter(X,y, shuffle = TRUE);
  
  if(file.exists(paste0("m",window,"-symbol.json"))) {
    
    message("Loading existing model")
    my <- mx.model.load(paste0("m",window), 1)
    
    mm <- mx.model.FeedForward.create(my$symbol,
                                      X = it,
                                      ctx = mx.cpu(),
                                      num.round = 10,
                                      epoch.end.callback = mx.callback.log.train.metric(10),
                                      eval.metric = mx.metric.rmse,
                                      array.layout = "colmajor",
                                      learning.rate = 20,
                                      arg.params = my$arg.params,
                                      aux.params = my$aux.params
    )
    
    mx.model.save(mm, paste0("m",window), 1)
    
  } else {
    message("Creating new model")
    mm <- mx.model.FeedForward.create(getModel(),
                                      X = it,
                                      ctx = mx.cpu(),
                                      num.round = 10,
                                      epoch.end.callback = mx.callback.log.train.metric(10),
                                      eval.metric = mx.metric.rmse,
                                      array.layout = "colmajor",
                                      learning.rate = 20
                                      
    )
    
    mx.model.save(mm,paste0("m",window), 1)
    
  }
  

  
}

#' Use saved model to predict the next byte.
#'
#' @param test - a string of exactly window-1 bytes (beware of utf-8 encoding !)
#' @param window - numbre of bytes ued for prdiction plus the predicted byte.
#'
#' @return
#' @export
#'
#' @examples
testPrediction <- function(test) {
  

  p <- as.integer(charToRaw(test))
  window <- 1 + length(p)
  mm <- mx.model.load(paste0("m",window),1)
  Xp <- matrix(0,255,(window-1))
  Xp [p] <- 1
  Xp <- matrix(c(Xp),255*(window - 1),1)
  rr <- predict(mm,Xp,array.layout = "colmajor")
  plot(rr)
  r <- tbl_df(rr) %>% rownames_to_column() %>% mutate(rowname= as.integer(rowname), rn = rowname) 
  r$rowname <- sapply(r$rowname,FUN=intToUtf8)
  r<-arrange(r, desc(V1)) 
  print(paste("Initial text : ",test))
  print(r[1:15,])
  
}



## ================================================= ##

con <- getConnection()

for(i in 1:2) {
  frag <- getFragment(con, debug = TRUE, size = 100000)
  trainModel(frag, window=4)
}

close(con)

testPrediction("ell")
testPrediction("men")




