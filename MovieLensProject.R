################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

###############################################################################################

# Create train and test sets within the training data for tuning
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = F)
edx_train <- edx[-test_index,]
edx_test_temp <- edx[test_index,]

# Make sure userId and movieId in edx_test set are also in edx_train set
edx_test <- edx_test_temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Add rows removed from edx_test set back into edx_train set
removed <- anti_join(edx_test_temp, edx_test)
edx_train <- rbind(edx_train, removed)

# Remove temporary tables and variables
rm(test_index, edx_test_temp, removed)

# Determine the average rating across all movies, from all users  
mu <- mean(edx_train$rating)
mu

# Generate a sequence of lambdas for tuning. Lambda will be used as a penalty, in order to 
# reduce the weight of ratings from a small number of users
lambdas <- seq(0, 10, 0.25)

# Test various values of lambda
rmses <- sapply(lambdas, function(l){
  # Determine how each movie's average rating deviates from the average for all movies (movie effect)
  b_i_tune <- edx_train %>% 
    group_by(movieId) %>%
    summarize(b_i_tune = sum(rating - mu)/(n()+l))
  # Join the movie effect data back to the training data and determine user effect
  b_u_tune <- edx_train %>% 
    left_join(b_i_tune, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u_tune = sum(rating - b_i_tune - mu)/(n()+l))
  # Join movie effect and user effect data to the partitioned test data (from the training data set)
  predicted_ratings_tune <- edx_test %>% 
    left_join(b_i_tune, by = "movieId") %>%
    left_join(b_u_tune, by = "userId") %>%
    mutate(pred = mu + b_i_tune + b_u_tune) %>%
    .$pred

  # Evaluate RMSE of remaining predictions and corresponding actual ratings from the test data
  return(RMSE(predicted_ratings_tune, edx_test$rating))
})

# Identify lambda with smallest RMSE
qplot(lambdas, rmses)
best_lambda <- lambdas[which.min(rmses)]
best_lambda

# Use the best lambda, movie effects, and user averages to generate predicted ratings for movies in validation data set

# Determine how each movie's average rating deviates from the average for all movies (movie effect)
b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n() + best_lambda))
# Join the movie effect data back to the training data and determine user effect
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n() + best_lambda))
# Join movie effect and user effect data to the validation data
predicted_ratings <- 
  validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

# Evaluate accuracy of predicted ratings for validation data against actual ratings for validation data, outputting RMSE
model_rmse <- RMSE(predicted_ratings, validation$rating)
model_rmse
#0.8648201
