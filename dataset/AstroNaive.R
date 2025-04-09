# training data : 
df<- read.csv("/Users/roberto/Downloads/classwork-01_train/train.csv")

# Select only features
features <- df[, c("u_g", "g_r", "r_i", "i_z")]

# Correlation matrix
cor_matrix <- cor(features)
cor_matrix



# Load necessary package
library(ggplot2)

# here we select the correlated features
features <- df[, c("u_g", "g_r", "r_i", "i_z")]

# now Scale the features
# PCA is sensitive to scale, so we standardize (mean = 0, sd = 1)
features_scaled <- scale(features)

# Applying PCA...
pca_result <- prcomp(features_scaled, center = TRUE, scale. = TRUE)

# see explained variance by each component
summary(pca_result)
# This shows how much variance each principal component explains.
# Goal: choose number of components that explain ~95% of variance.

# plot to visualize how many components are worth keeping
plot(pca_result, type = "l", main = "Scree Plot")

# Biplot just to visualize contribution of original features
biplot(pca_result, scale = 0)
# Arrows = original features projected into PC space

# Extracting the principal components
pca_data <- as.data.frame(pca_result$x)

# Examine correlation of PCs — should be ~0
round(cor(pca_data), 3)

# The first 2 PCs explain over 90% of the total variance. So, dimensionality reduction is justified and efficient here.


## here we try a different approch ICA : 

# Load ICA package
library(fastICA)

# Step 1: Apply ICA to standardized features
ica_result <- fastICA(features_scaled, n.comp = 4)  # same as #features

# Step 2: Extract the independent components
ica_data <- as.data.frame(ica_result$S)

# Step 3: Check correlations between components
round(cor(ica_data), 3)

# Optional: Visualize
pairs(ica_data, main = "ICA Components")

# ICA goes further than PCA: 
# it tries to produce components that are statistically independent, not just uncorrelated.
# That’s even more aligned with Naive Bayes’ assumptions, 
# though it’s more sensitive to noise and harder to interpret.


### here is a full implementation of the UMAP with deepseek prompt : 
#TRY AN APPROACH WITH UMAP !! nonlinear method ::::: 
#I've a project about statistical learning on classification using naive bayes classifier non parametric ,
#start by finding how to check if the variables are indipendent (decorellated), 
#i was trying with confusion matrix : show me all step in R , consider this dataset given :  
#id       u_g       g_r       r_i       i_z target
#1     1  2.538229  1.350439  0.697413  0.421685      2
#2     2  0.976145  0.330214  0.104654  0.015205      2
#3     3  1.177351 -0.173849 -0.130686 -0.094068      2
#4     6  1.653826  0.620245  0.204937  0.072859      2


# Load required packages
library(ggplot2)
library(umap)
library(infotheo)
library(ks)
library(caret)

# Load and prepare data
data <- df
data$target <- as.factor(data$target)
features <- c("u_g", "g_r", "r_i", "i_z")

# 1. Check variable independence with proper discretization
# ---------------------------------------------------------
# Discretize continuous variables for mutual information
discretized_data <- discretize(data[, features], disc="equalfreq", nbins=5)

# Mutual information analysis
mi_matrix <- matrix(NA, nrow=4, ncol=4, 
                    dimnames=list(features, features))
for(i in 1:4) {
  for(j in 1:4) {
    mi_matrix[i,j] <- mutinformation(discretized_data[,i], discretized_data[,j])
  }
}
print("Mutual Information Matrix:")
print(mi_matrix)

# Correlation analysis (for linear relationships)
cor_matrix <- cor(data[, features])
print("Correlation Matrix:")
print(cor_matrix)

# 2. UMAP Visualization
# ---------------------
set.seed(42)
umap_config <- umap.defaults
umap_config$n_neighbors <- 15
umap_config$min_dist <- 0.1

umap_result <- umap(data[, features], config=umap_config)
umap_df <- data.frame(UMAP1=umap_result$layout[,1], 
                      UMAP2=umap_result$layout[,2],
                      Target=data$target)

ggplot(umap_df, aes(x=UMAP1, y=UMAP2, color=Target)) +
  geom_point(alpha=0.7) +
  ggtitle("UMAP Projection of Features (n_neighbors=15, min_dist=0.1)") +
  theme_minimal()

# 3. Improved Nonparametric Naive Bayes
# -------------------------------------
# Bandwidth selection with cross-validation
optimize_bandwidth <- function(x) {
  h <- tryCatch(
    hpi(x),
    error = function(e) bw.nrd0(x)
  )
  return(max(h, 1e-6))
}

# Train Naive Bayes model with better density estimation
train_nb <- function(train_data) {
  classes <- unique(train_data$target)
  models <- list()
  
  for(cls in classes) {
    class_data <- train_data[train_data$target == cls, features]
    models[[as.character(cls)]] <- list(
      prior = nrow(class_data)/nrow(train_data),
      bandwidths = apply(class_data, 2, optimize_bandwidth),
      data = class_data
    )
  }
  return(models)
}

# Predict with log-sum-exp trick for numerical stability
predict_nb <- function(model, test_point) {
  log_posteriors <- sapply(names(model), function(cls) {
    class_model <- model[[cls]]
    log_likelihood <- 0
    
    for(i in seq_along(features)) {
      feat <- features[i]
      kde <- kde(class_model$data[[feat]], 
                 h=class_model$bandwidths[i],
                 eval.points=test_point[[feat]])
      log_likelihood <- log_likelihood + log(max(kde$estimate, 1e-10))
    }
    
    log(class_model$prior) + log_likelihood
  })
  
  names(which.max(log_posteriors))
}

# 4. Enhanced Model Evaluation
# ---------------------------
set.seed(42)
train_idx <- createDataPartition(data$target, p=0.7, list=FALSE)
train_data <- data[train_idx,]
test_data <- data[-train_idx,]

# Train model
nb_model <- train_nb(train_data)

# Make predictions
predictions <- sapply(1:nrow(test_data), function(i) {
  predict_nb(nb_model, test_data[i, features])
})

# Confusion matrix with metrics
conf_matrix <- confusionMatrix(factor(predictions, levels=levels(data$target)), 
                               test_data$target)
print(conf_matrix)

# Feature importance based on mutual information
target_disc <- discretize(data$target, disc="equalfreq", nbins=3)
feature_importance <- sapply(features, function(f) {
  mutinformation(discretized_data[,f], target_disc)
})
print("Feature Importance (Mutual Information with Target):")
print(sort(feature_importance, decreasing=TRUE))

