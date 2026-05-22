#ISLR 4 
set.seed(42)
lib <- .libPaths("C:\\Users\\Kamyar\\OneDrive\\Desktop\\r_lib")
install.packages("MASS")
library(MASS)
install.packages("ISLR2")  
library(ISLR2) 
library(e1071)
library(class)
library(caret)
library(pROC)




data("Default")            
head(Default)
n <- nrow(Default)
train_idx <- sample(n, size = 0.8 * n)
train     <- Default[train_idx, ]

test <- Default[-train_idx, ]

pca_fit <- pca(default ~ balance + income, data = train)

lda_fit <- lda(default ~ balance + income, data = train)
lda_fit

lda_pred <- predict(lda_fit, newdata = test)
names(lda_pred)
head(lda_pred$class)
head(lda_pred$posterior, 3)

#evaluation 
table(predicted = lda_pred$class, Actual = test$default)

# Performance metrics
TP <- 28; FN <- 37; TN <- 1928; FP <- 7
Sensitivity <- TP / (TP + FN)   # 0.431 (43% of defaults caught)
Specificity  <- TN / (TN + FP)  # 0.996
Accuracy     <- (TP + TN) / (TP+FN+TN+FP) # 0.978

lda_class_adj <- ifelse(lda_pred$posterior[, "Yes"] > 0.2, "Yes", "No")


# lets do lab 
data(Default)

str(Default)

head(Default)

table(Default$student)
table(Default$default)
table(Default$balance)
table(Default$income)
table(Default$default)


# you can look at probability from the start 

prop.table(table(Default$default))



# train and testing 

set.seed(42)                             # For reproducibility!
n         <- nrow(Default)              # 10,000
train_idx <- sample(n, size = 0.8 * n) # 8,000 random indices

train_data <- Default[ train_idx, ]    # 8,000 observations
test_data  <- Default[-train_idx, ]    # 2,000 observations

## Verify balance is preserved
cat("Train default rate:", mean(train_data$default=="Yes"), "\n") # ~0.033
cat("Test default rate: ", mean(test_data$default=="Yes"),  "\n") # ~0.033



lda_fit <- lda(default ~ balance + income, data = train_data)


lda(default ~ balance + income, data = train_data)



####QDA Lab 

qda_fit <- qda(default ~ balance + income, data = train_data)
qda_pred <- predict(qda_fit, newdata = test_data)
# confusion matrix 
qda_fit_tab_table <- table(predicted = qda_pred$class, Actual = test_data$default)

print(qda_fit_tab_table)

table(qda_pred$class, test_data$default)


cov_no <- cov(train_data[train_data$default=="No", c("balance","income")])

cov_yes <- cov(train_data[train_data$default=="Yes", c("balance","income")])

cat("Covariance matrix - No default:\n")
print(round(cov_no, 0))

cat("\nCovariance matrix - Yes default:\n")
print(round(cov_yes, 0))






# KNN 

# sometimes some data dominate... for instance, income here dominates substatially

#which is wrong 
library(class)

## WRONG: unscaled data — income dominates by 28x
knn_wrong <- knn(train = train_data[, c("balance","income")],
                 test  = test_data[,  c("balance","income")],
                 cl    = train_data$default, k = 5)

# --correcting the imbalance 

train_X <- train_data[, c("balance", "income")]
test_X <- test_data[, c("balance", "income")]


sc_center <- colMeans(train_X)
sc_scale   <- apply(train_X, 2, sd)
train_sc   <- scale(train_X, center = sc_center, scale = sc_scale)
test_sc    <- scale(test_X,  center = sc_center, scale = sc_scale)



# fit KNN with k= 5 

set.seed(1)
knn_pred_5 <- knn(train = train_sc,
                  test  = test_sc,
                  cl    = train_data$default,
                  k     = 5,
                  prob  = TRUE)  # prob=TRUE gets vote fraction

head(knn_pred_5, 6)


vote_frac <- attr(knn_pred_5, "prob")
head(vote_frac, 6)


tab_knn <- table(Predicted=knn_pred_5, Actual=test_data$default)
print(tab_knn)



k_vals    <- c(1, 3, 5, 7, 10, 15, 20, 30, 50, 100)
test_errs <- numeric(length(k_vals))
for (i in seq_along(k_vals)) {
  set.seed(1)
  pred <- knn(train_sc, test_sc, train_data$default, k = k_vals[i])
  test_errs[i] <- mean(pred != test_data$default)
}

## Print results
results <- data.frame(K = k_vals, TestError = round(test_errs, 4))
print(results)

## Find optimal K
best_k <- k_vals[which.min(test_errs)]
cat("Optimal K:", best_k, "\n")

## Plot
plot(k_vals, test_errs, type = "b", log = "x",
     xlab = "K (log scale)", ylab = "Test Error",
     main = "KNN: Choosing K via Test Error")
abline(v = best_k, lty = 2, col = "red")





#----a summary code for this lab --- by claude 

# i wanted this code to represent the chapter summary of the summary for future refereces 

# I think LDA has potentials for gene expressions data and has real use beenfits for dimentionality reduction 


## ── Complete comparison pipeline ─────────────────────────────────
library(ISLR2); library(MASS); library(e1071); library(class); library(pROC)
data(Default); set.seed(42)
n <- nrow(Default); idx <- sample(n, 0.8*n)
train <- Default[idx,]; test <- Default[-idx,]

## ── Helper: compute AUC from probability and actual ──────────────
get_auc <- function(prob, actual) {
  auc(roc(actual, prob, quiet=TRUE))
}

## ── 1. Logistic Regression ──────────────────────────────────────
glm_fit  <- glm(default ~ balance + income, data=train, family=binomial)
glm_prob <- predict(glm_fit, newdata=test, type="response")
glm_cls  <- ifelse(glm_prob > 0.5, "Yes", "No")

## ── 2. LDA ──────────────────────────────────────────────────────
lda_fit  <- lda(default ~ balance + income, data=train)
lda_pred <- predict(lda_fit, newdata=test)
lda_prob <- lda_pred$posterior[,"Yes"]

## ── 3. QDA ──────────────────────────────────────────────────────
qda_fit  <- qda(default ~ balance + income, data=train)
qda_pred <- predict(qda_fit, newdata=test)
qda_prob <- qda_pred$posterior[,"Yes"]

## ── 4. Naive Bayes ──────────────────────────────────────────────
nb_fit   <- naiveBayes(default ~ balance + income, data=train)
nb_prob  <- predict(nb_fit, newdata=test, type="raw")[,"Yes"]
nb_cls   <- predict(nb_fit, newdata=test)

## ── 5. KNN (K=7, scaled) ─────────────────────────────────────────
trainX <- scale(train[,c("balance","income")])
testX  <- scale(test[,c("balance","income")],
                center=attr(trainX,"scaled:center"),
                scale=attr(trainX,"scaled:scale"))
knn_cls <- knn(trainX, testX, train$default, k=7, prob=TRUE)

## ── Compute metrics ──────────────────────────────────────────────
methods <- list(
  Logistic = list(cls=glm_cls, prob=glm_prob),
  LDA      = list(cls=lda_pred$class, prob=lda_prob),
  QDA      = list(cls=qda_pred$class, prob=qda_prob),
  NaiveBayes=list(cls=nb_cls, prob=nb_prob)
)

for (m in names(methods)) {
  cls  <- factor(methods[[m]]$cls, levels=c("No","Yes"))
  prob <- methods[[m]]$prob
  tab  <- table(cls, test$default)
  cat(m, "— Sens:", round(tab[2,2]/sum(tab[,2]),3),
      "Spec:", round(tab[1,1]/sum(tab[,1]),3),
      "AUC:",  round(get_auc(prob, test$default),3), "\n")
}



