## 2. Predict the outcome without treatment (validation set approach)

source("R/00_setup.R")

# control group only
controls <- ihdp[W == 0, c("Y", xvars)]

# 75% training data, 25% validation data
set.seed(123)
train_rows <- sample(nrow(controls), round(0.75 * nrow(controls)))
train <- controls[train_rows, ]
validation <- controls[-train_rows, ]

X.train <- as.matrix(train[, xvars])
Y.train <- train$Y
X.validation <- as.matrix(validation[, xvars])
Y.validation <- validation$Y


#### Fit the models on the training data

#a) OLS with all covariates
mod_OLS <- lm(Y ~ ., data = train)
summary(mod_OLS)

#b) ridge and lasso, lambda chosen by cross-validation
set.seed(1)
mod_ridge <- cv.glmnet(X.train, Y.train, alpha = 0)
mod_lasso <- cv.glmnet(X.train, Y.train, alpha = 1)
log(mod_ridge$lambda.min)
log(mod_lasso$lambda.min)

#c) random forests with default m and with m = sqrt(p)
set.seed(124)
mod_rf_default <- regression_forest(X.train, Y.train)
mod_rf_sqrtp <- regression_forest(X.train, Y.train, mtry = floor(sqrt(ncol(X.train))))

#d) boosting with depth 1, 2 and 3, number of trees chosen by 5-fold CV
set.seed(123)
mod_boost1 <- gbm(Y ~ ., data = train, distribution = "gaussian", n.trees = 1000,
                  interaction.depth = 1, shrinkage = 0.02, cv.folds = 5)
mod_boost2 <- gbm(Y ~ ., data = train, distribution = "gaussian", n.trees = 1000,
                  interaction.depth = 2, shrinkage = 0.02, cv.folds = 5)
mod_boost3 <- gbm(Y ~ ., data = train, distribution = "gaussian", n.trees = 1000,
                  interaction.depth = 3, shrinkage = 0.02, cv.folds = 5)
ntrees1 <- gbm.perf(mod_boost1, method = "cv", plot.it = FALSE)
ntrees2 <- gbm.perf(mod_boost2, method = "cv", plot.it = FALSE)
ntrees3 <- gbm.perf(mod_boost3, method = "cv", plot.it = FALSE)


#### Predictions and MSE on the validation data

predV <- data.frame(
  OLS = predict(mod_OLS, newdata = validation),
  ridge = as.vector(predict(mod_ridge, newx = X.validation, s = "lambda.min")),
  lasso = as.vector(predict(mod_lasso, newx = X.validation, s = "lambda.min")),
  rf_default = predict(mod_rf_default, X.validation)$predictions,
  rf_sqrtp = predict(mod_rf_sqrtp, X.validation)$predictions,
  boost_depth1 = predict(mod_boost1, newdata = validation, n.trees = ntrees1),
  boost_depth2 = predict(mod_boost2, newdata = validation, n.trees = ntrees2),
  boost_depth3 = predict(mod_boost3, newdata = validation, n.trees = ntrees3)
)

MSEtable <- data.frame(model = names(predV), MSE = NA, R2 = NA)
for (i in 1:ncol(predV)) {
  MSEtable$MSE[i] <- mean((predV[, i] - Y.validation)^2)
  MSEtable$R2[i] <- 1 - sum((Y.validation - predV[, i])^2) / sum((Y.validation - mean(Y.validation))^2)
}
MSEtable <- MSEtable[order(MSEtable$MSE), ]
MSEtable$rank <- 1:nrow(MSEtable)
MSEtable
write.csv(MSEtable, "output/02_validation_mse.csv", row.names = FALSE)

# predicted vs observed for all models, same axes
ax_lim <- range(c(Y.validation, unlist(predV)))
png("output/02_pred_vs_obs.png", width = 2000, height = 1100, res = 150)
par(mfrow = c(2, 4), mar = c(4, 4, 3, 1))
for (m in MSEtable$model) {
  plot(Y.validation, predV[, m], xlim = ax_lim, ylim = ax_lim, pch = 16,
       col = adjustcolor("steelblue", alpha.f = 0.5), cex = 0.8,
       xlab = "Observed Y", ylab = "Predicted Y", main = m)
  abline(a = 0, b = 1, col = "red", lty = 2)
  legend("topleft", legend = paste("MSE =", round(mean((predV[, m] - Y.validation)^2), 2)), bty = "n")
}
dev.off()


#### Top predictors, models refit on all control children

X.full <- as.matrix(controls[, xvars])
Y.full <- controls$Y

set.seed(1)
lasso_full <- cv.glmnet(X.full, Y.full, alpha = 1)
lasso_coefs <- as.matrix(coef(lasso_full, s = "lambda.min"))[-1, 1]   # drop intercept
lasso_nonzero <- lasso_coefs[lasso_coefs != 0]
lasso_sorted <- sort(abs(lasso_nonzero), decreasing = TRUE)
round(lasso_sorted[1:10], 3)

set.seed(124)
rf_full <- regression_forest(X.full, Y.full)
varimp <- variable_importance(rf_full)
rownames(varimp) <- xvars
rf_sorted <- sort(varimp[, 1], decreasing = TRUE)
round(rf_sorted[1:10], 3)

# predictors in the top 5 of both models
intersect(names(lasso_sorted)[1:5], names(rf_sorted)[1:5])
