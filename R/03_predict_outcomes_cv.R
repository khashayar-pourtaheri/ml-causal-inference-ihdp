## 3. Same models with 10-fold cross-validation, then a binary outcome

source("R/00_setup.R")

controls <- ihdp[W == 0, c("Y", xvars)]

K <- 10
set.seed(123456)
controls$fold <- sample(rep(1:K, length.out = nrow(controls)))

models <- c("OLS", "ridge", "lasso", "rf_default", "rf_sqrtp",
            "boost_depth1", "boost_depth2", "boost_depth3")
preds <- matrix(NA, nrow = nrow(controls), ncol = length(models))
colnames(preds) <- models
MSEs <- matrix(NA, nrow = K, ncol = length(models))
colnames(MSEs) <- models

for (k in 1:K) {
  train <- controls[controls$fold != k, c("Y", xvars)]
  validate <- controls[controls$fold == k, c("Y", xvars)]
  X.train <- as.matrix(train[, xvars])
  X.validate <- as.matrix(validate[, xvars])
  rows <- which(controls$fold == k)

  #a) OLS
  mod_OLS <- lm(Y ~ ., data = train)
  preds[rows, "OLS"] <- predict(mod_OLS, newdata = validate)

  #b) ridge and lasso (lambda chosen by CV inside the training folds)
  set.seed(234567)
  mod_ridge <- cv.glmnet(X.train, train$Y, alpha = 0)
  preds[rows, "ridge"] <- predict(mod_ridge, newx = X.validate, s = "lambda.min")
  mod_lasso <- cv.glmnet(X.train, train$Y, alpha = 1)
  preds[rows, "lasso"] <- predict(mod_lasso, newx = X.validate, s = "lambda.min")

  #c) random forests
  mod_rf1 <- regression_forest(X.train, train$Y)
  preds[rows, "rf_default"] <- predict(mod_rf1, X.validate)$predictions
  mod_rf2 <- regression_forest(X.train, train$Y, mtry = floor(sqrt(ncol(X.train))))
  preds[rows, "rf_sqrtp"] <- predict(mod_rf2, X.validate)$predictions

  #d) boosting
  set.seed(123)
  for (d in 1:3) {
    mod_boost <- gbm(Y ~ ., data = train, distribution = "gaussian", n.trees = 1000,
                     interaction.depth = d, shrinkage = 0.02, cv.folds = 5)
    ntrees <- gbm.perf(mod_boost, method = "cv", plot.it = FALSE)
    preds[rows, paste0("boost_depth", d)] <- predict(mod_boost, newdata = validate, n.trees = ntrees)
  }

  # MSE in this fold
  for (m in models) MSEs[k, m] <- mean((preds[rows, m] - validate$Y)^2)
}

# CV MSE two ways: average over folds, and from all out-of-fold predictions
CV_MSE <- data.frame(model = models,
                     fold_average = colMeans(MSEs),
                     all_predictions = colMeans((preds - controls$Y)^2))
CV_MSE <- CV_MSE[order(CV_MSE$all_predictions), ]
CV_MSE
write.csv(CV_MSE, "output/03_cv_mse.csv", row.names = FALSE)


#### Binary outcome: Y below the median of the control group

controls$low <- ifelse(controls$Y < median(controls$Y), 1, 0)
table(controls$low)

models_bin <- c("logistic", "lasso_logistic", "random_forest", "boosting")
preds_bin <- matrix(NA, nrow = nrow(controls), ncol = length(models_bin))
colnames(preds_bin) <- models_bin

for (k in 1:K) {
  train <- controls[controls$fold != k, c("low", xvars)]
  validate <- controls[controls$fold == k, c("low", xvars)]
  X.train <- as.matrix(train[, xvars])
  X.validate <- as.matrix(validate[, xvars])
  rows <- which(controls$fold == k)

  #a) logistic regression
  mod_logit <- glm(low ~ ., data = train, family = "binomial")
  preds_bin[rows, "logistic"] <- predict(mod_logit, newdata = validate, type = "response")

  #b) lasso logistic regression
  set.seed(345678)
  mod_lasso <- cv.glmnet(X.train, train$low, family = "binomial", alpha = 1)
  preds_bin[rows, "lasso_logistic"] <- predict(mod_lasso, newx = X.validate, s = "lambda.min", type = "response")

  #c) random forest
  mod_rf <- ranger(as.factor(low) ~ ., data = train, probability = TRUE, num.trees = 1000, seed = 1)
  preds_bin[rows, "random_forest"] <- predict(mod_rf, data = validate)$predictions[, "1"]

  #d) boosting with depth 2
  set.seed(123)
  mod_boost <- gbm(low ~ ., data = train, distribution = "bernoulli", n.trees = 1000,
                   interaction.depth = 2, shrinkage = 0.02, cv.folds = 5)
  ntrees <- gbm.perf(mod_boost, method = "cv", plot.it = FALSE)
  preds_bin[rows, "boosting"] <- predict(mod_boost, newdata = validate, n.trees = ntrees, type = "response")
}

# ROC curves and AUC
AUCs <- rep(NA, length(models_bin))
names(AUCs) <- models_bin
cols <- c("blue", "red", "darkgreen", "orange")

png("output/03_roc.png", width = 1400, height = 1000, res = 150)
par(mar = c(4, 4, 3, 1))
for (i in 1:length(models_bin)) {
  currROC <- roc(controls$low, preds_bin[, i], quiet = TRUE)
  AUCs[i] <- as.numeric(currROC$auc)
  if (i == 1) {
    plot(currROC, col = cols[i], lwd = 2, legacy.axes = TRUE, main = "Cross-validated ROC curves")
  } else {
    lines(currROC, col = cols[i], lwd = 2)
  }
}
legend("bottomright", legend = paste0(models_bin, " (AUC ", round(AUCs, 3), ")"),
       col = cols, lwd = 2, bty = "n")
dev.off()

sort(AUCs, decreasing = TRUE)

# calibration: observed share of low outcomes in each decile of predicted risk
png("output/03_calibration.png", width = 1400, height = 1400, res = 150)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
for (m in models_bin) {
  decile <- cut(preds_bin[, m], quantile(preds_bin[, m], seq(0, 1, 0.1)), include.lowest = TRUE)
  mean_pred <- tapply(preds_bin[, m], decile, mean)
  mean_obs <- tapply(controls$low, decile, mean)
  plot(mean_pred, mean_obs, xlim = c(0, 1), ylim = c(0, 1), pch = 16, type = "b",
       col = "steelblue", xlab = "Mean predicted probability", ylab = "Observed share", main = m)
  abline(a = 0, b = 1, lty = 2, col = "grey50")
}
dev.off()
