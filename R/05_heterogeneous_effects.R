## 5. Heterogeneous treatment effects (CATE)
## compare simple and flexible models with the true effect of each child

source("R/00_setup.R")

n <- nrow(ihdp)
dat <- data.frame(Y = Y, W = W, X)

# 5-fold cross-fitting: each child's CATE comes from models that did not use that child
set.seed(123)
fold <- sample(rep(1:5, length.out = n))
cate <- matrix(NA, nrow = n, ncol = 4)
colnames(cate) <- c("OLS_interactions", "lasso_by_group", "forest_by_group", "causal_forest")

for (k in 1:5) {
  tr <- fold != k
  te <- fold == k

  #a) one OLS with treatment x covariate interactions
  mod_int <- lm(Y ~ W * ., data = dat[tr, ])
  d1 <- dat[te, ]; d1$W <- 1
  d0 <- dat[te, ]; d0$W <- 0
  cate[te, "OLS_interactions"] <- predict(mod_int, newdata = d1) - predict(mod_int, newdata = d0)

  #b) separate lasso for treated and control
  mod_lasso1 <- cv.glmnet(X[tr & W == 1, ], Y[tr & W == 1], alpha = 1)
  mod_lasso0 <- cv.glmnet(X[tr & W == 0, ], Y[tr & W == 0], alpha = 1)
  cate[te, "lasso_by_group"] <- predict(mod_lasso1, newx = X[te, ], s = "lambda.min") -
    predict(mod_lasso0, newx = X[te, ], s = "lambda.min")

  #c) separate regression forests for treated and control
  mod_rf1 <- regression_forest(X[tr & W == 1, ], Y[tr & W == 1])
  mod_rf0 <- regression_forest(X[tr & W == 0, ], Y[tr & W == 0])
  cate[te, "forest_by_group"] <- predict(mod_rf1, X[te, ])$predictions - predict(mod_rf0, X[te, ])$predictions
}

#d) causal forest, nuisance models fitted first, out-of-bag predictions
set.seed(123)
Y.forest <- regression_forest(X, Y, tune.parameters = "all")
W.forest <- regression_forest(X, W, tune.parameters = "all")
Y.hat <- predict(Y.forest)$predictions
W.hat <- predict(W.forest)$predictions
cf <- causal_forest(X, Y, W, Y.hat = Y.hat, W.hat = W.hat, tune.parameters = "all")
cate[, "causal_forest"] <- predict(cf)$predictions

# PEHE = root mean squared error against the true CATE
PEHEtable <- data.frame(model = colnames(cate), PEHE = NA, correlation = NA)
for (i in 1:ncol(cate)) {
  PEHEtable$PEHE[i] <- sqrt(mean((cate[, i] - tau_true)^2))
  PEHEtable$correlation[i] <- cor(cate[, i], tau_true)
}
PEHEtable <- PEHEtable[order(PEHEtable$PEHE), ]
PEHEtable
write.csv(PEHEtable, "output/05_cate_accuracy.csv", row.names = FALSE)

png("output/05_cate_vs_true.png", width = 1400, height = 1400, res = 150)
par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
for (m in PEHEtable$model) {
  plot(tau_true, cate[, m], pch = 16, col = adjustcolor("steelblue", alpha.f = 0.4), cex = 0.7,
       xlab = "True CATE", ylab = "Estimated CATE", main = m)
  abline(a = 0, b = 1, col = "red", lty = 2)
  legend("topleft", legend = paste("PEHE =", round(sqrt(mean((cate[, m] - tau_true)^2)), 3)), bty = "n")
}
dev.off()


#### More on the causal forest

# is there heterogeneity? (calibration test)
test_calibration(cf)

# doubly robust scores, averaged within quartiles of the predicted CATE
DR.scores <- get_scores(cf)
quartile <- cut(cate[, "causal_forest"], quantile(cate[, "causal_forest"], c(0, 0.25, 0.5, 0.75, 1)),
                include.lowest = TRUE, labels = 1:4)
quartile_table <- data.frame(quartile = 1:4,
                             DR_estimate = tapply(DR.scores, quartile, mean),
                             se = tapply(DR.scores, quartile, sd) / sqrt(tapply(DR.scores, quartile, length)),
                             true_CATE = tapply(tau_true, quartile, mean))
quartile_table

png("output/05_quartiles.png", width = 1200, height = 900, res = 150)
plot(1:4, quartile_table$DR_estimate, ylim = range(c(quartile_table$DR_estimate - 2 * quartile_table$se,
                                                    quartile_table$DR_estimate + 2 * quartile_table$se,
                                                    quartile_table$true_CATE)),
     pch = 16, col = "steelblue", xaxt = "n", xlab = "Quartile of predicted CATE", ylab = "Average effect",
     main = "Effect by quartile of predicted CATE")
axis(1, at = 1:4)
segments(1:4, quartile_table$DR_estimate - 1.96 * quartile_table$se,
         1:4, quartile_table$DR_estimate + 1.96 * quartile_table$se, col = "steelblue", lwd = 2)
points(1:4, quartile_table$true_CATE, pch = 4, col = "red", cex = 1.5)
legend("bottomright", legend = c("DR estimate (95% CI)", "true CATE"), pch = c(16, 4),
       col = c("steelblue", "red"), bty = "n")
dev.off()

# variable importance
varimp <- variable_importance(cf)
rownames(varimp) <- xvars
importance_df <- data.frame(variable = xvars, importance = varimp[, 1])
importance_df <- importance_df[order(-importance_df$importance), ]
head(importance_df, 8)

# best linear predictor of the CATE using the five most important variables
top5 <- importance_df$variable[1:5]
best_linear_projection(cf, X[, top5])
