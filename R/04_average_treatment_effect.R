## 4. Average treatment effect
## start with the simple difference in means, then adjust for the covariates

source("R/00_setup.R")

n <- nrow(ihdp)
true_ATE <- mean(tau_true)
true_ATE

ATEtable <- data.frame(method = c("difference in means", "OLS with treatment dummy",
                                  "separate OLS by group", "IPW", "AIPW (forests, cross-fitted)",
                                  "double ML", "causal forest"),
                       estimate = NA, se = NA)

#a) difference in means
ATEtable$estimate[1] <- mean(Y[W == 1]) - mean(Y[W == 0])
ATEtable$se[1] <- sqrt(var(Y[W == 1]) / sum(W == 1) + var(Y[W == 0]) / sum(W == 0))

#b) OLS with a treatment dummy and all covariates
mod_OLS <- lm(Y ~ W + X)
ATEtable$estimate[2] <- coef(mod_OLS)["W"]
ATEtable$se[2] <- summary(mod_OLS)$coefficients["W", "Std. Error"]

#c) separate OLS for treated and control, then average the predicted difference
dat <- data.frame(Y = Y, W = W, X)
mod_treated <- lm(Y ~ ., data = dat[W == 1, c("Y", xvars)])
mod_control <- lm(Y ~ ., data = dat[W == 0, c("Y", xvars)])
ATEtable$estimate[3] <- mean(predict(mod_treated, newdata = dat) - predict(mod_control, newdata = dat))

#d) inverse probability weighting with a logistic propensity score
ps_model <- glm(W ~ X, family = "binomial")
ps <- predict(ps_model, type = "response")
ps <- pmin(pmax(ps, 0.01), 0.99)   # trim extreme scores
ATEtable$estimate[4] <- sum(W * Y / ps) / sum(W / ps) - sum((1 - W) * Y / (1 - ps)) / sum((1 - W) / (1 - ps))

# bootstrap standard errors for c) and d)
set.seed(123)
B <- 300
boot_c <- rep(NA, B)
boot_d <- rep(NA, B)
for (b in 1:B) {
  i <- sample(n, replace = TRUE)
  d <- dat[i, ]
  m1 <- lm(Y ~ ., data = d[d$W == 1, c("Y", xvars)])
  m0 <- lm(Y ~ ., data = d[d$W == 0, c("Y", xvars)])
  boot_c[b] <- mean(predict(m1, newdata = d) - predict(m0, newdata = d))
  p <- predict(glm(W ~ X[i, ], family = "binomial"), type = "response")
  p <- pmin(pmax(p, 0.01), 0.99)
  boot_d[b] <- sum(d$W * d$Y / p) / sum(d$W / p) - sum((1 - d$W) * d$Y / (1 - p)) / sum((1 - d$W) / (1 - p))
}
ATEtable$se[3] <- sd(boot_c)
ATEtable$se[4] <- sd(boot_d)

#e) AIPW with 5-fold cross-fitting, regression forests for the outcome and the propensity score
set.seed(123)
fold <- sample(rep(1:5, length.out = n))
ps_hat <- rep(NA, n)
mu1_hat <- rep(NA, n)
mu0_hat <- rep(NA, n)
for (k in 1:5) {
  tr <- fold != k
  te <- fold == k
  ps_hat[te] <- predict(regression_forest(X[tr, ], W[tr]), X[te, ])$predictions
  mu1_hat[te] <- predict(regression_forest(X[tr & W == 1, ], Y[tr & W == 1]), X[te, ])$predictions
  mu0_hat[te] <- predict(regression_forest(X[tr & W == 0, ], Y[tr & W == 0]), X[te, ])$predictions
}
ps_hat <- pmin(pmax(ps_hat, 0.01), 0.99)
aipw_scores <- mu1_hat - mu0_hat + W * (Y - mu1_hat) / ps_hat - (1 - W) * (Y - mu0_hat) / (1 - ps_hat)
ATEtable$estimate[5] <- mean(aipw_scores)
ATEtable$se[5] <- sd(aipw_scores) / sqrt(n)

#f) double ML (interactive regression model) with random forest learners
set.seed(123)
dml_data <- DoubleMLData$new(dat, y_col = "Y", d_cols = "W")
dml <- DoubleMLIRM$new(dml_data,
                       ml_g = lrn("regr.ranger", num.trees = 500, min.node.size = 5),
                       ml_m = lrn("classif.ranger", num.trees = 500, min.node.size = 5),
                       n_folds = 5, score = "ATE", trimming_threshold = 0.01)
dml$fit()
dml$summary()
ATEtable$estimate[6] <- dml$coef
ATEtable$se[6] <- dml$se

#g) causal forest
set.seed(123)
cf <- causal_forest(X, Y, W)
cf_ate <- average_treatment_effect(cf)
ATEtable$estimate[7] <- cf_ate["estimate"]
ATEtable$se[7] <- cf_ate["std.err"]

# compare with the true ATE
ATEtable$lower <- ATEtable$estimate - 1.96 * ATEtable$se
ATEtable$upper <- ATEtable$estimate + 1.96 * ATEtable$se
ATEtable$error <- ATEtable$estimate - true_ATE
ATEtable$covers_true_ATE <- ATEtable$lower < true_ATE & ATEtable$upper > true_ATE
ATEtable
write.csv(ATEtable, "output/04_ate_estimates.csv", row.names = FALSE)

png("output/04_ate_estimates.png", width = 1400, height = 800, res = 150)
par(mar = c(4, 14, 3, 1))
plot(ATEtable$estimate, 7:1, xlim = range(c(ATEtable$lower, ATEtable$upper)), yaxt = "n",
     pch = 16, col = "steelblue", xlab = "Estimated ATE with 95% CI", ylab = "",
     main = "ATE by method (dashed line = true ATE)")
segments(ATEtable$lower, 7:1, ATEtable$upper, 7:1, col = "steelblue", lwd = 2)
axis(2, at = 7:1, labels = ATEtable$method, las = 2, cex.axis = 0.8)
abline(v = true_ATE, col = "red", lty = 2)
dev.off()

# overlap of the propensity scores
png("output/04_overlap.png", width = 1400, height = 800, res = 150)
hist(ps_hat[W == 0], breaks = 30, col = adjustcolor("grey50", 0.5), xlim = c(0, 1),
     main = "Propensity scores by group", xlab = "Estimated propensity score")
hist(ps_hat[W == 1], breaks = 30, col = adjustcolor("steelblue", 0.6), add = TRUE)
legend("topright", legend = c("control", "treated"),
       fill = c(adjustcolor("grey50", 0.5), adjustcolor("steelblue", 0.6)), bty = "n")
dev.off()
