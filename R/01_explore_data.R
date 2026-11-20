## 1. Look at the data

source("R/00_setup.R")

dim(ihdp)
table(W)
summary(Y)

# mean outcome by group
tapply(Y, W, mean)

# standardized mean differences between treated and control
smd <- rep(NA, length(xvars))
names(smd) <- xvars
for (v in xvars) {
  x1 <- ihdp[W == 1, v]
  x0 <- ihdp[W == 0, v]
  smd[v] <- (mean(x1) - mean(x0)) / sqrt((var(x1) + var(x0)) / 2)
}
round(sort(smd), 3)
sum(abs(smd) > 0.1)   # covariates that are not balanced

png("output/01_balance.png", width = 1000, height = 1100, res = 150)
par(mar = c(4, 5, 3, 1))
smd_sorted <- smd[order(abs(smd))]
plot(smd_sorted, 1:length(smd_sorted), yaxt = "n", pch = 16, col = "steelblue",
     xlab = "Standardized mean difference", ylab = "",
     main = "Balance between treated and control")
axis(2, at = 1:length(smd_sorted), labels = names(smd_sorted), las = 2, cex.axis = 0.8)
abline(v = 0)
abline(v = c(-0.1, 0.1), lty = 2, col = "grey50")
dev.off()

# true effects
summary(tau_true)
mean(tau_true)            # true ATE
mean(tau_true[W == 1])    # true ATT
