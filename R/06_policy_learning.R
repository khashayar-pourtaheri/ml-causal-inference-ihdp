## 6. Who should be treated if treatment has a cost?
## cost is in outcome units: treat a child only if the effect is larger than the cost

source("R/00_setup.R")

cost <- 4
n <- nrow(ihdp)
mean(tau_true > cost)   # share of children with a true effect above the cost

# policy tree of depth 2 on the full data (for interpretation)
set.seed(123)
cf <- causal_forest(X, Y, W)
gamma <- get_scores(cf)
gamma_net <- gamma - cost
tree <- policy_tree(X, cbind(-gamma_net, gamma_net), depth = 2)
print(tree)

# honest evaluation: fit on 70% of the data, evaluate on the other 30%
# with the true effects, repeated over 20 random splits
policies <- c("treat_none", "treat_all", "plug_in", "policy_tree", "oracle")
n_splits <- 20
value <- matrix(NA, nrow = n_splits, ncol = length(policies))
treated <- matrix(NA, nrow = n_splits, ncol = length(policies))
colnames(value) <- colnames(treated) <- policies

for (s in 1:n_splits) {
  set.seed(1000 + s)
  train <- sample(n, round(0.7 * n))
  test <- setdiff(1:n, train)

  cf_train <- causal_forest(X[train, ], Y[train], W[train])
  g <- get_scores(cf_train) - cost
  tree_train <- policy_tree(X[train, ], cbind(-g, g), depth = 2)

  rule <- cbind(treat_none = 0,
                treat_all = 1,
                plug_in = as.numeric(predict(cf_train, X[test, ])$predictions > cost),
                policy_tree = predict(tree_train, X[test, ]) - 1,
                oracle = as.numeric(tau_true[test] > cost))

  for (p in policies) {
    value[s, p] <- mean((tau_true[test] - cost) * rule[, p])   # net benefit per child
    treated[s, p] <- mean(rule[, p])
  }
}

policy_table <- data.frame(policy = policies,
                           net_benefit = colMeans(value),
                           se = apply(value, 2, sd) / sqrt(n_splits),
                           share_treated = colMeans(treated))
policy_table
write.csv(policy_table, "output/06_policy_values.csv", row.names = FALSE)

png("output/06_policy_values.png", width = 1400, height = 800, res = 150)
bp <- barplot(policy_table$net_benefit, names.arg = policy_table$policy, col = "steelblue",
              ylim = c(min(0, policy_table$net_benefit), max(policy_table$net_benefit) * 1.2),
              ylab = "Net benefit per child", main = paste("Value of each rule on held-out data, cost =", cost))
arrows(bp, policy_table$net_benefit - 1.96 * policy_table$se, bp,
       policy_table$net_benefit + 1.96 * policy_table$se, angle = 90, code = 3, length = 0.05)
dev.off()
