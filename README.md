# Prediction and causal machine learning with the IHDP data (R)

This project goes from simple prediction models to causal machine learning, using the
IHDP data. The Infant Health and Development Program gave home visits and child care
to low-birth-weight infants. I use the semi-simulated version from Hill (2011): the
covariates are real and the outcome is simulated, so the true effect for each child is
known and every estimate can be checked against it.

## Scripts

1. `01_explore_data.R`: sample, outcome by group, covariate balance
2. `02_predict_outcomes_validation.R`: predict the outcome of control children with OLS, ridge, lasso, random forests and boosting (75/25 split)
3. `03_predict_outcomes_cv.R`: the same models with 10-fold cross-validation; a binary outcome (below the median) with logistic regression, lasso logistic, random forest and boosting; ROC curves, AUC and calibration
4. `04_average_treatment_effect.R`: ATE with difference in means, OLS, separate OLS by group, IPW, cross-fitted AIPW, double ML and a causal forest
5. `05_heterogeneous_effects.R`: CATE with OLS interactions, lasso and forests by group, and a causal forest; PEHE, calibration test, effects by quartile, best linear projection
6. `06_policy_learning.R`: who to treat when treatment costs 4 outcome units; plug-in rule and depth-2 policy tree, evaluated on held-out data over 20 splits

`00_setup.R` loads the packages and the data. Tables and figures are saved in `output/`.

## Main results

- Boosting predicts the control outcome best (CV MSE 1.06, against 1.28 for OLS).
- Treated and control children differ on 16 of the 25 covariates, but all seven ATE estimates are within 0.09 of the true ATE (4.02) and all the 95% CIs include it.
- For individual effects, separate forests by group do best (PEHE 0.37), then separate lasso models (0.44) and the causal forest (0.52). OLS with interactions does worst (0.70).
- With a cost of 4, treating everyone gains 0.03 per child. Treating only children whose estimated effect is above the cost gains 0.27, close to the 0.30 we would get knowing the true effects. A depth-2 policy tree gains 0.21.

## Running the code

In R, from the main folder of the repository:

```r
source("run_all.R")
```

The data file (replication 1 of the IHDP version used by Shalit et al. 2017 and Louizos
et al. 2017) is downloaded into `data/` the first time. Covariates x1 to x6 are
continuous and x7 to x25 are binary.

Packages: ggplot2, glmnet, grf, gbm, ranger, pROC, DoubleML, mlr3, mlr3learners, policytree.

## References

- Athey, S., J. Tibshirani and S. Wager (2019). Generalized random forests. Annals of Statistics 47(2).
- Athey, S. and S. Wager (2021). Policy learning with observational data. Econometrica 89(1).
- Chernozhukov, V. et al. (2018). Double/debiased machine learning for treatment and structural parameters. Econometrics Journal 21(1).
- Hill, J. (2011). Bayesian nonparametric modeling for causal inference. Journal of Computational and Graphical Statistics 20(1).
- Louizos, C. et al. (2017). Causal effect inference with deep latent-variable models. NeurIPS.
- Shalit, U., F. Johansson and D. Sontag (2017). Estimating individual treatment effect: generalization bounds and algorithms. ICML.
