## Packages and data
## Run the scripts from the main folder of the repository

# install.packages(c("ggplot2", "glmnet", "grf", "gbm", "ranger", "pROC",
#                    "DoubleML", "mlr3", "mlr3learners", "policytree"))

library(ggplot2)       # graphs
library(glmnet)        # ridge and lasso
library(grf)           # regression forests and causal forests
library(gbm)           # boosting
library(ranger)        # random forest for the binary outcome
library(pROC)          # ROC curves and AUC
library(DoubleML)      # double machine learning
library(mlr3)
library(mlr3learners)
library(policytree)    # policy trees

lgr::get_logger("mlr3")$set_threshold("warn")   # less printing from mlr3

dir.create("output", showWarnings = FALSE)

# IHDP data, replication 1 (Hill 2011 simulation, csv from the CEVAE repository)
if (!file.exists("data/ihdp_npci_1.csv")) {
  dir.create("data", showWarnings = FALSE)
  download.file("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv",
                "data/ihdp_npci_1.csv")
}

ihdp <- read.csv("data/ihdp_npci_1.csv", header = FALSE)
names(ihdp) <- c("W", "Y", "Y_cf", "mu0", "mu1", paste0("x", 1:25))

# x1-x6 are continuous, x7-x25 are binary
xvars <- paste0("x", 1:25)
X <- as.matrix(ihdp[, xvars])
Y <- ihdp$Y
W <- ihdp$W

# true effect for each child (we know it only because the outcome is simulated)
tau_true <- ihdp$mu1 - ihdp$mu0
