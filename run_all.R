# run all scripts in order
source("R/01_explore_data.R", echo = TRUE)
source("R/02_predict_outcomes_validation.R", echo = TRUE)
source("R/03_predict_outcomes_cv.R", echo = TRUE)
source("R/04_average_treatment_effect.R", echo = TRUE)
source("R/05_heterogeneous_effects.R", echo = TRUE)
source("R/06_policy_learning.R", echo = TRUE)
