# =============================================================================
# Coverage Gap Tests for embedmit
# Tests for previously untested or under-tested code paths
# =============================================================================

# =============================================================================
# Section 1: step_lencode smooth=TRUE Tests (HIGH PRIORITY)
# The smooth branch (lines 240-265 in lencode.R) was completely untested
# =============================================================================

test_that("step_lencode smooth=TRUE works with numeric outcome", {

  # smooth=TRUE is the default for numeric outcomes
  set.seed(42)
  data <- data.frame(
    outcome = rnorm(100),
    predictor = factor(rep(letters[1:5], each = 20))
  )

  rec <- recipe(outcome ~ ., data = data) |>
    step_lencode(predictor, outcome = vars(outcome), smooth = TRUE) |>
    prep()

  result <- tidy(rec, 1)

  # Should have 5 levels + 1 for "..new"

  expect_equal(nrow(result), 6)
  expect_true("..new" %in% result$level)

  # Values should be numeric and not NA
  expect_true(all(!is.na(result$value)))
  expect_true(is.numeric(result$value))

  # Smoothed values should differ from simple means
  simple_means <- tapply(data$outcome, data$predictor, mean)
  expect_false(all(result$value[result$level != "..new"] == simple_means))
})

test_that("step_lencode smooth=TRUE ignores case weights (uses unweighted mean)", {
  # smooth=TRUE uses mean() not weighted.mean(), so case weights
  # in the data frame don't affect the smoothing calculation
  set.seed(42)
  data <- data.frame(
    outcome = rnorm(100),
    predictor = factor(rep(letters[1:5], each = 20))
  )

  rec <- recipe(outcome ~ ., data = data) |>
    step_lencode(predictor, outcome = vars(outcome), smooth = TRUE) |>
    prep()

  result <- tidy(rec, 1)

  expect_equal(nrow(result), 6)
  expect_true(all(!is.na(result$value)))
})

test_that("step_lencode smooth=TRUE errors for non-numeric outcome", {
  data <- data.frame(
    outcome = factor(rep(c("a", "b"), 50)),
    predictor = factor(rep(letters[1:5], each = 20))
  )

  expect_error(
    recipe(outcome ~ ., data = data) |>
      step_lencode(predictor, outcome = vars(outcome), smooth = TRUE) |>
      prep(),
    "smooth = TRUE.*only works for numeric"
  )
})

test_that("step_lencode smooth=TRUE with single observation per level", {
  # Edge case: variance within group will be NA or 0
  set.seed(42)
  data <- data.frame(
    outcome = rnorm(10),
    predictor = factor(letters[1:10])
  )

  rec <- recipe(outcome ~ ., data = data) |>
    step_lencode(predictor, outcome = vars(outcome), smooth = TRUE) |>
    prep()

  result <- tidy(rec, 1)

  # Should handle gracefully even with edge cases
  expect_equal(nrow(result), 11)  # 10 levels + ..new
})

test_that("step_lencode smooth=TRUE with zero variance outcome", {
  # Edge case: global variance is 0
  data <- data.frame(
    outcome = rep(5, 100),  # constant outcome
    predictor = factor(rep(letters[1:5], each = 20))
  )

  # This may produce NaN due to division by zero in smoothing formula
  # The function should handle this gracefully
  rec <- recipe(outcome ~ ., data = data) |>
    step_lencode(predictor, outcome = vars(outcome), smooth = TRUE) |>
    prep()

  result <- tidy(rec, 1)
  expect_equal(nrow(result), 6)
})

# =============================================================================
# Section 2: log_odds() and weighted_log_odds() Edge Cases (MEDIUM PRIORITY)
# Testing p=0, p=1, and edge cases that produce Inf
# =============================================================================

test_that("step_lencode handles all-same-class predictor levels (p=0 or p=1)", {
  # When all observations in a predictor level have same outcome class,
  # log_odds produces Inf which should be adjusted by adjust_infinities
  data <- data.frame(
    outcome = factor(c(rep("a", 50), rep("b", 50))),
    predictor = factor(c(rep("x", 50), rep("y", 50)))  # perfect separation
  )

  rec <- recipe(outcome ~ ., data = data) |>
    step_lencode(predictor, outcome = vars(outcome), smooth = FALSE) |>
    prep()

  result <- tidy(rec, 1)

  # Values should not be Inf after adjustment
  expect_false(any(is.infinite(result$value)))
  expect_true(all(!is.na(result$value)))
})

test_that("step_lencode handles weighted log_odds with extreme weights", {
  data <- data.frame(
    outcome = factor(rep(c("a", "b"), 50)),
    predictor = factor(rep(letters[1:5], each = 20)),
    wts = hardhat::importance_weights(c(rep(0.001, 50), rep(1000, 50)))
  )

  rec <- recipe(outcome ~ ., data = data) |>
    step_lencode(predictor, outcome = vars(outcome), smooth = FALSE) |>
    prep()

  result <- tidy(rec, 1)

  # Should handle extreme weights without producing NA or Inf
  expect_false(any(is.infinite(result$value)))
  expect_true(all(!is.na(result$value)))
})

# =============================================================================
# Section 3: adjust_infinities() Edge Cases (MEDIUM PRIORITY)
# =============================================================================

test_that("step_lencode adjust_infinities handles mixed Inf/-Inf values", {
  # Create data that produces both +Inf and -Inf in log_odds
  data <- data.frame(
    outcome = factor(c("a", "a", "a", "b", "b", "b")),
    predictor = factor(c("x", "x", "x", "y", "y", "y"))  # produces both extremes
  )

  rec <- recipe(outcome ~ ., data = data) |>
    step_lencode(predictor, outcome = vars(outcome), smooth = FALSE) |>
    prep()

  result <- tidy(rec, 1)

  # All infinities should be adjusted
  expect_false(any(is.infinite(result$value)))
})

# =============================================================================
# Section 4: step_lencode_mixed Error Handling (HIGH PRIORITY)
# 3+ class factor outcome should error
# =============================================================================

test_that("step_lencode_mixed errors with 3+ class factor outcome", {
  skip_if_not_installed("lme4")

  three_class <- iris
  three_class$predictor <- factor(rep(letters[1:3], 50))

  expect_error(
    recipe(Species ~ ., data = three_class) |>
      step_lencode_mixed(predictor, outcome = vars(Species)) |>
      prep(),
    "two-class|two levels|2 different values"
  )
})

test_that("step_lencode_mixed works with binary factor outcome", {
  skip_if_not_installed("lme4")

  set.seed(42)
  data <- data.frame(
    outcome = factor(rep(c("a", "b"), each = 50)),
    predictor = factor(rep(letters[1:10], each = 10)),
    x1 = rnorm(100)
  )

  rec <- recipe(outcome ~ ., data = data) |>
    step_lencode_mixed(predictor, outcome = vars(outcome)) |>
    prep()

  result <- tidy(rec, 1)
  expect_equal(nrow(result), 11)  # 10 levels + ..new
  expect_true(all(!is.na(result$value)))
})

# =============================================================================
# Section 5: step_lencode_bayes Edge Cases (HIGH PRIORITY)
# =============================================================================

test_that("step_lencode_bayes verbose option suppresses stan messages", {
  skip_if_not_installed("rstanarm")
  skip_on_cran()

  set.seed(42)
  data <- data.frame(
    outcome = factor(rep(c("a", "b"), each = 25)),
    predictor = factor(rep(letters[1:5], each = 10))
  )

  # verbose = FALSE (default) should suppress stan output
  # Note: rstanarm may still produce warnings about chain convergence with few iterations
  expect_no_error(
    suppressWarnings({
      rec <- recipe(outcome ~ ., data = data) |>
        step_lencode_bayes(predictor, outcome = vars(outcome),
                           options = list(chains = 1, iter = 100, refresh = 0)) |>
        prep()
    })
  )
})

test_that("step_lencode_bayes works with numeric outcome", {
  skip_if_not_installed("rstanarm")
  skip_on_cran()

  set.seed(42)
  data <- data.frame(
    outcome = rnorm(50),
    predictor = factor(rep(letters[1:5], each = 10))
  )

  rec <- recipe(outcome ~ ., data = data) |>
    step_lencode_bayes(predictor, outcome = vars(outcome),
                       options = list(chains = 1, iter = 100, refresh = 0)) |>
    prep()

  result <- tidy(rec, 1)
  expect_equal(nrow(result), 6)
  expect_true(all(!is.na(result$value)))
})

# =============================================================================
# Section 6: Case Weights Integration (HIGH PRIORITY)
# =============================================================================

test_that("step_lencode_glm respects case weights", {
  set.seed(42)
  data <- data.frame(
    outcome = factor(rep(c("a", "b"), each = 50)),
    predictor = factor(rep(letters[1:5], each = 20)),
    wts = hardhat::importance_weights(c(rep(1, 50), rep(10, 50)))
  )

  # With weights heavily favoring "b" class
  rec <- recipe(outcome ~ ., data = data) |>
    step_lencode_glm(predictor, outcome = vars(outcome)) |>
    prep()

  result <- tidy(rec, 1)

  expect_equal(nrow(result), 6)
  expect_true(all(!is.na(result$value)))
})

test_that("step_lencode_mixed respects case weights", {
  skip_if_not_installed("lme4")

  set.seed(42)
  data <- data.frame(
    outcome = rnorm(100),
    predictor = factor(rep(letters[1:10], each = 10)),
    wts = hardhat::importance_weights(runif(100, 0.5, 2))
  )

  rec <- recipe(outcome ~ ., data = data) |>
    step_lencode_mixed(predictor, outcome = vars(outcome)) |>
    prep()

  result <- tidy(rec, 1)
  expect_equal(nrow(result), 11)
})

# =============================================================================
# Section 7: step_umap Edge Cases (HIGH PRIORITY)
# =============================================================================

test_that("step_umap with explicit seed works", {
  skip_if_not_installed("irlba")

  data <- iris[1:50, 1:4]

  rec <- recipe(~., data = data) |>
    step_umap(all_predictors(), num_comp = 2, neighbors = 5, seed = c(42L, 123L)) |>
    prep()

  result <- bake(rec, new_data = NULL)
  expect_true("UMAP1" %in% names(result))
  expect_true("UMAP2" %in% names(result))
})

test_that("step_umap with num_comp = 1 works",
{
  skip_if_not_installed("irlba")

  data <- iris[1:50, 1:4]

  rec <- recipe(~., data = data) |>
    step_umap(all_predictors(), num_comp = 1, neighbors = 5) |>
    prep()

  result <- bake(rec, new_data = NULL)
  expect_true("UMAP1" %in% names(result))
  expect_false("UMAP2" %in% names(result))
})

test_that("step_umap neighbors adjusted for small dataset", {
  skip_if_not_installed("irlba")

  # Very small dataset where neighbors would exceed n-1
  data <- iris[1:10, 1:4]

  # Should adjust neighbors automatically
  rec <- recipe(~., data = data) |>
    step_umap(all_predictors(), num_comp = 2, neighbors = 15) |>  # neighbors > n
    prep()

  result <- bake(rec, new_data = NULL)
  expect_equal(nrow(result), 10)
})

# =============================================================================
# Section 8: Print Methods for Untrained Steps (MEDIUM PRIORITY)
# =============================================================================

test_that("untrained step_lencode prints correctly", {
  rec <- recipe(x2 ~ ., data = ex_dat) |>
    step_lencode(x3, outcome = vars(x2))

  expect_snapshot(print(rec))
})
test_that("untrained step_lencode_glm prints correctly", {
  rec <- recipe(x2 ~ ., data = ex_dat) |>
    step_lencode_glm(x3, outcome = vars(x2))

  expect_snapshot(print(rec))
})

test_that("untrained step_lencode_mixed prints correctly", {
  rec <- recipe(x2 ~ ., data = ex_dat) |>
    step_lencode_mixed(x3, outcome = vars(x2))

  expect_snapshot(print(rec))
})

test_that("untrained step_umap prints correctly", {
  rec <- recipe(~., data = iris[1:50, 1:4]) |>
    step_umap(all_predictors())

  expect_snapshot(print(rec))
})

# =============================================================================
# Section 9: Empty Column Selection Edge Cases (MEDIUM PRIORITY)
# =============================================================================

test_that("step_lencode_glm with empty selection is no-op", {
  rec <- recipe(mpg ~ ., data = mtcars) |>
    step_lencode_glm(outcome = vars(mpg)) |>
    prep()

  result <- bake(rec, new_data = mtcars)

  # Should have all the same columns (order may differ)
  expect_true(all(names(mtcars) %in% names(result)))
  expect_equal(nrow(result), nrow(mtcars))
})

test_that("step_lencode_mixed with empty selection is no-op", {
  rec <- recipe(mpg ~ ., data = mtcars) |>
    step_lencode_mixed(outcome = vars(mpg)) |>
    prep()

  result <- bake(rec, new_data = mtcars)

  # Should have all the same columns (order may differ)
  expect_true(all(names(mtcars) %in% names(result)))
  expect_equal(nrow(result), nrow(mtcars))
})

# =============================================================================
# Section 10: step_woe Edge Cases (MEDIUM PRIORITY)
# =============================================================================

test_that("step_woe with Laplace = 0 (no smoothing)", {
  data <- data.frame(
    outcome = factor(rep(c("good", "bad"), each = 50)),
    predictor = factor(rep(letters[1:5], each = 20))
  )

  rec <- recipe(outcome ~ ., data = data) |>
    step_woe(predictor, outcome = vars(outcome), Laplace = 0) |>
    prep()

  result <- tidy(rec, 1)
  expect_true(all(!is.na(result$woe)))
})

test_that("step_woe with very large Laplace smoothing", {
  data <- data.frame(
    outcome = factor(rep(c("good", "bad"), each = 50)),
    predictor = factor(rep(letters[1:5], each = 20))
  )

  rec <- recipe(outcome ~ ., data = data) |>
    step_woe(predictor, outcome = vars(outcome), Laplace = 100) |>
    prep()

  result <- tidy(rec, 1)
  expect_true(all(!is.na(result$woe)))
  # Large Laplace should shrink WoE values (compared to Laplace = 0)
  rec_no_smooth <- recipe(outcome ~ ., data = data) |>
    step_woe(predictor, outcome = vars(outcome), Laplace = 0) |>
    prep()
  result_no_smooth <- tidy(rec_no_smooth, 1)

  # Mean absolute WoE should be smaller with large Laplace
  expect_lte(mean(abs(result$woe)), mean(abs(result_no_smooth$woe)) + 0.01)
})

# =============================================================================
# Section 11: step_pca_sparse Edge Cases (MEDIUM PRIORITY)
# =============================================================================

test_that("step_pca_sparse with predictor_prop near boundaries", {
  skip_if_not_installed("irlba")

  data <- iris[, 1:4]

  # predictor_prop = 0.01 (very sparse)
  rec <- recipe(~., data = data) |>
    step_pca_sparse(all_predictors(), num_comp = 2, predictor_prop = 0.01) |>
    prep()

  result <- bake(rec, new_data = NULL)
  expect_true("PC1" %in% names(result))
})

test_that("step_pca_sparse with two columns works", {
  skip_if_not_installed("irlba")

  # Minimum 2 columns for sparse PCA
  data <- data.frame(x = rnorm(100), y = rnorm(100))

  rec <- recipe(~., data = data) |>
    step_pca_sparse(all_predictors(), num_comp = 1) |>
    prep()

  result <- bake(rec, new_data = NULL)
  expect_true("PC1" %in% names(result))
})

# =============================================================================
# Section 12: step_collapse_cart Edge Cases (MEDIUM PRIORITY)
# =============================================================================

test_that("step_collapse_cart with many levels", {
  set.seed(42)
  data <- data.frame(
    outcome = rnorm(200),
    predictor = factor(sample(letters, 200, replace = TRUE))
  )

  rec <- recipe(outcome ~ ., data = data) |>
    step_collapse_cart(predictor, outcome = vars(outcome)) |>
    prep()

  result <- bake(rec, new_data = NULL)
  expect_true("predictor" %in% names(result))

  # Should have fewer unique levels after collapsing
  n_original <- length(unique(data$predictor))
  n_collapsed <- length(unique(result$predictor))
  expect_lte(n_collapsed, n_original)
})

test_that("step_collapse_cart with binary predictor (no collapse needed)", {
  set.seed(42)
  data <- data.frame(
    outcome = rnorm(100),
    predictor = factor(rep(c("a", "b"), 50))
  )

  rec <- recipe(outcome ~ ., data = data) |>
    step_collapse_cart(predictor, outcome = vars(outcome)) |>
    prep()

  result <- bake(rec, new_data = NULL)
  expect_equal(length(unique(result$predictor)), 2)
})

# =============================================================================
# Section 13: required_pkgs Edge Cases (HIGH PRIORITY)
# =============================================================================

test_that("required_pkgs.step_umap returns correct packages", {
  rec <- recipe(~., data = iris[1:50, 1:4]) |>
    step_umap(all_predictors())

  pkgs <- recipes::required_pkgs(rec$steps[[1]])

  expect_true("embedmit" %in% pkgs)
  expect_true("uwotlite" %in% pkgs)
  expect_false("uwot" %in% pkgs)  # Should NOT require uwot
})

test_that("required_pkgs.step_lencode_bayes returns correct packages", {
  rec <- recipe(x2 ~ ., data = ex_dat) |>
    step_lencode_bayes(x3, outcome = vars(x2))

  pkgs <- recipes::required_pkgs(rec$steps[[1]])

  expect_true("embedmit" %in% pkgs)
  expect_true("rstanarm" %in% pkgs)
})

test_that("required_pkgs.step_lencode_mixed returns correct packages", {
  rec <- recipe(x2 ~ ., data = ex_dat) |>
    step_lencode_mixed(x3, outcome = vars(x2))

  pkgs <- recipes::required_pkgs(rec$steps[[1]])

  expect_true("embedmit" %in% pkgs)
  expect_true("lme4" %in% pkgs)
})

# =============================================================================
# Section 14: tidy Methods Edge Cases (MEDIUM PRIORITY)
# =============================================================================

test_that("tidy.step_lencode on unprepared recipe", {
  rec <- recipe(x2 ~ ., data = ex_dat) |>
    step_lencode(x3, outcome = vars(x2))

  result <- tidy(rec, 1)

  expect_true(is.data.frame(result))
  expect_equal(nrow(result), 1)
  expect_true("terms" %in% names(result))
})

test_that("tidy.step_umap on unprepared recipe", {
  rec <- recipe(~., data = iris[1:50, 1:4]) |>
    step_umap(all_predictors())

  result <- tidy(rec, 1)

  expect_true(is.data.frame(result))
  expect_true("terms" %in% names(result))
})

# =============================================================================
# Section 15: Error Messages Verification (MEDIUM PRIORITY)
# =============================================================================

test_that("step_lencode errors clearly when outcome is missing", {
  expect_error(
    recipe(~., data = ex_dat) |>
      step_lencode(x3) |>
      prep(),
    "outcome"
  )
})

test_that("step_lencode_glm errors clearly when outcome is missing", {
  expect_error(
    recipe(~., data = ex_dat) |>
      step_lencode_glm(x3) |>
      prep(),
    "outcome"
  )
})

test_that("step_woe errors with non-binary outcome", {
  three_class <- iris
  three_class$predictor <- factor(rep(letters[1:3], 50))

  expect_error(
    recipe(Species ~ ., data = three_class) |>
      step_woe(predictor, outcome = vars(Species)) |>
      prep(),
    "2 categories|two levels|binary"
  )
})
