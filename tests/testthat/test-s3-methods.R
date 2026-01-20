rlang::local_options(lifecycle_verbosity = "quiet")

r1 <- recipe(mpg ~ ., data = mtcars)
r2 <- r1 |> step_lencode_bayes(wt, outcome = vars(mpg))
r3 <- r1 |> step_discretize_cart(disp, outcome = vars(mpg))
r4 <- r1 |> step_discretize_xgb(disp, outcome = vars(mpg))
r5 <- r1 |> step_umap(disp, outcome = vars(mpg))

# Helper to check required packages, accepting either embedmit or embed variants
# When embed is installed, it overrides S3 methods and returns "embed"/"uwot"
check_required_pkgs <- function(actual, base_pkgs, mit_pkg, orig_pkg = NULL) {
  # Must contain base packages
  expect_true(all(base_pkgs %in% actual),
    info = sprintf("Missing base packages. Expected: %s, Got: %s",
                   paste(base_pkgs, collapse = ", "), paste(actual, collapse = ", ")))
  # Must contain either MIT fork or original package
  mit_or_orig <- if (is.null(orig_pkg)) mit_pkg else c(mit_pkg, orig_pkg)
  expect_true(any(mit_or_orig %in% actual),
    info = sprintf("Missing package variant. Expected one of: %s, Got: %s",
                   paste(mit_or_orig, collapse = ", "), paste(actual, collapse = ", ")))
}

test_that("required packages", {
  expect_equal(required_pkgs(r1), "recipes")

  # These tests accept either embedmit or embed (when embed overrides methods)
  check_required_pkgs(required_pkgs(r2), c("recipes", "rstanarm"), "embedmit", "embed")
  check_required_pkgs(required_pkgs(r3), c("recipes", "rpart"), "embedmit", "embed")
  check_required_pkgs(required_pkgs(r4), c("recipes", "xgboost"), "embedmit", "embed")


  # step_umap can return uwotmit or uwot, and embedmit or embed
  pkgs5 <- required_pkgs(r5)
  expect_true("recipes" %in% pkgs5)
  expect_true(any(c("uwotmit", "uwot") %in% pkgs5),
    info = sprintf("Expected uwotmit or uwot, got: %s", paste(pkgs5, collapse = ", ")))
  expect_true(any(c("embedmit", "embed") %in% pkgs5),
    info = sprintf("Expected embedmit or embed, got: %s", paste(pkgs5, collapse = ", ")))
})
