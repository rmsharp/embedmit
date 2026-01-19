library(embedmit)
library(dplyr)
library(testthat)

embedmit:::is_tf_available()

test_that("embedmit package loads correctly", {
  expect_true("embedmit" %in% loadedNamespaces())
})
