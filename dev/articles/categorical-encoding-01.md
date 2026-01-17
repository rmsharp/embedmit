# Encoding Categorical Data

This vignette demonstrates categorical encoding techniques using
`embedmit`, closely following the approach from [Chapter 17 of Tidy
Modeling with R](https://www.tmwr.org/categorical) by Max Kuhn and Julia
Silge.

## Introduction

For many models, the predictors must be encoded as numbers before
modeling. The most common encoding for categorical variables is to
create *dummy* or *indicator* variables. However, when a categorical
variable has many levels (high cardinality), dummy encoding creates many
columns and can lead to issues with:

- Computational efficiency
- Overfitting on rare categories
- Handling new/unseen categories at prediction time

This chapter explores *effect encodings* (also called likelihood
encodings) as an alternative approach that replaces categorical
variables with numeric values representing their relationship to the
outcome.

## Setup

``` r
library(embedmit)
library(recipes)
library(dplyr)
library(rsample)
library(ggplot2)
library(purrr)
library(modeldata)

# Load the Ames housing data
data(ames)

# Create train/test split
set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test <- testing(ames_split)
```

## Is an Encoding Necessary?

Before diving into encoding strategies, it’s worth noting that some
models can handle categorical predictors natively:

- **Tree-based models** (decision trees, random forests, boosted trees)
  can use categorical variables directly by finding optimal splits
- **Naive Bayes** models compute class probabilities without requiring
  numeric encoding

For these models, converting categories to dummy variables typically
doesn’t improve performance and may even hurt it. The `recipes` package
provides
[`step_dummy()`](https://recipes.tidymodels.org/reference/step_dummy.html)
for standard dummy encoding when needed.

## Using the Outcome for Encoding Predictors

Effect encodings (sometimes called likelihood or target encodings)
replace categorical levels with a numeric value that represents the
relationship between that level and the outcome. This approach:

1.  Reduces a high-cardinality categorical to a single numeric column
2.  Naturally handles novel categories at prediction time
3.  Can incorporate regularization to prevent overfitting

### Visualizing Neighborhood Effects

The Ames data has 28 different neighborhoods. Let’s visualize how sale
price varies across them:

``` r
ames_train %>%
  group_by(Neighborhood) %>%
  summarize(
    mean = mean(Sale_Price),
    std_err = sd(Sale_Price) / sqrt(length(Sale_Price)),
    .groups = "drop"
  ) %>%
  ggplot(aes(y = reorder(Neighborhood, mean), x = mean)) +
  geom_point() +
  geom_errorbar(aes(xmin = mean - 1.64 * std_err, xmax = mean + 1.64 * std_err)) +
  labs(y = NULL, x = "Price (mean, log scale)") +
  theme_minimal()
```

![Mean sale price by neighborhood with 90% confidence intervals.
Neighborhoods are ordered by their mean sale
price.](categorical-encoding-01_files/figure-html/neighborhood-viz-1.png)

Mean sale price by neighborhood with 90% confidence intervals.
Neighborhoods are ordered by their mean sale price.

Some neighborhoods (like Northridge Heights, Stone Brook) command
premium prices, while others (like Meadow Village, Iowa DOT and Rail
Road) are associated with lower prices.

### GLM-Based Effect Encoding

The
[`step_lencode_glm()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_glm.md)
function uses a generalized linear model to estimate the effect of each
category level. The recipe below demonstrates the approach:

``` r
ames_glm <-
recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_lencode_glm(Neighborhood, outcome = vars(Sale_Price)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact(~ Gr_Liv_Area:starts_with("Bldg_Type_")) %>%
  step_ns(Latitude, Longitude, deg_free = 20)

ames_glm
```

We can examine the learned encodings by preparing the recipe and using
[`tidy()`](https://generics.r-lib.org/reference/tidy.html):

``` r
glm_estimates <-
  prep(ames_glm) %>%
  tidy(number = 2)

glm_estimates
#> # A tibble: 29 × 4
#>    level                value terms        id               
#>    <chr>                <dbl> <chr>        <chr>            
#>  1 North_Ames         144416. Neighborhood lencode_glm_yj20u
#>  2 College_Creek      202763. Neighborhood lencode_glm_yj20u
#>  3 Old_Town           124999. Neighborhood lencode_glm_yj20u
#>  4 Edwards            130460. Neighborhood lencode_glm_yj20u
#>  5 Somerset           232310. Neighborhood lencode_glm_yj20u
#>  6 Northridge_Heights 321119. Neighborhood lencode_glm_yj20u
#>  7 Gilbert            191159. Neighborhood lencode_glm_yj20u
#>  8 Sawyer             137208. Neighborhood lencode_glm_yj20u
#>  9 Northwest_Ames     188726. Neighborhood lencode_glm_yj20u
#> 10 Sawyer_West        182651. Neighborhood lencode_glm_yj20u
#> # ℹ 19 more rows
```

Each neighborhood is replaced by a single numeric value representing its
effect on sale price (on the log scale). Higher values indicate
neighborhoods associated with higher prices.

### Handling Novel Categories

A key advantage of effect encoding is graceful handling of categories
not seen during training. The encoding includes a special `..new` level:

``` r
glm_estimates %>%
  filter(level == "..new")
#> # A tibble: 1 × 4
#>   level   value terms        id               
#>   <chr>   <dbl> <chr>        <chr>            
#> 1 ..new 183150. Neighborhood lencode_glm_yj20u
```

When the model encounters an unseen neighborhood at prediction time, it
uses this default encoding (typically close to the overall mean).

### Effect Encodings with Partial Pooling

The GLM approach estimates each neighborhood’s effect independently. For
neighborhoods with few observations, these estimates can be unreliable.

The
[`step_lencode_mixed()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_mixed.md)
function uses *hierarchical* or *mixed effects* models that apply
**partial pooling**. This shrinks estimates toward the overall mean,
with more shrinkage for categories with fewer observations:

``` r
ames_mixed <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_lencode_mixed(Neighborhood, outcome = vars(Sale_Price)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact(~ Gr_Liv_Area:starts_with("Bldg_Type_")) %>%
  step_ns(Latitude, Longitude, deg_free = 20)

ames_mixed
```

``` r
mixed_estimates <-
  prep(ames_mixed) %>%
  tidy(number = 2)

mixed_estimates
#> # A tibble: 29 × 4
#>    level                value terms        id                 
#>    <chr>                <dbl> <chr>        <chr>              
#>  1 North_Ames         144488. Neighborhood lencode_mixed_AmBz1
#>  2 College_Creek      202724. Neighborhood lencode_mixed_AmBz1
#>  3 Old_Town           125183. Neighborhood lencode_mixed_AmBz1
#>  4 Edwards            130698. Neighborhood lencode_mixed_AmBz1
#>  5 Somerset           232135. Neighborhood lencode_mixed_AmBz1
#>  6 Northridge_Heights 320533. Neighborhood lencode_mixed_AmBz1
#>  7 Gilbert            191145. Neighborhood lencode_mixed_AmBz1
#>  8 Sawyer             137433. Neighborhood lencode_mixed_AmBz1
#>  9 Northwest_Ames     188722. Neighborhood lencode_mixed_AmBz1
#> 10 Sawyer_West        182683. Neighborhood lencode_mixed_AmBz1
#> # ℹ 19 more rows
```

``` r
mixed_estimates %>%
  filter(level == "..new")
#> # A tibble: 1 × 4
#>   level   value terms        id                 
#>   <chr>   <dbl> <chr>        <chr>              
#> 1 ..new 183225. Neighborhood lencode_mixed_AmBz1
```

### Comparing Pooling Methods

Let’s visualize how partial pooling affects the estimates:

``` r
glm_estimates %>%
  rename(`no pooling` = value) %>%
  left_join(
    mixed_estimates %>%
      rename(`partial pooling` = value),
    by = "level"
  ) %>%
  left_join(
    ames_train %>%
      count(Neighborhood) %>%
      mutate(level = as.character(Neighborhood)),
    by = "level"
  ) %>%
  filter(!is.na(n)) %>%
  ggplot(aes(`no pooling`, `partial pooling`, size = sqrt(n))) +
  geom_abline(color = "gray50", lty = 2) +
  geom_point(alpha = 0.7) +
  coord_fixed() +
  labs(size = "sqrt(n)") +
  theme_minimal()
```

![Comparison of GLM (no pooling) versus mixed effects (partial pooling)
encodings. Point size represents the number of observations in each
neighborhood. Points below the diagonal indicate shrinkage toward the
mean.](categorical-encoding-01_files/figure-html/compare-pooling-1.png)

Comparison of GLM (no pooling) versus mixed effects (partial pooling)
encodings. Point size represents the number of observations in each
neighborhood. Points below the diagonal indicate shrinkage toward the
mean.

Neighborhoods with fewer observations (smaller points) show more
shrinkage—their partial pooling estimates are pulled toward the diagonal
(overall mean). Neighborhoods with many observations retain estimates
close to the unpooled GLM values.

### Bayesian Effect Encoding

For the most principled uncertainty quantification,
[`step_lencode_bayes()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_bayes.md)
uses a fully Bayesian approach (requires the `rstanarm` package):

``` r
ames_bayes <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_lencode_bayes(Neighborhood, outcome = vars(Sale_Price)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact(~ Gr_Liv_Area:starts_with("Bldg_Type_")) %>%
  step_ns(Latitude, Longitude, deg_free = 20)
```

## Feature Hashing

An alternative approach for high-cardinality categoricals is *feature
hashing* (the “hashing trick”). This uses a hash function to map
categories to a fixed number of columns:

``` r
library(rlang)

ames_hashed <-
  ames_train %>%
  mutate(Hash = map_chr(Neighborhood, hash))

ames_hashed %>%
  select(Neighborhood, Hash) %>%
  head(6)
#> # A tibble: 6 × 2
#>   Neighborhood    Hash                            
#>   <fct>           <chr>                           
#> 1 North_Ames      076543f71313e522efe157944169d919
#> 2 North_Ames      076543f71313e522efe157944169d919
#> 3 Briardale       b598bec306983e3e68a3118952df8cf0
#> 4 Briardale       b598bec306983e3e68a3118952df8cf0
#> 5 Northpark_Villa 6af95b5db968bf393e78188a81e0e1e4
#> 6 Northpark_Villa 6af95b5db968bf393e78188a81e0e1e4
```

We can reduce these to a smaller number of bins using the modulo
operation:

``` r
ames_hashed %>%
  mutate(
    Hash = strtoi(substr(Hash, 26, 32), base = 16L),
    Hash = Hash %% 16
  ) %>%
  select(Neighborhood, Hash) %>%
  head(10)
#> # A tibble: 10 × 2
#>    Neighborhood     Hash
#>    <fct>           <dbl>
#>  1 North_Ames          9
#>  2 North_Ames          9
#>  3 Briardale           0
#>  4 Briardale           0
#>  5 Northpark_Villa     4
#>  6 Northpark_Villa     4
#>  7 Sawyer_West         9
#>  8 Sawyer_West         9
#>  9 Sawyer              8
#> 10 Sawyer              8
```

The `textrecipes` package provides `step_dummy_hash()` for this
approach. Note that `embedmit` previously provided
[`step_feature_hash()`](https://rmsharp.github.io/embedmit/dev/reference/step_feature_hash.md)
but this is now deprecated in favor of `textrecipes::step_dummy_hash()`.

## More Encoding Options

The `embedmit` package offers additional encoding methods:

| Function                                                                                         | Description                                         |
|--------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| [`step_lencode_glm()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_glm.md)     | GLM-based effect encoding (no pooling)              |
| [`step_lencode_mixed()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_mixed.md) | Mixed effects encoding (partial pooling)            |
| [`step_lencode_bayes()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_bayes.md) | Bayesian encoding (full uncertainty quantification) |
| [`step_woe()`](https://rmsharp.github.io/embedmit/dev/reference/step_woe.md)                     | Weight of evidence transformation (binary outcomes) |
| [`step_umap()`](https://rmsharp.github.io/embedmit/dev/reference/step_umap.md)                   | UMAP embeddings via uwotlite                        |

## Chapter Summary

Encoding categorical predictors is a fundamental preprocessing step for
many models:

- **Dummy variables** work well for low-cardinality categoricals but
  create many columns for high-cardinality variables
- **Effect encodings** replace categories with a single numeric value
  representing the outcome relationship, handling high cardinality
  elegantly
- **Partial pooling** (via
  [`step_lencode_mixed()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_mixed.md))
  provides regularization for categories with few observations
- **Feature hashing** offers a fast, memory-efficient alternative that
  works well for very high cardinality
- Tree-based models often work better with untransformed categorical
  variables

The choice of encoding strategy depends on:

1.  The model being used
2.  The cardinality of categorical variables
3.  Whether interpretability is important
4.  How novel categories should be handled at prediction time

## References

- Kuhn, M., & Johnson, K. (2019). *Feature Engineering and Selection*.
  CRC Press.
- Kuhn, M., & Silge, J. (2022). *Tidy Modeling with R*. O’Reilly Media.
- Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality
  categorical attributes in classification and prediction problems. *ACM
  SIGKDD Explorations*, 3(1), 27-32.
