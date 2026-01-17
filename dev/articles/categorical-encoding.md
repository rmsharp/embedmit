# Encoding Categorical Data with embedmit

This vignette demonstrates categorical encoding techniques using
`embedmit`, following the approach from [Chapter 17 of Tidy Modeling
with R](https://www.tmwr.org/categorical).

The `embedmit` package provides MIT-licensed effect encoding methods
that transform categorical variables into numeric representations
suitable for machine learning models.

## Setup

``` r
library(embedmit)
library(tidymodels)
library(modeldata)

# Load the Ames housing data
data(ames)

# Create train/test split
set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test <- testing(ames_split)
```

## The Challenge: High-Cardinality Categorical Variables

The Ames dataset contains neighborhood information with many levels:

``` r
ames_train %>%
  count(Neighborhood, sort = TRUE) %>%
  print(n = 30)
#> # A tibble: 28 × 2
#>    Neighborhood                                n
#>    <fct>                                   <int>
#>  1 North_Ames                                354
#>  2 College_Creek                             221
#>  3 Old_Town                                  198
#>  4 Somerset                                  147
#>  5 Edwards                                   140
#>  6 Northridge_Heights                        131
#>  7 Sawyer                                    130
#>  8 Gilbert                                   126
#>  9 Northwest_Ames                            104
#> 10 Sawyer_West                                96
#> 11 Mitchell                                   91
#> 12 Brookside                                  87
#> 13 Crawford                                   85
#> 14 Iowa_DOT_and_Rail_Road                     74
#> 15 Timberland                                 59
#> 16 Northridge                                 55
#> 17 South_and_West_of_Iowa_State_University    40
#> 18 Stone_Brook                                37
#> 19 Clear_Creek                                36
#> 20 Meadow_Village                             29
#> 21 Bloomington_Heights                        26
#> 22 Briardale                                  24
#> 23 Northpark_Villa                            19
#> 24 Veenker                                    16
#> 25 Blueste                                     7
#> 26 Greens                                      7
#> 27 Green_Hills                                 2
#> 28 Landmark                                    1
```

Creating dummy variables for all these levels leads to many columns and
potential overfitting. Effect encoding offers an alternative approach.

## Effect Encoding Methods

Effect encoding replaces each categorical level with a single numeric
value that represents the relationship between that level and the
outcome. This reduces a high-cardinality categorical variable to a
single numeric column.

### Method 1: GLM-based Effect Encoding

The
[`step_lencode_glm()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_glm.md)
function uses a generalized linear model to estimate the effect of each
category level:

``` r
ames_glm <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_lencode_glm(Neighborhood, outcome = vars(Sale_Price)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact(~ Gr_Liv_Area:starts_with("Bldg_Type_")) %>%
  step_ns(Latitude, Longitude, deg_free = 20)

# Prepare the recipe and examine the encodings
glm_prep <- prep(ames_glm)
glm_estimates <- tidy(glm_prep, number = 2)

glm_estimates %>%
  arrange(desc(value)) %>%
  print(n = 15)
#> # A tibble: 29 × 4
#>    level                 value terms        id               
#>    <chr>                 <dbl> <chr>        <chr>            
#>  1 Stone_Brook         334526. Neighborhood lencode_glm_yj20u
#>  2 Northridge          331721. Neighborhood lencode_glm_yj20u
#>  3 Northridge_Heights  321119. Neighborhood lencode_glm_yj20u
#>  4 Green_Hills         280000  Neighborhood lencode_glm_yj20u
#>  5 Veenker             250878. Neighborhood lencode_glm_yj20u
#>  6 Timberland          247627. Neighborhood lencode_glm_yj20u
#>  7 Somerset            232310. Neighborhood lencode_glm_yj20u
#>  8 Clear_Creek         211551. Neighborhood lencode_glm_yj20u
#>  9 Crawford            205434. Neighborhood lencode_glm_yj20u
#> 10 College_Creek       202763. Neighborhood lencode_glm_yj20u
#> 11 Bloomington_Heights 197888. Neighborhood lencode_glm_yj20u
#> 12 Gilbert             191159. Neighborhood lencode_glm_yj20u
#> 13 Greens              190643. Neighborhood lencode_glm_yj20u
#> 14 Northwest_Ames      188726. Neighborhood lencode_glm_yj20u
#> 15 ..new               183150. Neighborhood lencode_glm_yj20u
#> # ℹ 14 more rows
```

The GLM method produces separate estimates for each neighborhood level.
Levels associated with higher sale prices get higher encoded values.

### Handling Novel Categories

A key advantage of effect encoding is handling categories not seen
during training:

``` r
# The "..new" level represents the encoding for unseen categories
glm_estimates %>%
  filter(level == "..new")
#> # A tibble: 1 × 4
#>   level   value terms        id               
#>   <chr>   <dbl> <chr>        <chr>            
#> 1 ..new 183150. Neighborhood lencode_glm_yj20u
```

### Method 2: Mixed Effects Encoding (Partial Pooling)

The
[`step_lencode_mixed()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_mixed.md)
function uses mixed effects models that apply **partial pooling**. This
shrinks estimates toward the overall mean, especially for categories
with few observations:

``` r
ames_mixed <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_lencode_mixed(Neighborhood, outcome = vars(Sale_Price)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact(~ Gr_Liv_Area:starts_with("Bldg_Type_")) %>%
  step_ns(Latitude, Longitude, deg_free = 20)

mixed_prep <- prep(ames_mixed)
mixed_estimates <- tidy(mixed_prep, number = 2)

mixed_estimates %>%
  arrange(desc(value)) %>%
  print(n = 15)
#> # A tibble: 29 × 4
#>    level                 value terms        id                 
#>    <chr>                 <dbl> <chr>        <chr>              
#>  1 Stone_Brook         332265. Neighborhood lencode_mixed_AmBz1
#>  2 Northridge          330222. Neighborhood lencode_mixed_AmBz1
#>  3 Northridge_Heights  320533. Neighborhood lencode_mixed_AmBz1
#>  4 Green_Hills         259323. Neighborhood lencode_mixed_AmBz1
#>  5 Veenker             248679. Neighborhood lencode_mixed_AmBz1
#>  6 Timberland          247046. Neighborhood lencode_mixed_AmBz1
#>  7 Somerset            232135. Neighborhood lencode_mixed_AmBz1
#>  8 Clear_Creek         211178. Neighborhood lencode_mixed_AmBz1
#>  9 Crawford            205316. Neighborhood lencode_mixed_AmBz1
#> 10 College_Creek       202724. Neighborhood lencode_mixed_AmBz1
#> 11 Bloomington_Heights 197672. Neighborhood lencode_mixed_AmBz1
#> 12 Gilbert             191145. Neighborhood lencode_mixed_AmBz1
#> 13 Greens              190440. Neighborhood lencode_mixed_AmBz1
#> 14 Northwest_Ames      188722. Neighborhood lencode_mixed_AmBz1
#> 15 ..new               183225. Neighborhood lencode_mixed_AmBz1
#> # ℹ 14 more rows
```

### Comparing GLM vs Mixed Effects

Let’s visualize how partial pooling affects the estimates:

``` r
comparison <- glm_estimates %>%
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
  filter(level != "..new")

library(ggplot2)

ggplot(comparison, aes(`no pooling`, `partial pooling`, size = sqrt(n))) +

geom_abline(color = "gray50", lty = 2) +
  geom_point(alpha = 0.7) +
  coord_fixed() +
  labs(
    title = "Effect of Partial Pooling on Neighborhood Encodings",
    subtitle = "Points off the diagonal show shrinkage toward the mean",
    size = "sqrt(n)"
  ) +
  theme_minimal()
```

![Comparison of GLM (no pooling) vs Mixed Effects (partial pooling)
encodings. Point size represents sample size for each
neighborhood.](categorical-encoding_files/figure-html/compare-methods-1.png)

Comparison of GLM (no pooling) vs Mixed Effects (partial pooling)
encodings. Point size represents sample size for each neighborhood.

Small neighborhoods (smaller points) show more shrinkage toward the
diagonal, while large neighborhoods retain estimates closer to the
unpooled GLM values.

### Method 3: Bayesian Effect Encoding

For the most principled uncertainty quantification, use
[`step_lencode_bayes()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_bayes.md):

``` r
# Requires rstanarm package
ames_bayes <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_lencode_bayes(Neighborhood, outcome = vars(Sale_Price)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact(~ Gr_Liv_Area:starts_with("Bldg_Type_")) %>%
  step_ns(Latitude, Longitude, deg_free = 20)
```

## Visualizing Neighborhood Effects

``` r
ames_train %>%
  group_by(Neighborhood) %>%
  summarize(
    mean = mean(Sale_Price),
    std_err = sd(Sale_Price) / sqrt(n()),
    .groups = "drop"
  ) %>%
  ggplot(aes(y = reorder(Neighborhood, mean), x = mean)) +
  geom_point() +
  geom_errorbar(aes(xmin = mean - 1.64 * std_err, xmax = mean + 1.64 * std_err)) +
  labs(
    y = NULL,
    x = "Sale Price (mean with 90% CI)",
    title = "Neighborhood Effects on Sale Price"
  ) +
  theme_minimal()
```

![Mean sale price by neighborhood with 90% confidence
intervals](categorical-encoding_files/figure-html/neighborhood-viz-1.png)

Mean sale price by neighborhood with 90% confidence intervals

## Building a Complete Model

Here’s how to use effect encoding in a complete modeling workflow:

``` r
# Define the recipe with mixed effect encoding
ames_rec <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_lencode_mixed(Neighborhood, outcome = vars(Sale_Price)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact(~ Gr_Liv_Area:starts_with("Bldg_Type_")) %>%
  step_ns(Latitude, Longitude, deg_free = 20)

# Define a linear regression model
lm_spec <- linear_reg() %>%
  set_engine("lm")

# Create and fit the workflow
ames_wf <- workflow() %>%
  add_recipe(ames_rec) %>%
  add_model(lm_spec)

ames_fit <- fit(ames_wf, data = ames_train)

# Evaluate on test set
ames_pred <- predict(ames_fit, ames_test) %>%
  bind_cols(ames_test %>% select(Sale_Price))

ames_pred %>%
  metrics(truth = Sale_Price, estimate = .pred)
#> # A tibble: 3 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 rmse    standard   38832.   
#> 2 rsq     standard       0.779
#> 3 mae     standard   25362.
```

## Additional Encoding Methods in embedmit

Beyond effect encoding, `embedmit` provides other useful
transformations:

- [`step_woe()`](https://rmsharp.github.io/embedmit/dev/reference/step_woe.md) -
  Weight of Evidence encoding
- [`step_collapse_cart()`](https://rmsharp.github.io/embedmit/dev/reference/step_collapse_cart.md) -
  Collapse factor levels using CART
- [`step_collapse_stringdist()`](https://rmsharp.github.io/embedmit/dev/reference/step_collapse_stringdist.md) -
  Collapse similar levels based on string distance
- [`step_discretize_cart()`](https://rmsharp.github.io/embedmit/dev/reference/step_discretize_cart.md) -
  Discretize numeric variables using CART
- [`step_discretize_xgb()`](https://rmsharp.github.io/embedmit/dev/reference/step_discretize_xgb.md) -
  Discretize using XGBoost
- [`step_umap()`](https://rmsharp.github.io/embedmit/dev/reference/step_umap.md) -
  UMAP embeddings (using uwotlite)

## Summary

Effect encoding is a powerful technique for handling high-cardinality
categorical variables. The `embedmit` package provides three approaches:

| Method   | Function                                                                                         | Characteristics                               |
|----------|--------------------------------------------------------------------------------------------------|-----------------------------------------------|
| GLM      | [`step_lencode_glm()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_glm.md)     | Fast, no pooling, may overfit rare categories |
| Mixed    | [`step_lencode_mixed()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_mixed.md) | Partial pooling, better for small samples     |
| Bayesian | [`step_lencode_bayes()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_bayes.md) | Full uncertainty quantification               |

All methods gracefully handle novel categories at prediction time by
using a learned default encoding.
