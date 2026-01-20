# Encoding Categorical Data

This vignette demonstrates categorical encoding techniques using
`embedmit`, closely following [Chapter 17 of Tidy Modeling with
R](https://www.tmwr.org/categorical) by Max Kuhn and Julia Silge.

## Introduction

For statistical modeling in R, the preferred representation for
categorical or nominal data is a *factor*, which is a variable that can
take on a limited number of different values. Internally, factors are
stored as a vector of integer values together with a set of text labels.

The most straightforward approach for transforming a categorical
variable to a numeric representation is to create dummy or indicator
variables from the levels. However, this approach does not work well
when you have a variable with high cardinality (too many levels) or when
you may encounter novel values at prediction time (new levels).

This vignette explores alternative encoding strategies that address
these challenges.

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

A minority of models, such as those based on trees or rules, can handle
categorical data natively and do not require encoding or transformation
of these kinds of features. For example:

- **Tree-based models** (decision trees, random forests, boosted trees)
  can find optimal splits on categorical variables directly
- **Naive Bayes** models compute class probabilities without requiring
  numeric encoding

For these models, research has shown that creating dummy variables
typically does not improve performance and can increase computation
time. The `recipes` package provides
[`step_dummy()`](https://recipes.tidymodels.org/reference/step_dummy.html)
for standard dummy encoding when it is needed.

## Using the Outcome for Encoding Predictors

There are multiple options for encodings more complex than dummy or
indicator variables. One method called *effect* or *likelihood
encodings* replaces the original categorical variables with a single
numeric column that measures the effect of those data.

For example, for the Ames housing data, we can compute the mean or
median sale price for each neighborhood and use this value to represent
that categorical level. Effect encodings can also seamlessly handle
situations where a novel factor level is encountered in the data.

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

### GLM-Based Effect Encoding

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
effect on sale price.

### Handling Novel Categories

Effect encodings can seamlessly handle situations where a novel factor
level is encountered in the data. The encoding includes a special
`..new` level:

``` r
glm_estimates %>%
  filter(level == "..new")
#> # A tibble: 1 × 4
#>   level   value terms        id               
#>   <chr>   <dbl> <chr>        <chr>            
#> 1 ..new 183150. Neighborhood lencode_glm_yj20u
```

When the model encounters an unseen neighborhood at prediction time, it
uses this default encoding.

## Effect Encodings with Partial Pooling

Creating an effect encoding with
[`step_lencode_glm()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_glm.md)
estimates the effect separately for each factor level (in this case,
neighborhood). However, some of these neighborhoods have many houses in
them, and some have only a few. There is much more uncertainty in our
measurement of price for neighborhoods with few training set homes than
for neighborhoods with many.

We can use *partial pooling* to adjust these estimates so that levels
with small sample sizes are shrunk toward the overall mean. The
[`step_lencode_mixed()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_mixed.md)
function uses hierarchical or mixed effects models to accomplish this:

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
neighborhood.](categorical-encoding-01_files/figure-html/compare-pooling-1.png)

Comparison of GLM (no pooling) versus mixed effects (partial pooling)
encodings. Point size represents the number of observations in each
neighborhood.

When we use partial pooling, we shrink the effect estimates toward the
mean because we don’t have as much evidence about the price in
neighborhoods with few observations. Neighborhoods with many
observations retain estimates close to the unpooled GLM values.

### Bayesian Effect Encoding

For fully Bayesian uncertainty quantification,
[`step_lencode_bayes()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_bayes.md)
provides an alternative approach (requires the `rstanarm` package):

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

*Feature hashing* methods also create dummy variables, but only consider
the value of the category to assign it to a predefined pool of dummy
variables. A hashing function takes an input of variable size and maps
it to an output of fixed size.

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

In feature hashing, the number of possible hashes is a hyperparameter
and is set by the model developer through computing the modulo of the
integer hashes:

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

Feature hashing can handle new category levels at prediction time, since
it does not rely on pre-determined dummy variables. The `textrecipes`
package provides `step_dummy_hash()` for this approach.

## More Encoding Options

The `embedmit` package offers additional encoding methods:

| Function                                                                                         | Description                                         |
|--------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| [`step_lencode_glm()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_glm.md)     | GLM-based effect encoding (no pooling)              |
| [`step_lencode_mixed()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_mixed.md) | Mixed effects encoding (partial pooling)            |
| [`step_lencode_bayes()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_bayes.md) | Bayesian encoding (full uncertainty quantification) |
| [`step_woe()`](https://rmsharp.github.io/embedmit/dev/reference/step_woe.md)                     | Weight of evidence transformation (binary outcomes) |
| [`step_umap()`](https://rmsharp.github.io/embedmit/dev/reference/step_umap.md)                   | UMAP embeddings via uwotmit                         |

## Chapter Summary

The most straightforward option for transforming a categorical variable
to a numeric representation is to create dummy variables from the
levels, but this option does not work well when you have a variable with
high cardinality (too many levels) or when you may see novel values at
prediction time (new levels).

Effect encodings and feature hashing address these challenges:

- **Effect encodings** replace categories with a single numeric value
  measuring the outcome relationship
- **Partial pooling** (via
  [`step_lencode_mixed()`](https://rmsharp.github.io/embedmit/dev/reference/step_lencode_mixed.md))
  adjusts estimates so that levels with small sample sizes are shrunk
  toward the overall mean
- **Feature hashing** maps categories to a predefined pool of dummy
  variables and can handle novel categories

Other options include entity embeddings (learned via a neural network
with
[`step_embed()`](https://rmsharp.github.io/embedmit/dev/reference/step_embed.md))
and weight of evidence transformation (via
[`step_woe()`](https://rmsharp.github.io/embedmit/dev/reference/step_woe.md)).

## References

- Kuhn, M., & Johnson, K. (2019). *Feature Engineering and Selection*.
  CRC Press.
- Kuhn, M., & Silge, J. (2022). *Tidy Modeling with R*. O’Reilly Media.
- Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality
  categorical attributes in classification and prediction problems. *ACM
  SIGKDD Explorations*, 3(1), 27-32.
