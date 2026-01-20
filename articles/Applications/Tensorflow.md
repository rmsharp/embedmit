# Entity Embeddings of Categorical Variables using TensorFlow

The approach encodes categorical data as multiple numeric variables
using a *word embedding* approach. Originally intended as a way to take
a large number of word identifiers and represent them in a smaller
dimension. Good references on this are [Guo and Berkhahn
(2016)](https://arxiv.org/abs/1604.06737) and Chapter 6 of [Francois and
Allaire (2018)](https://www.manning.com/books/deep-learning-with-r).

The methodology first translates the *C* factor levels as a set of
integer values then randomly allocates them to the new *D* numeric
columns. These columns are optionally connected in a neural network to
an intermediate layer of hidden units. Optionally, other predictors can
be added to the network in the usual way (via the `predictors` argument)
that also link to the hidden layer. This implementation uses a single
layer with ReLu activations. Finally, an output layer is used with
either linear activation (for numeric outcomes) or softmax (for
classification).

To translate this model to a set of embeddings, the coefficients of the
original embedding layer are used to represent the original factor
levels.

As an example, we use the Ames housing data where the sale price of
houses are being predicted. One predictor, neighborhood, has the most
factor levels of the predictors.

``` r
library(tidymodels)
data(ames)
length(levels(ames$Neighborhood))
```

    ## [1] 29

The distribution of data in the neighborhood is not uniform:

``` r
ames |>
  count(Neighborhood) |>
  ggplot(aes(n, reorder(Neighborhood, n))) +
  geom_col() +
  labs(y = NULL) +
  theme_bw()
```

![Horizontal bar chart. n along the x axis, neighborhoods along the
y-axis. The lengths of the bars vary from near zero for Landmarks and
Green_Hills, to almost 450 for
North_Ames.](Tensorflow_files/figure-html/ames-xtab-1.png)

Fo plotting later, we calculate the simple means per neighborhood:

``` r
means <-
  ames |>
  group_by(Neighborhood) |>
  summarise(
    mean = mean(log10(Sale_Price)),
    n = length(Sale_Price),
    lon = median(Longitude),
    lat = median(Latitude)
  )
```

We’ll fit a model with 10 hidden units and 3 encoding columns:

``` r
library(embedmit)
tf_embed <-
  recipe(Sale_Price ~ ., data = ames) |>
  step_log(Sale_Price, base = 10) |>
  # Add some other predictors that can be used by the network
  # We preprocess them first
  step_YeoJohnson(Lot_Area, Full_Bath, Gr_Liv_Area) |>
  step_range(Lot_Area, Full_Bath, Gr_Liv_Area) |>
  step_embed(
    Neighborhood,
    outcome = vars(Sale_Price),
    predictors = vars(Lot_Area, Full_Bath, Gr_Liv_Area),
    num_terms = 5,
    hidden_units = 10,
    options = embed_control(epochs = 75, validation_split = 0.2)
  ) |>
  prep(training = ames)

theme_set(theme_bw() + theme(legend.position = "top"))

tf_embed$steps[[4]]$history |>
  filter(epochs > 1) |>
  ggplot(aes(x = epochs, y = loss, col = type)) +
  geom_line() +
  scale_y_log10()
```

![Line chart with 2 lines. epochs along the x-axis, loss along the
y-axis. The two lines are colored according to the type of loss, red for
normal loss and blue for validation loss. The lines have high values for
small epochs and lower values for higher epochs, with the validation
loss being lower at all
times.](Tensorflow_files/figure-html/ames-linear-1.png)

The embeddings are obtained using the `tidy` method:

``` r
hood_coef <-
  tidy(tf_embed, number = 4) |>
  dplyr::select(-terms, -id) |>
  dplyr::rename(Neighborhood = level) |>
  # Make names smaller
  rename_at(
    vars(contains("emb")),
    funs(gsub("Neighborhood_", "", ., fixed = TRUE))
  )
hood_coef
```

    ## # A tibble: 30 × 6
    ##     embed_1  embed_2   embed_3  embed_4 embed_5 Neighborhood      
    ##       <dbl>    <dbl>     <dbl>    <dbl>   <dbl> <chr>             
    ##  1 -0.00155  0.0381  -0.0394    0.0460  -0.0212 ..new             
    ##  2 -0.0423  -0.00143 -0.000230 -0.0792   0.0230 North_Ames        
    ##  3 -0.0108  -0.0797  -0.00768  -0.0644  -0.0481 College_Creek     
    ##  4 -0.0478   0.0517   0.0661   -0.0552   0.0694 Old_Town          
    ##  5  0.0352   0.00799 -0.0253   -0.0331   0.0493 Edwards           
    ##  6 -0.0244  -0.113   -0.0286   -0.00368 -0.0835 Somerset          
    ##  7 -0.0399  -0.0512   0.0168   -0.0194  -0.167  Northridge_Heights
    ##  8 -0.0197  -0.0315   0.0344    0.0176  -0.0210 Gilbert           
    ##  9 -0.0439  -0.0231   0.0184   -0.0693   0.0312 Sawyer            
    ## 10  0.0415   0.00161  0.0358   -0.00910 -0.0346 Northwest_Ames    
    ## # ℹ 20 more rows

``` r
hood_coef <-
  hood_coef |>
  inner_join(means, by = "Neighborhood")
hood_coef
```

    ## # A tibble: 28 × 10
    ##    embed_1  embed_2   embed_3  embed_4 embed_5 Neighborhood     mean     n   lon
    ##      <dbl>    <dbl>     <dbl>    <dbl>   <dbl> <chr>           <dbl> <int> <dbl>
    ##  1 -0.0423 -0.00143 -0.000230 -0.0792   0.0230 North_Ames       5.15   443 -93.6
    ##  2 -0.0108 -0.0797  -0.00768  -0.0644  -0.0481 College_Creek    5.29   267 -93.7
    ##  3 -0.0478  0.0517   0.0661   -0.0552   0.0694 Old_Town         5.07   239 -93.6
    ##  4  0.0352  0.00799 -0.0253   -0.0331   0.0493 Edwards          5.09   194 -93.7
    ##  5 -0.0244 -0.113   -0.0286   -0.00368 -0.0835 Somerset         5.35   182 -93.6
    ##  6 -0.0399 -0.0512   0.0168   -0.0194  -0.167  Northridge_Hei…  5.49   166 -93.7
    ##  7 -0.0197 -0.0315   0.0344    0.0176  -0.0210 Gilbert          5.27   165 -93.6
    ##  8 -0.0439 -0.0231   0.0184   -0.0693   0.0312 Sawyer           5.13   151 -93.7
    ##  9  0.0415  0.00161  0.0358   -0.00910 -0.0346 Northwest_Ames   5.27   131 -93.6
    ## 10 -0.0593 -0.0213   0.0149   -0.0239  -0.0114 Sawyer_West      5.25   125 -93.7
    ## # ℹ 18 more rows
    ## # ℹ 1 more variable: lat <dbl>

We can make a simple, interactive plot of the new features versus the
outcome:

``` r
tf_plot <-
  hood_coef |>
  dplyr::select(-lon, -lat) |>
  gather(variable, value, starts_with("embed")) |>
  # Clean up the embedding names
  # Add a new variable as a hover-over/tool tip
  mutate(
    label = paste0(gsub("_", " ", Neighborhood), " (n=", n, ")"),
    variable = gsub("_", " ", variable)
  ) |>
  ggplot(aes(x = value, y = mean)) +
  geom_point_interactive(aes(size = sqrt(n), tooltip = label), alpha = .5) +
  facet_wrap(~variable, scales = "free_x") +
  theme_bw() +
  theme(legend.position = "top") +
  labs(y = "Mean (log scale)", x = "Embedding")

girafe(ggobj = tf_plot)
```

However, this has induced some between-predictor correlations:

``` r
hood_coef |>
  dplyr::select(contains("emb")) |>
  cor() |>
  round(2)
```

    ##         embed_1 embed_2 embed_3 embed_4 embed_5
    ## embed_1    1.00    0.19   -0.23    0.04    0.33
    ## embed_2    0.19    1.00    0.21    0.02    0.61
    ## embed_3   -0.23    0.21    1.00    0.22    0.01
    ## embed_4    0.04    0.02    0.22    1.00   -0.11
    ## embed_5    0.33    0.61    0.01   -0.11    1.00
