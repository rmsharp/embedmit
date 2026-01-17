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
    ##     embed_1  embed_2  embed_3  embed_4 embed_5 Neighborhood      
    ##       <dbl>    <dbl>    <dbl>    <dbl>   <dbl> <chr>             
    ##  1  0.0177   0.0165  -0.0191   0.0239   0.0203 ..new             
    ##  2 -0.0491  -0.0523  -0.00435 -0.0441   0.0688 North_Ames        
    ##  3 -0.0571  -0.0392  -0.0148  -0.0299  -0.0310 College_Creek     
    ##  4  0.0222  -0.0119   0.0740   0.0410   0.0755 Old_Town          
    ##  5 -0.0343   0.0489  -0.00678 -0.00713  0.0744 Edwards           
    ##  6 -0.0530  -0.0712   0.0128  -0.00662 -0.0897 Somerset          
    ##  7 -0.0883  -0.0651  -0.0567  -0.0275  -0.123  Northridge_Heights
    ##  8  0.0228  -0.0512  -0.0272   0.0337  -0.0128 Gilbert           
    ##  9  0.00406 -0.00394  0.00489 -0.0190   0.0189 Sawyer            
    ## 10  0.0587  -0.0281   0.0261  -0.0124  -0.0230 Northwest_Ames    
    ## # ℹ 20 more rows

``` r
hood_coef <-
  hood_coef |>
  inner_join(means, by = "Neighborhood")
hood_coef
```

    ## # A tibble: 28 × 10
    ##     embed_1  embed_2  embed_3  embed_4 embed_5 Neighborhood     mean     n   lon
    ##       <dbl>    <dbl>    <dbl>    <dbl>   <dbl> <chr>           <dbl> <int> <dbl>
    ##  1 -0.0491  -0.0523  -0.00435 -0.0441   0.0688 North_Ames       5.15   443 -93.6
    ##  2 -0.0571  -0.0392  -0.0148  -0.0299  -0.0310 College_Creek    5.29   267 -93.7
    ##  3  0.0222  -0.0119   0.0740   0.0410   0.0755 Old_Town         5.07   239 -93.6
    ##  4 -0.0343   0.0489  -0.00678 -0.00713  0.0744 Edwards          5.09   194 -93.7
    ##  5 -0.0530  -0.0712   0.0128  -0.00662 -0.0897 Somerset         5.35   182 -93.6
    ##  6 -0.0883  -0.0651  -0.0567  -0.0275  -0.123  Northridge_Hei…  5.49   166 -93.7
    ##  7  0.0228  -0.0512  -0.0272   0.0337  -0.0128 Gilbert          5.27   165 -93.6
    ##  8  0.00406 -0.00394  0.00489 -0.0190   0.0189 Sawyer           5.13   151 -93.7
    ##  9  0.0587  -0.0281   0.0261  -0.0124  -0.0230 Northwest_Ames   5.27   131 -93.6
    ## 10  0.00362 -0.0362   0.0415  -0.0239  -0.0108 Sawyer_West      5.25   125 -93.7
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
    ## embed_1    1.00    0.26    0.43    0.40    0.43
    ## embed_2    0.26    1.00    0.20    0.28    0.52
    ## embed_3    0.43    0.20    1.00    0.34    0.39
    ## embed_4    0.40    0.28    0.34    1.00    0.27
    ## embed_5    0.43    0.52    0.39    0.27    1.00
