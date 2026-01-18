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
    ##     embed_1 embed_2  embed_3   embed_4  embed_5 Neighborhood      
    ##       <dbl>   <dbl>    <dbl>     <dbl>    <dbl> <chr>             
    ##  1  0.0438  -0.0249 -0.0405  -0.0192   -0.00307 ..new             
    ##  2  0.0170  -0.0630  0.0792   0.0109    0.0185  North_Ames        
    ##  3  0.00930 -0.0430  0.0354  -0.0554   -0.0515  College_Creek     
    ##  4 -0.00731  0.0270  0.0527   0.0486    0.0693  Old_Town          
    ##  5 -0.0236   0.0130  0.00232  0.0124    0.0800  Edwards           
    ##  6 -0.0650  -0.0662  0.0400  -0.0627   -0.0742  Somerset          
    ##  7 -0.0420  -0.147   0.0215  -0.108    -0.0997  Northridge_Heights
    ##  8 -0.00765 -0.0138 -0.00134  0.0232   -0.0319  Gilbert           
    ##  9  0.0106  -0.0118  0.0687   0.000764  0.00184 Sawyer            
    ## 10  0.0360  -0.0207 -0.0104   0.00131  -0.0146  Northwest_Ames    
    ## # ℹ 20 more rows

``` r
hood_coef <-
  hood_coef |>
  inner_join(means, by = "Neighborhood")
hood_coef
```

    ## # A tibble: 28 × 10
    ##     embed_1 embed_2  embed_3   embed_4  embed_5 Neighborhood    mean     n   lon
    ##       <dbl>   <dbl>    <dbl>     <dbl>    <dbl> <chr>          <dbl> <int> <dbl>
    ##  1  0.0170  -0.0630  0.0792   0.0109    0.0185  North_Ames      5.15   443 -93.6
    ##  2  0.00930 -0.0430  0.0354  -0.0554   -0.0515  College_Creek   5.29   267 -93.7
    ##  3 -0.00731  0.0270  0.0527   0.0486    0.0693  Old_Town        5.07   239 -93.6
    ##  4 -0.0236   0.0130  0.00232  0.0124    0.0800  Edwards         5.09   194 -93.7
    ##  5 -0.0650  -0.0662  0.0400  -0.0627   -0.0742  Somerset        5.35   182 -93.6
    ##  6 -0.0420  -0.147   0.0215  -0.108    -0.0997  Northridge_He…  5.49   166 -93.7
    ##  7 -0.00765 -0.0138 -0.00134  0.0232   -0.0319  Gilbert         5.27   165 -93.6
    ##  8  0.0106  -0.0118  0.0687   0.000764  0.00184 Sawyer          5.13   151 -93.7
    ##  9  0.0360  -0.0207 -0.0104   0.00131  -0.0146  Northwest_Ames  5.27   131 -93.6
    ## 10  0.0177  -0.0424  0.0345  -0.00432  -0.0171  Sawyer_West     5.25   125 -93.7
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
    ## embed_1    1.00    0.47    0.30    0.59    0.37
    ## embed_2    0.47    1.00   -0.06    0.67    0.65
    ## embed_3    0.30   -0.06    1.00    0.20    0.25
    ## embed_4    0.59    0.67    0.20    1.00    0.61
    ## embed_5    0.37    0.65    0.25    0.61    1.00
