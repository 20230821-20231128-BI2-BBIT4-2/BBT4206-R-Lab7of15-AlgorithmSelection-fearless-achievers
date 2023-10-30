Business Intelligence Project
================
Team Champions
\| Date:30/10/2023

- [Student Details](#student-details)
- [Setup Chunk](#setup-chunk)
- [Understanding the Structure of
  Data](#understanding-the-structure-of-data)
  - [Loading the Dataset](#loading-the-dataset)
    - [Source:](#source)
    - [Reference:](#reference)
- [Clustering](#clustering)
- [Association](#association)

# Student Details

|                                              |                  |
|----------------------------------------------|------------------|
| **Student ID Number**                        | 126761           |
|                                              | 134111           |
|                                              | 133996           |
|                                              | 127707           |
|                                              | 135859           |
| **Student Name**                             | Virginia Wanjiru |
|                                              | Immaculate Haayo |
|                                              | Trevor Ngugi     |
|                                              | Clarice Muthoni  |
|                                              | Pauline Wairimu  |
| **BBIT 4.2 Group**                           | B                |
| **BI Project Group Name/ID (if applicable)** | Champions        |

# Setup Chunk

**Note:** the following KnitR options have been set as the global
defaults: <BR>
`knitr::opts_chunk$set(echo = TRUE, warning = FALSE, eval = TRUE, collapse = FALSE, tidy = TRUE)`.

More KnitR options are documented here
<https://bookdown.org/yihui/rmarkdown-cookbook/chunk-options.html> and
here <https://yihui.org/knitr/options/>.

# Understanding the Structure of Data

## Loading the Dataset

### Source:

The dataset that was used can be downloaded here:
<https://drive.google.com/drive/folders/1-BGEhfOwquXF6KKXwcvrx7WuZXuqmW9q?usp=sharing>

### Reference:

*  
Refer to the APA 7th edition manual for rules on how to cite datasets:
<https://apastyle.apa.org/style-grammar-guidelines/references/examples/data-set-references>*
\##### Linear Regression

##### Logistic Regression without caret

``` r
## 2. Logistic Regression ----
### 2.a. Logistic Regression without caret ----
# The glm() function is in the stats package and creates a
# generalized linear model for regression or classification.
# It can be configured to perform a logistic regression suitable for BINARY
# classification problems.

#### Load and split the dataset ----
library(readr)
chest_disease <- read_csv("../data/chest_disease.csv")
```

    ## Rows: 768 Columns: 9
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## dbl (9): pain_range, congestion, BloodPressure, abscess, Insulin, BMI, DFP, ...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(chest_disease$Outcome,
                                   p = 0.7,
                                   list = FALSE)
chest_disease_train <- chest_disease[train_index, ]
chest_disease_test <- chest_disease[-train_index, ]

#### Train the model ----

chest_disease_model_glm <- glm(Outcome ~ ., data = chest_disease_train,
                            family = binomial(link = "logit"))

#### Display the model's details ----
print(chest_disease_model_glm)
```

    ## 
    ## Call:  glm(formula = Outcome ~ ., family = binomial(link = "logit"), 
    ##     data = chest_disease_train)
    ## 
    ## Coefficients:
    ##   (Intercept)     pain_range     congestion  BloodPressure        abscess  
    ##    -8.7061753      0.1197299      0.0359745     -0.0140064      0.0050950  
    ##       Insulin            BMI            DFP            Age  
    ##    -0.0008998      0.0839628      1.4300515      0.0198078  
    ## 
    ## Degrees of Freedom: 537 Total (i.e. Null);  529 Residual
    ## Null Deviance:       701.1 
    ## Residual Deviance: 502.3     AIC: 520.3

``` r
#### Make predictions ----
probabilities <- predict(chest_disease_model_glm, chest_disease_test[, 1:8],
                         type = "response")
print(probabilities)
```

    ##           1           2           3           4           5           6 
    ## 0.049459546 0.036884036 0.125666433 0.862500152 0.938263668 0.288153049 
    ##           7           8           9          10          11          12 
    ## 0.707463554 0.564464483 0.251743040 0.211367802 0.643895597 0.113161144 
    ##          13          14          15          16          17          18 
    ## 0.615100163 0.977516709 0.041859955 0.039956394 0.037236617 0.866703309 
    ##          19          20          21          22          23          24 
    ## 0.782816048 0.019687012 0.901725024 0.171290052 0.507286361 0.030986055 
    ##          25          26          27          28          29          30 
    ## 0.005362408 0.613770865 0.537236830 0.175476317 0.065593209 0.266773071 
    ##          31          32          33          34          35          36 
    ## 0.133216607 0.053445237 0.028979799 0.583012053 0.318404457 0.116717498 
    ##          37          38          39          40          41          42 
    ## 0.046667045 0.274163977 0.193067343 0.205405988 0.645576051 0.734704517 
    ##          43          44          45          46          47          48 
    ## 0.061134835 0.221571778 0.008696385 0.626454312 0.899731092 0.597153980 
    ##          49          50          51          52          53          54 
    ## 0.879195047 0.978633497 0.210149286 0.098627563 0.428212901 0.334016135 
    ##          55          56          57          58          59          60 
    ## 0.128938988 0.208085372 0.753306196 0.324601974 0.507984242 0.062642628 
    ##          61          62          63          64          65          66 
    ## 0.677153536 0.970042164 0.416723515 0.662627340 0.086811342 0.844649379 
    ##          67          68          69          70          71          72 
    ## 0.313030324 0.344423472 0.087735151 0.855289354 0.119473939 0.153081513 
    ##          73          74          75          76          77          78 
    ## 0.030843022 0.071734675 0.102525702 0.170611912 0.363460750 0.703494006 
    ##          79          80          81          82          83          84 
    ## 0.041881322 0.445246458 0.607943871 0.744651423 0.538224673 0.468772590 
    ##          85          86          87          88          89          90 
    ## 0.488015232 0.288021342 0.403185837 0.229896768 0.368310431 0.638361747 
    ##          91          92          93          94          95          96 
    ## 0.154475347 0.168589615 0.939343221 0.743066650 0.089979300 0.394899119 
    ##          97          98          99         100         101         102 
    ## 0.185757246 0.012190509 0.068748094 0.197625614 0.490840366 0.080979918 
    ##         103         104         105         106         107         108 
    ## 0.033923488 0.192934790 0.146471674 0.143275698 0.839954306 0.108117188 
    ##         109         110         111         112         113         114 
    ## 0.708677660 0.567964795 0.027802838 0.824006986 0.102018593 0.445010846 
    ##         115         116         117         118         119         120 
    ## 0.523451811 0.227362943 0.160180128 0.673772640 0.186337200 0.392084255 
    ##         121         122         123         124         125         126 
    ## 0.180954602 0.011393158 0.324692130 0.097233642 0.110704404 0.348063523 
    ##         127         128         129         130         131         132 
    ## 0.746084569 0.232518760 0.023588405 0.117047603 0.669637242 0.018699976 
    ##         133         134         135         136         137         138 
    ## 0.238234953 0.171283224 0.853804520 0.180388567 0.069952952 0.850211194 
    ##         139         140         141         142         143         144 
    ## 0.955214708 0.918093420 0.718616385 0.075444215 0.108062602 0.340038865 
    ##         145         146         147         148         149         150 
    ## 0.179057525 0.682893548 0.360722225 0.219515774 0.506533574 0.043048730 
    ##         151         152         153         154         155         156 
    ## 0.026632398 0.119196909 0.340409897 0.292861520 0.542698389 0.319951482 
    ##         157         158         159         160         161         162 
    ## 0.065951348 0.975773642 0.635449437 0.095962663 0.099631872 0.152101290 
    ##         163         164         165         166         167         168 
    ## 0.383465322 0.165853831 0.190091018 0.179427040 0.102428462 0.369988725 
    ##         169         170         171         172         173         174 
    ## 0.414289734 0.957225498 0.545157036 0.034751976 0.855065850 0.203022312 
    ##         175         176         177         178         179         180 
    ## 0.713255721 0.101820316 0.129290022 0.852817767 0.343876101 0.493190783 
    ##         181         182         183         184         185         186 
    ## 0.334268692 0.072758142 0.013056198 0.263512853 0.191310632 0.145078903 
    ##         187         188         189         190         191         192 
    ## 0.091199306 0.406251028 0.140041372 0.094457660 0.149583564 0.791424030 
    ##         193         194         195         196         197         198 
    ## 0.492022890 0.313849424 0.097788488 0.602184242 0.562520936 0.977209395 
    ##         199         200         201         202         203         204 
    ## 0.841703686 0.079452445 0.103866889 0.808567614 0.201512392 0.302033404 
    ##         205         206         207         208         209         210 
    ## 0.312910253 0.106094242 0.685394503 0.849189020 0.497215693 0.546947779 
    ##         211         212         213         214         215         216 
    ## 0.338950431 0.641390819 0.275932815 0.418338451 0.836903735 0.238501565 
    ##         217         218         219         220         221         222 
    ## 0.195000315 0.271846310 0.122519423 0.158936519 0.087195434 0.843527811 
    ##         223         224         225         226         227         228 
    ## 0.563877889 0.270616685 0.691252643 0.506806983 0.300911745 0.938828140 
    ##         229         230 
    ## 0.353763582 0.068830114

``` r
predictions <- ifelse(probabilities > 0.5, "Yes", "No")
print(predictions)
```

    ##     1     2     3     4     5     6     7     8     9    10    11    12    13 
    ##  "No"  "No"  "No" "Yes" "Yes"  "No" "Yes" "Yes"  "No"  "No" "Yes"  "No" "Yes" 
    ##    14    15    16    17    18    19    20    21    22    23    24    25    26 
    ## "Yes"  "No"  "No"  "No" "Yes" "Yes"  "No" "Yes"  "No" "Yes"  "No"  "No" "Yes" 
    ##    27    28    29    30    31    32    33    34    35    36    37    38    39 
    ## "Yes"  "No"  "No"  "No"  "No"  "No"  "No" "Yes"  "No"  "No"  "No"  "No"  "No" 
    ##    40    41    42    43    44    45    46    47    48    49    50    51    52 
    ##  "No" "Yes" "Yes"  "No"  "No"  "No" "Yes" "Yes" "Yes" "Yes" "Yes"  "No"  "No" 
    ##    53    54    55    56    57    58    59    60    61    62    63    64    65 
    ##  "No"  "No"  "No"  "No" "Yes"  "No" "Yes"  "No" "Yes" "Yes"  "No" "Yes"  "No" 
    ##    66    67    68    69    70    71    72    73    74    75    76    77    78 
    ## "Yes"  "No"  "No"  "No" "Yes"  "No"  "No"  "No"  "No"  "No"  "No"  "No" "Yes" 
    ##    79    80    81    82    83    84    85    86    87    88    89    90    91 
    ##  "No"  "No" "Yes" "Yes" "Yes"  "No"  "No"  "No"  "No"  "No"  "No" "Yes"  "No" 
    ##    92    93    94    95    96    97    98    99   100   101   102   103   104 
    ##  "No" "Yes" "Yes"  "No"  "No"  "No"  "No"  "No"  "No"  "No"  "No"  "No"  "No" 
    ##   105   106   107   108   109   110   111   112   113   114   115   116   117 
    ##  "No"  "No" "Yes"  "No" "Yes" "Yes"  "No" "Yes"  "No"  "No" "Yes"  "No"  "No" 
    ##   118   119   120   121   122   123   124   125   126   127   128   129   130 
    ## "Yes"  "No"  "No"  "No"  "No"  "No"  "No"  "No"  "No" "Yes"  "No"  "No"  "No" 
    ##   131   132   133   134   135   136   137   138   139   140   141   142   143 
    ## "Yes"  "No"  "No"  "No" "Yes"  "No"  "No" "Yes" "Yes" "Yes" "Yes"  "No"  "No" 
    ##   144   145   146   147   148   149   150   151   152   153   154   155   156 
    ##  "No"  "No" "Yes"  "No"  "No" "Yes"  "No"  "No"  "No"  "No"  "No" "Yes"  "No" 
    ##   157   158   159   160   161   162   163   164   165   166   167   168   169 
    ##  "No" "Yes" "Yes"  "No"  "No"  "No"  "No"  "No"  "No"  "No"  "No"  "No"  "No" 
    ##   170   171   172   173   174   175   176   177   178   179   180   181   182 
    ## "Yes" "Yes"  "No" "Yes"  "No" "Yes"  "No"  "No" "Yes"  "No"  "No"  "No"  "No" 
    ##   183   184   185   186   187   188   189   190   191   192   193   194   195 
    ##  "No"  "No"  "No"  "No"  "No"  "No"  "No"  "No"  "No" "Yes"  "No"  "No"  "No" 
    ##   196   197   198   199   200   201   202   203   204   205   206   207   208 
    ## "Yes" "Yes" "Yes" "Yes"  "No"  "No" "Yes"  "No"  "No"  "No"  "No" "Yes" "Yes" 
    ##   209   210   211   212   213   214   215   216   217   218   219   220   221 
    ##  "No" "Yes"  "No" "Yes"  "No"  "No" "Yes"  "No"  "No"  "No"  "No"  "No"  "No" 
    ##   222   223   224   225   226   227   228   229   230 
    ## "Yes" "Yes"  "No" "Yes" "Yes"  "No" "Yes"  "No"  "No"

``` r
#### Display the model's evaluation metrics ----
table(predictions, chest_disease_test$Outcome)
```

    ##            
    ## predictions   0   1
    ##         No  127  29
    ##         Yes  27  47

``` r
# Read the following article on how to compute various evaluation metrics using
# the confusion matrix:
# https://en.wikipedia.org/wiki/Confusion_matrix
```

##### Logistic Regression with caret

``` r
library(readr)
chest_disease <- read_csv("../data/chest_disease.csv")
```

    ## Rows: 768 Columns: 9
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## dbl (9): pain_range, congestion, BloodPressure, abscess, Insulin, BMI, DFP, ...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
View(chest_disease)

# Define a 70:30 train:test data split of the dataset.
train_index <- createDataPartition(chest_disease$Outcome,
                                   p = 0.7,
                                   list = FALSE)
chest_disease_train <- chest_disease[train_index, ]
chest_disease_test <- chest_disease[-train_index, ]

#### Train the model ----
# We apply the 5-fold cross validation resampling method

train_control <- trainControl(method = "cv", number = 5)
# We can use "regLogistic" instead of "glm"
# Notice the data transformation applied when we call the train function
# in caret, i.e., a standardize data transform (centre + scale)
set.seed(7)

chest_disease_caret_model_logistic <-
  train(Outcome ~ ., data = chest_disease_train,
        method = "glm", metric = "RMSE",
        preProcess = c("center", "scale"), trControl = train_control)

#### Display the model's details ----
print(chest_disease_caret_model_logistic)
```

    ## Generalized Linear Model 
    ## 
    ## 538 samples
    ##   8 predictor
    ## 
    ## Pre-processing: centered (8), scaled (8) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 431, 430, 430, 431, 430 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared   MAE      
    ##   0.404534  0.2759685  0.3391235

``` r
#### Make predictions ----
predictions <- predict(chest_disease_caret_model_logistic,
                       chest_disease_test[, 1:8])
predictions<-as.factor(chest_disease_test$Outcome)

#### Display the model's evaluation metrics ----
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         as.factor(chest_disease_test[, 1:9]$Outcome))
print(confusion_matrix)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction   0   1
    ##          0 140   0
    ##          1   0  90
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9841, 1)
    ##     No Information Rate : 0.6087     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##                                      
    ##  Mcnemar's Test P-Value : NA         
    ##                                      
    ##             Sensitivity : 1.0000     
    ##             Specificity : 1.0000     
    ##          Pos Pred Value : 1.0000     
    ##          Neg Pred Value : 1.0000     
    ##              Prevalence : 0.6087     
    ##          Detection Rate : 0.6087     
    ##    Detection Prevalence : 0.6087     
    ##       Balanced Accuracy : 1.0000     
    ##                                      
    ##        'Positive' Class : 0          
    ## 

``` r
fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")
```

![](Lab_7_Submission_files/figure-gfm/Carry%20out%20Logistic%20Regression%20with%20caret-1.png)<!-- -->

# Clustering

``` r
# STEP 1. Install and Load the Required Packages ----
## readr ----
if (require("readr")) {
  require("readr")
} else {
  install.packages("readr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naniar ----
if (require("naniar")) {
  require("naniar")
} else {
  install.packages("naniar", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: naniar

``` r
## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## corrplot ----
if (require("corrplot")) {
  require("corrplot")
} else {
  install.packages("corrplot", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: corrplot

    ## corrplot 0.92 loaded

``` r
## ggcorrplot ----
if (require("ggcorrplot")) {
  require("ggcorrplot")
} else {
  install.packages("ggcorrplot", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: ggcorrplot

``` r
## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## dplyr ----
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: dplyr

    ## 
    ## Attaching package: 'dplyr'

    ## The following object is masked from 'package:MASS':
    ## 
    ##     select

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
# STEP 2. Load the Dataset ----
library(readr)
train <- read_csv("../data/train.csv")
```

    ## Rows: 8068 Columns: 11

    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (7): Gender, Ever_Married, Graduated, Profession, Spending_Score, Var_1,...
    ## dbl (4): ID, Age, Work_Experience, Family_Size
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
View(train)

train <-
  read_csv("../data/train.csv",
           col_types =
             cols(ID = col_double(),
                  Gender = col_character(),
                  Ever_Married = col_character(),
                  Age = col_double(),
                  Graduated = col_character(),
                  Profession = col_character(),
                  Work_Experience = col_double(),
                  Family_Size = col_double(),
                  Spending_Score = col_character(),
                  Var_1 = col_character(),
                  Segmentation = col_character()))


train$Profession <- factor(train$Profession)

str(train)
```

    ## spc_tbl_ [8,068 × 11] (S3: spec_tbl_df/tbl_df/tbl/data.frame)
    ##  $ ID             : num [1:8068] 462809 462643 466315 461735 462669 ...
    ##  $ Gender         : chr [1:8068] "Male" "Female" "Female" "Male" ...
    ##  $ Ever_Married   : chr [1:8068] "No" "Yes" "Yes" "Yes" ...
    ##  $ Age            : num [1:8068] 22 38 67 67 40 56 32 33 61 55 ...
    ##  $ Graduated      : chr [1:8068] "No" "Yes" "Yes" "Yes" ...
    ##  $ Profession     : Factor w/ 9 levels "Artist","Doctor",..: 6 3 3 8 4 1 6 6 3 1 ...
    ##  $ Work_Experience: num [1:8068] 1 NA 1 0 NA 0 1 1 0 1 ...
    ##  $ Spending_Score : chr [1:8068] "Low" "Average" "Low" "High" ...
    ##  $ Family_Size    : num [1:8068] 4 3 1 2 6 2 3 3 3 4 ...
    ##  $ Var_1          : chr [1:8068] "Cat_4" "Cat_4" "Cat_6" "Cat_6" ...
    ##  $ Segmentation   : chr [1:8068] "D" "A" "B" "B" ...
    ##  - attr(*, "spec")=
    ##   .. cols(
    ##   ..   ID = col_double(),
    ##   ..   Gender = col_character(),
    ##   ..   Ever_Married = col_character(),
    ##   ..   Age = col_double(),
    ##   ..   Graduated = col_character(),
    ##   ..   Profession = col_character(),
    ##   ..   Work_Experience = col_double(),
    ##   ..   Spending_Score = col_character(),
    ##   ..   Family_Size = col_double(),
    ##   ..   Var_1 = col_character(),
    ##   ..   Segmentation = col_character()
    ##   .. )
    ##  - attr(*, "problems")=<externalptr>

``` r
dim(train)
```

    ## [1] 8068   11

``` r
head(train)
```

    ## # A tibble: 6 × 11
    ##       ID Gender Ever_Married   Age Graduated Profession    Work_Experience
    ##    <dbl> <chr>  <chr>        <dbl> <chr>     <fct>                   <dbl>
    ## 1 462809 Male   No              22 No        Healthcare                  1
    ## 2 462643 Female Yes             38 Yes       Engineer                   NA
    ## 3 466315 Female Yes             67 Yes       Engineer                    1
    ## 4 461735 Male   Yes             67 Yes       Lawyer                      0
    ## 5 462669 Female Yes             40 Yes       Entertainment              NA
    ## 6 461319 Male   Yes             56 No        Artist                      0
    ## # ℹ 4 more variables: Spending_Score <chr>, Family_Size <dbl>, Var_1 <chr>,
    ## #   Segmentation <chr>

``` r
summary(train)
```

    ##        ID            Gender          Ever_Married            Age       
    ##  Min.   :458982   Length:8068        Length:8068        Min.   :18.00  
    ##  1st Qu.:461241   Class :character   Class :character   1st Qu.:30.00  
    ##  Median :463473   Mode  :character   Mode  :character   Median :40.00  
    ##  Mean   :463479                                         Mean   :43.47  
    ##  3rd Qu.:465744                                         3rd Qu.:53.00  
    ##  Max.   :467974                                         Max.   :89.00  
    ##                                                                        
    ##   Graduated                 Profession   Work_Experience  Spending_Score    
    ##  Length:8068        Artist       :2516   Min.   : 0.000   Length:8068       
    ##  Class :character   Healthcare   :1332   1st Qu.: 0.000   Class :character  
    ##  Mode  :character   Entertainment: 949   Median : 1.000   Mode  :character  
    ##                     Engineer     : 699   Mean   : 2.642                     
    ##                     Doctor       : 688   3rd Qu.: 4.000                     
    ##                     (Other)      :1760   Max.   :14.000                     
    ##                     NA's         : 124   NA's   :829                        
    ##   Family_Size      Var_1           Segmentation      
    ##  Min.   :1.00   Length:8068        Length:8068       
    ##  1st Qu.:2.00   Class :character   Class :character  
    ##  Median :3.00   Mode  :character   Mode  :character  
    ##  Mean   :2.85                                        
    ##  3rd Qu.:4.00                                        
    ##  Max.   :9.00                                        
    ##  NA's   :335

``` r
# STEP 3. Check for Missing Data and Address it ----
# Are there missing values in the dataset?
any_na(train)
```

    ## [1] TRUE

``` r
# How many?
n_miss(train)
```

    ## [1] 1582

``` r
# What is the proportion of missing data in the entire dataset?
prop_miss(train)
```

    ## [1] 0.01782575

``` r
# What is the number and percentage of missing values grouped by
# each variable?
miss_var_summary(train)
```

    ## # A tibble: 11 × 3
    ##    variable        n_miss pct_miss
    ##    <chr>            <int>    <dbl>
    ##  1 Work_Experience    829   10.3  
    ##  2 Family_Size        335    4.15 
    ##  3 Ever_Married       140    1.74 
    ##  4 Profession         124    1.54 
    ##  5 Graduated           78    0.967
    ##  6 Var_1               76    0.942
    ##  7 ID                   0    0    
    ##  8 Gender               0    0    
    ##  9 Age                  0    0    
    ## 10 Spending_Score       0    0    
    ## 11 Segmentation         0    0

``` r
# Which variables contain the most missing values?
gg_miss_var(train)
```

![](Lab_7_Submission_files/figure-gfm/Carry%20Out%20Clustering-1.png)<!-- -->

``` r
# Which combinations of variables are missing together?
gg_miss_upset(train)
```

![](Lab_7_Submission_files/figure-gfm/Carry%20Out%20Clustering-2.png)<!-- -->

``` r
# Where are missing values located (the shaded regions in the plot)?
vis_miss(train) +
  theme(axis.text.x = element_text(angle = 80))
```

![](Lab_7_Submission_files/figure-gfm/Carry%20Out%20Clustering-3.png)<!-- -->

``` r
## OPTION 1: Remove the observations with missing values ----
# We can decide to remove all the observations that have missing values
# as follows:
train_removed_obs <- train %>% filter(complete.cases(.))

train_removed_obs <-
  train %>%
  dplyr::filter(complete.cases(.))

# The initial dataset had 8068 observations and 11 variables
dim(train)
```

    ## [1] 8068   11

``` r
# The filtered dataset has 6665 observations and 11 variables
dim(train_removed_obs)
```

    ## [1] 6665   11

``` r
# Are there missing values in the dataset?
any_na(train_removed_obs)
```

    ## [1] FALSE

``` r
## OPTION 2: Remove the variables with missing values ----
# Alternatively, we can decide to remove the 2 variables that have missing data
train_removed_vars <-
  train %>%
  dplyr::select(-Work_Experience, -Family_Size)

# The initial dataset had 8068 observations and 11 variables
dim(train)
```

    ## [1] 8068   11

``` r
# The filtered dataset has 8068 observations and 11 variables
dim(train_removed_vars)
```

    ## [1] 8068    9

``` r
# Are there missing values in the dataset?
any_na(train_removed_vars)
```

    ## [1] TRUE

``` r
## OPTION 3: Perform Data Imputation ----

# CAUTION:
# 1. Avoid Over-imputation:
# Be cautious when imputing dates, especially if it is
# Missing Not at Random (MNAR).
# Over-Imputing can introduce bias into your analysis. For example, if dates
# are missing because of a specific event or condition, imputing dates might
# not accurately represent the data.

# 2. Consider the Business Context:
# Dates often have a significant business or domain context. Imputing dates
# may not always be appropriate, as it might distort the interpretation of
# your data. For example, imputing order dates could lead to incorrect insights
# into seasonality trends.


# STEP 4. Perform EDA and Feature Selection ----
## Compute the correlations between variables ----
# We identify the correlated variables because it is these correlated variables
# that can then be used to identify the clusters.

# Create a correlation matrix
# Option 1: Basic Table
cor(train_removed_obs[, c(1, 4, 7, 9)]) %>%
  View()

# Option 2: Basic Plot
cor(train_removed_obs[, c(1, 4, 7, 9)]) %>%
  corrplot(method = "square")
```

![](Lab_7_Submission_files/figure-gfm/Carry%20Out%20Clustering-4.png)<!-- -->

``` r
# Option 3: Fancy Plot using ggplot2
corr_matrix <- cor(train_removed_obs[, c(1, 4, 7, 9)])

p <- ggplot2::ggplot(data = reshape2::melt(corr_matrix),
                     ggplot2::aes(Var1, Var2, fill = value)) +
  ggplot2::geom_tile() +
  ggplot2::geom_text(ggplot2::aes(label = label_wrap(label, width = 10)),
                     size = 4) +
  ggplot2::theme_minimal() +
  ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))

ggcorrplot(corr_matrix, hc.order = TRUE, type = "lower", lab = TRUE)
```

![](Lab_7_Submission_files/figure-gfm/Carry%20Out%20Clustering-5.png)<!-- -->

``` r
## Plot the scatter plots ----
# A scatter plot to show a person's Work Experience against Variable
ggplot(train_removed_obs,
       aes(Work_Experience, Var_1,
           color = Age,
           shape = Spending_Score)) +
  geom_point(alpha = 0.5) +
  xlab("Work Experience") +
  ylab("Variable")
```

![](Lab_7_Submission_files/figure-gfm/Carry%20Out%20Clustering-6.png)<!-- -->

``` r
# A scatter plot to show Spending Score against Segmentation
ggplot(train_removed_obs,
       aes(Spending_Score, Segmentation,
           color = Graduated, shape = Segmentation)) +
  geom_point(alpha = 0.5) +
  xlab("Spending Score") +
  ylab("Segmentation")
```

![](Lab_7_Submission_files/figure-gfm/Carry%20Out%20Clustering-7.png)<!-- -->

``` r
## Transform the data ----
# The K Means Clustering algorithm performs better when data transformation has
# been applied. This helps to standardize the data making it easier to compare
# multiple variables.

summary(train_removed_obs)
```

    ##        ID            Gender          Ever_Married            Age       
    ##  Min.   :458982   Length:6665        Length:6665        Min.   :18.00  
    ##  1st Qu.:461349   Class :character   Class :character   1st Qu.:31.00  
    ##  Median :463575   Mode  :character   Mode  :character   Median :41.00  
    ##  Mean   :463520                                         Mean   :43.54  
    ##  3rd Qu.:465741                                         3rd Qu.:53.00  
    ##  Max.   :467974                                         Max.   :89.00  
    ##                                                                        
    ##   Graduated                 Profession   Work_Experience  Spending_Score    
    ##  Length:6665        Artist       :2192   Min.   : 0.000   Length:6665       
    ##  Class :character   Healthcare   :1077   1st Qu.: 0.000   Class :character  
    ##  Mode  :character   Entertainment: 809   Median : 1.000   Mode  :character  
    ##                     Doctor       : 592   Mean   : 2.629                     
    ##                     Engineer     : 582   3rd Qu.: 4.000                     
    ##                     Executive    : 505   Max.   :14.000                     
    ##                     (Other)      : 908                                      
    ##   Family_Size       Var_1           Segmentation      
    ##  Min.   :1.000   Length:6665        Length:6665       
    ##  1st Qu.:2.000   Class :character   Class :character  
    ##  Median :2.000   Mode  :character   Mode  :character  
    ##  Mean   :2.841                                        
    ##  3rd Qu.:4.000                                        
    ##  Max.   :9.000                                        
    ## 

``` r
model_of_the_transform <- preProcess(train_removed_obs, method = c("scale", "center"))
print(model_of_the_transform)
```

    ## Created from 6665 samples and 11 variables
    ## 
    ## Pre-processing:
    ##   - centered (4)
    ##   - ignored (7)
    ##   - scaled (4)

``` r
train_removed_obs_std <- predict(model_of_the_transform, train_removed_obs)
summary(train_removed_obs_std)  # Use 'train_removed_obs_std' here, not 'train_obs_std'
```

    ##        ID              Gender          Ever_Married            Age         
    ##  Min.   :-1.76815   Length:6665        Length:6665        Min.   :-1.5454  
    ##  1st Qu.:-0.84586   Class :character   Class :character   1st Qu.:-0.7587  
    ##  Median : 0.02149   Mode  :character   Mode  :character   Median :-0.1535  
    ##  Mean   : 0.00000                                         Mean   : 0.0000  
    ##  3rd Qu.: 0.86547                                         3rd Qu.: 0.5727  
    ##  Max.   : 1.73555                                         Max.   : 2.7514  
    ##                                                                            
    ##   Graduated                 Profession   Work_Experience   Spending_Score    
    ##  Length:6665        Artist       :2192   Min.   :-0.7720   Length:6665       
    ##  Class :character   Healthcare   :1077   1st Qu.:-0.7720   Class :character  
    ##  Mode  :character   Entertainment: 809   Median :-0.4784   Mode  :character  
    ##                     Doctor       : 592   Mean   : 0.0000                     
    ##                     Engineer     : 582   3rd Qu.: 0.4026                     
    ##                     Executive    : 505   Max.   : 3.3391                     
    ##                     (Other)      : 908                                       
    ##   Family_Size         Var_1           Segmentation      
    ##  Min.   :-1.2075   Length:6665        Length:6665       
    ##  1st Qu.:-0.5516   Class :character   Class :character  
    ##  Median :-0.5516   Mode  :character   Mode  :character  
    ##  Mean   : 0.0000                                        
    ##  3rd Qu.: 0.7601                                        
    ##  Max.   : 4.0393                                        
    ## 

``` r
sapply(train_removed_obs_std[, c(1, 4, 7, 9)], sd)
```

    ##              ID             Age Work_Experience     Family_Size 
    ##               1               1               1               1

``` r
## Select the features to use to create the clusters ----
# OPTION 1: Use all the numeric variables to create the clusters
train_vars <- train_removed_obs_std[, c(1, 4, 7, 9)]

train_vars <-
  train_removed_obs_std[, c("Age",
                            "Work_Experience")]

# STEP 5. Create the clusters using the K-Means Clustering Algorithm ----
# We start with a random guess of the number of clusters we need
set.seed(7)
kmeans_cluster <- kmeans(train_vars, centers = 3, nstart = 20)

# We then decide the maximum number of clusters to investigate
n_clusters <- 8

# Initialize total within sum of squares error: wss
wss <- numeric(n_clusters)

set.seed(7)

# Investigate 1 to n possible clusters (where n is the maximum number of 
# clusters that we want to investigate)
for (i in 1:n_clusters) {
  # Use the K Means cluster algorithm to create each cluster
  kmeans_cluster <- kmeans(train_vars, centers = i, nstart = 20)
  # Save the within cluster sum of squares
  wss[i] <- kmeans_cluster$tot.withinss
}

## Plot a scree plot ----
# The scree plot should help you to note when additional clusters do not make
# any significant difference (the plateau).
wss_df <- tibble(clusters = 1:n_clusters, wss = wss)

scree_plot <- ggplot(wss_df, aes(x = clusters, y = wss, group = 1)) +
  geom_point(size = 4) +
  geom_line() +
  scale_x_continuous(breaks = c(2, 4, 6, 8)) +
  xlab("Number of Clusters")


# OPTION 2: Use only the most significant variables to create the clusters
# This can be informed by feature selection, or by the business case.

scree_plot
```

![](Lab_7_Submission_files/figure-gfm/Carry%20Out%20Clustering-8.png)<!-- -->

``` r
# We can add guides to make it easier to identify the plateau (or "elbow").
scree_plot +
  geom_hline(
    yintercept = wss,
    linetype = "dashed",
    col = c(rep("#000000", 5), "#FF0000", rep("#000000", 2))
  )
```

![](Lab_7_Submission_files/figure-gfm/Carry%20Out%20Clustering-9.png)<!-- -->

``` r
# The plateau is reached at 8 clusters.
# We therefore create the final cluster with 8 clusters
# (not the initial 3 used at the beginning of this STEP.)
k <- 6
set.seed(7)
# Build model with k clusters: kmeans_cluster
kmeans_cluster <- kmeans(train_vars, centers = k, nstart = 20)

# STEP 6. Add the cluster number as a label for each observation ----
train_removed_obs$cluster_id <- factor(kmeans_cluster$cluster)

## View the results by plotting scatter plots with the labelled cluster ----
ggplot(train_removed_obs, aes(Work_Experience, Var_1,
                              color = cluster_id)) +
  geom_point(alpha = 0.5) +
  xlab("How long a person has worked") +
  ylab("Type of variable")
```

![](Lab_7_Submission_files/figure-gfm/Carry%20Out%20Clustering-10.png)<!-- -->

``` r
ggplot(train_removed_obs,
       aes(Spending_Score, Segmentation, color = cluster_id)) +
  geom_point(alpha = 0.5) +
  xlab("Total Spending Amount") +
  ylab("Characteristic of Each Segment")
```

![](Lab_7_Submission_files/figure-gfm/Carry%20Out%20Clustering-11.png)<!-- -->

``` r
# Note on Clustering for both Descriptive and Predictive Data Analytics ----
# Clustering can be used for both descriptive and predictive analytics.
# It is more commonly used around Exploratory Data Analysis which is
# descriptive analytics.

# The results of clustering, i.e., a label of the cluster can be fed as input
# to a supervised learning algorithm. The trained model can then be used to
# predict the cluster that a new observation will belong to.
```

# Association
