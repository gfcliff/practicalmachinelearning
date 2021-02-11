---
title: "Practical Machine Learning Course Project"
author: "Gabriel Filc"
date: "2/11/2021"
output: 
  html_document: 
    keep_md: yes
---

# PROJECT REPORT

For this project, we are given data from accelerometers on the belt, forearm, arm, and dumbell of 6 research study participants. Our training data consists of accelerometer data and a label identifying the quality of the activity the participant was doing. Our testing data consists of accelerometer data without the identifying label. Our goal is to predict the labels for the test set observations. The source of the data is: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

# How I built the model
In order to build a model I previously conducted an exploratory data analysis. Through the "skimmed" function I identified several variables with a high number of missing observations. Those variables with more than 65% of NAs were excluded. Additionally, seven variables that do not seem to have an effect on the result were removed (X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window and num_window).

Feature selection was made based on several plots of the remaining variables. These graphs were based on the training set in order not to affect cross-validation. Through them I tried to identify imbalance in outcomes/predictors. The density plots did not clearly identify any feature that could be removed, even though they suggested that variables related to the belt and to the dumbbell´s accelerometers were the most relevant. 


# How I used cross validation, 
Probably the simplest and most widely used method for estimating prediction error is cross-validation. This method directly estimates the expected extra-sample error Err = E[L(Y, fˆ(X))], the average generalization error when the method fˆ(X) is applied to an independent test sample from the joint distribution of X and Y . As mentioned earlier, we might hope that cross-validation estimates the conditional error, with the training set T held fixed. But as we will see in Section 7.12, cross-validation typically estimates well only the expected prediction error. # Because I want to be able to estimate the out-of-sample error, I randomly split the full training data (ptrain) into a smaller training set (ptrain1) and a validation set (ptrain2) as the The test data renamed: valid_in (validate data) will stay as is and will be used later to test the prodction algorithm on the 20 cases. It should be noted that Random Forest model does not need to perform cross-validation on a testing set because... However, I performed it anyway because the GML model required thid step.

# Expected out of sample error

I will report the Random Forest´s model out of sample error as it is the one that I chose to make the final prediction.
When the cross validation is done predicting on the testing set, the accuracy is 99,01%, thus the expected out of sample error is 0,99%. When calculated through the random forest model estimationinternal process, the OOB error estimate is 0.8%.


# Methodological choices justification 

As stated previously, I am not interested to explain, thus, random forests and boosting models were the best choice. Both models were fitted and its accuracy compared in order to select the best predictor.

The Confusion Matrixes of the Gradient Boosting and Random Forest models show a 96% accuracy and 99% accuracy respectively. Thus, I chose to use the Random Forest Model for the prediction on the validation data set.




## TRAINING; TESTING AND VALIDATION DATA-SETS CREATION


 
## EXPLORATORY DATA ANALYSIS: DESCRIPTIVE STATISTICS


```r
# irrelevant explicative factors are deleted
training <- training[,-c(1:7)]
validation.data.set <- validation.data.set[,-c(1:7)]
skimmed <- skim_to_wide(training)
```

```
## Warning: 'skim_to_wide' is deprecated.
## Use 'skim()' instead.
## See help("Deprecated")
```

```r
skimmed[, c(1:5, 9:11, 13, 15:16)]
```


Table: Data summary

|                         |      |
|:------------------------|:-----|
|Name                     |.data |
|Number of rows           |12757 |
|Number of columns        |153   |
|_______________________  |      |
|Column type frequency:   |      |
|character                |33    |
|factor                   |1     |
|numeric                  |119   |
|________________________ |      |
|Group variables          |None  |


**Variable type: character**

|skim_variable           | n_missing| complete_rate| min| whitespace|
|:-----------------------|---------:|-------------:|---:|----------:|
|kurtosis_roll_belt      |     12497|          0.02|   7|          0|
|kurtosis_picth_belt     |     12497|          0.02|   7|          0|
|kurtosis_yaw_belt       |     12497|          0.02|   7|          0|
|skewness_roll_belt      |     12497|          0.02|   7|          0|
|skewness_roll_belt.1    |     12497|          0.02|   7|          0|
|skewness_yaw_belt       |     12497|          0.02|   7|          0|
|max_yaw_belt            |     12497|          0.02|   3|          0|
|min_yaw_belt            |     12497|          0.02|   3|          0|
|amplitude_yaw_belt      |     12497|          0.02|   4|          0|
|kurtosis_roll_arm       |     12497|          0.02|   7|          0|
|kurtosis_picth_arm      |     12497|          0.02|   7|          0|
|kurtosis_yaw_arm        |     12497|          0.02|   7|          0|
|skewness_roll_arm       |     12497|          0.02|   7|          0|
|skewness_pitch_arm      |     12497|          0.02|   7|          0|
|skewness_yaw_arm        |     12497|          0.02|   7|          0|
|kurtosis_roll_dumbbell  |     12497|          0.02|   6|          0|
|kurtosis_picth_dumbbell |     12497|          0.02|   6|          0|
|kurtosis_yaw_dumbbell   |     12497|          0.02|   7|          0|
|skewness_roll_dumbbell  |     12497|          0.02|   6|          0|
|skewness_pitch_dumbbell |     12497|          0.02|   6|          0|
|skewness_yaw_dumbbell   |     12497|          0.02|   7|          0|
|max_yaw_dumbbell        |     12497|          0.02|   3|          0|
|min_yaw_dumbbell        |     12497|          0.02|   3|          0|
|amplitude_yaw_dumbbell  |     12497|          0.02|   4|          0|
|kurtosis_roll_forearm   |     12497|          0.02|   6|          0|
|kurtosis_picth_forearm  |     12497|          0.02|   6|          0|
|kurtosis_yaw_forearm    |     12497|          0.02|   7|          0|
|skewness_roll_forearm   |     12497|          0.02|   6|          0|
|skewness_pitch_forearm  |     12497|          0.02|   6|          0|
|skewness_yaw_forearm    |     12497|          0.02|   7|          0|
|max_yaw_forearm         |     12497|          0.02|   3|          0|
|min_yaw_forearm         |     12497|          0.02|   3|          0|
|amplitude_yaw_forearm   |     12497|          0.02|   4|          0|


**Variable type: factor**

|skim_variable | n_missing| complete_rate|ordered | n_unique|
|:-------------|---------:|-------------:|:-------|--------:|
|classe        |         0|             1|FALSE   |        5|


**Variable type: numeric**

|skim_variable            | n_missing| complete_rate|    mean|       p0|     p25|
|:------------------------|---------:|-------------:|-------:|--------:|-------:|
|roll_belt                |         0|          1.00|   64.13|   -28.90|    1.10|
|pitch_belt               |         0|          1.00|    0.31|   -55.80|    1.77|
|yaw_belt                 |         0|          1.00|  -11.52|  -180.00|  -88.30|
|total_accel_belt         |         0|          1.00|   11.28|     0.00|    3.00|
|max_roll_belt            |     12497|          0.02|   -5.53|   -94.30|  -88.10|
|max_picth_belt           |     12497|          0.02|   13.07|     3.00|    5.00|
|min_roll_belt            |     12497|          0.02|  -10.36|  -180.00|  -88.60|
|min_pitch_belt           |     12497|          0.02|   10.79|     0.00|    3.00|
|amplitude_roll_belt      |     12497|          0.02|    4.82|     0.00|    0.30|
|amplitude_pitch_belt     |     12497|          0.02|    2.28|     0.00|    1.00|
|var_total_accel_belt     |     12497|          0.02|    0.95|     0.00|    0.10|
|avg_roll_belt            |     12497|          0.02|   69.06|   -27.40|    1.20|
|stddev_roll_belt         |     12497|          0.02|    1.48|     0.00|    0.20|
|var_roll_belt            |     12497|          0.02|    8.38|     0.00|    0.00|
|avg_pitch_belt           |     12497|          0.02|    0.60|   -51.40|   -0.35|
|stddev_pitch_belt        |     12497|          0.02|    0.62|     0.00|    0.20|
|var_pitch_belt           |     12497|          0.02|    0.79|     0.00|    0.00|
|avg_yaw_belt             |     12497|          0.02|   -8.40|  -138.30|  -88.30|
|stddev_yaw_belt          |     12497|          0.02|    1.76|     0.00|    0.10|
|var_yaw_belt             |     12497|          0.02|  166.99|     0.00|    0.01|
|gyros_belt_x             |         0|          1.00|   -0.01|    -1.04|   -0.03|
|gyros_belt_y             |         0|          1.00|    0.04|    -0.64|    0.00|
|gyros_belt_z             |         0|          1.00|   -0.13|    -1.46|   -0.20|
|accel_belt_x             |         0|          1.00|   -5.65|   -82.00|  -21.00|
|accel_belt_y             |         0|          1.00|   30.07|   -69.00|    3.00|
|accel_belt_z             |         0|          1.00|  -72.16|  -269.00| -162.00|
|magnet_belt_x            |         0|          1.00|   55.50|   -52.00|    9.00|
|magnet_belt_y            |         0|          1.00|  593.66|   354.00|  581.00|
|magnet_belt_z            |         0|          1.00| -345.77|  -621.00| -375.00|
|roll_arm                 |         0|          1.00|   17.72|  -180.00|  -31.40|
|pitch_arm                |         0|          1.00|   -4.65|   -88.80|  -26.00|
|yaw_arm                  |         0|          1.00|    0.00|  -180.00|  -42.50|
|total_accel_arm          |         0|          1.00|   25.39|     1.00|   17.00|
|var_accel_arm            |     12497|          0.02|   52.74|     0.00|    6.89|
|avg_roll_arm             |     12497|          0.02|   14.95|  -166.67|  -38.57|
|stddev_roll_arm          |     12497|          0.02|   12.15|     0.00|    1.84|
|var_roll_arm             |     12497|          0.02|  512.38|     0.00|    3.39|
|avg_pitch_arm            |     12497|          0.02|   -5.18|   -81.77|  -24.70|
|stddev_pitch_arm         |     12497|          0.02|   10.76|     0.00|    3.42|
|var_pitch_arm            |     12497|          0.02|  204.54|     0.00|   11.73|
|avg_yaw_arm              |     12497|          0.02|    2.21|  -173.44|  -32.97|
|stddev_yaw_arm           |     12497|          0.02|   23.98|     0.00|    3.83|
|var_yaw_arm              |     12497|          0.02| 1241.53|     0.00|   14.68|
|gyros_arm_x              |         0|          1.00|    0.04|    -6.37|   -1.33|
|gyros_arm_y              |         0|          1.00|   -0.26|    -3.40|   -0.80|
|gyros_arm_z              |         0|          1.00|    0.27|    -2.33|   -0.07|
|accel_arm_x              |         0|          1.00|  -60.20|  -383.00| -239.00|
|accel_arm_y              |         0|          1.00|   33.18|  -318.00|  -54.00|
|accel_arm_z              |         0|          1.00|  -70.57|  -636.00| -143.00|
|magnet_arm_x             |         0|          1.00|  192.97|  -578.00| -298.00|
|magnet_arm_y             |         0|          1.00|  156.41|  -386.00|  -11.00|
|magnet_arm_z             |         0|          1.00|  308.47|  -597.00|  134.00|
|max_roll_arm             |     12497|          0.02|   11.35|   -73.10|   -2.65|
|max_picth_arm            |     12497|          0.02|   38.56|  -173.00|   -2.62|
|max_yaw_arm              |     12497|          0.02|   35.27|     5.00|   29.00|
|min_roll_arm             |     12497|          0.02|  -22.28|   -88.80|  -43.42|
|min_pitch_arm            |     12497|          0.02|  -36.62|  -180.00|  -75.67|
|min_yaw_arm              |     12497|          0.02|   14.76|     1.00|    8.00|
|amplitude_roll_arm       |     12497|          0.02|   33.64|     0.00|   12.04|
|amplitude_pitch_arm      |     12497|          0.02|   75.18|     0.00|   14.50|
|amplitude_yaw_arm        |     12497|          0.02|   20.51|     0.00|   10.00|
|roll_dumbbell            |         0|          1.00|   23.49|  -153.71|  -19.16|
|pitch_dumbbell           |         0|          1.00|  -10.74|  -149.59|  -40.73|
|yaw_dumbbell             |         0|          1.00|    1.68|  -150.87|  -77.57|
|max_roll_dumbbell        |     12497|          0.02|   18.22|   -70.10|  -25.05|
|max_picth_dumbbell       |     12497|          0.02|   38.05|  -108.00|  -65.15|
|min_roll_dumbbell        |     12497|          0.02|  -39.02|  -149.60|  -59.15|
|min_pitch_dumbbell       |     12497|          0.02|  -30.66|  -147.00|  -92.55|
|amplitude_roll_dumbbell  |     12497|          0.02|   57.24|     0.00|   16.74|
|amplitude_pitch_dumbbell |     12497|          0.02|   68.71|     0.00|   17.65|
|total_accel_dumbbell     |         0|          1.00|   13.71|     0.00|    4.00|
|var_accel_dumbbell       |     12497|          0.02|    5.30|     0.00|    0.41|
|avg_roll_dumbbell        |     12497|          0.02|   24.81|  -128.96|  -10.67|
|stddev_roll_dumbbell     |     12497|          0.02|   21.88|     0.00|    4.69|
|var_roll_dumbbell        |     12497|          0.02| 1115.63|     0.00|   22.00|
|avg_pitch_dumbbell       |     12497|          0.02|   -9.16|   -70.73|  -31.20|
|stddev_pitch_dumbbell    |     12497|          0.02|   13.37|     0.00|    3.89|
|var_pitch_dumbbell       |     12497|          0.02|  336.88|     0.00|   15.11|
|avg_yaw_dumbbell         |     12497|          0.02|    4.05|  -117.95|  -76.88|
|stddev_yaw_dumbbell      |     12497|          0.02|   17.28|     0.00|    4.20|
|var_yaw_dumbbell         |     12497|          0.02|  612.19|     0.00|   17.66|
|gyros_dumbbell_x         |         0|          1.00|    0.15|  -204.00|   -0.03|
|gyros_dumbbell_y         |         0|          1.00|    0.05|    -2.10|   -0.14|
|gyros_dumbbell_z         |         0|          1.00|   -0.12|    -2.38|   -0.31|
|accel_dumbbell_x         |         0|          1.00|  -28.46|  -419.00|  -50.00|
|accel_dumbbell_y         |         0|          1.00|   52.29|  -189.00|   -9.00|
|accel_dumbbell_z         |         0|          1.00|  -37.52|  -319.00| -141.00|
|magnet_dumbbell_x        |         0|          1.00| -327.77|  -639.00| -535.00|
|magnet_dumbbell_y        |         0|          1.00|  219.89| -3600.00|  231.00|
|magnet_dumbbell_z        |         0|          1.00|   47.26|  -262.00|  -45.00|
|roll_forearm             |         0|          1.00|   34.31|  -180.00|   -0.59|
|pitch_forearm            |         0|          1.00|   10.69|   -72.50|    0.00|
|yaw_forearm              |         0|          1.00|   19.58|  -180.00|  -67.60|
|max_roll_forearm         |     12497|          0.02|   24.81|   -66.60|    0.00|
|max_picth_forearm        |     12497|          0.02|   85.60|  -151.00|    0.00|
|min_roll_forearm         |     12497|          0.02|    0.48|   -72.50|   -5.90|
|min_pitch_forearm        |     12497|          0.02|  -60.31|  -180.00| -175.00|
|amplitude_roll_forearm   |     12497|          0.02|   24.32|     0.00|    0.98|
|amplitude_pitch_forearm  |     12497|          0.02|  145.91|     0.00|    1.00|
|total_accel_forearm      |         0|          1.00|   34.66|     0.00|   29.00|
|var_accel_forearm        |     12497|          0.02|   32.46|     0.00|    6.42|
|avg_roll_forearm         |     12497|          0.02|   30.63|  -177.23|   -3.05|
|stddev_roll_forearm      |     12497|          0.02|   46.76|     0.00|    0.33|
|var_roll_forearm         |     12497|          0.02| 6030.08|     0.00|    0.11|
|avg_pitch_forearm        |     12497|          0.02|   12.13|   -68.17|    0.00|
|stddev_pitch_forearm     |     12497|          0.02|    7.77|     0.00|    0.29|
|var_pitch_forearm        |     12497|          0.02|  129.51|     0.00|    0.09|
|avg_yaw_forearm          |     12497|          0.02|   18.67|  -155.06|  -19.34|
|stddev_yaw_forearm       |     12497|          0.02|   47.88|     0.00|    0.51|
|var_yaw_forearm          |     12497|          0.02| 5104.60|     0.00|    0.26|
|gyros_forearm_x          |         0|          1.00|    0.15|   -22.00|   -0.22|
|gyros_forearm_y          |         0|          1.00|    0.09|    -7.02|   -1.46|
|gyros_forearm_z          |         0|          1.00|    0.16|    -7.94|   -0.18|
|accel_forearm_x          |         0|          1.00|  -61.46|  -498.00| -179.00|
|accel_forearm_y          |         0|          1.00|  164.35|  -537.00|   56.00|
|accel_forearm_z          |         0|          1.00|  -54.11|  -410.00| -181.00|
|magnet_forearm_x         |         0|          1.00| -312.21| -1280.00| -615.00|
|magnet_forearm_y         |         0|          1.00|  379.52|  -896.00|   -2.00|
|magnet_forearm_z         |         0|          1.00|  392.08|  -973.00|  187.00|

## THOSE EXPLICATIVE VARIABLES WITH MORE THAN 65% NA ARE DELETED

```r
training <- training[colSums(is.na(training))/nrow(training) < .35]
validation.data.set <- validation.data.set[colSums(is.na(validation.data.set))/nrow(validation.data.set) < .35]
```



## EXPLORATORY DATA ANALYSIS: Visualizing the importance of explicative variables
In this case, for a variable to be important, I would expect the density curves to be significantly different for the 5 classes, both in terms of the height (kurtosis) and placement (skewness). However, it is not possible to eliminate any variable based on these plots (belt, arm, forearm and dumbbell).



```r
beltCols <- grep("belt$", names(training), value = TRUE)
featurePlot(x = training[, beltCols[1:4]], 
            y = training$classe,
            plot = "density", 
            ## Pass in options to xyplot() to 
            ## make it prettier
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(1, 4), 
            auto.key = list(columns = 5))
```

![](index_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

```r
#featurePlot(x=training[,c()], y= training$classe, plot="ellipse")
#featurePlot(y= training$classe, plot="ellipse")

# ARM
armCols <- grep("arm$", names(training), value = TRUE)
#transparentTheme(trans = .9)
featurePlot(x = training[, armCols[1:4]], 
            y = training$classe,
            plot = "density", 
            ## Pass in options to xyplot() to 
            ## make it prettier
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(1, 4), 
            auto.key = list(columns = 5))
```

![](index_files/figure-html/unnamed-chunk-3-2.png)<!-- -->

```r
# FOREARM
farmCols <- grep("forearm$", names(training), value = TRUE)
#transparentTheme(trans = .9)
featurePlot(x = training[, farmCols[1:4]], 
            y = training$classe,
            plot = "density", 
            ## Pass in options to xyplot() to 
            ## make it prettier
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(1, 4), 
            auto.key = list(columns = 5))
```

![](index_files/figure-html/unnamed-chunk-3-3.png)<!-- -->

```r
# DUMBELL
dbellCols <- grep("dumbbell$", names(training), value = TRUE)
#transparentTheme(trans = .9)
featurePlot(x = training[, dbellCols[1:4]], 
            y = training$classe,
            plot = "density", 
            ## Pass in options to xyplot() to 
            ## make it prettier
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(1, 4), 
            auto.key = list(columns = 5))
```

![](index_files/figure-html/unnamed-chunk-3-4.png)<!-- -->



## MODEL SELECTION
Random Forest Model Training and OOB error reporting


```r
t.Control <- trainControl(method="cv", number = 5, allowParallel = TRUE)


fit.model.rf <- train(classe ~ ., data = training, method = "rf", trControl = t.Control, verbose = TRUE, na.action=na.omit)
fit.model.rf
```

```
## Random Forest 
## 
## 12757 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10205, 10205, 10206, 10207, 10205 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9894962  0.9867122
##   27    0.9894963  0.9867128
##   52    0.9829902  0.9784804
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

```r
fit.model.rf$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, verbose = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.81%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3618    5    2    0    2 0.002481390
## B   18 2442    9    0    0 0.010935601
## C    0   15 2203    7    0 0.009887640
## D    0    2   30 2057    2 0.016260163
## E    0    1    3    7 2334 0.004690832
```
Gradient Boosting model training


```r
fit.model.gbm <- train(classe ~ ., data = training, method = "gbm", trControl = t.Control, verbose = TRUE)
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1282
##      2        1.5244             nan     0.1000    0.0869
##      3        1.4664             nan     0.1000    0.0700
##      4        1.4219             nan     0.1000    0.0565
##      5        1.3847             nan     0.1000    0.0435
##      6        1.3553             nan     0.1000    0.0452
##      7        1.3261             nan     0.1000    0.0410
##      8        1.3004             nan     0.1000    0.0333
##      9        1.2787             nan     0.1000    0.0353
##     10        1.2556             nan     0.1000    0.0334
##     20        1.0970             nan     0.1000    0.0190
##     40        0.9240             nan     0.1000    0.0085
##     60        0.8150             nan     0.1000    0.0072
##     80        0.7363             nan     0.1000    0.0051
##    100        0.6706             nan     0.1000    0.0045
##    120        0.6166             nan     0.1000    0.0026
##    140        0.5755             nan     0.1000    0.0016
##    150        0.5555             nan     0.1000    0.0028
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1899
##      2        1.4899             nan     0.1000    0.1268
##      3        1.4076             nan     0.1000    0.1112
##      4        1.3371             nan     0.1000    0.0827
##      5        1.2833             nan     0.1000    0.0728
##      6        1.2368             nan     0.1000    0.0745
##      7        1.1895             nan     0.1000    0.0590
##      8        1.1518             nan     0.1000    0.0522
##      9        1.1183             nan     0.1000    0.0436
##     10        1.0902             nan     0.1000    0.0433
##     20        0.8899             nan     0.1000    0.0201
##     40        0.6673             nan     0.1000    0.0093
##     60        0.5422             nan     0.1000    0.0061
##     80        0.4524             nan     0.1000    0.0055
##    100        0.3892             nan     0.1000    0.0042
##    120        0.3366             nan     0.1000    0.0030
##    140        0.2971             nan     0.1000    0.0020
##    150        0.2790             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2385
##      2        1.4587             nan     0.1000    0.1581
##      3        1.3583             nan     0.1000    0.1268
##      4        1.2775             nan     0.1000    0.1055
##      5        1.2097             nan     0.1000    0.0897
##      6        1.1531             nan     0.1000    0.0735
##      7        1.1062             nan     0.1000    0.0752
##      8        1.0584             nan     0.1000    0.0621
##      9        1.0198             nan     0.1000    0.0594
##     10        0.9828             nan     0.1000    0.0552
##     20        0.7509             nan     0.1000    0.0237
##     40        0.5224             nan     0.1000    0.0117
##     60        0.3976             nan     0.1000    0.0087
##     80        0.3153             nan     0.1000    0.0032
##    100        0.2600             nan     0.1000    0.0023
##    120        0.2179             nan     0.1000    0.0013
##    140        0.1849             nan     0.1000    0.0010
##    150        0.1703             nan     0.1000    0.0023
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1314
##      2        1.5225             nan     0.1000    0.0862
##      3        1.4645             nan     0.1000    0.0686
##      4        1.4191             nan     0.1000    0.0550
##      5        1.3832             nan     0.1000    0.0457
##      6        1.3533             nan     0.1000    0.0426
##      7        1.3263             nan     0.1000    0.0385
##      8        1.3015             nan     0.1000    0.0363
##      9        1.2786             nan     0.1000    0.0315
##     10        1.2571             nan     0.1000    0.0313
##     20        1.0998             nan     0.1000    0.0146
##     40        0.9271             nan     0.1000    0.0094
##     60        0.8196             nan     0.1000    0.0052
##     80        0.7400             nan     0.1000    0.0037
##    100        0.6776             nan     0.1000    0.0036
##    120        0.6250             nan     0.1000    0.0020
##    140        0.5803             nan     0.1000    0.0023
##    150        0.5616             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1823
##      2        1.4887             nan     0.1000    0.1338
##      3        1.4011             nan     0.1000    0.1062
##      4        1.3350             nan     0.1000    0.0891
##      5        1.2790             nan     0.1000    0.0689
##      6        1.2349             nan     0.1000    0.0691
##      7        1.1919             nan     0.1000    0.0567
##      8        1.1547             nan     0.1000    0.0480
##      9        1.1233             nan     0.1000    0.0525
##     10        1.0895             nan     0.1000    0.0393
##     20        0.8895             nan     0.1000    0.0176
##     40        0.6807             nan     0.1000    0.0196
##     60        0.5500             nan     0.1000    0.0103
##     80        0.4611             nan     0.1000    0.0055
##    100        0.4003             nan     0.1000    0.0035
##    120        0.3481             nan     0.1000    0.0040
##    140        0.3053             nan     0.1000    0.0020
##    150        0.2880             nan     0.1000    0.0021
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2357
##      2        1.4586             nan     0.1000    0.1645
##      3        1.3566             nan     0.1000    0.1218
##      4        1.2785             nan     0.1000    0.1160
##      5        1.2073             nan     0.1000    0.0930
##      6        1.1482             nan     0.1000    0.0710
##      7        1.1006             nan     0.1000    0.0642
##      8        1.0592             nan     0.1000    0.0651
##      9        1.0180             nan     0.1000    0.0477
##     10        0.9869             nan     0.1000    0.0552
##     20        0.7506             nan     0.1000    0.0247
##     40        0.5179             nan     0.1000    0.0106
##     60        0.4007             nan     0.1000    0.0054
##     80        0.3206             nan     0.1000    0.0048
##    100        0.2663             nan     0.1000    0.0026
##    120        0.2218             nan     0.1000    0.0027
##    140        0.1888             nan     0.1000    0.0013
##    150        0.1759             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1262
##      2        1.5259             nan     0.1000    0.0849
##      3        1.4679             nan     0.1000    0.0671
##      4        1.4240             nan     0.1000    0.0530
##      5        1.3889             nan     0.1000    0.0459
##      6        1.3587             nan     0.1000    0.0428
##      7        1.3310             nan     0.1000    0.0404
##      8        1.3052             nan     0.1000    0.0316
##      9        1.2842             nan     0.1000    0.0339
##     10        1.2621             nan     0.1000    0.0327
##     20        1.1064             nan     0.1000    0.0166
##     40        0.9325             nan     0.1000    0.0111
##     60        0.8222             nan     0.1000    0.0058
##     80        0.7437             nan     0.1000    0.0046
##    100        0.6828             nan     0.1000    0.0032
##    120        0.6292             nan     0.1000    0.0035
##    140        0.5867             nan     0.1000    0.0023
##    150        0.5688             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1875
##      2        1.4884             nan     0.1000    0.1256
##      3        1.4067             nan     0.1000    0.1015
##      4        1.3414             nan     0.1000    0.0862
##      5        1.2849             nan     0.1000    0.0657
##      6        1.2423             nan     0.1000    0.0552
##      7        1.2053             nan     0.1000    0.0620
##      8        1.1658             nan     0.1000    0.0514
##      9        1.1333             nan     0.1000    0.0475
##     10        1.1041             nan     0.1000    0.0384
##     20        0.9003             nan     0.1000    0.0235
##     40        0.6830             nan     0.1000    0.0108
##     60        0.5539             nan     0.1000    0.0058
##     80        0.4680             nan     0.1000    0.0055
##    100        0.4005             nan     0.1000    0.0029
##    120        0.3471             nan     0.1000    0.0024
##    140        0.3078             nan     0.1000    0.0016
##    150        0.2896             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2398
##      2        1.4587             nan     0.1000    0.1623
##      3        1.3568             nan     0.1000    0.1265
##      4        1.2784             nan     0.1000    0.1066
##      5        1.2109             nan     0.1000    0.0952
##      6        1.1530             nan     0.1000    0.0698
##      7        1.1070             nan     0.1000    0.0672
##      8        1.0654             nan     0.1000    0.0690
##      9        1.0227             nan     0.1000    0.0542
##     10        0.9872             nan     0.1000    0.0441
##     20        0.7590             nan     0.1000    0.0261
##     40        0.5288             nan     0.1000    0.0110
##     60        0.4009             nan     0.1000    0.0055
##     80        0.3191             nan     0.1000    0.0033
##    100        0.2644             nan     0.1000    0.0022
##    120        0.2181             nan     0.1000    0.0022
##    140        0.1867             nan     0.1000    0.0008
##    150        0.1731             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1291
##      2        1.5239             nan     0.1000    0.0919
##      3        1.4636             nan     0.1000    0.0654
##      4        1.4196             nan     0.1000    0.0543
##      5        1.3836             nan     0.1000    0.0506
##      6        1.3511             nan     0.1000    0.0403
##      7        1.3242             nan     0.1000    0.0401
##      8        1.2990             nan     0.1000    0.0328
##      9        1.2775             nan     0.1000    0.0347
##     10        1.2551             nan     0.1000    0.0262
##     20        1.0992             nan     0.1000    0.0194
##     40        0.9237             nan     0.1000    0.0097
##     60        0.8192             nan     0.1000    0.0071
##     80        0.7370             nan     0.1000    0.0035
##    100        0.6765             nan     0.1000    0.0024
##    120        0.6245             nan     0.1000    0.0027
##    140        0.5813             nan     0.1000    0.0023
##    150        0.5614             nan     0.1000    0.0025
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1864
##      2        1.4894             nan     0.1000    0.1297
##      3        1.4041             nan     0.1000    0.0996
##      4        1.3396             nan     0.1000    0.0864
##      5        1.2849             nan     0.1000    0.0735
##      6        1.2366             nan     0.1000    0.0587
##      7        1.1979             nan     0.1000    0.0669
##      8        1.1558             nan     0.1000    0.0513
##      9        1.1221             nan     0.1000    0.0452
##     10        1.0935             nan     0.1000    0.0457
##     20        0.8934             nan     0.1000    0.0229
##     40        0.6800             nan     0.1000    0.0094
##     60        0.5544             nan     0.1000    0.0081
##     80        0.4657             nan     0.1000    0.0050
##    100        0.4003             nan     0.1000    0.0025
##    120        0.3466             nan     0.1000    0.0038
##    140        0.3050             nan     0.1000    0.0024
##    150        0.2868             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2335
##      2        1.4626             nan     0.1000    0.1567
##      3        1.3610             nan     0.1000    0.1348
##      4        1.2768             nan     0.1000    0.0995
##      5        1.2144             nan     0.1000    0.0837
##      6        1.1605             nan     0.1000    0.0685
##      7        1.1161             nan     0.1000    0.0792
##      8        1.0660             nan     0.1000    0.0576
##      9        1.0303             nan     0.1000    0.0605
##     10        0.9924             nan     0.1000    0.0531
##     20        0.7530             nan     0.1000    0.0248
##     40        0.5256             nan     0.1000    0.0104
##     60        0.4053             nan     0.1000    0.0084
##     80        0.3233             nan     0.1000    0.0028
##    100        0.2629             nan     0.1000    0.0028
##    120        0.2219             nan     0.1000    0.0021
##    140        0.1876             nan     0.1000    0.0013
##    150        0.1741             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1227
##      2        1.5235             nan     0.1000    0.0842
##      3        1.4660             nan     0.1000    0.0673
##      4        1.4209             nan     0.1000    0.0535
##      5        1.3851             nan     0.1000    0.0488
##      6        1.3539             nan     0.1000    0.0441
##      7        1.3251             nan     0.1000    0.0415
##      8        1.2982             nan     0.1000    0.0307
##      9        1.2786             nan     0.1000    0.0317
##     10        1.2562             nan     0.1000    0.0305
##     20        1.1021             nan     0.1000    0.0182
##     40        0.9274             nan     0.1000    0.0105
##     60        0.8203             nan     0.1000    0.0054
##     80        0.7417             nan     0.1000    0.0046
##    100        0.6766             nan     0.1000    0.0040
##    120        0.6242             nan     0.1000    0.0027
##    140        0.5820             nan     0.1000    0.0018
##    150        0.5624             nan     0.1000    0.0027
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1898
##      2        1.4902             nan     0.1000    0.1309
##      3        1.4067             nan     0.1000    0.1075
##      4        1.3380             nan     0.1000    0.0836
##      5        1.2829             nan     0.1000    0.0732
##      6        1.2353             nan     0.1000    0.0592
##      7        1.1973             nan     0.1000    0.0580
##      8        1.1595             nan     0.1000    0.0546
##      9        1.1249             nan     0.1000    0.0437
##     10        1.0975             nan     0.1000    0.0365
##     20        0.8952             nan     0.1000    0.0184
##     40        0.6780             nan     0.1000    0.0097
##     60        0.5524             nan     0.1000    0.0055
##     80        0.4621             nan     0.1000    0.0067
##    100        0.3974             nan     0.1000    0.0037
##    120        0.3431             nan     0.1000    0.0027
##    140        0.3013             nan     0.1000    0.0016
##    150        0.2842             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2318
##      2        1.4609             nan     0.1000    0.1633
##      3        1.3577             nan     0.1000    0.1257
##      4        1.2788             nan     0.1000    0.1052
##      5        1.2134             nan     0.1000    0.0928
##      6        1.1551             nan     0.1000    0.0752
##      7        1.1065             nan     0.1000    0.0696
##      8        1.0618             nan     0.1000    0.0697
##      9        1.0176             nan     0.1000    0.0471
##     10        0.9856             nan     0.1000    0.0518
##     20        0.7530             nan     0.1000    0.0282
##     40        0.5271             nan     0.1000    0.0120
##     60        0.3997             nan     0.1000    0.0080
##     80        0.3186             nan     0.1000    0.0031
##    100        0.2605             nan     0.1000    0.0030
##    120        0.2184             nan     0.1000    0.0014
##    140        0.1851             nan     0.1000    0.0014
##    150        0.1703             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2361
##      2        1.4610             nan     0.1000    0.1631
##      3        1.3577             nan     0.1000    0.1273
##      4        1.2766             nan     0.1000    0.1030
##      5        1.2123             nan     0.1000    0.0856
##      6        1.1572             nan     0.1000    0.0726
##      7        1.1102             nan     0.1000    0.0703
##      8        1.0657             nan     0.1000    0.0623
##      9        1.0265             nan     0.1000    0.0644
##     10        0.9854             nan     0.1000    0.0576
##     20        0.7508             nan     0.1000    0.0212
##     40        0.5274             nan     0.1000    0.0102
##     60        0.4007             nan     0.1000    0.0064
##     80        0.3227             nan     0.1000    0.0049
##    100        0.2661             nan     0.1000    0.0028
##    120        0.2217             nan     0.1000    0.0030
##    140        0.1886             nan     0.1000    0.0016
##    150        0.1754             nan     0.1000    0.0013
```

```r
fit.model.gbm
```

```
## Stochastic Gradient Boosting 
## 
## 12757 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10206, 10206, 10205, 10205, 10206 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7568396  0.6917478
##   1                  100      0.8168842  0.7681930
##   1                  150      0.8515313  0.8120949
##   2                   50      0.8501991  0.8101859
##   2                  100      0.9027979  0.8769998
##   2                  150      0.9289798  0.9101420
##   3                   50      0.8953506  0.8675631
##   3                  100      0.9430110  0.9278884
##   3                  150      0.9598648  0.9492253
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150, interaction.depth =
##  3, shrinkage = 0.1 and n.minobsinnode = 10.
```

Confusion Matrixes are built after predicting the model over the testing data set. Accuracy estimates for each model will be used to decide which model predicts better.

## PREDICT

```r
predict.rf <- predict(fit.model.rf, newdata = testing)
confusionMatrix(predict.rf, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1947    6    0    0    0
##          B    5 1308    9    1    1
##          C    0   11 1181   20    1
##          D    0    2    7 1103    2
##          E    1    1    0    1 1258
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9901          
##                  95% CI : (0.9875, 0.9923)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9875          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9969   0.9849   0.9866   0.9804   0.9968
## Specificity            0.9988   0.9971   0.9944   0.9981   0.9995
## Pos Pred Value         0.9969   0.9879   0.9736   0.9901   0.9976
## Neg Pred Value         0.9988   0.9964   0.9972   0.9962   0.9993
## Prevalence             0.2845   0.1934   0.1744   0.1639   0.1838
## Detection Rate         0.2836   0.1905   0.1720   0.1607   0.1832
## Detection Prevalence   0.2845   0.1929   0.1767   0.1623   0.1837
## Balanced Accuracy      0.9979   0.9910   0.9905   0.9893   0.9981
```

```r
predict.gbm <- predict(fit.model.gbm, newdata = testing)
confusionMatrix(predict.gbm, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1915   36    0    0    2
##          B   27 1245   42    4   12
##          C    2   44 1146   43   16
##          D    6    1    9 1064   19
##          E    3    2    0   14 1213
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9589         
##                  95% CI : (0.954, 0.9635)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.948          
##                                          
##  Mcnemar's Test P-Value : 1.091e-08      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9805   0.9375   0.9574   0.9458   0.9612
## Specificity            0.9923   0.9846   0.9815   0.9939   0.9966
## Pos Pred Value         0.9805   0.9361   0.9161   0.9682   0.9846
## Neg Pred Value         0.9923   0.9850   0.9909   0.9894   0.9913
## Prevalence             0.2845   0.1934   0.1744   0.1639   0.1838
## Detection Rate         0.2790   0.1814   0.1669   0.1550   0.1767
## Detection Prevalence   0.2845   0.1937   0.1822   0.1601   0.1795
## Balanced Accuracy      0.9864   0.9611   0.9694   0.9698   0.9789
```

Further model comparison is performed throug the "resamples" function. This method confirms that the Random Forest model is a better predictor.


```r
# Compare model performances using resample()
models_compare <- resamples(list(RF=fit.model.rf, GradientBoosting=fit.model.gbm))

# Summary of the models performances
summary(models_compare)
```

```
## 
## Call:
## summary.resamples(object = models_compare)
## 
## Models: RF, GradientBoosting 
## Number of resamples: 5 
## 
## Accuracy 
##                       Min.   1st Qu.    Median      Mean   3rd Qu.      Max.
## RF               0.9870690 0.9894201 0.9894201 0.9894963 0.9898079 0.9917647
## GradientBoosting 0.9482556 0.9604077 0.9612069 0.9598648 0.9635580 0.9658957
##                  NA's
## RF                  0
## GradientBoosting    0
## 
## Kappa 
##                       Min.   1st Qu.    Median      Mean   3rd Qu.      Max.
## RF               0.9836408 0.9866143 0.9866162 0.9867128 0.9871089 0.9895835
## GradientBoosting 0.9345695 0.9499232 0.9509188 0.9492253 0.9538767 0.9568383
##                  NA's
## RF                  0
## GradientBoosting    0
```



Finally, we apply the random forest to the validation data.


```r
Results <- predict(fit.model.rf, newdata=validation.data.set)
Results
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

