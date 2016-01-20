---
title: "Practical Machine Learning Course Project"
author: "KDiggs86"
date: "January 19, 2016"
output: html_document
---
###Introduction
In a study conducted by Ugolino et. al. six participants were asked to perform barbell lifts in multiple ways (in the correct way and in four incorrect ways). Researchers then collected data from accelerometers located on the belt, forearm, and arm of the participants and on the dumbbell that the participants used. The goal of this study is to use the data from the accelerometers to predict whether or not the participants correctly performed the barbell lifts.

For this Coursera project we were give the data set of all observations collected during the barbell experiment. Our goal is to use the data set to predict the manner in which each participant did the exercise. In other words, we want to predict the "classe" variable. 

I started by downloading the training file to my computer and reading it into R. 


```r
weight <- read.csv("pml-training.csv")
```

###Choice of Variables
This data set has 19,622 observations and 160 variables. So, there is a lot going on in this data set, and I don't understand what many of the variables mean. However, there are really just four sensors (belt, dumbbell, arm and forearm), and then there are a lot of different measurements from each sensor. So, the 160 variables are just the "classe" variable and then various measurements from each of the four sensors. 

After perusing the data I noticed that all four sensors have roll, pitch, and yaw recordings. I googled and learned that these are some of the more important measurements taken by accelerometers. Moreover, these variables are very tidy and so should be easy to work with. Therefore, I decided to predict the "classe" variable using just these 12 variables. 


```r
rpy <- subset(weight, select = c("roll_belt", "pitch_belt", "yaw_belt", "roll_arm", "pitch_arm", "yaw_arm", "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", "roll_forearm", "pitch_forearm", "yaw_forearm", "classe"))
head(rpy)
```

```
##   roll_belt pitch_belt yaw_belt roll_arm pitch_arm yaw_arm roll_dumbbell
## 1      1.41       8.07    -94.4     -128      22.5    -161      13.05217
## 2      1.41       8.07    -94.4     -128      22.5    -161      13.13074
## 3      1.42       8.07    -94.4     -128      22.5    -161      12.85075
## 4      1.48       8.05    -94.4     -128      22.1    -161      13.43120
## 5      1.48       8.07    -94.4     -128      22.1    -161      13.37872
## 6      1.45       8.06    -94.4     -128      22.0    -161      13.38246
##   pitch_dumbbell yaw_dumbbell roll_forearm pitch_forearm yaw_forearm
## 1      -70.49400    -84.87394         28.4         -63.9        -153
## 2      -70.63751    -84.71065         28.3         -63.9        -153
## 3      -70.27812    -85.14078         28.3         -63.9        -152
## 4      -70.39379    -84.87363         28.1         -63.9        -152
## 5      -70.42856    -84.85306         28.0         -63.9        -152
## 6      -70.81759    -84.46500         27.9         -63.9        -152
##   classe
## 1      A
## 2      A
## 3      A
## 4      A
## 5      A
## 6      A
```

I then created training and test sets from the data.


```r
library(caret)
inTrain <- createDataPartition(y=rpy$classe, p = 0.7, list = FALSE)
training <- rpy[inTrain,]
testing <- rpy[-inTrain,]
```
 
###Exporatory Data Analysis

I wanted to determine if there are any clear patterns amongst the yaw, pitch, roll and classe variables. So, I created a few plots. I first created feature plots for the roll, pitch and yaw variables. I give the feature plots for the roll and pitch variables below. The color gives the classe. The plot for the yaw variables is similar.  Unfortunately, no clear pattern emerged. 
 

```r
 featurePlot(x=training[,c("roll_belt", "roll_dumbbell","roll_arm","roll_forearm")],y=training$classe, plot = "pairs")
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4-1.png) 

```r
 featurePlot(x=training[,c("pitch_belt", "pitch_dumbbell","pitch_arm","pitch_forearm")],y=training$classe, plot = "pairs")
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4-2.png) 


To continue my search for patterns I created some scatterplots that were colored by the "classe" variable. They were all similar, and so I only give two of these plots below.


```r
 qplot(roll_belt, roll_dumbbell, colour = classe, data = training)
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-1.png) 

```r
 qplot(roll_belt, yaw_belt, colour = classe, data = training)
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-2.png) 

While there seems to be some patterns in the scatterplots, I don't think we can use the patterns to distinguish the color (i.e. classe) of each observation. Therefore, the twelve predictors that I have chosen are weak.

###Boosting to the Rescue!
Since the predictors are weak, I used a boosting algorithm to create a prediction model. This strategy created a stronger predictor from the weak ones I have chosen.

I used the train function in the caret package with the "gbm" method to create a stochastic gradient boosted model.


```r
modFit <- train(classe~., method = "gbm", data = training, verbose = FALSE)
print(modFit)
```

```
## Stochastic Gradient Boosting 
## 
## 13737 samples
##    12 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD
##   1                   50      0.6803582  0.5941148  0.009400070
##   1                  100      0.7389434  0.6702500  0.006576118
##   1                  150      0.7635259  0.7014831  0.005026558
##   2                   50      0.7885358  0.7330691  0.005304793
##   2                  100      0.8496045  0.8101928  0.006533287
##   2                  150      0.8784491  0.8465538  0.005030602
##   3                   50      0.8483385  0.8085426  0.005623965
##   3                  100      0.8979112  0.8710473  0.004525685
##   3                  150      0.9224103  0.9019485  0.003770376
##   Kappa SD   
##   0.012098612
##   0.008230172
##   0.006295803
##   0.006709732
##   0.008230320
##   0.006338723
##   0.007042965
##   0.005706270
##   0.004753289
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
print(modFit$finalModel)
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 12 predictors of which 12 had non-zero influence.
```

###Error Discussion
I first used our model to predict the classe variable in the training data and looked at the corresponding confusion matrix. Both the accuracy and concordance (kappa) were greater than 90%. I am happy with that.


```r
 confusionMatrix(training$classe, predict(modFit, training))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3760   79   23   26   18
##          B   83 2401  134   35    5
##          C    4  117 2217   52    6
##          D    3   21   53 2169    6
##          E    5   32   35   58 2395
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9421         
##                  95% CI : (0.9381, 0.946)
##     No Information Rate : 0.2806         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9269         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9754   0.9060   0.9005   0.9269   0.9856
## Specificity            0.9852   0.9768   0.9841   0.9927   0.9885
## Pos Pred Value         0.9626   0.9033   0.9253   0.9631   0.9485
## Neg Pred Value         0.9903   0.9775   0.9784   0.9851   0.9969
## Prevalence             0.2806   0.1929   0.1792   0.1703   0.1769
## Detection Rate         0.2737   0.1748   0.1614   0.1579   0.1743
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9803   0.9414   0.9423   0.9598   0.9870
```
 
 Since this was encouraging I next applied the model to the testing set that I created. I used the model to predict the classe of each observation and then looked at the corresponding confusion matrix. The accuracy and the concordance remained greater than 90%. Fantastic!
 

```r
 confusionMatrix(testing$classe, predict(modFit, testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1604   38   21    6    5
##          B   45 1027   45   13    9
##          C    3   65  930   24    4
##          D    5    8   21  923    7
##          E    5   20   20   25 1012
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9339          
##                  95% CI : (0.9272, 0.9401)
##     No Information Rate : 0.2824          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9164          
##  Mcnemar's Test P-Value : 3.065e-06       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9651   0.8869   0.8968   0.9314   0.9759
## Specificity            0.9834   0.9763   0.9802   0.9916   0.9856
## Pos Pred Value         0.9582   0.9017   0.9064   0.9575   0.9353
## Neg Pred Value         0.9862   0.9724   0.9780   0.9862   0.9948
## Prevalence             0.2824   0.1968   0.1762   0.1684   0.1762
## Detection Rate         0.2726   0.1745   0.1580   0.1568   0.1720
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9743   0.9316   0.9385   0.9615   0.9807
```

Finally, this model correctly predicted 19 of the 20 test cases on the Project Quiz.

###Improvements

I am pleased with my results, but there are always ways to improve. I ignored 147 variables in building my model, so there is a lot left to explore. If I had more time to devote to this project I would try building multiple gradient boosted  models using different variables. I would then combine the classifiers using the ensembling methods discussed in the "Combining Predictors" lecture.
