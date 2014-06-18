Machine Learning Method for HAR Prediction Based on Movement Data
========================================================
$$Author: Q.F.Fang$$   

## Synopsis
According to the project [background](https://class.coursera.org/predmachlearn-002/human_grading/view/courses/972090/assessments/4/submissions), devices such as Jawbone Up, Nike FuelBand, and Fitbit now allow us to collect a large amount of data about human activity recognition (HAR) in a more economical fashion. With such data, we are allowed to predict the quality of human's movement.

We utilized the [training and testing dataset](http://groupware.les.inf.puc-rio.br/har) from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. We built a Machine Learning model on it, in order to predict the manner in which they performed in the experiment (i.e. the "classe" variable in the training set). Generally, what we did include:   
***
1. Load the data and filter out irrelavant variables.
2. Use the **PCA Method** to select out variables making a somewhat significant contribution to the outcome variable, "classe".
3. Split the training data into two parts, one for training and the other for cross-validation.
4. Train the training set using **"K-fold Algorithm" + "Random Forest Algorithm"** and verify its feasibility by calculating the "out of sample error" on cross-validation data.
5. Pass the machine learning method on the testing set to make prediction.

## Step 1: Loading & Cleaning Raw Data
First things first, we load all the packages that might be of use in our project, set the working directory and read the training and testing .csv datasets into "tr" and "te" objects respectively. Note that all blank and NA observations as read as NA's.

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(lattice)
library(ggplot2)
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
setwd("D:\\Learning Materials\\Coursera_Practical Machine Learning\\Project")
tr <- read.csv(".\\pml-training.csv", na.strings = c("NA", ""))
te <- read.csv(".\\pml-testing.csv", na.strings = c("NA", ""))  ##all blank and NA observations as read as NA's
str(tr)
```

By checking the structure of "tr", we find out that, of all 160 variables, the first 7 ones are merely recording labels of each observation instead of describing the activities. Hence they need filtering out.

```r
tr <- tr[, -c(1:7)]
```

Besides, many variables consist of a considerable proportion of blank observation values. We then only kept variables without any NAs in the "tr" dataset. The consequent datasets are named "training".

```r
bad <- rep(NA, ncol(tr))  ## Create a null vector 'bad' to record variables with NAs.
for (i in 1:ncol(tr)) {
    if (any(is.na(tr[, i]))) 
        {
            bad[i] <- i
        }  ## As long as a variable contains NA, it is labelled as 'bad' variable.
}
bad <- bad[!is.na(bad)]
training <- tr[, -bad]  ## Throw away 'tr' columns labelled as 'bad' and assign the remaining data to 'training'
```


## Step 2: Selecting Variables of Significant Contribution
Now we check the dimension of the "training" dataset.

```r
dim(training)
```

```
## [1] 19622    53
```

It turns out we reduce the number of variables from 160 to 53, including the outcome "classe". We may regard the remaining 52 as potential regressors. However, we still believe there too many regressors. Some insignificant ones ought to be excluded from our model.   
Correlations among all variables are measures of variable significance. Thereby, we use the PCA Algorithm to filter out variables with correlation values below a certain threshold. Setting the threshold level is quite crucial: if the threshold is too low, insignificant variables might fail to be excluded; if the threshold is too high, there might be too few regressors to predict the outcome so that the prediction accuracy can be negatively affected. After multiple trials, we set the threshold level at 0.6.   

```r
M <- abs(cor(training[, -53]))  ## M is the correlation matrix of all 52 potential regressors.
diag(M) <- 0  ## Obviously, cor(XX,XX)=1. We ignore such self-correlation values by let the diagonal values be 0.
A <- which(M > 0.6, arr.ind = T)  ## Keep matrix entries above the threshold level, 0.6
sigVars <- unique(c(A[, 1], A[, 2]))  ## With unique() function, we get the 'sigVars' vector consisting of column numbers correspondant to signicant variables.
length(unique(c(A[, 1], A[, 2])))
```

```
## [1] 40
```

Till now, we reduce the number of potential regressors from 52 to 40 and we obtain the new training set, "training":

```r
training <- cbind(training[, sigVars], tr$classe)
colnames(training)[ncol(training)] <- "classe"
```


## Step 3: Spliting "training" Dataset for Cross-Validation
Within the "caret" R-package, we split the training data into two parts, one for training (i.e., "realTraining") and the other for cross-validation (i.e., "Xvalidation").

```r
set.seed(10011)  ## Set randomization seed for reproduciblility
inTrain <- createDataPartition(y = training$classe, p = 0.2, list = FALSE)  ## The training set accounts for 20% of the 'training' dataser.
realTraining <- training[inTrain, ]
Xvalidation <- tr[-inTrain, ]
dim(Xvalidation)
```

```
## [1] 15695   153
```

```r
dim(realTraining)
```

```
## [1] 3927   41
```


## Step 4: Training & Cross-Validation
Train the training set using **"K-fold Algorithm" + "Random Forest Algorithm"** and verify its feasibility by calculating the "out of sample error" on cross-validation data. Hereby, K=4. Specially, to expediate the training process, we fix some settings of the "trControl" call. We expect a somehow acceptable out of sample error less than 10%.

```r
set.seed(10086)
modFit <- train(classe ~ ., data = realTraining, method = "rf", trControl = trainControl(method = "cv", 
    number = 4))
modFit
```

```
## Random Forest 
## 
## 3927 samples
##   40 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 2945, 2945, 2946, 2945 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     0.9       0.9    0.007        0.009   
##   20    0.9       0.9    0.005        0.007   
##   40    0.9       0.9    0.005        0.006   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 21.
```

**Note that the "Accuracy" values here in the outcome are truncated**. In fact, however, it should be "Accuracy=0.95". Hence:    
$$in-sample-error=1- max(Accuracy)= \approx 1-0.95=5.0\%$$
The in sample error is quite low.   
Then, we  pass the "modFit" to "Xvalidation" for cross-validation:

```r
Xprediction <- predict(modFit, Xvalidation)
Xtable <- table(Xprediction, Xvalidation$classe)
Xtable
```

```
##            
## Xprediction    A    B    C    D    E
##           A 4418   63   10   46   12
##           B   14 2889   58    9   20
##           C    4   81 2651  114   29
##           D   23    4   18 2398   16
##           E    5    0    0    5 2808
```

See the table above, we declaim that:    
$$out-of-sample-error(of cross-validation)=1-\frac{sum(diag(Xtable))}{ncol(Xvalidation)}=1-\frac{4418+2889+2651+2398+2808}{15695} \approx 3.39\%$$   
The out of sample error is even lower. Therefore we have reasons to believe in the feasibility of the machine learning model.

## Step 5: Final Prediction
Finally, we pass the machine learning method, "modFit", to the testing set, "te", to make prediction.

```r
pred <- predict(modFit, te)
pred
```

```
##  [1] B A A A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

"pred" is the 20 prediction results. However, according to [the course project auto-grader submission system](https://class.coursera.org/predmachlearn-002/assignment), the third outcome of "pred", "A",  is incorrect actually.   
   
Hence,   
$$out-of-sample-error(of testing set) =\frac{19}{20}= 5\%$$
   
   
## Summary & Declaration
The machine learning prediction model in this work is rather reliable and convenient.   

The out of sample errors may result from insufficient number of regressors and K value in the K-fold Algorithm. Also, the out of sample errors vary among different seed we set.
