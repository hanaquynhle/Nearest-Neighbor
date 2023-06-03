
# -----------------------------------------------------------------------------------------
# GIVEN: For use in all testing for the purpose of grading

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd
from scipy.spatial import distance
import timeit
from sklearn import model_selection
from sklearn.metrics import make_scorer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

"""
@author: Hana Quynh Le
@date: Oct. 12, 2022 - Nov. 2,2022
    
"""

# GIVEN: For use starting in the "Reading in the data" step
def readData(numRows=None):
    inputCols = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids",
                 "Nonflavanoid Phenols", "Proanthocyanins", "Color Intensity", "Hue", "Diluted", "Proline"]
    outputCol = 'Class'
    colNames = [outputCol] + inputCols  # concatenate two lists into one
    wineDF = pd.read_csv("data/wine.data", header=None, names=colNames, nrows=numRows)

    # Need to mix this up before doing CV
    wineDF = wineDF.sample(frac=1, random_state=50).reset_index(drop=True)

    return wineDF, inputCols, outputCol
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# GIVEN: For use starting in the "Testing AlwaysOneClassifier" step
def accuracyOfActualVsPredicted(actualOutputSeries, predOutputSeries):
    
    compare = (actualOutputSeries == predOutputSeries).value_counts()
    # if there are no Trues in compare, then compare[True] throws an error. So we have to check:
    if (True in compare):
        accuracy = compare[True] / actualOutputSeries.size
    else:
        accuracy = 0

    return accuracy
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# GIVEN: For use starting in the "Standardization on a DataFrame" step
def operationsOnDataFrames():
    d = {'x': pd.Series([1, 2], index=['a', 'b']),
         'y': pd.Series([10, 11], index=['a', 'b']),
         'z': pd.Series([30, 25], index=['a', 'b'])}
    df = pd.DataFrame(d)
    print("Original df:", df, type(df), sep='\n', end='\n\n')

    cols = ['x', 'z']

    df.loc[:, cols] = df.loc[:, cols] / 2
    print("Certain columns / 2:", df, type(df), sep='\n', end='\n\n')

    maxResults = df.loc[:, cols].max()
    print("Max results:", maxResults, type(maxResults), sep='\n', end='\n\n')
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# GIVEN: For use starting in the "Standardization on a DataFrame" step
def testStandardize():
    df, inputCols, outputCol = readData()
    colsToStandardize = inputCols[2:5]
    print("Before standardization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')
    standardize(df, colsToStandardize)
    print("After standardization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')

    # Proof of standardization:
    print("Means are approx 0:", df.loc[:, colsToStandardize].mean(), sep='\n', end='\n\n')
    print("Stds are approx 1:", df.loc[:, colsToStandardize].std(), sep='\n', end='\n\n')
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
# GIVEN: For use starting in the "Normalization on a DataFrame" step
def testNormalize():
    df, inputCols, outputCol = readData()
    colsToStandardize = inputCols[2:5]
    print("Before normalization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')
    normalize(df, colsToStandardize)
    print("After normalization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')

    # Proof of normalization:
    print("Maxes are 1:", df.loc[:, colsToStandardize].max(), sep='\n', end='\n\n')
    print("Mins are 0:", df.loc[:, colsToStandardize].min(), sep='\n', end='\n\n')
# -----------------------------------------------------------------------------------------

class AlwaysOneClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self):
        pass
    
    def fit(self, inputDF, outputSeries):
        return self
    
    def predict(self, testInput):
        howMany = all
        if isinstance(testInput, pd.core.series.Series):
            return 1
        else :
            howMany = testInput.shape[0]
            return pd.Series(np.ones(howMany), index=testInput.index, dtype="int64")
    
def testAlwaysOneClassifier():
    df, inputCols, outputCol = readData()
    
    testInputDF = df.iloc[0:10,1:]
    testOutputSeries = df.iloc[0:10, 0]
    trainInputDF = df.iloc[10:,1:]
    trainOutputSeries = df.iloc[10:, 0]
    
    print("testInputDF:", testInputDF, sep='\n', end='\n\n') 
    print("testOutputSeries:", testOutputSeries, sep='\n', end='\n\n')
    print("trainInputDF:", trainInputDF, sep='\n', end='\n\n') 
    print("trainOutputSeries:", trainOutputSeries, sep='\n', end='\n\n')
    
    testClass = AlwaysOneClassifier()
    testFit = testClass.fit(trainInputDF, trainOutputSeries)
    
    correct1stRow = df.iloc[0,0]
    firstRowTestInput = df.iloc[0,1:]
    predict1stRow = testClass.predict(firstRowTestInput)
    print("---------- Test one example\nCorrect answer:", correct1stRow) 
    print("Predicted answer:", predict1stRow, end='\n\n')
    
    correctTestSet = df.iloc[0:10,0]
    print("---------- Test the entire test set\nCorrect answer:")
    print(correctTestSet, end='\n\n')
    print("Predicted answers:")
    predictTestSet = testClass.predict(testInputDF)
    print(predictTestSet, end='\n\n')    
    print("Accuracy:", accuracyOfActualVsPredicted(correctTestSet, predictTestSet), end='\n\n')
    
def findNearestLoop(df, testRow):
    minID = 0
    minDistance = distance.euclidean(df.iloc[0,:], testRow)
    for i in range(df.shape[0]):
        if distance.euclidean(df.iloc[i,:], testRow) < minDistance:
            minDistance = distance.euclidean(df.iloc[i,:], testRow)
            minID = i
    return df.index[minID]

def findNearestHOF(df, testRow):
    distances = df.apply(lambda row: distance.euclidean(row, testRow), axis=1 )
    return distances.idxmin()
            
def testFindNearest():
    df, inputCols, outputCol = readData()
    startTime = timeit.default_timer()
    for i in range(100):
        findNearestLoop(df.iloc[100:107, :], df.iloc[90, :])
    elapsedTime = timeit.default_timer() - startTime
    #print(findNearestLoop(df.iloc[100:107, :], df.iloc[90, :]))
    print("findNearestLoop timing:")
    print(elapsedTime,"seconds \n")
    
    startTime = timeit.default_timer()
    for i in range(100):
        findNearestHOF(df.iloc[100:107, :], df.iloc[90, :])
    elapsedTime = timeit.default_timer() - startTime
    #print(findNearestHOF(df.iloc[100:107, :], df.iloc[90, :]))
    print("findNearestHOF timing:")
    print(elapsedTime,"seconds \n")
        
    #Using higher-order function to find the distance is faster than using loop to do so


class OneNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.inputDF = None
        self.outputSeries = None   
        
    def fit(self, inputDF, outputSeries):
        self.inputDF = inputDF
        self.outputSeries = outputSeries
        return self
    
    def __predictOne(self, testInput):
        
            labelOfNearestRow = findNearestHOF(self.inputDF,testInput)
            return self.outputSeries.loc[labelOfNearestRow] 
    
    def predict(self, testInput):
        if isinstance(testInput, pd.core.series.Series):
            return self.__predictOne(testInput)
        else:
            result = testInput.apply(lambda row: self.__predictOne(row), axis = 1)
            return result

def testOneNNClassifier():
    df, inputCols, outputCol = readData()
    
    testInputDF = df.loc[0:9, inputCols]
    testOutputSeries = df.loc[0:10, outputCol]
    trainInputDF = df.loc[10:,inputCols]
    trainOutputSeries = df.loc[10:,outputCol]

    testClass = OneNNClassifier()
    testFit = testClass.fit(trainInputDF, trainOutputSeries)
    
    correct3rdRow = df.loc[2,outputCol]
    thirdRowTestInput = df.loc[2, inputCols]
    predict3rdRow = testClass.predict(thirdRowTestInput)
    print("---------- Test one example\nCorrect answer:", correct3rdRow) 
    print("Predicted answer:", predict3rdRow, end='\n\n')
    
    correctTestSet = df.loc[0:9, outputCol]
    print("---------- Test the entire test set\nCorrect answer:")
    print(correctTestSet, end='\n\n')
    print("Predicted answers:")
    predictTestSet = testClass.predict(testInputDF)
    print(predictTestSet, end='\n\n')    
    print("Accuracy:", accuracyOfActualVsPredicted(correctTestSet, predictTestSet), end='\n\n')

def cross_val_score_manual(model, inputDF, outputSeries, k, verbose):
    results = []
    numberOfElements = inputDF.shape[0]
    foldSize = numberOfElements / k
    for i in range (0,k):
        start = int(i*foldSize)
        upToNotIncluding = int((i+1)*foldSize)
        testInputDF = inputDF.iloc[start:upToNotIncluding, :]
        trainInputDF = pd.concat([inputDF.iloc[:start,:],inputDF.iloc[upToNotIncluding:,:]])
        testOutputSeries = outputSeries.iloc[start:upToNotIncluding]
        trainOutputSeries = pd.concat([outputSeries.iloc[:start], outputSeries.iloc[upToNotIncluding:]])
        if (verbose):
            print("================================") 
            print("Iteration:", i)
            print("Train input:\n", list(trainInputDF.index)) 
            print("Train output:\n", list(trainOutputSeries.index)) 
            print("Test input:\n", testInputDF.index)
            print("Test output:\n", testOutputSeries.index)
            print("================================")
        
        model = OneNNClassifier().fit(trainInputDF, trainOutputSeries)
        predicted = model.predict(testInputDF)
        results.append(accuracyOfActualVsPredicted(testOutputSeries, predicted))
        
    return results

def testCVManual(model, k):
    df, inputCols, outputCol = readData()
    
    inputDF = df.loc[:,inputCols]
    outputSeries = df.loc[:, outputCol]
    
    #model  = OneNNClassifier()
    #k = 5
    
    accuracies = cross_val_score_manual(model, inputDF, outputSeries, k, verbose = True)
    print("Accuracies:", accuracies) 
    print("Average:", np.mean(accuracies))


def testCVBuiltIn(model, k):
    df, inputCols, outputCol = readData()
    
    inputDF = df.loc[:,inputCols]
    outputSeries = df.loc[:, outputCol]
    
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
    accuracies = model_selection.cross_val_score(model, inputDF, outputSeries, cv=k, scoring=scorer)
    print("Accuracies:", accuracies) 
    print("Average:", np.mean(accuracies))
    
def compareFolds():
    df, inputCols, outputCol = readData()
    
    inputDF = df.loc[:,inputCols]
    outputSeries = df.loc[:, outputCol]
    model = OneNNClassifier()

    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
    accuracies3 = model_selection.cross_val_score(model, inputDF, outputSeries, cv=3, scoring=scorer)
    accuracies10 = model_selection.cross_val_score(model, inputDF, outputSeries, cv=10, scoring=scorer) 
    
    print("Mean accuracy for 3:",np.mean(accuracies3))
    print("Mean accuracy for 10:",np.mean(accuracies10))
    
def standardize(df, names):
    newDF = df.loc[:,names] = (df.loc[:,names] - df.loc[:,names].mean()) / df.loc[:,names].std()
    return newDF

def normalize(df, names):
    newDF = df.loc[:,names] = (df.loc[:,names]- df.loc[:,names].min()) / (df.loc[:,names].max() - df.loc[:,names].min())
    return newDF

def comparePreprocessing():
    df, inputCols, outputCol = readData()
    k=10
    model = OneNNClassifier()
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
    accuracies = model_selection.cross_val_score(model, df.loc[:,inputCols], df.loc[:,outputCol], cv=k, scoring=scorer)
    print("Mean accuracy of original set: ",np.mean(accuracies))
     
    dfCopy = df.copy()
    standardize(dfCopy, inputCols)
    accuraciesStd = model_selection.cross_val_score(model, dfCopy.loc[:, inputCols], df.loc[:, outputCol], cv=k, scoring=scorer)
    print("Mean accuracy of standardized set: ", np.mean(accuraciesStd))
    
    dfCopy = df.copy()
    normalize(dfCopy, inputCols)
    accuraciesNml = model_selection.cross_val_score(model, dfCopy.loc[:, inputCols], df.loc[:, outputCol], cv=k, scoring=scorer)
    print("Mean accuracy of normalized set: ", np.mean(accuraciesNml))    

#Step 22 Comments:
#a. The standardized and normalized datasets are more accurate than the original set.
#   The wine data set is not yet in range and organized, which causes the original dataset's accuracy to be lower 
#   than the normalized and standardized one.
#   Importance for range and magnitude

#b. z-transformed data is data transformed into z-scores, which is also known as the standardization process that 
#   usesd distribution mean and standard deviation
#   Substracting the mean from each observation and divide that number by the standard deviation.

#c. Leave-one-out is the technique of spliting a dataset into a training set and a testing set, using all but one observation 
#   as part of the training set,
#   then build the model using data in training set and use it to predict the response value of the one left-out observation, 
#   and do the calculation needed.
#   Repeat the process n times (n = # of observations from dataset)
# - In leave-one-out technique, the model is trained with n-1 observations (n: total # of observations in wine.names dataset)
# - In k-fold cross validation, we split the dataset into k subsets, test on them, and train the n-k obsevations.
#   Because the training set is smaller here, the accuracy is reduced compared to using the leave-one-out techniques.
#   (The more training we have, the better performance the test set would be)

def visualization():
    fullDF, inputCols, outputCol = readData()
    standardize(fullDF, inputCols)
    
    sns.displot(fullDF.loc[:, 'Malic Acid'])
    print(fullDF.loc[:, 'Malic Acid'].skew())
    #Malic Acid skew measure = 1.0396511925814444
    #Displot for Malic Acid indicates that it is positively skewed
    
    #Histogram and Distribution Estimate
    sns.displot(fullDF.loc[:, 'Alcohol'])
    print(fullDF.loc[:, 'Alcohol'].skew())
    #a) Alcohol skew measure = -0.051482331077132064
    #Displot for alcohol indicates that it is negatively skewed
    
    #Histogram and Distribution Estimate for Two Attributes
    sns.jointplot(x='Malic Acid', y='Alcohol', data=fullDF.loc[:, ['Malic Acid', 'Alcohol']], kind='kde')
    sns.jointplot(x='Ash', y='Magnesium', data=fullDF.loc[:, ['Ash', 'Magnesium']], kind='kde')
   
    #Comments 
    #b) Value for Ash: ~ -0.1
    #   Value for Magnesium: ~ -0.5 
    
    #Pair Plots
    sns.pairplot(fullDF, hue=outputCol)

    #Comments
    #c) If Proline has a positive value, class 1 is most likely
    
    #d) If we dropped most input columns from your dataset, keeping only Diluted and Proline, 
    #    We would expect the accuracy would be descent because each class is clustered with its own type as we see in the figure

    #e) Keeping only Nonflavanoid Phenols and Ash:
    #   We would expect the accuracy would drop a lot because different classes are clustered together as we see in the figure.
    
    
    plt.show()

def testSubsets():
    df, inputCols, outputCol = readData()
    standardize(df, inputCols)
    model = OneNNClassifier()
    inputDF1 = df.loc[:,['Diluted','Proline']]
    outputSeries = df.loc[:,outputCol]
    
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
    accuraciesDP = model_selection.cross_val_score(model, inputDF1, outputSeries, cv=3, scoring=scorer)
    print("Accuracies for Diluted and Proline attributes: ", np.mean(accuraciesDP))
    
    inputDF2 = df.loc[:,['Nonflavanoid Phenols', 'Ash']]
    scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
    accuraciesNA = model_selection.cross_val_score(model, inputDF2, outputSeries, cv=3, scoring=scorer)
    print("Accuracies for Nonflavanoid Phenols and Ash attributes: ", np.mean(accuraciesNA))
    
    #Commets
    #f) Yes the experimental results match our hypotheses in (d) and (e) based on the pair plot
    

class kNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k=1):
        self.k=k
        self.inputDF = None
        self.outputSeries = None

        
    def fit(self, inputDF, outputSeries):
        self.inputDF = inputDF
        self.outputSeries = outputSeries
        return self
    
    def __predOfKNearest(self, testInput):
        distances = self.inputDF.apply(lambda row: distance.euclidean(row, testInput), axis=1)
        nearest = distances.nsmallest(self.k).index
        return self.outputSeries.loc[nearest].mode().loc[0]
        
    def predict(self, testInput):
         if isinstance(testInput, pd.core.series.Series):
             return self.__predOfKNearest(testInput)
         else:
             result = testInput.apply(lambda row: self.__predOfKNearest(row), axis = 1)
             return result
            
def testKNN():
     df, inputCols, outputCol = readData(None)
     model1 = OneNNClassifier()
     model8 = kNNClassifier(8)
     
     scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
     accuracies1 = model_selection.cross_val_score(model1, df.loc[:,inputCols], df.loc[:,outputCol], cv=10, scoring=scorer)
     print("Unaltered dataset, 1NN, accuracy: ",np.mean(accuracies1))
     
     dfCopy = df.copy()
     standardize(dfCopy, inputCols)
     accuraciesStd1 = model_selection.cross_val_score(model1, dfCopy.loc[:, inputCols], df.loc[:, outputCol], cv=10, scoring=scorer)
     print("Standardized dataset, 1NN, accuracy: ",np.mean(accuraciesStd1))
     
     dfCopy = df.copy()
     standardize(dfCopy, inputCols)
     accuraciesStd8NN = model_selection.cross_val_score(model8, dfCopy.loc[:, inputCols], df.loc[:, outputCol], cv=10, scoring=scorer)
     print("Standardized dataset, 8NN, accuracy:", np.mean(accuraciesStd8NN))
     
def paramSearchPlot():
     neighborList = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 50, 60, 80])
     accuracies = neighborList.apply(calMeanAccuracy)
     print(accuracies)
     plt.plot(neighborList, accuracies)
     plt.xlabel('Neighbors')
     plt.ylabel('Accuracy')
     plt.show()
     print(neighborList.loc[accuracies.idxmax()])
     
def calMeanAccuracy(k):
     model = kNNClassifier(k)
     df, inputCols, outputCol = readData(None)
     standardize(df, inputCols)
     scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)
     standardizedAccu = model_selection.cross_val_score(model, df.loc[:,inputCols], df.loc[:,outputCol], cv=10, scoring=scorer)
     meanAccu = np.mean(standardizedAccu)
     return meanAccu

def paramSearchPlotBuiltIn():
     df, inputCols, outputCol = readData(None)
     standardize(df, inputCols)
     stdInputDF = df.loc[:,inputCols]
     outputSeries = df.loc[:,outputCol] 
     neighborList = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 50, 60, 80])
     
     alg = KNeighborsClassifier(n_neighbors = 8)
     cvScores = model_selection.cross_val_score(alg, stdInputDF, outputSeries, cv=10, scoring='accuracy')
     print("Standardized dataset, 8NN, accuracy:", np.mean(cvScores))
     
     accuracies = neighborList.apply(lambda row: model_selection.cross_val_score(kNNClassifier(row), stdInputDF, outputSeries, cv=10, scoring='accuracy').mean())
     plt.plot(neighborList, accuracies)
     plt.xlabel('Neighbors')
     plt.ylabel('Accuracy')
     plt.show()
     print(accuracies)
        

# -----------------------------------------------------------------------------------------
def testMain():
    '''
    This function runs all the tests we'll use for grading. Please don't change it!
    When certain parts need to be graded, uncomment those parts only.
    Please keep all the other parts commented out for grading.
    '''
    pass

    print("========== testAlwaysOneClassifier ==========")
    testAlwaysOneClassifier()

    print("========== testFindNearest() ==========")
    testFindNearest()

    print("========== testOneNNClassifier() ==========")
    testOneNNClassifier()

    print("========== testCVManual(OneNNClassifier(), 5) ==========")
    testCVManual(OneNNClassifier(), 5)

    print("========== testCVBuiltIn(OneNNClassifier(), 5) ==========")
    testCVBuiltIn(OneNNClassifier(), 5)

    print("========== compareFolds() ==========")
    compareFolds()

    print("========== testStandardize() ==========")
    testStandardize()

    print("========== testNormalize() ==========")
    testNormalize()

    print("========== comparePreprocessing() ==========")
    comparePreprocessing()

    print("========== visualization() ==========")
    #visualization()

    print("========== testKNN() ==========")
    testKNN()

    print("========== paramSearchPlot() ==========")
    paramSearchPlot()

    print("========== paramSearchPlotBuiltIn() ==========")
    paramSearchPlotBuiltIn()
# -----------------------------------------------------------------------------------------
