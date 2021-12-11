import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def problem2(userIDs = [], fracSpent = [], fracComp = [], fracPaused = [], numPauses = [], avgPBR = [], numRWs = [], numFFs = [], s = []):
    #\\\\\\filter stuff///////#

    # create dictionary that holds number of times an ID appears and all other relevant parameters
    # [times, fracSpent, fracComp, fracPaused, numPauses, avgPBR, nuwRWs, numFFs, s] 
    vidsCompleted = {}
    for i in range(0,len(userIDs)):
        if userIDs[i] in vidsCompleted.keys():
            vidsCompleted[userIDs[i]][0] = vidsCompleted[userIDs[i]][0] + 1
        else:
            vidsCompleted[userIDs[i]] = [1, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(0,len(userIDs)):
        vidsCompleted[userIDs[i]][1] = vidsCompleted[userIDs[i]][1] + fracSpent[i]
        vidsCompleted[userIDs[i]][2] = vidsCompleted[userIDs[i]][2] + fracComp[i]
        vidsCompleted[userIDs[i]][3] = vidsCompleted[userIDs[i]][3] + fracPaused[i]
        vidsCompleted[userIDs[i]][4] = vidsCompleted[userIDs[i]][4] + numPauses[i]
        vidsCompleted[userIDs[i]][5] = vidsCompleted[userIDs[i]][5] + avgPBR[i]
        vidsCompleted[userIDs[i]][6] = vidsCompleted[userIDs[i]][6] + numRWs[i]
        vidsCompleted[userIDs[i]][7] = vidsCompleted[userIDs[i]][7] + numFFs[i]
        vidsCompleted[userIDs[i]][8] = vidsCompleted[userIDs[i]][8] + s[i]

    # remove entries that appear fewer than 47 times
    removeThese = []
    for id in vidsCompleted.keys():
        if(vidsCompleted[id][0] < 47):
            removeThese.append(id)
    
    for id in removeThese:
        del vidsCompleted[id]
    
    # take the average of the remaining entries
    for id in vidsCompleted.keys():
        vidsCompleted[id] = [x / vidsCompleted[id][0] for x in vidsCompleted[id]]
    
    fracSpentAvg = []
    fracCompAvg = []
    fracPausedAvg = []
    numPausesAvg = []
    avgPBRAvg = []
    numRWsAvg = []
    numFFsAvg = []
    sAvg = []
    for keys, values in vidsCompleted.items():
        fracSpentAvg.append(values[1])
        fracCompAvg.append(values[2])
        fracPausedAvg.append(values[3])
        numPausesAvg.append(values[4])
        avgPBRAvg.append(values[5])
        numRWsAvg.append(values[6])
        numFFsAvg.append(values[7])
        sAvg.append(values[8])

    #degrees = [1, 2, 3, 4, 5]
    degrees = [1]

    paramFits1 = getFits(sAvg, fracSpentAvg, degrees, "fracSpent")
    paramFits2 = getFits(sAvg, fracCompAvg, degrees, "fracComp")
    paramFits3 = getFits(sAvg, fracPausedAvg, degrees, "fracPaused")
    paramFits4 = getFits(sAvg, numPausesAvg, degrees, "numPauses")
    paramFits5 = getFits(sAvg, avgPBRAvg, degrees, "avgPBR")
    paramFits6 = getFits(sAvg, numRWsAvg, degrees, "numRWs")
    paramFits7 = getFits(sAvg, numFFsAvg, degrees, "numFFs")
    print("\nThe amount of time spent on the video (relative to video length)")
    printParams(paramFits1, degrees)
    print("\nThe fraction of the video watched")
    printParams(paramFits2, degrees)
    print("\nThe amount of time spent paused (relative to video length)")
    printParams(paramFits3, degrees)
    print("\nThe number of times the student pauses the video")
    printParams(paramFits4, degrees)
    print("\nThe average playback rate of the video")
    printParams(paramFits5, degrees)
    print("\nThe number of times the video was rewind-ed")
    printParams(paramFits6, degrees)
    print("\nThe number of times the video was fast-forwarded")
    printParams(paramFits7, degrees)


#Return fitted model parameters to the dataset at datapath for each choice in degrees.
#Input: datapath as a string specifying a .txt file, degrees as a list of positive integers.
#Output: paramFits, a list with the same length as degrees, where paramFits[i] is the list of
#coefficients when fitting a polynomial of d = degrees[i].
def getFits(scoreDataGiven, averageDataGiven, degrees, text):
    paramFits = []
    scoreData = []
    averageData = []

    #remove the outliers that we visually saw in the graphs
    index = 0
    for i in averageDataGiven:
        if(i > 200 or (text == "numRWs" and i > 10)):
            index = index + 1
        else:
            scoreData.append(scoreDataGiven[index])
            averageData.append(averageDataGiven[index])
            index = index + 1

    for n in degrees:
        features = feature_matrix(averageData, n)
        modelParams = least_squares(features, scoreData)
        paramFits.append(modelParams)
    #fill in
    #read the input file, assuming it has two columns, where each row is of the form [x y] as
    #in poly.txt.
    #iterate through each n in degrees, calling the feature_matrix and least_squares functions to solve
    #for the model parameters in each case. Append the result to paramFits each time.
    
    x_theory = np.linspace(min(averageData),max(averageData),1000)

    for k in range(0,len(degrees)):
        y_theory = [0] * len(x_theory)
        for i in range(0,len(y_theory)):
            for j in range(0,degrees[k]+1):
                y_theory[i] = y_theory[i] + x_theory[i] ** (degrees[k] - j) * paramFits[k][j]

        plt.plot(x_theory,y_theory)
        
    plt.scatter(averageData, scoreData, color="black")
    plt.xlabel("Average " + text)
    plt.ylabel("Average Score")
    plt.title(text + " Fitting")
    if len(degrees) == 5:
        plt.legend(["Degree 1 Fit", "Degree 2 Fit", "Degree 3 Fit", "Degree 4 Fit", "Degree 5 Fit", "Experimental Data"])
    elif len(degrees) == 1:
        plt.legend(["Degree 1 Fit", "Experimental Data"])
    
    plt.show()

    return paramFits


#Return the feature matrix for fitting a polynomial of degree d based on the explanatory variable
#samples in x.
#Input: x as a list of the independent variable samples, and d as an integer.
#Output: X, a list of features for each sample, where X[i][j] corresponds to the jth coefficient
#for the ith sample. Viewed as a matrix, X should have dimension #samples by d+1.
def feature_matrix(x, d):

    #fill in
    #There are several ways to write this function. The most efficient would be a nested list comprehension
    #which for each sample in x calculates x^d, x^(d-1), ..., x^0.
    degree_range = range(0,d+1)

    X = [[p**(d-i) for i in degree_range] for p in x]

    return X


#Return the least squares solution based on the feature matrix X and corresponding target variable samples in y.
#Input: X as a list of features for each sample, and y as a list of target variable samples.
#Output: B, a list of the fitted model parameters based on the least squares solution.
def least_squares(X, y):
    X = np.array(X)
    y = np.array(y)

    #fill in
    #Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.
    B = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),y)
    return B

#Function that calculates the mean squared error of the model on the input dataset.
#Input: Feature matrix X, target variable vector y, numpy model object
#Output: mse, the mean squared error
def rsquared(X, y, paramFits, degrees):

    predy = []
    if len(degrees) == 5:
        #print(paramFits[0][0], "X +", paramFits[0][1])
        #print(paramFits[1][0], "X^2 +", paramFits[1][1], "X +", paramFits[1][2])
        #print(paramFits[2][0], "X^3 +", paramFits[2][1], "X^2 +", paramFits[2][2], "X +", paramFits[2][3])
        #print(paramFits[3][0], "X^4 +", paramFits[3][1], "X^3 +", paramFits[3][2], "X^2 +", paramFits[3][3], "X +", paramFits[3][4])
        #print(paramFits[4][0], "X^5 +", paramFits[4][1], "X^4 +", paramFits[4][2], "X^3 +", paramFits[4][3], "X^2 +", paramFits[4][4], "X +", paramFits[4][5])
    elif len(degrees) == 1:
        predy.append(paramFits[0])
    
    predy = model.predict(X)
    mse = r2_score(y, predy)

    return rsquared

def printParams(paramFits, degrees):
    if len(degrees) == 5:
        print(paramFits[0][0], "X +", paramFits[0][1])
        print(paramFits[1][0], "X^2 +", paramFits[1][1], "X +", paramFits[1][2])
        print(paramFits[2][0], "X^3 +", paramFits[2][1], "X^2 +", paramFits[2][2], "X +", paramFits[2][3])
        print(paramFits[3][0], "X^4 +", paramFits[3][1], "X^3 +", paramFits[3][2], "X^2 +", paramFits[3][3], "X +", paramFits[3][4])
        print(paramFits[4][0], "X^5 +", paramFits[4][1], "X^4 +", paramFits[4][2], "X^3 +", paramFits[4][3], "X^2 +", paramFits[4][4], "X +", paramFits[4][5])
    elif len(degrees) == 1:
        print(paramFits[0][0], "X +", paramFits[0][1])