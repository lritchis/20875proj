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

    # remove entries that appear fewer than 5 times
    removeThese = []
    for id in vidsCompleted.keys():
        if(vidsCompleted[id][0] < 47):
            removeThese.append(id)
    
    for id in removeThese:
        del vidsCompleted[id]
    
    # take the average of the remaining entries
    for id in vidsCompleted.keys():
        vidsCompleted[id] = [x / vidsCompleted[id][0] for x in vidsCompleted[id]]

    ## TESTING ## 
    # print(list (vidsCompleted.keys())[0])

    # testSpent = 0
    # testComp = 0
    # testPaused = 0
    # testNumPause = 0
    # testPBR = 0
    # testRW = 0
    # testFF = 0
    # testS = 0
    # j = 0
    # for i in range(0,len(userIDs)):
    #     if(userIDs[i] == "210f854b0afc3d476d711b2b41379954e48cfa44"):
    #         j = j + 1
    #         testSpent = testSpent + fracSpent[i]
    #         testComp = testComp + fracComp[i]
    #         testPaused = testPaused + fracPaused[i]
    #         testNumPause = testNumPause + numPauses[i]
    #         testPBR = testPBR + avgPBR[i]
    #         testRW = testRW + numRWs[i]
    #         testFF = testFF + numFFs[i]
    #         testS = testS + s[i]

    # tester = [x / j for x in [j, testSpent,testComp,testPaused,testNumPause,testPBR,testRW,testFF,testS]]
    # print(tester)
    # print(vidsCompleted["210f854b0afc3d476d711b2b41379954e48cfa44"])
    #^^^^^filter stuff ^^^^^#
    # start Julian stuff#
    degrees = [1, 2, 3, 4, 5]

    paramFits1 = getFits(s, fracSpent, degrees)
    paramFits2 = getFits(s, fracComp, degrees)
    paramFits3 = getFits(s, fracPaused, degrees)
    paramFits4 = getFits(s, numPauses, degrees)
    paramFits5 = getFits(s, avgPBR, degrees)
    paramFits6 = getFits(s, numRWs, degrees)
    paramFits7 = getFits(s, numFFs, degrees)
    print("The amount of time spent on the video (relative to video length)")
    print(paramFits1)
    print("The fraction of the video watched")
    print(paramFits2)
    print("The amount of time spent paused (relative to video length)")
    print(paramFits3)
    print("The number of times the student pauses the video")
    print(paramFits4)
    print("The average playback rate of the video")
    print(paramFits5)
    print("The number of times the video was rewind-ed")
    print(paramFits6)
    print("The number of times the video was fast-forwarded")
    print(paramFits7)


#Return fitted model parameters to the dataset at datapath for each choice in degrees.
#Input: datapath as a string specifying a .txt file, degrees as a list of positive integers.
#Output: paramFits, a list with the same length as degrees, where paramFits[i] is the list of
#coefficients when fitting a polynomial of d = degrees[i].
def getFits(scoreData, averageData, degrees):
    paramFits = []
    
    for n in degrees:
        features = feature_matrix(scoreData, n)
        modelParams = least_squares(features, averageData)
        paramFits.append(modelParams)
    #fill in
    #read the input file, assuming it has two columns, where each row is of the form [x y] as
    #in poly.txt.
    #iterate through each n in degrees, calling the feature_matrix and least_squares functions to solve
    #for the model parameters in each case. Append the result to paramFits each time.
    
    x_theory = np.linspace(min(scoreData),max(scoreData),1000)

    for k in range(0,len(degrees)):
        y_theory = [0] * len(x_theory)
        for i in range(0,len(y_theory)):
            for j in range(0,degrees[k]+1):
                y_theory[i] = y_theory[i] + x_theory[i] ** (degrees[k] - j) * paramFits[k][j]

        plt.plot(x_theory,y_theory)
        
    plt.scatter(scoreData,averageData, color="black")
    plt.legend(["Degree 1 Fit", "Degree 2 Fit", "Degree 3 Fit", "Degree 4 Fit", "Degree 5 Fit", "Experimental Data"])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("poly.txt Fitting")
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
def rsquared(X,y,model):

    #Fill in
    predy = model.predict(X)
    mse = r2_score(y, predy)

    return mse