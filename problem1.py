import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def problem1(userIDs,fracSpent,fracComp,fracPaused,numPauses,avgPBR,numRWs,numFFs,s):
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
        if(vidsCompleted[id][0] < 5):
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

    filtFracSpent = []
    filtFracComp = []
    filtFracPaused = []
    filtNumPauses = []
    filtAvgPBR = []
    filtNumRWs = []
    filtNumFFs = []
    filtS = []

    for ids in vidsCompleted.keys():
        filtFracSpent.append(vidsCompleted[ids][1])
        filtFracComp.append(vidsCompleted[ids][2])
        filtFracPaused.append(vidsCompleted[ids][3])
        filtNumPauses.append(vidsCompleted[ids][4])
        filtAvgPBR.append(vidsCompleted[ids][5])
        filtNumRWs.append(vidsCompleted[ids][6])
        filtNumFFs.append(vidsCompleted[ids][7])
        filtS.append(vidsCompleted[ids][8])
    
    
    data = np.array([filtFracSpent,filtFracComp, filtFracPaused, filtNumPauses,filtAvgPBR,filtNumRWs,filtNumFFs])
    
    categorize(data, filtS)

    pass

def categorize(dataset, s):
    dataLen = np.shape(dataset)[1]
    cutoff0 = .25
    cutoff1 = .5
    cutoff2 = .75

    category = []
    for point in s:
        if(point <= cutoff0):
            category.append(0)
        elif(point <= cutoff1):
            category.append(1)
        elif(point <= cutoff2):
            category.append(2)
        else:
            category.append(3)


    split_ind = int(np.floor(.9*dataLen))
    training_data = np.array(dataset[:,0:split_ind])
    training_data = np.array(training_data.reshape(-1,len(training_data)))
    training_labels = np.array(category[0:split_ind])
    training_labels = np.ravel(training_labels.reshape(len(training_labels),-1))

    test_data = np.array(dataset[:,split_ind+1:])
    test_data = np.array(test_data.reshape(-1,len(test_data)))
    test_labels = np.array(category[split_ind+1:])
    test_labels = np.array(test_labels.reshape(len(test_labels),-1))


    #test_data = dataset[split_ind+1:len(dataset)]
    #test_labels = category[split_ind+1:len(dataset)]

    for i in range(1,6):
        print("Number of Neighbors =", i)
        # create KNN classifier
        knn = KNeighborsClassifier(n_neighbors=i)

        # train the model using the training sets
        knn.fit(training_data, training_labels)

        # predict the response for test dataset
        test_pred = knn.predict(test_data)

        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:",metrics.accuracy_score(test_labels,test_pred))

        # calcuating the confusion matrix
        confusion_matrix = metrics.confusion_matrix(test_labels,test_pred)
        print(confusion_matrix)



