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

from load_cifar_10 import load_cifar_10_data
import numpy as np
from skimage.color import rgb2gray
#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
######################################################################
def load_cifar_3(train_data, train_labels, test_data, test_labels):

    zipped_train = list(zip(train_data,train_labels))
    zipped_test = list(zip(test_data, test_labels))

    filtering_criteria = lambda x: x[1]<3
    filtered_train = list(filter(filtering_criteria, zipped_train))
    filtered_test = list(filter(filtering_criteria, zipped_test))

    filtered_trained_data = np.array([i for i,_ in filtered_train])
    filtered_trained_labels = np.array([j for _,j in filtered_train])

    filtered_test_data = np.array([i for i,_ in filtered_test])
    filtered_test_labels = np.array([j for _,j in filtered_test])

    return filtered_trained_data, filtered_trained_labels, filtered_test_data, filtered_test_labels

######################################################################

if __name__ == "__main__":

    cifar_10_dir = 'cifar-10-batches-py'

    # loading CIFAR-10 dataset
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
            load_cifar_10_data(cifar_10_dir)

    # filtering CIFAR-10 to make it CIFAR-3
    train_data, train_labels, test_data, test_labels = load_cifar_3(train_data, train_labels, test_data, test_labels)

    # converting rgb input images to gray then flattening the input images.
    train_data = rgb2gray(train_data).reshape(len(train_data),-1)
    test_data = rgb2gray(test_data).reshape(len(test_data),-1)
    
    i = train_labels.size()
    if i == 1500:
        print("yeah")
    else:
        print("neh")

    ### fill in any necessary code below to perform the task outlined in README.md document. ###
    for i in range(1,6):
        print("Number of Neighbors =", i)

        # create KNN classifier
        knn = KNeighborsClassifier(n_neighbors=i)

        # train the model using the training sets
        knn.fit(train_data, train_labels)

        # predict the response for test dataset
        test_pred = knn.predict(test_data)

        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:",metrics.accuracy_score(test_labels, test_pred))

        # calcuating the confusion matrix
        confusion_matrix = metrics.confusion_matrix(test_labels, test_pred)
        print(confusion_matrix, "\n")

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
    pass