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
    pass