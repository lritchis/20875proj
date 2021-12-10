def problem1(userIDs = [],vidIDs = [],fracSpent = [],fracComp = [],fracPlayed = [],fracPaused = [],numPauses = [],avgPBR = [],stdPBR = [],numRWs = [],numFFs = [],s = []):
    userIDSet = set(userIDs)
    print(len(userIDs))
    print(len(userIDSet))
    vidsCompleted = {}
    #for i in range(0,len(userIDs)):
    #    if(fracComp[i] == 1):