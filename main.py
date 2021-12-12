import re
from problem1 import *
from problem2 import *
from problem3 import *

if __name__ == '__main__':
    with open('behavior-performance.txt','r') as fil:
        fileText = fil.readlines()
        fil.close()
    userIDs = [] 
    vidIDs = []
    fracSpent = []
    fracComp = []
    fracPlayed = []
    fracPaused = []
    numPauses = []
    avgPBR = []
    stdPBR = []
    numRWs = []
    numFFs = []
    s = []
    #                      1:ID    2:VID     3:Spent      4:fracComp   5:fracPlay   6:fracPaused
    parser = re.compile('(\w+)\s+(\d+)\s+(\d+\.*\d*e*\-*\d*)\s(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.*\d*e*\-*\d*)\s+(\d+)\s+(\d+\.*\d*)\s+(\d+\.*\d*e*\-*\d*)\s+(\d+)\s+(\d+)\s+(\d+)\s*') 
    fileText.pop(0)

    for line in fileText:
        parsedLine = parser.search(line)
        if(parsedLine != None):
            userIDs.append(str(parsedLine.group(1)))
            vidIDs.append(int(parsedLine.group(2)))
            fracSpent.append(float(parsedLine.group(3)))
            fracComp.append(float(parsedLine.group(4)))
            fracPlayed.append(float(parsedLine.group(5)))
            fracPaused.append(float(parsedLine.group(6)))
            numPauses.append(int(parsedLine.group(7)))
            avgPBR.append(float(parsedLine.group(8)))
            stdPBR.append(float(parsedLine.group(9)))
            numRWs.append(int(parsedLine.group(10)))
            numFFs.append(int(parsedLine.group(11)))
            s.append(int(parsedLine.group(12)))
   
    #problem1(userIDs,fracSpent,fracComp,fracPaused,numPauses,avgPBR,numRWs,numFFs,s)
    #problem2(userIDs,fracSpent,fracComp,fracPaused,numPauses,avgPBR,numRWs,numFFs,s)
    problem3(fracSpent,fracComp,fracPaused,numPauses,avgPBR,numRWs,numFFs,s)
