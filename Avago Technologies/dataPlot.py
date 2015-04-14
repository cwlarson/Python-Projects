# -*- coding: utf-8 -*-
# Title: plotFinal.py
# Last Modified: 1.3.13
# Description: Reads in one or more data files, and graphs for each AND the difference of the set (if more than one):
#       1.) Frequency Error (PPM) CDF
#       2.) Output Power CDF
#       3.) Bias Current CDF
#       4.) 1 KHz Phase Noise CDF
#       5.) 800 KHz Phase Noise CDF
#       6.) Bar graph depicting # of failures in each category

import os
import numpy as NP
import scipy.special as SCI
import matplotlib.pyplot as PLOT
import matplotlib.axes as AX
from math import *
from sets import Set
import operator

from pylab import *
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import logging
import numpy.core.numeric as NX
import pprint
pp = pprint.PrettyPrinter(indent=4)

# Global variables:
dieListPath = ""        # Set in getFilenames()
testNames = []          # Set in getFilenames()
filenames = []          # Set in getFilenames()
pltDir = ""             # Set in getFilenames()
f0 = 0                  # Set in getFilenames()

dieList = {}            # Set in getDieList()

dates = {}              # Set in getHeaderInfo()
lots = {}               # Set in getHeaderInfo()
sublots = {}            # Set in getHeaderInfo()
columns = {}            # Set in getHeaderInfo()
lines = {}              # Set in getHeaderInfo()

testData = {'all':{}, 'final':{}, 'plot':{}}    # Set in readTestData()
finalPIDs = {}                                  # Set in readTestData()

bins = {}                               # Set in sortTestData()
mean = {'data':{}, 'diff':{}}           # mean['data'] set in sortTestData(), mean['diff'] set in getDifferences()
sigma = {'data':{}, 'diff':{}}          # sigma['data'] set in sortTestData(), sigma['diff'] set in getDifferences()

diffSets = []           		# Set in getDifferences()
differences = {'dict':{}, 'list':{}}    # Set in getDifferences()

ys = {'data':{}, 'diff':{}, 'bins':{}}  # y['data'] set in plotData(), y['diff'] set in plotDifferences(), y['bins'] set in plotBins()

parameters = ['DIE_X','DIE_Y','TEMPERATURE_VE__3.3','FreqOut_VE__3.3','Pout_VE__3.3','Icc_VE__3.3','PhaseNoise_1K_VE__3.3','PhaseNoise_800K_VE__3.3']
paramInfo = {
        'titles': {      
                'FreqOut_VE__3.3':'Frequency Error (PPM) CDF',
                'Pout_VE__3.3':'Output Power CDF',
                'Icc_VE__3.3':'Bias Current CDF',
                'PhaseNoise_1K_VE__3.3':'1 KHz Phase Noise CDF',
                'PhaseNoise_800K_VE__3.3':'800 KHz Phase Noise CDF'},

        'xLabels': {     
                'FreqOut_VE__3.3':'PPM',
                'Pout_VE__3.3':'dBm',
                'Icc_VE__3.3':'ma',
                'PhaseNoise_1K_VE__3.3':'dBc/hz @ 1 KHz',
                'PhaseNoise_800K_VE__3.3':'dBc/hz @ 800 KHz'},
        
        'plotted': ['FreqOut_VE__3.3','Pout_VE__3.3','Icc_VE__3.3','PhaseNoise_1K_VE__3.3','PhaseNoise_800K_VE__3.3']
}


def getFilenames():
        global dieListPath
        global testNames
        global filenames
        global pltDir
        global f0
        
        dataInfoFile = raw_input("Name of .gti file: ")
        dataInfoFP = open(dataInfoFile, 'r')
        dataInfo = dataInfoFP.readlines()
        dataInfoFP.close()
        
        pltDir = dataInfoFile.split('.gti')[0] + "_plts"
        if not os.path.exists(pltDir):
                os.makedirs(pltDir)
        pltDir += '/'
        
        for line in dataInfo:
                if line[0:2].lower() == "f0":
                        f0 = int(line.split(':')[1].strip())
                elif line[0:7].lower() == "dielist":
                        dieListPath = line.split(':')[1].strip()
                elif line[0] != '#': 
                        testNames.append(line.split(':')[0].strip())
                        filenames.append(line.split(':')[1].strip())
        return


def getDieList():
        global dieList
        
        dieFP = open(dieListPath, 'r')
        dieListContents = dieFP.readlines()
        dieFP.close()

        # Get Die List header information
        header = dieListContents[0].split(',')
        for val in range(len(header)): header[val] = header[val].split('\r')[0]
        for i in range(1,len(dieListContents)):
                line = dieListContents[i].split(',')
                dieID = line[2]
                dieList[dieID] = {}
                dieList[dieID][header[0]] = line[0].strip()
                dieList[dieID][header[1]] = line[1].strip()
                dieList[dieID][header[3]] = line[3].strip()
        return


def readTestData():
        global testData
        global finalPIDs
        
        # Read file contents
        testContents = []
        for file in filenames:
                testFP = open(file, 'r')
                test = testFP.readlines()
                if len(test) == 1: test = test[0].split('\r')
                testContents.append(test)
                testFP.close()
        
        getHeaderInfo(testContents)
        
        # Read in test data
        for testInfo in testContents:
                test = testNames[testContents.index(testInfo)]
                finalPIDs[test] = int(testInfo[-1].split(',')[0].split('PID-')[1])
                for line in range(lines[test], len(testInfo)):
                        splitLine = testInfo[line].split(',')
                        pid = splitLine[0].strip()
                        if not pid in testData['all']: testData['all'][pid] = {}
                        if not pid in testData['final']: testData['final'][pid] = {}
                        for parameter in parameters:
                                testData['all'][pid][parameter+test] = splitLine[columns[test][parameter]].strip()
                                testData['all'][pid]['pid'] = pid
                                testData['final'][pid][parameter+test] = splitLine[columns[test][parameter]].strip()
                                testData['final'][pid]['pid'] = pid
        
        # If PID info is missing from one or more tests, include in 'PID Discrepancies.txt'
        lastPID = 0
        for pid in finalPIDs: lastPID = max(finalPIDs[pid], lastPID)
        logFP = open(pltDir+"_PID Discrepancies.txt", "w+")
        logFP.write("PID discrepancies:")
        error1, error2 = "", ""
        for test in testNames:
                for pid in range(1, lastPID+1):
                        if not "PID-"+str(pid) in testData['all']:
                                # PID-X was skipped for all tests
                                error1 += "\n\tPID-{} was not included in any test".format(pid)
                                print error1
                        elif not parameters[0]+test in testData['all']['PID-'+str(pid)]:
                                # PID-X was skipped for one or more tests
                                error2 += "\n\tPID-{0} does not have data for test {1} (Lot {2}, Sublot {3})".format(pid, test, lots[test], sublots[test])
                                print error2
        if error1 == "":
                if error2 == "": logFP.write("\n\tNONE: All PIDs included in all tests")
                else: logFP.write(error2)
        else: logFP.write(error1+'\n\n'+error2)
        logFP.close()
        return


def getHeaderInfo(testContents):
        global dates
        global lots
        global sublots
        global columns
        global lines
        
        for test in testContents:
                testIndex = testContents.index(test)
                if len(test) == 1: test = test[0].split('\r')
                def getParamInfo(desired):
                        line = 0
                        while test[line].split(',')[0].strip() != desired:
                                line += 1
                        return test[line].split(',')[1].strip()
                dates[testNames[testIndex]] = getParamInfo("Date")
                lots[testNames[testIndex]] = getParamInfo("Lot")
                sublots[testNames[testIndex]] = getParamInfo("SubLot")

                line = 0
                cols = {}
                while test[line].split(',')[0].strip() != 'Parameter':
                        line += 1
                tempLine = test[line].split(',')
                for i in range(len(tempLine)):
                        index = tempLine[i].strip() 
                        if index in parameters:
                                cols[index] = i
                columns[testNames[testIndex]] = cols

                while test[line].split(',')[0].strip() != 'LowL':
                        line += 1
                lines[testNames[testIndex]] = line+1
        return


def sortTestData():
        global bins
        global mean
        global sigma
        badPIDs = {}
        
        # Set up "Failure Data.csv"
        logFP = open(pltDir+"_Failure Data.csv", "w+")
        logFP.write("Bin 0,Passed\nBin 1,Current Failure\nBin 2,Power Failure\nBin 3,Frequency Failure\nBin 4,1k Phase Noise Failure\nBin 5,800k Phase Noise Failure\n\n")
        logFP.write("PID,Test,Bin")
        for parameter in parameters: logFP.write(","+parameter)
        
        # Sort PIDs into bins
        for test in testNames:
                bins[test] = {}
                for i in range(6): bins[test][i] = 0
                
                testData['plot'][test] = {}
                for parameter in parameters: testData['plot'][test][parameter] = []
                testData['plot'][test]['xy'] = []
                
                for pid in testData['final']:
                        if 'FreqOut_VE__3.3'+test in testData['final'][pid]: 
                                bin = binSelect(pid, test)
                                bins[test][bin] += 1
                                if bin == 0:
                                        testData['plot'][test]['TEMPERATURE_VE__3.3'].append(float(testData['final'][pid]['TEMPERATURE_VE__3.3'+test]))
                                        ppm = 1e6*(float(testData['final'][pid]['FreqOut_VE__3.3'+test])-f0)/f0
                                        testData['plot'][test]['FreqOut_VE__3.3'].append(ppm)
                                        testData['final'][pid]['FreqOut_VE__3.3'+test] = ppm
                                        testData['plot'][test]['Pout_VE__3.3'].append(float(testData['final'][pid]['Pout_VE__3.3'+test]))
                                        testData['plot'][test]['Icc_VE__3.3'].append(float(testData['final'][pid]['Icc_VE__3.3'+test]))
                                        testData['plot'][test]['PhaseNoise_1K_VE__3.3'].append(float(testData['final'][pid]['PhaseNoise_1K_VE__3.3'+test]))
                                        testData['plot'][test]['PhaseNoise_800K_VE__3.3'].append(float(testData['final'][pid]['PhaseNoise_800K_VE__3.3'+test]))
                                        testData['plot'][test]['xy'].append([testData['final'][pid]['DIE_X'+test], testData['final'][pid]['DIE_Y'+test]])
                                else:
                                        if not pid in badPIDs: badPIDs[pid] = {}
                                        for tst in testNames:
                                                if not tst in badPIDs[pid]: badPIDs[pid][tst] = 0
                                        badPIDs[pid][test] = bin
        
        # Remove PIDs that failed any parameter in any test from final plotting data (and log)
        for pid in badPIDs:
                for test in testNames:
                        logFP.write("\n{0},{1},{2}".format(pid, test, badPIDs[pid][test]))
                        for parameter in parameters: logFP.write(","+str(testData['final'][pid][parameter+test]))
                del testData['final'][pid]
        
        for parameter in paramInfo['plotted']:
                mean['data'][parameter] = {}
                sigma['data'][parameter] = {}
                for test in testNames:
                        mean['data'][parameter][test] = NP.mean(NP.array(testData['plot'][test][parameter]))
                        sigma['data'][parameter][test] = NP.std(NP.array(testData['plot'][test][parameter]))
                        
        return


def binSelect(pid, test):
        bin = 0
        freq = float(testData['final'][pid]['FreqOut_VE__3.3'+test])
        current = float(testData['final'][pid]['Icc_VE__3.3'+test])
        power = float(testData['final'][pid]['Pout_VE__3.3'+test])
        pn1k = float(testData['final'][pid]['PhaseNoise_1K_VE__3.3'+test])
        pn800k = float(testData['final'][pid]['PhaseNoise_800K_VE__3.3'+test])
        
        if current < 5 or current > 25: bin = 1
        if (power < -20 or power > 0) and bin == 0: bin = 2
        if (freq < 2558 or freq > 2658) and bin == 0: bin = 3
        if pn1k > -68 and bin == 0: bin = 4
        if (pn800k < -180 or pn800k > -120) and bin == 0: bin = 5
        return bin


def getDifferences():
        global mean
        global sigma
        global diffSets
        global differences
        
        # Get diffSets
        for test in testNames[1:]:
                diffSets.append(test+'-'+testNames[0])
        
        # Get differences
        for pid in testData['final']:
                differences['dict'][pid] = {}
                for parameter in paramInfo['plotted']:
                        for test in testNames[1:]:
                                if (parameter+test in testData['final'][pid]) and (parameter+testNames[0] in testData['final'][pid]):
                                        diffSet = test+'-'+testNames[0]
                                        val1 = float(testData['final'][pid][parameter+testNames[0]])
                                        val2 = float(testData['final'][pid][parameter+test])
                                        differences['dict'][pid][parameter+diffSet] = val2 - val1
        
        # Find mean and sigma of differences
        for parameter in paramInfo['plotted']:
                mean['diff'][parameter] = {}
                sigma['diff'][parameter] = {}
                differences['list'][parameter] = {}
                for diffSet in diffSets:
                        differences['list'][parameter][diffSet] = []
                        temp = []
                        for pid in differences['dict']:
                                if parameter+diffSet in differences['dict'][pid]: temp.append(differences['dict'][pid][parameter+diffSet])
                        mean['diff'][parameter][diffSet] = NP.mean(NP.array(temp))
                        sigma['diff'][parameter][diffSet] = NP.std(NP.array(temp))
        
        # Find data points not within the range M±5σ        
        badPIDs = {}
        for pid in differences['dict']:
                for parameter in paramInfo['plotted']:
                        for diffSet in diffSets:
                                if parameter+diffSet in differences['dict'][pid]:
                                        val = differences['dict'][pid][parameter+diffSet]
                                        upBound = mean['diff'][parameter][diffSet]+(5*sigma['diff'][parameter][diffSet])
                                        lowBound = mean['diff'][parameter][diffSet]-(5*sigma['diff'][parameter][diffSet])
                                        if not ((val > lowBound) and (val < upBound)):
                                                # Value outside of range
                                                if not pid in badPIDs: badPIDs[pid] = {}
                                                if not diffSet in badPIDs[pid]: badPIDs[pid][diffSet] = []
                                                badPIDs[pid][diffSet].append(parameter)
                                        else: differences['list'][parameter][diffSet].append(val)
                                                
        
        # Log outlying data points and remove them from the final data set
        logFP = open(pltDir+"_∆ Error Log.csv", "w+")
        logFP.write("PID,Test")
        for parameter in parameters: logFP.write(","+parameter) 
        for pid in badPIDs:
                for test in testNames: 
                        logFP.write("\n{},{}".format(pid, test))
                        for parameter in parameters: logFP.write(",{}".format(testData['final'][pid][parameter+test]))
                for diffSet in diffSets:
                        logFP.write("\n{},{}".format(pid,diffSet))
                        if diffSet in badPIDs[pid]:
                                for parameter in parameters:
                                        if parameter in badPIDs[pid][diffSet]: 
                                                logFP.write(",{}".format(differences['dict'][pid][parameter+diffSet]))
                                                del differences['dict'][pid][parameter+diffSet]
                                        else: logFP.write(",")
        return


def plotData():
        global ys

        for parameter in paramInfo['plotted']:
                PLOT.figure(figsize=(8,7))
                PLOT.subplots_adjust(top=(1-(0.05*len(testNames))))
                ax=gca()
                PLOT.xlabel(paramInfo['xLabels'][parameter])
                PLOT.ylabel('cumulative probability')
                PLOT.yticks(yTicks, yLabels)
                PLOT.grid(True)
                
                plotTitle = paramInfo['titles'][parameter]
                xTotal = []
                maxLengthTest = len("Test:")
                maxLengthMean = len("Mean:")
                maxLengthSigma = len("Sigma:")
                for diffSet in diffSets: maxLengthTest = max(maxLengthTest, len(diffSet))
                print "\nFor {}:".format(parameter)

                for test in testNames:
                        x = testData['plot'][test][parameter]
                        x.sort()
                        xTotal.extend(x)
                        y = getY(len(x))
                        if not test in ys['data']: ys['data'][test] = {}
                        ys['data'][test][parameter] = y
                        fit, yFit = getFit(x,y)
                        Mean = "{0:1.3f}".format(mean['data'][parameter][test])
                        Sigma = "{0:1.3f}".format(sigma['data'][parameter][test])
                        maxLengthTest = max(len(test), maxLengthTest)
                        maxLengthMean = max(len(Mean), maxLengthMean)
                        maxLengthSigma = max(len(Sigma), maxLengthSigma)
                        print "Fit is: σ={0}, M={1} (Lot {2}, Sublot {3})".format(Sigma, Mean, lots[test], sublots[test])
                        PLOT.plot(x,y)
                        PLOT.plot(yFit, [-3,3], '--')
                
                # Show only 0.5% - 99.5% (to get rid of outliers)
                ind = int(ceil(0.005*len(xTotal)))
                xTotal.sort()
                xMin = xTotal[ind]
                xMax = xTotal[-ind]
                PLOT.axis([xMin, xMax, -3, 3])
                
                # Configure callout
                for diffSet in diffSets: maxLengthTest = max(len(diffSet), maxLengthTest)
                maxLengthTest += 2
                maxLengthMean += 2
                maxLengthSigma += 2
                callout = "Test".ljust(maxLengthTest) + "Mean".ljust(maxLengthMean) + "Sigma".ljust(maxLengthSigma)
                for test in testNames:
                        callout += "\n" + test.ljust(maxLengthTest)
                        callout += "{0:1.3f}".format(mean['data'][parameter][test]).ljust(maxLengthMean)
                        callout += "{0:1.3f}".format(sigma['data'][parameter][test]).ljust(maxLengthSigma)
                for diffSet in diffSets:
                        callout += "\n"+diffSet.ljust(maxLengthTest)
                        callout += "{0:1.3f}".format(mean['diff'][parameter][diffSet]).ljust(maxLengthMean)
                        callout += "{0:1.3f}".format(sigma['diff'][parameter][diffSet]).ljust(maxLengthSigma)
                PLOT.text(0.05, (0.9-(0.05*len(testNames))), callout, transform=ax.transAxes, fontsize=14, bbox=dict(boxstyle="round,pad=0.5",fc="w",ec="g"), family='Courier New')
                
                # Configure title
                maxLot, maxSublot, maxDate = 0,0,0
                for test in testNames:
                        maxLot = max(len("Lot: {}".format(lots[test])), maxLot)
                        maxSublot = max(len("Sublot: {}".format(sublots[test])), maxSublot)
                        maxDate = max(len("Date: {}".format(dates[test])), maxDate)
                maxLot += 2
                maxSublot += 2
                maxDate += 2
                for test in testNames:
                        plotTitle += "\n" + "Lot: {}".format(lots[test]).ljust(maxLot)
                        plotTitle += "Sublot: {}".format(sublots[test]).ljust(maxSublot)
                        plotTitle += "Date: {}".format(dates[test]).ljust(maxDate)
                PLOT.title(plotTitle, family="Courier New")
                PLOT.savefig(pltDir+paramInfo['titles'][parameter]+".png", format="png")
        return


def getY(length):
        y = []
        yTemp = NP.array(range(length))
        yTemp = (yTemp+1.)/float(len(yTemp))
        for cp in yTemp:
                if cp < 0.5: map = -sq*SCI.erfinv(1. - 2.*cp)
                elif cp > 0.5: map = sq*SCI.erfinv(2.*cp - 1.)
                else: map = 0.
                y.append(map)
        return y

        
def getFit(x, y):
        fitMin = int(0.25*len(y))
        fitMax = int(0.75*len(y))
        fit = NP.polyfit(y[fitMin:fitMax], x[fitMin:fitMax], 1)
        yFit = NP.polyval(fit, [-3,3])
        return fit, yFit


def plotDifferences():
        global ys
        
        for parameter in paramInfo['plotted']:
                PLOT.figure(figsize=(8,7))
                ax = gca()
                PLOT.xlabel(paramInfo['xLabels'][parameter])
                PLOT.ylabel('cumulative probability')
                PLOT.yticks(yTicks, yLabels)
                PLOT.grid(True)
                
                plotTitle = "Difference ({})".format(paramInfo['titles'][parameter])
                maxLengthTest = len("Set:")
                maxLengthMean = len("Mean:")
                maxLengthSigma = len("Sigma:")
                xTotal = []
                plts = []
                
                for diffSet in diffSets:
                      x = differences['list'][parameter][diffSet]
                      x.sort()
                      xTotal.extend(x)
                      y = getY(len(x))
                      if not diffSet in ys['diff']: ys['diff'][diffSet] = {}
                      ys['diff'][diffSet][parameter] = y
                      fit, yFit = getFit(x,y)
                      Mean = "{0:1.3f}".format(mean['diff'][parameter][diffSet])
                      Sigma = "{0:1.3f}".format(sigma['diff'][parameter][diffSet])
                      maxLengthTest = max(len(diffSet), maxLengthTest)
                      maxLengthMean = max(len(Mean), maxLengthMean)
                      maxLengthSigma = max(len(Sigma), maxLengthSigma)
                      pTmp = PLOT.plot(x,y)
                      plts.append(pTmp[0])
                      PLOT.plot(yFit, [-3,3], '--')
                
                # Show only 0.5% - 99.5% (to get rid of outliers)
                ind = int(ceil(0.005*len(xTotal)))
                xTotal.sort()
                xMin = xTotal[ind]
                xMax = xTotal[-ind]
                PLOT.axis([xMin, xMax, -3, 3])
                
                PLOT.legend(plts, diffSets, loc=2)
                PLOT.title(plotTitle)
                PLOT.savefig(pltDir+plotTitle+".png", format="png")
        return


def plotBins():
        global bins
        global ys
        
        failedBins = {}
        width = 0.5
        index = NP.arange(5)
        colors = ('#000099', '#ff7700', '#992244', '#cc88ff', 'b')
        
        # Create dictionary of failures
        for test in bins:
                failedBins[test] = {}
                failedBins[test]['bins'] = {}
                failedBins[test]['total'] = 0
                for bin in range(1, len(bins[test])):
                        failedBins[test]['bins'][bin] = bins[test][bin]
                        failedBins[test]['total'] += bins[test][bin]

        for test in testNames:
                if failedBins[test]['total'] > 0:
                        # Sort failures high -> low
                        pidsPassed = len(testData['plot'][test][paramInfo['plotted'][0]])
                        pidsFailed = failedBins[test]['total']
                        percentYield = float(100.*pidsPassed/(pidsPassed + pidsFailed))
                        percentFailed = []
                        binNums = []
                        xTicks = []
                        legendInfo = []
                        sorted_fails = sorted(failedBins[test]['bins'].items(), key=lambda x: x[1], reverse=True)
                        for tuple in sorted_fails:
                                pct = float("{0:1.3f}".format(100.*tuple[1]/(pidsPassed + pidsFailed)))
                                percentFailed.append(pct)
                                xTicks.append("Bin {}".format(tuple[0]))
                                legendInfo.append("Bin {}: {} failed ({}%)".format(tuple[0], tuple[1], pct))
                        rangeTop = 2*max(percentFailed)
                        ys['bins'] = []
                        if rangeTop > 15: ys['bins'] = range(0, int(rangeTop), int(rangeTop/15))
                        elif rangeTop < 5:
                                for i in range(1,11):
                                        ys['bins'].append(float("{0:1.2f}".format(rangeTop/10*i)))
                        else: ys['bins'] = range(int(rangeTop))
                        
                        # Plot bins
                        PLOT.figure()
                        ax = gca()
                        PLOT.title("Yield\nLot: {}      Sublot: {}".format(lots[test], sublots[test]))
                        PLOT.ylabel("failure (%)")
                        plt = PLOT.bar(index, percentFailed, width, color=colors)
                        PLOT.yticks(ys['bins'])
                        PLOT.xticks(index+width/2, xTicks)
                        PLOT.legend((plt[0], plt[1], plt[2], plt[3], plt[4]), legendInfo)
                        PLOT.text(.6, .5, "Total Passing: {0}\nTotal Failures: {1}\nYield: {2:1.3f}%".format(pidsPassed, pidsFailed, percentYield), transform=ax.transAxes, fontsize=14, bbox=dict(boxstyle="round,pad=0.5",fc="w",ec="b"))
                        PLOT.text(0.75, -0.09, dates[test], transform=ax.transAxes)
                        PLOT.savefig("{0}Yield (Lot {1}, Sublot {2}).png".format(pltDir, lots[test], sublots[test]), format='png')
        return


sq = 1./sqrt(0.5)
yTicks = []
yLabels = []
cumDistFun = [.005, .01, .02, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .98, .99, .995]
for val in cumDistFun:
    yLabels.append(repr(val))
    if val < .5:
        map = -sq*SCI.erfinv(1. - 2.*val)
    elif val > .5:
        map = sq*SCI.erfinv(2.*val-1.)
    else:
        map = 0
    yTicks.append(map)

getFilenames()
getDieList()
readTestData()
sortTestData()
getDifferences()
plotData()
plotDifferences()
plotBins()

PLOT.show()
