# coding: utf-8

# reference: http://blog.csdn.net/u012162613/article/details/41929171#comments

import csv
from numpy import *
import  operator  # add

def loadTrainData():
    trainData=[]
    with open('./data/train.csv') as file:
        lines=csv.reader(file)
        for line in lines:
            trainData.append(line)
    trainData.remove(trainData[0])
    trainData=array(trainData)
    trainLabel=trainData[:,0]
    trainData=trainData[:,1:]
    return nomalizing(toInt(trainData)),toInt(trainLabel)

def toInt(array):
    array=mat(array)
    m,n=shape(array)
    newArray=zeros((m,n))
    for i in range(m):
        for j in range(n):
            newArray[i,j]=int(array[i,j])
    return newArray

def nomalizing(array):
    m,n=shape(array)
    for i in range(m):
        for j in range(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array

def loadTestData():
    data=[];
    with open('./data/test.csv') as file:
        lines=csv.reader(file)
        for line in lines:
            data.append(line)
    data.remove(data[0])
    data=array(data)
    return nomalizing(toInt(data))

def loadTestResult():
    label=[]
    with open('./data/sample_submission.csv') as file:
        lines=csv.reader(file)
        for line in lines:
            label.append(line)
    label.remove(label[0])
    label=array(label)
    return toInt(label[:,1])

def classify(inX, dataSet, labels, k):
    inX=mat(inX)
    dataSet=mat(dataSet)
    labels=mat(labels)
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = array(diffMat)**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[0,sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  #'dict' object has no attribute 'iteritems'  Python3.5中：iteritems变为items
    return sortedClassCount[0][0]

def saveResult(result):
    with open('result.csv','w') as myFile:
        myWriter=csv.writer(myFile)
        for i in result:
            tmp=[]
            tmp.append(i)
            myWriter.writerow(tmp)  #python 3.5: TypeError: a bytes-like object is required, not 'str';出现该错误往往是通过open()函数打开文本文件时，使用了‘rb’属性


def handwritingClassTest():
    trainData,trainLabel=loadTrainData()
    print ("\n loadTrainData finished...")
    testData=loadTestData()
    print ("\n loadTestData finished...")
    testLabel=loadTestResult()
    print ("\n loadTestResult finished...")
    m,n=shape(testData)
    errorCount=0
    resultList=[]
    for i in range(m):
         classifierResult = classify(testData[i], trainData, trainLabel, 5)
         resultList.append(classifierResult)
         print ("the classifier came back with: %d, the real answer is: %d, predict order: %d" % (classifierResult, testLabel[0,i], i))
         if (classifierResult != testLabel[0,i]): errorCount += 1.0
    print ("\nthe total number of errors is: %d" % errorCount)
    print ("\nthe total error rate is: %f" % (errorCount/float(m)))
    saveResult(resultList)


if __name__=="__main__":
    
    handwritingClassTest()