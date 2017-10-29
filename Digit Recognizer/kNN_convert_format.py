# coding: utf-8

from kNN_mnist import *
import csv


def loadTestData1():
    data = [];
    with open('test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            data.append(line)
    # data.remove(data[0])
    data = array(data)
    return nomalizing(toInt(data))


def loadResult():
    label = [];
    with open('result.csv', 'r', newline='') as file:
        lines = csv.reader(file, doublequote=False)  # doubequote=False
        for line in lines:
            label.append(line)
    # label.remove(label[0])
    # label=array(label)
    # label=[i for i in label if i%2==0]
    label = label[::2]  # 那么获得奇数的语句print x[::2];偶数的语句print x[1::2]
    # reverse list :[int(i) for i in label]
    data = []
    data = array(label)  # use two times array error
    m, n = shape(data)
    #newArray = zeros((m, n), int)  ##bug

    newArray = [];
    for i in range(m):
        for j in range(n):
            temp = float(data[i, j])  # data[i,j]='2.0'
            #newArray[i, j] = int(temp)  #********
            newArray.append(int(temp))

    with open('result_temp.csv', 'w', newline='') as myFile:
        myWriter = csv.writer(myFile)
        for i in newArray:  # i is list
            tmp = []
            tmp.append(i)
            #print (tmp)
            myWriter.writerow(tmp)



def saveResult1(result):
    with open('result1.csv', 'w') as myFile:
        myWriter = csv.writer(myFile)
        for i in result:
            tmp = []
            tmp.append(i)
            myWriter.writerow(tmp)

def saveCsvfile(listfile):
    csvfile = open('kNN_Digit_Recognize.csv', 'w', newline = '')
    #要有参数 newline = '' 否则会出现每一行后空一行的现象。
    writer = csv.writer(csvfile)
    writer.writerow(['ImageId', 'Label']) #标题
    data = []
    for i in enumerate(listfile):
        data.append((i[0]+1,i[1])) #enumerate的序号是从0开始的，所以要加1
    writer.writerows(data)
    csvfile.close()

if __name__ == "__main__":

    sampleResult = loadResult()
    # saveResult1(sampleResult)

# result = []
# l = range(10)
# for i in l:
#     if i % 2 == 0:
#         result.append(i)
# print(result)


# a=range(40)
# csvFile = open("test_execel.csv", "w",newline='')  ## newline='' py3.5需要
# writer = csv.writer(csvFile,quoting=csv.QUOTE_MINIMAL)
# # 写入的内容都是以列表的形式传入函数
# writer.writerow(a)
# csvFile.close()


## execise csv
# import csv
# # 文件头，一般就是数据名
# fileHeader = ["name", "score"]
# # 假设我们要写入的是以下两行数据
# d1 = ["Wang", "100"]
# d2 = ["Li", "80"]
# # 写入数据
# csvFile = open("instance.csv", "w",newline='')  ## newline='' py3.5需要
# writer = csv.writer(csvFile,quoting=csv.QUOTE_MINIMAL)
# # 写入的内容都是以列表的形式传入函数
# writer.writerow(fileHeader)
# writer.writerow(d1)
# writer.writerow(d1)
# csvFile.close()

# import csv
#
# with open('names.csv', 'w') as csvfile:
#     fieldnames = ['first_name', 'last_name']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#     writer.writeheader()
#     writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
#     writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
#     writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})


# import csv
# with open('eggs.csv', 'w', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=' ',
#                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
#     spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
#     spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
