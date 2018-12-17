import time
# 获得时间 B - A 的时间戳之差, 再 scaling 处理
def getTimeStampIntervalFromAToB(start, end='2016/6/30'):

    scale = 10000
    timeStart = time.strptime(start, "%Y/%m/%d")
    timestampStart = int(time.mktime(timeStart))

    timeEnd = time.strptime(end, "%Y/%m/%d")
    timestampEnd = int(time.mktime(timeEnd))
    return (timestampEnd - timestampStart) / scale


print(getTimeStampIntervalFromAToB('1970/1/1'))
