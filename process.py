import csv
import numpy as np
import time
import pickle

# 保存 str_id => int_id 映射关系
enterprise_id_mapping = {}

# 保存 int_id => str_id 映射关系
enterprise_str_mapping_predict = {}

# 保存 industry_code  'Char' => 'int' 映射关系
industry_code_mapping = {}


def serialize(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def deSerialize(filename):
    with open(filename , 'rb') as f:
        obj = pickle.load(f)
    return obj


# 加载csv文件，并返回数据 data, m 是数据行数
def loadCSVfilePredict(filename,encoding='utf8'):
    data = []
    label = []
    m = 0
    isfirst = 1
    csv_file = csv.reader(open(filename,'r',encoding=encoding))
    for line in csv_file:
        if (isfirst == 1):
            label.append(line)
            isfirst = 0
        else:
            if (line[2] != '' ):
                data.append(line)
                m += 1

    return [data, label, m]

# 加载csv文件，并返回数据 data, m 是数据行数
def loadCSVfile(filename,encoding='utf8'):
    data = []
    label = []
    m = 0
    isfirst = 1
    csv_file = csv.reader(open(filename,'r',encoding=encoding))
    for line in csv_file:
        if (isfirst == 1):
            label.append(line)
            isfirst = 0
        else:
            if (line[3] != '' ):
                data.append(line)
                m += 1

    return [data, label, m]

# 将data 保存到指定文件
def saveCSVfile(data,filename):
    out = open(filename, 'w', newline='')
    csv_write = csv.writer(out, dialect='excel')
    for i in range(0,len(data)):
        csv_write.writerow(data[i])

# 转置矩阵
def trans(m):
    a = [[] for i in m[0]]
    for i in m:
        for j in range(len(i)):
            a[j].append(i[j])
    return a

# 增加一个维度： 股东的数量
def add_partner_info(data, filename):
    [p_data,p_label,p_m] = loadCSVfile(filename,'utf8')
    for i in range(0,len(data)):
        data[i].append(0)

    for d in p_data:
        enterprise_id = d[0]
        try:
            int_id = enterprise_id_mapping[enterprise_id]
            data[int_id][-1] += 1
        except:
            continue

# 增加一个维度： 股东的数量
def add_question_partner_info(data, filename):
    [p_data,p_label,p_m] = loadCSVfile(filename,'utf8')
    for i in range(0,len(data)):
        data[i].append(0)

    for d in p_data:
        enterprise_id = d[0]
        try:
            int_id = enterprise_id_mapping[enterprise_id]
            data[int_id][-1] += 1
        except:
            continue

# 增加 3 维度： 专利的数量
def add_patent_info(data, filename):
    [p_data, p_label, p_m] = loadCSVfile(filename,'utf8')
    for i in range(0, len(data)):
        # 每一行增加3 个维度： 不同种类的专利数量
        data[i].append(0)   # -3
        data[i].append(0)   # -2
        data[i].append(0)   # -1

    for d in p_data:
        enterprise_id = d[0]
        patent_type = d[3]
        try:
            int_id = enterprise_id_mapping[enterprise_id]
            if patent_type == '1':
                data[int_id][-3] += 1
            elif patent_type == '2':
                data[int_id][-2] += 1
            elif patent_type == '3':
                data[int_id][-1] += 1
        except:
            continue


# 增加 3 维度： 专利的数量
def add_question_patent_info(data, filename):
    add_patent_info(data, filename)


# 增加 4 维度： 历史的各类投资的数量
def add_invest_info(data, filename):
    [p_data, p_label, p_m] = loadCSVfile(filename,'utf8')
    for i in range(0, len(data)):
        # 每一行增加3 个维度： 不同种类的投资数量
        data[i].append(0)  # -4
        data[i].append(0)  # -3
        data[i].append(0)  # -2
        data[i].append(0)  # -1

    for d in p_data:
        enterprise_id = d[0]
        # NOTE : 此处的invest 表是在删除invest_name 后处理的 -- 防止编码错误      ok
        invest_level = d[4]
        try:
            int_id = enterprise_id_mapping[enterprise_id]
            invest_time = d[2]
            # isValidInvestment = getTimeStampIntervalFromAToB(invest_time, '2010/6/30')
            # if isValidInvestment > 0:
            if invest_level == '1':
                data[int_id][-4] += 1
            elif invest_level == '2':
                data[int_id][-3] += 1
            elif invest_level == '3':
                data[int_id][-2] += 1
            elif invest_level == '4':
                data[int_id][-1] += 1
        except:
            continue


# 增加 4 维度： 历史的各类投资的数量
def add_question_invest_info(data, filename):
    [p_data, p_label, p_m] = loadCSVfile(filename)
    for i in range(0, len(data)):
        # 每一行增加3 个维度： 不同种类的投资数量
        data[i].append(0)  # -4
        data[i].append(0)  # -3
        data[i].append(0)  # -2
        data[i].append(0)  # -1

    for d in p_data:
        enterprise_id = d[0]
        # NOTE : 此处的invest 表是没有删除 invest_name 列
        invest_level = d[4]
        try:
            int_id = enterprise_id_mapping[enterprise_id]
            invest_time = d[2]
            # isValidInvestment = getTimeStampIntervalFromAToB(invest_time, '2010/6/30')
            # if isValidInvestment > 0:
            if invest_level == '1':
                data[int_id][-4] += 1
            elif invest_level == '2':
                data[int_id][-3] += 1
            elif invest_level == '3':
                data[int_id][-2] += 1
            elif invest_level == '4':
                data[int_id][-1] += 1
        except:
            continue


# 增加 6 维度： 历史的各类判决的数量
def add_judgement_info(data, filename):
    p_data = []
    m = 0
    isfirst = 1
    csv_file = csv.reader(open(filename, 'r', encoding='utf8'))
    for line in csv_file:
        if (isfirst == 1):
            isfirst = 0
        else:
            p_data.append(line)
            m += 1

    for i in range(0, len(data)):
        # 每一行增加3 个维度： 不同种类的投资数量
        data[i].append(0)  # -6
        data[i].append(0)  # -5
        data[i].append(0)  # -4
        data[i].append(0)  # -3
        data[i].append(0)  # -2
        data[i].append(0)  # -1

    for d in p_data:
        enterprise_id = d[0]
        # NOTE : 此处的judgement 表是在删除judgement_name 后处理的 -- 防止编码错误    ok
        judgement_type = d[3]
        try:
            int_id = enterprise_id_mapping[enterprise_id]
            if judgement_type == '1':
                data[int_id][-6] += 1
            elif judgement_type == '2':
                data[int_id][-5] += 1
            elif judgement_type == '3':
                data[int_id][-4] += 1
            elif judgement_type == '4':
                data[int_id][-3] += 1
            elif judgement_type == '5':
                data[int_id][-2] += 1
            elif judgement_type == '6':
                data[int_id][-1] += 1
        except:
            continue


# 增加 6 维度： 历史的各类判决的数量
def add_question_judgement_info(data, filename):
    p_data = []
    m = 0
    isfirst = 1
    csv_file = csv.reader(open(filename, 'r', encoding='utf8'))
    for line in csv_file:
        if (isfirst == 1):
            isfirst = 0
        else:
            p_data.append(line)
            m += 1

    for i in range(0, len(data)):
        # 每一行增加3 个维度： 不同种类的投资数量
        data[i].append(0)  # -6
        data[i].append(0)  # -5
        data[i].append(0)  # -4
        data[i].append(0)  # -3
        data[i].append(0)  # -2
        data[i].append(0)  # -1

    for d in p_data:
        enterprise_id = d[0]
        # NOTE : 此处的judgement 表是 没有 删除judgement_name 列
        judgement_type = d[3]
        try:
            int_id = enterprise_id_mapping[enterprise_id]
            if judgement_type == '1':
                data[int_id][-6] += 1
            elif judgement_type == '2':
                data[int_id][-5] += 1
            elif judgement_type == '3':
                data[int_id][-4] += 1
            elif judgement_type == '4':
                data[int_id][-3] += 1
            elif judgement_type == '5':
                data[int_id][-2] += 1
            elif judgement_type == '6':
                data[int_id][-1] += 1
        except:
            continue

# 考虑关键字的影响，权重 1 维度
def getKeywordWeight(product):

    keywords = {"共享": 0.6122, "人工智能": 0.9285, "机器人": 0.8219, "分布式": 1.0, "区块链": 1.0, "互联网": 0.5792, "系统集成": 0.4,
                "跨境": 0.6875, "智能制造": 0.6667, "教育": 0.5344, "新能源汽车": 0.9091, "VR": 0.8620, "养老": 0.8571}
    weight = 0.5
    for (k, v) in keywords.items():
        if k in product:
            if v > weight:
                weight = v
    return weight


# 将原始数据在索引数组 位置的数据 转换成处理后的数据
def transform_data(primitive_data, indexes):

    data = []
    for i in indexes:
        # 将类型由str转换成float
        t = primitive_data[:, i]
        t = [float(x) for x in t]
        # 把若干个列表合成一个大二维列表
        data.append(t)

    # 转置矩阵
    data = trans(data)
    data = np.array(data)

    return data

# 获得时间 B - A 的时间戳之差, 再 scaling 处理
def getTimeStampIntervalFromAToB(start, end='2016/6/30'):

    scale = 10000
    timeStart = time.strptime(start, "%Y/%m/%d")
    timestampStart = int(time.mktime(timeStart))

    timeEnd = time.strptime(end, "%Y/%m/%d")
    timestampEnd = int(time.mktime(timeEnd))
    return (timestampEnd - timestampStart) / scale

# 获得索引数组，规则: 选中的chooseArr 索引 + 某位置开始剩下所有的索引
def get_index_array(len, chooseArr, the_rest_index):
    indexes = []
    indexes += chooseArr
    for i in range(the_rest_index,len):
        indexes.append(i)
    return indexes

def processTrainingData():
    # 增加企业信息，及初始化
    industry_code_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11,
                             'L': 12,'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'null': 20}

    [data,label, m] = loadCSVfile('data/train/enterprise.csv','utf8')
    per_line = ['tag1','tag2','tag3','tag4']
    label[0] += per_line
    col = 0

    # 添加有用数据到 data 中
    for i in range(0,m):
        per_line = [0,0,0,0]

        # 增加tag 4 维度
        tag = data[i][5]
        tag = tag.split("，")
        for j in range(0,len(tag)):
            if (tag[j] != ''):
                per_line[int(tag[j]) - 1] = 1

        # 增加企业的注册时间维度
        registered_time = data[i][2]
        timestamp = getTimeStampIntervalFromAToB(registered_time)
        per_line.append(timestamp)

        # 增加企业代码维度 20 维度
        for ins in range(0, 20):
            per_line.append(0)
        industry_code = data[i][7]
        try:
            industry_index = industry_code_mapping[industry_code] + 3
            per_line[industry_index] = 1
        except:
            # 如'' , 'NULL' 等异常情况
            per_line[-1] = 1

        # 增加关键字权重 1 维度
        # per_line.append(0)
        # keyword = data[i][6]
        # weight = getKeywordWeight(keyword)
        # per_line[-1] = weight

        # 添加到每一行的数据项中
        data[i] += per_line
        enterprise_id_mapping[data[i][0]] = col
        data[i][0] = col
        col += 1

    # 使用partner 表信息
    add_partner_info(data, 'data/train/partner.csv')

    # 使用patent 表信息
    add_patent_info(data, 'data/train/patent.csv')

    # 使用invest 表信息
    add_invest_info(data, 'data/train/invest.csv')

    # 使用judgement 表信息
    add_judgement_info(data, 'data/train/judgement.csv')

    m = np.array(data)  # 转换为numpy.array

    # 把所需要的信息列举出来
    indexes = get_index_array(len(m[0]),[0,1,3,9],10)
    print(indexes)

    # 并将类型由str转换成float
    data = transform_data(m, indexes)

    # 打印列的数量,以第一行为例
    print(len(data[0,:]))
    print(data[0,:])

    # 保存有用的信息
    saveCSVfile(data, "all.csv")

def processPredictData():

    # 增加企业信息，及初始化
    enterprise_id_mapping.clear()

    industry_code_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11,
                             'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'null': 20}

    [data, label, m] = loadCSVfilePredict('data/question/enterprise.csv','utf8')
    per_line = ['tag1', 'tag2', 'tag3', 'tag4']
    label[0] += per_line
    col = 0

    # 添加有用数据到 data 中
    for i in range(0, m):
        per_line = [0, 0, 0, 0]

        # 增加tag 4 维度
        tag = data[i][4]
        tag = tag.split("，")
        for j in range(0, len(tag)):
            if (tag[j] != ''):
                per_line[int(tag[j]) - 1] = 1

        # 增加企业的注册时间 1 维度
        registered_time = data[i][1]
        timestamp = getTimeStampIntervalFromAToB(registered_time)
        per_line.append(timestamp)

        # 增加企业代码维度 20 维度
        for ins in range(0, 20):
            per_line.append(0)
        industry_code = data[i][6]
        try:
            industry_index = industry_code_mapping[industry_code] + 3
            per_line[industry_index] = 1
        except:
            # 如'' , 'NULL' 等异常情况
            per_line[-1] = 1

        # 增加关键字权重 1 维度
        # per_line.append(0)
        # keyword = data[i][6]
        # weight = getKeywordWeight(keyword)
        # per_line[-1] = weight

        # 添加到每一行的数据项中
        data[i] += per_line
        enterprise_id_mapping[data[i][0]] = col
        enterprise_str_mapping_predict[col] = data[i][0]
        data[i][0] = col
        col += 1

    # 使用partner 表信息
    add_question_partner_info(data, 'data/question/partner.csv')

    # 使用patent 表信息
    add_question_patent_info(data, 'data/question/patent.csv')

    # 使用invest 表信息
    add_question_invest_info(data, 'data/question/invest.csv')

    # 使用judgement 表信息
    add_question_judgement_info(data, 'data/question/judgement.csv')


    m = np.array(data)  # 转换为numpy.array

    # 把所需要的信息列举出来
    indexes = get_index_array(len(m[0]), [0, 2, 8], 9)

    print(indexes)

    # 并将类型由str转换成float
    data = transform_data(m, indexes)

    # 打印列的数量,以第一行为例
    print(len(data[0, :]))
    print(data[2,:])

    # 保存有用的信息
    saveCSVfile(data, "all_question.csv")
    serialize(enterprise_id_mapping, "id_mapping.binary")



if __name__ == '__main__':

    # 处理 training data 成 all.csv
    processTrainingData()

    # 处理 question data 成 all_question.csv + id_mapping.binary
    processPredictData()
