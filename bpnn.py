#encoding=utf8
import csv
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle

# 加载之前处理后的数据 ，并返回数据 data, m 是数据行数
def loadProcessedCSVfile(filename,encoding='utf8'):
    data = []
    m = 0
    csv_file = csv.reader(open(filename,'r',encoding=encoding))
    for line in csv_file:
        data.append(line)
        m += 1

    return [data, m]

def deSerialize(filename):
    with open(filename , 'rb') as f:
        obj = pickle.load(f)
    return obj


# 将data 保存到指定文件
def saveCSVfile(data,filename):
    out = open(filename, 'w', newline='')
    csv_write = csv.writer(out, dialect='excel')
    for i in range(0,len(data)):
        csv_write.writerow(data[i])


class BPNN(object):

    def __init__(self,input_n,hidden_n,output_n,lamada):
        """
        这是BP神经网络类的构造函数
        :param input_n:输入层神经元个数
        :param hidden_n: 隐藏层神经元个数
        :param output_n: 输出层神经元个数
        :param lamada: 正则化系数
        """
        self.sess = tf.Session()
        self.Train_Data = tf.placeholder(tf.float64,shape=(None,input_n),name='input_dataset')      # 训练数据集
        self.Train_Label = tf.placeholder(tf.float64,shape=(None,output_n),name='input_labels')     # 训练数据集标签
        self.input_n = input_n                                        # 输入层神经元个数
        self.hidden_n = hidden_n                                      # 隐含层神经元个数
        self.output_n = output_n                                      # 输出层神经元个数
        self.lamada = lamada                                            # 正则化系数
        self.input_weights = tf.Variable(tf.random_normal((self.input_n, self.hidden_n),mean=0,stddev=1,dtype=tf.float64),trainable=True)                                       # 输入层与隐含层之间的权重
        self.hidden_weights = tf.Variable(tf.random_normal((self.hidden_n,self.output_n),mean=0,stddev=1,dtype=tf.float64),trainable=True)                                      # 隐含层与输出层之间的权重
        self.hidden_threshold = tf.Variable(tf.random_normal((1,self.hidden_n),mean=0,stddev=1,dtype=tf.float64),trainable=True)                                            # 隐含层的阈值
        self.output_threshold = tf.Variable(tf.random_normal((1,self.output_n),mean=0,stddev=1,dtype=tf.float64),trainable=True)                                            # 输出层的阈值
        # 将层与层之间的权重与偏置项加入损失集合
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.lamada)(self.input_weights))
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.lamada)(self.hidden_weights))
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.lamada)(self.hidden_threshold))
        tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(self.lamada)(self.output_threshold))
        # 定义前向传播过程
        self.hidden_cells = tf.nn.sigmoid(tf.matmul(self.Train_Data, self.input_weights) + self.hidden_threshold)
        self.output_cells = tf.nn.sigmoid(tf.matmul(self.hidden_cells, self.hidden_weights) + self.output_threshold)
        # 定义损失函数,并加入损失集合
        self.MSE = tf.reduce_mean(tf.square(self.output_cells-self.Train_Label))
        tf.add_to_collection('loss',self.MSE)
        # 定义损失函数,均方误差加入L2正则化
        self.loss = tf.add_n(tf.get_collection('loss'))

    def train_test(self,Train_Data,Train_Label,Test_Data,Test_Label,learn_rate,epoch,iteration,batch_size):
        """
        这是BP神经网络的训练函数
        :param Train_Data: 训练数据集
        :param Train_Label: 训练数据集标签
        :param Test_Data: 测试数据集
        :param Test_Label: 测试数据集标签
        :param learn_rate:  学习率
        :param epoch:  时期数
        :param iteration: 一个epoch的迭代次数
        :param batch_size:  小批量样本规模
        """
        train_loss = []                 # 训练损失
        test_loss = []                  # 测试损失


        datasize = len(Train_Label)
        self.train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        for e in np.arange(epoch):
            for i in range(iteration):
                start = (i*batch_size)%datasize
                end = np.min([start+batch_size,datasize])
                self.sess.run(self.train_step,
                         feed_dict={self.Train_Data:Train_Data[start:end],self.Train_Label:Train_Label[start:end]})
                if i % 10000 == 0:
                    total_MSE = self.sess.run(self.MSE,
                                         feed_dict={self.Train_Data:Train_Data,self.Train_Label:Train_Label})
                    print("第%d个epoch中，%d次迭代后，训练MSE为:%g"%(e+1,i+10000,total_MSE))
            # 训练损失
            _train_loss = self.sess.run(self.MSE,feed_dict={self.Train_Data:Train_Data,self.Train_Label:Train_Label})
            train_loss.append(_train_loss)
            # 测试损失
            _test_loss = self.sess.run(self.MSE, feed_dict={self.Train_Data:Test_Data, self.Train_Label: Test_Label})
            test_loss.append(_test_loss)
            # 测试精度
            test_result = self.sess.run(self.output_cells,feed_dict={self.Train_Data:Test_Data})

        return train_loss,test_loss

    # def Accuracy(self,test_result,test_label):
    #     """
    #     这是BP神经网络的测试函数
    #     :param test_result: 测试集预测结果
    #     :param test_label: 测试集真实标签
    #     """
    #     predict_ans = []
    #     label = []
    #     for (test,_label) in zip(test_result,test_label):
    #         test = np.exp(test)
    #         test = test/np.sum(test)
    #         predict_ans.append(np.argmax(test))
    #         label.append(np.argmax(_label))
    #     return accuracy_score(label,predict_ans)

    def predict(self, case):
        return self.sess.run(self.output_cells,feed_dict={self.Train_Data:case})

def run_main():
    """
       这是主函数
    """
    [data, m] = loadProcessedCSVfile('all.csv')
    data = np.array(data)
    Data = data[:, 2:m]
    Label = []
    for i in range(0,m):
        la = []
        la.append(data[i, 1])
        Label.append(la)

    # 分割数据集,并对数据集进行标准化
    Train_Data,Test_Data,Train_Label,Test_Label = train_test_split(Data,Label,test_size=1/5,random_state=10)
    scaler = preprocessing.StandardScaler()
    Train_Data = scaler.fit_transform(Train_Data)
    Test_Data = scaler.fit_transform(Test_Data)

    # 设置网络参数
    input_n = np.shape(Data)[1]
    output_n = np.shape(Label)[1]
    # hidden_n = int(np.sqrt(input_n*output_n))
    hidden_n = input_n
    lamada = 0.0001
    batch_size = 64
    learn_rate = 0.1
    epoch = 15
    iteration = 10000

    # 训练并测试网络
    bpnn = BPNN(input_n,hidden_n,output_n,lamada)
    train_loss,test_loss = bpnn.train_test(Train_Data,Train_Label,Test_Data,Test_Label,learn_rate,epoch,iteration, batch_size)

    # 看看训练的拟合正确率
    predict_data = Test_Data[:,:]
    predict_data = scaler.fit_transform(predict_data)

    result = bpnn.predict(predict_data)
    res = []
    right_count = 0
    count = len(Test_Label)
    win_lable_count = 0
    for i in range(0,count):
        if (np.float64(Test_Label[i][0]) == 1.0):
            win_lable_count += 1
        # else:
        #     print(Test_Label[i])
        result[i][0] = (result[i][0] > 0.5)
        res.append(result[i][0])
        if np.float64(Test_Label[i][0]) == result[i][0]:
            right_count += 1

    # print(result[0][0])

    #print(Test_Data)
    #print(Test_Label)
    print('label')
    print(win_lable_count)
    print(count - win_lable_count)
    print("predict")
    print(len(predict_data))
    # print(predict_data)
    #print(Test_Label)
    #print('res')
    #print(res)
    # 正确率
    print('result')
    print(right_count/count)


    # 结果可视化
    col = ['Train_Loss','Test_Loss']
    epoch = np.arange(epoch)
    plt.plot(epoch,train_loss,'r')
    plt.plot(epoch,test_loss,'b-.')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(labels = col,loc='best')
    plt.savefig('训练与测试损失.png')
    plt.show()
    plt.close()

    # 对于question 数据进行处理，并预测
    question_data = []
    isfirst = 1
    csv_file = csv.reader(open('data/question/enterprise.csv', 'r',encoding='utf8'))
    for line in csv_file:
        if (isfirst == 1):
            isfirst = 0
        else:
            question_data.append(line)

    # 预测
    [question_predict_data, m] = loadProcessedCSVfile('all_question.csv')
    question_predict_data = np.array(question_predict_data)
    question_predict_data = question_predict_data[:, 1:m]
    question_predict_data = scaler.fit_transform(question_predict_data)
    question_result = bpnn.predict(question_predict_data)

    # 获得之前保存的id => 序号 映射
    enterprise_id_mapping_predict = deSerialize("id_mapping.binary")

    # 格式为 : id , probability
    question_answer = []
    # 加上label
    per_answer = []
    per_answer.append('enterprise_id')
    per_answer.append('probability')
    question_answer.append(per_answer)

    # 开始处理数据
    for i in range(0,len(question_data)):
        ent_id = question_data[i][0]
        per_probability = 0
        try:
            index_id = enterprise_id_mapping_predict[ent_id]
            per_probability = question_result[index_id][0]
        except:
            pass
        per_answer = []
        per_answer.append(ent_id)
        per_answer.append(per_probability)
        question_answer.append(per_answer)
        #print('ent id ',ent_id,' and result ',per_probability)

    # 将结果保存
    saveCSVfile(question_answer, 'answer.csv')


if __name__ == '__main__':
    run_main()

