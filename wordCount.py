#encoding=utf8
import jieba
import jieba.analyse
import xlwt

#写入Excel表的库

if __name__ == "__main__":
    wbk = xlwt.Workbook(encoding = 'ascii')
    sheet = wbk.add_sheet("wordCount")
    word_list = []
    key_list = []
    for line in open('products.txt'):
        item = line.strip('\n\r').split('\t')
        tags = jieba.analyse.extract_tags(item[0])
        for t in tags:
            word_list.append(t)
    word_dict = {}
    with open("wordCount.txt",'w') as wf2:
        for item in word_list:
            if item not in word_dict:
                word_dict[item] = 1
            else:
                word_dict[item] += 1
        orderList = list(word_dict.values())
        orderList.sort(reverse=True)

        for i in range(len(orderList)):
            for key in word_dict:
                if word_dict[key] == orderList[i]:
                    wf2.write(key+' '+str(word_dict[key])+'\n')
                    key_list.append(key)
                    word_dict[key] = 0
    for i in range(len(key_list)):
        sheet.write(i,1,label= orderList[i])
        sheet.write(i, 0, label=key_list[i])
    wbk.save('wordCount.xls')