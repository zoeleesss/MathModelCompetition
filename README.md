#ReadMe

##File Overview
- / data
    -   / train       `进行训练的数据`
        - enterprise.csv
        - invest.csv
        - judgement.csv
        - partner.csv
        - patent.csv
    -  / question    `需要预测的数据`
        - enterprise.csv
        - invest.csv
        - judgement.csv
        - partner.csv
        - patent.csv
- all.csv           `训练数据处理结果`
- all_question.csv  `需要预测数据处理结果`
- answer.csv        `预测结果`
- bpnn.py           `BPNN训练并预测`
- process.py        `数据处理`
- id_mapping.binary `数据处理中间结果`
- README.md

##  Run 
- 先运行 `process.py` 文件，可以得到 `all.csv` `all_question.csv` `id_mapping.binary`
- 后运行 `bpnn.py` 文件，可以得到 `answer.csv` 

## Note
- 部分文件由于编码格式的原因，因此手工删除了部分列。

## Credit
- `Data Ultimate 战队`
- `各位指导老师`
