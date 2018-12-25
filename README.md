# ReadMe

## File Overview
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

## Input Params `47`

- Enterprise:   `31 PARAMS`
    - Tag `4 params` 
    - Registered time   `1 param`
    - Industry code     `20 params`
    - Product           `1 param`
    - Address           `3 params`
    - 企业员工            `2 params`
- Partner:      `3 PARAMS`
    - 个人股东的数量  &nbsp;&nbsp;&nbsp;  `1`
    - 普通企业股东数量  `1`
    - 投资机构股东数量  `1`
- Patent:       `3 PARAMS`
    - 历史的3类专利的数量   `3`
- Invest:       `4 PARAMS`
    - 历史的4类投资的数量 `4`
- Judgement:    `6 PARAMS`
    - 历史的6类判决的数量    `6`

## Credit
- `Data Ultimate 战队`
- `各位指导老师`
