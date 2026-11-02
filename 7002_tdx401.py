import pandas as pd
import numpy as np

def generate_bins(start: float, length: int, step: float) -> list:
    """
    生成用于 Token 化的 bins 列表

    参数:
        start: 起始值
        length: bins 列表的长度
        step: 增量值
    返回:
        list: 生成的 bins 列表
    """
    return [round(start + i * step, 4) for i in range(length)]

BINS10S = generate_bins(start=-9.5, length=62, step=19./61)  # 63 个有效

# 日报表gb2312转utf8
# input file: D:/zd_huatai/T0002/export/00_1500.txt
# output file: ../ds6124csv/u1500.csv
def ex401utf8(input_file: str='D:/zd_huatai/T0002/export/0000_0401.txt', output_file: str='r:/401.txt', min_tabs=4):
    # 以 gbk 编码读取源文件，以 utf-8 编码写入目标文件
    # errors='ignore' 可根据需要添加，用来忽略原文件中可能存在的个别损坏字符
    with open(input_file, 'r', encoding='gbk') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line in fin:
            # 统计当前行中制表符 '\t' 的数量
            if line.count('\t') > min_tabs:
                fout.write(line)

    df = pd.read_csv(output_file, sep='\t', dtype={'代码': str})

    df["名称"] = df["名称"].str.replace(r"\s+", "", regex=True)
    df.columns = df.columns.str.replace(r"\s+", "", regex=True)

    df['id0']=df['代码']
    df['code2']=df['代码']//10000
    df1=df[(df['code2']==0) | (df['code2']==60)]
    print(df1)
    print(len(df1))

    out_cols=['id0','A0','A1','O100','H100','L100','C100']
    df2 = df1[out_cols].reset_index(drop=True)
    df3 = df2[df2['A0']>1.0]

    print(df3)
    print(len(df3))

    df=df3

    mapping = [
        ('to', 'O100'),
        ('th', 'H100'),
        ('tl', 'L100'),
        ('tc', 'C100'),
    ]
    offset = 1 # 🔴🔴

    for token_col, pct_col in mapping:
        if pct_col in df.columns:
            s = df[pct_col]
            tokens = pd.Series(np.digitize(s.fillna(0), BINS10S), index=df.index, dtype='Int64')
            tokens[s.isna()] = pd.NA
            df[token_col] = tokens + offset

    print(df)



    # df.to_csv('../tmp/401.csv',index=False, encoding='utf-8')
    # print(df.dtypes)

def read_ds2():
    df=pd.read_parquet("../0002_autodl/ds_h_s1_60_ds2.parquet")
    print(df)
    df1=df.iloc[:,:-6]
    print(df1)

ex401utf8()
read_ds2()
