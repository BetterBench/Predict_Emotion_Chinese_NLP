# 情感预测Baseline

# 1 题目

给定一个对话训练集

每一段话有多个句子

每个句子都有情感标注

要求根据历史对话，预测最后一个句子的情感。总共有6类情感。

![question](https://z3.ax1x.com/2021/11/04/Im0Lxx.jpg)

# 2 项目结构

```b
├── data
│   ├── clear_data.csv # 清洗后的文本数据
│   └── train_data.csv # 原始数据集
├── preprocess_train.ipynb # 文本清洗
├── stop
│   └── cn_stopwords.txt # 中文停用词文件
├── train.py # LGB模型训练baseline
```

这只是一个简单的机器学习Baseline，仅仅作为一个入门NLP的简单的学习例子，程序的精度并不能保证

解决思路是：

将问题看成是文本分类问题。

通过TF_IDF 提取文本的特征后得到特征矩阵，再将每个句子的标签作为特征合并到提取到的特征矩阵 中，每个句子的标签不包括最后一个句子的标签。

由于每个样本的标签个数不一样，数据分析可知最多有10个标签，就通过前面补0来填充长度到10。比如标签是123456，补充后为0000123456。只取前9个标签合并到特征矩阵，最后一个标签作为整个样本的label 。

