## 文本多标签分类

本项目对来自Wikipedia’s talk页面的评论数据集建立一个多标签识别模型，识别恶意评论的类别，比如威胁，侮辱，身份歧视等，使得用户可以选择某类感兴趣的评论。
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

+	text normalization：修复错拼单词（out-of-vocabulary），创建一个词表包含恶意评论中的高频词，对每个错拼单词查询这个列表，选择编辑距离小的词来纠正这个错拼单词。利用TextBlob dictionary修复错拼单词。利用Fasttext tool创建错拼单词的词向量。
+	data augmentation：通过把文本翻译成外文再翻译回来的方式增加数据，减小过拟合。需要注意的是信息泄露的可能性，要保证原始评论和翻译评论在同一边，也就是同一句的语料要被分在相同的数据集。
+	model ensemble：在不同的预训练的embedding上独立训练多个模型进行平均（FastText and Glove embedding）。
+	模型框架是 Embedding + Dropout + Bi-GRU + Pooling(average pooling + max Pooling) + Fully connected。


模型的AUC Score为0.9862。
