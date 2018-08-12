# Coursera-Deep-Learning-of-ZHF
记录一下将代码从coursera打包下载后运行遇到的问题

默认Anaconda 常规环境已具备，包括numpy tensorflow keras等常规包

Course 5

Week 1

Improvise a Jazz Solo with an LSTM Network

![image](https://github.com/hongfeizhang/Coursera-Deep-Learning-of-ZHF/blob/master/images/1.PNG)

问题：缺少music21 

解决：conda install -c mbonix music21


Week2
Operations on word vectors

![image](https://github.com/hongfeizhang/Coursera-Deep-Learning-of-ZHF/blob/master/images/2.png)

问题：gbk 编码

解决: 找到w2v_utils.py 添加encoding='utf-8'

![image](https://github.com/hongfeizhang/Coursera-Deep-Learning-of-ZHF/blob/master/images/3.png)
