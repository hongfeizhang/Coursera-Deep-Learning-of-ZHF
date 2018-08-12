# Coursera-Deep-Learning-of-ZHF  
记录一下将代码从coursera打包下载后运行遇到的问题  
本地环境：
Windows  
conda -V  
conda 4.5.9  
默认Anaconda 常规环境已具备，包括numpy tensorflow keras等常规包  

Course 1  
Planar data classification with one hidden layer  包括后面的多个作业都有类似该问题  
![image](https://github.com/hongfeizhang/Coursera-Deep-Learning-of-ZHF/blob/master/images/4.PNG)  
问题：ValueError: c of shape (1, 400) not acceptable as a color sequence for x with size 400, y with size 400  
解决：改为 
```python
plt.scatter(X[0, :], X[1, :], c=Y.flatten(), s=40, cmap=plt.cm.Spectral); 
```
后续的类似问题同样使用这个方法解决

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
```
def read_glove_vecs(glove_file):
    with open(glove_file, 'r',encoding='utf-8') as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec_map
```
