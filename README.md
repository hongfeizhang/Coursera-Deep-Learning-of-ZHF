# Coursera-Deep-Learning-of-ZHF  
记录一下将代码从coursera打包下载后运行遇到的问题  
本地环境：
Windows  
conda -V  
conda 4.5.9  
默认Anaconda 常规环境已具备，包括numpy tensorflow keras等常规包  

Course 1  
Week 2  
Planar data classification with one hidden layer  包括后面的多个作业都有类似该问题  
```python
# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);

ValueError: c of shape (1, 400) not acceptable as a color sequence for x with size 400, y with size 400
```
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
```python
words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

UnicodeDecodeError: 'gbk' codec can't decode byte 0x93 in position 3136: illegal multibyte sequence
```
问题：gbk 编码  
解决: 找到w2v_utils.py 添加encoding='utf-8'   
```python
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
Emojify  
问题1 同上  
问题2 缺少emoji  
pip install emoji  

Week 3  
Neural machine translation with attention
缺少 faker tqdm 装之  

问题：  
```python
EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:
    
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis = -1)
    output = [inv_machine_vocab[int(i)] for i in prediction]
    
    print("source:", example)
    print("output:", ''.join(output))
```
```python
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-44-7ae395d727dd> in <module>()
      6     #source=source.T
      7     #source =source.reshape(1,source.shape[0],source.shape[1])
----> 8     prediction = model.predict([source, s0, c0])
      9     prediction = np.argmax(prediction, axis = -1)
     10     output = [inv_machine_vocab[int(i)] for i in prediction]

H:\Anaconda3\lib\site-packages\keras\engine\training.py in predict(self, x, batch_size, verbose, steps)
   1145                              'argument.')
   1146         # Validate user data.
-> 1147         x, _, _ = self._standardize_user_data(x)
   1148         if self.stateful:
   1149             if x[0].shape[0] > batch_size and x[0].shape[0] % batch_size != 0:

H:\Anaconda3\lib\site-packages\keras\engine\training.py in _standardize_user_data(self, x, y, sample_weight, class_weight, check_array_lengths, batch_size)
    747             feed_input_shapes,
    748             check_batch_axis=False,  # Don't enforce the batch size.
--> 749             exception_prefix='input')
    750 
    751         if y is not None:

H:\Anaconda3\lib\site-packages\keras\engine\training_utils.py in standardize_input_data(data, names, shapes, check_batch_axis, exception_prefix)
    125                         ': expected ' + names[i] + ' to have ' +
    126                         str(len(shape)) + ' dimensions, but got array '
--> 127                         'with shape ' + str(data_shape))
    128                 if not check_batch_axis:
    129                     data_shape = data_shape[1:]

ValueError: Error when checking input: expected input_4 to have 3 dimensions, but got array with shape (37, 30)
```
解决：  
```python
EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:
    
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
    source=source.T
    source =source.reshape(1,source.shape[0],source.shape[1])
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis = -1)
    output = [inv_machine_vocab[int(i)] for i in prediction]
    
    print("source:", example)
    print("output:", ''.join(output))
```
