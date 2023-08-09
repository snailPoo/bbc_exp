Code for reproducing results of **Implementation and Performance Analysis of Lossless Variational Image Compression using Bits-Back Coding**

# Table of Contents
* [Introduction](#1)
* [Getting Started](#2)
* [Usage](#3)
* [Questions](#4)

<h2 id="1">Introduction</h2>

這份工作整合 [bbans](https://github.com/bits-back/bits-back), [bit-swap](https://github.com/fhkingma/bitswap), [hilloc](https://github.com/hilloc-submission/hilloc) 的 code，在相同硬體條件下實驗三個模型在重要指標的表現，並嘗試刻出 [shvc](https://arxiv.org/pdf/2204.02071.pdf)。hilloc 使用 pytorch 改寫。

<h2 id="2">Getting Started</h2>

* 環境設置
```
conda env create -f bbc.yaml
```
* 設定環境變數
```
 export PYTHONPATH=$PYTHONPATH:~/bbc_exp
```
* 安裝 [craystack](https://github.com/j-towns/craystack)（vectorized ANS package）

* 下載 [ImageNet32 Dataset](https://www.image-net.org/download.php)

<h2 id="3">Usage</h2>

**如何訓練 xxx model？**  
1. 到 [config.py](./config.py) 修改 class Config_xxx 的超參數
2. [model/train.py](./model/train.py) 修改 cf = Config_xxx()
3. 執行訓練
```
python model/train.py
```

**如何執行壓縮測試？**  

0. [下載 model checkpoint](https://drive.google.com/drive/folders/1LuL_slRQxpq9Jx3v1lxzN_FaatzYsesS?usp=share_link)
1. 到 [config.py](./config.py) 修改 class Config_xxx 的超參數
2. 執行測試：  

**bbans**
```
# using original ANS
python bbans_ANS_compress.py
# using vectorized ANS
python vANS_compress.py
``` 
**bit-swap**
```
python compress.py
```
**hilloc**
```
python vANS_compress.py
```


<h2 id="4">Questions</h2>
Please contact me (r10944054@cmlab.csie.ntu.edu.tw) if you have any questions.

