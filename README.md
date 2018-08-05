# constant.py
パラメータを羅列したもの。
主に、train\_2.pyとpredict\_2.pyとvisualize\_2.ipynbで使う。

# dataset.py
## class Dataset
データセットを生成する。
one-hotベクトルである。

# decoder.py
デコーダ。

# discriminator.py
識別器。

# encoder.py
エンコーダ。

# evaluate.py
使ってない。

# phi\_loss\_calculator.py
(eq. 3.7)。

# predict.py
予測器。

# psi\_loss\_calculator.py
(eq. 3.3)。

# sampler.py
## class Sampler
データセットからのサンプリング、ガウシアンからのサンプリング。

# theta\_loss\_calculator.py
使ってない。

# train.py
訓練。

# visualize.ipynb
train.pyとpredict.pyの結果を描画する。 

# train\_2.py
- train.pyはpixel\_size=4だが、これを可変長にしたもの。
- GPU対応にしたもの。CPUも可。
- 引数受け取りをやめてファイルから読み込むようにしたもの。

# predict\_2.py
train\_2.pyに対応して使うもの。

# visualize\_2.ipynb
train\_2.pyとpredict\_2.pyの結果を描画する。 
