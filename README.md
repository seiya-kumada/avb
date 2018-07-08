# constant.py
パラメータを羅列したもの。
主に、train_2.pyとpredict_2.pyとvisualize_2.ipynbで使う。

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

# phi_loss_calculator.py
(eq. 3.7)。

# predict.py
予測器。

# psi_loss_calculator.py
(eq. 3.3)。

# sampler.py
## class Sampler
データセットからのサンプリング、ガウシアンからのサンプリング。

# theta_loss_calculator.py
使ってない。

# train.py
訓練。

# visualize.ipynb
train.pyとpredict.pyの結果を描画する。 

# train_2.py
train.pyはpixel_size=4だが、これを可変長にしたもの。
さらに、GPU対応にしたもの。CPUも可。

# predict_2.py
train_2.pyに対応して使うもの。

# visualize_2.ipynb
train_2.pyとpredict_2.pyの結果を描画する。 
