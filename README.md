Learning Tensorflow
==

#Setup

```bash
$ virtualenv tfvenv
$ source tfvenv/bin/activate
$ pip install tensorflow ipython jupyter matplotlib
$ pip install scipy Pillow
$ git clone https://github.com/aymericdamien/TensorFlow-Examples
$ git clone https://github.com/tensorflow/tensorflow
$ git clone https://github.com/tensorflow/models
```

#Following various examples to create my own networks
https://github.com/martin-gorner/tensorflow-mnist-tutorial/
https://docs.google.com/presentation/d/1TVixw6ItiZ8igjp6U17tcgoFrLSaHWQmMOwjlgQY9co/pub?slide=id.g140797b42d_0_60

https://github.com/BinRoot/TensorFlow-Book

http://eblearn.sourceforge.net/lib/exe/lenet5.png

https://www.tensorflow.org/tutorials/mnist/pros/

#Interesting Links

http://learningtensorflow.com/lesson7/

http://sebastianruder.com/optimizing-gradient-descent/

https://github.com/aqibsaeed/Multilabel-timeseries-classification-with-LSTM

#Notes
The `iris.py` example uses tf.contrib.learn, a simplified scikit-learn like interface. Not important to learn.

# hyperparameter tuning
Got this with decay = 2000, max_lr = 0.01, min_lr = max_lr / 100
At epoch 0,   accuracy: 0.130000, learning rate: 0.010000
At epoch 100, accuracy: 0.870000, learning rate: 0.009517
At epoch 200, accuracy: 0.940000, learning rate: 0.009058
At epoch 300, accuracy: 0.950000, learning rate: 0.008621
At epoch 400, accuracy: 0.960000, learning rate: 0.008205
At epoch 500, accuracy: 0.960000, learning rate: 0.007810

# Questions
- How to make it less sensitive to hyperparameters?
- Why multiply loss by 100 instead of turning up learning rate?
