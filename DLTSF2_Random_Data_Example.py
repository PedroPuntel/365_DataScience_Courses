# Date: 17/04/2020                                                   
# Author: Pedro H. Puntel                                            
# Email: pedro.puntel@gmail.com                                      
# Topic: 365 Data Science Course - Deep Learning With Tensorflow 2.0 
# Ecoding: UTF-8                                                     

#%% I - Setup
import numpy as np
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def get_random_data(n=1000, k=4, use_seed=42):
    np.random.seed(use_seed)
    inputs = np.random.uniform(-10, 10, (n,k-1))
    noise = np.random.normal(0, 1, (n,1))
    weights_and_bias = np.random.randint(-100, 100, (k,1))
    targets = np.dot(inputs, weights_and_bias[1:k]) + weights_and_bias[0] + noise
    print("True model coefficients are (b0, b1, ..., bk): {}".format(weights_and_bias.T))
    return weights_and_bias, inputs, targets
coefs, inputs, targets = get_random_data(n=10000, k=5, use_seed=1)

train_size= int(0.8*inputs.shape[0])
val_size= int(0.1*train_size)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2)
x_val = x_train[0:val_size,:]
y_val = y_train[0:val_size,:]
x_train = x_train[val_size:,:]
y_train = y_train[val_size:,:]

history_dict = {}

#%% II - Model Training

epochs = 100
learning_rate = 0.01
# momentum = 0.9

input_size = coefs.shape[0]
output_size = 1
loss_function = "mean_squared_error"
optmizer = tf.keras.optimizers.Adam(lr=learning_rate,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=1e-08,
                                    decay=0.0)
net = tf.keras.Sequential([
      tf.keras.layers.Dense(units=output_size,
                            activation=None,
                            use_bias=True,
                            kernel_initializer="glorot_uniform",
                            bias_initializer="zeros")])
net.compile(optimizer=optmizer,
            loss=loss_function,
            metrics=["mse"])
fit = net.fit(x=x_train,
              y=y_train,
              batch_size=None,
              epochs=epochs,
              verbose=1,
              validation_data=(x_val, y_val),
              use_multiprocessing=True)

#%% III - Model inspection

fit.model.summary()

fit.model.layers[0].get_weights()[0].round(1)
fit.model.layers[0].get_weights()[1].round(1)

fit.model.evaluate(x_train, y_train, verbose=2)

history_dict["third_attempt"] = fit
loss_plot = tfdocs.plots.HistoryPlotter(metric="MSE", smoothing_std=10)
loss_plot.plot(history_dict)
plt.title("Model Loss Function Evaluation")
plt.xlabel("Epochs")
plt.ylabel("Loss Function")

pred = fit.model.predict(x_test, verbose=2, use_multiprocessing=True)

fit.model.evaluate(x_test, y_test, verbose=2)

plt.plot(np.squeeze(pred), np.squeeze(y_test), c="green", marker="o", alpha=0.7)
plt.xlabel("Predicted")
plt.ylabel("Target")
plt.title("Predicted x Targets Scatterplot")

resid = (pred - y_test)
sns.distplot(resid, color="orange")

from scipy.stats import shapiro
norm_test = shapiro(resid)
norm_test[0]
norm_test[1]

#%% IV - Notes on different deep learning models and its performance
"""
First Attempt
-------------
    . 10k observations on 5 variables (including intercept). Used seed = 1.
    . 100 epochs, 0.001 Learning Rate, SGD optimizer with no momentum or learning schedules.
    . Training on 7.2k samples, Validation on 0.8k samples and predicting on 2k samples.
    . Already got really close to the true parameters.
    . Train Mean Squared Error of 0.995 (pretty good given the range of the data).
    . Overlapping between train and validation loss function. Took < 20 epochs to learn.
    . Prediction Mean Squared Error of 0.992, meaning that we actually didn't overfitted.
    . Residuals were pretty much N(0, Sigma), strong backend by Shapiro-Wilk's test.

Second Attempt
--------------
    . Same data with same training approach (7.2k - 0.8k - 2k).
    . Same epochs and leraning rate, SGD with Nesterov's momentum and 0.9 momentum weight.
    . Just as good on guessing the parameters.
    . Train and test were MSE a bit higher of 1.011 and 1.0033, respectively.
    . Compared with the previous loss function, it learned reasonably quicker.
    . Residuals were much closer to N(0,Sigma) than the previous as pointed by the SW test.
    . Overall a marginaly better model then the previous.

Third Attempt
-------------
    . Same data and training approach
    . 100 epochs, 0.01 learning rate, but now with adaptive learning momentum (Adam) schedules.
    . Train and test MSE closer to each other (0.998 and 1.004, respectively).
    . Much more smoother learning curve. Clearly shows that the previous two simply learned
      too quickly indeed.
    . Normality of the residuals, but not much as in the second attempt.

Notes
-----
    . Although the second attempt arguably achived slightly better results, using the Adam
      proved to be a superior and arguably more robust training approach since the model
      learned from the data in a more natural form.
    . Nevertheless, the accounting form momentum in the second attempt proved marginally
      better than the first one.
    . Given the reasonable amount of data, one could also split the data into batches in order
      for computational advantage. Overall, the network structure remaind unchanged throughout
      attempts due to the simplicity of a linear model, but this is something that should not
      be done for more complex model in which you don't know the true realtion a priori.
"""