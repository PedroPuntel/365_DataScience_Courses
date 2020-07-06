# Date: 26/05/2020                                                   
# Author: Pedro H. Puntel                                            
# Email: pedro.puntel@gmail.com                                      
# Topic: 365 Data Science Course - Deep Learning With Tensorflow 2.0 
# Ecoding: UTF-8                                                     

# References:
# . https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
# . https://michaelblogscode.wordpress.com/2017/12/20/visualizing-model-performance-statistics-with-tensorflow/
# . https://www.ritchieng.com/machine-learning-evaluate-classification-model/
# . https://stats.stackexchange.com/questions/388552/
# . https://stats.stackexchange.com/questions/310952/
# . https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
# . https://parasite.id/blog/2018-12-13-model-evaluation/

# Multi-Class Classification
# . Problem of predicting an integer value, where each class is assigned a unique integer from 0 to (n–1).

# Modules
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Random data generation
from sklearn.datasets import make_blobs
inputs, targets = make_blobs(n_samples=10000,
                             n_features=2,
                             centers=5,
                             cluster_std=2,
                             shuffle=True,
                             random_state=42)

# Visualizing
for i in range(len(set(targets))):
	samples_ix = np.where(targets == i)
	plt.scatter(inputs[samples_ix, 0], inputs[samples_ix, 1], label=str(i))
plt.title("Multi-Class Classification Example - Random Points")
plt.legend()
plt.show()

# Checking for class imbalance
# . Since our data set is quite balanced, we mey discard threshold moving analysis
sns.catplot(x="y", kind="count", data=pd.DataFrame(targets, columns=["y"]))
plt.title("Checking for Class Imbalance")
plt.xlabel("Classes")
plt.ylabel("Counts")
plt.show()

# Multi-Class Cross-Entropy Loss Function
# . Preffered loss function for Multi-Class classification problems under the framework of MLE.
# . Calculates a score that summarizes the average difference between the actual and predicted
#   probability distributions for all classes in the problem.
# . Perfect cross-entropy value is 0.
# . The function requires that the output layer is configured with n nodes (one for each class),
#   and a ‘softmax‘ activation in order to predict the probability for each class.
# . Reponse variable must be one-hot encoded.

from sklearn.model_selection import train_test_split
x_test, x_train, y_test_cat, y_train_cat = train_test_split(inputs, targets, test_size=0.8, random_state=42)
y_train_cat = tf.keras.utils.to_categorical(y_train_cat)
y_test_cat = tf.keras.utils.to_categorical(y_test_cat)

epochs = 1000
learning_rate = 0.01
loss_function = "categorical_crossentropy"
optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)

cce_net = tf.keras.Sequential([
    tf.keras.layers.Dense(units=50,
                          activation="relu",
                          use_bias=True,
                          kernel_initializer="glorot_uniform",
                          bias_initializer="zeros",
                          input_shape=[inputs.shape[1]]),
    tf.keras.layers.Dense(units=len(set(targets)), activation="softmax")
])

cce_net.compile(loss=loss_function, optimizer=optimizer, metrics=["acc"])

cce_model = cce_net.fit(x=x_train,
                        y=y_train_cat,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test,y_test_cat),
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)])

# Sparse Multi-Class Cross-Entropy Loss Function
# . Adresses possible large number of labels in one-hot encoding problem (memory hungry)
# . Does not require one-hot encoding of the labels

x_test, x_train, y_test, y_train = train_test_split(inputs, targets, test_size=0.8, random_state=42)

epochs = 1000
learning_rate = 0.01
loss_function = "sparse_categorical_crossentropy"
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

scce_net = tf.keras.Sequential([
    tf.keras.layers.Dense(units=50,
                          activation="relu",
                          use_bias=True,
                          kernel_initializer="glorot_uniform",
                          bias_initializer="zeros",
                          input_shape=[inputs.shape[1]]),
    tf.keras.layers.Dense(units=len(set(targets)), activation="softmax")
])

scce_net.compile(loss=loss_function, optimizer=optimizer, metrics=["acc"])

scce_model = scce_net.fit(x=x_train,
                          y=y_train,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(x_test,y_test),
                          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)])

# Kullback-Leibler Divergence Loss Function
# . Measure of how one probability distribution differs from a baseline distribution.
# . A KL divergence loss of 0 suggests the distributions are identical. In practice,
#   the behavior of KL Divergence is very similar to cross-entropy. It calculates how much
#   information is lost (in terms of bits) if the predicted probability distribution is used to
#   approximate the desired target probability distribution.
# . Commonly used when using models that learn to approximate a more complex function than
#   simply multi-class classification, such as in the case of an autoencoder used for learning a
#   dense feature representation under a model that must reconstruct the original input.
# . Requires one-hot encoding of the labels.

x_test, x_train, y_test_cat, y_train_cat = train_test_split(inputs, targets, test_size=0.8, random_state=42)
y_train_cat = tf.keras.utils.to_categorical(y_train_cat)
y_test_cat = tf.keras.utils.to_categorical(y_test_cat)

epochs = 1000
learning_rate = 0.01
loss_function = "kullback_leibler_divergence"
optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate, rho=0.9)

klb_net = tf.keras.Sequential([
    tf.keras.layers.Dense(units=100,
                          activation="relu",
                          use_bias=True,
                          kernel_initializer="glorot_uniform",
                          bias_initializer="zeros",
                          input_shape=[inputs.shape[1]]),
    tf.keras.layers.Dense(units=len(set(targets)), activation="softmax")
])

klb_net.compile(loss=loss_function, optimizer=optimizer, metrics=["acc"])

klb_model = klb_net.fit(x=x_train,
                        y=y_train_cat,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test,y_test_cat),
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)])

# Comparing the learning curves of the models
plt.plot(cce_model.history['loss'], label='Train Loss', c="blue")
plt.plot(cce_model.history['val_loss'], label='Test Loss', c="orange")
plt.title("Multi-Class Categorical Cross-Entropy Model")
plt.xlabel("Epochs")
plt.ylabel("Categorical Cross-Entropy Loss")
plt.legend()
plt.show()

plt.plot(scce_model.history['loss'], label='Train Loss', c="purple")
plt.plot(scce_model.history['val_loss'], label='Test Loss', c="orange")
plt.title("Sparse Categorical Cross-Entropy Loss Model")
plt.xlabel("Epochs")
plt.ylabel("Sparse Categorial Cross-Entropy Loss")
plt.legend()
plt.show()

plt.plot(klb_model.history['loss'], label='Train Loss', c="green")
plt.plot(klb_model.history['val_loss'], label='Test Loss', c="orange")
plt.title("Kullback Leibler DivergenceLoss Model")
plt.xlabel("Epochs")
plt.ylabel("Kullback Leibler Divergence Loss")
plt.legend()
plt.show()

# Funcion responsible for computing evaluation metrics
def compute_metrics(model, x_test, y_test, thresh):
  """ 
  -----------
  Description
  -----------
    Provided an Tensorflow multi-class classification model object and test data, this function omputes various
    evaluation metrics. Such metrics are both class-specific and model-specific, meaning that metrics like
    True Positive Rate for example, are computed both for each class and for the model as a whole.

  ----------
  Parameters
  ----------
    . model: Tensorflow.python.keras.callbacks.History object.
    . x_test: Test feature set.
    . y_test: Test response set.

  -------
  Outputs
  -------
    . predict: Model predictions in the form of a probability n x k matrix ("n" samples by "k" classes).
    . null_model: Null-Accuracy "dumb" model por comparison.
    . cm_plot: Confusion-Matrix plot of the model. One for each class and one for the whole model.
    . metrics_df: Dataframe containing computed evalutaions metrics. One for each class and one for the whole model.
  """
  
  # Class probabilities matrix
  predict = model.model.predict(x_test, verbose=0, use_multiprocessing=True)*y_test

  # When class imbalance is an issue, maybe use something like this. Or taking q = class proportion
  # thresh = [np.quantile(predict[:,i], q=0.5) if thresh is None else thresh for i in range(predict.shape[1])]

  class_biggest_prob = [max(predict[i,:]) for i in range(predict.shape[0])]
  predict = np.array(
    [1 if predict[i,j] == class_biggest_prob[i] else 0 for i in range(y_test.shape[0]) for j in range(y_test.shape[1])]
  ).reshape(y_test.shape[0], y_test.shape[1])

  tp_by_class = []  # Correctly classified positive labels.
  tn_by_class = []  # Correctly classified negative labels.
  fp_by_class = []  # Type I error. Something that we want to minimize.
  fn_by_class = []  # Type II error. Something that we want to minimize.
  acc_by_class = [] # Overall, how often is the classifier correct ? Must be compared with Null Accuracy.
  pre_by_class = [] # When the actual value is negative, how often is the classifier correct ? (Maximize).
  rec_by_class = [] # When the actual value is postive, how often is the classifier correct ? (Maximize).
  f1_by_class = []  # How efficient is the model when we care evenly for precision an recall ?

  for i in range(y_test.shape[1]):
    # predict * y_test != 0 iff is when both the label and the prediction were 1
    tp_by_class.append(np.count_nonzero((predict[:,i]*y_test[:,i]).reshape(-1,1)))
    # (prediction_integer - 1) * (y_test - 1) != 0 iff is label and the prediction were both 0
    tn_by_class.append(np.count_nonzero(((predict[:,i]-1)*(y_test[:,i]-1)).reshape(-1,1)))
    # If predict * (y_test - 1) != 0, it’s because the prediction was 1, but the label was 0.
    fp_by_class.append(np.count_nonzero((predict[:,i]*(y_test[:,i]-1)).reshape(-1,1)))
    # If (predict - 1) * y_test !=0, it’s because the prediction was 0, but the label was 1.
    fn_by_class.append(np.count_nonzero(((predict[:,i]-1)*y_test[:,i]).reshape(-1,1)))
    acc_by_class.append(
      (np.sum([1 if predict[j,i] == y_test[j,i] else 0 for j in range(predict.shape[0])]))/predict.shape[0]
    )
    pre_by_class.append(np.divide(tp_by_class[i], (tp_by_class[i] + fp_by_class[i])))
    rec_by_class.append(np.divide(tp_by_class[i], (tp_by_class[i] + fn_by_class[i])))
    f1_by_class.append(np.divide((2*pre_by_class[i]*rec_by_class[i]), (pre_by_class[i] + rec_by_class[i])))

  # Metrics dataframe containing named rows, one for each class being the last one full model metrics
  metrics_df = pd.DataFrame(np.reshape([acc_by_class, pre_by_class, rec_by_class, f1_by_class], (y_test.shape[0],4)),
                            columns=["Accuracy","Precision","Recall","F1"])

  # Plots confusion matrix of model
  for i in range(y_test.shape[1]):

    confusion_matrix = pd.DataFrame(np.reshape([tp, fp, fn, tn],(2,2)))
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    sns.heatmap(confusion_matrix, annot=True, annot_kws={"size":16}, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix Plot")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

  ## Think of a way returning everything in a accsesible way