# Date: 19/05/2020                                                   
# Author: Pedro H. Puntel                                            
# Email: pedro.puntel@gmail.com                                      
# Topic: 365 Data Science Course - Deep Learning With Tensorflow 2.0 
# Ecoding: UTF-8                                                     

# References:
# . https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
# . https://michaelblogscode.wordpress.com/2017/12/20/visualizing-model-performance-statistics-with-tensorflow/
# . https://www.ritchieng.com/machine-learning-evaluate-classification-model/

# Modules
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.4)

# Random data generation
from sklearn.datasets import make_circles
inputs, targets = make_circles(n_samples=10000, shuffle=True, noise=0.05, random_state=42)

# Data preview
for i in range(2):
  samples_ix = np.where(targets == i)
  plt.scatter(inputs[samples_ix, 0], inputs[samples_ix, 1], label=str(i))
plt.title("Binary Classification Example - Random Circles")
plt.legend()
plt.show()

# Binary Cross-Entropy
# . Default loss function for most binary classification problems.
# . Intended for use when the binary labels are in the set {0,1}.
# . Like MSE, its the preffered loss function under maximum likelihood estimation.
# . Calculates the average difference between the actual and predicted probabilty
#   distribution for predicting class 1. A perfect cross-entropy value is 0.

from sklearn.model_selection import train_test_split
x_test, x_train, y_test, y_train = train_test_split(inputs, targets, test_size=0.8, random_state=42)

epochs = 1000
learning_rate = 0.01
loss_function = "binary_crossentropy"
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

bc_net = tf.keras.Sequential([
  tf.keras.layers.Dense(units=50,
                        activation="relu",
                        use_bias=True,
                        kernel_initializer="glorot_uniform",
                        bias_initializer="zeros",
                        input_shape=[inputs.shape[1]]),
  tf.keras.layers.Dense(units=1, activation="sigmoid")])

bc_net.compile(loss=loss_function, optimizer=optimizer, metrics=["acc"])

bc_model = bc_net.fit(x=x_train,
                      y=y_train,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(x_test,y_test),
                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)])

# Hinge Loss
# . Alternative to the binary cross-entropy loss function, developed for SVM models.
# . Inteded for use when the labels are for set {-1,1}.
# . Encourages samples to have the correct sign, penalizing greater for differences in signs.
# . Requires an modification to the activation function due to the change in domain.
# . Reports on its performance are somewhat mixed, it may perform worse the cross-entropy.

hinge_y_train = np.copy(y_train)
hinge_y_test = np.copy(y_test)
hinge_y_train[np.where(hinge_y_train == 0)] = -1
hinge_y_test[np.where(hinge_y_test == 0)] = -1

epochs = 1000
learning_rate = 0.01
loss_function = "hinge"
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

hinge_net = tf.keras.Sequential([
  tf.keras.layers.Dense(units=50,
                        activation="relu",
                        use_bias=True,
                        kernel_initializer="glorot_uniform",
                        bias_initializer="zeros",
                        input_dim=inputs.shape[1]),
  tf.keras.layers.Dense(units=1, activation="tanh")])

hinge_net.compile(loss=loss_function, optimizer=optimizer, metrics=["acc"])

hinge_model = hinge_net.fit(x=x_train,
                            y=hinge_y_train,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, hinge_y_test),
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)])

# Squared Hinge Loss
# . The Hinge loss functions has many extensions, one of which is the Squared Hinge Loss.
# . Has the effect of smoothing out the error function and makes it numerically more stable.
# . At times, if the Hinge Loss functions does not perform well, Squared Hinge Loss might.

epochs = 1000
learning_rate = 0.01
loss_function = "squared_hinge"
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

sqr_hinge_net = tf.keras.Sequential([
  tf.keras.layers.Dense(units=50,
                        activation="relu",
                        use_bias=True,
                        kernel_initializer="glorot_uniform",
                        bias_initializer="zeros",
                        input_dim=inputs.shape[1]),
  tf.keras.layers.Dense(units=1, activation="tanh")])

sqr_hinge_net.compile(loss=loss_function, optimizer=optimizer, metrics=["acc"])

sqr_hinge_model = sqr_hinge_net.fit(x=x_train,
                                    y=hinge_y_train,
                                    epochs=epochs,
                                    verbose=1,
                                    validation_data=(x_test, hinge_y_test),
                                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)])

# Comparing the learning curves of the models
plt.plot(bc_model.history['loss'], label='Train Loss', c="blue")
plt.plot(bc_model.history['val_loss'], label='Test Loss', c="orange")
plt.title("Binary Cross-Entropy Model")
plt.xlabel("Epochs")
plt.ylabel("BCE Loss")
plt.legend()
plt.show()

plt.plot(hinge_model.history['loss'], label='Train Loss', c="purple")
plt.plot(hinge_model.history['val_loss'], label='Test Loss', c="orange")
plt.title("Hinge Loss Model")
plt.xlabel("Epochs")
plt.ylabel("Hinge Loss")
plt.legend()
plt.show()

plt.plot(sqr_hinge_model.history['loss'], label='Train Loss', c="green")
plt.plot(sqr_hinge_model.history['val_loss'], label='Test Lss', c="orange")
plt.title("Squared Hinge Loss Model")
plt.xlabel("Epochs")
plt.ylabel("Square Hinge Loss")
plt.legend()
plt.show()

## Update!!!
# Funcion responsible for computing evaluation metrics
def compute_metrics(model, test_data, test_targets):
  """ 
  -----------
  Description
  -----------
    Given an Binary classification model, computes various model evaluation metrics.

  ----------
  Parameters
  ----------
    . model: Tensorflow.python.keras.callbacks.History object
    . test_data: Model test set.
    . test_targets: Model test targets.

  -------
  Outputs
  -------
    . predictions: Model predictions.
    . baseline_model: Associated Null-Accuracy "dumb" model.
    . confusion_matrix: Confusion-Matrix plot of the model.
    . metrics_df: Dataframe containing all the computed evalutaions metrics.
  """

  predictions = model.model.predict(test_data, verbose=0, use_multiprocessing=True)
  
  ## Null model = accuracy that would be achieved by predicting the MOST FREQUENT CLASS!!
  if any(test_targets == -1):
    baseline_model = len(test_targets[np.where(test_targets == -1)])/len(test_targets)
    test_targets[np.where(test_targets == -1)] = 0 # relabel the data for easier manipulation
  else:
    baseline_model = len(test_targets[np.where(test_targets == 0)])/len(test_targets)
  
  tp = np.count_nonzero(predictions * test_targets)
  tn = np.count_nonzero(((predictions-1)*(test_targets-1)))
  fp = np.count_nonzero(predictions * (test_targets-1))
  fn = np.count_nonzero((predictions-1)*test_targets)
  _, accuracy = model.model.evaluate(test_data, test_targets, verbose=0)
  precision = np.divide(tp, (tp + fp))
  recall = np.divide(tp, (tp + fn))
  f1_score = np.divide((2*precision*recall), (precision + recall))
  metrics_df = pd.DataFrame(np.reshape([accuracy, precision, recall, f1_score],(1,4)),
                            columns=["Accuracy","Precision","Recall","F1"])

  confusion_matrix = pd.DataFrame(np.reshape([tp, fp, fn, tn],(2,2)))
  confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
  sns.heatmap(confusion_matrix, annot=True, annot_kws={"size":16}, cmap=plt.cm.Blues)
  plt.title("Confusion Matrix Plot")
  plt.xlabel("Predicted Labels")
  plt.ylabel("True Labels")
  plt.show()

  return predictions, baseline_model, metrics_df

# Comparing the models
bc_pred, bc_dumb, bc_metrics = compute_metrics(bc_model, x_test, y_test)
hinge_pred, hinge_dumb, hinge_metrics = compute_metrics(hinge_model, x_test, hinge_y_test)
sqrhinge_pred, sqrhinge_dumb, sqrhinge_metrics = compute_metrics(sqr_hinge_model, x_test, hinge_y_test)