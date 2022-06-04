from preprocessing import getData 

import pandas as pd
import numpy as np
from keras.models import Sequential
# from tensorflow.python.keras.models import Sequential
from keras.layers import Dense, Dropout
# from tensorflow.python.keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def landNN():
  _, _, flagsScaled, land, _, _ = getData()

  model = Sequential()

  layers = [
    Dense(20, activation='relu', name='layer1'),
    Dense(10, activation='relu', name='layer2'),
    Dense(6, activation='sigmoid', name='Output')
  ]

  for layer in layers:
    model.add(layer)

  model.compile(optimizer=SGD(learning_rate=.01), loss='mse', metrics=['accuracy'])
  hist = model.fit(flagsScaled, land, validation_split=.2, epochs=500, verbose=0)

  flagsTrain, flagsTest, landTrain, landTest = train_test_split(flagsScaled, land, test_size = 0.20, random_state=5)

  pred_train= model.predict(flagsTrain)
  land_train_score, land_train_acc = model.evaluate(flagsTrain, landTrain, verbose=0)
  # print('Training Error: %.3f' % score)
  # print('Training Accuracy: %.3f' % acc)

  pred_train= model.predict(flagsTest)
  land_test_score, land_test_acc = model.evaluate(flagsTest, landTest, verbose=0)
  # print('Test Error: %.3f' % score)
  # print('Test Accuracy: %.3f' % acc)

  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.title('Model Loss Curve')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['train', 'test'], loc='upper right')

  plt.savefig('static/images/land-loss-plot.png')
  plt.clf()

  plt.plot(hist.history['accuracy'])
  plt.plot(hist.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epochs')
  plt.legend(['train', 'test'], loc='upper left')

  plt.savefig('static/images/land-acc-plot.png')
  plt.clf()
  
  def createmodel(n_layers, first_layer_nodes, second_layer_nodes, learning_rate=.1):
    model = Sequential()
    
    #n_nodes = FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes)
    for i in range(2):
      if i==1:
        model.add(Dense(first_layer_nodes, activation='relu'))
      else:
        model.add(Dense(second_layer_nodes, activation='relu'))
            
    #Finally, the output layer should have a single node in binary classification
    model.add(Dense(6, activation='sigmoid'))
    model.compile(optimizer=SGD(learning_rate=learning_rate), 
      loss='mse', metrics = ["accuracy"]) #note: metrics could also be 'mse'
    
    return model

  param_grid = dict(n_layers=[2], first_layer_nodes = [20, 10, 5], second_layer_nodes=[20,10,5],
    epochs = [150, 300, 500], learning_rate= [.01, .15, .3, .5])

  model = KerasClassifier(build_fn=createmodel, verbose=0)
  grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
  grid_result = grid.fit(flagsScaled, land)

  # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  # means = grid_result.cv_results_['mean_test_score']
  # stds = grid_result.cv_results_['std_test_score']
  # params = grid_result.cv_results_['params']
  # for mean, stdev, param in zip(means, stds, params):
    # print("%f (%f) with: %r" % (mean, stdev, param))

  return land_train_score, land_train_acc, land_test_score, land_test_acc, grid_result.best_score_, grid_result.best_params_
  # return land_train_score, land_train_acc, land_test_score, land_test_acc

def landDT():
  _, _, flagsScaled, land, _, _ = getData()

  flagsTrain, flagsTest, landTrain, landTest = train_test_split(flagsScaled, land, test_size = 0.20, random_state=5)
  tree_model = DecisionTreeClassifier().fit(flagsTrain, landTrain)
  tree_pred = tree_model.predict(flagsTest)

  land_DT_acc = accuracy_score(landTest, tree_pred)
  land_DT_class_report = pd.DataFrame(classification_report(landTest, tree_pred, output_dict=True)).transpose()
  land_DT_class_report.to_csv('land_DT_class_report.csv', index=True)
  land_DT_conf_matrix = confusion_matrix(np.argmax(landTest, axis=1), np.argmax(tree_pred, axis=1))
  return land_DT_acc, land_DT_class_report, land_DT_conf_matrix

def landSVM():
  flags, flagsOut, flagsScaled, land, _, _ = getData()
  flagsTrain, flagsTest, landTrain, landTest = train_test_split(flagsScaled, land, test_size = 0.20, random_state=5)

  scaler = StandardScaler()
  clf_1 = SVC(kernel='linear')
  scaler.fit(flagsOut)
  clf_1.fit(scaler.transform(flagsOut), np.asarray(flags['Landmass']))

  land_SVM_class_report = pd.DataFrame(classification_report(np.argmax(landTest, axis=1)+1, clf_1.predict(flagsTest), output_dict=True)).transpose()
  land_SVM_class_report.to_csv('land_SVM_class_report.csv', index=True)

  return land_SVM_class_report