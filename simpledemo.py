#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report


# In[4]:


flags = pd.read_csv('flag.data')
pd.set_option('display.max_columns', None)

flagsOut = flags.drop('Name', axis = 1)

landmass = pd.get_dummies(flagsOut.Landmass)
language = pd.get_dummies(flagsOut.Language)
religion = pd.get_dummies(flagsOut.Religion)
zone = pd.get_dummies(flagsOut.Zone)

flagsOut = flagsOut.drop('Landmass', axis = 1)
flagsOut = flagsOut.drop('Language', axis = 1)
flagsOut = flagsOut.drop('Religion', axis = 1)
flagsOut = flagsOut.drop('Zone', axis = 1)

x = pd.get_dummies(flagsOut.Topleft, prefix='topleft')
y = pd.get_dummies(flagsOut.Botright, prefix = 'botright')
z = pd.get_dummies(flagsOut.Mainhue, prefix = 'mainhue')

flagsOut = flagsOut.drop('Botright', axis=1)
flagsOut = flagsOut.drop('Topleft', axis=1)
flagsOut = flagsOut.drop('Mainhue', axis=1)

flagsOut = flagsOut.join(x)
flagsOut = flagsOut.join(y)
flagsOut = flagsOut.join(z)

scaler = MinMaxScaler()
flagsScaled = scaler.fit_transform(flagsOut)  #flagsScaled is a numpy array
land = landmass.to_numpy()
lang = language.to_numpy()
rlgn = religion.to_numpy()
zn = zone.to_numpy()
flagsOut = flagsOut.drop('Area', axis = 1)
flagsOut = flagsOut.drop('Population', axis = 1)
flagsScaled = scaler.fit_transform(flagsOut)

#Same as using ovo
scaler = StandardScaler()

clf_1 = SVC(kernel='linear')

scaler.fit(flagsOut)

clf_1.fit(scaler.transform(flagsOut), np.asarray(flags['Landmass']))

clf_2 = SVC(kernel='linear')

scaler.fit(flagsOut)

clf_2.fit(scaler.transform(flagsOut), np.asarray(flags['Language']))

clf_3 = SVC(kernel='linear')

scaler.fit(flagsOut)

clf_3.fit(scaler.transform(flagsOut), np.asarray(flags['Religion']))


# In[5]:


input_val = [0,3,5,1,1,0,1,1,1,0,'green',0,0,0,0,1,0,0,1,0,0,'black','green'] # input values we get from html 
if input_val[21]=='black':
    topleft = [1,0,0,0,0,0,0]
elif input_val[21]=='blue':
    topleft = [0,1,0,0,0,0,0]
elif input_val[21]=='gold':
    topleft = [0,0,1,0,0,0,0]
elif input_val[21]=='green':
    topleft = [0,0,0,1,0,0,0]
elif input_val[21]=='orange':
    topleft = [0,0,0,0,1,0,0]
elif input_val[21]=='red':
    topleft = [0,0,0,0,0,1,0]
elif input_val[21]=='white':
    topleft = [0,0,0,0,0,0,1]
if input_val[22] == 'black':
    botright = [1,0,0,0,0,0,0,0]
elif input_val[22] == 'blue':
    botright = [0,1,0,0,0,0,0,0]
elif input_val[22] == 'brown':
    botright = [0,0,1,0,0,0,0,0]
elif input_val[22] == 'gold':
    botright = [0,0,0,1,0,0,0,0]
elif input_val[22] == 'green':
    botright = [0,0,0,0,1,0,0,0]
elif input_val[22] == 'orange':
    botright = [0,0,0,0,0,1,0,0]
elif input_val[22] == 'red':
    botright = [0,0,0,0,0,0,1,0]
elif input_val[22] == 'white':
    botright = [0,0,0,0,0,0,0,1]
if input_val[10]=='black':
    mainhue = [1,0,0,0,0,0,0,0]
elif input_val[10]=='blue':
    mainhue = [0,1,0,0,0,0,0,0]
elif input_val[10]=='brown':
    mainhue = [0,0,1,0,0,0,0,0]
elif input_val[10]=='gold':
    mainhue = [0,0,0,1,0,0,0,0]
elif input_val[10]=='green':
    mainhue = [0,0,0,0,1,0,0,0]
elif input_val[10]=='orange':
    mainhue = [0,0,0,0,0,1,0,0]
elif input_val[10]=='red':
    mainhue = [0,0,0,0,0,0,1,0]
elif input_val[10]=='white':
    mainhue = [0,0,0,0,0,0,0,1]

indices = [10,21,22]
for i in sorted(indices, reverse=True):
    input_val = np.delete(input_val, i)
all_input = np.concatenate((input_val,topleft,botright, mainhue), axis=None)
df = pd.DataFrame(all_input).T
df = scaler.transform(df)

print(clf_1.predict(df),clf_2.predict(df),clf_3.predict(df))

