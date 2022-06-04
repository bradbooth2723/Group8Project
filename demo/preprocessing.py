import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def getData():
  flags = pd.read_csv('../../Group8Project/flag(1).data', sep=',')

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

  #flagsOut = flagsOut.join(w)
  flagsOut = flagsOut.join(x)
  flagsOut = flagsOut.join(y)
  flagsOut = flagsOut.join(z)

  scaler = MinMaxScaler()
  flagsScaled = scaler.fit_transform(flagsOut)  #flagsScaled is a numpy array
  land = landmass.to_numpy()
  lang = language.to_numpy()
  rlgn = religion.to_numpy()
  zn = zone.to_numpy()

  return flags, flagsOut, flagsScaled, land, lang, rlgn