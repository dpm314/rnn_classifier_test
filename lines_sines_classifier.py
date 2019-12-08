import matplotlib.pyplot as plt
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
#%%
def plotClass1Class2(c1,c2):
    plt.figure()
    for x,y in zip(c1, c2):
        plt.plot(np.linspace(0.0, 1.0, len(x)), x, 'r')
        plt.plot(np.linspace(0.0, 1.0, len(y)), y, 'b')
    plt.xlabel('Time', fontsize = 16)
    plt.ylabel('Amplitude', fontsize = 16)
#from keras.layers.embeddings import Embedding
#from keras.preprocessing import sequence
#%%
def makeLines(num_batches, sample_length, offset_minmax, slope_minmax, noiseAmp = 0.0):
    #input is [batch_size, x (y, z ...)]
    offsets = np.random.uniform(offset_minmax[0], offset_minmax[1], num_batches)
    slopes  = np.random.uniform(slope_minmax[0],  slope_minmax[1] , num_batches)
    x = np.zeros( [sample_length, num_batches] )
    for i in range(num_batches):
        x[:,i] = slopes[i]*( np.linspace(0.0, 1.0, sample_length) - .5 ) + offsets[i] + noiseAmp * np.random.random(sample_length)
    return x
#%%
def makeSines(num_batches, sample_length, offset_minmax, amplitude_minmax, omega_minmax, noiseAmp = 0.0):
    #assume t_final is num_batches/(2*np.pi) so omega = 1 is one complete cycles
    #hardcoding random phase offsets evenly distributed from 0 to 2pi
    t = np.linspace(0, 2.0*np.pi, sample_length)
    offsets       = np.random.uniform(offset_minmax[0],     offset_minmax[1],     num_batches)
    amplitudes    = .5*np.random.uniform(amplitude_minmax[0],  amplitude_minmax[1],  num_batches)
    omegas        = np.random.uniform(omega_minmax[0],      omega_minmax[1],      num_batches)
    phase_offsets = np.linspace(0, 2.0*np.pi, num_batches)
    
    x = np.zeros( [sample_length, num_batches] )
    for i in range(num_batches):
        x[:,i] = amplitudes[i]*np.sin(t*omegas[i] + phase_offsets[i]) + offsets[i] + noiseAmp * np.random.random(sample_length)
    return x
#%%
num_batches = 25
sample_length = 100
c1_noise_amplitude = 0.0
c2_noise_amplitude = 0.0
#right now same number of test and train data
#no need to randomly splice them since no coherance across index of the 'batch'
c1 = makeLines(num_batches, sample_length,  (-1.0, 1.0), (-1.0,1.0),              c1_noise_amplitude)
c2 = makeSines(num_batches, sample_length, (-1.0, 1.0), (-.250,.250), (.5, 10.0), c2_noise_amplitude)
#%%
labels = np.zeros([2*num_batches], dtype = np.int16)
labels[:num_batches] = 1
np.random.shuffle(labels)
#%%
c1_inds = np.where(labels==0)[0]
c2_inds = np.where(labels==1)[0]
#%%
#labels = np.reshape(labels,[2*num_batches,1])#need extra column for 'features'
#%%
X_train = np.zeros([num_batches*2, sample_length,1])
X_train[c1_inds,:,0] = c1
X_train[c2_inds,:,0] = c2
X_train = X_train.swapaxes(1,0)
#%%
plotClass1Class2(c1,c2)


#%%
model = Sequential()
#return_sequences=True
model.add(LSTM(128, input_shape = (num_batches*2,1)))   

model.add(Dropout(.1))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, labels, epochs=3)
#%%
test_c1 makeSines(num_batches, sample_length, (-1.0, 1.0), (-.250,.250), (.01, 10.0), c2_noise_amplitude)
model.predict(test_c1.reshape(num_batches, sample_length, 1))