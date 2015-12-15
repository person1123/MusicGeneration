import sys
import numpy as np
import scikits.audiolab as audiolab
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.utils import np_utils, generic_utils

f = audiolab.Sndfile('05 Woodstock.aif', 'r')

seq_len = 10
buckets = 2048
hidden_layer_size = 256
batch_size = 128
np_epoch = 10
output_n_frames = f.samplerate*10
data_frames_to_use = 9

data = ((f.read_frames(f.samplerate*15)[:,0] + 1.0)*(buckets/2))
data = data.astype(int)
sampling_rate = f.samplerate
nframes = f.nframes
output_filename = 'testshorterm' + str(data_frames_to_use) + '.wav'

f.close()

model = Sequential()
model.add(Embedding(buckets, hidden_layer_size, input_length=seq_len))
model.add(LSTM(output_dim=hidden_layer_size, activation="sigmoid", inner_activation="hard_sigmoid", init="uniform"))
model.add(Dense(input_dim=hidden_layer_size, output_dim=buckets, init="glorot_uniform"))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
 
model.load_weights("test2.hdf")

frames = np.empty(output_n_frames, dtype=int)

frames[0:seq_len] = data[0:seq_len]

max_i = output_n_frames-seq_len
for i in range(max_i):
	sys.stdout.write("Status %d/%d           \r" % (i+1,max_i))
	sys.stdout.flush()
	seed = np.empty(seq_len, dtype=int)
	seed[0:data_frames_to_use] = data[i:i+data_frames_to_use]
	seed[data_frames_to_use:seq_len] = frames[i+data_frames_to_use:i+seq_len]
	frames[i+seq_len] = model.predict_classes(np.array([seed]), batch_size=1, verbose=0)

sys.stdout.write("Status Saving file...          \r")
frames = frames/(buckets/2.0) - 1.0
print frames

f2 = audiolab.Sndfile(output_filename, 'w', audiolab.Format('wav'), 1, sampling_rate)

f2.write_frames(frames)
f2.close()

sys.stdout.write("Done!                                    \n")
