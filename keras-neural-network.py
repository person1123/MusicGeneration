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

data = (f.read_frames(f.nframes)[:,0] + 1.0)*(buckets/2)
sampling_rate = f.samplerate
nframes = f.nframes

f.close()

all_x = []
all_y = []

for i in range(44100):
	train_x = data[i*seq_len:i*seq_len+seq_len].astype(int)
	train_y = data[i*seq_len+seq_len].astype(int)

	#categorical_x = []
	#for x in train_x:
	#	categorical = np.zeros(buckets)
	#	categorical[x] = 1
	#	categorical_x.append(categorical)

	categorical_y = np.zeros(buckets)
	categorical_y[train_y] = 1

	#all_x.append(np.array(categorical_x))
	all_y.append(categorical_y)
	all_x.append(train_x)
	#all_y.append(train_y)

train_x = np.array(all_x)
train_y = np.array(all_y)

model = Sequential()
model.add(Embedding(buckets, hidden_layer_size, input_length=seq_len))
model.add(LSTM(output_dim=hidden_layer_size, activation="sigmoid", inner_activation="hard_sigmoid", init="uniform"))
model.add(Dense(input_dim=hidden_layer_size, output_dim=buckets, init="glorot_uniform"))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

#model.fit(train_x, train_y, nb_epoch=1000)


#model.save_weights("test.hdf", overwrite=True)
model.load_weights("test.hdf")

classes = model.predict_classes(train_x)
print classes

classes = classes/(buckets/2.0) - 1.0
print classes

f2 = audiolab.Sndfile('output.wav', 'w', audiolab.Format('wav'), 1, len(classes))

f2.write_frames(classes)
f2.close()

