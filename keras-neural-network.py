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

data = ((f.read_frames(f.nframes)[0:44100*5,0] + 1.0)*(buckets/2))
data = data.astype(int)
sampling_rate = f.samplerate
nframes = f.nframes

f.close()

model = Sequential()
model.add(Embedding(buckets, hidden_layer_size, input_length=seq_len))
model.add(LSTM(output_dim=hidden_layer_size, activation="sigmoid", inner_activation="hard_sigmoid", init="uniform"))
model.add(Dense(input_dim=hidden_layer_size, output_dim=buckets, init="glorot_uniform"))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
loss = [0.0]
 
for epoch in range(np_epoch):
	max_h = len(data)/batch_size
	for h in range((len(data)-seq_len-1)/batch_size):
		sys.stdout.write("Epoch %d/%d step %d/%d loss = %f   \r" % (epoch+1,np_epoch,h+1,max_h, loss[0]))
		sys.stdout.flush()

		train_x = np.empty([batch_size, seq_len], dtype=int)
		train_y = np.zeros([batch_size, buckets], dtype=int)

		for i in range(batch_size):
        		train_x[i] = data[batch_size*h+i:batch_size*h+i+seq_len]
        		train_y[i][data[batch_size*h+i+seq_len]] = 1

		loss = model.train_on_batch(train_x, train_y)

	sys.stdout.write("Epoch %d/%d saving weights!        \r" % (epoch+1,np_epoch))
	model.save_weights("test2.hdf", overwrite=True)
	sys.stdout.write("Epoch %d/%d done!            \r" % (epoch+1,np_epoch))

#model.load_weights("test.hdf")

classes = model.predict_classes(train_x)
print classes

classes = classes/(buckets/2.0) - 1.0
print classes

f2 = audiolab.Sndfile('output.wav', 'w', audiolab.Format('wav'), 1, len(classes))

f2.write_frames(classes)
f2.close()

