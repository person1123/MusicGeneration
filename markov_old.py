import numpy as np
import scikits.audiolab as audiolab
import random

# data, sampling_rate, encoding = aiffread('Woodstock.aif')

f = audiolab.Sndfile('05 Woodstock.aif', 'r')

sampling_rate = f.samplerate
channels = f.channels
encoding = f.encoding
#format = f.format

format = audiolab.Format('wav')

data = f.read_frames(f.nframes)

data = data[30 * f.samplerate:45 * f.samplerate]

f2 = audiolab.Sndfile('copy.wav', 'w', format, channels, f.samplerate)

f2.write_frames(data)
f2.close()

print data

#data = [[1, 0], [2, 0], [3, 0], [2, 0], [1, 0]]

chain_depth = 2
bin_size = 0.00001
def start_val(i):   # An arbitrary unique value representing i spaces before the start of the audio that doesn't interfere with the audio values
                    # Since these are generally in the range (-1, 1)
    return -1-i

chain = {}  # The actual Markov chain
uniq = []   # How many unique frequencies are there? This is to figure out whether we need to bucket similar frequencies in order to make a good Markov chain
tc = chain

num_buckets = 2 / bin_size + 1 # The total number of buckets, including one special bucket for before the start of the audio.
bucket_index = 0 # The bucket index can be viewed as a base-num_buckets number with chain depth digits. The nth rightmost digit (counting the ones place as the first) is the bucket selected for the frequency n places back in the audio.
for i in range(chain_depth): # Sets all digits of the bucket index equal to the special value
	bucket_index = bucket_index * num_buckets + num_buckets - 1

i = 0
total_endpoints = 0
num_categories = 0
for i in range(len(data)):   
    if i % (5 * f.samplerate) == 0:   # Print progress every 5 seconds
        print i / f.samplerate

    tc = chain
    if bucket_index not in tc:
        tc[bucket_index] = {}

    tc = tc[bucket_index]

    # Compute the bucket_index for the next iteration
    a = data[i][0] 
    val = (a - a % bin_size) / bin_size + 1 / bin_size   
    bucket_index = (bucket_index * num_buckets + val) % (num_buckets ** chain_depth) # Shifts all digits of the bucket_index to the left, cuts off the highest digit, and adds a new rightmost digit.

    if data[i][0] not in tc:
        tc[data[i][0]] = 1
        total_endpoints += 1
    else:
        tc[data[i][0]] += 1
    if "sum" not in tc:
        tc["sum"] = 1
        num_categories += 1.0
    else:
        tc["sum"] += 1

    if a not in uniq:
        uniq.append(a)

print "Average endpoints: " + str(total_endpoints / num_categories)
print "TE: " + str(total_endpoints) + " NC: " + str(num_categories)
print str(len(uniq)) + " unique out of " + str(len(data))

# Reset the bucket index to its starting value
bucket_index = 0
for i in range(chain_depth):
	bucket_index = bucket_index * num_buckets + num_buckets - 1

new_data = np.empty([f.samplerate * 15])
for i in range(new_data.size):
    if i % (5 * f.samplerate) == 0:   # Print progress every 5 seconds
        print i / f.samplerate
    tc = chain[bucket_index]
    new_datum = 0
    
    #print tc
    weighted_index = random.randrange(tc["sum"])
    for k in tc.keys():
        if k is not "sum":
            if weighted_index < tc[k]:
                new_datum = k
                break
            else:
                weighted_index -= tc[k]
    new_data[i] = new_datum
    #print new_data[0:i]

    # Compute the bucket_index for the next iteration
    val = (new_data[i] - new_data[i] % bin_size) / bin_size + 1 / bin_size
    bucket_index = (bucket_index * num_buckets + val) % (num_buckets ** chain_depth)


f3 = audiolab.Sndfile('gen.wav', 'w', format, 1, f.samplerate)
f3.write_frames(new_data)
f3.close()

#audiolab.play(data)