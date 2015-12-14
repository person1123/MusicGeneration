import numpy as np
import scikits.audiolab as audiolab
import random
import json
import pickle
import argparse


# Best values so far: for chain_depth 4, bin size .004 is 100% uniq, any less is 64% unique
# for chain depth 5, bin size .004 is 90% unique
# chain depth 5, 

def train(filepath, chain, chain_depth, bin_size):
    global samplerate

    chain_depth = chain_depth or 2
    bin_size = bin_size or 0.00001
    
    f = audiolab.Sndfile(filepath, 'r')

    samplerate = f.samplerate

    if chain is not None and chain["samplerate"] != samplerate:
        print "Sample rate mismatch"
        return
    
    data = f.read_frames(f.nframes)
    data = data[30 * f.samplerate:45 * f.samplerate]

    def start_val(i):   # An arbitrary unique value representing i spaces before the start of the audio that doesn't interfere with the audio values
                    # Since these are generally in the range (-1, 1)
        return -1-i

    if chain is None:
        chain = {}  # The actual Markov chain
        chain["samplerate"] = samplerate
        chain["chain_depth"] = chain_depth
        chain["bin_size"] = bin_size
    
    uniq = []   # How many unique frequencies are there? This is to figure out whether we need to bucket similar frequencies in order to make a good Markov chain
    tc = chain

    num_buckets = 2 / bin_size + 1 # The total number of buckets, including one special bucket for before the start of the audio.
    bucket_index = 0 # The bucket index can be viewed as a base-num_buckets number with chain depth digits. The nth rightmost digit (counting the ones place as the first) is the bucket selected for the frequency n places back in the audio.
    for i in range(chain_depth): # Sets all digits of the bucket index equal to the special value
	   bucket_index = bucket_index * num_buckets + num_buckets - 1
    
    num_uniq = 0

    i = 0
    total_endpoints = 0
    num_categories = 0
    for i in range(len(data)):
        if i % (5 * f.samplerate) == 0:   # Print progress every 5 seconds
            print i / f.samplerate
            if i != 0:
                print "un: " + str(num_uniq * 1.0 / i)

        tc = chain
        if bucket_index not in tc:
            tc[bucket_index] = {}

        tc = tc[bucket_index]

        # Compute the bucket_index for the next iteration
        a = data[i][0] 
        val = (a - a % bin_size) / bin_size + 1 / bin_size
        bucket_index = int((bucket_index * num_buckets + val) % (num_buckets ** chain_depth)) # Shifts all digits of the bucket_index to the left, cuts off the highest digit, and adds a new rightmost digit.

        if data[i][0] not in tc:
            tc[data[i][0]] = 1
            total_endpoints += 1
            num_uniq += 1
        else:
            if tc[data[i][0]] == 1:
                num_uniq -= 1
            tc[data[i][0]] += 1
        if "sum" not in tc:
            tc["sum"] = 1
            num_categories += 1.0
        else:
            tc["sum"] += 1

        if a not in uniq:
            uniq.append(a)

    print "unique: " +str(num_uniq * 1.0 / len(data))
    return chain

def write(chain, filepath):
    with open(filepath, 'w') as outfile:
        pickle.dump(chain,outfile)

def read(filename):
    with open(filename) as infile:
        chain = pickle.load(infile)
        return chain
    return None
    # chain = simplemarkov.train("05 Woodstock.aif", None, None, None)
    # simplemarkov.gen(chain, "tsttst.wav", None, None)
        
def train_and_write(args):#filepath, chainpath, chain_depth = 2, bin_size = 0.00001):
    try:
        chain = read(args.chainpath)
    except Exception:
        chain = None
    chain = train(args.filepath, chain, args.chain_depth, args.bin_size)
    write(chain, args.chainpath)

def read_and_gen(args):#chainpath, filepath):
    chain = read(args.chainpath)
    print args.chainpath
    gen(chain,args.filepath)
"""
print "Average endpoints: " + str(total_endpoints / num_categories)
print "TE: " + str(total_endpoints) + " NC: " + str(num_categories)
print str(len(uniq)) + " unique out of " + str(len(data))
"""

def gen(chain,filepath):
    chain_depth = chain["chain_depth"]
    bin_size = chain["bin_size"]
    # Reset the bucket index to its starting value
    bucket_index = 0

    num_buckets = 2 / bin_size + 1 # The total number of buckets, including one special bucket for before the start of the audio.
    for i in range(chain_depth):
	   bucket_index = bucket_index * num_buckets + num_buckets - 1

    new_data = np.empty([chain["samplerate"] * 15])   # 15 seconds of sound
    try:
        for i in range(new_data.size):
            if i % (5 * chain["samplerate"]) == 0:   # Print progress every 5 seconds
                print i / chain["samplerate"]
            tc = chain[bucket_index]
            new_datum = 0
        
            #print tc
            weighted_index = random.randrange(tc["sum"])
            for k in tc.keys():
                if k not in ["sum", "samplerate", "bin_size", "chain_depth"]:
                    if weighted_index < tc[k]:
                        new_datum = k
                        break
                    else:
                        weighted_index -= tc[k]
            new_data[i] = new_datum
            #print new_data[0:i]

            # Compute the bucket_index for the next iteration
            val = (new_data[i] - new_data[i] % bin_size) / bin_size + 1 / bin_size
            print val
            bucket_index = int((bucket_index * num_buckets + val) % (num_buckets ** chain_depth))
    finally:
        format = audiolab.Format('wav')
        f3 = audiolab.Sndfile(filepath, 'w', format, 1, chain["samplerate"])
        f3.write_frames(new_data)
        f3.close()

#audiolab.play(data)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument("filepath")
    parser_train.add_argument("chainpath")
    parser_train.add_argument("-chain_depth",type=int)
    parser_train.add_argument("-bin_size",type=float)
    parser_train.set_defaults(func=train_and_write)

    parser_gen = subparsers.add_parser('gen')
    parser_gen.add_argument("chainpath")
    parser_gen.add_argument("filepath")
    parser_gen.set_defaults(func=read_and_gen)
    
    args = parser.parse_args()
    print args
    args.func(args)
    
