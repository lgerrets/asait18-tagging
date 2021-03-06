import numpy as np
import os
import scipy
from keras.utils import Sequence
import librosa 
from parameters import *

import argparse
parser = argparse.ArgumentParser(description='Preprocess the data into mel features and binary targets.')
parser.add_argument('--data', type=str, default="data", metavar='DA',
                    help='path to the data folder (where the *.csv and *.wav files are located) (default: data)')
parser.add_argument('--fs', type=int, default=16000, metavar='FS',
                    help='sampling rate in Hz (default: 16000)')
parser.add_argument('--frame', type=int, default=20, metavar='FR',
                    help='frame length in ms (default: 20)')
parser.add_argument('--hop', type=int, default=10, metavar='HOP',
                    help='frame hop size in ms (default: 10)')
# These default parameters are the ones used in the original paper. With these, the size of resulting numpy files
# created by `create_mfb_features` are ~50% of the size of the raw data, and the arrays are ready to be fed into the neural nets.
# Computing these features once and for all to save them in file for later reuse, should be more time efficient than
# re-computing them at every neural net training experiment (ie `python train.py`).

def create_mfb_features(data_path,fs,frame_ms,hop_ms):
	"""Converts raw data files (wav and csv) to numpy arrays of mfb features and tags.
	Inputs: (documentation given by argparse above, run `python prepare_data.py --help`)
		data_path (str)
		fs (float)
		frame_ms (float)
		hop_ms (float)
	Outputs:
		None
	This function loads the files: there is 1 chunk for each pair of wav/csv file.
	The wav sound is divided into frames which are converted to Mel features.
	The csv contains the tag(s) associated to the chunk, and therefore to all the subsequent frames.
	Numpy arrays of (1) mel features and (2) tags are stored in the `data_path` directory as *.npy.
	"""

	frame_length = int(fs*frame_ms/1000)
	frame_step = int(fs*hop_ms/1000)
	NFFT = frame_length # either frame_length or next power of 2
	num_mel_feat = 40

	all_mfb_data = []
	all_y_data = []

	# append data: mel features (X) and tags (y)
	num_chunks = 0
	valid_chunks = 0
	for file in os.listdir(data_path): # loop for every file in given path
		file = os.path.join(data_path,file)

		if not os.path.isfile(file):
			continue

		if not file[-7:] == "kHz.wav":
			continue

		num_chunks += 1

		chunk_is_valid = True

		# at this point we assume that file[-10:]+".csv" also exists; it is needed because it contains the tags

		### tags

		# load the set of tags associated to the previous chunk
		y_data = np.genfromtxt(file[:-10]+'.csv',delimiter=',',dtype=str)
		y_data = y_data[loc_targets_in_file[0],loc_targets_in_file[1]] # string of tags from the 7 possible tags: bcfmopv

		# convert into a binary vector
		targets = np.zeros(n_classes)
		for c in y_data:
			if c in ind_to_tag:
				targets[tag_to_ind[c]] = 1 
			else:
				# print("Unexpected value "+str(c)+" in "+str(y_data))
				chunk_is_valid = False

		if np.sum(targets) < 1:
			chunk_is_valid = False

		if not chunk_is_valid:
			continue # tags are not valid so we skip this chunk
		else:
			valid_chunks += 1

		targets = targets.astype(np.int32)

		# append this data
		all_y_data.append(targets)

		### mel features

		# load the chunk of sound
		rate,x_data = scipy.io.wavfile.read(file)
		x_data = x_data.astype(np.float32)
		assert rate==fs, "Expected " + str(rate) + " but got " + str(fs)
		num_frames = int((len(x_data)-frame_length)/frame_step)+1
		
		#assert num_frames == 399, num_frames

		# divide this chunk into num_frames frames 
		indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
		indices = indices.astype(np.int32,copy=False)
		frames = x_data[indices] # shape (num_frames,frame_length)

		# apply short term fourier transform with hamming window
		frames *= np.hamming(frame_length)
		mag_frames = np.abs(np.fft.rfft(frames, NFFT))  # (num_frames,1+nfft/2 ?) Magnitude of the FFT
		pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
		log_frames = np.log(pow_frames)

		# build the mel features
		Mfb = librosa.filters.mel(fs, NFFT, n_mels=num_mel_feat) # (num_mel_feat,1+nfft/2)
		mfb_frames = log_frames.dot(Mfb.T) # (num_frames,num_mel_feat)
		mfb_frames = mfb_frames.astype(np.float32)

		# append this data
		all_mfb_data.append(mfb_frames)

	print("Got %d/%d valid chunks"%(valid_chunks,num_chunks))
	np.save(os.path.join(data_path,file_feat),all_mfb_data) # (num_chunks,num_frames,num_mel_feat)
	np.save(os.path.join(data_path,file_y),all_y_data) # (num_chunks,n_classes)



class DataLoader(Sequence):
	"""Generates data for Keras. This serves as a fast data loader with optimized memory usage (instead of storing every possible set of `num_expansion_frames` contextual frames).
	"""
	def __init__(self, data_path, partition='all',balance=False, batch_size=32, shuffle=True, samples_per_chunk=5):
		"""Initialization. Create an indexation of all possible samples, eg all possible sets of contextual frames.
		Inputs:
			data_path (str): path where the .npy mel features and target vectors can be found
			partition (str): one of 'all', 'train', 'validation', 'evaluation'
			balance (bool): wether to balance the data or not by taking more samples from chunks that have rarer tags (only intended for training, otherwise no balancing is done)
			batch_size (int): number of samples per batch (only intended for training, otherwise `batch_size` is fixed to be `samples_per_chunk`)
			shuffle (bool): wether to shuffle the data loader
			samples_per_chunk (int): number of samples to be extracted from each chunk; if `balance` is True, this is actually instead the **minimum** number of samples per chunks
		"""
		assert not (balance and partition=='evaluation'), "Cannot balance AND evaluate"

		self.y_dim = (n_classes,)
		self.batch_size = batch_size
		# self.labels = labels
		self.n_channels = 1
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.partition = partition

		# load data and infer sizes
		self.__load_data(data_path)
		num_mel_feat = self.mfb_frames.shape[2]
		self.x_dim = ((num_expansion_frames+1)*num_mel_feat,) # eg 92*40
		num_frames = self.mfb_frames.shape[1]
		num_chunks = self.targets.shape[0]

		# preprocess the data (scaling)
		self.__preprocess()

		if not balance:
			# create an exhaustive list of indexes for all possible samples
			chunks = np.arange(num_chunks)
			middle_frames = np.linspace(int(num_expansion_frames/2), num_frames-int(num_expansion_frames/2)-1, samples_per_chunk).astype(int) # eg if num_expansion_frames=91 and num_frames=399, this is an int array [40, ... 358]
			grid_chunks,grid_middle_frames = np.meshgrid(chunks,middle_frames)
			self.list_IDs = np.transpose(np.stack([grid_chunks,grid_middle_frames],axis=2),[1,0,2]).reshape(-1,2) # exhaustive list of all [chunk_id,middle_frame_id], in lexicographic order such that same chunks are grouped
		else:
			chunks = np.arange(num_chunks)
			self.list_IDs = []
			max_tag_proportion = np.max(list(tag_proportions.values()))
			for chunk in chunks:
				tags = [ind_to_tag[cl] for cl in range(n_classes) if self.targets[chunk,cl]]
				proportions = [tag_proportions[tag] for tag in tags]
				chunk_balancing = max_tag_proportion/np.mean(proportions) # this is one simple way to have more samples for chunks that have rare tags
				chunk_balancing = int(np.round(chunk_balancing*samples_per_chunk)) # ensures that every chunk is sampled at least `samples_per_chunk` times
				self.list_IDs += list(np.stack([[chunk]*chunk_balancing, np.linspace(int(num_expansion_frames/2),num_frames-int(num_expansion_frames/2)-1,chunk_balancing).astype(int)],axis=1)) # idea: could be randomint indexes instead of linspace
			self.list_IDs = np.array(self.list_IDs)
		self.num_samples = self.list_IDs.shape[0]


		# partition = 'evaluation' is a special case: a batch has a fixed length and is made of all samples from one chunk, also shuffle is False.
		if partition == 'evaluation':
			self.shuffle = False
			self.batch_size = samples_per_chunk

		self.on_epoch_end()

	def __load_data(self,data_path):
		"""Subroutine for `__init__`; loads the data from numpy array .npy files (created by `create_mfb_features`) 
		"""
		# load data from .npy files
		self.mfb_frames = np.load(os.path.join(data_path,file_feat))
		self.mfb_averages = self.mfb_frames[:,:first_background_noise_aware,:].mean(axis=1)
		self.targets = np.load(os.path.join(data_path,file_y))

		num_chunks = self.targets.shape[0]

		# compute chunk ranges depending on the partition
		if self.partition == 'all':
			selec_chunks = np.arange(num_chunks)
		elif self.partition == 'train':
			selec_chunks = np.arange(0,int(num_chunks*train_val))
		elif self.partition == 'validation':
			selec_chunks = np.arange(int(num_chunks*train_val),num_chunks)
		elif self.partition == 'evaluation':
			selec_chunks = np.arange(int(num_chunks*train_val),num_chunks)
		else:
			assert False

		# keep only a part of the data corresponding to the given chunk range
		self.mfb_frames = self.mfb_frames[selec_chunks,:,:]
		self.mfb_averages = self.mfb_averages[selec_chunks,:]
		self.targets = self.targets[selec_chunks,:]

	def __preprocess(self):
		"""Performs feature scaling and zero mean"""
		self.mfb_frames -= self.mfb_frames.mean(axis=0)
		self.mfb_averages -= self.mfb_averages.mean(axis=0)

		self.mfb_frames /= self.mfb_frames.std(axis=0)
		self.mfb_averages /= self.mfb_averages.std(axis=0)


	def __len__(self):
		"""Denotes the number of batches per epoch. Required by Keras."""
		return int(np.floor(self.num_samples / self.batch_size))

	def __getitem__(self, index):
		"""Generate one batch of data. Required by Keras."""
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X = np.empty((self.batch_size, *self.x_dim, self.n_channels))
		y = np.empty((self.batch_size, *self.y_dim), dtype=int)
		for i,IDs in enumerate(list_IDs_temp):
			[chunk_id,middle_frame_id] = IDs
			X[i,:], y[i,:] = self.data_generation(chunk_id, middle_frame_id)

		X = X.reshape(self.batch_size,*self.x_dim)

		return X, y

	def on_epoch_end(self):
		"""Shuffle indexes after each epoch. Required by Keras."""
		self.indexes = np.arange(self.num_samples)
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def data_generation(self, chunk_id, middle_frame_id):
		"""Generates one sample"""
		expanded_frames = self.mfb_frames[chunk_id,middle_frame_id-int(num_expansion_frames/2):middle_frame_id+1+int(num_expansion_frames/2),:]
		background_noise_aware_frame = self.mfb_averages[chunk_id,:].reshape(1,-1)
		X = np.concatenate([expanded_frames, background_noise_aware_frame], axis=0)
		X = X.reshape(self.x_dim[0],1)
		
		y = self.targets[chunk_id,:]
		
		return X,y

if __name__ == '__main__':
	args = parser.parse_args()
	create_mfb_features(args.data,args.fs,args.frame,args.hop)

