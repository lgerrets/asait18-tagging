import numpy as np
import os
import scipy
import keras 
import librosa 
from src.parameters import *


def create_mfb_features(data_path,fs,frame_ms,hop_ms):

	frame_length = int(fs*frame_ms/1000)
	frame_step = int(fs*hop_ms/1000)
	NFFT = frame_length # either frame_length or next power of 2
	num_mel_feat = 40

	all_mfb_data = []
	all_y_data = []

	# append data: mel features (X) and tags (y)
	num_chunks = 0
	for file in os.listdir(data_path):
		file = os.path.join(data_path,file)

		if not os.path.isfile(file):
			continue

		if not file[-10:] == ".16kHz.wav":
			continue

		### mel features

		# load the chunk of sound
		rate,x_data = scipy.io.wavfile.read(file)
		x_data = x_data.astype(np.float32)
		assert rate==fs, "Expected " + str(rate) + " but got " + str(fs)
		num_frames = int((len(x_data)-frame_length)/frame_step)+1
		print(num_frames)

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

		### tags

		# load the set of tags associated to the previous chunk
		y_data = np.genfromtxt(file[:-10]+'.csv',delimiter=',',dtype=str)
		y_data = y_data[loc_targets_in_file[0],loc_targets_in_file[1]] # string of tags from the 7 possible tags: bcfmopv

		# convert into a binary vector
		targets = np.zeros(n_classes)
		for c in y_data:
			if c == 'b':
				targets[0] = 1
			elif c == 'c':
				targets[1] = 1
			elif c == 'f':
				targets[2] = 1
			elif c == 'm':
				targets[3] = 1
			elif c == 'o':
				targets[4] = 1
			elif c == 'p':
				targets[5] = 1
			elif c == 'v':
				targets[6] = 1
			else:
				assert False, "Unexpected value "+str(c)+" "+str(y_data)
		targets = targets.astype(np.int32)

		# append this data
		all_y_data.append(targets)

		num_chunks += 1


	np.save(os.path.join(data_path,file_feat),all_mfb_data) # (num_chunks,num_frames,num_mel_feat)
	np.save(os.path.join(data_path,file_y),all_y_data) # (num_chunks,n_classes)



class DataLoader(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_path, partition='all', batch_size=32, shuffle=True):
        'Initialization'
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

        # create an exhaustive list of indexes for all possible samples
        self.chunks = np.arange(self.targets.shape[0])
        self.middle_frames = np.arange(int(num_expansion_frames/2), num_frames-int(num_expansion_frames/2)) # eg if num_expansion_frames=91 and num_frames=399, this is an int array [40, ... 358]
        grid_chunks,grid_middle_frames = np.meshgrid(self.chunks,self.middle_frames)
        self.list_IDs = np.stack([grid_chunks,grid_middle_frames],axis=2).reshape(-1,2) # exhaustive list of all [chunk_id,middle_frame_id]
        self.num_samples = self.list_IDs.shape[0]

        self.on_epoch_end()

    def __load_data(self,data_path):
    	self.mfb_frames = np.load(os.path.join(data_path,file_feat))
    	self.mfb_averages = self.mfb_frames[:,:first_background_noise_aware,:].mean(axis=1)
    	self.targets = np.load(os.path.join(data_path,file_y))

    	num_chunks = self.targets.shape[0]

    	if self.partition == 'all':
    		selec_chunks = np.arange(num_chunks)
    	elif self.partition == 'train':
    		selec_chunks = np.arange(0,int(num_chunks*train_val))
    	elif self.partition == 'validation':
    		selec_chunks = np.arange(int(num_chunks*train_val),num_chunks)
    	else:
    		assert False

    	self.mfb_frames = self.mfb_frames[selec_chunks,:,:]
    	self.mfb_averages = self.mfb_averages[selec_chunks,:]
    	self.targets = self.targets[selec_chunks,:]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = np.empty((self.batch_size, *self.x_dim, self.n_channels))
        y = np.empty((self.batch_size, *self.y_dim), dtype=int)
        for i,IDs in enumerate(list_IDs_temp):
        	[chunk_id,middle_frame_id] = IDs
        	X[i,:], y[i,:] = self.__data_generation(chunk_id, middle_frame_id)

        X = X.reshape(self.batch_size,*self.x_dim)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, chunk_id, middle_frame_id):
        'Generates one sample'
        expanded_frames = self.mfb_frames[chunk_id,middle_frame_id-int(num_expansion_frames/2):middle_frame_id+1+int(num_expansion_frames/2),:]
        background_noise_aware_frame = self.mfb_averages[chunk_id,:].reshape(1,-1)
        X = np.concatenate([expanded_frames, background_noise_aware_frame], axis=0)
        X = X.reshape(self.x_dim[0],1)
        
        y = self.targets[chunk_id,:]
        
        return X,y
