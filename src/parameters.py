
### global parameters, common to several scripts

# fixed by the given data
n_classes = 7 # number of classes (see `prepare_data` for more details about each class)
loc_targets_in_file = [9,1] # 9th line, 1st column (',' separated) in the raw csv files; that position points to the tag word

# could actually become script arguments, for more modular experiments
file_feat = "mfbfeat_x.npy" # file name (not path) of the numpy array of mel features
file_y = "chunk_y.npy" # file name (not path) of the numpy array of binarized tags
num_expansion_frames = 91 # number of frames to surround a single frame for context (a neural net is given such context as an input rather than frames alone)
first_background_noise_aware = 6 # first frames of a chunk to be used for estimating average ambient noise
train_val = 0.5 # ratio of the train set (between 0 and 1) 
