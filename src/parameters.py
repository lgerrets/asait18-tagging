
### global parameters, common to several scripts

# Parameters that are fixed by the given data. The 7 classes are:
# - broadband noise
# - child speech
# - adult female speech
# - adult male speech
# - other identifiable sounds
# - percussive sounds eg crash bang knock footsteps
# - video game / TV
n_classes = 7 # number of classes (see `prepare_data` for more details about each class)
loc_targets_in_file = [9,1] # 9th line, 1st column (',' separated) in the raw csv files; that position points to the tag word
tag_to_ind = {'b':0 ,'c':1 ,'f':2 ,'m':3 ,'o':4 ,'p':5 ,'v':6}
ind_to_tag = list(tag_to_ind.keys())



### The following parameters could actually become script arguments, for more modular experiments

# to create mel features
file_feat = "mfbfeat_x.npy" # file name (not path) of the numpy array of mel features
file_y = "chunk_y.npy" # file name (not path) of the numpy array of binarized tags

# to preprocess the data ; every model must be trained and tested with these params fixed
num_expansion_frames = 91 # number of frames to surround a single frame for context (a neural net is given such context as an input rather than frames alone)
first_background_noise_aware = 6 # first frames of a chunk to be used for estimating average ambient noise

# for train.py
train_val = 0.5 # ratio of the train set (between 0 and 1) 
tag_proportions = {'c' :1214, 'm' :174, 'f' :409, 'v' :1181, 'p' :765, 'b' :19, 'o' :361}
