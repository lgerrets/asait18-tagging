import keras
from keras.callbacks import ModelCheckpoint
import time
import os

import models as models
import prepare_data as pp_data

from parameters import *

import argparse
parser = argparse.ArgumentParser(description='Train Mel feature + Deep model for audio tagging')
parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                    help='learning rate (default: 0.005)')
parser.add_argument('--data', type=str, default="data", metavar='DA',
                    help='path to the data folder (where the files %s and %s are located) (default: data)'%(file_feat,file_y))
parser.add_argument('--model', type=str, default="models", metavar='MD',
                    help='path where the model checkpoints should be saved to')
parser.add_argument('--epochs', type=int, default=3, metavar='EP',
                    help='number of training epochs (default = 3)')


def main(args):
	data_train = pp_data.DataLoader(args.data,partition='train',batch_size=100,balance=True,samples_per_chunk=5)
	data_val = pp_data.DataLoader(args.data,partition='validation')

	model = models.build_dense(input_shape=((num_expansion_frames+1)*40,),lr=args.lr)
	model.summary()

	dump_file = os.path.join(args.model, time.strftime("%Y%m%d%H%M",time.localtime()) + '_keras_weights.{epoch:02d}-{val_loss:.2f}.ckpt')
	eachmodel=ModelCheckpoint(dump_file,monitor='val_loss',verbose=0,save_best_only=False,save_weights_only=False,mode='auto')

	model.fit_generator(generator=data_train,validation_data=data_val,nb_epoch=args.epochs,verbose=1,
						use_multiprocessing=True,workers=6,callbacks=[eachmodel])

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
