import keras
from keras.callbacks import ModelCheckpoint
import time

import src.models as models
import src.prepare_data as pp_data


def main():
	data_train = pp_data.DataLoader('data',partition='train')
	data_val = pp_data.DataLoader('data',partition='validation')

	model = models.build_dense(input_shape=(92*40,))
	model.summary()

	dump_file = 'models/' + time.strftime("%Y%m%d%H%M",time.localtime()) + '_keras_weights.{epoch:02d}-{val_loss:.2f}.ckpt'
	eachmodel=ModelCheckpoint(dump_file,monitor='val_loss',verbose=0,save_best_only=False,save_weights_only=False,mode='auto')

	model.fit_generator(generator=data_train,validation_data=data_val,nb_epoch=3,verbose=1,
						use_multiprocessing=True,workers=6,callbacks=[eachmodel])

if __name__ == '__main__':
	main()
