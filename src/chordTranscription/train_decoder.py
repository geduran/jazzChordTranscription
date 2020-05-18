import sys
import glob
import numpy         as     np
import tensorflow    as     tf
from   keras         import backend as K
from   keras.backend import tensorflow_backend
from   dlUtils       import functional_decoder, define_callBacks, decoder_split_data, DecoderDataGenerator


# tensorflow configuration
K.set_image_dim_ordering('th')
print(K.image_data_format())
config  = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

method = 'cqt'
model_paths = '../../models/chordTranscription/'
data_path = '../../data/JAAH/chordTranscription/intermediate/' + method + '/'
data_temp_dir = './temp_data_decoder/'

best_model    = 'decoder_'+method+'_{epoch:02d}.h5'
last_model    = 'decoder_'+method+'_last.h5'

# prepare callbacks
callbacks    = define_callBacks(model_paths + best_model)

epochs         = 40
n_hidden       = 60
batch_size     = 128
# seq_len        = 50
num_features   = 26
n_labels = 61



# decoder_split_data(data_path,  out_dir=data_temp_dir)

training_filenames = glob.glob(data_temp_dir + '*train*.pkl')
validation_filenames = glob.glob(data_temp_dir + '*validation*.pkl')
my_training_batch_generator   = DecoderDataGenerator(training_filenames, num_features=num_features, n_labels=n_labels)
my_validation_batch_generator = DecoderDataGenerator(validation_filenames, num_features=num_features, n_labels=n_labels)

n_train_samples = len(training_filenames)
n_val_samples = len(validation_filenames)

input_shape  = (None, num_features)

model =  functional_decoder(input_shape)

model.summary()


history = model.fit_generator(generator=my_training_batch_generator,
                              steps_per_epoch = (n_train_samples),
                              epochs          = epochs,
                              verbose         = 1,
                              validation_data = my_validation_batch_generator,
                              validation_steps= (n_val_samples),
                              use_multiprocessing = True,
                              workers         = 1,
                              max_queue_size  = 32,
                              shuffle         = True,
                              callbacks       = callbacks
                              )


#
model.save_weights(model_paths + last_model)
