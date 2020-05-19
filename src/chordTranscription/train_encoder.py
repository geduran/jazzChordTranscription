import sys
import glob
import numpy         as     np
import tensorflow    as     tf
from   keras         import backend as K
from   keras.backend import tensorflow_backend
from   dlUtils       import functional_encoder, encoder_callBacks, encoder_split_data, EncoderDataGenerator


# tensorflow configuration
K.set_image_dim_ordering('th')
print(K.image_data_format())
config  = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

method = 'cqt'
model_paths = '../../models/chordTranscription/encoder/'
data_path = '../../data/JAAH/chordTranscription/encoder/' + method + '/'
data_temp_dir = './temp_data_encoder/'

best_model    = 'encoder_'+method+'_{epoch:02d}.h5'
last_model    = 'encoder_'+method+'_last.h5'

# prepare callbacks
callbacks    = encoder_callBacks(model_paths + best_model)

epochs         = 40
# n_hidden       = 60
batch_size     = 128
seq_len        = 50
num_features   = 84
n_labels_root  = 13
n_labels_notes = 13
n_labels_beats = 2



n_train_samples, n_val_samples = encoder_split_data(data_path, seq_len=seq_len, out_dir=data_temp_dir)

training_filenames = glob.glob(data_temp_dir + '*train*.pkl')
validation_filenames = glob.glob(data_temp_dir + '*validation*.pkl')
my_training_batch_generator   = EncoderDataGenerator(training_filenames,   batch_size=batch_size, seq_len=seq_len, num_features=num_features)
my_validation_batch_generator = EncoderDataGenerator(validation_filenames, batch_size=batch_size, seq_len=seq_len, num_features=num_features)



input_shape  = (None, num_features, 1)

model =  functional_encoder(input_shape)


history = model.fit_generator(generator=my_training_batch_generator,
                                      steps_per_epoch = (n_train_samples // batch_size),
                                      epochs          = epochs,
                                      verbose         = 1,
                                      validation_data = my_validation_batch_generator,
                                      validation_steps= (n_val_samples // batch_size),
                                      use_multiprocessing = True,
                                      workers         = 16,
                                      max_queue_size  = 32,
                                      shuffle         = True,
                                      callbacks       = callbacks
                                      )


#
model.save_weights(model_paths + last_model)
