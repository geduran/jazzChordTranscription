import sys
import glob
import pickle
import numpy         as     np
import tensorflow    as     tf
from   keras         import backend as K
from   keras.backend import tensorflow_backend
from   dlUtils       import functional_both, encoder_callBacks, both_split_data, BothDataGenerator
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use id from $ nvidia-smi

# tensorflow configuration
K.set_image_dim_ordering('th')
print(K.image_data_format())
config  = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

method = 'cqt'
model_paths = '../../models/chordTranscription/both/'
data_path = '../../data/JAAH/chordTranscription/both_da/' + method + '/'
data_temp_dir = './temp_data_both/'

best_model    = 'both_'+method+'_{epoch:02d}.h5'
last_model    = 'both_'+method+'_last.h5'

# prepare callbacks
callbacks    = encoder_callBacks(model_paths + best_model)


epochs         = 40
# n_hidden       = 60
batch_size     = 64
seq_len        = 120
num_features   = 84
n_labels_root  = 13
n_labels_notes = 12
n_labels_beats = 2


both_split_data(data_path, seq_len=seq_len, out_dir=data_temp_dir)
# with open(model_paths + 'mean_std.pkl', 'wb') as f:
#     pickle.dump({'mean':mean, 'std':std}, f)

training_filenames = glob.glob(data_temp_dir + '*train*.pkl')
validation_filenames = glob.glob(data_temp_dir + '*validation*.pkl')
n_train_samples = len(training_filenames)
n_val_samples = len(validation_filenames)

my_training_batch_generator   = BothDataGenerator(training_filenames,   batch_size=batch_size, seq_len=seq_len, num_features=num_features)
my_validation_batch_generator = BothDataGenerator(validation_filenames, batch_size=batch_size, seq_len=seq_len, num_features=num_features)

input_shape  = (None, num_features, 1)

model =  functional_both(input_shape)
model.summary()

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














#
