import sys
import numpy         as     np
import tensorflow    as     tf
from   keras         import backend as K
from   keras.backend import tensorflow_backend
from   dlUtils       import functional_encoder, define_callBacks, load_data


# tensorflow configuration
K.set_image_dim_ordering('th')
print(K.image_data_format())
config  = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

method = 'cqt'
model_paths = '../../models/chordTranscription/'
data_path = '../../data/JAAH/chordTranscription/' + method + '/'


best_model    = 'encoder_'+method+'_best.h5'
last_model    = 'encoder_'+method+'_last.h5'

# prepare callbacks
callbacks    = define_callBacks(model_paths + best_model)

epochs        = 100
n_hidden      = 60
batch_size    = 128
seq_len       = 100

train_d, test_d, train_r, test_r, train_n, test_n, train_b, test_b = load_data(data_path, seq_len=seq_len)

num_features = train_d.shape[2]

input_shape  = (None, num_features, 1)

model =  functional_encoder(input_shape)

# history = model.fit(train_d, {'root': train_r,
#                              'notes': train_n,
#                              'beats': train_b},
#                   batch_size      = batch_size,
#                 	validation_data=(test_d,
#                 		    {'root': test_r,
#                              'notes': test_n,
#                              'beats': test_b}),
#                   epochs=epochs,
#                   verbose=1,
#                   shuffle   = True,
#                   callbacks = callbacks)


# model.save_weights(model_paths + last_model)
