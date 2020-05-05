def construct_model(pump):
    '''build the model'''
    model_inputs = 'cqt/mag'

    # Build the input layer
    x = pump.layers()[model_inputs]

    # Apply batch normalization
    x_bn = K.layers.BatchNormalization()(x)

    # First convolutional filter: a single 5x5
    conv1 = K.layers.Convolution2D(1, (5, 5),
                                   padding='same',
                                   activation='relu',
                                   data_format='channels_last')(x_bn)

    c1bn = K.layers.BatchNormalization()(conv1)

    # Second convolutional filter: a bank of full-height filters
    conv2 = K.layers.Convolution2D(12*6, (1, int(conv1.shape[2])),
                                   padding='valid', activation='relu',
                                   data_format='channels_last')(c1bn)

    c2bn = K.layers.BatchNormalization()(conv2)

    # Squeeze out the frequency dimension
    squeeze = crema.layers.SqueezeLayer(axis=2)(c2bn)

    # BRNN layer
    rnn1 = K.layers.Bidirectional(K.layers.GRU(128,
                                               return_sequences=True))(squeeze)

    r1bn = K.layers.BatchNormalization()(rnn1)

    rnn2 = K.layers.Bidirectional(K.layers.GRU(128,
                                              return_sequences=True))(r1bn)

    # 1: pitch class predictor
    pc = K.layers.Dense(pump.fields['chord_struct/pitch'].shape[1],
                        activation='sigmoid')

    pc_p = K.layers.TimeDistributed(pc, name='chord_pitch')(rnn2)

    # 2: root predictor
    root = K.layers.Dense(13, activation='softmax')
    root_p = K.layers.TimeDistributed(root, name='chord_root')(rnn2)

    # 3: bass predictor
    bass = K.layers.Dense(13, activation='softmax')
    bass_p = K.layers.TimeDistributed(bass, name='chord_bass')(rnn2)

    # 4: merge layer
    codec = K.layers.concatenate([rnn2, pc_p, root_p, bass_p])

    codecbn = K.layers.BatchNormalization()(codec)

    p0 = K.layers.Dense(len(pump['chord_tag'].vocabulary()),
                        activation='softmax',
                        bias_regularizer=K.regularizers.l2())

    tag = K.layers.TimeDistributed(p0, name='chord_tag')(codecbn)

    model = K.models.Model(x, [tag, pc_p, root_p, bass_p])
    model_outputs = ['chord_tag/chord',
                     'chord_struct/pitch',
                     'chord_struct/root',
                     'chord_struct/bass']

    return model, [model_inputs], model_outputs
