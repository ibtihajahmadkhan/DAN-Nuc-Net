from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.initializers import *
from tensorflow.keras.activations import *
from loss import *
from metrics import *


def spa_atten(enc_feat, dec_feat):
    enc_line = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(enc_feat)
    enc_line = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(enc_line)
    enc_line = Conv2D(filters=1, kernel_size=(1, 1), activation=None, padding='same',
                      kernel_initializer='glorot_normal', bias_initializer=Constant(0.1))(enc_line)
    enc_line = Conv2D(filters=1, kernel_size=(7, 7), activation=None, padding='same',
                      kernel_initializer='glorot_normal', bias_initializer=Constant(0.1))(enc_line)

    dec_line = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(dec_feat)
    dec_line = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(dec_line)
    dec_line = Conv2D(filters=1, kernel_size=(1, 1), activation=None, padding='same',
                      kernel_initializer='glorot_normal', bias_initializer=Constant(0.1))(dec_line)
    dec_line = Conv2D(filters=1, kernel_size=(7, 7), activation=None, padding='same',
                      kernel_initializer='glorot_normal', bias_initializer=Constant(0.1))(dec_line)

    out = sigmoid(add([enc_line, dec_line]))

    return out


def cha_atten(enc_feat, dec_feat, filters):
    enc_line_a = AveragePooling2D(pool_size=(2, 2), strides=(1, 1),  padding="same")(enc_feat)
    enc_line_a = Conv2D(filters=filters / 16,    kernel_size=(1, 1), activation=None,  padding='same',
                        kernel_initializer='glorot_normal', bias_initializer=Constant(0.1))(enc_line_a)
    enc_line_m = MaxPooling2D(pool_size=(2, 2),  strides=(1, 1),     padding="same")(enc_feat)
    enc_line_m = Conv2D(filters=filters / 16,    kernel_size=(1, 1), activation=None,  padding='same',
                        kernel_initializer='glorot_normal', bias_initializer=Constant(0.1))(enc_line_m)
    enc_line = add([enc_line_a, enc_line_m])
    enc_line = Conv2D(filters=filters / 16,    kernel_size=(1, 1), activation=None,  padding='same',
                      kernel_initializer='glorot_normal', bias_initializer=Constant(0.1))(enc_line)

    dec_line_a = AveragePooling2D(pool_size=(2, 2), strides=(1, 1),  padding="same")(dec_feat)
    dec_line_a = Conv2D(filters=filters / 16,    kernel_size=(1, 1), activation=None,  padding='same',
                        kernel_initializer='glorot_normal', bias_initializer=Constant(0.1))(dec_line_a)
    dec_line_m = MaxPooling2D(pool_size=(2, 2),  strides=(1, 1),     padding="same")(dec_feat)
    dec_line_m = Conv2D(filters=filters / 16,    kernel_size=(1, 1), activation=None,  padding='same',
                        kernel_initializer='glorot_normal', bias_initializer=Constant(0.1))(dec_line_m)
    dec_line = add([dec_line_a, dec_line_m])
    dec_line = Conv2D(filters=filters / 16,    kernel_size=(1, 1), activation=None,  padding='same',
                      kernel_initializer='glorot_normal', bias_initializer=Constant(0.1))(dec_line)

    out = add([enc_line, dec_line])
    out = Conv2D(filters=filters, kernel_size=(1, 1), activation='sigmoid', padding='same',
                 kernel_initializer='glorot_normal', bias_initializer=Constant(0.1))(out)

    return out


def spa_cha_atten_gate(enc_feat, dec_feat, filters):
    spa_aten_gate = spa_atten(enc_feat, dec_feat)
    cha_aten_gate = cha_atten(enc_feat, dec_feat, filters)

    mul = multiply([spa_aten_gate, enc_feat])
    mul = multiply([cha_aten_gate, mul])

    cat = concatenate([mul, dec_feat], axis=-1)

    out = Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same',
                 kernel_initializer='glorot_normal', bias_initializer=Constant(0.1))(cat)
    out = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.01), gamma_initializer=Constant(1.0),
                             momentum=0.5)(out)

    return out


def conv_block(input, filters):
    x = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal',
               bias_initializer=Constant(0.1))(input)
    x = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal',
               bias_initializer=Constant(0.1))(x)
    x = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0),
                           momentum=0.5)(x)
    return x


def bottleneck_block(input, filters):
    x = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal',
               bias_initializer=Constant(0.1))(input)
    x = Conv2D(filters, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='glorot_normal',
               bias_initializer=Constant(0.1))(x)
    x = BatchNormalization(epsilon=1e-3, beta_initializer=Constant(0.0), gamma_initializer=Constant(1.0),
                           momentum=0.5)(x)
    return x


def robust_residual_block(input, filt):
    xa_0 = Conv2D(filters=filt, kernel_size=(1,1), padding='same',activation='relu',
                  kernel_initializer='glorot_normal', bias_initializer=Constant(0.1))(input)
    bn_0 = BatchNormalization(epsilon=1e-3,beta_initializer=Constant(0.01), gamma_initializer=Constant(1.0),
                              momentum=0.5)(xa_0)

    xa_1 = Conv2D(filters=filt, kernel_size=(3,3), padding='same',activation='relu',
                  kernel_initializer='glorot_normal', bias_initializer=Constant(0.1))(bn_0)
    bn_1 = BatchNormalization(epsilon=1e-3,beta_initializer=Constant(0.01), gamma_initializer=Constant(1.0),
                              momentum=0.5)(xa_1)

    xa_2 = Conv2D(filters=filt, kernel_size=(1,1), padding='same',activation='relu',
                  kernel_initializer='glorot_normal',bias_initializer=Constant(0.1))(bn_1)
    bn_2 = BatchNormalization(epsilon=1e-3,beta_initializer=Constant(0.01), gamma_initializer=Constant(1.0),
                              momentum=0.5)(xa_2)

    xb_0 = Conv2D(filters=filt, kernel_size=(1,1), padding='same',activation='relu',
                  kernel_initializer='glorot_normal',bias_initializer=Constant(0.1))(input)
    bn_3 = BatchNormalization(epsilon=1e-3,beta_initializer=Constant(0.01), gamma_initializer=Constant(1.0),
                              momentum=0.5)(xb_0)
    x = concatenate([bn_2,bn_3],axis=-1)
    x = Conv2D(filters=filt, kernel_size=(3, 3), activation='relu', padding='same', use_bias=True,
               kernel_initializer='glorot_normal', bias_initializer=Constant(0.1))(x)

    return x


def dan_nuc_net(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)
    filters = [32, 64, 128, 256, 512]

    r_conv1 = robust_residual_block(inputs, filters[0])
    pool = MaxPooling2D( (2, 2), strides=(2, 2), padding="same")(r_conv1)

    r_conv2 = robust_residual_block(pool, filters[1])
    pool = MaxPooling2D( (2, 2), strides=(2, 2), padding="same")(r_conv2)

    r_conv3 = robust_residual_block(pool, filters[2])
    pool = MaxPooling2D( (2, 2), strides=(2, 2), padding="same")(r_conv3)

    r_conv4 = robust_residual_block(pool, filters[3])
    pool = MaxPooling2D( (2, 2), strides=(2, 2), padding="same")(r_conv4)

    r_conv5 = bottleneck_block(pool, filters[4])

    up_sam_1 = Conv2DTranspose(filters[3], (3, 3), strides=(2, 2), padding='same')(r_conv5)
    up_conv1 = spa_cha_atten_gate(r_conv4, up_sam_1, filters[3])

    up_sam_2 = Conv2DTranspose(filters[2], (3, 3), strides=(2, 2), padding='same')(up_conv1)
    up_conv2 = spa_cha_atten_gate(r_conv3, up_sam_2, filters[2])

    up_sam_3 = Conv2DTranspose(filters[1], (3, 3), strides=(2, 2), padding='same')(up_conv2)
    up_conv3 = spa_cha_atten_gate(r_conv2, up_sam_3, filters[1])

    up_sam_4 = Conv2DTranspose(filters[0], (3, 3), strides=(2, 2), padding='same')(up_conv3)
    up_conv4 = spa_cha_atten_gate(r_conv1, up_sam_4, filters[0])

    cat = Conv2D(filters[0], kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(up_conv4)

    outputs = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid', padding='same')(cat)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=optimizers.Adam(lr=0.001), loss=loss, metrics=[f1_score], run_eagerly=True)

    model.summary()

    tf.keras.utils.plot_model(model, to_file="Model_Plot_SCA_NucNet.png", show_shapes=True, show_layer_names=False,
                              expand_nested=False)

    return model
