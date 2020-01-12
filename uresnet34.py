from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D, BatchNormalization, Add
from keras.layers.core import Activation

# U-Net + ResNet 34
def uresnet(image_rows, image_cols, img_channels=1):

    def Conv2D_block(filters, strides=(3, 3), activation="relu", padding="same"):
        assert(isinstance(strides, tuple))
        assert(2 == len(strides))

        def f(x):
            conv1 = Conv2D(filters, strides, padding=padding)(x)
            batch_norm1 = BatchNormalization()(conv1)
            act1 = Activation(activation=activation)(batch_norm1)
            conv1 = Conv2D(filters, strides, padding=padding)(act1)
            batch_norm1 = BatchNormalization()(conv1)
            act1 = Activation(activation=activation)(batch_norm1)
            add1 = Add()([x, act1])
            return add1
        return f

    inputs = Input((image_rows, image_cols, img_channels))
    conv0 = Conv2D(32, (3, 3), padding='same')(inputs)
    batch_norm0 = BatchNormalization()(conv0)
    act0 = Activation(activation='relu')(batch_norm0)
    pool0 = MaxPooling2D(pool_size=(2, 2))(act0)

    conv1 = Conv2D_block(32, (3, 3), activation='relu', padding='same')(pool0)
    conv1 = Conv2D_block(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D_block(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D_block(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D_block(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D_block(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D_block(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D_block(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D_block(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D_block(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D_block(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D_block(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D_block(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D_block(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D_block(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D_block(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D_block(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D_block(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
        
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model
