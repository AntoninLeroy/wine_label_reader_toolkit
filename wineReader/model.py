import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Conv2D , MaxPooling2D ,concatenate ,Input ,Dropout ,UpSampling2D
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping ,ReduceLROnPlateau ,ModelCheckpoint ,TensorBoard
from tensorflow import keras
from keras.preprocessing.image import save_img
import datetime

class Unet:
    
    # class made to build, compile, train and predict with the u-net model
    # must pass config as argument

    def build_model(self, Config):
        
        inputs = tf.keras.layers.Input((256,256,3))
        s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

        #Contraction path
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
        c1 = tf.keras.layers.Dropout(Config['contraction_1_dropout'])(c1)
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(Config['contraction_2_dropout'])(c2)
        c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(Config['contraction_3_dropout'])(c3)
        c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Dropout(Config['contraction_4_dropout'])(c4)
        c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tf.keras.layers.Dropout(Config['contraction_5_dropout'])(c5)
        c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        #Expansive path 
        u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c4])
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(Config['expansive_1_dropout'])(c6)
        c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = tf.keras.layers.concatenate([u7, c3])
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = tf.keras.layers.Dropout(Config['expansive_2_dropout'])(c7)
        c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = tf.keras.layers.concatenate([u8, c2])
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.Dropout(Config['expansive_3_dropout'])(c8)
        c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Dropout(Config['expansive_4_dropout'])(c9)
        c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        return model

    def fit(self,X_train,X_valid,y_train,y_valid,Config):

        model=self.build_model(Config)
        optimizer_adam=Adam(learning_rate=Config['learning_rate'],beta_1=0.9,beta_2=0.99)
        EarlyStop=EarlyStopping(patience=Config['patience'],restore_best_weights=True)
        Reduce_LR=ReduceLROnPlateau(monitor='val_accuracy',verbose=1,factor=Config['lr_dec_factor'],min_lr=0.00001)
        model_check=ModelCheckpoint('models/unet-{}.hdf5'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),monitor='val_loss',verbose=1,save_best_only=True)
        log_dir = Config['tensorboards_path'] + "unet-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorbord=TensorBoard(log_dir=log_dir)
        callback=[EarlyStop , Reduce_LR,model_check,tensorbord]
        model.compile(optimizer=optimizer_adam,loss='binary_crossentropy',metrics=['accuracy'])

        model.fit(
                X_train,
                y_train,
                validation_data=(X_valid,y_valid),
                epochs=Config['max_epoch'],
                batch_size=Config['batch_size'],
                callbacks=callback,
                verbose=1)

        model.save("./models/unet.h5")

    def predict(self, X, model, fileNames, Config):
        
        mask_vectors = model.predict(X)

        for filename, mask in zip(fileNames, mask_vectors):
            save_img(Config['results_path'] + filename + "/2_raw_predict.jpg", mask)

        return mask_vectors
