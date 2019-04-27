
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


class SmallerVGGNet:
    def __init__(self, width, height, depth, classes, initialize_learning_rate=1e-3, epoches=100, batch_size=32):
        self.input_shape = (height, width, depth)
        self.classes = classes
        self.batch_size = batch_size
        self.initialize_learning_rate = initialize_learning_rate
        self.epoches = epoches

        self.model = self._create_model()
        self.compile()

    def _create_model(self):
        axis = -1

        model = Sequential()
        model.add(
            Conv2D(
                32,
                (3, 3),
                padding="same",
                input_shape=self.input_shape
            )
        )
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=axis))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=axis))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=axis))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=axis))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=axis))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(self.classes))
        model.add(Activation("softmax"))

        return model

    def compile(self):
        opt = Adam(
            lr=self.initialize_learning_rate,
            decay=self.initialize_learning_rate / self.epoches
        )

        self.model.compile(
            loss="binary_crossentropy",
            optimizer=opt,
            metrics=["accuracy"],
        )

    def fit(self, data, labels, verbose=1):
        trainX, testX, trainY, testY = train_test_split(
            data,
            labels,
            test_size=0.25,
            random_state=42,
        )
        self.model.fit(
            trainX,
            trainY,
            epochs=self.epoches,
            batch_size=self.batch_size,
            verbose=verbose,
        )
        return self.model.evaluate(
            testX,
            testY,
            batch_size=self.batch_size,
            verbose=verbose,
        )

    def save(self, path):
        self.model.save(path)


class LeNet:
    def __init__(self, width, height, depth, classes, initialize_learning_rate=1e-3, epoches=100, batch_size=32):
        self.input_shape = (height, width, depth)
        self.classes = classes
        self.batch_size = batch_size
        self.initialize_learning_rate = initialize_learning_rate
        self.epoches = epoches

        self.model = self._create_model()
        self.compile()

    def _create_model(self):
        model = Sequential()
        model.add(
            Conv2D(
                20,
                (5, 5),
                padding="same",
                input_shape=self.input_shape,
            )
        )
        model.add(Activation("relu"))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(50, (5, 5), padding="same"))        
        model.add(Activation("relu"))

        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))        
        model.add(Activation("relu"))

        model.add(Dense(self.classes))
        model.add(Activation("softmax"))

        return model

    def compile(self):
        opt = Adam(
            lr=self.initialize_learning_rate,
            decay=self.initialize_learning_rate / self.epoches
        )

        self.model.compile(
            loss="binary_crossentropy",
            optimizer=opt,
            metrics=["accuracy"],
        )

    def fit(self, data, labels, verbose=1):
        trainX, testX, trainY, testY = train_test_split(
            data,
            labels,
            test_size=0.25,
            random_state=42,
        )
        aug = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )
        self.model.fit_generator(
            aug.flow(trainX, trainY, batch_size=self.batch_size),
            validation_data=(testX, testY),
            steps_per_epoch=len(trainX) // self.batch_size,
            epochs=self.epoches,
            verbose=1
        )
        return self.model.evaluate(
            testX,
            testY,
            batch_size=self.batch_size,
            verbose=verbose,
        )

    def save(self, path):
        self.model.save(path)
