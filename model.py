import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_folder', '', "Training folder (containing the 'driving_log.csv' file and the 'IMG' folder)")
flags.DEFINE_string('output_model_file', None, "Save the weights of the model to a file (.h5)")
flags.DEFINE_string('input_model_file', None, "Load the weights of the model from a file (.h5)")
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")

import os
import csv

def load(training_folder):
    samples = []
    with open(training_folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples

import cv2
import numpy as np
import sklearn
# import tensorflow as tf
# tf.python.control_flow_ops = tf

def flip_img(image):
    return cv2.flip(image, 1)

def rotate_img(image, angle_deg):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle_deg, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

def zoom_img(image, factor):
    orig_shape = image.shape[:-1]
    size = np.array(image.shape[:-1])
    crop_size = size * factor
    start = (size - crop_size) / 2
    end = start + crop_size
    image = image[int(start[0]):int(end[0]),int(start[1]):int(end[1])]
    return cv2.resize(image, orig_shape[::-1])

def generator(samples, training_folder, batch_size=32, angle_offset=0.2):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = training_folder + '/IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                if center_image is None:
                    continue
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                name = training_folder + '/IMG/' + batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                if left_image is None:
                    continue
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_angle = min(1.0, float(batch_sample[3]) + angle_offset)
                images.append(left_image)
                angles.append(left_angle)

                name = training_folder + '/IMG/' + batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                if right_image is None:
                    continue
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                right_angle = max(-1.0, float(batch_sample[3]) - angle_offset)
                images.append(right_image)
                angles.append(right_angle)

            # augment images
            # flip
            new_images = []
            new_angles = []
            for img in images:
                new_images.append(flip_img(img))
            for angle in angles:
                new_angles.append(-angle)
            images += new_images
            angles += new_angles
            # rotate
            new_images = []
            new_angles = []
            for img in images:
                new_images.append(rotate_img(img, -5))
                new_images.append(rotate_img(img, 5))
            for angle in angles:
                new_angles.append(angle)
                new_angles.append(angle)
            images += new_images
            angles += new_angles
            # zoom
            new_images = []
            new_angles = []
            for img in images:
                new_images.append(zoom_img(img, 0.8))
                new_images.append(zoom_img(img, 0.9))
                new_images.append(zoom_img(img, 1.1))
                new_images.append(zoom_img(img, 1.2))
            for angle in angles:
                new_angles.append(angle)
                new_angles.append(angle)
                new_angles.append(angle)
                new_angles.append(angle)
            images += new_images
            angles += new_angles

            images, angles = sklearn.utils.shuffle(images, angles)
            for offset2 in range(0, len(images), batch_size):
                batch_images = images[offset2:offset2+batch_size]
                batch_angles = angles[offset2:offset2+batch_size]
                X_train = np.array(batch_images)
                y_train = np.array(batch_angles)
                yield X_train, y_train

def main(_):
    if not(FLAGS.output_model_file is None):
        output_model_file_without_ext = os.path.splitext(FLAGS.output_model_file)
    else:
        output_model_file_without_ext = "model"

    # load the samples
    train_samples, validation_samples = load(FLAGS.training_folder)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, FLAGS.training_folder, batch_size=FLAGS.batch_size)
    validation_generator = generator(validation_samples, FLAGS.training_folder, batch_size=FLAGS.batch_size)

    # Initial Setup for Keras
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten, Dropout
    from keras.layers import Lambda, Cropping2D
    from keras.layers.convolutional import Convolution2D
    from keras.layers.pooling import MaxPooling2D
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    if not(FLAGS.input_model_file is None):
        # load weights into new model
        from keras.models import load_model
        model = load_model(FLAGS.input_model_file)
        print("Loaded model from disk")
    else:
        row, col, ch = 160, 320, 3  # image format

        model = Sequential()
        # Preprocess incoming data, centered around zero with small standard deviation
        model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))
        # Preprocess incoming data, clip to ROI
        model.add(Cropping2D(cropping=((50,20), (0,0))))
        # Build the Neural Network in Keras Here
        model.add(Convolution2D(24, 5, 5))
        model.add(MaxPooling2D((2, 2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 5, 5))
        model.add(MaxPooling2D((2, 2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(48, 5, 5))
        model.add(MaxPooling2D((2, 2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))

        model.add(Flatten())

        model.add(Dense(128))
        model.add(Dropout(0.3))
        model.add(Activation('tanh'))
        model.add(Dense(64))
        model.add(Dropout(0.3))
        model.add(Activation('tanh'))
        model.add(Dense(32))
        model.add(Dropout(0.3))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('tanh'))

        model.compile(loss='mse', optimizer='adam')

    # output current model
    print("")
    print("Current model:")
    print(model.summary())

    es = EarlyStopping(monitor='val_loss',
        min_delta=0,
        patience=4,
        verbose=1, mode='auto')

    filepath = output_model_file_without_ext + "-checkpoint-{epoch:02d}-{val_loss:.3f}.h5"
    cp = ModelCheckpoint(filepath,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1, mode='auto')

    print("")
    history = model.fit_generator(train_generator,
        samples_per_epoch=len(train_samples)*90,
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples)*90,
        nb_epoch=FLAGS.epochs,
        callbacks=[es, cp])

    if not(FLAGS.output_model_file is None):
        # serialize weights to HDF5
        print("")
        model_file = output_model_file_without_ext + ".h5"
        model.save(model_file)
        print("Saved model to disk: '" + model_file + "'")

    # print the keys contained in the history object
    print("")
    print(history.history.keys())

    # plot the training and validation loss for each epoch
    import matplotlib.pyplot as plt
    plt.subplot(211)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='lower right')
    plt.subplot(212)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='lower right')
    plt.show()


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
