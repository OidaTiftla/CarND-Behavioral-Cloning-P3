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

def generator(samples, batch_size=32, angle_offset=0.2, training=True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(batch_sample[0])
                if center_image is None:
                    continue
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                left_image = cv2.imread(batch_sample[1])
                if left_image is None:
                    continue
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_angle = min(1.0, float(batch_sample[3]) + angle_offset)
                images.append(left_image)
                angles.append(left_angle)

                right_image = cv2.imread(batch_sample[2])
                if right_image is None:
                    continue
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                right_angle = max(-1.0, float(batch_sample[3]) - angle_offset)
                images.append(right_image)
                angles.append(right_angle)

            if training:
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

hist = {}
def log_history_start(input_file=None):
    global hist
    hist['loss'] = []
    hist['val_loss'] = []
    if not(input_file is None):
        import pickle
        hist_loaded = pickle.load(open(input_file + "-history.p", 'rb'))
        hist['loss'] = hist_loaded['loss']
        hist['val_loss'] = hist_loaded['val_loss']

def log_history_add(loss, val_loss):
    global hist
    hist['loss'].append(loss)
    hist['val_loss'].append(val_loss)

def log_history_save(output_file=None):
    global hist
    if not(output_file is None):
        import pickle
        pickle.dump(hist, open(output_file + "-history.p", 'wb'))

def log_history_add_save(loss, val_loss, output_file=None):
    log_history_add(loss, val_loss)
    log_history_save(output_file)

def main(_):
    global hist
    # split file extension
    if not(FLAGS.output_model_file is None):
        output_model_file_without_ext = os.path.splitext(FLAGS.output_model_file)[0]
    else:
        output_model_file_without_ext = "model"
    if not(FLAGS.input_model_file is None):
        input_model_file_without_ext = os.path.splitext(FLAGS.input_model_file)[0]
    else:
        input_model_file_without_ext = None
    # add date and time
    import time
    output_model_file_without_ext += '_' + time.strftime("%Y-%m-%d_%H-%M")

    # load the samples from multiple folders
    train_samples = []
    validation_samples = []
    for folder in FLAGS.training_folder.split(','):
        ts, vs = load(folder)
        # adjust path to img
        for sample in ts:
            sample[0] = folder + '/IMG/' + sample[0].split('/')[-1]
            sample[1] = folder + '/IMG/' + sample[1].split('/')[-1]
            sample[2] = folder + '/IMG/' + sample[2].split('/')[-1]
        for sample in vs:
            sample[0] = folder + '/IMG/' + sample[0].split('/')[-1]
            sample[1] = folder + '/IMG/' + sample[1].split('/')[-1]
            sample[2] = folder + '/IMG/' + sample[2].split('/')[-1]
        train_samples.extend(ts)
        validation_samples.extend(vs)

    # load history
    log_history_start(input_model_file_without_ext)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=FLAGS.batch_size, training=True)
    validation_generator = generator(validation_samples, batch_size=FLAGS.batch_size, training=False)

    # Initial Setup for Keras
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Flatten, Dropout
    from keras.layers import Lambda, Cropping2D
    from keras.layers.convolutional import Convolution2D
    from keras.layers.pooling import MaxPooling2D
    from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback

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
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(Dropout(0.6))
        model.add(Activation('relu'))

        model.add(Flatten())

        model.add(Dense(128))
        model.add(Dropout(0.6))
        model.add(Activation('tanh'))
        model.add(Dense(64))
        model.add(Dropout(0.4))
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
    print("")
    print("Current model was previously trained for {epochs} epochs".format(epochs=len(hist['loss'])))

    # # plot current model
    # print("")
    # from keras.utils.visualize_util import plot_model
    # plot_file = output_model_file_without_ext + ".png"
    # plot_model(model, to_file=plot_file, show_shapes=True, show_layer_names=True)
    # print("Plotted model to disk: '" + plot_file + "'")

    filepath_per_epoch = output_model_file_without_ext + "-checkpoint-{epoch:02d}-{val_loss:.3f}"

    es = EarlyStopping(monitor='val_loss',
        min_delta=0,
        patience=4,
        verbose=1, mode='auto')

    cp = ModelCheckpoint(filepath_per_epoch + '.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1, mode='auto')

    lc = LambdaCallback(
        on_epoch_end=lambda epoch, logs:
            log_history_add_save(
                logs['loss'], logs['val_loss'],
                filepath_per_epoch.format(epoch=epoch, loss=logs['loss'], val_loss=logs['val_loss']))
    )

    print("")
    history = model.fit_generator(train_generator,
        samples_per_epoch=len(train_samples)*3,
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples)*3,
        nb_epoch=FLAGS.epochs,
        callbacks=[es, cp, lc])

    if not(output_model_file_without_ext is None):
        # serialize weights to HDF5
        print("")
        model_file = output_model_file_without_ext + ".h5"
        model.save(model_file)
        print("Saved model to disk: '" + model_file + "'")

    # print the keys contained in the history object
    print("")
    print(history.history.keys())

    # combine history with history on disk
    log_history_save(output_model_file_without_ext)

    # plot the training and validation loss for each epoch
    import matplotlib.pyplot as plt
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
