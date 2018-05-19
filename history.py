import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('input_model_file', None, "Load the weights of the model from a file (.h5)")

import os

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

def main(_):
    global hist
    # split file extension
    if not(FLAGS.input_model_file is None):
        input_model_file_without_ext = os.path.splitext(FLAGS.input_model_file)[0]
    else:
        input_model_file_without_ext = "model"

    # load history
    log_history_start(input_model_file_without_ext)

    print("Current model was previously trained for {epochs} epochs".format(epochs=len(hist['loss'])))

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
