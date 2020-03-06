import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import forward
from preWork import my_preWork

path = 'processed_face'
def load_data():
    prework = my_preWork()
    img, label = prework.splice_pic(choice=0)

    return img, label

def load_test_data():
    prework = my_preWork()
    img, label = prework.splice_pic(choice=1)

    return img, label

path = os.getcwd()
MODEL_SAVE_PATH = path
MODEL_NAME = "my_model"

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
epochs = 10000
MOVING_AVERAGE_DECAY = 0.99

INPUT_NODE = 784
OUTPUT_NODE = 2
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 2

def plot_acc(list):
    x_axis = np.array([x for x in range(len(list))])
    y_list = np.array([y for y in list])

    plt.scatter(x_axis,y_list,color = 'red')
    plt.show()

def train():
    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(tf.float32,
                        [None, OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    y = forward.forward(x, True, regularizer)

    global_step = tf.Variable(0, trainable=False)
    pred_y = tf.argmax(y, 1)
    y_label = tf.argmax(y_, 1)

    correct_pred = tf.equal(pred_y, y_label)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    train_step = tf.train.GradientDescentOptimizer(0.000001).minimize(loss, global_step=global_step)



    train_accuracy = []
    accuracy_draw = []
    #saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(epochs):
            data, label = load_data()
            data_after = data.eval(session = tf.Session())
            data = np.reshape(data_after, (
                BATCH_SIZE,
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS))

            _, loss_value, step = sess.run([train_step, loss, global_step], feed_dict={x: data, y_: label})
            accuracy_ = sess.run(accuracy, feed_dict={x: data, y_: label})

            train_accuracy.append(accuracy_)
            if step % 10 == 0:
                print("After %d training steps, loss on training batch is %g,Training Accuracy=%g" % (step, loss_value, accuracy_))
                accuracy_draw.append(accuracy_)

        print("Training Finished!")
        train_acc_avg = tf.reduce_mean(tf.cast(train_accuracy, tf.float32))
        print("Average Training Accuracy=", sess.run(train_acc_avg))

        for k in range(3):
            # test on test set
            data_test, label_test = load_test_data()
            data_after_test = data_test.eval(session=tf.Session())
            # print("training on {}".format(step))
            data_test = np.reshape(data_after_test, (
                BATCH_SIZE,
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS))
            acc_test = sess.run(accuracy, feed_dict={x: data_test, y_: label_test})
            print("Testing {}, accuracy on test set is {}".format(k+1,acc_test))
        #
        # print("saving model")
        # saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))


if __name__ == '__main__':
    load_data()
    train()

