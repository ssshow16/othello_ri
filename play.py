import numpy as np
import os
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim

from board import Board  # from part 1 of this series

BATCH_SIZE = 32
UPDATE_FREQ = 4
TAU = 1E-3  # Rate to update target network toward primary network.
Y = 0.99
LOAD_MODEL = False  # Load a saved model?
PATH = './dqn'  # Path to save model.
BUFFER_SIZE = 5E4  # Num. moves to keep in buffer.
H_SIZE = 64  # Num. filters on final convolution layer.
NUM_GAMES = int(1E4)
SAVE_GAMES = int(1E3)


class QNetwork(object):
    def __init__(self, h_size=H_SIZE):
        self.current_player = 0

        self.scalar_input = tf.placeholder(shape=[None, 64], dtype=tf.int32)
        self._n_scalars = tf.shape(self.scalar_input)[0]
        self.move_mask = tf.placeholder(shape=[None, 64], dtype=tf.float32)

        self.board_onehot = tf.one_hot(self.scalar_input, 3, dtype=tf.float32)
        self._X = tf.split(self.board_onehot, 3, 2)
        self._Y = tf.transpose(
            tf.stack([
                tf.ones([self._n_scalars, 64, 1]),
                self._X[self.current_player],
                self._X[1 - self.current_player],
            ])
        )
        self.pre_pad = tf.reshape(self._Y, (-1, 8, 8, 3))
        self._pads = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        self.board_state = tf.pad(self.pre_pad, self._pads, 'CONSTANT')

        # Convolution layers decreasing board size each step
        self._conv1 = tf.tanh(slim.conv2d(
            inputs=self.board_state, num_outputs=16, kernel_size=[3, 3],
            stride=[1, 1], padding='VALID', biases_initializer=None))
        self._conv2 = tf.tanh(slim.conv2d(
            inputs=self._conv1, num_outputs=32, kernel_size=[2, 2],
            stride=[1, 1], padding='VALID', biases_initializer=None))
        self._conv3 = tf.tanh(slim.conv2d(
            inputs=self._conv2, num_outputs=32, kernel_size=[2, 2],
            stride=[1, 1], padding='VALID', biases_initializer=None))
        self._conv4 = tf.tanh(slim.conv2d(
            inputs=self._conv3, num_outputs=32, kernel_size=[2, 2],
            stride=[1, 1], padding='VALID', biases_initializer=None))
        self._conv5 = tf.tanh(slim.conv2d(
            inputs=self._conv4, num_outputs=32, kernel_size=[2, 2],
            stride=[1, 1], padding='VALID', biases_initializer=None))
        self._conv6 = tf.tanh(slim.conv2d(
            inputs=self._conv5, num_outputs=32, kernel_size=[2, 2],
            stride=[1, 1], padding='VALID', biases_initializer=None))
        self._conv7 = tf.tanh(slim.conv2d(
            inputs=self._conv6, num_outputs=32, kernel_size=[2, 2],
            stride=[1, 1], padding='VALID', biases_initializer=None))
        self._conv8 = slim.flatten(tf.tanh(slim.conv2d(
            inputs=self._conv7, num_outputs=h_size, kernel_size=[2, 2],
            stride=[1, 1], padding='VALID', biases_initializer=None)))

        # Break apart for Dueling DQN
        self._streamA, self._streamV = tf.split(self._conv8, 2, 1)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([int(h_size / 2), 64]))
        self.VW = tf.Variable(xavier_init([int(h_size / 2), 1]))
        self.advantage = tf.matmul(self._streamA, self.AW)
        self.value = tf.matmul(self._streamV, self.VW)

        # Combine together to get final Q-values.
        self._Q_all = self.value + tf.subtract(
            self.advantage,
            tf.reduce_mean(self.advantage, axis=1, keep_dims=True))
        self.Q_out = tf.multiply(tf.exp(self._Q_all), self.move_mask)
        self.predict = tf.multinomial(tf.log(self.Q_out), 1)[0][0]

        # Obtain loss function by taking the sum-of-squares difference
        # between the target and prediction Q-values.
        self.Q_target = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self._actions_onehot = tf.one_hot(self.actions, 64, dtype=tf.float32)

        self._Q = tf.reduce_sum(tf.multiply(
            self.Q_out, self._actions_onehot), axis=1)
        self._td_error = tf.square(self.Q_target - self._Q)
        self._loss = tf.reduce_mean(self._td_error)
        self._trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update_model = self._trainer.minimize(self._loss)


class ExperienceBuffer():
    def __init__(self, buffer_size=BUFFER_SIZE):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        tot_len = len(self.buffer) + len(experience)
        if tot_len >= self.buffer_size:
            self.buffer = self.buffer[int(tot_len - self.buffer_size):]
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(
            np.array(random.sample(self.buffer, size)), [size, 6])


def move_index_to_coord(idx):
    coord_move = np.zeros(64)
    coord_move[idx] = 1
    return tuple(np.argwhere(coord_move.reshape(8, 8))[0])


def update_target_graph(tf_vars, tau=TAU):
    """
    Update parameters of target network.

    target = tau*primary + (1-tau)*target"""
    total_vars = len(tf_vars)
    op_holder = []
    for idx, var in enumerate(tf_vars[:int(total_vars / 2)]):
        op_holder.append(tf_vars[idx + int(total_vars / 2)].assign(
            (var.value() * tau) + (
                (1 - tau) * tf_vars[idx + total_vars // 2].value()
            )))
    return op_holder


def update_target(op_holder, sess):
    for op in op_holder:
        sess.run(op)


tf.reset_default_graph()
main_QN = QNetwork()
Q_targetN = QNetwork()

init = tf.global_variables_initializer()
saver = tf.train.Saver()
target_ops = update_target_graph(tf.trainable_variables())
my_buffer = ExperienceBuffer()

r_list = []  # Final score of each game.
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(PATH):
    os.makedirs(PATH)

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)

        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(PATH)
        saver.restore(sess, ckpt.model_checkpoint_path)

        update_target(target_ops, sess)
        b = Board(verbose=False)
        b.print_board()

        while b.game_over is False:

            for row in b.board_state:
                vals = tuple([cell.is_valid_move for cell in row])
                print(b.print_row % vals)

            next_disk = input("your turn?")

            b.human_move(next_disk)

            valid_moves = np.where(
                [[x.is_valid_move for x in row] for row in b.board_state])

            move_mask = np.zeros_like(b.board_state, dtype='float32')
            move_mask[valid_moves] = 1
            move_mask = move_mask.flatten()[np.newaxis, :]

            s = b.board_state_list()

            main_QN.current_player = b.current_player
            a = sess.run(
            main_QN.predict,
                feed_dict={
                    main_QN.scalar_input: s,
                    main_QN.move_mask: move_mask
                })
            b.coord_move(move_index_to_coord(a))
            b.print_board()

        print([b.p0_score, b.p1_score])

        score = [b.p0_score, b.p1_score]
        my_score = score[0]
        their_score = score[1]
        if my_score > their_score:
            print("winner")
        else:
            print("loser")

