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
LOAD_MODEL = True  # Load a saved model?
PATH = './dqn'  # Path to save model.
BUFFER_SIZE = 5E4  # Num. moves to keep in buffer.
H_SIZE = 64  # Num. filters on final convolution layer.
NUM_GAMES = int(1E5)
SAVE_GAMES = int(1E3)


class QNetwork(object):
    def __init__(self, h_size=H_SIZE):
        self.current_player = 0

        self.scalar_input = tf.placeholder(shape=[None, 64], dtype=tf.int32)
        self._n_scalars = tf.shape(self.scalar_input)[0]  # 입력 상태의 갯수
        self.move_mask = tf.placeholder(shape=[None, 64], dtype=tf.float32)

        # 하나의 cell을 3차열 배열을 갖도록 한다. 0,1은 나와 상대의 입력이 있다면 해당 index에 1이 들어간다
        self.board_onehot = tf.one_hot(self.scalar_input, 3, dtype=tf.float32)
        self._X = tf.split(self.board_onehot, 3, 2)  # 0번 1번 인덱스의 값만 추출한다. board_onehot의 위치값
        self._Y = tf.transpose(
            tf.stack([
                tf.ones([self._n_scalars, 64, 1]), # 걍 1로 채운다.
                self._X[self.current_player],   # 나의 위치 (0이면 상대는 1)(1이면 상대는 0)
                self._X[1 - self.current_player],  # 상대의 위치
            ])
        )
        # [ 1, 내가 있다면 1 아니면 0, 상대가 있다면 1, 없다면 0] 배열을 만든다

        self.pre_pad = tf.reshape(self._Y, (-1, 8, 8, 3))  # 3개씩 묶는다
        self._pads = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        self.board_state = tf.pad(self.pre_pad, self._pads, 'CONSTANT')  # 위아래로  0으로 패딩한다

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
        self._Q_all = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keep_dims=True))
        self.Q_out = tf.multiply(tf.exp(self._Q_all), self.move_mask) # 가능한 곳을 미리 마스킹하고 그 값중에 좋은 값을 고른다??

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
        if LOAD_MODEL is True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(PATH)
            saver.restore(sess, ckpt.model_checkpoint_path)
        update_target(target_ops, sess)
        for i in range(NUM_GAMES):
            coin_flip = random.randint(0, 1)
            episode_buffer = ExperienceBuffer()
            b = Board(verbose=False)
            s = b.board_state_list()

            r = 0
            while b.game_over is False:
                valid_moves = np.where(
                    [[x.is_valid_move for x in row] for row in b.board_state])

                move_mask = np.zeros_like(b.board_state, dtype='float32')
                move_mask[valid_moves] = 1
                move_mask = move_mask.flatten()[np.newaxis, :]

                if b.current_player == coin_flip:
                    # If my move.
                    main_QN.current_player = b.current_player
                    a = sess.run(
                        main_QN.predict,
                        feed_dict={
                            main_QN.scalar_input: s,
                            main_QN.move_mask: move_mask
                        })
                    b.coord_move(move_index_to_coord(a))
                    s1 = b.board_state_list()

                    total_steps += 1
                    episode_buffer.add(
                        np.reshape(np.array([s, a, r, s1, b.game_over, move_mask]), [1, 6]))

                    if i > 1 and total_steps % UPDATE_FREQ == 0:
                        # Get a random batch of experiences
                        # Perform Double-DQN update to the target Q-values
                        train_batch = my_buffer.sample(BATCH_SIZE)
                        main_QN.current_player = b.current_player
                        Q_targetN.current_player = b.current_player
                        A1 = sess.run(
                            main_QN.predict,
                            feed_dict={
                                main_QN.scalar_input:  np.vstack(train_batch[:, 3]),
                                main_QN.move_mask:  np.vstack(train_batch[:, 5])})
                        Q2 = sess.run(
                            Q_targetN.Q_out,
                            feed_dict={
                                Q_targetN.scalar_input: np.vstack(train_batch[:, 3]),
                                Q_targetN.move_mask: np.vstack(train_batch[:, 5])})
                        doubleQ = Q2[range(BATCH_SIZE), A1]
                        Q_target = train_batch[:, 2] + Y * doubleQ

                        # Update the network with our target values.
                        _ = sess.run(
                            main_QN.update_model,
                            feed_dict={
                                main_QN.scalar_input: np.vstack(train_batch[:, 0]),
                                main_QN.move_mask: np.vstack(train_batch[:, 5]),
                                main_QN.Q_target: Q_target,
                                main_QN.actions: train_batch[:, 1]})

                        # Set target network equal to primary network.
                        update_target(target_ops, sess)
                else:
                    # If opponent's move.
                    Q_targetN.current_player = b.current_player
                    a = sess.run(Q_targetN.predict,
                                 feed_dict={
                                     Q_targetN.scalar_input: s,
                                     Q_targetN.move_mask: move_mask
                                 })
                    b.coord_move(move_index_to_coord(a))
                    # s1 = b.board_state_list[np.newaxis, :]
                    s1 = b.board_state_list()

                if b.game_over is True:
                    score = [b.p0_score, b.p1_score]
                    my_score = score[coin_flip]
                    their_score = score[1 - coin_flip]
                    if my_score > their_score:
                        r = 1
                    else:
                        r = -1
                s = s1

            my_buffer.add(episode_buffer.buffer)
            r_list += [r]

            # Periodically save the model.
            if i % SAVE_GAMES == 0:
                saver.save(sess, '%s/model-%s.cptk' % (PATH, i))
                print('Saved Model')
            if len(r_list) % 10 == 0:
                print(i + 1, total_steps, np.mean(r_list[-10:]))

        # Save at completion.
        saver.save(sess, '%s/model-%s.cptk' % (PATH, i))