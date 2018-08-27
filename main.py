import tensorflow as tf
import gym
import os
import numpy as np
import random


class ReplayMemory:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = np.empty(shape=buffer_size, dtype=np.object)
        self.index = 0
        self.length = 0

    def append(self, data):
        self.buffer[self.index] = data
        self.length = min(self.length + 1, self.buffer_size)
        self.index = (self.index + 1) % self.buffer_size

    def sample(self):
        index = np.random.randint(self.length)
        return self.buffer[index]


class A3C:
    def __init__(self, sess, input_size, output_size):
        self.sess = sess

        self.n_in = input_size
        self.n_out = output_size

        self.lr = 0.00025
        self.gamma = 0.999
        self.rand_chance = 0.15
        self.entropy_beta = 0.01

        self.s = tf.placeholder(tf.float32, [1, self.n_in], "state")
        self.a = tf.placeholder(tf.int32, None, "action")
        self.r = tf.placeholder(tf.float32, None, "reward")
        self.s_ = tf.placeholder(tf.float32, [1, self.n_in], "next_state")
        self.v_ = tf.placeholder(tf.float32, None, "next_v")
        self.end = tf.placeholder(tf.float32, None, "done")

        with tf.variable_scope("network"):
            self.shared = tf.layers.dense(self.s, 200, activation=tf.nn.relu)

            self.hidden_v = tf.layers.dense(self.shared, 200, activation=tf.nn.relu)
            self.v = tf.squeeze(tf.layers.dense(self.hidden_v, 1))

            self.hidden_policy = tf.layers.dense(self.shared, 200, activation=tf.nn.relu)
            self.policy = tf.layers.dense(self.hidden_policy, self.n_out)

        self.probs = tf.nn.softmax(self.policy)
        self.real_action = tf.argmax(self.probs[0, :], 0)
        self.rand_action = tf.squeeze(tf.multinomial(tf.log(self.probs), num_samples=1))

        with tf.variable_scope("training"):
            self.critic_loss = self.r + self.gamma * self.v_ * (1.0 - self.end) - self.v

            self.log_prob = tf.log(self.probs[0, self.a] + 1e-6)
            self.actor_loss = -tf.reduce_mean(self.log_prob * tf.stop_gradient(self.critic_loss))

            entropy = tf.reduce_sum(self.probs * tf.log(self.probs + 1e-6))
            self.entropy_loss = self.entropy_beta * entropy

            self.total_loss = 0.5 * tf.square(self.critic_loss) + self.actor_loss + self.entropy_loss

            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = optimizer.minimize(self.total_loss)

    def train(self, s, a, r, s_, end):
        s = s[np.newaxis, :]
        s_ = s_[np.newaxis, :]
        v_ = self.sess.run(self.v, feed_dict={self.s: s_})

        feed_dict = {self.s: s, self.a: a, self.r: r, self.s_: s_, self.v_: v_, self.end: end}
        _, training_error = self.sess.run([self.train_op, self.total_loss], feed_dict=feed_dict)
        return training_error

    def get_rand_action(self, s):
        s = s[np.newaxis, :]
        # if random.random() < self.rand_chance:
        #     return random.randint(0, self.n_out)
        return self.sess.run(self.rand_action, feed_dict={self.s: s})

    def get_real_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.real_action, feed_dict={self.s: s})


n_epochs = 500
n_games = 10
n_learn = 1000
n_max_iter = 1300

save_index = int(input("Save Index: "))
session = tf.InteractiveSession()

env = gym.make('LunarLander-v2')
env._max_episode_steps = n_max_iter
env_obs_n = env.observation_space.shape[0]
env_act_n = env.action_space.n

memory = ReplayMemory(50000)
agent = A3C(session, env_obs_n, env_act_n)

tf.global_variables_initializer().run()
saver = tf.train.Saver()

checkpoint = "./run-" + str(save_index) + ".ckpt"
if os.path.isfile(checkpoint + ".meta"):
    saver.restore(session, checkpoint)
elif save_index != 0:
    raise Exception("Session data not found!!")

user_select = input("Show result (y / n) ")
if user_select == "y":
    while True:
        obs = env.reset()
        for tick in range(n_max_iter):
            env.render()
            actor_val = agent.get_real_action(obs)
            print(actor_val)
            obs, reward, done, info = env.step(actor_val)
            if done:
                break

else:
    root_logdir = "tf_logs/"
    logdir = "{}/run-{}/".format(root_logdir, save_index + 1)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    for epoch in range(n_epochs):
        print("Starting Epoch " + str(epoch))
        for game in range(n_games):
            total_reward = 0
            # noinspection PyRedeclaration
            obs = env.reset()
            actions = [0 for _ in range(env_act_n)]

            for tick in range(n_max_iter):
                actor_val = agent.get_rand_action(obs)
                new_obs, reward, done, info = env.step(actor_val)

                if tick == n_max_iter - 1 and not done:
                    reward -= 100

                actions[actor_val] += 1
                memory.append((obs, actor_val, reward, new_obs, int(done)))
                total_reward += reward

                obs = new_obs
                if done:
                    break
            print(total_reward)
            sum_actions = 0
            for action in range(len(actions)):
                sum_actions += actions[action]
            print([x / sum_actions for x in actions])
            summary = tf.Summary()
            summary.value.add(tag='Total Reward', simple_value=total_reward)
            file_writer.add_summary(summary, epoch * n_games + game)

        print("Starting Learning")
        for learn in range(n_learn):
            sam_s, sam_a, sam_r, sam_s_, end = memory.sample()
            error = agent.train(sam_s, sam_a, sam_r, sam_s_, end)
            value = session.run(agent.v, feed_dict={agent.s: sam_s.reshape(1, 8)})

            summary = tf.Summary()
            summary.value.add(tag='Total Error', simple_value=error)
            summary.value.add(tag='Value', simple_value=value)
            file_writer.add_summary(summary, epoch * n_learn + learn)

    env.close()
    saver.save(session, "./run-" + str(save_index + 1) + ".ckpt")
    file_writer.flush()
    file_writer.close()
