import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from keras.losses import MeanSquaredError

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0 # Exploration vs exploration
        self.epsilon_decay_rate = 0.999
        self.min_epsilon = 0.2
        self.gamma = 0.9 # Discount factor
        self.update_rate = 100
        self.replay_buffer = deque(maxlen=2000)
        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.target_network.set_weights(self.main_network.get_weights())
        self.mse_loss = MeanSquaredError()  # (True, Pred)
        self.optimizer_function = Adam(learning_rate=0.001)
        pass

    def build_network(self):
        model = Sequential()
        model.add(Dense(units=32, input_shape=(self.state_size,)))
        model.add(Activation('relu'))
        model.add(Dense(units=64))
        model.add(Activation('relu'))
        model.add(Dense(units=32))
        model.add(Activation('relu'))
        # Don't softmax the final layer. The aim is to get all Q(s, a) values, not just the Q value for a single action
        model.add(Dense(units=self.action_size, activation='linear'))
        #model.compile(loss='mse', optimizer=Adam())
        return model

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))  # push it into the queue
        pass

    def epsilon_greedy(self, state):
        # Generate random number
        if random.uniform(0,1) < self.get_epsilon():
            # Below epsilon, explore
            Q_values = np.random.randint(self.action_size)
        else:
            # Otherwise, exploit using the main network
            Q_values = int(tf.argmax(self.main_network.predict(state, verbose=0)[0]))
            pass
        return Q_values

    def train(self, batch_size):
        # Get a mini batch from the replay memory
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target_Q = reward + self.gamma*np.amax(self.target_network.predict(next_state, verbose=0))
            else:
                target_Q = reward
                pass

            Q_values = self.main_network.predict(state, verbose=0)
            Q_values[0][action] = target_Q  # batch size = 1
            self.main_network.fit(state, Q_values, epochs=1, verbose=0)
        pass

    def train_double_DQN(self, batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if not done:
                # Select action with the maximum Q-value from the main network
                next_action = np.argmax(self.main_network.predict(next_state, verbose=0)[0])

                # Evaluate the Q-value of the selected action using the target network
                target_Q = reward + self.gamma * self.target_network.predict(next_state, verbose=0)[0][next_action]
            else:
                target_Q = reward

            Q_values = self.main_network.predict(state, verbose=0)
            Q_values[0][action] = target_Q
            #self.main_network.fit(state, Q_values, epochs=1, verbose=0)
            self.train_NN(input=state, target = Q_values)
            pass
        pass

    def train_NN(self, input, target):
        with tf.GradientTape() as tape:
            logits = self.main_network(input)
            loss_value = self.mse_loss(y_true=target, y_pred=logits)
            gradients = tape.gradient(loss_value, self.main_network.trainable_weights)
            self.optimizer_function.apply_gradients(zip(gradients, self.main_network.trainable_weights))
            pass
        pass



    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay_rate
        pass

    def get_epsilon(self):
        return max(self.epsilon, self.min_epsilon)

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())
        pass


