import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU') #set a specific gpu device based on checking
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
else:
    print('Running on CPU')

H = 32 # number of hidden layer neurons
D = 6400 #input dimensions
num_actions = 2 #up or down

#Policy estimation DNN model
class PolicyModel(tf.keras.Model):
    def __init__(self):
        super(PolicyModel, self).__init__(self)
        self.num_actions = num_actions
        self.hidden0 = tf.keras.layers.Dense(units=H, activation='relu',kernel_initializer='he_normal')
        self.hidden1 = tf.keras.layers.Dense(units=H, activation='relu',kernel_initializer='he_normal')
        self.logits = tf.keras.layers.Dense(self.num_actions,kernel_initializer='he_normal', activation = 'tanh') #action probabilities estimation

    def call(self, obs):
        x = tf.convert_to_tensor(obs)
        hidden0 = self.hidden0(x)
        hidden1 = self.hidden1(hidden0)
        out = self.logits(hidden1)
        return out

#Value estimation DNN model
class ValueModel(tf.keras.Model):
    def __init__(self):
        super(ValueModel, self).__init__(self)
        self.hidden0 = tf.keras.layers.Dense(units=H, activation='relu',kernel_initializer='he_normal')
        self.hidden1 = tf.keras.layers.Dense(units=H, activation='relu',kernel_initializer='he_normal')
        self.logits = tf.keras.layers.Dense(1,kernel_initializer='he_normal', activation = None) #value estimation

    def call(self, obs):
        x = tf.convert_to_tensor(obs)
        hidden0 = self.hidden0(x)
        hidden1 = self.hidden1(hidden0)
        out = self.logits(hidden1)
        return out
#Agent
class Agent:
    def __init__(self, policymodel, valuemodel):
        self.policymodel = policymodel
        self.valuemodel = valuemodel
        self.memory = self.Memory()
        self.policymodel.build((1,D))
        self.valuemodel.build((1,D))
        self.optimizer_policy = tf.keras.optimizers.RMSprop(learning_rate = 0.004, rho = 0.99)
        self.optimizer_value = tf.keras.optimizers.Adam(learning_rate = 0.002)

    #loss for the REINFORCE algorithm - learned baseline (value estimation) needs to be subtracted
    def loss(self,rewards,neg_log):
        loss = tf.reduce_mean(rewards*neg_log)
        return loss

    def neg_log(self, action, logits):
        return tf.squeeze(tf.nn.softmax_cross_entropy_with_logits(labels=action, logits=logits))

    #loss for the value estimation model - mean squared error between the learned baseline and rewards 
    def loss_baseline(self,val,rewards):
        return tf.reduce_mean(tf.math.squared_difference(val, rewards))

    def value_estimation(self,obs):
        value_est = self.valuemodel(obs.reshape(1,-1))
        return tf.squeeze(value_est)

    #select action from action-probabilities estimation
    def action(self, obs):
        logits = self.policymodel(obs.reshape(1,-1))
        action = self.select_action(logits=logits)
        action = tf.keras.utils.to_categorical(action,num_classes=num_actions)
        return action, logits

    def select_action(self, logits):
        return tf.random.categorical(logits,1)

    def zero_vector(self,grad_vector):
        for ix, grad in enumerate(grad_vector):
            grad_vector[ix] = grad * 0

    def prepro(self,I):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        return I.astype(np.float).ravel()

    def train(self, env, num_episodes = 10000):
        
        #obtain state space
        obs = env.reset()

        #vectorize grads for policy and value networks
        gradvector_policy = self.policymodel.trainable_weights
        gradvector_value = self.valuemodel.trainable_weights

        #initialize the vectorized grads to 0
        self.zero_vector(gradvector_policy)
        self.zero_vector(gradvector_value)

        episode_number = 1 #episode iteration number
        batch_size = 5 #batch size
        total_rewards = [] #save total rewards for each episode
        prev_obs = None #preprocessing parameter
        while episode_number < num_episodes:
            
            #preprocessing of images
            new_obs = agent.prepro(obs)
            new_obs = new_obs - prev_obs if prev_obs is not None else np.zeros(D)
            prev_obs = new_obs

            #calculate gradients and save
            with tf.GradientTape() as policy_tape, tf.GradientTape() as value_tape:
                action, logits = self.action(new_obs) #action space and probabilities
                value_estimation = self.value_estimation(new_obs) #value estimation
                a = 2 + np.argmax(action) #down or up
                obs, reward, done, _ = env.step(a) #take action
                reward_value = reward - value_estimation #reward for value estimation model
                loss = self.loss(reward, self.neg_log(action,logits)) - value_estimation #REINFORCE loss with baseline for the policy model
                loss_value = self.loss_baseline(value_estimation,reward) #loss for the value estimation model - mean squared error between reward and value estimation
            grads_policy = policy_tape.gradient(loss,self.policymodel.trainable_weights)
            grads_value = value_tape.gradient(loss_value,self.valuemodel.trainable_weights)
            
            self.memory.save(reward, reward_value, grads_policy, grads_value) #save for backpropagation

            if done:

                #update grad values for policy and value estimators

                #policy network - discounted rewards
                for grads, r in zip(self.memory.gradients,self.memory.discounted_rewards): 
                    for ix, grad in enumerate(grads):
                        gradvector_policy[ix] += grad * r

                #value estimation network - discounted rewards with baseline
                for grads_value, r_value in zip(self.memory.gradients_value,self.memory.discounted_rewards_value):
                    for iy, grad_value in enumerate(grads_value):
                        gradvector_value[iy] += grad_value * r_value

                #update networks for every batch size 
                if episode_number % batch_size == 0:
                    self.optimizer_policy.apply_gradients(zip(gradvector_policy, policymodel.trainable_variables))
                    self.optimizer_value.apply_gradients(zip(gradvector_value, valuemodel.trainable_variables))
                    self.zero_vector(gradvector_policy)
                    self.zero_vector(gradvector_value)

                #save and print
                total_rewards.append(self.memory.sum_rewards) #save sum of rewards for the episode
                print ('Episode: %s. Reward total: %f.' % (episode_number, self.memory.sum_rewards))
                episode_number += 1

                #clear variables for the next episode
                self.memory.clear()
                obs = env.reset()
                prev_obs = None

        return total_rewards

    #Save for backpropagation for each batch size
    class Memory():
        def __init__(self):
            self.__rewards = []
            self.__rewards_value = []
            self.__gradients = []
            self.__gradients_value = []

        def save(self, reward, reward_value, gradient, grad_value):
            gradient = [item.numpy() for item in gradient]
            grad_value = [item.numpy() for item in grad_value]
            self.__rewards.append(reward)
            self.__rewards_value.append(reward_value)
            self.__gradients.append(gradient)
            self.__gradients_value.append(grad_value)

        def clear(self):
            self.__rewards = []
            self.__rewards_value = []
            self.__gradients = []
            self.__gradients_value = []

        @property
        def sum_rewards(self):
            return np.sum(self.rewards)

        @property
        def rewards(self):
            return np.array(self.__rewards)

        @property
        def rewards_value(self):
            return np.array(self.__rewards_value)

        @property
        def gradients(self):
            return np.array(self.__gradients)

        @property
        def gradients_value(self):
            return np.array(self.__gradients_value)
        
        @property
        def discounted_rewards(self, gamma=0.99):
            discounted_r = np.zeros_like(self.rewards)
            running_add = 0
            for t in reversed(range(0, len(self.rewards))):
                if self.rewards[t] != 0: 
                    running_add = 0 # game boundary
                running_add = running_add * gamma + self.rewards[t]
                discounted_r[t] = running_add

            # standardize the rewards
            discounted_r -= np.mean(discounted_r)
            discounted_r /= np.std(discounted_r)

            return discounted_r

        @property
        def discounted_rewards_value(self, gamma=0.99):
            discounted_r = np.zeros_like(self.rewards_value)
            running_add = 0
            for t in reversed(range(0, len(self.rewards_value))):
                if self.rewards_value[t] != 0: 
                    running_add = 0 # game boundary
                running_add = running_add * gamma + self.rewards_value[t]
                discounted_r[t] = running_add

            # standardize the rewards
            discounted_r -= np.mean(discounted_r)
            discounted_r /= np.std(discounted_r)

            return discounted_r

#Start learning
env = gym.make("Pong-v0")
env.reset()
policymodel = PolicyModel()
valuemodel = ValueModel()
agent = Agent(policymodel,valuemodel)

num_episodes = 100 #number of total episodes
batch_size = 5 #batch size - update networks every batch size
total_rewards = agent.train(env,num_episodes) #returns total rewards for the given number of episodes and batch size
batch_mean = [np.mean(total_rewards[max(0,i-batch_size):i+1]) for i in range(len(total_rewards))] #batch mean of the total rewards

#Plotting
plt.plot(batch_mean)
plt.xlabel('Episodes')
plt.ylabel('Batch mean rewards')
plt.ylim((-21.0, -10.0))
plt.title('REINFORCE with Baseline for Pong-v0')
plt.show()
