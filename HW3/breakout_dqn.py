
"""
MSIA 490 HW3 
Sungsoo Lim

DQN for Breakout-v0

Code based on Ghani's code from the lab session - Please excuse me for the code being very similar. It was helpful for me to have an example code. 

"""

import numpy as np
import gym
import tensorflow as tf
from collections import deque

import os
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

env = gym.make('Breakout-v0')
obs = env.reset()

#Compatibility between tf.1.x and tf.2.x
tf.compat.v1.disable_eager_execution()

#Parameters for the convolution networks and the input image
H = 80 #height
W= 80 #width
C = 1 #channels

#Momentum optimizer
learning_rate = 0.001
momentum = 0.95

#Parameters for iterations
n_steps = 1000000 #10 million iterations
training_start = 10000 #start training after first 10000 iterations 
training_interval = 4 #frame-skip 
sv = 1000 #save 
checkpoint = './packman.ckpt'
cp = 10000 #copy
discount_rate = 0.95
skip_start = 90 #skip the first 90 actions in the game when calculating the reward 
batch_size = 50 

#Pre-process the observation images
def preprocess_observation(obs):
	image = obs[35:195]
	image = image[::2,::2,0]
	image[image == 144] = 0
	image[image == 109] = 0
	image[image != 0] = 1
	image = np.reshape(image.astype(np.float).ravel(),[80,80])
	image = np.expand_dims(image, axis=0)
	return image

replay_memory_size = 500000
replay_memory = deque([],maxlen=replay_memory_size)

eps_min = 0.1
eps_max = 1.0
eps_decay = 2000000

#greedily update the epsilon value 
def epsilon_greedy(q_values, step):
	epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step/eps_decay_steps)
	if np.random.rand() < epsilon:
		return np.random.randint(n_outputs)
	else:
		return np.argmax(q_values)

#Replay buffer
class Replay:
	def __init__(self,max_length):
		self.max_length = max_length
		self.buffer = np.empty(shape=max_length, dtype=np.object)
		self.index = 0
		self.length = 0

	def append(self,data): #append the data to the buffer
		self.buffer[self.index] = data
		self.length = min(self.length + 1, self.max_length)
		self.index = (self.index + 1) % self.max_length

	def sample(self, batch_size, with_replacement=True): #Sample from the buffer with replacement
		if with_replacement:
			indices = np.random.randint(self.length,size=batch_size)
		else:
			indices = np.random.permutation(self.length)[:batch_size]
		return self.buf[indices]

def sample_memories(batch_size):
	cols = [[],[],[],[],[]] #state, action, reward, next_state, 1.0-done
	for memory in replay_memory.sample(batch_size):
		for col, val in zip(cols,memory):
			col.append(val)

	cols = [np.array(col) for col in cols]

	return cols[0],cols[1],cols[2].reshape(-1,1),cols[3],cols[4].reshape(-1,1) #state, action, reward, next_state, 1.0-done

replay_memory_size = 500000
replay_memory = Replay(replay_memory_size) #initialize replay_memory

#The networks are initialized serially
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ['SAME'] * 3
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64*10*10
n_hidden = 256
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n #action space
initializer = tf.compat.v1.variance_scaling_initializer() #variance scaling initializer 

def DeepQNetwork(X_state,name):
	prev_layer = X_state / 255.0
	with tf.compat.v1.variable_scope(name) as scope: #Three convolution networks at once 
		for n_maps, kernel_size, strides, padding, activation in zip(conv_n_maps, conv_kernel_sizes, conv_strides, conv_paddings, conv_activation):
			previous_layer = tf.compat.v1.layers.conv2d(prev_layer, filters=n_maps, kernel_size = kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer = initializer)
		
		#flatten the output layer for keras dense network
		last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1,n_hidden_in])
		
		#hidden network
		hidden = tf.compat.v1.layers.dense(last_conv_layer_flat, n_hidden,activation=hidden_activation,kernel_initializer = initializer)
		
		#logits output
		outputs = tf.compat.v1.layers.dense(hidden, n_outputs, kernel_initializer = initializer)

	#determine the variables to be copied from online to target 
	trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,scope=scope.name)
	trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}

	return outputs, trainable_vars_by_name

#Initialize the q-values and the variables to be copied
tf.compat.v1.reset_default_graph()
X_state = tf.compat.v1.placeholder(tf.float32, shape=[None, H, W, C])
online_q_values, online_vars = DeepQNetwork(X_state,name='DeepQNetwork/online')
target_q_values, target_vars = DeepQNetwork(X_state,name='DeepQNetwork/target')

#initialize the variable copy from online to target 
copy = [target_var.assign(online_vars[var_name]) for var_name, target_var in target_vars.items()]
online_to_target = tf.group(*copy)

#Define the q_values and the loss function 
with tf.compat.v1.variable_scope('train'):
	action = tf.compat.v1.placeholder(tf.int32, shape=[None])
	y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])
	
	#Q(a) - qvalues 
	qvalue = tf.reduce_sum(input_tensor=online_q_values * tf.one_hot(action, n_outputs), axis=1, keepdims=True)
	
	#MSE with clipping
	error = tf.abs(y-qvalue)
	clipped = tf.clip_by_value(error,0.0,1.0)
	linear_error = 2 * (error - clipped)
	loss = tf.reduce_mean(input_tensor=tf.square(clipped_error) + linear_error)
	
	#have an global step variable for keeping track of epsilon-greedy update of the action  
	global_step = tf.Variable(0, trainable=False, name='global_step') 
	optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
	training_op = optimizer.minimize(loss,global_step=global_step)

#Intiialize the variables for the q-values 
init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver() #save check-points 

#Q-value iteration variables
game_length = 0
total_max_q = 0
mean_max_q = 0.0
iteration = 0
done = True

with tf.compat.v1.Session() as sess:

	#Update from saved session or initialize 
	if os.path.isfile(checkpoint + ".index"):
		saver.restore(sess,checkpoint)
	else:
		init.run()
		online_to_target.run()

	while True: #while not done 

		step = global_step.eval() #keep track of the iteration steps 

		if step >= n_steps:
			break

		iteration += 1

		print ('Iteration: %s. Mean Max-Q: %f.' % (iteration, mean_max_q))
		
		if done:
			obs = env.reset()
			for skip in range(skip_start): #skip first 90
				obs, reward, done, info = env.step(0)
			state = preprocess_observation(obs)
		
		#calculate q_values for the online network
		q_values = online_q_values.eval(feed_dict={X_state: [state]})
		action = epsilon_greedy(q_values,step) #take greedy action
		obs,reward,done,info = env.step(action)
		next_state = preprocess_observation(obs)

		#Append to the replay buffer 
		replay_memory.append((state,action,reward,next_state,1.0-done))
		state = next_state

		#Calculate the mean_max_q and output to a txt file 
		total_max_q += q_values.max()
		game_length += 1
		if done:
			mean_max_q = total_max_q / game_length
			text_file = open("Output_mean_max_q_breakout.txt", "a")
			text_file.write('%f\n' %(mean_max_q))
			text_file.close()
			total_max_q = 0.0
			game_length = 0

		#Only train during the given interval 
		if iteration < training_start or iteration % training_interval != 0:
			continue

		#Optimize based on the loss for the clipped error
		X_state_val, X_action_val, rewards, X_next_state_val, continues = (sample_memories(batch_size))
		next_q_values = target_q_values.eval(feed_dict={X_state: X_next_state_val})
		max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
		y_val = rewards + continues * discount_rate * max_next_q_values #loss for the clipped error - depends on whether it is continued (continues) 

		sess.run([training_op, loss], feed_dict={X_state: X_state_val, action: X_action_val, y: y_val})

		#Copy from online to target 
		if step % cp == 0:
			copy_online_to_target.run()

		#Save to checkpoint 
		if step % sv == 0:
			saver.save(sess,checkpoint)


#Run the game and plot the results 
n_games = 200
num = 1
total_score = []
with tf.compat.v1.Session() as sess:
	saver.restore(sess,checkpoint_path)
	obs = env.reset()
	score = 0
	while num < n_games:
		#print(num)
		state = preprocess_observation(obs)
		q_values = target_q_values.eval(feed_dict={X_state: [state]})
		action = np.argmax(q_values)
		#action= np.random.randint(n_outputs)
		obs,reward,done,info = env.step(action)
		score += reward
		if reward != 0.0:
			print(reward)
		next_state = preprocess_observation(obs)
		state = next_state
		if done:
			total_score.append(score)
			obs = env.reset()
			num+=1
			score = 0
