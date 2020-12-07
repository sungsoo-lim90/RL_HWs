import numpy as np
import tensorflow as tf

"""

MSIA 490 HW4 - Sungsoo Lim

Calculate the regret from the trained UCB model on the Jester data

Training = A, b, theta

"""

def sample_jester_data(dataset, context_dim = 32, num_actions = 8, num_contexts = 19181,
shuffle_rows=True, shuffle_cols=False):

	"""Samples bandit game from (user, joke) dense subset of Jester dataset.

	Args:

	file_name: Route of file containing the modified Jester dataset.
	context_dim: Context dimension (i.e. vector with some ratings from a user).
	num_actions: Number of actions (number of joke ratings to predict).
	num_contexts: Number of contexts to sample.
	shuffle_rows: If True, rows from original dataset are shuffled.
	shuffle_cols: Whether or not context/action jokes are randomly shuffled.
	
	Returns:

	dataset: Sampled matrix with rows: (context, rating_1, ..., rating_k).
	opt_vals: Vector of deterministic optimal (reward, action) for each context.

	"""
	
	#np.random.seed(0)

	#with tf.io.gfile.GFile(file_name, 'rb') as fid:

	#	dataset = np.load(fid)

	if shuffle_cols:
		dataset = dataset[:, np.random.permutation(dataset.shape[1])]
	
	if shuffle_rows:
		np.random.shuffle(dataset)
		
	dataset = dataset[:num_contexts, :]
		
	assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'
		
	opt_actions = np.argmax(dataset[:, context_dim:], axis=1) #needed for regret calculation
		
	opt_rewards = np.array([dataset[i, context_dim + a] for i, a in enumerate(opt_actions)]) #needed for regret calculation
		
	return dataset, opt_rewards, opt_actions

#Parameters
num_train = 50
num_train_users = 18000
num_features = 32
num_actions = 8
file_name = 'jester_data_40jokes_19181users.npy'

#Split into training and test data sets
with tf.io.gfile.GFile(file_name,'rb') as fid:
	data_set = np.load(fid)

np.random.shuffle(data_set)
dataset_train = data_set[:num_train_users,:]
dataset_test = data_set[num_train_users:,:]
num_test_users = len(dataset_test)

choices = np.zeros((num_train,num_train_users)) #(10000,18000)
rewards = np.zeros((num_train,num_train_users)) #(10000,18000)
b = np.zeros((num_train_users,num_actions,num_features)) #(18000,8,32) - loop through for each user
A = np.zeros((num_train_users,num_actions,num_features,num_features)) #(18000, 8, 32,32) - loop through for each user

#UCB hyperparameter
alph_max = 10
alph_min = 0.001
alph_total = np.linspace(alph_min,alph_max,num_train)
alpha_total= alph_total[::-1]

#initialize A
for j in range(num_train_users):
	A_i_j = A[j,:,:,:]
	for a in range(num_actions):
		A_i_j[a] = np.identity(num_features)

#initialize theta, p, and regret
th_hat = np.zeros((num_actions,num_features))
p = np.zeros(num_actions)
regret = np.zeros((num_train,num_train_users))

for i in range(num_train):
	
	print(i)
	
	dataset, opt_rewards, opt_actions = sample_jester_data(dataset_train,num_contexts=num_train_users,shuffle_rows=False,shuffle_cols=False)

	x_i = dataset[:,:num_features]

	alph = alpha_total[i]

	for j in range(len(x_i)): #for each user, find the UCB solution action

		x_i_j = x_i[j,:] #(1,32) #feature vector

		A_i_j = A[j,:,:,:] #(8,32,32)

		b_i_j = b[j,:,:] #(8,32)

		for a in range(num_actions):

			A_inv = np.linalg.inv(A_i_j[a]) #32x32

			th_hat[a] = A_inv.dot(b_i_j[a]) #1x32
			
			ta = x_i_j.dot(A_inv).dot(x_i_j) #
			
			a_upper_ci = alph * np.sqrt(ta) #(18000,)

			#print(a_upper_ci)
			
			a_mean = th_hat[a].dot(x_i_j)#(18000,)

			#print(a_mean)
			
			p[a] = a_mean + a_upper_ci #(18000,)

		p = p + (np.random.random(len(p)) * 0.00000001) #(8,1)

		choices[i,j] = p.argmax() #(1,1)

		#reward based on dataset
		rewards[i,j] = dataset[j,num_features+int(choices[i,j])]

		A_i_j[int(choices[i,j])] += np.outer(x_i_j,x_i_j) #(32,32)

		b_i_j[int(choices[i,j])] += rewards[i,j] * x_i_j #(1,32)

	#regret for training
	regret[i] = i*opt_rewards - sum(rewards[:i,:],0)

dataset_test, opt_rewards_test, opt_actions_test = sample_jester_data(dataset_test,num_contexts=num_test_users, shuffle_rows = False, shuffle_cols = False)
x_i = dataset_test[:,:num_features]
regret_test = np.zeros(num_test_users)
choices_test = np.zeros(num_test_users)
rewards_test = np.zeros(num_test_users)

#Play the game for the rest of the rows
for k in range(num_test_users):
	x_i_k = x_i[k,:]
	A_i_k = A[k,:,:,:]
	b_i_k = b[k,:,:]
	for a in range(num_actions):
		#A_inv = np.linalg.inv(A_i_j[a]) #32x32
		th_hat[a] = A_inv.dot(b_i_j[a]) #1x32
		#ta = x_i_j.T.dot(A_inv).dot(x_i_j)
		#a_upper_ci = alph * np.sqrt(ta) #(18000,)
		a_mean = x_i_j.T.dot(th_hat[a]) #(18000,)
		p[a] = a_mean #+ a_upper_ci #(18000,)
		#p = p + (np.random.random(len(p)) * 0.00000001) #(8,1)
	choices_test[k] = p.argmax()
	rewards_test[k] = dataset_test[k,num_features+int(choices_test[k])]

regret_test = opt_rewards_test - rewards_test
