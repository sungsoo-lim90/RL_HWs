"""
MSIA 490 HW1
Value iteration algorithm
Sungsoo Lim - 10/19/20

Problem
- One state is assumed that represents the number of customers at the station
- Two actions are assumed that either a bus is dispatched or it is not
- Reward is assumed to be negative, as there are only costs involved
- Assume that a bus can be dispatched below its capacity
"""

import numpy as np 

NO_DISPATCH = 0 #not dispatching a bus is set as 0
DISPATCH = 1 #dispatching a bus is set as 1

def value_iteration(theta, discount_factor):

	nS = 1 #A state in which a bus is dispatched or not
	nA = 2 #number of actions - dispatch a bus or not

	K = 15 #capacity of a shuttle if dispatched
	c_f = 100 #cost of dispatching a shuttle
	c_h = 2 #cost of waiting per customer

	P = {} #transition probability and reward tuple for each action
	P = {a: [] for a in range(nA)}
		
	s = np.random.randint(1,6) #initial number of customers

	def plus_one(nA,P,V):

		"""
		Calculate the value function for all action in a given state
		"""
		A = np.zeros(nA) #either dispatch or not dispatch
		for a in range(nA):
			for prob, reward in P[a]:
				A[a] += prob*(reward + discount_factor * V)
		return A

	#Initialize vectors for function output
	V = [0,0] #value
	policy = [0] #policy
	custm =[0] #total number of customers at each time point
	i = 1 #iterative index

	while True:

		#stopping condition for iteration
		delta = 0 

		cust = np.random.randint(1,6) #uniform random increasing customers

		#next state for dispatching a shuttle
		if K > s: #can dispatch a shuttle not at maximum capacity in the beginning
			next_cust_dispatch = cust
		else:
			next_cust_dispatch = (s-K+cust)

		next_cust_no_dispatch = s+cust #next state for not dispatching a shuttle

		reward_no_dispatch = -next_cust_no_dispatch*c_h #reward for not dispatching
		reward_dispatch =  -next_cust_dispatch*c_h - c_f #reward for dispatching
		
		#prob, reward for not dispatching and dispatching
		P[NO_DISPATCH] = [(1.0, reward_no_dispatch)]
		P[DISPATCH] = [(1.0, reward_dispatch)]

		#the capacity capped at 200 customers
		if next_cust_no_dispatch >= 200: #always dispatch
			P[NO_DISPATCH] = P[DISPATCH]

		A = plus_one(nA,P,V[i-1]) #calculate 

		if next_state_no_dispatch >= 200: #always dispatch
			best_action = 1
			best_action_vale = A[1]
		else: #take the action for maximum value
			best_action_value = np.max(A)
			best_action = np.argmax(A)

		delta = max(delta, np.abs(best_action_value - V[i]))

		if best_action == 0: #re-define for iteration
			s = next_cust_no_dispatch
		else:
			s = next_cust_dispatch

		i = i + 1

		V.append(best_action_value)
		policy.append(best_action)
		custm.append(s)

		if delta < theta: #below threshold
			break

	return custm, policy, V


