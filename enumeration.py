"""
MSIA 490 HW1
Enumeration algorithm
Sungsoo Lim - 10/19/20

Problem
- One state is assumed that represents the number of customers at the station
- Two actions are assumed that either a bus is dispatched or it is not
- Reward is assumed to be negative, as there are only costs involved
- Assume that a bus can be dispatched below its capacity
"""

import numpy as np 
import matplotlib.pyplot as plt

NO_DISPATCH = 0 #not dispatching a bus is set as 0
DISPATCH = 1 #dispatching a bus is set as 1

def enumeration(discount_factor, T):

	nA = 2 #number of actions - dispatch a bus or not

	K = 15 #capacity of a shuttle if dispatched
	c_f = 100 #cost of dispatching a shuttle
	c_h = 2 #cost of waiting per customer

	P = {} #transition probability and reward tuple for each action
	P = {a: [] for a in range(nA)}
		
	s = np.random.randint(1,6) #initial number of customers

	def plus_one(nA,P,t,V):

		"""
		Calculate the value function for all action in a given state
		"""
		A = np.zeros(nA) #either dispatch or not dispatch
		for a in range(nA):
			for prob, reward in P[a]:
				A[a] += prob*(reward + discount_factor * V[t+1])
		return A

	#Initialize vectors for saving
	V_t = np.zeros(T+2) #V[state]
	V_t1 = np.zeros(T+2) #V[next_state]
	save_state = np.zeros(T+1) #number of customers at each time step
		
	for t in range(T, -1, -1):

		cust = np.random.randint(1,6) #uniform random increasing customers

		#next state for dispatching a shuttle
		#assume that K can be dispatched below its capacity
		if K > s:
			next_state_dispatch = cust
		else:
			next_state_dispatch = (s-K+cust)

		next_state_no_dispatch = s+cust #next state for not dispatching a shuttle

		reward_no_dispatch = -next_state_no_dispatch*c_h #reward for dispatching
		reward_dispatch =  -next_state_dispatch*c_h - c_f #reward for not dispatching
	
		#prob, next_state_number of customers, reward_dispatching
		P[NO_DISPATCH] = [(1.0, reward_no_dispatch)]
		P[DISPATCH] = [(1.0, reward_dispatch)]

		#the capacity capped at 200 customers
		if next_state_no_dispatch >= 200: #always dispatch
			P[NO_DISPATCH] = P[DISPATCH]

		A = plus_one(nA,P,t,V_t1) #calculate action-values

		if next_state_no_dispatch >= 200: #always dispatch
			best_action = 1
			best_action_vale = A[1]
		else: #take the action for maximum value
			best_action_value = np.max(A)
			best_action = np.argmax(A)

		V_t[t] = best_action_value

		if best_action == 0: #re-define for iteration
			s = next_state_no_dispatch
			save_state[t] = s
		else:
			s = next_state_dispatch
			save_state[t] = s

		V_t1 = V_t.copy()

	return save_state, V_t

#Run the algorithm 1000 times, average the results
save_state = np.zeros(1000)
save_V = np.zeros(1000)
for i in range(0,1000):
	s, V = enumeration(0.95,500)
	save_state[i] = s[-1]
	save_V[i] = V[-2]

#Initially, there are between 1 and 5 people at the station for t = 0
s_plot = [1,2,3,4,5]
V_plot = [np.mean(save_V[save_state == 1]), np.mean(save_V[save_state == 2]), np.mean(save_V[save_state == 3]), np.mean(save_V[save_state == 4]), np.mean(save_V[save_state == 5])]

plt.plot(s_plot,V_plot)
plt.title("Enumeration algorithm")
plt.xlabel('Number of people')
plt.ylabel('Optimal Value Function')
plt.savefig("enum_single.png",dpi=300)