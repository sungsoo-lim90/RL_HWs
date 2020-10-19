"""
MSIA 490 HW1
Policy iteration algorithm - multiclass
Sungsoo Lim - 10/19/20

Problem
- One state is assumed that represents the number of customers at the station at given time
- Two actions are assumed that either a bus is dispatched or it is not at given time and state
- Reward is assumed to be negative, as there are only costs involved
- Assume that a bus can be dispatched below its capacity

Multiclass
- Each class has its own initial and incoming customer numbers
- Assume that each class contributes equally to the bus
- Reward is calculated based on each c_h and c_f
- Each class capped at 100 members each - not yet incorporated into the algorithm written here
"""

import numpy as np
import matplotlib.pyplot as plt 

NO_DISPATCH = 0 #State is no dispatched bus, and dispatches a bus
DISPATCH = 1    #State is dispatched bus, and dispatches a bus

def policy_iteration_multiclass(policy, theta, discount_factor):

	nA = 2 #number of actions - dispatch a bus or not

	K = 30 #capacity of a shuttle if dispatched
	c_f = 100 #cost of dispatching a shuttle
	c_h = [1, 1.5, 2, 2.5, 3] #cost of waiting per customer - 5 classes

	P = {}
	P = {a: [] for a in range(nA)}
		
	s1 = np.random.randint(1,6) #initial number of customers for class 1
	s2 = np.random.randint(1,6) #initial number of customers for class 2
	s3 = np.random.randint(1,6) #initial number of customers for class 3
	s4 = np.random.randint(1,6) #initial number of customers for class 4
	s5 = np.random.randint(1,6) #initial number of customers for class 5

	def expected_value(policy,P,V):

		"""
		Calculate the expected value function for all actions in the given policy
		"""
		v = 0
		for a, action_prob in enumerate(policy):
			for prob, reward in P[a]:
				v += action_prob*prob*(reward + discount_factor * V)
		return v

	V = [0,0]
	pol = [0]
	custm =[[0,0,0,0,0]]

	i = 1

	while True:

		#stopping condition
		delta = 0 

		cust1 = np.random.randint(1,6) #uniform random increasing customers for class 1
		cust2 = np.random.randint(1,6) #uniform random increasing customers for class 2
		cust3 = np.random.randint(1,6) #uniform random increasing customers for class 3
		cust4 = np.random.randint(1,6) #uniform random increasing customers for class 4
		cust5 = np.random.randint(1,6) #uniform random increasing customers for class 5

		#next state for dispatching a shuttle for each class
		if K/5 > s1:
			dispatch1 = cust1
		else:
			dispatch1 = (s1 - K/5 + cust1)

		if K/5 > s2:
			dispatch2 = cust2
		else:
			dispatch2 = (s2 - K/5 + cust2)

		if K/5 > s3:
			dispatch3 = cust3
		else:
			dispatch3 = (s3 - K/5 + cust3)
			
		if K/5 > s4:
			dispatch4 = cust4
		else:
			dispatch4 = (s4 - K/5 + cust4)

		if K/5 > s5:
			dispatch5 = cust5
		else:
			dispatch5 = (s5 - K/5 + cust5)

		no_dispatch1 = s1 + cust1 #next state for not dispatching a shuttle - class 1
		no_dispatch2 = s2 + cust2 #next state for not dispatching a shuttle - class 1
		no_dispatch3 = s3 + cust3 #next state for not dispatching a shuttle - class 1
		no_dispatch4 = s4 + cust4 #next state for not dispatching a shuttle - class 1
		no_dispatch5 = s5 + cust5 #next state for not dispatching a shuttle - class 1

		#total number of customers when not dispatching a shuttle
		next_state_no_dispatch = no_dispatch1 + no_dispatch2 + no_dispatch3 + no_dispatch4 + no_dispatch5 
		next_state_dispatch = dispatch1 + dispatch2 + dispatch3 + dispatch4 + dispatch5 


		reward_no_dispatch = -(no_dispatch1*c_h[0] + no_dispatch2*c_h[1] + no_dispatch3*c_h[2] + no_dispatch4*c_h[3] + no_dispatch5*c_h[4]) #reward for not dispatching
		reward_dispatch =  -(dispatch1*c_h[0] + dispatch2*c_h[1] + dispatch3*c_h[2] + dispatch4*c_h[3] + dispatch5*c_h[4]) - c_f #reward for dispatching

		#prob, reward for not/dispatching
		P[NO_DISPATCH] = [(1.0, reward_no_dispatch)]
		P[DISPATCH] = [(1.0, reward_dispatch)]

		#the capacity capped at 500 customers
		if next_state_no_dispatch >= 500: #always dispatch
			P[NO_DISPATCH] = P[DISPATCH]

		#First calculate the expected value function based on the next state
		v = expected_value(policy,P,V[i-1])

		delta = max(delta, np.abs(v - V[i]))

		#random uniformly update the state (number of customers)
		if next_state_no_dispatch >= 500: #always dispatch
			s1 = dispatch1
			s2 = dispatch2
			s3 = dispatch3
			s4 = dispatch4
			s5 = dispatch5
			best_action = 1

		if np.random.uniform(0,1.0) <=0.50:
			s1 = no_dispatch1
			s2 = no_dispatch2
			s3 = no_dispatch3
			s4 = no_dispatch4
			s5 = no_dispatch5
			best_action = 0
		else:
			s1 = dispatch1
			s2 = dispatch2
			s3 = dispatch3
			s4 = dispatch4
			s5 = dispatch5
			best_action = 1

		i = i + 1
		V.append(v)
		pol.append(best_action)
		custm.append([s1, s2, s3, s4, s5])

		if delta < theta:
			break

	return custm,pol,V

def policy_improvement_multiclass(discount_factor, theta, policy_iteration_fn=policy_iteration_multiclass):

	"""
	Greedily improve the random policy
	"""
	def plus_one(nA,P,V):

		"""
		Calculate the value function for all action in a given state
		"""
		A = np.zeros(nA) #either dispatch or not dispatch
		for a in range(nA):
			for prob, reward in P[a]:
				A[a] += prob*(reward + discount_factor * V)
		return A

	policy = [0.5, 0.5] #random policy
	nA = 2 #two actions - dispatch or not dispatch a bus

	P = {} #transition probability and reward tuple for each action
	P = {a: [] for a in range(nA)}
	c_f = 100 #cost of dispatching a shuttle
	c_h = [1, 1.5, 2, 2.5, 3] #cost of waiting per customer - 5 classes
	K = 30 #capacity

	while True:
		
		custm, pol, V = policy_iteration_fn(policy,theta,discount_factor)
		
		del V[0]

		policy_stable = True

		for i in range(len(V)-1,-1,-1): #for each state
			
			chosen_action = pol[i]
			
			s1 = custm[i][0]
			s2 = custm[i][1]
			s3 = custm[i][2]
			s4 = custm[i][3]
			s5 = custm[i][4]
			s = s1 + s2 + s3 + s4
			reward_no_dispatch = -(s1*c_h[0] + s2*c_h[1] + s3*c_h[2] + s4*c_h[3] + s5*c_h[4]) #reward for not dispatching
			reward_dispatch =  -(s1*c_h[0] + s2*c_h[1] + s3*c_h[2] + s4*c_h[3] + s5*c_h[4]) - c_f #reward for dispatching

			#prob, reward for not/dispatching
			P[NO_DISPATCH] = [(1.0, reward_no_dispatch)]
			P[DISPATCH] = [(1.0, reward_dispatch)] 

			A = plus_one(nA,P,V[i-1])
			best_action = np.argmax(A)

			if chosen_action != best_action:
				policy_stable = False
				policy = np.eye(nA)[best_action].tolist()

			if policy_stable:
				return custm, pol, V

custm, pol, V = policy_improvement_multiclass(0.95,0.01)

flat_list = np.zeros(len(custm))
for i in range(0,len(custm)-1):
	flat_list[i] = np.sum(custm[i][:])

plt.scatter(flat_list,pol)
plt.title("Policy iteration - multiclass")
plt.xlabel('Number of people')
plt.ylabel('Optimal Policy')
plt.savefig("policy_multi.png",dpi=300)