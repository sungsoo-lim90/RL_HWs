"""
MSIA 490 HW1
Policy iteration algorithm
Sungsoo Lim - 10/19/20

Problem
- One state is assumed that represents the number of remaining customers at the station
- Two actions are assumed that either a bus is dispatched or it is not at the given time and state
- Reward is assumed to be negative, as there are only costs involved
- Assume that a bus can be dispatched below its capacity
"""
import numpy as np
import matplotlib.pyplot as plt 

NO_DISPATCH = 0 #not dispatching a bus is set as 0
DISPATCH = 1 #dispatching a bus is set as 1

def policy_iteration(policy, theta, discount_factor):

	nA = 2 #number of actions - dispatch a bus or not

	K = 15 #capacity of a shuttle if dispatched
	c_f = 100 #cost of dispatching a shuttle
	c_h = 2 #cost of waiting per customer

	P = {} #transition probability and reward tuple for each action
	P = {a: [] for a in range(nA)}
		
	s = np.random.randint(1,6) #initial number of customers

	def expected_value(policy,P,V):

		"""
		Calculate the expected value function for all actions in the given policy
		Assume equal action probability
		"""
		v = 0
		for a, action_prob in enumerate(policy):
			for prob, reward in P[a]:
				v += action_prob*prob*(reward + discount_factor * V)
		return v

	#Initialize vectors for function output
	V = [0,0] #value
	pol = [0] #policy
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
		
		#prob, reward for not/dispatching
		P[NO_DISPATCH] = [(1.0, reward_no_dispatch)]
		P[DISPATCH] = [(1.0, reward_dispatch)]

		#the capacity capped at 200 customers
		if next_cust_no_dispatch >= 200: #always dispatch
			P[NO_DISPATCH] = P[DISPATCH]

		#First calculate the expected value function based on the next state
		v = expected_value(policy,P,V[i-1])

		delta = max(delta, np.abs(v - V[i]))

		#random uniformly update the state (number of customers)
		if s >= 200: #always dispatch
			s = next_cust_dispatch
			best_action = 1

		if np.random.uniform(0,1.0) <=0.50:
			s = next_cust_no_dispatch
			best_action = 0
		else:
			s = next_cust_dispatch
			best_action = 1

		i = i + 1

		V.append(v)
		pol.append(best_action)
		custm.append(s)

		if delta < theta:
			break

	return custm,pol,V

def policy_improvement(discount_factor, theta, policy_iteration_fn=policy_iteration):
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
	c_h = 2 #cost of waiting per customer
	K = 15 #capacity

	while True:
		
		custm, pol, V = policy_iteration_fn(policy,theta,discount_factor)
		
		del V[0] #delete the first unnecessary element from V

		policy_stable = True

		for i in range(len(V)-1,-1,-1): #for each state
			
			chosen_action = pol[i]
			
			reward_no_dispatch = -custm[i]*c_h #reward for not dispatching
			reward_dispatch =  -(custm[i]-K)*c_h - c_f #reward for dispatching
			
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

custm, pol, V = policy_improvement(0.95,0.0001)
plt.scatter(custm,pol)
plt.title("Policy iteration")
plt.xlabel('Number of people')
plt.ylabel('Optimal Policy')
plt.savefig("policy_single.png",dpi=300)