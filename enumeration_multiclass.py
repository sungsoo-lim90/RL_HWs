
"""
MSIA 490 HW1
Enumeration algorithm - multiclass
Sungsoo Lim - 10/19/20

Problem
- One state is assumed that represents the number of customers at the station
- Two actions are assumed that either a bus is dispatched or it is not
- Reward is assumed to be negative, as there are only costs involved
- Assume that a bus can be dispatched below its capacity

Multiclass
- Each class has its own initial and incoming customer numbers
- Assume that each class contributes equally to the bus
- Reward is calculated based on each c_h and c_f
- Each class capped at 100 members each - not yet incorporated into the algorithm written here
"""

import numpy as np 

NO_DISPATCH = 0 #not dispatching a bus is set as 0
DISPATCH = 1 #dispatching a bus is set as 1

def enumeration_multiclass(discount_factor, T):

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

		cust1 = np.random.randint(1,6) #uniform random increasing customers for class 1
		cust2 = np.random.randint(1,6) #uniform random increasing customers for class 2
		cust3 = np.random.randint(1,6) #uniform random increasing customers for class 3
		cust4 = np.random.randint(1,6) #uniform random increasing customers for class 4
		cust5 = np.random.randint(1,6) #uniform random increasing customers for class 5

		#at each state of decision epoch, the total number of customers is the sum
		#for the shuttle, each class contributes equally (6 per class)

		#next state for dispatching a shuttle for each class
		#assume that K can be dispatched below its capacity
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
		no_dispatch2 = s2 + cust2 #next state for not dispatching a shuttle - class 2
		no_dispatch3 = s3 + cust3 #next state for not dispatching a shuttle - class 3
		no_dispatch4 = s4 + cust4 #next state for not dispatching a shuttle - class 4
		no_dispatch5 = s5 + cust5 #next state for not dispatching a shuttle - class 5

		#total number of customers when not dispatching a shuttle
		next_state_no_dispatch = no_dispatch1 + no_dispatch2 + no_dispatch3 + no_dispatch4 + no_dispatch5 

		reward_no_dispatch = -(no_dispatch1*c_h[0] + no_dispatch2*c_h[1] + no_dispatch3*c_h[2] + no_dispatch4*c_h[3] + no_dispatch5*c_h[4]) #reward for not dispatching
		reward_dispatch =  -(dispatch1*c_h[0] + dispatch2*c_h[1] + dispatch3*c_h[2] + dispatch4*c_h[3] + dispatch5*c_h[4]) - c_f #reward for dispatching
		
		#prob, next_state_number of customers, reward_dispatching
		P[NO_DISPATCH] = [(1.0, reward_no_dispatch)]
		P[DISPATCH] = [(1.0, reward_dispatch)]

		#the capacity capped at 500 customers
		if next_state_no_dispatch  >= 500: #always dispatch
			P[NO_DISPATCH] = P[DISPATCH]

		A = plus_one(nA,P,t,V_t1) #calculate action-values

		if next_state_no_dispatch >= 500: #always dispatch
			best_action = 1
			best_action_value = A[1] 
		else: #take the action for maximum value
			best_action_value = np.max(A) 
			best_action = np.argmax(A) 

		V_t[t] = best_action_value

		if best_action == 0: #redefine for iteration
			s1 = no_dispatch1
			s2 = no_dispatch2
			s3 = no_dispatch3
			s4 = no_dispatch4
			s5 = no_dispatch5
			save_state[t] = s1 + s2 + s3 + s4 + s5 #save the total customer number
		else:
			s1 = dispatch1
			s2 = dispatch2
			s3 = dispatch3
			s4 = dispatch4
			s5 = dispatch5
			save_state[t] = s1 + s2 + s3 + s4 + s5

		V_t1 = V_t.copy()

	return save_state, V_t

#Run the algorithm 1000 times, average the results
save_state = np.zeros(1000)
save_V = np.zeros(1000)
for i in range(0,1000):
	s, V = enumeration_multiclass(0.95,500)
	save_state[i] = s[-1]
	save_V[i] = V[-2]

#People at the station for t = 0
s_plot = np.unique(save_state)
V_plot = np.zeros(len(s_plot))
for i in range(0,len(s_plot)-1):
	V_plot[i] = np.mean(save_V[save_state == s_plot[i]])

plt.plot(s_plot,V_plot)
plt.title("Enumeration - multiclass")
plt.xlabel('Number of people')
plt.ylabel('Optimal Value Function')
plt.savefig("enum_multi.png",dpi=300)