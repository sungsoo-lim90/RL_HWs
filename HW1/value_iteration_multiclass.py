"""
MSIA 490 HW1
Value iteration algorithm - multiclass
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
import matplotlib.pyplot as plt 

NO_DISPATCH = 0 #not dispatching a bus is set as 0
DISPATCH = 1 #dispatching a bus is set as 1

def value_iteration_multiclass(theta, discount_factor):

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

		cust1 = np.random.randint(1,6) #uniform random increasing customers for class 1
		cust2 = np.random.randint(1,6) #uniform random increasing customers for class 2
		cust3 = np.random.randint(1,6) #uniform random increasing customers for class 3
		cust4 = np.random.randint(1,6) #uniform random increasing customers for class 4
		cust5 = np.random.randint(1,6) #uniform random increasing customers for class 5

		#at each state of decision epoch, the total number of customers is the sum
		#for the shuttle, each class contributes equally (6 per class)

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

		reward_no_dispatch = -(no_dispatch1*c_h[0] + no_dispatch2*c_h[1] + no_dispatch3*c_h[2] + no_dispatch4*c_h[3] + no_dispatch5*c_h[4]) #reward for not dispatching
		reward_dispatch =  -(dispatch1*c_h[0] + dispatch2*c_h[1] + dispatch3*c_h[2] + dispatch4*c_h[3] + dispatch5*c_h[4]) - c_f #reward for dispatching

		#prob, reward for not/dispatching
		P[NO_DISPATCH] = [(1.0, reward_no_dispatch)]
		P[DISPATCH] = [(1.0, reward_dispatch)]

		#the capacity capped at 200 customers
		if next_state_no_dispatch >= 500: #always dispatch
			P[NO_DISPATCH] = P[DISPATCH]

		A = plus_one(nA,P,V[i-1])

		if next_state_no_dispatch >= 500: #always dispatch
			best_action = 1 
			best_action_value = A[1] 
		else:
			best_action_value = np.max(A) #find max A
			best_action = np.argmax(A) 

		delta = max(delta, np.abs(best_action_value - V[i]))

		if best_action == 0:
			s1 = no_dispatch1
			s2 = no_dispatch2
			s3 = no_dispatch3
			s4 = no_dispatch4
			s5 = no_dispatch5
		else:
			s1 = dispatch1
			s2 = dispatch2
			s3 = dispatch3
			s4 = dispatch4
			s5 = dispatch5

		i = i + 1
		
		s = s1 + s2 + s3 + s4 + s5

		V.append(best_action_value)
		policy.append(best_action)
		custm.append(s)

		if delta < theta:
			break

	return custm, policy, V


#Plotting
s,pol,V = value_iteration_multiclass(0.001,0.95)
del V[0]
s_plot = np.unique(s)
V_plot = np.zeros(len(s_plot))
for j in range(0,len(s_plot)-1):
	V_plot[j]= np.mean(V[s == s_plot[j]])

plt.plot(s,V)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.title("Value iteration - multiclass")
plt.xlabel('Number of people')
plt.ylabel('Optimal Value Function')
plt.savefig("value_multi.png")
