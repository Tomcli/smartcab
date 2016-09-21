import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.actions = [None, 'forward', 'left', 'right']
        self.waypoints = ['forward','left', 'right']
        self.lights = ['red','green']
        self.alpha = 0.1 #learning rate for q-learning algorithm
        self.gamma = 0.7 #discount factor for q-learning algorithm
        self.epsilon = 0.01 #exploration rate for q-learning algorithm
        # create q_values table
        self.q_values = {((u,v,w,x,y),z): 0 for u in self.lights for v in self.actions for w in self.actions for x in self.waypoints 
                        for y in range(6) for z in self.actions}

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None

    def getState(self, inputs, waypoint, deadline):
        """
        Return a state with the given information
        """
        return (inputs['light'], inputs['oncoming'], inputs['left'], waypoint, deadline//12) #not allow to make left turn

    def getPolicy(self, state):
        """
        Return the best policy with the given state
        """
        policies = []
        value = float('-inf')
        for action in self.actions:  #for each action
            q_values = self.q_values[state,action]#compute the q_value for that action
            if value < q_values: #set policies to the actions with maximum q_values
                value = q_values
                policies = [action]
            elif value == q_values: #if the q_value is the same, put them in the same list
                policies.append(action)
        policy = random.choice(policies) #in order to avoid bias, I put all the actions with the highest q_value in the same list and choose randomly 
        return policy

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # TODO: Update state
        self.state = self.getState(inputs,self.next_waypoint, deadline)
            
        # TODO: Select action according to your policy
        if random.randint(0,100) >= (self.epsilon*100):
            action = self.getPolicy(self.state)
        else: #choose a random action to explore based on epsilon(exploration rate)
            action = random.choice(self.actions)
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        nextState = self.getState(self.env.sense(self), self.planner.next_waypoint(), self.env.get_deadline(self))
        alpha_q = (self.alpha)*(reward + self.gamma * self.q_values[nextState, self.getPolicy(nextState)]) #discount nextState's q-value and weight it with reward 
        self.q_values[(self.state,action)] = (1-self.alpha)*(self.q_values[self.state,action]) + alpha_q

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
