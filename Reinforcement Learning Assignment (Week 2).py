
# coding: utf-8

# # HomeWork Assignment (Week 2)

# In[1]:


import numpy as np
import gym
import time


# In[2]:


#@title functions to evaluate how good our policy is
'''def runPolicy(env,policy):
    state = env.reset
    done = False
    totalReward = 0
    while not done:
        state,reward,done,_ = env.step(policy[state])
        totalReward += reward
    return totalReward'''
def runPolicy(env, policy):
    state = env.reset()
    done = False
  
    totalReward = 0
    while not done:
        state, reward, done, _ = env.step(policy[state])
        totalReward += reward
    return totalReward


# In[3]:


def evaluatePolicy(env,policy,iterations):
    totalRewards = 0
    for i in range(iterations):
        totalRewards += runPolicy(env,policy)
    return totalRewards/iterations


# In[4]:


eps = 1e-10


# In[5]:


def constructGreedyPolicy(env,values,gamma):
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        returns = [sum(p*(r+gamma*values[ns]) for p,ns,r,_ in env.env.P[s][a]) for a in range(env.env.nA)]
        policy[s] = np.argmax(returns)
    return policy


# In[6]:


def computeStateValues(env,gamma,policy=None,selectBest=False):
    if policy is None and not selectBest:
        raise 'When using computeStateValues either specify policy or pass selectBest=True'
    if policy is not None and selectBest:
        raise 'You cannot use policy and selectBest at the same time'
    values = np.zeros(env.env.nS)
    while True:
        nextValues = values.copy()
        for s in range(env.env.nS):
            if policy is not None:
                action = policy[s]
                nextValues[s] = sum(p*(r+gamma*values[ns]) for p,ns,r,_ in env.env.P[s][action])
            else:
                nextValues[s] = max(sum(p*(r+gamma*values[ns]) for p,ns,r,_ in env.env.P[s][a]) for a in range(env.env.nA))
        diff = np.fabs(nextValues-values).sum()
        values = nextValues
        if diff <= eps:
            break
    return values


# # Value iteration

# In[7]:


def valueIteration(env,gamma):
    stateValues = computeStateValues(env,gamma,selectBest=True)
    policy = constructGreedyPolicy(env,stateValues,gamma)
    return policy


# # Policy iteration

# In[8]:


def randomPolicy(env):
    return np.random.choice(env.env.nA,size=(env.env.nS))


# In[9]:


def policyIteration(env,gamma):
    policy=randomPolicy(env)
    while True:
        stateValues = computeStateValues(env,gamma,policy)
        nextPolicy = constructGreedyPolicy(env,stateValues,gamma)
        if np.all(policy == nextPolicy):
            break
        policy = nextPolicy
    return policy


# # Testing our methods

# In[10]:


evaluateIterations = 1000


# In[11]:


def solveEnv(env,methods,envName):
    print(f'Solving environment {envName}')
    for method in methods:
        name,f,gamma = method
        tstart = time.time()
        policy = f(env,gamma)
        tend = time.time()
        print(f'It took {tend-tstart} seconds to compute a policy using "{name}" with gamma = {gamma}')
        score = evaluatePolicy(env,policy,evaluateIterations)
        print(f'Policy average reward is {score}')


# In[12]:


methods = [
    ('Value Iteration',valueIteration,0.9),
    ('Policy Iteration',policyIteration,0.9),
    ('Value Iteration',valueIteration,0.98),
    ('Policy Iteration',policyIteration,0.98),
    ('Value Iteration',valueIteration,1),
    ('Policy Iteration',policyIteration,1),
]


# In[13]:


frozenLake4 = gym.make('FrozenLake-v0')
solveEnv(frozenLake4,methods,'Frozen Lake 4x4')

