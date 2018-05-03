
""" The policy gradient algorithm
a policy-based method (instead of policy-value method)

input: 
t: step
Gt: the long-term return at step t
w_grad_a_with_s: the direction of param which most increases the possibility of repeating the current action on future visits to the current state
"""
import numpy as np

def cal_policy_gradient(t, Gt, discount, w_grad_a_with_s):
    delta_w = discount**t*Gt*w_grad_a_with_s
    return delta_w

def cal_longterm_ret(rewards, t, discount, t_end=None):
    if t_end is None:
        t_end = len(rewards)

    Gt = 0.0
    discount_rt = 1.0
    for k in range(t, t_end):
        Gt += discount_rt*rewards[k]
        discount_rt = discount_rt*discount
    return Gt


def cal_longterm_ret_episode(rewards, discount):

    nT = len(rewards)
    Gt_episode = np.zeros(nT)

    for t in reversed(range(nT)):
        if t == nT-1:
            Gt_episode[t] = rewards[-1]
        else:
            Gt_episode[t] = rewards[t] + discount*Gt_episode[t+1]

    return Gt_episode



if __name__ == "__main__":

    rewards = np.array([1, 3, 0, 2, 4])
    discount = 0.5

    Gt_episode = cal_longterm_ret_episode(rewards, discount)

    print(Gt_episode)
