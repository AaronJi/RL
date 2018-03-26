
""" The policy gradient algorithm
a policy-based method (instead of policy-value method)

input: 
t: step
Gt: the long-term return at step t
w_grad_a_with_s: the direction of param which most increases the possibility of repeating the current action on future visits to the current state
"""


def cal_policy_gradient(t, Gt, discount, w_grad_a_with_s):
    delta_w = discount**t*Gt*w_grad_a_with_s
    return delta_w

def cal_longterm_ret(labels, t, discount, t_end=None):
    if t_end is None:
        t_end = len(labels)

    Gt = 0.0
    discount_rt = 1.0
    for k in range(t, t_end):
        Gt += discount_rt*labels[k]
        discount_rt = discount_rt*discount
    return Gt
