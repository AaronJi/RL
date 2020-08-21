
import time
import collections
import torch


def dqn_play(hyperparams, env, agent, net, exp_dir):
    visualize = True
    FPS = 25

    model_name = exp_dir + '/' + 'PongNoFrameskip-v4-best.dat'

    net.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()
    while True:
        start_ts = time.time()
        if visualize:
            env.render()

        action = agent.play(state, net, epsilon=0.0, device="cpu")

        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        if visualize:
            delta = 1 / FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    env.close()