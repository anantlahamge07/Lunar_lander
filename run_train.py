import os, time, numpy as np, tensorflow as tf, gymnasium as gym
from dqn.networks import QNetwork, hard_update, soft_update
from dqn.replay import ReplayBuffer
from dqn.policy import EpsilonScheduler, select_action
from dqn.train_step import dqn_update

# ======= hyperparams (you can move these to config.py later) =======
ENV_ID = "LunarLander-v3"
SEED = 7
GAMMA = 0.99
LR = 1e-3
BUFFER_SIZE = 100_000
BATCH_SIZE = 64
START_TRAIN_AFTER = 10_000       # collect this many steps before first update
TRAIN_EVERY = 4                  # do a gradient step every N env steps
TARGET_TAU = 0.005               # >0 use soft updates each step; set 0 for hard
TARGET_HARD_PERIOD = 1_000       # if tau==0, do hard copy every this many steps
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 200_000
TOTAL_STEPS = 800_000
LOG_EVERY = 1_000
CKPT_DIR = "checkpoints"
DOUBLE_DQN = True                # flip to False for vanilla DQN
# ====================================================================

def set_seed(seed, env):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

def evaluate(env, qnet, episodes=5, render=False):
    """Greedy policy evaluation (ε=0). Returns average return."""
    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            q = qnet(obs[None, :])
            act = int(tf.argmax(q, axis=1).numpy()[0])
            obs, r, term, trunc, _ = env.step(act)
            ep_r += r
            done = term or trunc
            if render:
                env.render()
        scores.append(ep_r)
    return float(np.mean(scores))

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ----- env -----
    env = gym.make(ENV_ID)
    set_seed(SEED, env)

    # ----- nets & optimizer -----
    online = QNetwork()       # build by calling once
    target = QNetwork()
    # Build weights
    _ = online(np.zeros((1,8), dtype=np.float32))
    _ = target(np.zeros((1,8), dtype=np.float32))
    hard_update(target, online)

    opt = tf.keras.optimizers.Adam(LR)

    # ----- buffer & epsilon -----
    buf = ReplayBuffer(obs_dim=8, size=BUFFER_SIZE)
    eps_sched = EpsilonScheduler(EPS_START, EPS_END, EPS_DECAY)

    # ----- training loop -----
    obs, _ = env.reset()
    episode_return, episode_len = 0.0, 0
    returns_hist = []

    t0 = time.time()
    for step in range(1, TOTAL_STEPS + 1):
        epsilon = eps_sched.at(step)
        a = select_action(online, obs.astype(np.float32), epsilon)
        obs2, r, term, trunc, _ = env.step(a)
        d = float(term or trunc)

        # store transition
        buf.add(obs.astype(np.float32), a, float(r), obs2.astype(np.float32), d)

        obs = obs2
        episode_return += r
        episode_len += 1

        # end of episode -> reset
        if d:
            returns_hist.append(episode_return)
            obs, _ = env.reset()
            episode_return, episode_len = 0.0, 0

        # train
        if step >= START_TRAIN_AFTER and step % TRAIN_EVERY == 0:
            s, a_b, r_b, s2, d_b = buf.sample(BATCH_SIZE)
            loss = dqn_update(
                online, target, opt,
                (s, a_b, r_b, s2, d_b),
                gamma=GAMMA,
                double=DOUBLE_DQN
            )

            # target update
            if TARGET_TAU > 0.0:
                soft_update(target, online, tau=TARGET_TAU)
            elif step % TARGET_HARD_PERIOD == 0:
                hard_update(target, online)

        # log
        if step % LOG_EVERY == 0:
            avg_ret = np.mean(returns_hist[-10:]) if returns_hist else 0.0
            print(f"step {step:,} | eps {epsilon:.3f} | avg_return(10) {avg_ret:7.2f} | "
                  f"buffer {len(buf):,} | time {(time.time()-t0):.1f}s")

        # optional: periodic eval (greedy)
        if step % (LOG_EVERY * 10) == 0 and step >= START_TRAIN_AFTER:
            eval_env = gym.make(ENV_ID)
            set_seed(SEED+1, eval_env)
            eval_score = evaluate(eval_env, online, episodes=5)
            eval_env.close()
            print(f"  >> eval_avg_return over 5 eps: {eval_score:.2f}")
            # save
            online.save_weights(os.path.join(CKPT_DIR, f"online_{step}.weights.h5"))

    env.close()
    # final save
    online.save_weights(os.path.join(CKPT_DIR, f"online_final.weights.h5"))
    print("Training done.")

if __name__ == "__main__":
    main()