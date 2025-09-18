CONFIG = dict(
  env_id="LunarLander-v2",
  seed=7,
  gamma=0.99,
  lr=1e-3,
  buffer_size=100_000,
  batch_size=64,
  start_training_after=10_000,   # steps to fill buffer before updates
  train_every=4,                 # how often to update
  target_update_tau=0.005,       # soft update (set 0 for hard)
  target_update_period=1_000,    # hard update every N steps (if tau==0)
  eps_start=1.0,
  eps_end=0.05,
  eps_decay_steps=200_000,
  total_env_steps=800_000,
  eval_episodes=10,
  log_every=1000,
  save_dir="checkpoints",
)
