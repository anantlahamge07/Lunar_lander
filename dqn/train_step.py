# dqn/train_step.py
import tensorflow as tf

# Robust loss for DQN
_HUBER = tf.keras.losses.Huber()

def dqn_update(online, target, optimizer, batch, gamma: float, double: bool = True):
    """
    One DQN (vanilla or Double) gradient step.
    Args:
      online: QNetwork (trainable)
      target: QNetwork (target/slow)
      optimizer: tf.keras.optimizers.Optimizer
      batch: tuple (s, a, r, s2, d) from replay
      gamma: discount factor
      double: True -> Double DQN target, False -> vanilla DQN target
    Returns:
      loss scalar tensor
    """
    s, a, r, s2, d = batch

    # ---- ensure TF tensors & dtypes ----
    s  = tf.convert_to_tensor(s,  dtype=tf.float32)   # [B,8]
    s2 = tf.convert_to_tensor(s2, dtype=tf.float32)   # [B,8]
    a  = tf.convert_to_tensor(a,  dtype=tf.int32)     # [B]
    r  = tf.convert_to_tensor(r,  dtype=tf.float32)   # [B]
    d  = tf.convert_to_tensor(d,  dtype=tf.float32)   # [B]

    with tf.GradientTape() as tape:
        # Forward through ONLINE net; be explicit with training=True
        q_all = online(s, training=True)                       # [B,4]
        # Pick Q(s,a) for the actions actually taken
        q_sa  = tf.gather(q_all, a, batch_dims=1)              # [B]

        # --------- targets ---------
        if double:
            # Double DQN: choose with ONLINE, evaluate with TARGET
            a_star = tf.argmax(online(s2, training=False), axis=1)          # [B]
            q_next = tf.gather(target(s2, training=False), a_star, batch_dims=1)  # [B]
        else:
            # Vanilla DQN: max over TARGET
            q_next = tf.reduce_max(target(s2, training=False), axis=1)      # [B]

        # Bellman target
        y = r + gamma * (1.0 - d) * q_next                                  # [B]

        # Huber loss (mean over batch) between Q_online(s,a) and target y
        loss = tf.reduce_mean(_HUBER(y, q_sa))

    vars_ = online.trainable_variables
    # (diagnostic) ensure model has variables
    tf.debugging.assert_positive(
        tf.cast(tf.size(vars_[0]), tf.int32),
        message="Online network has no variables. Did you build it first?"
    )

    grads = tape.gradient(loss, vars_)

    # (diagnostic) if any grad is None, raise a clear error
    if any(g is None for g in grads):
        raise RuntimeError(
            "Got None gradients. Common causes:\n"
            "- online net not built before first update\n"
            "- wrong dtypes (actions must be int32)\n"
            "- loss not connected to model outputs"
        )

    optimizer.apply_gradients(zip(grads, vars_))
    return loss