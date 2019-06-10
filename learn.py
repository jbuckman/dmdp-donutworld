import tensorflow as tf
import numpy as np
import itertools

from dmdp import Autoencoder, DeepMDP

## settings
AUTOENCODER = True
ENV = "4x"

if ENV == "regular":
    import env
    OBS_X, OBS_Y = 32, 32
    TOTAL_UPDATES = 30000
else:
    import env4 as env
    OBS_X, OBS_Y = 64, 64
    TOTAL_UPDATES = 30000

## hyperparameters
LATENT_SIZE = 2
ACTION_SIZE = 16
if AUTOENCODER:
    BATCH_SIZE = 1024
else:
    BATCH_SIZE = 256
    GAMMA = .9 # reward discounting in environment
    LAMBDA = .01 # gradient penalty coefficient
    EVAL_BATCH_SIZE = 1000
EVAL_SEQ_LEN = 1000

## feed placeholders
feed_obs = tf.placeholder(tf.float32, [None, OBS_X, OBS_Y])
feed_actions = tf.placeholder(tf.int32, [None])
feed_rewards = tf.placeholder(tf.float32, [None])
feed_next_obs1 = tf.placeholder(tf.float32, [None, OBS_X, OBS_Y])
feed_next_obs2 = tf.placeholder(tf.float32, [None, OBS_X, OBS_Y])
feed_actions_sequence = tf.placeholder(tf.int32, [EVAL_SEQ_LEN, None])

## initialize model
if AUTOENCODER: model = Autoencoder([OBS_X, OBS_Y, 1], LATENT_SIZE)
else:           model = DeepMDP([OBS_X, OBS_Y, 1], ACTION_SIZE, LATENT_SIZE, GAMMA, LAMBDA)
loss = model.build_training_graph(feed_obs, feed_actions, feed_rewards, feed_next_obs1, feed_next_obs2)
if not AUTOENCODER: predicted_return = model.build_evaluate_action_sequence_graph(feed_obs, feed_actions_sequence, EVAL_SEQ_LEN)

## create optimizer
opt = tf.train.AdamOptimizer(learning_rate=3e-4)

## create train and init ops
train_op = opt.minimize(loss)
init_op = tf.global_variables_initializer()

## start session
sess = tf.Session()
sess.run(init_op)

## clear file
with open("out.csv", "w") as f: f.write("")

## begin training loop
_losses = []
_lrs = []
_lpphis = []
_lips = []
for i in range(TOTAL_UPDATES):
    _obs, _action, _reward, _next_obs1, _next_obs2 = env.sample_env_tuple(BATCH_SIZE)
    if AUTOENCODER:
        _, _loss, _ds = sess.run([train_op, loss, model.deep_state], feed_dict={feed_obs: _obs})
        _losses.append(_loss)
    else:
        _, _loss, _ds, _lr, _lpphi, _lip = sess.run([train_op, loss, model.deep_state, model.L_R, model.L_pi_phi, model.K_Vhat_bound],
                                              feed_dict={feed_obs: _obs,
                                                         feed_actions: _action,
                                                         feed_rewards: _reward,
                                                         feed_next_obs1: _next_obs1,
                                                         feed_next_obs2: _next_obs2})
        _losses.append(_loss)
        _lrs.append(np.mean(_lr))
        _lpphis.append(np.mean(_lpphi))
        _lips.append(np.mean(_lip))
    if i % 100 == 0: ## evaluate and log
        if AUTOENCODER:
            outstr = ','.join(str(logitem) for logitem in [i, np.mean(_losses)])
            _losses = []
        else:
            ## evaluate |V - V_hat|
            _states = env.get_initial_latent_states(EVAL_BATCH_SIZE)
            _obs = env.convert_to_obs(_states)
            _actions_sequence = np.stack([env.sample_random_actions(EVAL_BATCH_SIZE) for _ in range(EVAL_SEQ_LEN)], axis=0)
            answer = env.get_return_from_action_sequence(_states, _actions_sequence, GAMMA)
            guess = sess.run([predicted_return], feed_dict={feed_obs: _obs,
                                                            feed_actions_sequence: _actions_sequence})
            empirical_abs_diff = np.mean(np.abs(answer - guess))
            _obs, _action, _reward, _next_obs1, _next_obs2 = env.sample_env_tuple(BATCH_SIZE)
            theoretical_abs_diff = sess.run(model.theoretical_bound, feed_dict={feed_obs: _obs,
                                                                                feed_actions: _action,
                                                                                feed_rewards: _reward,
                                                                                feed_next_obs1: _next_obs1,
                                                                                feed_next_obs2: _next_obs2})
            outstr = ','.join(str(logitem) for logitem in [i, np.mean(_losses), np.mean(_lrs), np.mean(_lpphis), np.mean(_lips), empirical_abs_diff, theoretical_abs_diff])
            _losses = []
            _lrs = []
            _lpphis = []
            _lips = []
        print outstr
        with open("out.csv", "a") as f: f.write(outstr+"\n")

## create heatmap to visualize learned representations
obs_batch, locs = env.all_valid_obs()

deep_states = sess.run(model.deep_state, feed_dict={feed_obs: obs_batch})

# recon = sess.run([model.soft_reconstruction], feed_dict={feed_obs: obs_batch})
# import matplotlib.pyplot as plt
# plt.imshow(obs_batch[0], cmap='gray', vmin=0, vmax=1)
# plt.show()
# plt.imshow(recon[0].reshape([32,32]), cmap='gray')
# plt.show()

## output data for heatmap
for source_i in ([70, 300, 325, 525] if ENV=="regular" else [i+j for i in [70, 300, 325, 525] for j in [0,602,602*2,602*3]]):
    source = deep_states[source_i]
    distances = np.sqrt(np.sum(np.square(deep_states - np.expand_dims(source, 0)), -1))
    distances /= np.max(distances)
    heat = 1. - distances
    heatmap = np.zeros([OBS_X,OBS_Y])
    for i, loc in enumerate(locs):
        heatmap[loc[0], loc[1]] = heat[i]
    heatmapstr = ""
    envstr = ""
    for i,j in itertools.product(range(OBS_X), range(OBS_Y)):
        heatmapstr += "%d,%d,%f\n" % (i,j,heatmap[i,j])
        envstr += "%d,%d,%f\n" % (i,j,obs_batch[source_i,i,j])
    with open("heatmap%d.csv" % source_i, "w") as f: f.write(heatmapstr)
    with open("env%d.csv" % source_i, "w") as f: f.write(envstr)
