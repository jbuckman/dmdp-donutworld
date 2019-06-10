import numpy as np
import tensorflow as tf
import nn
import env

class Autoencoder(object):
    def __init__(self, obs_shape, latent_size):
        self.obs_shape = obs_shape
        self.latent_size = latent_size

        ## set up neural networks
        self.phi_up = nn.ConvNet("encoder", self.obs_shape, [self.latent_size], layers=[[[4,4], [2,2], 2], [[4,4], [2,2], 4], [[4,4], [2,2], 8]])
        self.phi_down = nn.FeedForwardNet("decoder", self.latent_size, [np.prod(self.obs_shape)], layers=3, hidden_size=32)

    def build_training_graph(self, state, *args):
        self.deep_state = self.phi_up(tf.reshape(tf.one_hot(tf.argmax(tf.reshape(state, [-1, np.prod(self.obs_shape)]),-1),np.prod(self.obs_shape)), [-1] + self.obs_shape), add_x_channel_dim=False)
        self.reconstruction = self.phi_down(self.deep_state)
        self.soft_reconstruction = tf.nn.softmax(self.reconstruction, axis=1)
        self.recon_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.reshape(tf.cast(tf.equal(state, 1.0), tf.float32), [-1, np.prod(self.obs_shape)]), logits=self.reconstruction)

        return tf.reduce_mean(self.recon_loss)

class DeepMDP(object):
    def __init__(self, obs_shape, action_size, latent_size, gamma, lamb):
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.latent_size = latent_size
        self.gamma = gamma
        self.lamb = lamb

        ## set up neural networks
        self.phi = nn.ConvNet("encoder", self.obs_shape, [self.latent_size], layers=[[[4,4], [2,2], 2], [[4,4], [2,2], 4], [[4,4], [2,2], 8]], final_nonlinearity=tf.nn.sigmoid)
        self.P = nn.FeedForwardNet("transition", self.action_size + self.latent_size, [self.latent_size], layers=3, hidden_size=32, final_nonlinearity=tf.nn.sigmoid)
        self.R = nn.FeedForwardNet("reward", self.action_size + self.latent_size, [], layers=3, hidden_size=32)

    def build_training_graph(self, state, actions, rewards, next_state1, next_state2):
        self.deep_state = self.phi(state)
        self.deep_state_with_actions = tf.concat([self.deep_state, tf.one_hot(actions, self.action_size)], -1)

        self.predicted_reward = self.R(self.deep_state_with_actions)
        self.predicted_next_deep_state = self.P(self.deep_state_with_actions)

        self.next_deep_state1 = self.phi(next_state1)
        self.next_deep_state2 = self.phi(next_state2)

        ## lipschitz penalties
        self.R_lipschitz_gp = self.get_lipschitz_gp(self.deep_state_with_actions, self.predicted_reward)
        self.P_lipschitz_gp = self.get_lipschitz_gp(self.deep_state_with_actions, self.predicted_next_deep_state)

        self.L_R = tf.abs(self.predicted_reward - rewards)
        self.L_pi_phi = tf.reduce_sum(2. * tf.square(self.predicted_next_deep_state - self.next_deep_state1) -
                                      tf.square(self.next_deep_state1 - self.next_deep_state2),
                                      -1)

        self.K_Vhat_bound = self.get_K_Vhat_bound(self.deep_state_with_actions, self.predicted_reward)
        self.theoretical_bound = (1 / (1 - self.gamma)) * (tf.reduce_mean(self.L_R) + self.gamma * self.K_Vhat_bound * tf.reduce_mean(self.L_pi_phi))

        self.dmdp_loss = (1. / (1. - self.gamma)) * (self.L_R + self.gamma * self.L_pi_phi)
        self.lipschitz_reg = self.R_lipschitz_gp + self.P_lipschitz_gp

        return tf.reduce_mean(self.dmdp_loss) + self.lipschitz_reg

    def build_evaluate_action_sequence_graph(self, state, actions_sequence, n):
        deep_state = self.phi(state)
        rewards = 0.
        for i in range(n):
            deep_state_with_actions = tf.concat([deep_state, tf.one_hot(actions_sequence[i], self.action_size)], -1)
            rewards += self.gamma**i * self.R(deep_state_with_actions)
            deep_state = self.P(deep_state_with_actions)
        return rewards

    def get_lipschitz_gp(self, input, output):
        gradients = tf.gradients(output, [input])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean(slopes ** 2)
        return self.lamb * gradient_penalty

    def get_K_Vhat_bound(self, deep_states_with_actions, predicted_rewards):
        # K_Vpi <= K_R / (1 - gamma)
        dswa = deep_states_with_actions
        a = tf.abs(tf.expand_dims(predicted_rewards, 1) - tf.expand_dims(predicted_rewards, 0))
        b = tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(dswa, 0) - tf.expand_dims(dswa, 1)), -1))
        b = tf.where(b > tf.contrib.distributions.percentile(b, 20), b, tf.zeros_like(b)) # smooth by ignoring the 10% smallest distances (20 includes the diagonal)
        c = tf.where(b > tf.zeros_like(b), (a / b), tf.zeros_like(b)) # on the diagonal, a=b, so we get divide-by-zero issues
        K_R = tf.reduce_max(c)
        return K_R / (1. - self.gamma)

class HDeepMDP(object):
    def __init__(self, obs_shape, action_size, latent_size, latent_action_size, gamma, lamb):
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.latent_size = latent_size
        self.latent_action_size = latent_action_size
        self.gamma = gamma
        self.lamb = lamb

        ## set up neural networks
        self.phi = nn.ConvNet("encoder", self.obs_shape, [self.latent_size], layers=[[[4,4], [2,2], 2], [[4,4], [2,2], 4], [[4,4], [2,2], 8]], final_nonlinearity=tf.nn.sigmoid)
        self.P = nn.FeedForwardNet("transition", self.latent_action_size + self.latent_size, [self.latent_size], layers=3, hidden_size=32, final_nonlinearity=tf.nn.sigmoid)
        self.R = nn.FeedForwardNet("reward", self.latent_action_size + self.latent_size, [], layers=3, hidden_size=32)
        self.a_u_state_encoder = nn.ConvNet("action_to_deep_action_state_encoder", self.obs_shape, [self.latent_size], layers=[[[4, 4], [2, 2], 2], [[4, 4], [2, 2], 4], [[4, 4], [2, 2], 8]], final_nonlinearity=tf.nn.sigmoid)
        self.a_u = nn.FeedForwardNet("action_to_deep_action", self.action_size + self.latent_size, [self.latent_action_size], layers=3, hidden_size=32, final_nonlinearity=tf.nn.tanh)
        self.a_d_state_encoder = nn.ConvNet("deep_action_to_action_state_encoder", self.obs_shape, [self.latent_size], layers=[[[4, 4], [2, 2], 2], [[4, 4], [2, 2], 4], [[4, 4], [2, 2], 8]], final_nonlinearity=tf.nn.sigmoid)
        self.a_d = nn.FeedForwardNet("deep_action_to_action", self.latent_action_size + self.latent_size, [self.action_size, 2], layers=3, hidden_size=32)
        # self.a_d = nn.FeedForwardNet("deep_action_to_action", self.latent_action_size + self.latent_size, [self.action_size], layers=3, hidden_size=32, final_nonlinearity=tf.nn.tanh)

    def build_training_graph(self, state, actions_up, deep_actions_down, rewards_up, next_state1_up, next_state2_up):
        self.deep_state = self.phi(state)

        ## up-losses
        # self.deep_actions_up = actions_up
        self.deep_actions_up = self.a_u(tf.concat([self.a_u_state_encoder(state), actions_up], -1))
        self.deep_state_with_deep_actions_up = tf.concat([self.deep_state, self.deep_actions_up], -1)

        self.predicted_reward_up = self.R(self.deep_state_with_deep_actions_up)
        self.predicted_next_deep_state_up = self.P(self.deep_state_with_deep_actions_up)

        self.next_deep_state1_up = self.phi(next_state1_up)
        self.next_deep_state2_up = self.phi(next_state2_up)

        self.L_R_up = tf.abs(self.predicted_reward_up - rewards_up)
        self.L_pi_phi_up = tf.reduce_sum(2. * tf.square(self.predicted_next_deep_state_up - self.next_deep_state1_up) -
                                         tf.square(self.next_deep_state1_up - self.next_deep_state2_up),
                                         -1)

        ## down-losses
        self.deep_state_with_deep_actions_down = tf.concat([self.deep_state, deep_actions_down], -1)

        self.predicted_reward_down = self.R(self.deep_state_with_deep_actions_down)
        self.predicted_next_deep_state_down = self.P(self.deep_state_with_deep_actions_down)

        self.actions_down_muls = self.a_d(tf.concat([self.a_d_state_encoder(state), deep_actions_down], -1))
        self.actions_down_mu = tf.tanh(self.actions_down_muls[...,0])
        # self.actions_down_sigma = tf.exp(self.actions_down_muls[...,1]-10.)
        self.actions_down_sigma = tf.ones_like(self.actions_down_muls[...,1])*.1
        self.actions_dist = tf.distributions.Normal(loc=self.actions_down_mu, scale=self.actions_down_sigma)
        self.actions_down = tf.clip_by_value(self.actions_dist.sample(), -1., 1.)
        self.actions_logp = tf.reduce_sum(self.actions_dist.log_prob(self.actions_down), axis=1)
        # self.actions_down = deep_actions_down
        self.reward_down, self.next_state_down = tf.py_func(apply_action_in_env, [state, self.actions_down], [tf.float64, tf.float64])
        self.reward_down = tf.cast(tf.reshape(self.reward_down, tf.shape(rewards_up)), tf.float32)
        self.next_state_down = tf.cast(tf.reshape(self.next_state_down, tf.shape(next_state1_up)), tf.float32)

        self.next_deep_state_down = self.phi(self.next_state_down)
        # self.next_deep_state_down = tf.stop_gradient(self.phi(self.next_state_down))

        self.L_R_down = tf.abs(self.predicted_reward_down - self.reward_down)
        self.L_pi_phi_down = tf.reduce_sum(2. * tf.square(self.predicted_next_deep_state_down - self.next_deep_state_down) - 0., -1)

        ## lipschitz penalties
        self.R_lipschitz_gp = 0.5 * self.get_lipschitz_gp(self.deep_state_with_deep_actions_up, self.predicted_reward_up) + \
                              0.5 * self.get_lipschitz_gp(self.deep_state_with_deep_actions_down, self.predicted_reward_down)
        self.P_lipschitz_gp = 0.5 * self.get_lipschitz_gp(self.deep_state_with_deep_actions_up, self.predicted_next_deep_state_up) + \
                              0.5 * self.get_lipschitz_gp(self.deep_state_with_deep_actions_down, self.predicted_next_deep_state_down)

        ## bounds
        self.K_Vhat_bound = self.get_K_Vhat_bound(self.deep_state_with_deep_actions_down, self.predicted_reward_down)
        self.theoretical_bound = (1 / (1 - self.gamma)) * (tf.reduce_mean(self.L_R_down) + self.gamma * self.K_Vhat_bound * tf.reduce_mean(self.L_pi_phi_down))

        self.dmdp_loss = (1. / (1. - self.gamma)) * (0.5*self.L_R_up + 0.5*self.L_R_down + self.gamma * (0.5*self.L_pi_phi_up + 0.5*self.L_pi_phi_down))
        self.dmdp_loss += self.actions_logp * (1. / (1. - self.gamma)) * tf.stop_gradient(self.L_R_down + self.gamma*self.L_pi_phi_down)
        # self.dmdp_loss = tf.Print(self.dmdp_loss, [self.actions_down])
        self.lipschitz_reg = self.R_lipschitz_gp + self.P_lipschitz_gp

        return tf.reduce_mean(self.dmdp_loss) + self.lipschitz_reg

    def build_evaluate_action_sequence_graph(self, state, deep_actions_sequence, n):
        deep_state = self.phi(state)
        rewards = 0.
        for i in range(n):
            deep_state_with_deep_actions = tf.concat([deep_state, deep_actions_sequence[i]], -1)
            rewards += self.gamma**i * self.R(deep_state_with_deep_actions)
            deep_state = self.P(deep_state_with_deep_actions)
        return rewards

    def get_lipschitz_gp(self, input, output):
        gradients = tf.gradients(output, [input])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean(slopes ** 2)
        return self.lamb * gradient_penalty

    def get_K_Vhat_bound(self, deep_states_with_actions, predicted_rewards):
        # K_Vpi <= K_R / (1 - gamma)
        dswa = deep_states_with_actions
        a = tf.abs(tf.expand_dims(predicted_rewards, 1) - tf.expand_dims(predicted_rewards, 0))
        b = tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(dswa, 0) - tf.expand_dims(dswa, 1)), -1))
        b = tf.where(b > tf.contrib.distributions.percentile(b, 20), b, tf.zeros_like(b)) # smooth by ignoring the 10% smallest distances (20 includes the diagonal)
        c = tf.where(b > tf.zeros_like(b), (a / b), tf.zeros_like(b)) # on the diagonal, a=b, so we get divide-by-zero issues
        K_R = tf.reduce_max(c)
        return K_R / (1. - self.gamma)

def apply_action_in_env(states, actions):
    # print 'a', states.shape
    lats = env.convert_to_latents(states)
    # print 'b', lats.shape
    new_lats = env.apply_actions(lats, actions, simplify=False)
    # print 'c', new_lats.shape
    rewards = env.get_reward(lats, actions, new_lats, simplify=False)
    # print 'd', rewards.shape
    new_states = env.convert_to_obs(lats)
    # print 'e', new_states.shape
    return [rewards, new_states]
