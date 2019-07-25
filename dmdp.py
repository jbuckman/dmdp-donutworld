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

def apply_action_in_env(states, actions):
    lats = env.convert_to_latents(states)
    new_lats = env.apply_actions(lats, actions, simplify=False)
    rewards = env.get_reward(lats, actions, new_lats, simplify=False)
    new_states = env.convert_to_obs(lats)
    return [rewards, new_states]
