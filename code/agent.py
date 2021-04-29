import os
import visdom
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from rltorch.memory import MultiStepMemory, PrioritizedMemory

from model import TwinnedQNetwork, GaussianPolicy
from utils import grad_false, hard_update, soft_update, to_batch,\
    update_params, RunningMeanStats
import random
from multi_step import *

PREF = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8],[0.1,0.9]]

#PREF = [[0.9, 0.1], [0.5,0.5], [0.1,0.9]]

class Monitor(object):

    def __init__(self, spec,train=True):
        self.vis = visdom.Visdom()
        self.train = train
        self.spec = spec

        self.value_window = None
        self.text_window = None

    def update(self, eps, tot_reward, Rew_1, Rew_2):

        if self.value_window == None:
            self.value_window = self.vis.line(X=torch.Tensor([eps]).cpu(),
                                              Y=torch.Tensor([tot_reward, Rew_1, Rew_2]).unsqueeze(0).cpu(),
                                              opts=dict(xlabel='episode',
                                                        ylabel='Reward value',
                                                        title='Value Dynamics' + str(self.spec),
                                                        legend=['Total Reward', 'Rew_1', 'Rew_2']))
        else:
            self.vis.line(
                X=torch.Tensor([eps]).cpu(),
                Y=torch.Tensor([tot_reward, Rew_1, Rew_2]).unsqueeze(0).cpu(),
                win=self.value_window,
                update='append')


class SacAgent:

    def __init__(self, env, log_dir, num_steps=3000000, batch_size=256,
                 lr=0.0003, hidden_units=[256, 256], memory_size=1e6,
                 gamma=0.99, tau=0.005, entropy_tuning=True, ent_coef=0.2,
                 multi_step=1, per=False, alpha=0.6, beta=0.4,
                 beta_annealing=0.0001, grad_clip=None, updates_per_step=1,
                 start_steps=10000, log_interval=10, target_update_interval=1,
                 eval_interval=1000, cuda=True, seed=0):
        self.env = env

        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True  # It harms a performance.
        torch.backends.cudnn.benchmark = False
        
        self.monitor = []
        for i in PREF:
            moni = Monitor(spec = i,train=True )
            self.monitor.append(moni)

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")
        print(self.device)
        self.policy = GaussianPolicy(
            self.env.observation_space.shape[0]+self.env.reward_num,
            self.env.action_space.shape[0],
            hidden_units=hidden_units).to(self.device)
        self.critic = TwinnedQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            self.env.reward_num,
            hidden_units=hidden_units).to(self.device)
        self.critic_target = TwinnedQNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            self.env.reward_num,
            hidden_units=hidden_units).to(self.device).eval()

        # copy parameters of the learning network to the target network
        hard_update(self.critic_target, self.critic)
        # disable gradient calculations of the target network
        grad_false(self.critic_target)

        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr)

        if entropy_tuning:
            # Target entropy is -|A|.
            self.target_entropy = -torch.prod(torch.Tensor(
                self.env.action_space.shape).to(self.device)).item()
            # We optimize log(alpha), instead of alpha.
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
        else:
            # fixed alpha
            self.alpha = torch.tensor(ent_coef).to(self.device)

        if per:
            # replay memory with prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = PrioritizedMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step,
                alpha=alpha, beta=beta, beta_annealing=beta_annealing)
        else:

            # replay memory without prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = MOMultiStepMemory(
                memory_size, self.env.observation_space.shape, self.env.reward_num,
                self.env.action_space.shape, self.device, gamma, multi_step)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_rewards = RunningMeanStats(log_interval)
        
        self.set_num = 16 # set of ω'
        
        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.num_steps = num_steps
        self.tau = tau
        self.per = per
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.entropy_tuning = entropy_tuning
        self.grad_clip = grad_clip
        self.updates_per_step = updates_per_step
        self.log_interval = log_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval
    def get_pref(self):
        preference = np.random.rand( self.env.reward_num)
        preference = preference.astype(np.float32)
        preference /= preference.sum()
        return preference


    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return len(self.memory) > self.batch_size and\
            self.steps >= self.start_steps

    def act(self, state, preference=None):
        if preference is None:
            #rand = random.randint(0, len(PREF)-1)
            #preference = np.array(PREF[rand])
            preference = self.get_pref()
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state,preference)
        return action

    def explore(self, state, preference):
        # act with randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        preference = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy.sample(state, preference)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state, preference):
        # act without randomness
 
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        preference = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, action = self.policy.sample(state, preference)
        return action.cpu().numpy().reshape(-1)

    def calc_current_q(self, states, preference, actions, rewards, next_states, dones):

        curr_q1, curr_q2 = self.critic(states, preference, actions)
        

        return curr_q1, curr_q2

    def calc_target_q(self, states, preference, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy.sample(next_states, preference)
            next_q1, next_q2 = self.critic_target(next_states, preference, next_actions)           
            

            w_q1 = torch.einsum('ij,j->i',[next_q1, preference[0] ])
            w_q2 = torch.einsum('ij,j->i',[next_q2, preference[0] ])
            mask = torch.lt(w_q1,w_q2)
            mask = mask.repeat([1,self.env.reward_num])
            mask = torch.reshape(mask, next_q1.shape)

            minq = torch.where( mask, next_q1, next_q2)
                
           # next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies
            next_q = minq + self.alpha * next_entropies

        target_q = rewards + (1.0 - dones) * self.gamma_n * next_q

        return target_q

    def train_episode(self):
        self.episodes += 1
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = self.env.reset()

        #rand = random.randint(0, len(PREF)-1)
        #preference = np.array(PREF[rand])
        preference = self.get_pref()
        while not done:
            ## Just fixed
            action = self.act(state, preference)
            #action = self.act(state)
            next_state, reward, done = self.env.step(action)
            self.steps += 1
            episode_steps += 1
            episode_reward += reward

            # ignore done if the agent reach time horizons
            # (set done=True only when the agent fails)
            if episode_steps >= self.env._max_episode_steps:
                masked_done = False
            else:
                masked_done = done

            if self.per:
                batch = to_batch(
                    state, preference, action, reward, next_state, masked_done,
                    self.device)
     
                with torch.no_grad():
                    curr_q1, curr_q2 = self.calc_current_q(*batch)
                target_q = self.calc_target_q(*batch)
                error = torch.abs(curr_q1 - target_q).item()
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.
                self.memory.append(
                    state, preference, action, reward, next_state, masked_done, error,
                    episode_done=done)
            else:
                # We need to give true done signal with addition to masked done
                # signal to calculate multi-step rewards.

                self.memory.append(
                    state, preference, action, reward, next_state, masked_done,
                    episode_done=done)

            if self.is_update():
                for _ in range(self.updates_per_step):
                    self.learn()

            if self.steps % self.eval_interval == 0:
                for i in range(len(PREF)):
                    self.evaluate(PREF[i],self.monitor[i])
                self.save_models()

            state = next_state

        # We log running mean of training rewards.
        self.train_rewards.append(episode_reward)

        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'reward/train', self.train_rewards.get(), self.steps)

        print(f'episode: {self.episodes:<4}  '
              f'episode steps: {episode_steps:<4}  '
              f'reward:', episode_reward)

    def learn(self):
        self.learning_steps += 1
        if self.learning_steps % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        if self.per:
            # batch with indices and priority weights
            batch, indices, weights = \
                self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)
            # set priority weights to 1 when we don't use PER.
            weights = 1.

        

        rand = random.randint(0, len(PREF)-1)
        preference = self.get_pref()
        preference = torch.tensor(preference ,device = self.device)
        PREF_SET = []
        for _ in range(self.set_num):
            p = self.get_pref()
            PREF_SET.append(p)


        q1_loss, q2_loss, errors, mean_q1, mean_q2 =\
            self.calc_critic_loss(batch, weights, preference, PREF_SET)
        
        policy_loss, entropies = self.calc_policy_loss(batch, weights, preference, PREF_SET)

        update_params(
            self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip)
        update_params(
            self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip)
        update_params(
            self.policy_optim, self.policy, policy_loss, self.grad_clip)

        if self.entropy_tuning:
            entropy_loss = self.calc_entropy_loss(entropies, weights)
            update_params(self.alpha_optim, None, entropy_loss)
            self.alpha = self.log_alpha.exp()
            self.writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(), self.steps)
        if self.per:
            # update priority weights
            self.memory.update_priority(indices, errors.cpu().numpy())

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self.learning_steps)
    def calc_critic_loss(self, batch, weights, preference, PREF):
        

        states, _, actions, rewards, next_states, dones = batch

        q1_losses = []
        q2_losses = []
        errorses = []
        mean_q1s = []
        mean_q2s = []

        for i in PREF:

            D_pref = torch.tensor(i,device = self.device)
            D_pref = D_pref.repeat(self.batch_size,1)

            curr_q1, curr_q2 = self.calc_current_q(states, D_pref, actions, rewards, next_states, dones)
            
        
            target_q = self.calc_target_q(states, D_pref, actions, rewards, next_states, dones)

            curr_q1 = torch.tensordot(curr_q1, preference, dims = 1)
            curr_q2 = torch.tensordot(curr_q2, preference, dims = 1)

            target_q = torch.tensordot( target_q, preference, dims = 1)

            # TD errors for updating priority weights
            errors = torch.abs(curr_q1.detach() - target_q)
            # We log means of Q to monitor training.
            mean_q1 = curr_q1.detach().mean().item()
            mean_q2 = curr_q2.detach().mean().item()
          

            # Critic loss is mean squared TD errors with priority weights.
            q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
            q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

            q1_losses.append(q1_loss)
            q2_losses.append(q2_loss)
            errorses.append(errors)
            mean_q1s.append(mean_q1)
            mean_q2s.append(mean_q1)


        q1_loss = min(q1_losses)
        q2_loss = min(q2_losses)
        error = min(errors)
        mean_q1 = min(mean_q1s)
        mean_q2 = min(mean_q2s)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights, preference, PREF):
        states, preferences, actions, rewards, next_states, dones = batch
        
        
        losses = []
        for i in PREF:
            D_pref = torch.tensor(i,device = self.device)
            D_pref = D_pref.repeat(self.batch_size,1)
            preferences = preference.repeat(self.batch_size,1)
            # We re-sample actions to calculate expectations of Q.
            sampled_action, entropy, _ = self.policy.sample(states, preferences) ############################## w'?
            # expectations of Q with clipped double Q technique
            
            q1, q2 = self.critic(states, D_pref, sampled_action)
            
            q1 = torch.tensordot(q1, preference, dims = 1)
            q2 = torch.tensordot(q2, preference, dims = 1)

            q = torch.min(q1, q2)

            # Policy objective is maximization of (Q + alpha * entropy) with
            # priority weights.
            policy_loss = torch.mean((- q - self.alpha * entropy) * weights)
            
            losses.append(policy_loss)
            
        policy_loss = min(losses)

        return policy_loss, entropy

    def calc_entropy_loss(self, entropy, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropy).detach()
            * weights)
        return entropy_loss

    def evaluate(self, preference, monitor):
        episodes = 1
        returns = np.empty((episodes,self.env.reward_num))
        preference = np.array(preference)
        for i in range(episodes):
            state = self.env.reset()
            episode_reward = np.zeros(self.env.reward_num)
            done = False
            trace = []
            actions = []
            while not done:
                action = self.exploit(state,preference )
                
                trace.append(list(state))
                actions.append(list(action))
                next_state, reward, done = self.env.step(action)
                episode_reward += reward
                state = next_state


            returns[i] = episode_reward
            print('state', trace )
            print('action', actions )
        mean_return = np.mean(returns, axis=0)
        '''
        self.writer.add_scalar(
            'reward/test', mean_return, self.steps)
        '''
        monitor.update(self.steps/self.eval_interval, np.dot(preference,mean_return), mean_return[0], mean_return[1])
        print('-' * 60)
        print(f'preference ', preference,
              f'Num steps: {self.steps:<5}  '
              f'reward:', mean_return)
        print('-' * 60)

    def save_models(self):
        self.policy.save(os.path.join(self.model_dir, 'policy.pth'))
        self.critic.save(os.path.join(self.model_dir, 'critic.pth'))
        self.critic_target.save(
            os.path.join(self.model_dir, 'critic_target.pth'))

    def __del__(self):
        #self.writer.close()
        self.env.close()
