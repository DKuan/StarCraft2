import os, dill, tempfile, zipfile, configparser, datetime

import baselines.deepq.utils as U
import baselines.common.tf_util as TU

from baselines import logger, deepq
from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.logger import Logger, TensorBoardOutputFormat

from pysc2.lib import actions as sc2_actions
from pysc2.env import environment
from pysc2.lib import features, actions

from common import common
import numpy as np
import tensorflow as tf

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_DENSITY_UNIT = features.SCREEN_FEATURES.unit_density.index

_SELECTED_MINI = features.MINIMAP_FEATURES.selected.index
_PLAYER_RELATIVE_MINI = features.MINIMAP_FEATURES.player_relative.index

_PLAYER_FRIENDLY = 1
_PLAYER_SELECTED_GROUP = 2
_PLAYER_SELECTED_Army = 49
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_UNIT_ID = 1

_CONTROL_GROUP_SET = 1
_CONTROL_GROUP_RECALL = 0

_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

_NOT_QUEUED = 0
_SELECT_ALL = 0

UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'

class ActWrapper(object):
  def __init__(self, act):
    self._act = act
    #self._act_params = act_params

  @staticmethod
  def load(path, act_params, num_cpu=2):
    with open(path, "rb") as f:
      model_data = dill.load(f)
    act = deepq.build_act(**act_params)
    sess = TU.make_session(num_cpu=num_cpu)
    sess.__enter__()
    with tempfile.TemporaryDirectory() as td:
      arc_path = os.path.join(td, "packed.zip")
      with open(arc_path, "wb") as f:
        f.write(model_data)

      zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
      U.load_state(os.path.join(td, "model"))

    return ActWrapper(act)

  def __call__(self, *args, **kwargs):
    return self._act(*args, **kwargs)

  def save(self, path):
    """Save model to a pickle located at `path`"""
    with tempfile.TemporaryDirectory() as td:
      U.save_state(os.path.join(td, "model"))
      arc_name = os.path.join(td, "packed.zip")
      with zipfile.ZipFile(arc_name, 'w') as zipf:
        for root, dirs, files in os.walk(td):
          for fname in files:
            file_path = os.path.join(root, fname)
            if file_path != arc_name:
              zipf.write(file_path,
                         os.path.relpath(file_path, td))
      with open(arc_name, "rb") as f:
        model_data = f.read()
    with open(path, "wb") as f:
      dill.dump((model_data), f)


def load(path, act_params, num_cpu=16):
  """Load act function that was returned by learn function.
Parameters
----------
path: str
    path to the act function pickle
num_cpu: int
    number of cpus to use for executing the policy
Returns
-------
act: ActWrapper
    function that takes a batch of observations
    and returns actions.
"""
  return ActWrapper.load(path, num_cpu=num_cpu, act_params=act_params)


class deepq_two(object):

    def load_ini(self):
        """从ini文件提取出参数"""
        """Train a deepq model.
        Parameters
        -------
        q_func: (tf.Variable, int, str, bool) -> tf.Variable
            the model that takes the following inputs:
                observation_in: object
                    the output of observation placeholder
                num_actions: int
                    number of actions
                scope: str
                reuse: bool
                    should be passed to outer variable scope
            and returns a tensor of shape (batch_size, num_actions) with values of every action.
        checkpoint_freq: int
            how often to save the model. This is so that the best version is restored
            at the end of the training. If you do not wish to restore the best version at
            the end of the training set this variable to None.
        prioritized_replay_alpha: float
            alpha parameter for prioritized replay buffer
        prioritized_replay_beta0: float
            initial value of beta for prioritized replay buffer
        prioritized_replay_beta_iters: int
            number of iterations over which beta will be annealed from initial value
            to 1.0. If set to None equals to max_timesteps.
        prioritized_replay_eps: float
            epsilon to add to the TD errors when updating priorities.
        Returns
        -------
        act: ActWrapper
            Wrapper over act function. Adds ability to save it and load it.
            See header of baselines/deepq/categorical.py for details on the act function.
        """
        config = configparser.ConfigParser()
        self.flag_common = common.flag_common()
        config.read_file(open("./defeat_zerglings/deepq_par.ini"))

        # save model par
        self.PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
        self.max_mean_reward = 0
        self.best_reward_episode = 0
        self.last_filename = ""

        self.checkpoint_freq = 10000
        self.num_actions = config.getint('train_two_par', 'num_actions')
        self.max_timesteps = config.getint('train_two_par', 'max_timesteps')
        self.buffer_size = config.getint('train_two_par', 'buffer_size')
        self.train_freq = config.getint('train_two_par', 'train_freq')
        self.batch_size = config.getint('train_two_par', 'batch_size')
        self.print_freq = config.getint('train_two_par', 'print_freq')
        self.learning_starts = config.getint('train_two_par', 'learning_starts')
        self.target_network_update_freq  = config.getint('train_two_par', 'target_network_update_freq')

        self.lr = config.getfloat('train_two_par', 'lr')
        self.gamma = config.getfloat('train_two_par', 'gamma')
        self.exploration_fraction = config.getfloat('train_two_par', 'exploration_fraction')
        self.exploration_final_eps = config.getfloat('train_two_par', 'exploration_final_eps')
        self.gamma = config.getfloat('train_two_par', 'gamma')

        self.prioritized_replay = config.getboolean('train_two_par', 'prioritized')
        self.prioritized_replay_alpha = 0.6
        self.prioritized_replay_beta0 = 0.4
        self.prioritized_replay_beta_iters = None
        self.prioritized_replay_eps = 1e-6
        self.param_noise = False
        self.dueling = True
        self.param_noise_threshold = 0.05
        self.prioritized_replay_beta_iters = None
        self.Action_Choose = False
        self.reset = False
        self.episode_rewards = [0.0]
        self.old_episode = 0
        self.saved_mean_reward = 0
        self.sc_action = 0

    def __init__(self):
        self.load_ini()
        self.model_two = deepq.models.cnn_to_mlp(
            # convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1), (64, 3, 1), (64, 3, 1), (32, 3, 1)],
            convs=[(64, 8, 4), (64, 4, 2), (64, 3, 1), (64, 3, 1), (64, 3, 1), (32, 3, 1)],
            hiddens=[256],
            dueling=self.dueling
        )

        def make_obs_ph(name):
            return U.BatchInput((64, 64), name=name)

        act_params = {
            'make_obs_ph': make_obs_ph,
            'q_func': self.model_two,
            'num_actions': self.num_actions,
        }

        self.act, self.train, self.update_target, self.debug = deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=self.model_two,
            num_actions=self.num_actions,
            optimizer=tf.train.AdamOptimizer(learning_rate=self.lr),
            gamma=self.gamma,
            grad_norm_clipping=10,
            scope="deepq_2")


        self.start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
        logdir = "./tensorboard/zergling/%s/%s_%s_prio%s_duel%s_lr%s/%s_two" % (
            "deepq",
            self.max_timesteps,
            self.exploration_fraction,
            self.prioritized_replay,
            self.dueling,
            self.lr,
            self.start_time
        )

        Logger.DEFAULT \
            = Logger.CURRENT \
            = Logger(dir='',
                     output_formats=[TensorBoardOutputFormat(logdir)])

        # Create the replay buffer
        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.buffer_size, alpha=self.prioritized_replay_alpha)
            if self.prioritized_replay_beta_iters is None:
                self.prioritized_replay_beta_iters = self.max_timesteps
            self.beta_schedule = LinearSchedule(
                    self.prioritized_replay_beta_iters,
                    initial_p=self.prioritized_replay_beta0,
                    final_p=1.0)
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)
            beta_schedule = None
        # Create the schedule for exploration starting from 1.
        self.exploration = LinearSchedule(
            schedule_timesteps=int(self.exploration_fraction * self.max_timesteps),
            initial_p=1.0,
            final_p=self.exploration_final_eps)

        # create the model save
        self.act_save = ActWrapper(act=self.act)


    def action_network(self, env, obs, t):

        kwargs = {}

        if not self.param_noise:
            update_eps = self.exploration.value(t)
            update_param_noise_threshold = 0.
        else:
            update_eps = 0.
            if self.param_noise_threshold >= 0.:
                update_param_noise_threshold = self.param_noise_threshold
            else:
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(
                    1. - self.exploration.value(t) +
                    self.exploration.value(t) / float(self.num_actions))
            kwargs['reset'] = self.reset
            kwargs[
                'update_param_noise_threshold'] = update_param_noise_threshold
            kwargs['update_param_noise_scale'] = True
        # custom process for DefeatZerglingsAndBanelings
        self.Action_Choose = not (self.Action_Choose)

        if self.Action_Choose == True:
            # the first action
            obs, self.screen, group_id, self.player = common.select_marine(env, obs, self.flag_common)

        else:
            # the second action
            self.sc_action = None
            self.action = self.act(
                np.array(self.screen)[None], update_eps=update_eps, **kwargs)[0]
            obs, self.sc_action = common.marine_action(env, obs, self.player, self.action)

        return env, obs, self.Action_Choose, self.sc_action


    def learn(self, obs, t):
        new_screen = []
        done = obs[0].step_type == environment.StepType.LAST

        with tempfile.TemporaryDirectory() as td:
            model_saved = False
            model_file = os.path.join(td, "model")
            # if callback is not None:
            #     if callback(locals(), globals()):
            #         break

        # get the new screen in action 2
        player_y, player_x = np.nonzero(obs[0].observation["screen"][_SELECTED] == 1)
        new_screen = obs[0].observation["screen"][_UNIT_TYPE]
        for i in range(len(player_y)):
            new_screen[player_y[i]][player_x[i]] = _PLAYER_SELECTED_Army

        # update every step
        rew = obs[0].reward
        self.episode_rewards[-1] += rew
        reward = self.episode_rewards[-1]

        if (self.Action_Choose == False) or (done==True):  # only store the screen after the action is done or done
            self.replay_buffer.add(self.screen, self.action, rew, new_screen, float(done))
            mirror_new_screen, mirror_action = common.map_mirror(new_screen, self.action)
            mirror_screen = common._map_mirror(self.screen)
            self.replay_buffer.add(mirror_screen, mirror_action, rew, mirror_new_screen, float(done))

        if t > self.learning_starts and t % self.train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if self.prioritized_replay:
                experience = self.replay_buffer.sample(
                    self.batch_size, beta=self.beta_schedule.value(t))
                (obses_t, actions, rewards, obses_tp1, dones, weights,
                 batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(
                    self.batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
            td_errors = self.train(obses_t, actions, rewards, obses_tp1, dones,
                                   weights)
            if self.prioritized_replay:
                new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                self.replay_buffer.update_priorities(batch_idxes,
                                                     new_priorities)

        if t > self.learning_starts and t % self.target_network_update_freq == 0:
            # Update target network periodically.
            self.update_target()

        num_episodes = len(self.episode_rewards)
        if (num_episodes > 102):
            mean_100ep_reward = round(np.mean(self.episode_rewards[-101:-1]), 1)
        else:
            mean_100ep_reward = round(np.mean(self.episode_rewards), 1)


        if done and self.print_freq is not None and len(
                self.episode_rewards) % self.print_freq == 0:
            print("get the log")
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("reward", reward)
            logger.record_tabular("mean 100 episode reward",
                                  mean_100ep_reward)
            logger.record_tabular("% time spent exploring",
                                  int(100 * self.exploration.value(t)))
            logger.dump_tabular()

        if (self.checkpoint_freq is not None and t > self.learning_starts
                and num_episodes > 100 and t % self.checkpoint_freq == 0):
            if self.saved_mean_reward is None or mean_100ep_reward > self.saved_mean_reward:
                if self.print_freq is not None:
                    logger.log(
                        "Saving model due to mean reward increase: {} -> {}".
                            format(self.saved_mean_reward, mean_100ep_reward))
                U.save_state(model_file)
                model_saved = True
                self.saved_mean_reward = mean_100ep_reward

        if model_saved:
            if self.print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(
                    self.saved_mean_reward))
            U.load_state(model_file)

        #Model save callback
        if  (num_episodes >= 300
        and ( ((num_episodes - self.best_reward_episode) % 40 == 0) or (mean_100ep_reward > self.max_mean_reward))
        and (num_episodes >  self.old_episode)
        ):
            # in case always into this judge
            self.old_episode = num_episodes

            if (not os.path.exists(os.path.join(self.PROJ_DIR, 'models/deepq_two/{}'.format(self.start_time)))):
                try:
                    os.mkdir(os.path.join(self.PROJ_DIR, 'models/'))
                except Exception as e:
                    print(str(e))
                try:
                    os.mkdir(os.path.join(self.PROJ_DIR, 'models/deepq_two/{}'.format(self.start_time)))
                except Exception as e:
                    print(str(e))
            if mean_100ep_reward > self.max_mean_reward:
                if (self.last_filename != ""):
                    os.remove(self.last_filename)
                    print("delete last model file : %s" % self.last_filename)

                self.max_mean_reward = mean_100ep_reward
                self.best_reward_episode = num_episodes
                filename = os.path.join(self.PROJ_DIR, 'models/deepq_two/{}/zergling_{}.pkl'.format(self.start_time,mean_100ep_reward))
                self.act_save.save(filename)
                print("save best self.mean_100ep_reward model to %s" % filename)
                self.last_filename = filename
            else:
                filename = os.path.join(self.PROJ_DIR, 'models/deepq_two/{}/zergling_{}.pkl'.format(self.start_time,mean_100ep_reward))
                self.act_save.save(filename)