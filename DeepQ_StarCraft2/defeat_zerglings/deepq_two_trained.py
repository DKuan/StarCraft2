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
        """get the par from the ini file"""
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

        #trained model
        self.trained_model = "/home/kuan/project/deepq_new/models/deepq/zergling_79.3.pkl"

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
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1), (64, 3, 1), (64, 3, 1), (32, 3, 1)],  #old
            # convs=[(64, 8, 4), (64, 4, 2), (64, 3, 1), (64, 3, 1), (64, 3, 1), (32, 3, 1)],  5.4 train
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

        # self.act, self.train, self.update_target, self.debug = deepq.build_train(
        #     make_obs_ph=make_obs_ph,
        #     q_func=self.model_two,
        #     num_actions=self.num_actions,
        #     optimizer=tf.train.AdamOptimizer(learning_rate=self.lr),
        #     gamma=self.gamma,
        #     grad_norm_clipping=10,
        #     scope="deepq_2")
        self.act = load(
            self.trained_model, act_params=act_params)


        start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
        logdir = "./tensorboard/zergling/%s/%s_%s_prio%s_duel%s_lr%s/%s_two" % (
            "deepq",
            self.max_timesteps,
            self.exploration_fraction,
            self.prioritized_replay,
            self.dueling,
            self.lr,
            start_time
        )

        Logger.DEFAULT \
            = Logger.CURRENT \
            = Logger(dir='',
                     output_formats=[TensorBoardOutputFormat(logdir)])


    def action_network(self, env, obs, t):

        kwargs = {}

        # custom process for DefeatZerglingsAndBanelings
        self.Action_Choose = not (self.Action_Choose)

        if self.Action_Choose == True:
            # the first action
            obs, self.screen, group_id, self.player = common.select_marine(env, obs, self.flag_common)

        else:
            # the second action
            self.sc_action = None
            self.action = self.act(
                np.array(self.screen)[None], **kwargs)[0]
            obs, self.sc_action = common.marine_action(env, obs, self.player, self.action)

        return env, obs, self.Action_Choose, self.sc_action


    def learn(self, obs, t):
        new_screen = []
        done = obs[0].step_type == environment.StepType.LAST

        # update every step
        rew = obs[0].reward
        self.episode_rewards[-1] += rew
        reward = self.episode_rewards[-1]

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
            logger.dump_tabular()