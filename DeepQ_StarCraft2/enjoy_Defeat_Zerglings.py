import sys

from absl import flags
import baselines.deepq.utils as U
import baselines.common.tf_util as TU
import numpy as np
from baselines import deepq
from pysc2.env import environment
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import actions as sc2_actions
from pysc2.lib import features
from common import common

import defeat_zerglings.dqfd as deep_Defeat_zerglings

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_PLAYER_FRIENDLY = 1
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

_NOT_QUEUED = [0]
_SELECT_ALL = [0]

step_mul = 1
steps = 2000

FLAGS = flags.FLAGS
UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'

def main():
  FLAGS(sys.argv)
  with sc2_env.SC2Env(
      map_name="DefeatZerglingsAndBanelings",
      step_mul=step_mul,
      visualize=True,
      game_steps_per_episode=steps * step_mul) as env:

    model = deepq.models.cnn_to_mlp(
      convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
      hiddens=[256],
      dueling=True
    )

    def make_obs_ph(name):
      return U.BatchInput((64, 64), name=name)

    act_params = {
      'make_obs_ph': make_obs_ph,
      'q_func': model,
      'num_actions': 3,
    }

    act = deep_Defeat_zerglings.load(
      "/home/cz/DKuan/StarCraft2-master/DeepQ_StarCraft2/models/deepq/zergling_107.0.pkl", act_params=act_params)

    while True:
        episode_rewards = [0.0]
        saved_mean_reward = None
        episode_rew = 0
        rew = 0
        done = False

        obs = env.reset()
        obs, xy_per_marine = common.init(env, obs)

        while not done:

            obs, screen, player = common.select_marine(env, obs)

            action = act(
                np.array(screen)[None])[0]

            obs, new_action = common.marine_action(env, obs, player, action)

            army_count = env._obs[0].observation.player_common.army_count

            try:
                if army_count > 0 and _ATTACK_SCREEN in obs[0].observation["available_actions"]:
                    obs = env.step(actions=new_action)
                else:
                    new_action = [sc2_actions.FunctionCall(_NO_OP, [])]
                    obs = env.step(actions=new_action)
            except Exception as e:
                # print(e)
                1  # Do nothing

            player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
            new_screen = player_relative

            rew += obs[0].reward

            done = obs[0].step_type == environment.StepType.LAST

            episode_rewards[-1] += rew
            reward = episode_rewards[-1]

            if done:
                if (len(episode_rewards)>100) :
                    mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
                    num_episodes = len(episode_rewards)
                    print("the mean 100ep_reward is {0}".format(mean_100ep_reward))
                print("Episode Reward : %s" % episode_rewards[-1])

                obs = env.reset()
                player_relative = obs[0].observation["screen"][
                    _PLAYER_RELATIVE]

                screen = player_relative

                group_list = common.init(env, obs)

                # Select all marines first
                # env.step(actions=[sc2_actions.FunctionCall(_SELECT_UNIT, [_SELECT_ALL])])
                episode_rewards=[0.0]


if __name__ == '__main__':
  main()
