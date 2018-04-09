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
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

step_mul = 1
steps = 2000

FLAGS = flags.FLAGS


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
      'num_actions': 4,
    }

    act = deep_Defeat_zerglings.load(
      "defeat_zerglings.pkl", act_params=act_params)

    while True:

      obs = env.reset()
      episode_rew = 0

      done = False

      while not done:

        obs, screen, player = common.select_marine(env, obs)

        action = act(
            np.array(screen)[None])[0]
        coord = [player[0], player[1]]

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

        selected = obs[0].observation["screen"][_SELECTED]
        player_y, player_x = (selected == _PLAYER_FRIENDLY).nonzero()
        # test for me-----------------------------
        # a = common.enemy_postion(obs, 1)
        # print(len(a))
        screen_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
        player_y, player_x = (screen_relative == _PLAYER_FRIENDLY).nonzero()
        enemy_y, enemy_x = (screen_relative == _PLAYER_HOSTILE).nonzero()
        army1_count = obs[0].observation["player"][8]
        # print("the army num is {0} and the pix_num is {1}".format(army1_count, player_x.size))
        # print("the enemy pix_num is {0}".format(enemy_x.size))
        # test for me----------------------------
        if (len(player_y) > 0):
            player = [int(player_x.mean()), int(player_y.mean())]

        if (len(player) == 2):

            if (player[0] > 32):
                new_screen = common.shift(LEFT, player[0] - 32, new_screen)
            elif (player[0] < 32):
                new_screen = common.shift(RIGHT, 32 - player[0],
                                          new_screen)

            if (player[1] > 32):
                new_screen = common.shift(UP, player[1] - 32, new_screen)
            elif (player[1] < 32):
                new_screen = common.shift(DOWN, 32 - player[1], new_screen)

        # Store transition in the replay buffer.
        replay_buffer.add(screen, action, rew, new_screen, float(done))
        screen = new_screen

        episode_rewards[-1] += rew
        reward = episode_rewards[-1]
        # test for me-----------------------------
        # test for me----------------------------

        if done:
            print("Episode Reward : %s" % episode_rewards[-1])
            obs = env.reset()
            player_relative = obs[0].observation["screen"][
                _PLAYER_RELATIVE]

            screen = player_relative

            group_list = common.init(env, obs)

            # Select all marines first
            # env.step(actions=[sc2_actions.FunctionCall(_SELECT_UNIT, [_SELECT_ALL])])
            episode_rewards.append(0.0)

            reset = True


if __name__ == '__main__':
  main()
