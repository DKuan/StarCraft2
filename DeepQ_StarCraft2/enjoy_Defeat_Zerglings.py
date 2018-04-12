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

UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_actions", 3, "numbers of your action")
flags.DEFINE_integer("step_mul", 2, "the time of every step spends")
flags.DEFINE_integer("episode_steps", 2000, "the steps of every episode spends")

def main():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(
            map_name="DefeatZerglingsAndBanelings",
            step_mul=FLAGS.step_mul,
            visualize=True,
            game_steps_per_episode=FLAGS.episode_steps * FLAGS.step_mul) as env:

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
            'num_actions': FLAGS.num_actions,
        }

        act = deep_Defeat_zerglings.load(
            "/home/cz/DKuan/StarCraft2-master/DeepQ_StarCraft2/models/deepq/zergling_112.6.pkl", act_params=act_params)

        while True:
            rew = 0
            old_num = 0
            done = False
            episode_rew = 0
            Action_Choose = False
            episode_rewards = [0.0]
            saved_mean_reward = None

            obs = env.reset()
            obs, xy_per_marine = common.init(env, obs)

            while True:

                Action_Choose = not (Action_Choose)
                if Action_Choose == True:
                    # the first action
                    obs, screen, player = common.select_marine(env, obs)
                else:
                    # the second action
                    screen = obs[0].observation["screen"][_UNIT_TYPE]
                    action = act(
                        np.array(screen)[None])[0]
                    obs, new_action = common.marine_action(env, obs, player, action)
                    army_count = env._obs[0].observation.player_common.army_count

                    try:
                        if army_count > 0 and action == 1 and (
                                _ATTACK_SCREEN in obs[0].observation["available_actions"]):
                            obs = env.step(actions=new_action)
                        elif army_count > 0 and ((action == 0) or (action == 2)) and (
                                _MOVE_SCREEN in obs[0].observation["available_actions"]):
                            obs = env.step(actions=new_action)
                        else:
                            new_action = [sc2_actions.FunctionCall(_NO_OP, [])]
                            obs = env.step(actions=new_action)
                    except Exception as e:
                        print(new_action)
                        print(e)
                        new_action = [sc2_actions.FunctionCall(_NO_OP, [])]
                        obs = env.step(actions=new_action)

                rew = obs[0].reward
                done = obs[0].step_type == environment.StepType.LAST
                episode_rewards[-1] += rew

                if done:
                    obs = env.reset()
                    group_list = common.init(env, obs)
                    episode_rewards.append(0.0)

                # test for me
                num_episodes = len(episode_rewards)
                if (num_episodes > 102):
                    mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
                else:
                    mean_100ep_reward = round(np.mean(episode_rewards), 1)

                if num_episodes > old_num:
                    old_num = num_episodes
                    print("now the episode is {}".format(num_episodes))
                    print("the mean 100ep_reward is {0}".format(mean_100ep_reward))



if __name__ == '__main__':
  main()
