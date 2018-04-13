import sys
import datetime
from absl import flags
import baselines.deepq.utils as U
import baselines.common.tf_util as TU
import numpy as np
from common import common
from baselines import deepq
from pysc2.env import environment
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import actions as sc2_actions
import defeat_zerglings.dqfd as deep_Defeat_zerglings

from baselines import logger
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat

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

#to record the output
start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
logdir = "./tensorboard/enjoy/%s" % start_time
Logger.DEFAULT \
      = Logger.CURRENT \
      = Logger(dir=None,
               output_formats=[TensorBoardOutputFormat(logdir)])

FLAGS = flags.FLAGS
flags.DEFINE_string("map_name", "DefeatZerglingsAndBanelings", "the map you want to see.")
flags.DEFINE_string("trained_model", "/home/cz/DKuan/StarCraft2-master/DeepQ_StarCraft2/models/deepq/zergling_44.6.pkl",
                    "the model you has trained.")
flags.DEFINE_bool("visualize", True, "if you want to see the game")
flags.DEFINE_integer("num_actions", 4, "numbers of your action")
flags.DEFINE_integer("step_mul", 2, "the time of every step spends")
flags.DEFINE_integer("episode_steps", 2000, "the steps of every episode spends")


def main():
    FLAGS(sys.argv)
    with sc2_env.SC2Env(
            map_name=FLAGS.map_name,
            step_mul=FLAGS.step_mul,
            visualize=FLAGS.visualize,
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
            FLAGS.trained_model, act_params=act_params)

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
                    action = act(
                        np.array(screen)[None])[0]
                    obs, new_action = common.marine_action(env, obs, player, action)
                    army_count = env._obs[0].observation.player_common.army_count

                    try:
                        if army_count > 0 and (_MOVE_SCREEN in obs[0].observation["available_actions"]):
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
                    if old_num>2:
                        logger.record_tabular("reward now", episode_rewards[-2])
                    logger.record_tabular("the number of episode", num_episodes)
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.dump_tabular()
                    print("the number of episode", num_episodes)
                    print("mean 100 episode reward",mean_100ep_reward)


if __name__ == '__main__':
  main()
