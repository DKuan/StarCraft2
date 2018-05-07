# four _ udlr
# update date 4-30 to make the time-net
import os, log_config
import common.common as common
from pysc2.env import sc2_env, environment
from pysc2.lib import actions
from pysc2.lib import actions as sc2_actions

import baselines.common.tf_util as TU
import tensorflow as tf
from defeat_zerglings import deepq_two_trained as deepq_two
from defeat_zerglings import deepq_one_trained as deepq_one

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_STOP_QUICK = actions.FUNCTIONS.Stop_quick.id
_NO_OP = actions.FUNCTIONS.no_op.id

_SELECT_UNIT_ALL = 2
_SELECT_ALL = [0]
_NOT_QUEUED = [0]


def main():
    Action_Choose = False
    done = False
    flag_changeNetwork = False
    sc_action = [sc2_actions.FunctionCall(_NO_OP, [])]
    FLAGS = log_config.log_all()

    with sc2_env.SC2Env(
            map_name=FLAGS.map_name,
            step_mul=FLAGS.step_mul,
            visualize=FLAGS.visualize,
            minimap_size_px=(FLAGS.minimap_size_px, FLAGS.minimap_size_px),
            difficulty=FLAGS.difficulty,
            game_steps_per_episode=FLAGS.episode_steps) as env:

        # step1 init the train net.
        obs = env.reset()

        sess = TU.make_session(num_cpu=4)
        sess.__enter__()

        # then init the deepq_two_trained
        train_two = deepq_two.deepq_two()
        train_one = deepq_one.deepq_one()

        # 1 start train the two deepq network
        max_timesteps = 2500000
        for t in range(max_timesteps):
            # step2 use the training network
            if not flag_changeNetwork:
                env, obs, Action_Choose, sc_action = train_one.action_network(env, obs, t)
            else:
                env, obs, Action_Choose, sc_action = train_two.action_network(env, obs, t)
            # the function will be done when the action is chosen
            if Action_Choose is not True:
                try:
                    army_count = env._obs[0].observation.player_common.army_count
                    if army_count > 0 and (_MOVE_SCREEN in obs[0].observation["available_actions"]):
                        obs = env.step(actions=sc_action)
                    else:
                        sc_action = [sc2_actions.FunctionCall(_NO_OP, [])]
                        obs = env.step(actions=sc_action)
                except Exception as e:
                    print(sc_action)
                    print(e)
                    sc_action = [sc2_actions.FunctionCall(_NO_OP, [])]
                    obs = env.step(actions=sc_action)

            # step3 train the network
            if not flag_changeNetwork:
                train_one.learn(obs, t)
            else:
                train_two.learn(obs, t)

            # step 4 learn and change the flag
            done = obs[0].step_type == environment.StepType.LAST
            if done:
                # no matter the action is what,we just want the screen and the reward
                train_one.Action_Choose = False  # should regroup the army
                train_one.learn(obs, t)
                print("episode is {}".format(len(train_one.episode_rewards)))
                print("mean_100ep_reward  is {}".format(train_one.mean_100ep_reward))

                # game/flags reset
                obs = env.reset()
                train_one.episode_rewards.append(0.0)  # add another record
                train_two.episode_rewards.append(0.0)  # add another record
                flag_changeNetwork = False  # should restart the game

            # change the train_one to train_two     when the army is close to the enemy
            if (train_two.flag_common.changeNetwork(obs)) and (flag_changeNetwork == False):
                train_two.Action_Choose = False
                train_two.flag_common.secondperiod = False
                flag_changeNetwork = True

            # change the train_two to train_one     when the army is far away from the enemy
            # because the flag just can be changed in check_group is true,so the flag_secondperiod is safe
            if (train_two.flag_common.secondperiod) and (flag_changeNetwork == True):
                try:
                    obs = env.step(actions=[sc2_actions.FunctionCall(_STOP_QUICK, [[0]])])
                except:
                    obs = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])
                train_one.Action_Choose = False
                flag_changeNetwork = False


if __name__ == '__main__':
    main()

# for training
# step1 init the train net.
# obs = env.reset()
# train_one = deepq_one.deepq_one()
#
# sess = TU.make_session(num_cpu=4)
# sess.__enter__()
#
# # 1 Initialize the parameters and copy them to the target network.
# TU.initialize()
# train_one.update_target()
#
# # then init the deepq_two_trained
# train_two = deepq_two.deepq_two()