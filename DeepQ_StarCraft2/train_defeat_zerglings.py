# people experience  algorithm
import sys
import os
import datetime

from absl import flags
from baselines import deepq
from pysc2.env import sc2_env
from pysc2.lib import actions

from defeat_zerglings import dqfd
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

FLAGS = flags.FLAGS
# people experience  algorithm
flags.DEFINE_string("algorithm", "deepq", "RL algorithm to use.")
flags.DEFINE_string("log", "tensorboard", "logging type(stdout, tensorboard)")

flags.DEFINE_boolean("dueling", True, "dueling")
flags.DEFINE_boolean("prioritized", True, "prioritized_replay")
flags.DEFINE_bool("visualize", False, "if you want to see the game")

flags.DEFINE_float("exploration_final_eps",  0.01, "your final Exploration Fraction")
flags.DEFINE_float("exploration_fraction",  0.47, "Exploration Fraction")
flags.DEFINE_float("gamma", 0.99, " the speed of exploration")
flags.DEFINE_float("lr",  0.001, "Learning rate")

flags.DEFINE_integer("train_freq", 1000, "the freq that you train your model")
flags.DEFINE_integer("batch_size", 1500, "the number of your examples that you want to train your model")
flags.DEFINE_integer("print_freq", 15, "the freq that you print you result")
flags.DEFINE_integer("learning_starts", 45000, "Learning start time")
flags.DEFINE_integer("timesteps", 2500000, "most Steps to train")
flags.DEFINE_integer("num_actions", 4, "numbers of your action")    #3
flags.DEFINE_integer("step_mul", 5, "the time of every step spends")
flags.DEFINE_integer("episode_steps", 2000, "the steps of every episode spends")# 2000
flags.DEFINE_integer("buffer_size", 45000, "the number of actions that you want to store")
flags.DEFINE_integer("target_network_update_freq", 1000, "the freq that your network update")

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

best_reward_episode = 0
max_mean_reward = 0
last_filename = ""
start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")

def main():
  FLAGS(sys.argv)

  logdir = "tensorboard"
  if(FLAGS.algorithm == "deepq"):
    logdir = "./tensorboard/zergling/%s/%s_%s_prio%s_duel%s_lr%s/%s" % (
      FLAGS.algorithm,
      FLAGS.timesteps,
      FLAGS.exploration_fraction,
      FLAGS.prioritized,
      FLAGS.dueling,
      FLAGS.lr,
      start_time
    )

  if(FLAGS.log == "tensorboard"):
    Logger.DEFAULT \
      = Logger.CURRENT \
      = Logger(dir='log.txt',
               output_formats=[TensorBoardOutputFormat(logdir)])

  elif(FLAGS.log == "stdout"):
    os.mkdir(logdir)
    Logger.DEFAULT \
      = Logger.CURRENT \
      = Logger(dir=None,
               output_formats=[HumanOutputFormat(logdir+"/log.txt")])

  with sc2_env.SC2Env(
      map_name="DefeatZerglingsAndBanelings",
      step_mul=FLAGS.step_mul,
      visualize=FLAGS.visualize,
      game_steps_per_episode= FLAGS.episode_steps) as env:

    model = deepq.models.cnn_to_mlp(
      convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1), (64, 3, 1), (64, 3, 1), (32, 3, 1)],
      hiddens=[256],
      dueling=True
    )

    act = dqfd.learn(
      env,
      q_func=model,
      num_actions=FLAGS.num_actions,
      lr=FLAGS.lr,
      print_freq= FLAGS.print_freq,
      max_timesteps=FLAGS.timesteps,
      buffer_size=FLAGS.buffer_size,
      exploration_fraction=FLAGS.exploration_fraction,
      exploration_final_eps=FLAGS.exploration_final_eps,
      train_freq=FLAGS.train_freq,
      learning_starts=FLAGS.learning_starts,
      target_network_update_freq=FLAGS.target_network_update_freq,
      gamma=FLAGS.gamma,
      prioritized_replay=FLAGS.prioritized,
      callback=deepq_callback
    )
    act.save("defeat_zerglings.pkl")

def deepq_callback(locals, globals):
  global max_mean_reward, last_filename, best_reward_episode
  if('done' in locals and locals['done'] == True):

    print("mean_100ep_reward : %s max_mean_reward : %s" %
          (locals['mean_100ep_reward'], max_mean_reward))

    if ('mean_100ep_reward' in locals
          and locals['num_episodes'] >= 500
          and ( ((locals['num_episodes']-best_reward_episode)%100 ==0) or (locals['mean_100ep_reward'] > max_mean_reward))
      ):
      if(not os.path.exists(os.path.join(PROJ_DIR,'models/deepq/'))):
        try:
          os.mkdir(os.path.join(PROJ_DIR,'models/'))
        except Exception as e:
          print(str(e))
        try:
          os.mkdir(os.path.join(PROJ_DIR,'models/deepq/'))
        except Exception as e:
          print(str(e))
      if locals['mean_100ep_reward'] > max_mean_reward:
        if(last_filename != ""):
          os.remove(last_filename)
          print("delete last model file : %s" % last_filename)

        max_mean_reward = locals['mean_100ep_reward']
        best_reward_episode = locals['num_episodes']
        act = dqfd.ActWrapper(locals['act'])

        filename = os.path.join(PROJ_DIR,'models/deepq/zergling_%s.pkl' % locals['mean_100ep_reward'])
        act.save(filename)
        print("save best mean_100ep_reward model to %s" % filename)
        last_filename = filename
      else:
        act = dqfd.ActWrapper(locals['act'])
        filename = os.path.join(PROJ_DIR, 'models/deepq/zergling_%s.pkl' % locals['mean_100ep_reward'])
        act.save(filename)


if __name__ == '__main__':
  main()
