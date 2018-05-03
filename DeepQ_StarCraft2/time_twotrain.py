# four _ udlr
# update date 4-27 to make the time-net
import sys
import os
import datetime

from absl import flags
from baselines import deepq
from pysc2.env import sc2_env
from pysc2.lib import actions

from defeat_zerglings import deepq_two
from defeat_zerglings import deepq_one
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

max_mean_reward = 0
best_reward_episode = 0
last_filename = ""

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")

FLAGS = flags.FLAGS
# four _ udlr
flags.DEFINE_string("algorithm", "deepq", "RL algorithm to use.")
flags.DEFINE_string("log", "tensorboard", "logging type(stdout, tensorboard)")

flags.DEFINE_boolean("dueling", True, "dueling")
flags.DEFINE_boolean("prioritized", True, "prioritized_replay")
flags.DEFINE_bool("visualize", True, "if you want to see the game")

flags.DEFINE_float("exploration_final_eps",  0.01, "your final Exploration Fraction")
flags.DEFINE_float("exploration_fraction",  0.47, "Exploration Fraction")
flags.DEFINE_float("gamma", 0.99, " the speed of exploration")
flags.DEFINE_float("lr",  0.001, "Learning rate")

flags.DEFINE_integer("minimap_size_px", 32, "minimap size that show in the down_left ")
flags.DEFINE_integer("train_freq", 100, "the freq that you train your model")
flags.DEFINE_integer("batch_size", 1500, "the number of your examples that you want to train your model")
flags.DEFINE_integer("print_freq", 15, "the freq that you print you result")
flags.DEFINE_integer("learning_starts", 150000, "Learning start time")
flags.DEFINE_integer("timesteps", 2500000, "most Steps to train")
flags.DEFINE_integer("num_actions", 4, "numbers of your action")    #3
flags.DEFINE_integer("step_mul", 5, "the time of every step spends")
flags.DEFINE_integer("episode_steps", 2000, "the steps of every episode spends")# 2000
flags.DEFINE_integer("buffer_size", 45000, "the number of actions that you want to store")
flags.DEFINE_integer("target_network_update_freq", 100, "the freq that your network update")

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
      minimap_size_px=(FLAGS.minimap_size_px, FLAGS.minimap_size_px),
      game_steps_per_episode= FLAGS.episode_steps) as env:

    model_one = deepq.models.cnn_to_mlp(
      convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1), (64, 3, 1), (64, 3, 1), (32, 3, 1)],
      hiddens=[256],
      dueling=True
    )

    model_two = deepq.models.cnn_to_mlp(
      convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1), (64, 3, 1), (64, 3, 1), (32, 3, 1)],
      hiddens=[256],
      dueling=True
    )