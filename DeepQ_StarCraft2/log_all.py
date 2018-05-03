import sys, os, datetime
from absl import flags
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat

def log_all():
    FLAGS = flags.FLAGS
    logdir = "tensorboard"
    start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")

    # FLAGS(sys.argv)
    #
    # if (FLAGS.algorithm == "deepq"):
    #     logdir = "./tensorboard/zergling/%s/%s_%s_prio%s_duel%s_lr%s/%s" % (
    #         FLAGS.algorithm,
    #         FLAGS.timesteps,
    #         FLAGS.exploration_fraction,
    #         FLAGS.prioritized,
    #         FLAGS.dueling,
    #         FLAGS.lr,
    #         start_time
    #     )
    #
    # Logger.DEFAULT \
    #     = Logger.CURRENT \
    #     = Logger(dir='',
    #         output_formats=[TensorBoardOutputFormat(logdir)])
