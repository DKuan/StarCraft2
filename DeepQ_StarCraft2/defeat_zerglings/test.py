from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat
from baselines import logger

logdir = "./tensorboard"
Logger.DEFAULT \
      = Logger.CURRENT \
      = Logger(dir=None,
               output_formats=[TensorBoardOutputFormat(logdir)])

for i in range(10):
    logger.record_tabular("steps", i)
    logger.record_tabular("episodes", i)
    logger.record_tabular("reward", i)
    logger.record_tabular("mean 100 episode reward",
                          i)
    logger.dump_tabular()

