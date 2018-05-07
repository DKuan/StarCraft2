import sys, os
from absl import flags

def log_all():
    FLAGS = flags.FLAGS

    flags.DEFINE_boolean("dueling", True, "dueling")
    flags.DEFINE_bool("visualize", True, "if you want to see the game")

    flags.DEFINE_string("map_name", "DefeatZerglingsAndBanelings", "the map you want to see.")
    flags.DEFINE_string("algorithm", "deepq", "the learning method you want to use.")

    flags.DEFINE_integer("minimap_size_px", 32, "minimap size that show in the down_left ")
    flags.DEFINE_integer("step_mul", 5, "the time of every step spends")
    flags.DEFINE_integer("episode_steps", 2000, "the steps of every episode spends")  # 2000
    flags.DEFINE_integer("difficulty", 2, "Bot's strength.")

    FLAGS(sys.argv)

    return FLAGS