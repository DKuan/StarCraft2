from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat
from baselines import logger
import numpy as np

# logdir = "./tensorboard"
# Logger.DEFAULT \
#       = Logger.CURRENT \
#       = Logger(dir=None,
#                output_formats=[TensorBoardOutputFormat(logdir)])

def map_mirror(screen):
    length = len(screen[0])
    mirror_screen = np.empty([length,length], int)
    for i in range(length):
      for j in range(length):
        np.put(mirror_screen, j+i*length, screen[i][length-j-1])

    return mirror_screen


def main():
    screen = np.array([[1,2],[3,4]])
    new_screen = map_mirror(screen)
    print(new_screen)


if __name__ == '__main__':
    main()
