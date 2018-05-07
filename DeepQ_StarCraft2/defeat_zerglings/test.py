import os, dill, tempfile, zipfile, configparser, datetime

import baselines.deepq.utils as U
import baselines.common.tf_util as TU

from baselines import logger, deepq
from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.logger import Logger, TensorBoardOutputFormat

from pysc2.lib import actions as sc2_actions
from pysc2.env import environment
from pysc2.lib import features, actions

from common import common
import numpy as np
import tensorflow as tf

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_DENSITY_UNIT = features.SCREEN_FEATURES.unit_density.index

_SELECTED_MINI = features.MINIMAP_FEATURES.selected.index
_PLAYER_RELATIVE_MINI = features.MINIMAP_FEATURES.player_relative.index

_PLAYER_FRIENDLY = 1
_PLAYER_SELECTED_GROUP = 2
_PLAYER_SELECTED_Army = 49
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

_NOT_QUEUED = 0
_SELECT_ALL = 0

UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'

class ActWrapper(object):
  def __init__(self, act):
    self._act = act
    #self._act_params = act_params

  @staticmethod
  def load(path, act_params, num_cpu=2):
    with open(path, "rb") as f:
      model_data = dill.load(f)
    act = deepq.build_act(**act_params)
    sess = TU.make_session(num_cpu=num_cpu)
    sess.__enter__()
    with tempfile.TemporaryDirectory() as td:
      arc_path = os.path.join(td, "packed.zip")
      with open(arc_path, "wb") as f:
        f.write(model_data)

      zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
      U.load_state(os.path.join(td, "model"))

    return ActWrapper(act)

  def __call__(self, *args, **kwargs):
    return self._act(*args, **kwargs)

  def save(self, path):
    """Save model to a pickle located at `path`"""
    with tempfile.TemporaryDirectory() as td:
      U.save_state(os.path.join(td, "model"))
      arc_name = os.path.join(td, "packed.zip")
      with zipfile.ZipFile(arc_name, 'w') as zipf:
        for root, dirs, files in os.walk(td):
          for fname in files:
            file_path = os.path.join(root, fname)
            if file_path != arc_name:
              zipf.write(file_path,
                         os.path.relpath(file_path, td))
      with open(arc_name, "rb") as f:
        model_data = f.read()
    with open(path, "wb") as f:
      dill.dump((model_data), f)


def load(path, act_params, num_cpu=16):
  """Load act function that was returned by learn function.
Parameters
----------
path: str
    path to the act function pickle
num_cpu: int
    number of cpus to use for executing the policy
Returns
-------
act: ActWrapper
    function that takes a batch of observations
    and returns actions.
"""
  return ActWrapper.load(path, num_cpu=num_cpu, act_params=act_params)

model_two = deepq.models.cnn_to_mlp(
            # convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1), (64, 3, 1), (64, 3, 1), (32, 3, 1)],
            convs=[(64, 8, 4), (64, 4, 2), (64, 3, 1), (64, 3, 1), (64, 3, 1), (32, 3, 1)],
            hiddens=[256],
            dueling=True,
        )


def make_obs_ph(name):
    return U.BatchInput((64, 64), name=name)


act_params = {
    'make_obs_ph': make_obs_ph,
    'q_func': model_two,
    'num_actions': 4,
}




