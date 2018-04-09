import numpy as np

from pysc2.lib import actions as sc2_actions
from pysc2.lib import features
from pysc2.lib import actions

import random



_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index

_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4

_NOT_QUEUED = 0
_SELECT_ALL = 0
_SELECT_UNIT_ID = 1
_CONTROL_GROUP_SET = 1
_CONTROL_GROUP_RECALL = 0

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

def init(env, obs):
    player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
    player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

    player_relative1 = obs[0].observation["screen"][6]
    Defeat_y, Defeat_x = (player_relative == _PLAYER_HOSTILE).nonzero()

    obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [[0]])])
    obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_UNIT, [[0], [0]])])
    player_selected = obs[0].observation["screen"][_SELECTED]
    select_y, selected_x = (player_selected == 1).nonzero()
    print(selected_x)
    print(select_y)
    obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [[0]])])
    obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_UNIT, [[0], [2]])])
    player_selected = obs[0].observation["screen"][_SELECTED]
    select_y, selected_x = (player_selected == 1).nonzero()
    print(selected_x)
    print(select_y)

    return obs

