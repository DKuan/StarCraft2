import numpy as np

from pysc2.lib import actions as sc2_actions
from pysc2.lib import features
from pysc2.lib import actions
from common import common
import random



_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_HEALTH_VALUE = features.SCREEN_FEATURES.unit_energy.index
_HEALTH_ratio = features.SCREEN_FEATURES.unit_energy_ratio.index

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


def check_coord(coord):
    if (coord[0] < 0):
        coord[0] = 0
    elif (coord[0] > 63):
        coord[0] = 63

    if (coord[1] < 0):
        coord[1] = 0
    elif (coord[1] > 63):
        coord[1] = 63
    return coord


def marine_action(env, obs, player, action):

  if (len(player) == 2):
      if (action == 0):  # UP
          coord = [player[0], player[1] - 10]
          coord = check_coord(coord)
          new_action = [
              sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
          ]

      elif (action == 1):  # DOWN
          coord = [player[0], player[1] + 10]
          coord = check_coord(coord)
          new_action = [
              sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
          ]

      elif (action == 2):  # LEFT
          coord = [player[0] - 10, player[1]]
          coord = check_coord(coord)
          new_action = [
              sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
          ]

      elif (action == 3):  # RIGHT
          coord = [player[0] + 10, player[1]]
          coord = check_coord(coord)
          new_action = [
              sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
          ]
      else:
          new_action = [sc2_actions.FunctionCall(_NO_OP, [])]
  else:
      new_action = [sc2_actions.FunctionCall(_NO_OP, [])]

  return obs, new_action


def marine_action_old(env, obs, player, action):

  player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]

  enemy_y, enemy_x = (player_relative == _PLAYER_HOSTILE).nonzero()

  closest, min_dist = None, None

  if (len(player) == 2):
    for p in zip(enemy_x, enemy_y):
      dist = np.linalg.norm(np.array(player) - np.array(p))
      if not min_dist or dist < min_dist:
        closest, min_dist = p, dist

  player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
  friendly_y, friendly_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

  closest_friend, min_dist_friend = None, None
  if (len(player) == 2):
    for p in zip(friendly_x, friendly_y):
      dist = np.linalg.norm(np.array(player) - np.array(p))
      if not min_dist_friend or dist < min_dist_friend:
        closest_friend, min_dist_friend = p, dist

  if (closest == None):

    new_action = [sc2_actions.FunctionCall(_NO_OP, [])]

  elif (action == 0 and closest_friend != None and min_dist_friend < 3):
    # Friendly marine is too close => Sparse!

    mean_friend = [int(friendly_x.mean()), int(friendly_x.mean())]

    diff = np.array(player) - np.array(closest_friend)

    norm = np.linalg.norm(diff)

    if (norm != 0):
      diff = diff / norm

    coord = np.array(player) + diff * 4

    if (coord[0] < 0):
      coord[0] = 0
    elif (coord[0] > 63):
      coord[0] = 63

    if (coord[1] < 0):
      coord[1] = 0
    elif (coord[1] > 63):
      coord[1] = 63

    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

  elif (action <= 1):  #Attack

    # nearest enemy

    coord = closest

    new_action = [
      sc2_actions.FunctionCall(_ATTACK_SCREEN, [[_NOT_QUEUED], coord])
    ]

    #print("action : %s Attack Coord : %s" % (action, coord))

  elif (action == 2):  # Oppsite direcion from enemy

    # nearest enemy opposite

    diff = np.array(player) - np.array(closest)

    norm = np.linalg.norm(diff)

    if (norm != 0):
      diff = diff / norm

    coord = np.array(player) + diff * 7

    if (coord[0] < 0):
      coord[0] = 0
    elif (coord[0] > 63):
      coord[0] = 63

    if (coord[1] < 0):
      coord[1] = 0
    elif (coord[1] > 63):
      coord[1] = 63

    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

  elif (action == 3):  #UP
    coord = [player[0], player[1] - 3]
    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

  elif (action == 4):  #DOWN
    coord = [player[0], player[1] + 3]
    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

  elif (action == 5):  #LEFT
    coord = [player[0] - 3, player[1]]
    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

  elif (action == 6):  #RIGHT
    coord = [player[0] + 3, player[1]]
    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

    #print("action : %s Back Coord : %s" % (action, coord))

  return obs, new_action



def marine_action_changed(env, obs, player, action):

  player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]

  enemy_y, enemy_x = (player_relative == _PLAYER_HOSTILE).nonzero()

  closest, min_dist = None, None

  if (len(player) == 2):
    for p in zip(enemy_x, enemy_y):
      dist = np.linalg.norm(np.array(player) - np.array(p))
      if not min_dist or dist < min_dist:
        closest, min_dist = p, dist

  player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
  friendly_y, friendly_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

  closest_friend, min_dist_friend = None, None
  if (len(player) == 2):
    for p in zip(friendly_x, friendly_y):
      dist = np.linalg.norm(np.array(player) - np.array(p))
      if not min_dist_friend or dist < min_dist_friend:
        closest_friend, min_dist_friend = p, dist

  if (closest == None):

    new_action = [sc2_actions.FunctionCall(_NO_OP, [])]

  elif (action == 0 and closest_friend != None and min_dist_friend < 3):
    # Friendly marine is too close => Sparse!

    mean_friend = [int(friendly_x.mean()), int(friendly_x.mean())]

    diff = np.array(player) - np.array(closest_friend)

    norm = np.linalg.norm(diff)

    if (norm != 0):
      diff = diff / norm
      coord = np.array(player) + diff * 5


    if (coord[0] < 0):
      coord[0] = 0
    elif (coord[0] > 63):
      coord[0] = 63

    if (coord[1] < 0):
      coord[1] = 0
    elif (coord[1] > 63):
      coord[1] = 63

    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

  elif (action <= 1):  #Attack

    # nearest enemy

    diff = np.array(closest) - np.array(player)

    norm = np.linalg.norm(diff)

    if (norm > 5):
      diff = diff / norm
      coord = np.array(player) + diff * 5
    else:
      coord = np.array(player) + diff
    new_action = [
      sc2_actions.FunctionCall(_ATTACK_SCREEN, [[_NOT_QUEUED], coord])
    ]

    #print("action : %s Attack Coord : %s" % (action, coord))

  elif (action == 2):  # Oppsite direcion from enemy

    # nearest enemy opposite

    diff = np.array(player) - np.array(closest)

    norm = np.linalg.norm(diff)

    if (norm != 0):
      diff = diff / norm

    coord = np.array(player) + diff * 5

    if (coord[0] < 0):
      coord[0] = 0
    elif (coord[0] > 63):
      coord[0] = 63

    if (coord[1] < 0):
      coord[1] = 0
    elif (coord[1] > 63):
      coord[1] = 63

    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

  elif (action == 3):  #UP
    coord = [player[0], player[1] - 3]
    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

  elif (action == 4):  #DOWN
    coord = [player[0], player[1] + 3]
    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

  elif (action == 5):  #LEFT
    coord = [player[0] - 3, player[1]]
    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

  elif (action == 6):  #RIGHT
    coord = [player[0] + 3, player[1]]
    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

    #print("action : %s Back Coord : %s" % (action, coord))

  return obs, new_action


def Model_Cal(env, obs, player, action):
    player_list = common.unit_postion(obs,1)
    enemy_list = common.unit_postion(obs, 2)
    health_army = []
    injured_army = []
    zergling_enemy = []
    banelings_enemy = []
    health_map =  obs[0].observation["screen"][_HEALTH_ratio]

    for i in range(player_list.__len__()):
        health = health_map[player_list[i][1], player_list[i][0]]
        if health>50:
            health_army.append(player_list[i])
        else:
            a=1
    for i in range(enemy_list.__len__()):
        health_enemy = health_map[enemy_list[i][1], enemy_list[i][0]]

    closest, min_dist = None, None

    if (len(player) == 2):
        for p in zip(enemy_list):
            dist = np.linalg.norm(np.array(player) - np.array(p))
            if not min_dist or dist < min_dist:
                closest, min_dist = p, dist

    player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
    friendly_y, friendly_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

    closest_friend, min_dist_friend = None, None
    if (len(player) == 2):
        for p in zip(friendly_x, friendly_y):
            dist = np.linalg.norm(np.array(player) - np.array(p))
            if not min_dist_friend or dist < min_dist_friend:
                closest_friend, min_dist_friend = p, dist

    if (closest == None):

        new_action = [sc2_actions.FunctionCall(_NO_OP, [])]

    elif (action == 0 and closest_friend != None and min_dist_friend < 3):
        # Friendly marine is too close => Sparse!

        mean_friend = [int(friendly_x.mean()), int(friendly_x.mean())]

        diff = np.array(player) - np.array(closest_friend)

        norm = np.linalg.norm(diff)

        if (norm != 0):
            diff = diff / norm

        coord = np.array(player) + diff * 4

        if (coord[0] < 0):
            coord[0] = 0
        elif (coord[0] > 63):
            coord[0] = 63

        if (coord[1] < 0):
            coord[1] = 0
        elif (coord[1] > 63):
            coord[1] = 63

        new_action = [
            sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
        ]

    elif (action <= 1):  # Attack

        # nearest enemy

        coord = closest

        new_action = [
            sc2_actions.FunctionCall(_ATTACK_SCREEN, [[_NOT_QUEUED], coord])
        ]

        # print("action : %s Attack Coord : %s" % (action, coord))

    elif (action == 2):  # Oppsite direcion from enemy

        # nearest enemy opposite

        diff = np.array(player) - np.array(closest)

        norm = np.linalg.norm(diff)

        if (norm != 0):
            diff = diff / norm

        coord = np.array(player) + diff * 7

        if (coord[0] < 0):
            coord[0] = 0
        elif (coord[0] > 63):
            coord[0] = 63

        if (coord[1] < 0):
            coord[1] = 0
        elif (coord[1] > 63):
            coord[1] = 63

        new_action = [
            sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
        ]

    elif (action == 3):  # UP
        coord = [player[0], player[1] - 3]
        new_action = [
            sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
        ]

    elif (action == 4):  # DOWN
        coord = [player[0], player[1] + 3]
        new_action = [
            sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
        ]

    elif (action == 5):  # LEFT
        coord = [player[0] - 3, player[1]]
        new_action = [
            sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
        ]

    elif (action == 6):  # RIGHT
        coord = [player[0] + 3, player[1]]
        new_action = [
            sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
        ]

        # print("action : %s Back Coord : %s" % (action, coord))

    return obs, new_action





