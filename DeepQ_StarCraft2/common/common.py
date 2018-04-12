import numpy as np

from pysc2.lib import actions as sc2_actions
from pysc2.lib import features
from pysc2.lib import actions
import common.Common_T as common_T

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_DENSITY_UNIT = features.SCREEN_FEATURES.unit_density.index

_PLAYER_FRIENDLY = 1
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


def init(env, obs):
  obs = env.step(actions=[
    sc2_actions.FunctionCall(_NO_OP, [])
  ])

  xy_per_marine = {}
  army_count = env._obs[0].observation.player_common.army_count
  selected_s = obs[0].observation["single_select"][0]
  selected_m = obs[0].observation["multi_select"]
  _pos_y, _pos_x = np.nonzero(obs[0].observation["screen"][_SELECTED] == 1)

  if _SELECT_UNIT in obs[0].observation["available_actions"]:
        if (selected_s[0] == 0) and (_pos_y.size > 1):
            obs = env.step(actions=[
                sc2_actions.FunctionCall(_SELECT_UNIT, [[3], [0]])
            ])

  _pos_y, _pos_x = np.nonzero(obs[0].observation["screen"][_SELECTED] == 1)

  if _pos_y.size != 0:
        obs = env.step(actions=[
            sc2_actions.FunctionCall(_SELECT_POINT, [[1], [ _pos_x[0], _pos_y[0]]])
        ])

  unitlist = unit_postion(obs, 1)
  for i in range(unitlist.__len__() if unitlist.__len__() <11 else 10):
    xy_per_marine[str(i)] = unitlist[i]
    obs = env.step(actions=[
              sc2_actions.FunctionCall(_SELECT_POINT, [[1], [unitlist[i][1], unitlist[i][0]]])
            ])
    obs = env.step(actions=[
          sc2_actions.FunctionCall(_SELECT_CONTROL_GROUP,
                                   [[_CONTROL_GROUP_SET], [i]])
        ])
    obs = env.step(actions=[
        sc2_actions.FunctionCall(_SELECT_POINT, [[1], [unitlist[i][1], unitlist[i][0]]])
     ])

  return obs, xy_per_marine


def update_group_list(obs):
  control_groups = obs[0].observation["control_groups"]
  group_count = 0
  group_list = []
  for id, group in enumerate(control_groups):
    if (group[0] != 0):
      group_count += 1
      group_list.append(id)
  return group_list


def check_group_list(env, obs):
  error = False
  control_groups = obs[0].observation["control_groups"]
  army_count = 0
  for id, group in enumerate(control_groups):
    if (group[0] == 48):
      army_count += group[1]

  if (army_count < env._obs[0].observation.player_common.army_count - 4): #need to be 5
      error = True
      if army_count != 0:
        print("the army count is {0}".format(army_count))

  return error


UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'


def select_marine(env, obs):

  player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
  screen = player_relative

  group_list = update_group_list(obs)

  if (check_group_list(env, obs)):
    obs, xy_per_marine = init(env, obs)
    group_list = update_group_list(obs)

  player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]

  friendly_y, friendly_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

  enemy_y, enemy_x = (player_relative == _PLAYER_HOSTILE).nonzero()

  player = []

  danger_closest, danger_min_dist = None, None
  for e in zip(enemy_x, enemy_y):
    for p in zip(friendly_x, friendly_y):
      dist = np.linalg.norm(np.array(p) - np.array(e))
      if not danger_min_dist or dist < danger_min_dist:
        danger_closest, danger_min_dist = p, dist

  marine_closest, marine_min_dist = None, None
  for e in zip(friendly_x, friendly_y):
    for p in zip(friendly_x, friendly_y):
      dist = np.linalg.norm(np.array(p) - np.array(e))
      if not marine_min_dist or dist < marine_min_dist:
        if dist >= 2:
          marine_closest, marine_min_dist = p, dist

  if (danger_min_dist != None and danger_min_dist <= 5):
    obs = env.step(actions=[
      sc2_actions.FunctionCall(_SELECT_POINT, [[0], danger_closest])
    ])

    selected = obs[0].observation["screen"][_SELECTED]
    player_y, player_x = (selected == _PLAYER_FRIENDLY).nonzero()
    if (len(player_y) > 0):
      player = [int(player_x.mean()), int(player_y.mean())]

  elif (marine_closest != None and marine_min_dist <= 3):
    obs = env.step(actions=[
      sc2_actions.FunctionCall(_SELECT_POINT, [[0], marine_closest])
    ])

    selected = obs[0].observation["screen"][_SELECTED]
    player_y, player_x = (selected == _PLAYER_FRIENDLY).nonzero()
    if (len(player_y) > 0):
      player = [int(player_x.mean()), int(player_y.mean())]

  else:
    # If there is no marine in danger, select random
    while (len(group_list) > 0):
      group_id = np.random.choice(group_list)

      obs = env.step(actions=[
        sc2_actions.FunctionCall(_SELECT_CONTROL_GROUP, [[
          _CONTROL_GROUP_RECALL
        ], [int(group_id)]])
      ])

      selected = obs[0].observation["screen"][_SELECTED]
      player_y, player_x = (selected == _PLAYER_FRIENDLY).nonzero()
      if (len(player_y) > 0):
        player = [int(player_x.mean()), int(player_y.mean())]
        break
      else:
        group_list.remove(group_id)

  screen = obs[0].observation["screen"][_UNIT_TYPE]

  return obs, screen, player


def marine_action(env, obs, player, action):

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

  elif (action == 4 and closest_friend != None and min_dist_friend < 3):
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

  elif (action <= 5):  #Attack

    # nearest enemy

    coord = closest

    new_action = [
      sc2_actions.FunctionCall(_ATTACK_SCREEN, [[_NOT_QUEUED], coord])
    ]

    #print("action : %s Attack Coord : %s" % (action, coord))

  elif (action == 6):  # Oppsite direcion from enemy

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

  elif (action == 0):  #UP
    coord = [player[0], player[1] - 3]
    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

  elif (action == 1):  #DOWN
    coord = [player[0], player[1] + 3]
    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

  elif (action == 2):  #LEFT
    coord = [player[0] - 3, player[1]]
    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

  elif (action == 3):  #RIGHT
    coord = [player[0] + 3, player[1]]
    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

    #print("action : %s Back Coord : %s" % (action, coord))

  return obs, new_action


def unit_postion(obs, flag):
  list_unit = []
  screen_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
  screen_density_unit = obs[0].observation["screen"][_DENSITY_UNIT]
  screen_type_unit = obs[0].observation["screen"][_UNIT_TYPE]
  enemy_y, enemy_x = (screen_relative == _PLAYER_HOSTILE).nonzero()
  player_y, player_x = (screen_relative == _PLAYER_FRIENDLY ).nonzero()

  unit_y, unit_x = (player_y,player_x) if flag == 1 else (enemy_y,enemy_x)  #which unit to get
  unit_y = unit_y.tolist()
  unit_x = unit_x.tolist()
  while(len(unit_y) > 0):
    pos = [unit_y[0], unit_x[0]]
    _record = np.array([ [pos[0], pos[1]], [pos[0], pos[1]+1],
                    [pos[0]+1, pos[1]], [pos[0]+1, pos[1]+1] ])

    #make sure there are three pix is the same type
    cnt = 0
    for j in range(4):
      pos = [_record[j][0], _record[j][1]]
      if pos in _record:
        cnt += 1
    if cnt < 4:
      break

    for j in range(len(_record)):
      if(screen_density_unit[_record[j][0], _record[j][1]] > 1):
        screen_density_unit[_record[j][0], _record[j][1]] -= 1
        np.delete(_record, j, axis=0)
      else:
        if _record[j][0] in unit_y:
          unit_y.remove(_record[j][0])
        if _record[j][1] in unit_x:
          unit_x.remove(_record[j][1])
    if _record.size > 5:
      list_unit.append(_record[0])

  return list_unit
