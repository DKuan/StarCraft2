import numpy as np

from pysc2.lib import actions as sc2_actions
from pysc2.lib import features
from pysc2.lib import actions
from pysc2.env import environment
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
  selected_s = obs[0].observation["single_select"][0]
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
  multi_select = obs[0].observation["multi_select"]
  army_count = 0
  for id, group in enumerate(control_groups):
    if (group[0] == 48):
      army_count += group[1]

  if len(multi_select) > 0:
      error = True

  if (army_count < env._obs[0].observation.player_common.army_count - 4): #need to be 5
      error = True
      if army_count != 0:
        print("the army count is {0}".format(army_count))

  return error

UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'

def select_marine(env, obs):
    player_y = []
    group_list = update_group_list(obs)

    if (check_group_list(env, obs)):
        obs, xy_per_marine = init(env, obs)
        group_list = update_group_list(obs)

    # If there is no marine in danger, select random
    if  (len(group_list) > 0):
        while(len(player_y)==0):
            group_id = np.random.choice(group_list)

            obs = env.step(actions=[
                    sc2_actions.FunctionCall(_SELECT_CONTROL_GROUP, [[
                    _CONTROL_GROUP_RECALL], [int(group_id)]])
                    ])
            player_y, player_x = np.nonzero(obs[0].observation["screen"][_SELECTED] == 1)
            #to make our same type_screen different
            screen = obs[0].observation["screen"][_UNIT_TYPE]
            for i in range(len(player_y)):
                screen[player_y[i]][player_x[i]] = 49
            #in case done
            done = obs[0].step_type == environment.StepType.LAST
            if done:
                return obs, screen, [[0], [0]]
    else:
        player_x = [1]
        player_y = [1]
        obs = env.step(actions=[
            sc2_actions.FunctionCall(_NO_OP, [])
        ])
        screen = obs[0].observation["screen"][_UNIT_TYPE]
    try:
        return obs, screen, [player_x[0], player_y[0]]
    except Exception as e:
        a = 1

def _map_mirror(screen):
    #to make the replay mirror
    length = len(screen[0])
    mirror_screen = np.empty([length, length], int)
    for i in range(length):
        for j in range(length):
            np.put(mirror_screen, j + i * length, screen[i][length - j - 1])

    return mirror_screen

def map_mirror(screen, action):
    # to make the replay mirror
    length = len(screen[0])
    mirror_screen = np.empty([length, length], int)
    for i in range(length):
        for j in range(length):
            np.put(mirror_screen, j + i * length, screen[i][length - j - 1])
    mirror_action = action
    return mirror_screen, mirror_action

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
    # ---------
    # 0 gather  2 attack
    # 1 sread   3  run
    # ---------
  max_length = 4    #as the four_udlr
  pos_friend=np.array(unit_postion(obs, 1))
  pos_enemy = np.array(unit_postion(obs, 0))
  [player_y, player_x] = [pos_friend[:, 0],pos_friend[:,1]]
  [enemy_y, enemy_x] = [pos_enemy[:, 0],pos_enemy[:,1]]
  closest, min_dist = None, None

  #to calculate the min dist enemy
  if (len(player) == 2):
    for p in zip(enemy_x, enemy_y):
      dist = np.linalg.norm(np.array(player) - np.array(p))
      if not min_dist or dist < min_dist:
        closest, min_dist = p, dist

  # to calculate the mean enemy point
  mean_friend = np.mean([player_x,  player_y],  1)
  mean_enemy  = np.mean([enemy_x,   enemy_y],   1)

  if (min_dist == None):
    new_action = [sc2_actions.FunctionCall(_NO_OP, [])] #it means no enemy

  elif (action == 0 or action==1 or action==3):
    # gather towards our friend player
    if action == 0:
        diff_pos = np.array(mean_friend) - np.array(player)
    elif action == 1:
        diff_pos = np.array(player) - np.array(mean_friend)
    else:
        diff_pos = np.array(player) - np.array(mean_enemy)

    norm = np.linalg.norm(diff_pos)
    #calculate the unit pos
    if (norm > max_length):
        diff_pos = diff_pos / norm
        coord = np.array(player) + diff_pos * max_length
    else:
        coord = np.array(player) + diff_pos

    coord = check_coord(coord)
    new_action = [
        sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

  elif (action == 2):  #Attack
    # nearest enemy
    diff = np.array(closest) - np.array(player)
    norm = np.linalg.norm(diff)

    if (norm > max_length):
        diff = diff / norm
        coord = np.array(player) + diff * max_length
        coord = check_coord(coord)
        new_action = [
            sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
        ]
    else:
        coord = np.array(player) + diff
        coord = check_coord(coord)
        new_action = [
            sc2_actions.FunctionCall(_ATTACK_SCREEN, [[_NOT_QUEUED], coord])
        ]
  else:
      new_action = [sc2_actions.FunctionCall(_NO_OP, [])]  # it means no enemy

  return obs, new_action


def unit_postion(obs, flag):
    # by the screen's density map and pix number
    # to cal the army(enemy)'s position
    #input  1 army      0 enemy
    # output : list=[y, x]

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
    try:
        pos = [unit_y[0], unit_x[0]]
    except Exception:
        print("the unit is {} {}".format(unit_y,unit_x))
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
