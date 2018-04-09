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
  unitlist = unit_postion(obs, 1)
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

  for i in range(unitlist.__len__() if unitlist.__len__() <11 else 10):
    unitlist = unit_postion(obs, 1)  # keep the newest position

    if i>=unitlist.__len__():       #in case the step make some unit dead
      break

    xy_per_marine[str(i)] = unitlist[i]
    obs = env.step(actions=[
              sc2_actions.FunctionCall(_SELECT_POINT, [[1], [unitlist[i][1], unitlist[i][0]]])
            ])

    obs = env.step(actions=[
          sc2_actions.FunctionCall(_SELECT_CONTROL_GROUP,
                                   [[_CONTROL_GROUP_SET], [i]])
        ])

    unitlist = unit_postion(obs, 1) # in case the unit is moving
    # if(_pos_y.size > 0) :
    print(i)
    obs = env.step(actions=[
        sc2_actions.FunctionCall(_SELECT_POINT, [[1], [unitlist[i][1], unitlist[i][0]]])
     ])
    # else:
    #   a = 1

  return obs, xy_per_marine


# def solve_tsp(player_relative, selected, group_list, group_id, dest_per_marine,
#               xy_per_marine):
#   my_dest = None
#   other_dest = None
#   closest, min_dist = None, None
#   actions = []
#   neutral_y, neutral_x = (player_relative == 1).nonzero()
#   player_y, player_x = (selected == 1).nonzero()
#
#   #for group_id in group_list:
#   if ("0" in dest_per_marine and "1" in dest_per_marine):
#     if (group_id == 0):
#       my_dest = dest_per_marine["0"]
#       other_dest = dest_per_marine["1"]
#     else:
#       my_dest = dest_per_marine["1"]
#       other_dest = dest_per_marine["0"]
#
#   if (len(player_x) > 0):
#     if (group_id == 0):
#       xy_per_marine["1"] = [int(player_x.mean()), int(player_y.mean())]
#     else:
#       xy_per_marine["0"] = [int(player_x.mean()), int(player_y.mean())]
#
#     player = xy_per_marine[str(group_id)]
#     points = [player]
#
#     for p in zip(neutral_x, neutral_y):
#
#       if (other_dest):
#         dist = np.linalg.norm(np.array(other_dest) - np.array(p))
#         if (dist < 10):
#           # print("continue since partner will take care of it ", p)
#           continue
#
#       pp = [p[0], p[1]]
#       if (pp not in points):
#         points.append(pp)
#
#       dist = np.linalg.norm(np.array(player) - np.array(p))
#       if not min_dist or dist < min_dist:
#         closest, min_dist = p, dist
#
#     solve_tsp = False
#     if (my_dest):
#       dist = np.linalg.norm(np.array(player) - np.array(my_dest))
#       if (dist < 0.5):
#         solve_tsp = True
#
#     if (my_dest is None):
#       solve_tsp = True
#
#     if (len(points) < 2):
#       solve_tsp = False
#
#     if (solve_tsp):
#       # function for printing best found solution when it is found
#       from time import clock
#       init = clock()
#
#       def report_sol(obj, s=""):
#         print("cpu:%g\tobj:%g\ttour:%s" % \
#               (clock(), obj, s))
#
#       #print("points: %s" % points)
#       n, D = mk_matrix(points, distL2)
#       # multi-start local search
#       #print("random start local search:", n)
#       niter = 50
#       tour, z = multistart_localsearch(niter, n, D)
#
#       #print("best found solution (%d iterations): z = %g" % (niter, z))
#       #print(tour)
#
#       left, right = None, None
#       for idx in tour:
#         if (tour[idx] == 0):
#           if (idx == len(tour) - 1):
#             #print("optimal next : ", tour[0])
#             right = points[tour[0]]
#             left = points[tour[idx - 1]]
#           elif (idx == 0):
#             #print("optimal next : ", tour[idx+1])
#             right = points[tour[idx + 1]]
#             left = points[tour[len(tour) - 1]]
#           else:
#             #print("optimal next : ", tour[idx+1])
#             right = points[tour[idx + 1]]
#             left = points[tour[idx - 1]]
#
#       left_d = np.linalg.norm(np.array(player) - np.array(left))
#       right_d = np.linalg.norm(np.array(player) - np.array(right))
#       if (right_d > left_d):
#         closest = left
#       else:
#         closest = right
#
#     #print("optimal next :" , closest)
#     dest_per_marine[str(group_id)] = closest
#     #print("dest_per_marine", self.dest_per_marine)
#     #dest_per_marine {'0': [56, 26], '1': [52, 6]}
#
#     if (closest):
#       if (group_id == 0):
#         actions.append({
#           "base_action": group_id,
#           "x0": closest[0],
#           "y0": closest[1]
#         })
#       else:
#         actions.append({
#           "base_action": group_id,
#           "x1": closest[0],
#           "y1": closest[1]
#         })
#
#     elif (my_dest):
#       if (group_id == 0):
#         actions.append({
#           "base_action": group_id,
#           "x0": my_dest[0],
#           "y0": my_dest[1]
#         })
#       else:
#         actions.append({
#           "base_action": group_id,
#           "x1": my_dest[0],
#           "y1": my_dest[1]
#         })
#
#   # elif(len(group_list)>0):
#   #
#   #   group_id = random.randint(0,len(group_list)-1)
#   #   actions.append({"base_action":group_id})
#
#   if (group_id == 0):
#     group_id = 1
#   else:
#     group_id = 0
#
#   if("0" not in xy_per_marine):
#     xy_per_marine["0"] = [0,0]
#   if("1" not in xy_per_marine):
#     xy_per_marine["1"] = [0,0]
#
#   return actions, group_id, dest_per_marine, xy_per_marine


def group_init_queue(player_relative):

  actions = []

  player_x, player_y = (player_relative == _PLAYER_FRIENDLY).nonzero()
  # try:
  #
  #   player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
  #   actions.append({"base_action":_SELECT_ARMY, "sub7":_SELECT_ALL})
  #
  # except Exception as e:
  #   print(e)
  # for i in range(len(player_x)):
  #   if i % 4 != 0:
  #     continue
  #
  #   xy = [player_x[i], player_y[i]]
  #   actions.append({"base_action":_SELECT_POINT, "sub6":0, "x0":xy[0], "y0":xy[1]})

  group_id = 0
  group_list = []
  unit_xy_list = []
  for i in range(len(player_x)):

    if group_id > 9:
      break

    xy = [player_x[i], player_y[i]]
    unit_xy_list.append(xy)
    # 2/select_point (6/select_point_act [4]; 0/screen [84, 84])
    # 4/select_control_group (4/control_group_act [5]; 5/control_group_id [10])
    if (len(unit_xy_list) >= 1):
      for idx, xy in enumerate(unit_xy_list):
        if (idx == 0):
          actions.append({
            "base_action": _SELECT_POINT,
            "sub6": 0,
            "x0": xy[0],
            "y0": xy[1]
          })
        else:
          actions.append({
            "base_action": _SELECT_POINT,
            "sub6": 1,
            "x0": xy[0],
            "y0": xy[1]
          })

      actions.append({
        "base_action": _SELECT_CONTROL_GROUP,
        "sub4": _CONTROL_GROUP_SET,
        "sub5": group_id
      })
      unit_xy_list = []

      group_list.append(group_id)
      group_id += 1

  if (len(unit_xy_list) >= 1):
    for idx, xy in enumerate(unit_xy_list):
      if (idx == 0):
        actions.append({
          "base_action": _SELECT_POINT,
          "sub6": 0,
          "x0": xy[0],
          "y0": xy[1]
        })
      else:
        actions.append({
          "base_action": _SELECT_POINT,
          "sub6": 1,
          "x0": xy[0],
          "y0": xy[1]
        })

    actions.append({
      "base_action": _SELECT_CONTROL_GROUP,
      "sub4": _CONTROL_GROUP_SET,
      "sub5": group_id
    })

    group_list.append(group_id)
    group_id += 1

  return actions


def update_group_list2(control_group):

  group_count = 0
  group_list = []

  for control_group_id, data in enumerate(control_group):

    unit_id = data[0]
    count = data[1]

    if (unit_id != 0):
      group_count += 1
      group_list.append(control_group_id)

  return group_list


def check_group_list2(extra):
  army_count = 0
  # (64, 64, 3)
  for control_group_id in range(10):
    unit_id = extra[control_group_id, 1]
    count = extra[control_group_id, 2]
    if (unit_id != 0):
      army_count += count

  if (army_count != extra[0, 0]):
    return True

  return False


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
      if (group[1] != 1):
        #print("group error group_id : %s count : %s" % (id, group[1]))
        error = True
        return error
  if (army_count != env._obs[0].observation.player_common.army_count):
    error = True
    # print("army_count %s !=  %s env._obs.observation.player_common.army_count "
    #      % (army_count, env._obs.observation.player_common.army_count))

  return error


UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'


def shift(direction, number, matrix):
  ''' shift given 2D matrix in-place the given number of rows or columns
    in the specified (UP, DOWN, LEFT, RIGHT) direction and return it
'''
  if direction in (UP):
    matrix = np.roll(matrix, -number, axis=0)
    matrix[number:, :] = -2
    return matrix
  elif direction in (DOWN):
    matrix = np.roll(matrix, number, axis=0)
    matrix[:number, :] = -2
    return matrix
  elif direction in (LEFT):
    matrix = np.roll(matrix, -number, axis=1)
    matrix[:, number:] = -2
    return matrix
  elif direction in (RIGHT):
    matrix = np.roll(matrix, number, axis=1)
    matrix[:, :number] = -2
    return matrix
  else:
    return matrix


def select_marine(env, obs):

  player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
  screen = player_relative

  group_list = update_group_list(obs)

  if (check_group_list(env, obs)):
    check_group_list(env, obs)
    obs, xy_per_marine = init(env, obs)
    group_list = update_group_list(obs)

  # if(len(group_list) == 0):
  #   obs = init(env, player_relative, obs)
  #   group_list = update_group_list(obs)

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
      # units = env._obs.observation.raw_data.units
      # marine_list = []          # for unit in units:
      #   if(unit.alliance == 1):
      #     marine_list.append(unit)

      group_id = np.random.choice(group_list)
      #xy = [int(unit.pos.y - 10), int(unit.pos.x+8)]
      #print("check xy : %s - %s" % (xy, player_relative[xy[0],xy[1]]))
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

  if (len(player) == 2):

    if (player[0] > 32):
      screen = shift(LEFT, player[0] - 32, screen)
    elif (player[0] < 32):
      screen = shift(RIGHT, 32 - player[0], screen)

    if (player[1] > 32):
      screen = shift(UP, player[1] - 32, screen)
    elif (player[1] < 32):
      screen = shift(DOWN, 32 - player[1], screen)

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

  elif (action == 4):  #UP
    coord = [player[0], player[1] - 3]
    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

  elif (action == 5):  #DOWN
    coord = [player[0], player[1] + 3]
    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

  elif (action == 6):  #LEFT
    coord = [player[0] - 3, player[1]]
    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

  elif (action == 7):  #RIGHT
    coord = [player[0] + 3, player[1]]
    new_action = [
      sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
    ]

    #print("action : %s Back Coord : %s" % (action, coord))

  return obs, new_action

def Remove(array, value):
  _array=[]
  size = array.size
  for i in range(size):
    if value == array[i]:
      _array[i:size-1] = array[i+1:size]
      return _array
    else:
      _array[i] = array[i]


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
