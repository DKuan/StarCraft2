# update date 4-27
import numpy as np
from pysc2.lib import actions as sc2_actions
from pysc2.lib import features
from pysc2.lib import actions
from pysc2.env import environment

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_DENSITY_UNIT = features.SCREEN_FEATURES.unit_density.index

_SELECTED_MINI = features.MINIMAP_FEATURES.selected.index
_PLAYER_RELATIVE_MINI = features.MINIMAP_FEATURES.player_relative.index

_PLAYER_FRIENDLY = 1
_PLAYER_SELECTED_GROUP = 2
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_UNIT_ID = 1

_CONTROL_GROUP_SET = 1
_CONTROL_GROUP_RECALL = 0

_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_MOVE_MINIMAP = actions.FUNCTIONS.Move_minimap.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_STOP_QUICK = actions.FUNCTIONS.Stop_quick.id

_NOT_QUEUED = 0
_SELECT_ALL = 0

check_coord_mapsize = 64

def init(env, obs):
    obs = env.step(actions=[
        sc2_actions.FunctionCall(_NO_OP, [])
        ])

    xy_per_marine = {}
    selected_s = obs[0].observation["single_select"][0]
    _pos_y, _pos_x = np.nonzero(obs[0].observation["screen"][_SELECTED] == 1)

    if _SELECT_UNIT in obs[0].observation["available_actions"]:
        if (selected_s[0] == 0) and (_pos_y.size > 1):
            obs = env.step(actions=[sc2_actions.FunctionCall(_STOP_QUICK, [[0]])])  # stop all actions
            obs = env.step(actions=[
                sc2_actions.FunctionCall(_SELECT_UNIT, [[3], [0]])
            ])

    _pos_y, _pos_x = np.nonzero(obs[0].observation["screen"][_SELECTED] == 1)

    if _pos_y.size != 0:
        obs = env.step(actions=[
            sc2_actions.FunctionCall(_SELECT_POINT, [[1], [ _pos_x[0], _pos_y[0]]])
            ])

    #get the real every one position
    # and thread the group into three groups(if the number is nine)
    unitlist = unit_postion(obs, 1)
    for i in range(unitlist.__len__()//3):
        xy_per_marine[str(i)] = np.mean(np.array([unitlist[3*i], unitlist[3*i + 1], unitlist[3*i + 2]]), axis=0)

        obs = env.step(actions=[
            sc2_actions.FunctionCall(_SELECT_POINT, [[0], [unitlist[3 * i][1], unitlist[3 * i][0]]])
        ])
        obs = env.step(actions=[
            sc2_actions.FunctionCall(_SELECT_POINT, [[1], [unitlist[3 * i + 1][1], unitlist[3 * i + 1][0]]])
        ])
        obs = env.step(actions=[
            sc2_actions.FunctionCall(_SELECT_POINT, [[1], [unitlist[3 * i + 2][1], unitlist[3 * i + 2][0]]])
        ])
        obs = env.step(actions=[
            sc2_actions.FunctionCall(_SELECT_CONTROL_GROUP,
                                     [[_CONTROL_GROUP_SET], [i]])
        ])
    #if number isn't nine
    if unitlist.__len__()%3 != 0:
        length_all = unitlist.__len__()
        length_group =  unitlist.__len__()//3
        obs = env.step(actions=[
            sc2_actions.FunctionCall(_SELECT_POINT, [[0], [unitlist[length_all-1][1], unitlist[length_all-1][0]]])
        ])
        if unitlist.__len__()%3 == 2:
            obs = env.step(actions=[
                sc2_actions.FunctionCall(_SELECT_POINT,
                                         [[1], [unitlist[length_all - 2][1], unitlist[length_all - 2][0]]])
            ])
        obs = env.step(actions=[
            sc2_actions.FunctionCall(_SELECT_CONTROL_GROUP,
                                     [[_CONTROL_GROUP_SET], [length_group]])
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
            if group[1] > 5:    #1 group is too many army
                error = True
            army_count += group[1]
            
            if (id > 4) and (army_count < 8):   #2 group is not right in the common_T
                error = True

    if (army_count < env._obs[0].observation.player_common.army_count - 4):  #3 the period is new need to be 5
        error = True
        if army_count != 0:
            print("the army count is {0}".format(army_count))

    return error

UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'

def select_marine(env, obs):
    """make sure!!  the screen data is using the minimap data
                    the action data is using the screen data!"""
    player_y = 0
    player_x = 0
    group_id = 0
    group_list = update_group_list(obs)
    screen = []

    if (check_group_list(env, obs)):
        obs, xy_per_marine = init(env, obs)
        group_list = update_group_list(obs)

    # select random
    if  (len(group_list) > 0 ):
        while(player_y==0):
            group_id = np.random.choice(group_list)

            obs = env.step(actions=[
                    sc2_actions.FunctionCall(_SELECT_CONTROL_GROUP, [[
                    _CONTROL_GROUP_RECALL], [int(group_id)]])
                    ])
            #use the average group place to stand for the position
            _player_y, _player_x = np.nonzero(obs[0].observation["minimap"][_SELECTED_MINI] == 1)
            screen = obs[0].observation["minimap"][_PLAYER_RELATIVE_MINI]
            for i in range(len(_player_y)):
                screen[_player_y[i]][_player_x[i]] = _PLAYER_SELECTED_GROUP
            # use the screen data to get the action data,so the step_episode can be same
            _player_y, _player_x = np.nonzero(obs[0].observation["screen"][_SELECTED] == 1)
            player_y, player_x = np.average([_player_y, _player_x], axis=1)
            done = obs[0].step_type == environment.StepType.LAST
            if done:
                return obs, screen, group_id, [0, 0]
    else:
        group_id = 0
        player_x = 1
        player_y = 1
        obs = env.step(actions=[
            sc2_actions.FunctionCall(_NO_OP, [])
        ])
        screen = obs[0].observation["minimap"][_PLAYER_RELATIVE_MINI]

    try:
        #the priority is the same as the if...else
        return obs, screen, group_id, [player_x, player_y]
    except Exception as e:
        a = 1

def _map_mirror(screen):
    length = len(screen[0])
    mirror_screen = np.empty([length, length], dtype=np.int32)
    for i in range(length):
        for j in range(length):
            np.put(mirror_screen, j + i * length, screen[i][length - j - 1])

    return mirror_screen

def map_mirror(screen, action):
    length = len(screen[0])
    mirror_screen = np.empty([length, length], dtype=np.int32)
    for i in range(length):
        for j in range(length):
            np.put(mirror_screen, j + i * length, screen[i][length - j - 1])

    if (action == 3):  # LEFT
        mirror_action = 2
    elif (action == 2):  # RIGHT
        mirror_action = 3
    else:
        mirror_action = 0
    return mirror_screen, mirror_action


def check_coord(coord):
    if (coord[0] < 0):
        coord[0] = 0
    elif (coord[0] > check_coord_mapsize):
        coord[0] = check_coord_mapsize

    if (coord[1] < 0):
        coord[1] = 0
    elif (coord[1] > check_coord_mapsize):
        coord[1] = check_coord_mapsize
    return coord


def marine_action(env, obs, player, action):
    coord = []
    step_length = 3

    if (len(player) == 2):
        if (action == 0):  # UP
            coord = [player[0], player[1] - step_length]
            coord = check_coord(coord)
            new_action = [
                sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
            ]

        elif (action == 1):  # DOWN
            coord = [player[0], player[1] + step_length]
            coord = check_coord(coord)
            new_action = [
                sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
            ]

        elif (action == 3):  # LEFT
            coord = [player[0] - step_length, player[1]]
            coord = check_coord(coord)
            new_action = [
                sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
            ]

        elif (action == 2):  # RIGHT
            coord = [player[0] + step_length, player[1]]
            coord = check_coord(coord)
            new_action = [
                sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])
            ]
        else:
            new_action = [sc2_actions.FunctionCall(_NO_OP, [])]
    else:
        new_action = [sc2_actions.FunctionCall(_NO_OP, [])]

    # deal the run_record problem
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
  while(len(unit_y) > 0 ) and (len(unit_x) > 0):
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

def flag_changeNetwork(obs):
    min_len = 80
    min_len_flag = 25
    screen_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
    enemy_y, enemy_x = np.average((screen_relative == _PLAYER_HOSTILE).nonzero(), axis=1)
    player_y, player_x = (screen_relative == _PLAYER_FRIENDLY).nonzero()

    for pos_y, pos_x in zip(player_y, player_x):
        length = np.linalg.norm([pos_y-enemy_y, pos_x-enemy_x])
        if length<min_len:
            min_len = length
            if min_len < min_len_flag:
                return True
    return False

