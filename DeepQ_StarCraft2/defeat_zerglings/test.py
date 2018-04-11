import numpy as np

unitlist = [1, 2, 3]
for t in range(10):
    if t == 0:
        Action_Choose = False
    Action_Choose = not(Action_Choose)
    print(Action_Choose)