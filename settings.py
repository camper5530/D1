import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Settings:
    tasks_count = 100
    edge_min = 0
    edge_max = 60

    COLORS = {
        0: "Orchid",
        1: "Red",
        2: "Green",
        3: "Blue",
        4: "Khaki",
        5: "Pink",
        6: "Chocolate",
        7: "Gold",
        8: "Coral",
        9: "Orange",
        10: "Teal",
        11: "Maroon",
    }

    demo = False
    frame = True
    scenario_count = 1          # 3
    experiments = 1             # 100
    scenario_enemy_count = 4    # 4

    agents_count_array = [12, 56, 100]
    enemies_count_array = [0, 1, 25, 50]
    diagonal = ((edge_max - edge_min) ** 2 + (edge_max - edge_min) ** 2) ** (1. / 2)


class Results:
    res_max = [[], [], [], []]

    def __init__(self):
        self.results = []

    def Update(self):
        self.results.clear()

    def Collect(self, n):
        Results.res_max[n].append(np.mean(self.results))


class Agent:
    v = 0.5

    def __init__(self, x, y, i):
        self.x = x
        self.y = y
        self.x_s = [x]
        self.y_s = [y]
        self.theta = 0
        self.dist = 0
        self.task = False
        self.id = i
        self.complete = False

    def Print(self):
        print(self.x, self.y, self.theta)

    def Update(self):
        self.theta = 0
        self.dist = 0
        self.task = False
        self.complete = False
        self.x_s = [self.x]
        self.y_s = [self.y]

    def CalculateDistance(self, p):
        return ((self.x - p[0]) ** 2 + (self.y - p[1]) ** 2) ** (1. / 2)

    def AngleBetweenPoints(self, p1, p2):
        d1 = p2[0] - p1[0]
        d2 = p2[1] - p1[1]
        if d1 == 0:
            if d2 == 0:  # same points?
                deg = 0
            else:
                deg = 0 if p1[1] > p2[1] else 180
        elif d2 == 0:
            deg = 90 if p1[0] < p2[0] else 270
        else:
            deg = math.atan(d2 / d1) / np.pi * 180
            lowering = p1[1] < p2[1]
            if (lowering and deg < 0) or (not lowering and deg > 0):
                deg += 270
            else:
                deg += 90
        return deg

    def Control(self, goal_x, goal_y):
        if self.CalculateDistance([goal_x, goal_y]) > self.v:
            self.theta = self.AngleBetweenPoints([self.x, self.y], [goal_x, goal_y])
            self.x += Agent.v * math.sin(np.deg2rad(self.theta))
            self.y -= Agent.v * math.cos(np.deg2rad(self.theta))
            self.dist += Agent.v
            return False
        else:
            return True

    def Allocate(self, tasks):
        solutions = []
        for t in tasks:
            solutions.append(self.CalculateDistance([t.x, t.y]))
        timer = 0
        done = False
        while not done:
            best = np.argmin(solutions)
            check_array = [Settings.diagonal for i in range(Settings.tasks_count)]
            if tasks[best].taken == -1 and tasks[best].done is False:
                self.task = best
                return best
            else:
                solutions[best] = Settings.diagonal
            if np.array_equal(solutions, check_array):
                self.task = -1
                return False
            if timer > Settings.tasks_count:
                self.task = -1
                return False
            timer += 1


class Task:
    x = 0
    y = 0
    done = False
    taken = -1

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def Update(self):
        self.done = False
        self.taken = -1

