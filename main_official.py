# -*- coding: utf8 -*-
import glob
import math
import os
import copy
import random
import sys

import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from settings import Settings as S
from settings import Results as R
from settings import Agent as A
from settings import Task as T


def CreateRawDataGraphic(k):
    colors = ["gray", "b", "g", "r"]
    alpha = 0.3
    index = [i + 1 for i in range(S.experiments)]
    plt.cla()
    plt.figure(figsize=(10, 4))
    plt.xlabel("Experiment number", fontsize=10)
    plt.ylabel("Distance, m")
    for i in range(S.scenario_enemy_count):
        malicious_count = int(S.agents_count_array[k] * S.enemies_count_array[i] / 100)
        if i == 1:
            malicious_count = 1
        label = ""
        if malicious_count == 0:
            label += "No malicious agents"
        elif malicious_count == 1:
            label += "1 malicious agent"
        else:
            label += str(malicious_count) + " malicious agents"
        plt.plot(index, np.array(R.res_max[i]), color=colors[i], alpha=alpha)
        plt.plot([index[0], index[-1]], [np.mean(R.res_max[i]), np.mean(R.res_max[i])], label=label, color=colors[i])
    plt.legend(loc='upper right')
    plt.grid(True)
    pic = 'results/raw_graphic_for_' + str(S.agents_count_array[k]) + "_agents.png"
    plt.savefig(pic)
    CreateResultImage(pic)

    plt.cla()
    plt.figure(figsize=(10, 4))
    bp = plt.boxplot(x=R.res_max, patch_artist=True, medianprops=dict(color="black"))
    for i in range(len(bp['boxes'])):
        bp['boxes'][i].set(facecolor=colors[i])
    y1 = int(S.agents_count_array[k] * S.enemies_count_array[1] / 100)
    y2 = int(S.agents_count_array[k] * S.enemies_count_array[2] / 100)
    y3 = int(S.agents_count_array[k] * S.enemies_count_array[3] / 100)
    plt.xticks([1, 2, 3, 4], [0, y1, y2, y3])
    plt.xlabel("Count of malicious agents")
    plt.ylabel("Distance, m")
    plt.grid(True)
    pic = 'results/result_for_' + str(S.agents_count_array[k]) + "_robots.png"
    plt.savefig(pic)
    CreateResultImage(pic)


def CreateResultImage(pic):
    image = Image.open(pic)
    new_im = Image.new('RGB', (850, 360))
    new_im.paste(image.crop((60, 40, 910, 400)), (0, 0))
    new_im.save(pic)


def AddDataToTable(k):
    values = ["Max value", "Mean value", "Min value"]
    file_m = "a"
    if k == 0:
        file_m = "w"
    with open("results/table.txt", file_m) as file:
        if k == 0:    # table header
            file.write("Criteria" + '\t')
            file.write("Distance, m" + '\t')
            file.write('\r\n')
        file.write("Agents count / malicious agents count" + '\t')
        for n in range(S.scenario_enemy_count):
            m = int(S.agents_count_array[k] * S.enemies_count_array[n] / 100)
            if n == 1:
                m = 1
            t = str(S.agents_count_array[k]) + " / " + str(m)
            file.write(t + '\t')
        file.write('\r\n')

        for v in range(len(values)):
            file.write(str(values[v]) + '\t')
            for n in range(S.scenario_enemy_count):
                if v == 0:
                    d = np.max(R.res_max[n])
                elif v == 1:
                    d = np.mean(R.res_max[n])
                else:
                    d = np.min(R.res_max[n])
                file.write(str(round(d, 2)).replace('.', ',') + '\t')
            file.write('\r\n')


def Main():
    for k in range(S.scenario_count):       # scaling of agents count: 0 - 12, 1 - 56, 2 - 100
        for j in range(S.experiments):
            # generate the same input data for every simulation
            print("Experiment #", str(k) + "." + str(j))
            tasks_positions = np.random.normal(S.edge_max / 2, S.edge_max / 4, [S.tasks_count, 2])
            agents_positions = np.random.normal(S.edge_min, 1.5, [S.agents_count_array[k], 2])
            agents, tasks = [], []
            for t in tasks_positions:
                if t[0] > S.edge_max:
                    t[0] = S.edge_max
                elif t[0] < S.edge_min:
                    t[0] = S.edge_min
                if t[1] > S.edge_max:
                    t[1] = S.edge_max
                elif t[1] < S.edge_min:
                    t[1] = S.edge_min
                tasks.append(T(t[0], t[1]))
                plt.scatter(t[0], t[1])
            for i in range(S.agents_count_array[k]):
                agents.append(A(agents_positions[i][0], agents_positions[i][1], i))

            for n in range(S.scenario_enemy_count):
                # generate different count of malicious agents for every simulation
                malicious_agents_count = int(S.agents_count_array[k] * S.enemies_count_array[n] / 100)
                if n == 1:
                    malicious_agents_count = 1
                malicious_agents = random.sample(range(S.agents_count_array[k]), malicious_agents_count)

                counter = 0
                while counter < S.tasks_count + S.agents_count_array[k]:
                    if S.demo:
                        plt.cla()
                        plt.scatter([-5, S.edge_max], [-5, S.edge_max], c="white")

                    for t in tasks:
                        if not t.done and S.demo:
                            c = "grey"
                            if t.taken > -1:
                                c = S.COLORS[t.taken]
                            plt.scatter(t.x, t.y, marker="x", c=c)
                    for i in range(S.agents_count_array[k]):
                        if agents[i].task is False:  # agent has not task
                            taken_task = agents[i].Allocate(tasks)
                            if taken_task is not False:
                                tasks[taken_task].taken = i
                            else:
                                continue
                            if S.demo:
                                plt.text(agents[i].x + .5, agents[i].y + .5, s="?..")
                        elif agents[i].task == -1:  # all tasks are complete - return to base
                            if not agents[i].complete:
                                if agents[i].Control(agents[i].x_s[0], agents[i].y_s[0]):
                                    counter += 1
                                    agents[i].complete = True
                            else:
                                continue
                        else:
                            if agents[i].id in malicious_agents:
                                tasks[agents[i].task].Update()
                                agents[i].task = -1
                                continue
                            else:
                                if agents[i].Control(tasks[agents[i].task].x, tasks[agents[i].task].y):
                                    if not tasks[agents[i].task].done:
                                        tasks[agents[i].task].done = True
                                        agents[i].x_s.append(tasks[agents[i].task].x)
                                        agents[i].y_s.append(tasks[agents[i].task].y)
                                        counter += 1
                                        agents[i].task = False
                    if S.demo:
                        for i in range(S.agents_count_array[k]):
                            plt.scatter(agents[i].x, agents[i].y, c=S.COLORS[i])
                            plt.plot(agents[i].x_s, agents[i].y_s, c=S.COLORS[i], alpha=0.2)
                            last = len(agents[i].x_s) - 1
                            plt.plot(
                                [agents[i].x_s[last], agents[i].x],
                                [agents[i].y_s[last], agents[i].y],
                                c=S.COLORS[i],
                                alpha=0.2
                            )
                        plt.title(str(S.agents_count_array[k]) + " agents / " + str(
                            malicious_agents_count) + " malicious agents")
                        plt.grid(True)
                        plt.xlabel("x, m", fontsize=12)
                        plt.ylabel("y, m", fontsize=12)
                        plt.pause(0.00000001)
                r = R()
                for a in agents:
                    if a.id not in malicious_agents:
                        r.results.append(a.dist)
                    a.Update()
                for t in tasks:
                    t.Update()
                r.Collect(n)
        CreateRawDataGraphic(k)
        AddDataToTable(k)
        R.res_max = [[] for i in range(S.scenario_enemy_count)]


if __name__ == '__main__':
    Main()
