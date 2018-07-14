import numpy as np
import copy

def maximal_decompositions(graph, r_matching):
    n = len(graph)
    num = 0
    gr = copy.deepcopy(graph)
    ans = []
    for i in r_matching:
        j = r_matching[i]
        if gr[i][j] == 1:
            gr[i][j] = 0
            gr[j][i] = 0
            ans.append("{}{}".format(i,j))
            num += 1
        elif gr[i][i] + gr[j][j] == 4:
            gr[i][i] = 0
            gr[j][j] = 0
            ans.append("{}{}".format(i,j))
            num += 1
    Gr = Graph(gr, r_matching)
    Gr.decompose()
    t, c = Gr.cycle_size, Gr.decompositons
    # t, c = decompose(gr, r_matching, v, w, cycles=cycles)
    return t + num, [cycles.union({cycle for cycle in ans}) for cycles in c]

def make_cycle(cycle):
    first = [str(i) for i in cycle[:-1]]
    second = cycle[1:]
    pairs = list(zip(first, second)) 
    start = pairs.index(min(pairs))
    return "".join(first[start - len(first):] + first[:start])



class Graph:
    def __init__(self, graph, r_matching):
        self.graph = graph
        self.r_matching = r_matching
        self.size = len(graph)
        self.decompositons = []
        self.cycle_size = 0

    def decompose(self, current_cycle=[], cycles=[]):
        if current_cycle:
            current = current_cycle[-1]
            current = self.r_matching[current]
            current_cycle.append(current)
            
            if current_cycle[0] == current:
                cycles.append(current_cycle)
                if len(cycles) >= self.cycle_size:
                    normalized = {min(make_cycle(i), make_cycle(i[::-1])) for i in cycles}
                    if len(cycles) > self.cycle_size:
                        self.decompositons = [normalized]
                        self.cycle_size = len(cycles)
                    elif normalized not in self.decompositons:
                        self.decompositons.append(normalized)
                self.decompose([], cycles)
                cycles.pop()

            else:
                for i in range(self.size):
                    if self.graph[current][i] > 0:
                        self.graph[current][i] = 0
                        self.graph[i][current] = 0
                        current_cycle.append(i)
                        self.decompose(current_cycle, cycles)
                        current_cycle.pop()
                        current_cycle.pop()
                        self.graph[current][i] += 1
                        self.graph[i][current] += 1

        else:
            for i in range(self.size):
                for j in range(self.size):
                    if self.graph[i][j] != 0:
                        self.graph[i][j] = 0
                        self.graph[j][i] = 0
                        current_cycle = [i, j]
                        self.decompose(current_cycle, cycles)
                        self.graph[i][j] += 1
                        self.graph[j][i] += 1

        





