import argparse
import numpy as np
import copy
import itertools

 
def check(counts):
    return max(counts).all() == 0
 
def dfs(graph, start):
    visited, stack = [], [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.append(vertex)
            for i in range(len(graph)):
                if i not in visited and graph[vertex][i] != 0:
                    stack.append(i)
    return visited
 
def filter1(graph): #does not consist adequate 0-0
    n = len(graph)
    for i in range(n):
        for j in range(i+1, n):
            if graph[i][i]*graph[j][j]*graph[i][j] != 0:
                return False
    return True

def  is_identical(graph1, graph2, p):
    n = len(graph1)
    for i in range(n):
        for j in range(i+1, n):
            if graph1[i][j] != graph2[p[i]][p[j]]:
                return False
    return True

def trace(graph):
    n = len(graph)
    tr = 0
    for i in range(n):
        tr += graph[i][i]
    return tr

def is_isomorhic(graph1, graph2, k): # the graphs should have k degrees degree1 and n-k degrees degree2  
    n = len(graph1) 
    if trace(graph1) != trace(graph2):
        return False
  
    permutations_1 = itertools.permutations([i for i in range(k)])
    permutations_2 = itertools.permutations([i for i in range(k,n)])

    perm2 =[]
    for p2 in permutations_2:
        perm2.append(p2)
    for p1 in permutations_1:
        for p2 in perm2:
            p = list(p1) + list(p2)
            if  is_identical(graph1, graph2, p):
                return True
    return False

def make_abgraph(gr, matching):
    n = len(gr)
    graph = copy.deepcopy(gr)
    for i in matching:
        graph[i][matching[i]] = -1
    return graph    

def is_сontains_adequate_subgraphs(gr, matching, ad_index):
    n = len(gr)
    graph = make_abgraph(gr,matching)
    if len(dfs(graph,0)) != len(graph): #is conneccted
        return True
    if not filter1(graph):
        return True
    if len(adequates[ad_index]) == 0:
        return False
    a_size = len(adequates[ad_index][0])
    if a_size == 0:
        return False
    permutations = itertools.permutations([i for i in range(n)], a_size)
    for perm in permutations:
        for agraph in adequates[ad_index]:
            if is_identical(agraph, graph, perm):
                return True
    return False

def cycles_number(graph, r_matching):
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


def is_deg_list(count):
    sum = 0
    for i in range(len(count)):
        sum += count[i]
    sum = int(sum)
    return sum%2

def next(count, k):
    n = len(count)
    for i in range(n-1,-1, -1):
        if count[i] < k:
            count[i] += 1
            for j in range(i,n):
                count[j] = count[i]
            if is_deg_list(count) == 0:
                return count
            return next(count, k) 
    count[0] = k+1
    return count


def ABcycles(graph):
    Bgraphs = set()
    for num_mR in range(len(matchings)):
        cycA = cycles_number(graph,matchings[num_mR])[0]
        for num_mB in range(len(bmatchings)):
            if num_mB in Bgraphs:
                continue
            bmatching = copy.deepcopy(bmatchings[num_mB])
            is_cotinue = False
            for i in range(len(counts)):
                if i in bmatching:
                    if graph[i][bmatching[i]] != 0 or graph[i][i] + graph[bmatching[i]][bmatching[i]] == 4:
                        t = bmatching[i]
                        del bmatching[t]
                        del bmatching[i]
            cycB = 0
            if len(bmatching) !=0:
                graphB = np.zeros((size, size))
                for i in bmatching:
                    graphB[i][bmatching[i]] = 1
                    graphB[bmatching[i]][i] = 1
                cycB = cycles_number(graphB,matchings[num_mR])[0]
            if cycA + cycB >= (a_dupl + b_dupl) * size / 4:
                is_cont = False
                for idx in range(len(adequates)):
                    if is_сontains_adequate_subgraphs(graph, bmatching, idx):
                        is_cont = True
                        break
                if not is_cont:
                    Bgraphs.add(num_mB)

    if len(Bgraphs) != 0:
        AB_graphs = []
        for el in Bgraphs:
            ab_graph = make_abgraph(graph, bmatchings[el])
        
            is_OK = True
            for gr in AB_graphs:
                if is_isomorhic(ab_graph, gr, 0):
                    is_OK = False
                    break
            if is_OK:
                is_cotinue = False
                for i in bmatchings[el]:
                    if graph[bmatchings[el][i]][i] == 1:
                        is_cotinue = True
                        break
                if is_cotinue:
                    continue
                #print(ab_graph)
                write_graph(ab_graph,graph_file)
                draw_tex(ab_graph, tex_file, 5);
                AB_graphs.append(ab_graph)


def write_graph(graph, file):
    file.write("[")
    for i in range(len(graph)):
        file.write("[ ")
        for j in range(len(graph)):
            file.write(str(int(graph[i][j])) + ' ')
        file.write("]")
        if (i+1 == len(graph)):
            file.write("]\n\n")
        else:
            file.write("\n")

num = 0
def draw_tex(graph, file,  columns_number):
    global num
    if num != 0 and num % columns_number == 0:
        file.write("\\\\ \n \\\\ \n")
    else:
        file.write("&\n")

    vertices = ["{{(0, 0)/0/}, {(0,1)/1/}, {(1,1)/2/}, {(1,0)/3/}}",
    "{{(0, 0)/0/}, {(1,1.5)/1/}, {(2,1.5)/2/}, {(3,0)/3/}, {(2,-1.5)/4/}, {(1,-1.5)/5/}}",
    "{{(1,0)/0/}, {(2,0)/1/}, {(3,1)/2/}, {(3,2)/3/}, {(2,3)/4/}, {(1,3)/5/},{(0,2)/6/}, {(0,1)/7/}"]

    file.write("\\begin{tikzpicture}[scale=0.5, baseline] \n" +
        "\\foreach \pos/\\name/\weight in" + vertices[int(size/2-2)] +
        "\n\t\\node[vertex] (\\name) at \pos {\weight};\n")
    #file.write(start_strings)
    file.write("\\foreach \source/ \dest in {")
    A_edges = ""
    for i in range(len(graph)):
        for j in range(i+1, len(graph)):
            if graph[i][j] > 0:
                A_edges += str(i) + "/" + str(j) + ", "

    file.write(A_edges[:-2] + "}\n\t\path[A-edge] (\source) edge (\dest);")

    file.write("\\foreach \source/ \dest in {")
    B_edges = ""
    for i in range(len(graph)):
        for j in range(i+1, len(graph)):
            if graph[i][j] < 0:
                B_edges += str(i) + "/" + str(j) + ", "
    file.write(B_edges[:-2] + "}\n\t\path[B-edge] (\source) edge (\dest);")
    file.write("\\foreach \source/ \dest in {")
    loops =""
    for i in range(len(graph)):
        if graph[i][i] == 2:
            loops += str(i) + ", "
    file.write(loops[:-2] + "}\n\t \path[A-edge] (\source) edge[my loop]  (\dest);")
    file.write("\n\end{tikzpicture}\n")
    num += 1

def run(v, graph, counts, A_forms, prev=None):
    if prev is None:
        prev = v-1
    if v == len(graph):
        if check(counts):
            is_cotinue = True
            for g in A_forms:
                if is_isomorhic(g, graph, 0):
                    is_cotinue = False
                    break
            if is_cotinue:
                A_forms.append(copy.deepcopy(graph))
                ABcycles(graph)
        return 
    if counts[v] <= 0:
        run(v+1, graph, counts, A_forms)
   
    for w in range(prev+1, len(graph)):
        add = 1
        if v == w:
            add = 2
        if counts[w] >= add and graph[v][w] == 0 and counts[v] > 0:
            graph[v][w] += 1
            graph[w][v] += 1
            counts[w] -= add
            if v != w:
                counts[v] -= 1

            run(v, graph, counts, A_forms, w)
            if v != w:
                counts[v] += 1
            graph[v][w] -= 1
            graph[w][v] -= 1
            counts[w] += add

def read_matchings(file_name):
    mathings_file = open(file_name, "r")
    matchings = []
    bmatchings = [{}]
    for line in mathings_file:
        line = line.split();
        matching = {}
        i = 0
        while(i < len(line)):
            matching[int(line[i])] = int(line[i+1])
            matching[int(line[i+1])] = int(line[i])
            i += 2
        if len(matching) == size:
            matchings.append(matching)
        if (b_dupl == 1):
            bmatchings.append(matching)
    mathings_file.close()
    return (matchings, bmatchings)

def read_adequates(file_name):
    adequates = [[] for i in range(10)]
    ad_file = open(file_name, "r")
    graph = []
    for line in ad_file:
        line = line.split();
        if len(line) == 0:
            continue
        #print(line)
        if line[0] == "[[":
            if (len(graph) != 0):
                adequates[int(len(graph)/2)-2].append(graph)
            i = 0
            len_graph = len(line) - 2
            graph = [[0 for i in range(len_graph)] for j in range(len_graph)]
        else:
            i += 1
        for j in range(1, len_graph + 1):
            graph[i][j-1] = int(line[j])
    adequates[int(len(graph)/2)-2].append(graph)
    return adequates


parser = argparse.ArgumentParser()
parser.add_argument("-l", help='size of subgraphs')
parser.add_argument("-A", help='number dublications of a')
parser.add_argument("-B", help='is b or not (1 or 0)')
parser.add_argument("-m", help='path to file with l-size mathingsis')
parser.add_argument("-ad", help='path to file with adequate graphs with size <l')
args = parser.parse_args()
size = int(args.l)
a_dupl = int(args.A)
b_dupl = int(args.B)


matchings, bmatchings = read_matchings(args.m)
adequates = read_adequates(args.ad) 

graph  = [[0 for i in range(size)] for j in range(size)]
A_forms = []
counts = np.ones(size) * 0

tex_file = open("graphs.tex", "w")
graph_file = open("graphs.text", "w")

while (counts[0] < a_dupl):   
    counts = next(counts, a_dupl)
    A_forms = []
    run(0, graph, counts, A_forms)
print(num)

