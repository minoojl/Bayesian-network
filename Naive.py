import os
import copy
import distutils.util
from gunfolds.utils import bfutils
from gunfolds.utils import zickle as zkl
import numpy as np
import time, socket
import scipy
from gunfolds.solvers.clingo_rasl import drasl, drasl_command
from gunfolds.utils import graphkit as gk
from gunfolds.utils.calc_procs import get_process_count
from gunfolds.estimation import linear_model as lm
from gunfolds import conversions as cv
from progressbar import ProgressBar, Percentage
from numpy import linalg as la
import networkx as nx
from math import log

# Manually set the arguments
class Args:
    CAPSIZE = 0
    BATCH = 1
    PNUM = int(min(64, get_process_count(1)))
    NODE = 8
    DEN = 0.14
    GTYPE = 'f'
    TIMEOUT = 120
    THRESHOLD = 5
    SCC = 't'
    SCCMEMBERS = 't'
    UNDERSAMPLING = 2
    MAXU = 15

args = Args()
TIMEOUT = args.TIMEOUT * 60 * 60
GRAPHTYPE = bool(distutils.util.strtobool(args.GTYPE))
DENSITY = float(args.DEN)
graphType = 'ringmore' if GRAPHTYPE else 'bp_mean'
SCC = bool(distutils.util.strtobool(args.SCC))
SCC_members = bool(distutils.util.strtobool(args.SCCMEMBERS))
SCC = True if SCC_members else SCC
u_rate = args.UNDERSAMPLING
k_threshold = args.THRESHOLD
EDGE_CUTOFF = 0.01
noise_naive_bayesian = 0.1

drop_bd_normed_errors_comm = []
drop_bd_normed_errors_omm = []
dir_errors_omm = []
dir_errors_comm = []
opt_dir_errors_omm = []
opt_dir_errors_comm = []
g_dir_errors_omm = []
g_dir_errors_comm = []
Gu_opt_dir_errors_omm = []
Gu_opt_dir_errors_comm = []
error_normalization = True

def round_tuple_elements(input_tuple, decimal_points=3):
    return tuple(round(elem, decimal_points) if isinstance(elem, (int, float)) else elem for elem in input_tuple)

def partition_distance(G1, G2):
    scc1 = list(nx.strongly_connected_components(G1))
    scc2 = list(nx.strongly_connected_components(G2))
    vi = 0
    for s1 in scc1:
        for s2 in scc2:
            intersection = len(s1.intersection(s2))
            if intersection > 0:
                vi += intersection * (log(intersection) - log(len(s1)) - log(len(s2)))
    vi *= 2
    vi /= log(len(G1.nodes))
    return vi

def get_strongly_connected_components(graph):
    return [c for c in nx.strongly_connected_components(graph)]

def calculate_jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def quantify_graph_difference(graph1, graph2):
    scc1 = get_strongly_connected_components(graph1)
    scc2 = get_strongly_connected_components(graph2)
    scc_sets1 = set([frozenset(component) for component in scc1])
    scc_sets2 = set([frozenset(component) for component in scc2])
    intersection = len(scc_sets1.intersection(scc_sets2))
    union = len(scc_sets1.union(scc_sets2))
    jaccard_similarity = intersection / union if union > 0 else 0
    return jaccard_similarity

def makeConnected(G):
    weakly_connected_components = list(nx.weakly_connected_components(G))
    for i in range(len(weakly_connected_components) - 1):
        G.add_edge(list(weakly_connected_components[i])[0], list(weakly_connected_components[i + 1])[0])
    return G

def rmBidirected(gu):
    g = copy.deepcopy(gu)
    for v in g:
        for w in list(g[v]):
            if g[v][w] == 2:
                del g[v][w]
            elif g[v][w] == 3:
                g[v][w] = 1
    return g

def transitionMatrix4(g, minstrength=0.1, distribution='normal', maxtries=1000):
    A = cv.graph2adj(g)
    edges = np.where(A == 1)
    s = 2.0
    c = 0
    pbar = ProgressBar(widgets=['Searching for weights: ', Percentage(), ' '], maxval=maxtries).start()
    while s > 1.0:
        minstrength -= 0.001
        A = lm.initRandomMatrix(A, edges, distribution=distribution)
        x = A[edges]
        delta = minstrength / np.min(np.abs(x))
        A[edges] = delta * x
        l = lm.linalg.eig(A)[0]
        s = np.max(np.real(l * scipy.conj(l)))
        c += 1
        if c > maxtries:
            return None
        pbar.update(c)
    pbar.finish()
    return A

def genData(A, rate=2, burnin=100, ssize=5000, noise=0.1, dist='normal'):
    Agt = A
    data = drawsamplesLG(Agt, samples=burnin + (ssize * rate), nstd=noise)
    data = data[:, burnin:]
    return data[:, ::rate]

def drawsamplesLG(A, nstd=0.1, samples=100):
    n = A.shape[0]
    data = np.zeros([n, samples])
    data[:, 0] = nstd * np.random.randn(A.shape[0])
    for i in range(1, samples):
        data[:, i] = A @ data[:, i - 1] + nstd * np.random.randn(A.shape[0])
    return data

def drawsamplesMA(A, nstd=0.1, samples=100, order=5):
    n = A.shape[0]
    data = scipy.zeros([n, samples])
    data[:, 0] = nstd * scipy.random.randn(A.shape[0])
    for i in range(1, samples):
        if i > order:
            result = 0
            for j in range(order):
                result += np.dot(1 / (j + 1) * A, data[:, i - 1 - j]) + nstd * np.dot(1 / (j + 1) * A, scipy.random.randn(A.shape[0]))
            data[:, i] = result
        else:
            data[:, i] = scipy.dot(A, data[:, i - 1]) + nstd * scipy.random.randn(A.shape[0])
    return data

def AB2intAB_1(A, B, th=0.09):
    A[amap(lambda x: abs(x) > th, A)] = 1
    A[amap(lambda x: abs(x) < 1, A)] = 0
    B[amap(lambda x: abs(x) > th, B)] = 1
    B[amap(lambda x: np.abs(x) < 1, B)] = 0
    np.fill_diagonal(B, 0)
    return A, B

def amap(f, a):
    v = np.vectorize(f)
    return v(a)

print('_____________________________________________')
dataset = zkl.load('datasets/ringmore_n10d13.zkl')
GT = dataset[args.BATCH-1]
mask = cv.graph2adj(GT)

G = np.clip(np.random.randn(*mask.shape) * 0.2 + 0.5, 0.3, 0.7)
Con_mat = G * mask

w, v = la.eig(Con_mat)
res = all(ele <= 1 for ele in abs(w))

while not res:
    G = np.clip(np.random.randn(*mask.shape) * 0.2 + 0.5, 0.3, 0.7)
    Con_mat = G * mask
    w, v = la.eig(Con_mat)
    res = all(ele <= 1 for ele in abs(w))

'''Naive Bayesian'''
dd = genData(Con_mat, rate=u_rate, ssize=2000*u_rate, noise=noise_naive_bayesian)

MAXCOST = 10000
g_estimated, A, B = lm.data2graph(dd, th=EDGE_CUTOFF * k_threshold)
DD = (np.abs((np.abs(A/np.abs(A).max()) + (cv.graph2adj(g_estimated) - 1))*MAXCOST)).astype(int)
BD = (np.abs((np.abs(B/np.abs(B).max()) + (cv.graph2badj(g_estimated) - 1))*MAXCOST)).astype(int)

GT_at_actual_U = bfutils.undersample(GT, u_rate)
jaccard_similarity = quantify_graph_difference(gk.graph2nx(g_estimated), gk.graph2nx(GT_at_actual_U))
g_estimated_errors_GT_at_actual_U = gk.OCE(g_estimated, GT_at_actual_U, undirected=False, normalized=error_normalization)['total']

print("Gtype : {0:}, intended sampling rate : {1:} Num nodes  : {2:}, dens : {3:}\nBatch : {4:}\n"
      "g_estimated error with GT at intended U: {5:}\n"
      "using estimated SCC: {6:}".format(graphType, u_rate, args.NODE, DENSITY, args.BATCH,
                                         round_tuple_elements(g_estimated_errors_GT_at_actual_U), SCC_members))

print('jaccard similarity is: ' +str(jaccard_similarity))
'''task optimization'''
if SCC_members:
    members = nx.strongly_connected_components(gk.graph2nx(g_estimated))
else:
    members = nx.strongly_connected_components(gk.graph2nx(GT_at_actual_U))
    
startTime = int(round(time.time() * 1000))

r_estimated = drasl([g_estimated], weighted=True, capsize=0, timeout=TIMEOUT,
                    urate=min(args.MAXU, (3 * len(g_estimated) + 1)),
                    dm=[DD],
                    bdm=[BD],
                    scc=SCC,
                    scc_members=members,
                    GT_density=int(1000*gk.density(GT)),
                    #edge_weights=(1, 1), pnum=args.PNUM, optim='optN')
                    edge_weights=(1, 1, 1, 1, 1),
                    pnum=args.PNUM, optim='optN')
endTime = int(round(time.time() * 1000))
sat_time = endTime - startTime

print('number of optimal solutions is', len(r_estimated))

min_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
min_norm_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
min_val = 1000000
min_cost = 10000000

for answer in r_estimated:
    curr_errors = gk.OCE(bfutils.undersample(bfutils.num2CG(answer[0][0], len(GT)),answer[0][1][0]), g_estimated)
    curr_normed_errors = gk.OCE(bfutils.undersample(bfutils.num2CG(answer[0][0], len(GT)),answer[0][1][0]), g_estimated, normalized=True)
    curr_cost = answer[1]
    if  (curr_errors['total'][0] + curr_errors['total'][1]) < min_val:
        min_err = curr_errors
        min_norm_err = curr_normed_errors
        min_cost = curr_cost
        min_val =  (curr_errors['total'][0] + curr_errors['total'][1])
        min_answer_WRT_GuOptVsGest = answer
    elif (curr_errors['total'][0] + curr_errors['total'][1]) == min_val:
        if curr_cost < min_cost:
            min_err = curr_errors
            min_norm_err = curr_normed_errors
            min_cost = curr_cost
            min_val = (curr_errors['total'][0] + curr_errors['total'][1])
            min_answer_WRT_GuOptVsGest = answer

'''G1_opt - the solution of optimization problem (r_estimated from g_estimated) in causal time scale'''
G1_opt_WRT_GuOptVsGest = bfutils.num2CG(min_answer_WRT_GuOptVsGest[0][0], len(g_estimated))

'''Gu_opt - the solution of optimization problem (r_estimated from g_estimated) in measured time scale'''
Gu_opt_WRT_GuOptVsGest = bfutils.undersample(G1_opt_WRT_GuOptVsGest, min_answer_WRT_GuOptVsGest[0][1][0])
'''network_GT_U - the GT  in measured time scale'''
network_GT_U_WRT_GuOptVsGest = bfutils.undersample(GT, min_answer_WRT_GuOptVsGest[0][1][0])

Gu_opt_errors_network_GT_U_WRT_GuOptVsGest = gk.OCE(Gu_opt_WRT_GuOptVsGest, network_GT_U_WRT_GuOptVsGest, undirected=False, normalized=error_normalization)['total']
Gu_opt_errors_g_estimated_WRT_GuOptVsGest = gk.OCE(Gu_opt_WRT_GuOptVsGest, g_estimated, undirected=False, normalized=error_normalization)['total']
G1_opt_error_GT_WRT_GuOptVsGest = gk.OCE(G1_opt_WRT_GuOptVsGest, GT, undirected=False, normalized=error_normalization)['total']
print('*******************************************')
print('results with respect to Gu_opt Vs. G_estimate ')
print('U rate found to be:' + str(min_answer_WRT_GuOptVsGest[0][1][0]))
print('Gu_opt_errors_network_GT_U = ', round_tuple_elements(Gu_opt_errors_network_GT_U_WRT_GuOptVsGest))
print('Gu_opt_errors_g_estimated', round_tuple_elements(Gu_opt_errors_g_estimated_WRT_GuOptVsGest))
print('G1_opt_error_GT', round_tuple_elements(G1_opt_error_GT_WRT_GuOptVsGest))

### minimizing with respect to Gu_opt Vs. GTu
min_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
min_norm_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
min_val = 1000000
min_cost = 10000000
for answer in r_estimated:
    curr_errors = gk.OCE(bfutils.undersample(bfutils.num2CG(answer[0][0], len(GT)),answer[0][1][0]), bfutils.undersample(GT, answer[0][1][0]))
    curr_normed_errors = gk.OCE(bfutils.undersample(bfutils.num2CG(answer[0][0], len(GT)),answer[0][1][0]), bfutils.undersample(GT, answer[0][1][0]), normalized=True)
    curr_cost = answer[1]
    if  (curr_errors['total'][0] + curr_errors['total'][1]) < min_val:
        min_err = curr_errors
        min_norm_err = curr_normed_errors
        min_cost = curr_cost
        min_val =  (curr_errors['total'][0] + curr_errors['total'][1])
        min_answer_WRT_GuOptVsGTu = answer
    elif (curr_errors['total'][0] + curr_errors['total'][1]) == min_val:
        if curr_cost < min_cost:
            min_err = curr_errors
            min_norm_err = curr_normed_errors
            min_cost = curr_cost
            min_val = (curr_errors['total'][0] + curr_errors['total'][1])
            min_answer_WRT_GuOptVsGTu = answer

'''G1_opt - the solution of optimization problem (r_estimated from g_estimated) in causal time scale'''
G1_opt_WRT_GuOptVsGTu = bfutils.num2CG(min_answer_WRT_GuOptVsGTu[0][0], len(g_estimated))

'''Gu_opt - the solution of optimization problem (r_estimated from g_estimated) in measured time scale'''
Gu_opt_WRT_GuOptVsGTu = bfutils.undersample(G1_opt_WRT_GuOptVsGTu, min_answer_WRT_GuOptVsGTu[0][1][0])
'''network_GT_U - the GT  in measured time scale'''
network_GT_U_WRT_GuOptVsGTu = bfutils.undersample(GT, min_answer_WRT_GuOptVsGTu[0][1][0])

Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu = gk.OCE(Gu_opt_WRT_GuOptVsGTu, network_GT_U_WRT_GuOptVsGTu, undirected=False, normalized=error_normalization)['total']
Gu_opt_errors_g_estimated_WRT_GuOptVsGTu = gk.OCE(Gu_opt_WRT_GuOptVsGTu, g_estimated, undirected=False, normalized=error_normalization)['total']
G1_opt_error_GT_WRT_GuOptVsGTu = gk.OCE(G1_opt_WRT_GuOptVsGTu, GT, undirected=False, normalized=error_normalization)['total']
print('*******************************************')
print('results of minimizing with respect to Gu_opt Vs. GTu')
print('U rate found to be:' + str(min_answer_WRT_GuOptVsGTu[0][1][0]))
print('Gu_opt_errors_network_GT_U = ', round_tuple_elements(Gu_opt_errors_network_GT_U_WRT_GuOptVsGTu))
print('Gu_opt_errors_g_estimated', round_tuple_elements(Gu_opt_errors_g_estimated_WRT_GuOptVsGTu))
print('G1_opt_error_GT', round_tuple_elements(G1_opt_error_GT_WRT_GuOptVsGTu))

### minimizing with respect to G1_opt Vs. GT
min_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
min_norm_err = {'directed': (0, 0), 'bidirected': (0, 0), 'total': (0, 0)}
min_val = 1000000
min_cost = 10000000
for answer in r_estimated:
    curr_errors = gk.OCE(bfutils.num2CG(answer[0][0], len(GT)),GT)
    curr_normed_errors = gk.OCE(bfutils.num2CG(answer[0][0], len(GT)), GT, normalized=True)
    curr_cost = answer[1]
    if (curr_errors['total'][0] + curr_errors['total'][1]) < min_val:
        min_err = curr_errors
        min_norm_err = curr_normed_errors
        min_cost = curr_cost
        min_val = (curr_errors['total'][0] + curr_errors['total'][1])
        min_answer_WRT_G1OptVsGT = answer
    elif (curr_errors['total'][0] + curr_errors['total'][1]) == min_val:
        if curr_cost < min_cost:
            min_err = curr_errors
            min_norm_err = curr_normed_errors
            min_cost = curr_cost
            min_val = (curr_errors['total'][0] + curr_errors['total'][1])
            min_answer_WRT_G1OptVsGT = answer

'''G1_opt - the solution of optimization problem (r_estimated from g_estimated) in causal time scale'''
G1_opt_WRT_G1OptVsGT = bfutils.num2CG(min_answer_WRT_G1OptVsGT[0][0], len(g_estimated))

'''Gu_opt - the solution of optimization problem (r_estimated from g_estimated) in measured time scale'''
Gu_opt_WRT_G1OptVsGT = bfutils.undersample(G1_opt_WRT_G1OptVsGT, min_answer_WRT_G1OptVsGT[0][1][0])
'''network_GT_U - the GT  in measured time scale'''
network_GT_U_WRT_G1OptVsGT = bfutils.undersample(GT, min_answer_WRT_G1OptVsGT[0][1][0])

Gu_opt_errors_network_GT_U_WRT_G1OptVsGT = gk.OCE(Gu_opt_WRT_G1OptVsGT, network_GT_U_WRT_G1OptVsGT, undirected=False, normalized=error_normalization)['total']
Gu_opt_errors_g_estimated_WRT_G1OptVsGT = gk.OCE(Gu_opt_WRT_G1OptVsGT, g_estimated, undirected=False, normalized=error_normalization)['total']
G1_opt_error_GT_WRT_G1OptVsGT = gk.OCE(G1_opt_WRT_G1OptVsGT, GT, undirected=False, normalized=error_normalization)['total']
print('*******************************************')
print('results of minimizing with respect to G1_opt Vs. GT')
print('U rate found to be:' + str(min_answer_WRT_G1OptVsGT[0][1][0]))
print('Gu_opt_errors_network_GT_U = ', round_tuple_elements(Gu_opt_errors_network_GT_U_WRT_G1OptVsGT))
print('Gu_opt_errors_g_estimated', round_tuple_elements(Gu_opt_errors_g_estimated_WRT_G1OptVsGT))
print('G1_opt_error_GT', round_tuple_elements(G1_opt_error_GT_WRT_G1OptVsGT))
