from graphing import *
from search import search, get_se_nodes

import os.path
from ctypes import *
import random
import itertools
from scipy.special import comb

import csv
from matplotlib import collections as mc
from matplotlib import patches as mpatch

import math
import argparse
import networkx as nx
from network import *

extension = '.pickle'

parser = argparse.ArgumentParser(description='find an order for repairing bridges')
parser.add_argument('-n', '--net_name', type=str, help='network name')
parser.add_argument('-b', '--num_broken', type=int, help='number of broken bridges')
parser.add_argument('-a', '--approx', type=int, help=('approximation methods enabled \
                    - LAFO/LASR, 1: display only min of LAFO/LASR, 2: display both, \
                    3: also display altLASR'), default=0)
parser.add_argument('--arc', help='build arc flow MILP using approx values and solve using Gurobi',
                    type=bool, default=False)
parser.add_argument('-r', '--reps', type=int, help='number of scenarios with the given \
                    parameters', default=5)
parser.add_argument('-t', '--tables', type=bool, help='table output mode',default=False)
parser.add_argument('-g', '--graphing', type=bool, help='save results graphs and \
                    solution quality vs runtime graph', default=False)
parser.add_argument('-l', '--loc', type=int, help='location based sampling', default=3)
parser.add_argument('-y', '--onlybeforeafter', type=bool, help='to get before and \
                    after tstt', default=False)
parser.add_argument('-z', '--output_sequences', type=bool,
                    help='to get sequences and params to csv from net/#', default=False)
parser.add_argument('-f', '--full', type=bool, help='to use full algorithm not beam \
                    search', default=False)
parser.add_argument('-s', '--beamsearch', type=bool, help='solve using beam search',
                    default=False)
parser.add_argument('-j', '--random', help='generate scenarios randomly, to disable, \
                    enter -j without arguments', action='store_false')
parser.add_argument('-c', '--gamma', type=int, help='hyperparameter to expand search',
                    default=128)
parser.add_argument('-v', '--beta', type=int, help='hyperparameter that comtrols \
                    frequency of purge', default=128)
parser.add_argument('-d', '--num_crews', nargs='+', type=int, help='number of work \
                    crews available', default=1)
""" When multiple values are given for num_crews, order is found based on first number,
and postprocessing is performed to find OBJ for other crew numbers """
parser.add_argument('--mdecomp', nargs='+', type=int, default=0,
                    help='find multicrew sequences using decomp, min makespan (LPT)')
parser.add_argument('--idecomp', nargs='+', type=int, default=0,
                    help='find multicrew sequences using decomp, icassign')
""" within-crew methods: 1=global optimal, 2=local optimal, 3=greedy, 4=IF, 5=SPT """
parser.add_argument('--bf', nargs='+', type=int, help='1: brute force (opt), 2: brute \
                    force using ML values', default=0)
parser.add_argument('--sp', nargs='+', type=int, help='1: shortest path (opt), 2: \
                    sp using ML values, 3: sp using free flow times', default=0)
parser.add_argument('--ff', type=bool, help='calc lower bound using free flow times',
                    default=False)
parser.add_argument('--mip', nargs='+', type=int, help=('1: opt w/ precalced TSTTs, \
                    2: ML TSTT values, 3: estimated deltaTSTT[t,b] values'), default=0)
                    # mip uses Girobi solver, need license to use, coded for single-crew
parser.add_argument('--sa', type=int, help='solve using simulated annealing starting \
                    at bfs, int is number of times to solve using sa', default=0)
parser.add_argument('--sacompare', nargs='+', type=int, default=0,
    help='solve using simulated annealing method; 0=None, xyz=see additional options')
""" Additional testing options for sa (3-digit number): single-crew {x=k-neighborhood,
y=0 for L-M or y=1 for var-geom, z=0 without fail_dict z=1 for fail_dict using min in
neighborhood z=2 for fail_dict with random out direction}, multi-crew {x=1 for single-
sequence structure x=2 for multi-sequence structure, y=0 to alternate within and cross-
crew swaps y=1 to swap cross-crew only every N iters y=2 to swap cross-crew when within
crew fails (y=0 for single-sequence structure), z=0 to seed with LAFO/LASR/IF (if
available) z=1 to seed with SQG/IF}. Number of times to run each method is determined by
'--sa' parameter"""
parser.add_argument('--damaged', type=str, help='set damaged_dict to previously stored \
                    values', default='')
parser.add_argument('--mc', type=bool, help='display separate TSTTs for each class of \
                    demand', default=False)
parser.add_argument('-w', '--mc_weights', nargs='+', type=int,
                    help='TSTT weights for each class of demand', default=1)
parser.add_argument('--demand', type=float, help='demand multiplier', default=1)
args = parser.parse_args()

SEED = 42
FOLDER = 'TransportationNetworks'
MAX_DAYS = 180
MIN_DAYS = 21
memory = {}


class BestSoln():
    """A node class for bi-directional search for pathfinding"""
    def __init__(self):
        self.cost = None
        self.path = None


def eval_working_sequence(
        net, order_list, after_eq_tstt, before_eq_tstt, is_approx=False, num_crews=1,
        approx_params=None):
    """evaluates total tstt for a repair sequence, using memory to minimize runtime"""
    tap_solved = 0
    days_list = []
    tstt_list = []
    times = [0]*3 # eval combinations, prep for tap, eval taps
    global memory
    global ML_mem
    try:
        if net.free_flow:
            free_flow = True
            global FF_mem
        else:
            free_flow = False
    except:
        free_flow = False

    if isinstance(order_list[0],str):
        to_visit = order_list
    else:
        to_visit = list(reduce(op.concat,order_list))
    added = []

    # Crew order list is the order in which projects complete
    if isinstance(order_list[0], str):
        crew_order_list, which_crew, days_list = gen_crew_order(
            order_list, damaged_dict=net.damaged_dict, num_crews=num_crews)
    else:
        crew_order_list, which_crew, days_list = gen_decomp_crew_order(
            order_list, damaged_dict=net.damaged_dict, num_crews=num_crews)

    for link_id in crew_order_list:
        added.append(link_id)
        not_fixed = set(to_visit).difference(set(added))
        net.not_fixed = set(not_fixed)
        prep_time, tap_time = 0, 0

        if is_approx:
            if frozenset(net.not_fixed) in ML_mem.keys():
                tstt_after = ML_mem[frozenset(net.not_fixed)]
            else:
                prep_st = time.time()
                damaged_links = list(net.damaged_dict.keys())
                state = list(set(damaged_links).difference(net.not_fixed))
                state = [damaged_links.index(i) for i in state]
                pattern = np.zeros(len(damaged_links))
                pattern[(state)] = 1
                prep_time = time.time() - prep_st
                tap_st = time.time()
                tstt_after = approx_params[0].predict(pattern.reshape(1, -1),
                    verbose=0) * approx_params[2] + approx_params[1]
                tap_time = time.time() - tap_st
                tstt_after = tstt_after[0][0]
                ML_mem[frozenset(net.not_fixed)] = tstt_after
        elif free_flow:
            if frozenset(net.not_fixed) in FF_mem.keys():
                tstt_after = FF_mem[frozenset(net.not_fixed)]
            else:
                tstt_after, prep_time, tap_time = solve_UE(net=net, eval_seq=True)
                FF_mem[frozenset(net.not_fixed)] = tstt_after
                tap_solved += 1
        else:
            if frozenset(net.not_fixed) in memory.keys():
                tstt_after = memory[frozenset(net.not_fixed)]
            else:
                tstt_after, prep_time, tap_time = solve_UE(net=net, eval_seq=True)
                memory[frozenset(net.not_fixed)] = tstt_after
                tap_solved += 1

        times[1] += prep_time
        times[2] += tap_time
        tstt_list.append(tstt_after)

    tot_area = 0
    for i in range(len(days_list)):
        if i == 0:
            tstt = after_eq_tstt
        else:
            tstt = tstt_list[i - 1]
        tot_area += (tstt - before_eq_tstt) * days_list[i]

    return tot_area, tap_solved, tstt_list, times


def last_benefit(before, links_to_remove, before_eq_tstt, relax=False, bsearch=False,
                 ext_name=''):
    """builds last benefits dict by finding benefit of repairing each link last"""
    if relax:
        ext_name = '_relax'
    elif bsearch:
        ext_name = ext_name
    fname = before.save_dir + '/last_benefit_dict' + ext_name

    # For each bridge, find the effect on TSTT when that bridge is repaired last
    if not os.path.exists(fname + extension):
        last_b = {}
        not_fixed = []
        for link in links_to_remove:
            test_net = deepcopy(before)
            not_fixed = [link]
            test_net.not_fixed = set(not_fixed)

            tstt, __, __ = solve_UE(net=test_net, eval_seq=True)
            global memory
            memory[frozenset(test_net.not_fixed)] = tstt
            last_b[link] = tstt - before_eq_tstt
        save(fname, last_b)
    else:
        last_b = load(fname)

    return last_b


def first_benefit(after, links_to_remove, after_eq_tstt, relax=False, bsearch=False,
                  ext_name=''):
    """builds first benefits dict by finding benefit of repairing each link first"""
    if relax:
        ext_name = '_relax'
    elif bsearch:
        ext_name = ext_name
    fname = after.save_dir + '/first_benefit_dict' + ext_name

    # For each bridge, find the effect on TSTT when that bridge is repaired first
    if not os.path.exists(fname + extension):
        start = time.time()
        first_b = {}
        to_visit = links_to_remove
        added = []
        for link in links_to_remove:
            test_net = deepcopy(after)
            added = [link]
            not_fixed = set(to_visit).difference(set(added))
            test_net.not_fixed = set(not_fixed)

            tstt_after, __, __ = solve_UE(net=test_net, eval_seq=True)
            global memory
            memory[frozenset(test_net.not_fixed)] = tstt_after
            first_b[link] = after_eq_tstt - tstt_after
        elapsed = time.time() - start
        save(fname, first_b)
    else:
        first_b = load(fname)

    return first_b, elapsed


def state_after(damaged_links, save_dir, relax=False, real=False, bsearch=False,
                ext_name=''):
    """creates network and solves for tstt after the damage and before any repairs"""
    if relax:
        ext_name = '_relax'
    elif real:
        ext_name = '_real'
    elif bsearch:
        ext_name = ext_name

    fname = save_dir + '/net_after' + ext_name
    if not os.path.exists(fname + extension):
        start = time.time()
        net_after = create_network(NETFILE, TRIPFILE, mc_weights=mc_weights,
                demand_mult=demand_mult)
        net_after.not_fixed = set(damaged_links)
        net_after.art_links = art_link_dict
        net_after.damaged_dict = damaged_dict
        net_after.save_dir = save_dir
        net_after.maxruntime=str(10)

        after_eq_tstt, __, __ = solve_UE(net=net_after, eval_seq=True, warm_start=False,
                initial=True)
        if isinstance(mc_weights, list):
            test_net = deepcopy(net_after)
            test_net.mc_weights = 1
            after_eq_tstt_mcunw, __, __ = solve_UE(net=test_net, eval_seq=True,
                    warm_start=False, multiclass=True)

        global memory
        memory[frozenset(net_after.not_fixed)] = after_eq_tstt
        elapsed = time.time() - start
        save(fname, net_after)
        save(fname + '_tstt', after_eq_tstt)
        if isinstance(mc_weights, list):
            save(fname + '_tstt_mcunw', after_eq_tstt_mcunw)

        temp = 1
        if isinstance(net_after.tripfile, list):
            temp = len(net_after.tripfile)
        for i in range(temp):
            shutil.copy('batch'+str(i)+'.bin', 'after-batch'+str(i)+'.bin')
            shutil.copy('matrix'+str(i)+'.bin', 'after-matrix'+str(i)+'.bin')
        return net_after, after_eq_tstt, elapsed

    else:
        net_after = load(fname)
        after_eq_tstt = load(fname + '_tstt')

    temp = 1
    if isinstance(net_after.tripfile, list):
        temp = len(net_after.tripfile)
    for i in range(temp):
        shutil.copy('batch'+str(i)+'.bin', 'after-batch'+str(i)+'.bin')
        shutil.copy('matrix'+str(i)+'.bin', 'after-matrix'+str(i)+'.bin')
    return net_after, after_eq_tstt


def state_before(
        damaged_links, save_dir, relax=False, real=False, bsearch=False, ext_name=''):
    """creates network and solves for tstt before damage occured
    (equivalently after all repairs are complete)"""
    if relax:
        ext_name = '_relax'
    elif real:
        ext_name = '_real'
    elif bsearch:
        ext_name = ext_name

    fname = save_dir + '/net_before' + ext_name
    if not os.path.exists(fname + extension):
        start = time.time()
        net_before = create_network(NETFILE, TRIPFILE, mc_weights=mc_weights,
                demand_mult=demand_mult)
        net_before.not_fixed = set([])
        net_before.art_links = art_link_dict
        net_before.damaged_dict = damaged_dict
        net_before.save_dir = save_dir
        net_before.maxruntime=str(10)

        if isinstance(mc_weights, list):
            test_net = deepcopy(net_before)
            test_net.mc_weights = 1
            before_eq_tstt_mcunw, __, __ = solve_UE(net=test_net, eval_seq=True,
                    warm_start=False, multiclass=True)
        before_eq_tstt, __, __ = solve_UE(net=net_before, eval_seq=True,
                warm_start=False, flows=True, initial=True)

        global memory
        memory[frozenset(net_before.not_fixed)] = before_eq_tstt
        elapsed = time.time() - start
        save(fname, net_before)
        save(fname + '_tstt', before_eq_tstt)
        if isinstance(mc_weights, list):
            save(fname + '_tstt_mcunw', before_eq_tstt_mcunw)

        temp = 1
        if isinstance(net_before.tripfile, list):
            temp = len(net_before.tripfile)
        for i in range(temp):
            shutil.copy('batch'+str(i)+'.bin', 'before-batch'+str(i)+'.bin')
            shutil.copy('matrix'+str(i)+'.bin', 'before-matrix'+str(i)+'.bin')
        return net_before, before_eq_tstt, elapsed

    else:
        net_before = load(fname)
        before_eq_tstt = load(fname + '_tstt')

    temp = 1
    if isinstance(net_before.tripfile, list):
        temp = len(net_before.tripfile)
    for i in range(temp):
        shutil.copy('batch'+str(i)+'.bin', 'before-batch'+str(i)+'.bin')
        shutil.copy('matrix'+str(i)+'.bin', 'before-matrix'+str(i)+'.bin')
    return net_before, before_eq_tstt


def safety(last_b, first_b):
    swapped_links = {}
    bb = {}
    wb = {}
    for a_link in first_b:
        if first_b[a_link] < last_b[a_link]:
            bb[a_link] = last_b[a_link]
            wb[a_link] = first_b[a_link]
            swapped_links[a_link] = bb[a_link] - wb[a_link]
        else:
            wb[a_link] = last_b[a_link]
            bb[a_link] = first_b[a_link]
    return wb, bb, swapped_links


def eval_state(state, after, damaged_links, eval_seq=False):
    """finds tstt for a repair state (checks memory first for cached solution)"""
    times = [0]*3
    test_net = deepcopy(after)
    added = []
    num_tap = 0
    global memory

    for link in state:
        added.append(link)
    not_fixed = set(damaged_links).difference(set(added))
    test_net.not_fixed = set(not_fixed)

    if frozenset(test_net.not_fixed) in memory.keys():
        tstt_after = memory[frozenset(test_net.not_fixed)]
    else:
        tstt_after, prep_time, tap_time = solve_UE(net=test_net, eval_seq=eval_seq)
        memory[frozenset(test_net.not_fixed)] = tstt_after
        num_tap += 1
        times[1] += prep_time
        times[2] += tap_time

    return tstt_after, num_tap, times


def find_approx(approx, damaged_links, net_after, last_b, first_b):
    """uses sampling as described in Rey et al. 2019 to find approximated
    average first-order effects. If approx==3, uses exact first and last
    effects, and samples for middle values only"""
    X_train = []
    Y_train = []
    Z_train = [0]*len(damaged_links)
    for i in range(len(damaged_links)):
        Z_train[i] = []
    if approx==3:
        alt_Z_train = deepcopy(Z_train)

    preprocessing_num_tap = 0
    damaged_links = [i for i in damaged_links]

    for k, v in memory.items():
        pattern = np.ones(len(damaged_links))
        state = [damaged_links.index(i) for i in k]
        pattern[(state)] = 0
        X_train.append(pattern)
        Y_train.append(v)
        preprocessing_num_tap += 1

    ns = 1
    card_P = len(damaged_links)
    denom = 2**card_P

    # capture benefit of repairing each link first and last
    if approx==3:
        alt_Z_bar = np.zeros((3,card_P))
        for i in range(card_P):
            alt_Z_bar[0,i] = first_b[damaged_links[i]]
            alt_Z_bar[2,i] = last_b[damaged_links[i]]
    else:
        alt_Z_bar = None

    # A value of 1 means the link has already been repaired in that state
    mid = True
    for i in range(1,card_P):
        nom = ns * comb(card_P, i)
        num_to_sample = math.ceil(nom / denom)

        for j in range(num_to_sample):
            mid = True
            pattern = np.zeros(card_P)
            temp_state = random.sample(damaged_links, i)
            state = [damaged_links.index(i) for i in temp_state]
            pattern[(state)] = 1

            if any((pattern is test) or (pattern == test).all() for test in X_train):
                try:
                    TSTT = Y_train[X_train.index(pattern)]
                except:
                    continue
            else:
                TSTT, tap, __ = eval_state(temp_state, net_after, damaged_links,
                                           eval_seq=True)
                preprocessing_num_tap += tap
                X_train.append(pattern)
                Y_train.append(TSTT)

            for el in range(len(pattern)):
                new_pattern = np.zeros(card_P)
                new_state = deepcopy(temp_state)
                if pattern[el] == 1:
                    new_state.remove(damaged_links[el])
                    if sum(pattern) == 1 or sum(pattern) == card_P:
                        mid = False
                else:
                    new_state.append(damaged_links[el])
                    if sum(pattern) == 0 or sum(pattern) == card_P - 1:
                        mid = False
                state = [damaged_links.index(i) for i in new_state]
                new_pattern[(state)] = 1

                if any((new_pattern is test) or (new_pattern == test).all() for test in
                        X_train):
                    try:
                        new_TSTT = Y_train[X_train.index(new_pattern)]
                    except:
                        continue
                else:
                    new_TSTT, tap, __ = eval_state(new_state, net_after, damaged_links,
                        eval_seq=True)
                    preprocessing_num_tap += tap
                    X_train.append(new_pattern)
                    Y_train.append(new_TSTT)
                Z_train[el].append(abs(new_TSTT - TSTT))
                if mid and approx==3:
                    alt_Z_train[el].append(abs(new_TSTT - TSTT))

    Z_bar = np.zeros(card_P)
    for i in range(card_P):
        Z_bar[i] = np.mean(Z_train[i])
    print('Z_bar values are: ',Z_bar)

    if approx==3:
        for i in range(card_P):
            alt_Z_bar[1,i] = np.mean(alt_Z_train[i])
        print('alt Z_bar values are: ',alt_Z_bar)

    return Z_bar, alt_Z_bar, preprocessing_num_tap


def find_deltaTSTT(damaged_links, net_after, after_eq_tstt, before_eq_tstt, last_b,
                   first_b):
    """uses sampling adapted from that described in Rey et al. 2019 to find
    approximated average first-order effects for EACH position in the repair
    sequence"""
    X_train = []
    Y_train = []
    Z_train = [0]*len(damaged_links)
    for i in range(len(damaged_links)):
        Z_train[i] = []
    alt_Z_train = [0]*(len(damaged_links)-2)
    for i in range(len(damaged_links)-2):
        alt_Z_train[i] = deepcopy(Z_train)

    preprocessing_num_tap = 0
    damaged_links = [i for i in damaged_links]

    ns = 1
    card_P = len(damaged_links)
    denom = 2**card_P

    # presolve for benefit of repairing each link first and last
    alt_Z_bar = np.zeros((card_P,card_P))
    for i in range(card_P):
        alt_Z_bar[0,i] = first_b[damaged_links[i]]
        alt_Z_bar[card_P-1,i] = last_b[damaged_links[i]]

    # A value of 1 means the link has already been repaired in that state
    for k in range(1,card_P):
        nom = ns * comb(card_P, k)
        num_to_sample = math.ceil(nom / denom)

        for j in range(num_to_sample):
            pattern = np.zeros(card_P)
            temp_state = random.sample(damaged_links, k)
            state = [damaged_links.index(i) for i in temp_state]
            pattern[(state)] = 1

            count = 0
            while any((pattern is test) or (pattern == test).all() for test in X_train):
                pattern = np.zeros(card_P)
                temp_state = random.sample(damaged_links, k)
                state = [damaged_links.index(i) for i in temp_state]
                pattern[(state)] = 1
                count += 1
                if count >= card_P:
                    break

            TSTT, tap, __ = eval_state(temp_state, net_after, damaged_links,
                                       eval_seq=True)
            preprocessing_num_tap += tap
            X_train.append(pattern)
            Y_train.append(TSTT)

            for el in range(len(pattern)):
                new_pattern = np.zeros(card_P)
                new_state = deepcopy(temp_state)
                if pattern[el] == 1: # break link at index el
                    new_state.remove(damaged_links[el])
                    stage = k-1
                else: # fix link at index el
                    new_state.append(damaged_links[el])
                    stage = k
                state = [damaged_links.index(i) for i in new_state]
                new_pattern[(state)] = 1

                if stage > 0 and stage < card_P - 1:
                    if any((new_pattern is test) or (new_pattern == test).all() for test
                            in X_train):
                        try:
                            new_TSTT = Y_train[X_train.index(new_pattern)]
                            Z_train[el].append(abs(new_TSTT - TSTT))
                            alt_Z_train[stage-1][el].append(abs(new_TSTT - TSTT))
                        except:
                            pass
                    else:
                        new_TSTT, tap, __ = eval_state(new_state, net_after,
                                damaged_links, eval_seq=True)
                        preprocessing_num_tap += tap
                        X_train.append(new_pattern)
                        Y_train.append(new_TSTT)
                        Z_train[el].append(abs(new_TSTT - TSTT))
                        alt_Z_train[stage-1][el].append(abs(new_TSTT - TSTT))

    for k in range(1,card_P-1): # pos
        for el in range(card_P): # link
            while len(alt_Z_train[k-1][el]) <= 1:
                pattern = np.zeros(card_P)
                while True:
                    temp_state = random.sample(damaged_links, k)
                    if damaged_links[el] not in temp_state:
                        break
                state = [damaged_links.index(i) for i in temp_state]
                pattern[(state)] = 1

                if any((pattern is test) or (pattern==test).all() for test in X_train):
                    try:
                        TSTT = Y_train[X_train.index(pattern)]
                    except:
                        TSTT, tap, __ = eval_state(temp_state, net_after, damaged_links,
                                eval_seq=True)
                        preprocessing_num_tap += tap
                        X_train.append(pattern)
                        Y_train.append(TSTT)
                else:
                    TSTT, tap, __ = eval_state(temp_state, net_after, damaged_links,
                                               eval_seq=True)
                    preprocessing_num_tap += tap
                    X_train.append(pattern)
                    Y_train.append(TSTT)

                new_pattern = np.zeros(card_P)
                new_state = deepcopy(temp_state)
                new_state.append(damaged_links[el])
                state = [damaged_links.index(i) for i in new_state]
                new_pattern[(state)] = 1
                if any((new_pattern is test) or (new_pattern == test).all() for test
                        in X_train):
                    try:
                        new_TSTT = Y_train[X_train.index(new_pattern)]
                    except:
                        new_TSTT, tap, __ = eval_state(new_state, net_after,
                                damaged_links, eval_seq=True)
                        preprocessing_num_tap += tap
                        X_train.append(new_pattern)
                        Y_train.append(new_TSTT)
                else:
                    new_TSTT, tap, __ = eval_state(new_state, net_after, damaged_links,
                            eval_seq=True)
                    preprocessing_num_tap += tap
                    X_train.append(new_pattern)
                    Y_train.append(new_TSTT)
                Z_train[el].append(abs(new_TSTT - TSTT))
                alt_Z_train[k-1][el].append(abs(new_TSTT - TSTT))

    for j in range(card_P-2):
        for i in range(card_P):
            alt_Z_bar[j+1,i] = np.mean(alt_Z_train[j][i])
    print('alt Z_bar values are: ',alt_Z_bar)

    return alt_Z_bar, preprocessing_num_tap


def ML_preprocess(damaged_links, net_after):
    """trains a TensorFlow model to predict tstt from the binary repair state"""
    X_train = []
    Y_train = []
    Z_train = [0]*len(damaged_links)
    for i in range(len(damaged_links)):
        Z_train[i] = []

    preprocessing_num_tap = 0
    damaged_links = [i for i in damaged_links]

    for k, v in memory.items():
        pattern = np.ones(len(damaged_links))
        state = [damaged_links.index(i) for i in k]
        pattern[(state)] = 0
        X_train.append(pattern)
        Y_train.append(v)
        preprocessing_num_tap += 1
    print(str(preprocessing_num_tap)+' patterns initially stored')

    ns = 1
    card_P = len(damaged_links)
    denom = 2 ** card_P

    # A value of 1 means the link has already been repaired in that state
    for i in range(card_P):
        nom = ns * comb(card_P, i)
        num_to_sample = math.ceil(nom / denom)

        for j in range(num_to_sample):
            pattern = np.zeros(len(damaged_links))
            temp_state = random.sample(damaged_links, i)
            state = [damaged_links.index(i) for i in temp_state]
            pattern[(state)] = 1

            if any((pattern is test) or (pattern == test).all() for test in X_train):
                pass
            else:
                TSTT, tap, __ = eval_state(temp_state, net_after, damaged_links,
                                           eval_seq=True)
                preprocessing_num_tap += tap
                X_train.append(pattern)
                Y_train.append(TSTT)

                for el in range(len(pattern)):
                    new_pattern = np.zeros(len(damaged_links))
                    new_state = deepcopy(temp_state)
                    if pattern[el] == 1:
                        new_state.remove(damaged_links[el])
                    else:
                        new_state.append(damaged_links[el])
                    state = [damaged_links.index(i) for i in new_state]
                    new_pattern[(state)] = 1

                    if any((new_pattern is test) or (new_pattern == test).all() for test
                           in X_train):
                        pass
                    else:
                        new_TSTT, tap, __ = eval_state(new_state, net_after,
                                damaged_links, eval_seq=True)
                        preprocessing_num_tap += tap
                        X_train.append(new_pattern)
                        Y_train.append(new_TSTT)
                        Z_train[el].append(abs(new_TSTT - TSTT))

    Z_bar = np.zeros(len(damaged_links))
    for i in range(len(damaged_links)):
        Z_bar[i] = np.mean(Z_train[i])

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from tensorflow import keras

    c = list(zip(X_train, Y_train))
    random.shuffle(c)
    X_train, y_train = zip(*c)

    X_train_full = np.array(X_train)
    Y_train_full = np.array(Y_train)

    meany = np.mean(Y_train_full)
    stdy = np.std(Y_train_full)
    Y_train_full = (Y_train_full - meany) / stdy

    cutt = int(X_train_full.shape[0] * 0.1)
    X_train = X_train_full[cutt:]
    Y_train = Y_train_full[cutt:]
    X_valid = X_train_full[:cutt]
    Y_valid = Y_train_full[:cutt]

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
        100, activation='relu', input_shape=X_train.shape[1:]))
    model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(
        learning_rate=0.001))
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=30)


    history = model.fit(X_train, Y_train, validation_data=(
        X_valid, Y_valid), epochs=1000, verbose=0, callbacks=[early_stopping_cb])

    # Tests
    state = random.sample(damaged_links, 1)
    TSTT, tap, __ = eval_state(state, net_after, damaged_links)
    state = [damaged_links.index(i) for i in state]
    pattern = np.zeros(len(damaged_links))
    pattern[(state)] = 1
    predicted_TSTT = model.predict(pattern.reshape(1, -1)) * stdy + meany
    print('Test 1: predicted tstt vs real tstt, percent error:', predicted_TSTT[0][0],
          TSTT, (predicted_TSTT[0][0]-TSTT)/TSTT*100)

    state = random.sample(damaged_links, 4)
    TSTT, tap, __ = eval_state(state, net_after, damaged_links)
    state = [damaged_links.index(i) for i in state]
    pattern = np.zeros(len(damaged_links))
    pattern[(state)] = 1
    predicted_TSTT = model.predict(pattern.reshape(1, -1)) * stdy + meany
    print('Test 2: predicted tstt vs real tstt, percent error:', predicted_TSTT[0][0],
          TSTT, (predicted_TSTT[0][0]-TSTT)/TSTT*100)

    return model, meany, stdy, Z_bar, preprocessing_num_tap


def sim_anneal(method, sanum, bfs, net_after, after_eq_tstt, before_eq_tstt,
               num_crews=1):
    """starts at bfs (greedy or IF) and conducts simulated annealing to get solution"""
    start = time.time()
    fname = net_after.save_dir + '/sim_anneal_solution_' + str(method) +'_'+ str(sanum)

    if not os.path.exists(fname + extension):
        print('Finding the simulated annealing solution using method '+str(method)
              +', run '+str(sanum+1))
        tap_solved = 0
        if num_crews!=1 and str(method)[2]=='0':
            try:
                current = list(deepcopy(bfs_LA.path))
                best_soln = list(deepcopy(bfs_LA.path))
                curcost = deepcopy(bfs_LA.cost)
                best_cost = deepcopy(bfs_LA.cost)
                print('Starting soln cost from LAFO/LASR/IF:', curcost)
            except:
                current = list(deepcopy(bfs.path))
                best_soln = list(deepcopy(bfs.path))
                curcost = deepcopy(bfs.cost)
                best_cost = deepcopy(bfs.cost)
                print('Starting soln cost from SQG/IF:', curcost)
        else:
            current = list(deepcopy(bfs.path))
            best_soln = list(deepcopy(bfs.path))
            curcost = deepcopy(bfs.cost)
            best_cost = deepcopy(bfs.cost)
            print('Starting soln cost from SQG/IF:', curcost)
        curnet = deepcopy(net_after)
        damaged_links = list(curnet.damaged_dict.keys())
        if graphing and sanum == 0:
            global sa_time_list
            global sa_OBJ_list
            sa_time_list[method].append(0)
            sa_OBJ_list[method].append(deepcopy(best_cost))

        global memory
        t = 0
        fail = 0
        ratio = 0
        lastMvmt = 0

        if num_crews==1:
            maxiters = int(1.2 * len(current)**3)
            if method == 100:
                neighborhood = len(current) - 1
            else:
                T = 0
                T0 = 0.044814
                match str(method)[0]:
                    case '3':
                        neighborhood = len(current)*5-11
                        print('Using 3-neighborhood')
                    case '2':
                        neighborhood = len(current)*3-5
                        print('Using 2-neighborhood')
                    case '1':
                        neighborhood = len(current) - 1
                    case _:
                        neighborhood = len(current) - 1
                        print('k-neighborhood input: {} not understood, using k=1',
                              str(method)[0])
                match str(method)[1]:
                    case '1':
                        alpha = (-0.01 / (T0 * np.log(0.1)))**(1/(len(current)-1))
                        print('Calculated alpha for geometric cooling scheme is : ',
                              alpha)
                    case '0':
                        pass
                    case _:
                        print('Cooling scheme input: {} not understood, use Lundy-Mees',
                              str(method)[1])
                match str(method)[2]:
                    case '1' | '2':
                        fail_dict = dict()
                        force = False
                        print('Using failure dictionary with sub-method ',
                              str(method)[2])
                    case '0':
                        pass
                    case _:
                        print('Failure dict input: {} not understood, do not use',
                              str(method)[2])

            while t < maxiters:
                t += 1
                idx_temp = random.randrange(0,neighborhood)
                if str(method)[2]=='1':
                    # fail dict which selects min solution in k-neigh after rejecting
                    # all in k-neigh
                    if len(fail_dict) == neighborhood:
                        idx_temp = min(fail_dict, key=lambda x: fail_dict[x])
                        force = True
                        fail_dict = {}
                    while idx_temp in fail_dict:
                        idx_temp = random.randrange(0,neighborhood)
                elif str(method)[2]=='2':
                    # fail dict which selects random solution after rejecting all in
                    # k-neigh
                    if len(fail_dict) == neighborhood:
                        idx_temp = random.randrange(0,neighborhood)
                        force = True
                        fail_dict = {}
                    while idx_temp in fail_dict:
                        idx_temp = random.randrange(0,neighborhood)
                
                if idx_temp < len(current) - 1:
                    curhood = 1
                    idx = idx_temp
                elif idx_temp < len(current)*3-5:
                    curhood = 2
                else:
                    curhood = 3

                if curhood == 3:
                    nextord = deepcopy(current)
                    if idx_temp < len(current)*4-8:
                        idx = idx_temp - len(current)*3 + 5
                        el = nextord.pop(idx)
                        nextord.insert(idx+3,el)
                    else:
                        idx = idx_temp - len(current)*4 + 11
                        el = nextord.pop(idx)
                        nextord.insert(idx-3,el)
                    nextcost, tap, __, __ = eval_working_sequence(
                        curnet, nextord, after_eq_tstt, before_eq_tstt)
                    tap_solved += tap
                elif curhood == 2:
                    nextord = deepcopy(current)
                    if idx_temp < len(current)*2-3:
                        idx = idx_temp - len(current) + 1
                        el = nextord.pop(idx)
                        nextord.insert(idx+2,el)
                    else:
                        idx = idx_temp - len(current)*2 + 5
                        el = nextord.pop(idx)
                        nextord.insert(idx-2,el)
                    nextcost, tap, __, __ = eval_working_sequence(
                        curnet, nextord, after_eq_tstt, before_eq_tstt)
                    tap_solved += tap

                else:
                    nextord = deepcopy(current)
                    el = nextord.pop(idx)
                    swap = nextord[idx]
                    nextord.insert(idx+1,el)

                    # Get tstt before fixing el or swap, then tstt if fixing each first
                    # to find difference in total area
                    startstate = current[:idx]
                    startTSTT, tap, __ = eval_state(startstate, curnet, damaged_links,
                            eval_seq=True)
                    tap_solved += tap

                    endstate = current[:idx+2]
                    endTSTT, tap, __ = eval_state(endstate, curnet, damaged_links,
                                                    eval_seq=True)
                    tap_solved += tap

                    elstate = current[:idx+1]
                    elTSTT, tap, __ = eval_state(elstate, curnet, damaged_links,
                                                    eval_seq=True)
                    tap_solved += tap

                    swapstate = nextord[:idx+1]
                    swapTSTT, tap, __ = eval_state(swapstate, curnet, damaged_links,
                                                    eval_seq=True)
                    tap_solved += tap

                    nextcost = deepcopy(curcost)
                    nextcost -= startTSTT*(damaged_dict[el]-damaged_dict[swap])
                    nextcost -= elTSTT*damaged_dict[swap]
                    nextcost += swapTSTT*damaged_dict[el]

                negdelta = curcost - nextcost

                if negdelta > 0:
                    current = deepcopy(nextord)
                    curcost = deepcopy(nextcost)
                else:
                    if str(method)[1]=='0':
                        if str(method)[2]!='0' and force:
                            prob = 1
                        else:
                            try:
                                prob = math.exp(negdelta/curcost*(t**(2/3)))
                            except:
                                prob = 0
                    else:
                        if t % np.floor(max_iters/len(damaged_dict)) == 0:
                            T+=1
                        if str(method)[2]!='0' and force:
                            prob = 1
                        else:
                            try:
                                prob = math.exp(negdelta/curcost/(T0*alpha**T))
                            except:
                                prob = 0
                    if random.random() <= prob:
                        current = deepcopy(nextord)
                        curcost = deepcopy(nextcost)
                    else:
                        fail += 1
                        if str(method)[2]!='0':
                            fail_dict[idx] = nextcost

                if curcost < best_cost:
                    best_soln = deepcopy(current)
                    best_cost = deepcopy(curcost)
                    if graphing and sanum==0:
                        sa_time_list[method].append(time.time()-start)
                        sa_OBJ_list[method].append(deepcopy(best_cost))
                    lastMvmt = t
                    print('On iteration {}, new best solution cost is {} with \
                            sequence {}'.format(t, best_cost, best_soln))
                    
            ratio = fail/t
            print('Finished on iteration {} with ratio {} with best solution \
                    found on iteration {}'.format(t, ratio, lastMvmt))

        elif str(method)[0]=='1': # multi-crew, single sequence structure
            if not isinstance(current[0], str):
                current = gen_single_seq(current, damaged_dict, num_crews)
            maxiters = int(1.5 * (len(current)-num_crews+1)**3)
            while t < maxiters:
                t += 1
                idx = random.randrange(0,len(current)-1)
                nextord = deepcopy(current)
                el = nextord.pop(idx)
                if idx+1 >= num_crews:
                    nextord.insert(idx+1,el)
                else:
                    nextord.insert(num_crews,el)

                nextcost, tap, __, __ = eval_working_sequence(
                    curnet, nextord, after_eq_tstt, before_eq_tstt, num_crews=num_crews)
                tap_solved += tap

                negdelta = curcost - nextcost

                if negdelta > 0:
                    current = deepcopy(nextord)
                    curcost = deepcopy(nextcost)
                else:
                    prob = math.exp(negdelta/curcost*(t**(2/3)))
                    if random.random() <= prob:
                        current = deepcopy(nextord)
                        curcost = deepcopy(nextcost)
                    else:
                        fail += 1

                if curcost < best_cost:
                    test1, __, __, __ = eval_sequence(curnet, current, after_eq_tstt,
                            before_eq_tstt, num_crews=num_crews)
                    if best_cost < test1:
                        print('Inaccuracy in new best soln: ' + str(test1-curcost)
                              + ', test1 = ' + str(test1) + ', curcost = '
                              + str(curcost) + '. Do not use.')
                    else:
                        best_soln = deepcopy(current)
                        best_cost = deepcopy(curcost)
                        if graphing and sanum==0:
                            sa_time_list[method].append(time.time()-start)
                            sa_OBJ_list[method].append(deepcopy(best_cost))
                        lastMvmt = t
                        print('New best solution cost is ' + str(best_cost)
                              + ' with sequence ' + str(best_soln))
                        
            ratio = fail/t
            print('Finished on iteration {} with ratio {} with best solution found on \
                  iteration {}'.format(t, ratio, lastMvmt))

        else: # multi-crew, multi-sequence structure
            if isinstance(current[0],str):
                current, __ = gen_crew_seqs(current, damaged_dict, num_crews)
            maxiters = int(1.5 * (sum([len(i) for i in current])-num_crews+1)**3)
            frequency = math.ceil(num_broken/num_crews)
            cross = False
            while t < maxiters:
                t += 1
                if (str(method)[1]=='0' and t % 2) or (str(method)[1]=='1'
                                                       and t % frequency):
                    cross = True

                idx = random.randrange(0,num_broken - num_crews)
                for j in range(num_crews):
                    if idx < len(current[j]) - 1:
                        crewidx = j
                    else:
                        idx -= len(current[j]) - 1
                nextord = deepcopy(current)
                el = nextord[crewidx].pop(idx)

                if not cross:
                    nextord[crewidx].insert(idx+1,el)
                else:
                    newcrew = random.randrange(0,num_crews)
                    if newcrew == crewidx and crewidx != num_crews - 1:
                        newcrew += 1
                    elif newcrew == crewidx:
                        newcrew -= 1
                    idx2 = min(idx,len(nextord[newcrew])-1)
                    el2 = nextord[newcrew].pop(idx2)
                    nextord[newcrew].insert(idx2,el)
                    nextord[crewidx].insert(idx,el2)
                    crews = [sum([damaged_dict[link] for link in sub]) for sub in
                             nextord]
                    last = [damaged_dict[sub[-1]] for sub in nextord]
                    penult = [i - j for (i,j) in zip(crews,last)]
                    while max(penult) > min(crews):
                        el = nextord[np.argmax(penult)].pop()
                        nextord[np.argmin(crews)].append(el)
                        crews = [sum([damaged_dict[link] for link in sub]) for sub in
                                 nextord]
                        last = [damaged_dict[sub[-1]] for sub in nextord]
                        penult = [i - j for (i,j) in zip(crews,last)]

                nextcost, tap, __, __ = eval_working_sequence(
                    curnet, nextord, after_eq_tstt, before_eq_tstt, num_crews=num_crews)
                tap_solved += tap

                negdelta = curcost - nextcost

                if negdelta > 0:
                    current = deepcopy(nextord)
                    curcost = deepcopy(nextcost)
                else:
                    prob = math.exp(negdelta/curcost*(t**(2/3)))
                    if random.random() <= prob:
                        current = deepcopy(nextord)
                        curcost = deepcopy(nextcost)
                        cross = False
                    else:
                        fail += 1
                        if str(method)[1]=='2' and not cross:
                            cross = True

                if curcost < best_cost:
                    test1, __, __, __ = eval_sequence(curnet, current, after_eq_tstt,
                            before_eq_tstt, num_crews=num_crews)
                    if best_cost < test1:
                        print('Inaccuracy in new best soln: ' + str(test1-curcost)
                              + ', test1 = ' + str(test1) + ', curcost = '
                              + str(curcost) + '. Do not use.')
                    else:
                        best_soln = deepcopy(current)
                        best_cost = deepcopy(curcost)
                        if graphing and sanum==0:
                            sa_time_list[method].append(time.time()-start)
                            sa_OBJ_list[method].append(deepcopy(best_cost))
                        lastMvmt = t
                        print('New best solution cost is ' + str(best_cost)
                              + ' with sequence ' + str(best_soln))

            ratio = fail/t
            print('Finished on iteration {} with ratio {} with best solution found on \
                  iteration {}'.format(t, ratio, lastMvmt))


        elapsed = time.time() - start
        path = best_soln
        bound = best_cost
        if graphing:
            sa_time_list[method].append(elapsed)
            sa_OBJ_list[method].append(deepcopy(best_cost))

        test2, __, __, __ = eval_sequence(curnet, best_soln, after_eq_tstt,
                before_eq_tstt, num_crews=num_crews)

        if abs(test2-bound)> 5:
            print('Inaccuracy in best soln: ' + str(test2-bound) + ', test2 = '
                  + str(test2) + ', bound = ' + str(bound))
        bound = test2
        if num_crews != 1 and str(method)[2]==1:
            elapsed += LASR_res[1] + importance_res[1] + 3*evaluation_time
            tap_solved += LASR_res[2]
        else:
            elapsed += greedy_res[1] + importance_res[1] + 2*evaluation_time
            tap_solved += greedy_res[2] + num_broken - 1

        save(fname + '_obj', bound)
        save(fname + '_path', path)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', tap_solved)
    else:
        bound = load(fname + '_obj')
        path = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    print('Simulated annealing objective value: ',bound)
    if num_crews != 1:
        crew_seqs, make_span = gen_crew_seqs(path, curnet.damaged_dict, num_crews)
        return [bound, elapsed, tap_solved, make_span], crew_seqs
    else:
        return [bound, elapsed, tap_solved], path


def LAFO(net_before, after_eq_tstt, before_eq_tstt, time_before, Z_bar):
    """approx solution method based on estimating Largest Average First Order
    effects using the preprocessing function"""
    start = time.time()
    tap_solved = 0

    fname = net_before.save_dir + '/LAFO_bound'
    if not os.path.exists(fname + extension):
        LAFO_net = deepcopy(net_before)

        c = list(zip(Z_bar, damaged_links))
        sorted_c = sorted(c,reverse=True)
        __, path = zip(*sorted_c)

        elapsed = time.time() - start + time_before
        bound, eval_taps, __, __ = eval_sequence(
            LAFO_net, path, after_eq_tstt, before_eq_tstt, num_crews=num_crews)

        save(fname + '_obj', bound)
        save(fname + '_path', path)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', 0)
    else:
        bound = load(fname + '_obj')
        path = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    print('LAFO objective value: ',bound)
    if num_crews != 1:
        crew_seqs, make_span = gen_crew_seqs(path, damaged_dict, num_crews)
        return [bound, elapsed, tap_solved, make_span], crew_seqs
    else:
        return [bound, elapsed, tap_solved], path


def LASR(net_before, after_eq_tstt, before_eq_tstt, time_before, Z_bar):
    """approx solution method based on estimating Largest Average Smith Ratio
    using the preprocessing function"""
    start = time.time()
    tap_solved = 0

    fname = net_before.save_dir + '/LASR_bound'
    if not os.path.exists(fname + extension):
        LASR_net = deepcopy(net_before)

        order = np.zeros(len(damaged_links))
        for i in range(len(damaged_links)):
            order[i] = Z_bar[i]/list(damaged_dict.values())[i]

        c = list(zip(order, damaged_links))
        sorted_c = sorted(c,reverse=True)
        __, path = zip(*sorted_c)

        elapsed = time.time() - start + time_before
        bound, eval_taps, __, __ = eval_sequence(
            LASR_net, path, after_eq_tstt, before_eq_tstt, num_crews=num_crews)

        save(fname + '_obj', bound)
        save(fname + '_path', path)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', 0)
    else:
        bound = load(fname + '_obj')
        path = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    print('LASR objective value: ',bound)
    if num_crews != 1:
        crew_seqs, make_span = gen_crew_seqs(path, damaged_dict, num_crews)
        return [bound, elapsed, tap_solved, make_span], crew_seqs
    else:
        return [bound, elapsed, tap_solved], path


def altLASR(net_before, after_eq_tstt, before_eq_tstt, time_before, alt_Z_bar):
    """approx solution method based on estimating Largest Average Smith Ratio
    using the alternate preprocessing function"""
    start = time.time()
    tap_solved = 0

    fname = net_before.save_dir + '/altLASR_bound'
    if not os.path.exists(fname + extension):
        LASR_net = deepcopy(net_before)

        order = np.zeros((3,len(damaged_links)))
        for i in range(len(damaged_links)):
            order[:,i] = alt_Z_bar[:,i]/list(damaged_dict.values())[i]

        tops = np.argmax(order, axis=1)
        bottoms = np.argmin(order, axis=1)

        mod_order = order[1]
        mod_order[bottoms[2]] = order[1,bottoms[1]]/2
        mod_order[tops[0]] = order[1,tops[1]]*2

        c = list(zip(mod_order, damaged_links))
        sorted_c = sorted(c,reverse=True)
        __, path = zip(*sorted_c)

        elapsed = time.time() - start + time_before
        bound, eval_taps, __, __ = eval_sequence(
            LASR_net, path, after_eq_tstt, before_eq_tstt, num_crews=num_crews)

        save(fname + '_obj', bound)
        save(fname + '_path', path)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', 0)
    else:
        bound = load(fname + '_obj')
        path = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    print('AltLASR objective value: ',bound)
    if num_crews != 1:
        crew_seqs, make_span = gen_crew_seqs(path, damaged_dict, num_crews)
        return [bound, elapsed, tap_solved, make_span], crew_seqs
    else:
        return [bound, elapsed, tap_solved], path


def mip_delta(net_before, after_eq_tstt, before_eq_tstt, time_before, deltaTSTT):
    """approx solution method based on estimating change in TSTT due to repairing
    each link in each repair position (i.e. first through last). deltaTSTT values
    are used as OBJ function coefficients for the modified OBJ function, and the
    mip is solving using Gurobi. Single-crew use only."""
    import gurobipy as gp
    from gurobipy import GRB
    start = time.time()
    tap_solved = 0

    fname = net_before.save_dir + '/mip_delta'
    if not os.path.exists(fname + extension):
        mip_net = deepcopy(net_before)
        N = len(damaged_dict)
        delta = deltaTSTT # [pos,link]
        D = np.array(list(damaged_dict.values()))

        # create model and add vars and constraints
        m = gp.Model('mip_delta')
        y = m.addMVar(shape=(N,N),vtype=GRB.BINARY)
        m.addConstrs((y[i,:].sum()==1 for i in range(N)), 'stages')
        m.addConstrs((y[:,i].sum()==1 for i in range(N)), 'links')

        obj = (y[0,:] @ delta[0,:]) * (y[1,:].T @ D)
        for i in range(2,N):
            for j in range(i):
                obj += (y[j,:] @ delta[j,:]) * (y[i,:].T @ D)
        m.setObjective(obj, GRB.MAXIMIZE)

        # optimize and print solution
        m.optimize()
        if m.Status == GRB.OPTIMAL:
            y_sol = y.getAttr('X')
            path = []
            for i in range(N):
                path.append(list(damaged_dict.keys())[int(np.argmax(y_sol[i]))])
            print('path found using Gurobi: ' + str(path))
        else:
            print('Gurobi solver status not optimal...')

        elapsed = time.time() - start + time_before
        bound, eval_taps, __, __ = eval_sequence(
            mip_net, path, after_eq_tstt, before_eq_tstt, num_crews=num_crews)

        save(fname + '_obj', bound)
        save(fname + '_path', path)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', 0)
    else:
        bound = load(fname + '_obj')
        path = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    print('DeltaTSTT objective value: ',bound)
    if num_crews != 1:
        crew_seqs, make_span = gen_crew_seqs(path, damaged_dict, num_crews)
        return [bound, elapsed, tap_solved, make_span], crew_seqs
    else:
        return [bound, elapsed, tap_solved], path


def arc_flow(net_before, after_eq_tstt, before_eq_tstt, time_before, Z_bar, first_b):
    """approx solution method using arc flows to schedule estimating arc weights
    using the preprocessing function"""
    import gurobipy as gp
    from gurobipy import GRB
    start = time.time()
    tap_solved = 0

    fname = net_before.save_dir + '/arc_soln'
    if not os.path.exists(fname + extension):
        arc_net = deepcopy(net_before)

        # Find benefit of repairing each link first
        lg_dict = deepcopy(first_b)

        # Base data (damaged_links is already a list)
        order = np.zeros(len(damaged_links))
        weights = dict()
        for i in range(len(damaged_links)):
            order[i] = Z_bar[i]/list(damaged_dict.values())[i]
            weights[list(damaged_dict.keys())[i]] = Z_bar[i]
        c = list(zip(order, damaged_links))
        sorted_ = sorted(c,reverse=True)
        __, J = zip(*sorted_)
        p = [round(damaged_dict[j]) for j in J]
        print('J = {}, p = {}'.format(J,p))

        # Build set of valid nodes and arcs (from Kramer, 2019)
        T = math.floor(1/num_crews * sum(p) + (num_crews-1) / num_crews * max(p))
        P = [0] * (T+1)
        N = []
        A_ = [ [] for _ in range(len(damaged_links)+1)]
        P[0] = 1
        for j in range(len(damaged_dict)):
            for t in range(T-p[j],-1,-1):
                if P[t]==1:
                    P[t+p[j]] = 1
                    A_[j+1].append((t,t+p[j],J[j]))
        for t in range(0,T+1):
            if P[t]==1:
                N.append(t)
                A_[0].append((t,T,0))
        if T not in N:
            N.append(T)
        A = [el for elem in A_[1:] for el in elem]

        # Establish arc costs and node balances
        cost = {arc: weights[arc[2]] * arc[1] for arc in A if arc[0]!=0}
        temp = {arc: lg_dict[arc[2]] * arc[1] for arc in A if arc[0]==0}
        cost.update(temp)
        temp = {arc: 0 for arc in A_[0]}
        cost.update(temp)
        balance = {node: 0 for node in N}
        balance[0], balance[T] = num_crews, -num_crews

        # Create model and add vars and constraints
        m = gp.Model('arcflow')
        flow = m.addVars(A, obj=cost, vtype=GRB.BINARY, name='flow')
        flow0 = m.addVars(A_[0], obj=cost, name='flow0')
        m.addConstrs((flow.sum('*',q,'*') + flow0.sum('*',q,'*') + balance[q]
             == flow.sum(q,'*','*') + flow0.sum(q,'*','*') for q in N), 'node')
        m.addConstrs((flow.sum('*','*',j) >= 1 for j in J), 'links')

        # Optimize and print solution
        arc_set = []
        m.optimize()
        if m.Status == GRB.OPTIMAL:
            solution = m.getAttr('X', flow)
            for arc in A:
                if solution[arc] > 0 and arc[2] != 0:
                    arc_set.append(arc)
        else:
            print('Gurobi solver status not optimal...')

        print('arc set (without loss arcs): ', arc_set)
        temp = sorted(arc_set, key = lambda x: x[0])
        path = [link[2] for link in temp]

        elapsed = time.time() - start + time_before
        bound, eval_taps, __, __ = eval_sequence(
            arc_net, path, after_eq_tstt, before_eq_tstt, num_crews=num_crews)

        save(fname + '_obj', bound)
        save(fname + '_path', path)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', 0)
    else:
        bound = load(fname + '_obj')
        path = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    if num_crews != 1:
        crew_seqs, make_span = gen_crew_seqs(path, damaged_dict, num_crews)
        return [bound, elapsed, tap_solved, make_span], crew_seqs
    else:
        return [bound, elapsed, tap_solved], path


def SPT_solution(net_before, after_eq_tstt, before_eq_tstt, time_net_before):
    """simple heuristic which orders link for repair based on shortest repair time"""
    start = time.time()
    tap_solved = 0

    fname = net_before.save_dir + '/SPT_bound'
    if not os.path.exists(fname + extension):
        SPT_net = deepcopy(net_before)
        sorted_d = sorted(damaged_dict.items(), key=lambda x: x[1])
        path, __ = zip(*sorted_d)

        elapsed = time.time() - start
        bound, eval_taps, __, __ = eval_sequence(
            SPT_net, path, after_eq_tstt, before_eq_tstt, num_crews=num_crews)

        save(fname + '_obj', bound)
        save(fname + '_path', path)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', 0)
    else:
        bound = load(fname + '_obj')
        path = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    print('Shortest processing time objective value: ',bound)
    if num_crews != 1:
        crew_seqs, make_span = gen_crew_seqs(path, damaged_dict, num_crews)
        return [bound, elapsed, tap_solved, make_span], crew_seqs
    else:
        return [bound, elapsed, tap_solved], path


def importance_factor_solution(net_before, after_eq_tstt, before_eq_tstt,
                               time_net_before):
    """simple heuristic which orders links based on predamage flow"""
    start = time.time()

    fname = net_before.save_dir + '/importance_factor_bound'
    if not os.path.exists(fname + extension):
        tot_flow = 0
        if_net = deepcopy(net_before)
        for ij in if_net.linkDict:
            tot_flow += if_net.linkDict[ij]['flow']

        damaged_links = damaged_dict.keys()
        if_dict = {}
        for link_id in damaged_links:
            link_flow = if_net.linkDict[link_id]['flow']
            if_dict[link_id] = link_flow / tot_flow
        sorted_d = sorted(if_dict.items(), key=lambda x: x[1])
        path, if_importance = zip(*sorted_d)
        path = path[::-1]
        if_importance = if_importance[::-1]

        elapsed = time.time() - start + time_net_before
        bound, eval_taps, __, __ = eval_sequence(if_net, path, after_eq_tstt,
                before_eq_tstt, if_dict, importance=True, num_crews=num_crews)
        tap_solved = 1

        save(fname + '_obj', bound)
        save(fname + '_path', path)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', 1)
    else:
        bound = load(fname + '_obj')
        path = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    print('Importance factor objective value: ',bound)
    if num_crews != 1:
        crew_seqs, make_span = gen_crew_seqs(path, damaged_dict, num_crews)
        return [bound, elapsed, tap_solved, make_span], crew_seqs
    else:
        return [bound, elapsed, tap_solved], path


def brute_force(net_after, after_eq_tstt, before_eq_tstt, is_approx=False, num_crews=1,
                calc_ff=False):
    """enumerates all possible sequences to find the lowest overall tstt impacts"""
    net = deepcopy(net_after)
    start = time.time()
    if is_approx:
        tap_solved = 0
    else:
        tap_solved = len(memory)
    damaged_links = damaged_dict.keys()
    times = [0]*3 # eval combinations, prep for tap, eval taps
    approx_ext = ''
    if is_approx:
        approx_ext = '_approx'
        global approx_params
    if calc_ff:
        net.free_flow = True
        approx_ext = '_ff'
    else:
        net.free_flow = False

    fname = net.save_dir + '/min_seq' + approx_ext
    if not os.path.exists(fname + extension):
        print('Finding the optimal sequence ...')
        all_sequences = itertools.permutations(damaged_links)

        i = 0
        min_cost = 1e+80
        min_seq = None

        for sequence in all_sequences:
            if num_crews != 1:
                check = True
                for el in range(num_crews-1):
                    if sequence[el] > sequence[el+1]:
                        check = False
                if not check:
                    continue

            seq_net = deepcopy(net)
            cost, eval_taps, __, seq_times = eval_working_sequence(seq_net, sequence,
                after_eq_tstt, before_eq_tstt, is_approx=is_approx, num_crews=num_crews,
                approx_params=approx_params)
            times[1] += seq_times[1]
            times[2] += seq_times[2]
            tap_solved += eval_taps

            if cost < min_cost or min_cost == 1e+80:
                min_cost = cost
                min_seq = sequence
            i += 1

        elapsed = time.time() - start
        times[0] = elapsed - times[1] - times[2]
        bound, __, __, __ = eval_sequence(
            seq_net, min_seq, after_eq_tstt, before_eq_tstt, num_crews=num_crews)
        if calc_ff:
            print('Lower bound using free flow with {} crew(s): {}, path: {}'.format(
                  num_crews, bound, min_seq))
        elif is_approx:
            print('ML brute force objective with {} crew(s): {}, path: {}'.format(
                num_crews, bound, min_seq))
        else:
            print('Optimal objective with {} crew(s): {}, optimal path: {}'.format(
                num_crews, bound, min_seq))
        print('Run time break down: upper algo, prep for taps, run taps: ', times)

        save(fname + '_obj', bound)
        save(fname + '_path', min_seq)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', tap_solved)
    else:
        bound = load(fname + '_obj')
        min_seq = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    if num_crews != 1:
        crew_seqs, make_span = gen_crew_seqs(min_seq, damaged_dict, num_crews)
        return [bound, elapsed, tap_solved, make_span], crew_seqs, times
    else:
        return [bound, elapsed, tap_solved], min_seq, times


def opt_sp(sp, net_after, after_eq_tstt, before_eq_tstt, time_before, vecTSTT):
    """uses TSTT values for every state (calc or est) to find the lowest overall tstt
    impacts by finding shortest path from immediate post disruption state to fully
    repaired state in an induced network. Single-crew only"""
    tap_solved=0

    if sp==1:
        fname = net_after.save_dir + '/sp_OPT'
    elif sp==2:
        fname = net_after.save_dir + '/sp_ML'
    elif sp==3:
        fname = net_after.save_dir + '/sp_FF'
    
    if not os.path.exists(fname + extension):
        start = time.time()
        sp_net = Network() # init empty network using Network.py
        sp_net.init_sp_net(net_after.damaged_dict, vecTSTT, before_eq_tstt)
        backlink, cost = sp_net.altAcyclic(1, after_eq_tstt
                                           * sum(net_after.damaged_dict.values()))
        bound, min_seq = sp_net.build_sp_seq(backlink, cost, net_after.damaged_dict)

        sp_time = time.time() - start
        times = [sp_time, 0, time_before]
        elapsed = sp_time + time_before

        bound, __, __, __ = eval_sequence(net_after, min_seq, after_eq_tstt,
                                          before_eq_tstt)
        if sp==2:
            print('ML shortest path objective: {}, path: {}'.format(bound, min_seq))
        else:
            print('Optimal shortest path objective: {}, optimal path: {}'.format(bound,
                                                                                 min_seq))
        print('Run time break down: upper algo, prep for taps, run taps: ', times)

        save(fname + '_obj', bound)
        save(fname + '_path', min_seq)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', tap_solved)
    else:
        bound = load(fname + '_obj')
        min_seq = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    if num_crews != 1:
        crew_seqs, make_span = gen_crew_seqs(min_seq, damaged_dict, num_crews)
        return [bound, elapsed, tap_solved, make_span], crew_seqs, times
    else:
        return [bound, elapsed, tap_solved], min_seq, times


def mip_bf(mip, net_before, after_eq_tstt, before_eq_tstt, time_before, vecTSTT):
    """uses TSTT values for every state (calc or est) to build single level IP
    and solve using gurobi; vecTSTT is a vector of length 2^N with TSTTs
    encoded from z-vectors where vecTSTT[0] is the TSTT for z=[0,...,0] and
    vecTSTT[2^N-1] is the TSTT for z=[1,...1] with num_broken=N; mip=1 uses
    actual TSTT values to find optimal sequence, mip=2 uses ML TSTT values"""
    import gurobipy as gp
    from gurobipy import GRB
    start = time.time()
    tap_solved = 0

    if mip==1:
        fname = net_before.save_dir + '/mip_OPT'
    elif mip==2:
        fname = net_before.save_dir + '/mip_ML'
    if not os.path.exists(fname + extension):
        mip_net = deepcopy(net_before)
        N = len(damaged_dict)
        valstemp = vecTSTT - before_eq_tstt
        vals = valstemp/valstemp[0]
        Dtemp = np.array(list(damaged_dict.values()))
        D = Dtemp/sum(Dtemp)

        # create model and add vars and constraints
        m = gp.Model('mip_TSTT')
        y = m.addMVar(shape=(N,N),vtype=GRB.BINARY) # [stage,link]
        z = m.addMVar(shape=(N,N),vtype=GRB.BINARY) # [stage,link]
        ze = m.addMVar(shape=(N,2**N),vtype=GRB.BINARY) # [stage,state]
        m.addConstrs((y[t,:].sum()==1 for t in range(N)), 'stages')
        m.addConstrs((y[:,b].sum()==1 for b in range(N)), 'links')
        m.addConstrs((z[0,b]==0 for b in range(N)), 'state0')
        m.addConstrs((y[0:t,b].sum()==z[t,b] for t in range(1,N) for b in range(N)),
            'states')
        m.addConstrs(((ze[t,s]==1) >> (sum(z[t,b]*2**(N-1-b) for b in range(N))==s)
                      for t in range(N) for s in range(2**N)),'expstates')
        m.addConstrs((ze[t,:].sum()==1 for t in range(N)), 'extra')

        obj = (ze[0,:] @ vals[:]) * (y[0,:].T @ D)
        for t in range(1,N):
            obj += ze[t,:] @ vals[:] * (y[t,:].T @ D)
        m.setObjective(obj, GRB.MINIMIZE)

        # optimize and print solution
        m.optimize()
        if m.Status == GRB.OPTIMAL:
            y_sol = y.getAttr('X')
            path = []
            for i in range(N):
                print(y_sol[i])
                path.append(list(damaged_dict.keys())[int(np.argmax(y_sol[i]))])
            print('path found using Gurobi: ' + str(path))
        else:
            print('Gurobi solver status not optimal...')

        elapsed = time.time() - start + time_before
        bound, eval_taps, __, __ = eval_sequence(
            mip_net, path, after_eq_tstt, before_eq_tstt, num_crews=num_crews)

        save(fname + '_obj', bound)
        save(fname + '_path', path)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', tap_solved)
    else:
        bound = load(fname + '_obj')
        path = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    if mip==1:
        print('OPT IP objective value: ',bound)
    elif mip==2:
        print('ML IP objective value: ',bound)
    if num_crews != 1:
        crew_seqs, make_span = gen_crew_seqs(path, damaged_dict, num_crews)
        return [bound, elapsed, tap_solved, make_span], crew_seqs
    else:
        return [bound, elapsed, tap_solved], path


def lazy_greedy_heuristic(net_after, after_eq_tstt, before_eq_tstt, first_b, bb_time):
    """heuristic which orders links for repair based on tstt effect if repaired first"""
    start = time.time()
    ordered_days = []
    orderedb_benefits = []
    sorted_d = sorted(damaged_dict.items(), key=lambda x: x[1])
    for key, value in sorted_d:
        ordered_days.append(value)
        orderedb_benefits.append(first_b[key])

    ob, od, path = orderlists(orderedb_benefits, ordered_days, rem_keys=sorted_d)
    path = [i[0] for i in path]
    elapsed = time.time() - start + bb_time

    test_net = deepcopy(net_after)
    bound, eval_taps, __, __ = eval_sequence(
        test_net, path, after_eq_tstt, before_eq_tstt, num_crews=num_crews)
    tap_solved = len(damaged_dict)+1

    fname = save_dir + '/lazygreedy_solution'
    save(fname + '_obj', bound)
    save(fname + '_path', path)
    save(fname + '_elapsed', elapsed)
    save(fname + '_num_tap', tap_solved)

    print('Lazy greedy objective value: ',bound)
    if num_crews != 1:
        crew_seqs, make_span = gen_crew_seqs(path, damaged_dict, num_crews)
        return [bound, elapsed, tap_solved, make_span], crew_seqs
    else:
        return [bound, elapsed, tap_solved], path


def greedy_heuristic(net_after, after_eq_tstt, before_eq_tstt, time_net_before,
                     time_net_after):
    """heuristic which orders links for repair at each stage based on immediate effect
    on tstt"""
    start = time.time()
    tap_solved = 0
    fname = net_after.save_dir + '/greedy_solution'

    if not os.path.exists(fname + extension):
        print('Finding the greedy solution ...')
        tap_solved = 0

        damaged_links = [link for link in damaged_dict.keys()]
        eligible_to_add = deepcopy(damaged_links)

        test_net = deepcopy(net_after)
        after_ = after_eq_tstt

        decoy_dd = deepcopy(damaged_dict)
        path = []
        for i in range(len(damaged_links)):
            new_tstts = []
            new_bb = {}

            for link in eligible_to_add:
                added = [link]
                not_fixed = set(eligible_to_add).difference(set(added))
                test_net.not_fixed = set(not_fixed)

                after_fix_tstt, __, __ = solve_UE(net=test_net, eval_seq=True)
                global memory
                memory[frozenset(test_net.not_fixed)] = after_fix_tstt
                tap_solved += 1

                diff = after_ - after_fix_tstt
                new_bb[link] = after_ - after_fix_tstt

                global wb_update
                global bb_update
                if wb_update[link] > diff:
                    wb_update[link] = diff
                if bb_update[link] < diff:
                    bb_update[link] = diff

                new_tstts.append(after_fix_tstt)

            ordered_days = []
            orderedb_benefits = []
            sorted_d = sorted(decoy_dd.items(), key=lambda x: x[1])
            for key, value in sorted_d:
                ordered_days.append(value)
                orderedb_benefits.append(new_bb[key])

            __, __, ord = orderlists(orderedb_benefits, ordered_days, rem_keys=sorted_d)

            link_to_add = ord[0][0]
            path.append(link_to_add)
            min_index = eligible_to_add.index(link_to_add)
            after_ = new_tstts[min_index]
            eligible_to_add.remove(link_to_add)
            decoy_dd = deepcopy(decoy_dd)
            del decoy_dd[link_to_add]

        net = deepcopy(net_after)

        tap_solved += 1
        elapsed = time.time() - start + time_net_before + time_net_after

        bound, __, __, __ = eval_sequence(
            net, path, after_eq_tstt, before_eq_tstt, num_crews=num_crews)

        save(fname + '_obj', bound)
        save(fname + '_path', path)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', tap_solved)
    else:
        bound = load(fname + '_obj')
        path = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    print('Greedy objective value: ',bound)
    if num_crews != 1:
        crew_seqs, make_span = gen_crew_seqs(path, damaged_dict, num_crews)
        return [bound, elapsed, tap_solved, make_span], crew_seqs
    else:
        return [bound, elapsed, tap_solved], path


def greedy_heuristic_mult(net_after, after_eq_tstt, before_eq_tstt, time_net_before,
                          time_net_after, num_crews):
    """multicrew heuristic which orders links for repair at each step based on
    immediate effect on tstt if that link completed before any additional links"""
    start = time.time()
    tap_solved = 0
    fname = net_after.save_dir + '/greedy_solution'
    global memory

    if not os.path.exists(fname + extension):
        print('Finding the greedy solution ...')
        tap_solved = 0

        damaged_links = [link for link in damaged_dict.keys()]
        eligible_to_add = deepcopy(damaged_links)

        test_net = deepcopy(net_after)
        after_ = after_eq_tstt

        decoy_dd = deepcopy(damaged_dict)
        path = []
        crew_order_list = []
        crews = [0]*num_crews # Crews is the current finish time for that crew
        which_crew = dict()
        completion = dict() # Completion time of each link
        baseidx = -1

        for i in range(len(damaged_links)):
            if i > 0 and i < num_crews:
                continue

            new_tstts = []
            new_bb = {}

            for link in eligible_to_add:
                added = [link]
                not_fixed = set(damaged_dict).difference(
                    set(crew_order_list[:baseidx+1]))
                not_fixed.difference_update(set(added))
                test_net.not_fixed = set(not_fixed)

                after_fix_tstt, __, __ = solve_UE(net=test_net, eval_seq=True)
                memory[frozenset(test_net.not_fixed)] = after_fix_tstt
                tap_solved += 1

                diff = after_ - after_fix_tstt
                new_bb[link] = after_ - after_fix_tstt

                global wb_update
                global bb_update
                if wb_update[link] > diff:
                    wb_update[link] = diff
                if bb_update[link] < diff:
                    bb_update[link] = diff

                new_tstts.append(after_fix_tstt)

            ordered_days = []
            orderedb_benefits = []

            sorted_d = sorted(decoy_dd.items(), key=lambda x: x[1])
            for key, value in sorted_d:
                ordered_days.append(value)
                orderedb_benefits.append(new_bb[key])

            __, __, order = orderlists(orderedb_benefits,ordered_days,rem_keys=sorted_d)

            if i == 0:
                links_to_add = []
                after_ = []
                for j in range(num_crews):
                    links_to_add.append(order[j][0])

                for j in range(num_crews):
                    temp = damaged_dict[links_to_add[0]]
                    crew_order_list.append(links_to_add[0])
                    for ij in links_to_add[1:]:
                        if damaged_dict[ij] < temp:
                            temp = damaged_dict[ij]
                            crew_order_list[j] = ij
                    crews[j]+=damaged_dict[crew_order_list[j]]
                    completion[crew_order_list[j]] = deepcopy(crews[j])
                    which_crew[crew_order_list[j]] = j
                    path.append(crew_order_list[j])
                    links_to_add.remove(crew_order_list[j])

                min_index = eligible_to_add.index(crew_order_list[0])
                after_ = new_tstts[min_index]
                baseidx = 0

                decoy_dd = deepcopy(decoy_dd)
                for link in path:
                    eligible_to_add.remove(link)
                    del decoy_dd[link]

            else:
                link_to_add = order[0][0]
                which_crew[link_to_add] = crews.index(min(crews))
                crews[which_crew[link_to_add]] += damaged_dict[link_to_add]
                completion[link_to_add] = deepcopy(crews[which_crew[link_to_add]])

                if completion[link_to_add] == max(crews):
                    crew_order_list.append(link_to_add)
                else:
                    crew_order_list.insert(len(crew_order_list) - num_crews +
                        sorted(crews).index(crews[which_crew[link_to_add]])+1,
                        link_to_add)
                path.append(link_to_add)

                if completion[link_to_add] == min(crews):
                    min_index = eligible_to_add.index(link_to_add)
                    after_ = new_tstts[min_index]
                    baseidx = crew_order_list.index(link_to_add)
                else:
                    base = [k for k, v in completion.items() if v==min(crews)][0]
                    baseidx = crew_order_list.index(base)
                    not_fixed = set(damaged_dict).difference(
                        set(crew_order_list[:baseidx+1]))
                    test_net.not_fixed = set(not_fixed)
                    after_, __, __ = solve_UE(net=test_net, eval_seq=True)
                    memory[frozenset(test_net.not_fixed)] = after_
                    tap_solved += 1

                eligible_to_add.remove(link_to_add)
                decoy_dd = deepcopy(decoy_dd)
                del decoy_dd[link_to_add]

        net = deepcopy(net_after)

        tap_solved += 1
        elapsed = time.time() - start + time_net_before + time_net_after

        bound, __, __, __ = eval_sequence(
            net, path, after_eq_tstt, before_eq_tstt, num_crews=num_crews)

        save(fname + '_obj', bound)
        save(fname + '_path', path)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', tap_solved)
    else:
        bound = load(fname + '_obj')
        path = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    print('Greedy objective value: ',bound)
    if num_crews != 1:
        crew_seqs, make_span = gen_crew_seqs(path, damaged_dict, num_crews)
        return [bound, elapsed, tap_solved, make_span], crew_seqs
    else:
        return [bound, elapsed, tap_solved], path


def find_decomp(decomp_method, decomp_list, net_before, net_after, after_eq_tstt,
                before_eq_tstt, time_net_before, num_crews):
    """Finds a decomposition into crews, then solves for within-crew sequences"""
    start = time.time()

    fname = net_after.save_dir + '/'+decomp_method+'_solution'
    if not os.path.exists(fname + extension):
        if decomp_method == 'minmake':
            which_crew, taps, make_span = minspan(net_after, num_crews)
        else: # 'icassign'
            which_crew, taps, make_span = icassign(net_before, num_crews)
        init_time = time.time() - start
        print('Time to find '+decomp_method+' crew assignments: '+str(init_time))

        # temp_seq is a tuples of tuples, where each subtuple is order within a crew
        decomp_res, decomp_seq = list(), list()
        for el in decomp_list:
            match el:
                case 1:
                    temp_res, temp_seq = decomp_brute_global(net_after, which_crew,
                        after_eq_tstt, before_eq_tstt, num_crews, make_span)
                case 2:
                    temp_res, temp_seq = decomp_brute_local(net_after, which_crew,
                        after_eq_tstt, before_eq_tstt, num_crews, make_span)
                case 3:
                    temp_res, temp_seq = decomp_greedy(net_after, which_crew,
                        after_eq_tstt, before_eq_tstt, num_crews, make_span)
                case 4:
                    temp_res, temp_seq = decomp_IF(net_before, which_crew,
                        after_eq_tstt, before_eq_tstt, num_crews, make_span)
                case 5:
                    temp_res, temp_seq = decomp_SPT(net_after, which_crew,
                        after_eq_tstt, before_eq_tstt, num_crews, make_span)
            temp_res[1] += init_time + time_net_before
            temp_res[2] += taps
            decomp_res.append(temp_res)
            decomp_seq.append(temp_seq)
    else:
        pass

    return decomp_res, decomp_seq


def minspan(net_after, num_crews):
    """finds a min makespan assignment to crews using a longest processing time first
    greedy algorithm (4/3 approximation)"""
    test_net = deepcopy(net_after)
    sorted_d = sorted(test_net.damaged_dict.items(), key=lambda x: x[1], reverse=True)
    order_list, __ = zip(*sorted_d)

    # which_crew is a dict where keys are links and values are crew assignments
    __, which_crew, days_list = gen_crew_order(order_list,
            damaged_dict=test_net.damaged_dict, num_crews=num_crews)

    return which_crew, 0, sum(days_list)


def icassign(net_before, num_crews):
    """finds an interaction coefficient-based assignment to crews"""
    taps = 0
    test_net = deepcopy(net_before)
    initflow = dict()
    damaged_links = list(test_net.damaged_dict.keys())
    toassign = deepcopy(damaged_links)
    for link in damaged_links:
        initflow[link] = test_net.linkDict[link]['flow']

    delta = np.zeros((len(damaged_links),len(damaged_links)))
    for link in damaged_links:
        test_net.not_fixed = set([link])
        solve_UE(net=test_net, eval_seq=True, flows=True)
        taps += 1
        for l2 in damaged_links:
            if initflow[l2] == 0:
                if test_net.linkDict[l2]['flow'] > 1:
                    temp = 1
                else:
                    temp = 0
            else:
                temp = (test_net.linkDict[l2]['flow'] - initflow[l2])/initflow[l2]
            delta[damaged_links.index(link),damaged_links.index(l2)] = temp

    delprime = deepcopy(delta)
    which_crew = dict()
    crews = [0]*num_crews
    safety = sum(test_net.damaged_dict.values())/num_crews
    for crew in range(num_crews):
        i,j = np.unravel_index(np.argmax(delprime), np.array(delprime).shape)
        l1 = damaged_links[i]
        l2 = damaged_links[j]
        try:
            if test_net.damaged_dict[l1] + test_net.damaged_dict[l2] > safety:
                if test_net.damaged_dict[l1] > test_net.damaged_dict[l2]:
                    which_crew[l1] = crew
                    crews[crew] += test_net.damaged_dict[l1]
                    toassign.remove(l1)
                else:
                    which_crew[l2] = crew
                    crews[crew] += test_net.damaged_dict[l2]
                    toassign.remove(l2)
            else:
                which_crew[l1] = crew
                crews[crew] += test_net.damaged_dict[l1]
                toassign.remove(l1)
                which_crew[l2] = crew
                crews[crew] += test_net.damaged_dict[l2]
                toassign.remove(l2)
                delprime[i,:] = -1
                delprime[:,i] = -1
                delprime[:,j] = -1
                delprime[j,:] = -1
        except:
            pass

    while toassign != []:
        crew = crews.index(min(crews))
        temp = []
        for link in toassign:
            t1 = sum([delta[damaged_links.index(link),damaged_links.index(j)]
                        for j in which_crew if which_crew[j]==crew])
            t2 = sum([delta[damaged_links.index(i),damaged_links.index(link)]
                        for i in which_crew if which_crew[i]==crew])
            temp.append(max(t1,t2))
        ind = temp.index(max(temp))
        new = toassign[ind]
        which_crew[new] = crew
        crews[crew] += test_net.damaged_dict[new]
        toassign.remove(new)

    return which_crew, taps, max(crews)


def decomp_brute_global(net_after, which_crew, after_eq_tstt, before_eq_tstt, num_crews,
                        make_span):
    """find optimal global solution by brute force given that links are pre-assigned to
    crews"""
    start = time.time()
    tap_solved = 0
    damaged_link_list = []
    for crew in range(num_crews):
        damaged_link_list.append([])
    for link in net_after.damaged_dict.keys():
        damaged_link_list[which_crew[link]].append(link)
    print(damaged_link_list)

    print('Finding the optimal sequence for the decomposed problem...')
    sub_sequences = [0]*num_crews
    for crew in range(num_crews):
        sub_sequences[crew] = itertools.permutations(damaged_link_list[crew])
    all_sequences = itertools.product(*sub_sequences)

    i = 0
    min_cost = 1e+80
    min_seq = None

    for sequence in all_sequences:
        seq_net = deepcopy(net_after)
        cost, eval_taps, __, __ = eval_working_sequence(
            seq_net, sequence, after_eq_tstt, before_eq_tstt, num_crews=num_crews)
        tap_solved += eval_taps

        if cost < min_cost or min_cost == 1e+80:
            min_cost = cost
            min_seq = sequence
        i += 1

    elapsed = time.time() - start
    bound, __, __, __ = eval_sequence(
        seq_net, min_seq, after_eq_tstt, before_eq_tstt, num_crews=num_crews)

    return [bound, elapsed, tap_solved, make_span], min_seq


def decomp_brute_local(net_after, which_crew, after_eq_tstt, before_eq_tstt, num_crews,
                    make_span):
    """find optimal order within each crew by brute force given that links are pre-
    assigned to crews, and acting as if only the links within the crew being optimized
    are broken"""
    start = time.time()
    tap_solved = 0
    damaged_link_list = []
    for crew in range(num_crews):
        damaged_link_list.append([])
    for link in damaged_dict.keys():
        damaged_link_list[which_crew[link]].append(link)

    print('Finding the quasi-optimal sequence for the decomposed problem...')
    sub_sequences = [0]*num_crews
    for crew in range(num_crews):
        sub_sequences[crew] = itertools.permutations(damaged_link_list[crew])
    min_cost = [1e+80]*num_crews
    min_seq = [ [] for _ in range(num_crews)]

    for crew in range(num_crews):
        sub_net = deepcopy(net_after)
        subdict = {k: sub_net.damaged_dict[k] for k in damaged_link_list[crew]}
        sub_net.damaged_dict = subdict
        sub_net.not_fixed = set(damaged_link_list[crew])
        sub_after_eq_tstt, __, __ = solve_UE(net=sub_net, eval_seq=True,
                                             warm_start=False)
        tap_solved += 1
        for sequence in sub_sequences[crew]:
            seq_net = deepcopy(sub_net)
            cost, eval_taps, __, __ = eval_working_sequence(
                seq_net, sequence, sub_after_eq_tstt, before_eq_tstt, num_crews=1)
            tap_solved += eval_taps

            if cost < min_cost[crew] or min_cost[crew] == 1e+80:
                min_cost[crew] = cost
                min_seq[crew] = sequence

    elapsed = time.time() - start
    test_net = deepcopy(net_after)
    bound, __, __, __ = eval_sequence(
        test_net, min_seq, after_eq_tstt, before_eq_tstt, num_crews=num_crews)

    return [bound, elapsed, tap_solved, make_span], min_seq


def decomp_greedy(net_after, which_crew, after_eq_tstt, before_eq_tstt, num_crews,
                  make_span):
    """find sequential greedy solution given that links are pre-assigned to crews"""
    start = time.time()
    print('Finding the greedy assignment for the decomposed problem...')
    tap_solved = 0
    damaged_link_list = []
    decoy_dd = []
    path = []

    for crew in range(num_crews):
        damaged_link_list.append([])
        decoy_dd.append({})
        path.append([])
    for link in net_after.damaged_dict.keys():
        damaged_link_list[which_crew[link]].append(link)
        decoy_dd[which_crew[link]][link] = net_after.damaged_dict[link]

    eligible_to_add = deepcopy(damaged_link_list)
    flat_eligible = [link for crew in eligible_to_add for link in crew]
    test_net = deepcopy(net_after)
    after_ = after_eq_tstt
    crew_order_list = []
    crews = [0]*num_crews
    completion = dict()
    base_idx = -1

    new_tstts = []
    new_bb = {}
    for link in flat_eligible:
        not_fixed = set(test_net.damaged_dict).difference(set(crew_order_list[
            :base_idx+1]))
        not_fixed.difference_update(set([link]))
        test_net.not_fixed = set(not_fixed)

        after_fix_tstt, __, __ = solve_UE(net=test_net, eval_seq=True)
        memory[frozenset(test_net.not_fixed)] = after_fix_tstt
        tap_solved += 1

        diff = after_ - after_fix_tstt
        new_bb[link] = after_ - after_fix_tstt
        new_tstts.append(after_fix_tstt)

    sorted_d = []
    ordered_days = []
    orderedb_benefits = []
    order = []
    for crew in range(num_crews):
        sorted_d.append(sorted(decoy_dd[crew].items(), key=lambda x: x[1]))
        ordered_days.append([])
        orderedb_benefits.append([])
        order.append([])
        for key, value in sorted_d[crew]:
            ordered_days[crew].append(value)
            orderedb_benefits[crew].append(new_bb[key])
        _, _, order[crew] = orderlists(orderedb_benefits[crew], ordered_days[crew],
                                       rem_keys=sorted_d[crew])

    temp = []
    for crew in range(num_crews):
        temp.append(test_net.damaged_dict[order[crew][0][0]])
    for el in sorted(temp):
        crew = temp.index(el)
        crew_order_list.append(order[crew][0][0])
        crews[crew] += el
        completion[order[crew][0][0]] = deepcopy(crews[crew])
        path[crew].append(order[crew][0][0])

    min_index = flat_eligible.index(crew_order_list[0])
    after_ = new_tstts[min_index]
    base_idx = 0
    decoy_dd = deepcopy(decoy_dd)
    for crew in range(num_crews):
        flat_eligible.remove(path[crew][0])
        eligible_to_add[crew].remove(path[crew][0])
        del decoy_dd[crew][path[crew][0]]

    while flat_eligible != []:
        active = crews.index(min(crews))
        new_tstts = []
        new_bb = {}
        if eligible_to_add[active] == []:
            duration = 0
            for link in flat_eligible:
                if decoy_dd[which_crew[link]][link] > duration:
                    duration = decoy_dd[which_crew[link]][link]
                    to_move = link
            eligible_to_add[which_crew[to_move]].remove(to_move)
            del decoy_dd[which_crew[to_move]][to_move]
            which_crew[to_move] = active
            eligible_to_add[active].append(to_move)
            decoy_dd[active][to_move] = duration
        for link in eligible_to_add[active]:
            not_fixed = set(test_net.damaged_dict).difference(set(crew_order_list[
                :base_idx+1]))
            not_fixed.difference_update(set([link]))
            test_net.not_fixed = set(not_fixed)

            after_fix_tstt, __, __ = solve_UE(net=test_net, eval_seq=True)
            memory[frozenset(test_net.not_fixed)] = after_fix_tstt
            tap_solved += 1

            diff = after_ - after_fix_tstt
            new_bb[link] = after_ - after_fix_tstt
            new_tstts.append(after_fix_tstt)

        ordered_days = []
        orderedb_benefits = []
        order = []
        sorted_d = sorted(decoy_dd[active].items(), key=lambda x: x[1])
        for key, value in sorted_d:
            ordered_days.append(value)
            orderedb_benefits.append(new_bb[key])
        _, _, order = orderlists(orderedb_benefits, ordered_days, rem_keys=sorted_d)

        try:
            link_to_add = order[0][0]
        except:
            print('order: {}'.format(order))
            link_to_add = order[0][0]
        crews[active] += test_net.damaged_dict[link_to_add]
        completion[link_to_add] = deepcopy(crews[active])

        if completion[link_to_add] == max(crews):
            crew_order_list.append(link_to_add)
        else:
            crew_order_list.insert(len(crew_order_list) - num_crews +
                sorted(crews).index(crews[active]) + 1 ,link_to_add)
        path[active].append(link_to_add)

        if completion[link_to_add] == min(crews):
            min_index = eligible_to_add[active].index(link_to_add)
            after_ = new_tstts[min_index]
            baseidx = crew_order_list.index(link_to_add)
        else:
            base = [k for k, v in completion.items() if v==min(crews)][0]
            baseidx = crew_order_list.index(base)
            not_fixed = set(test_net.damaged_dict).difference(set(crew_order_list[
                :baseidx+1]))
            test_net.not_fixed = set(not_fixed)
            after_, __, __ = solve_UE(net=test_net, eval_seq=True)
            memory[frozenset(test_net.not_fixed)] = after_
            tap_solved += 1

        flat_eligible.remove(link_to_add)
        eligible_to_add[active].remove(link_to_add)
        decoy_dd = deepcopy(decoy_dd)
        del decoy_dd[active][link_to_add]

    net = deepcopy(net_after)
    tap_solved += 1
    elapsed = time.time() - start

    bound, __, __, __ = eval_sequence(
            net, path, after_eq_tstt, before_eq_tstt, num_crews=num_crews)

    return [bound, elapsed, tap_solved, make_span], path


def decomp_IF(net_before, which_crew, after_eq_tstt, before_eq_tstt, num_crews,
              make_span):
    start = time.time()
    print('Finding the IF assignment for the decomposed problem ...')
    tot_flow = 0
    if_net = deepcopy(net_before)
    for ij in if_net.linkDict:
        tot_flow += if_net.linkDict[ij]['flow']

    damaged_link_list = []
    path = []
    for crew in range(num_crews):
        damaged_link_list.append([])
        path.append([])
    for link in if_net.damaged_dict.keys():
        damaged_link_list[which_crew[link]].append(link)

    if_dict = {}
    for link_id in if_net.damaged_dict.keys():
        link_flow = if_net.linkDict[link_id]['flow']
        if_dict[link_id] = link_flow / tot_flow
    sorted_d = sorted(if_dict.items(), key=lambda x: x[1], reverse=True)

    for link_id, v in sorted_d:
        path[which_crew[link_id]].append(link_id)

    elapsed = time.time() - start
    bound, __, __, __ = eval_sequence(if_net, path, after_eq_tstt, before_eq_tstt,
        num_crews=num_crews)
    tap_solved = 1

    return [bound, elapsed, tap_solved, make_span], path


def decomp_SPT(net_after, which_crew, after_eq_tstt, before_eq_tstt, num_crews,
               make_span):
    start = time.time()
    print('Finding the SPT assignment for the decomposed problem ...')

    test_net = deepcopy(net_after)
    sorted_d =  sorted(test_net.damaged_dict.items(), key=lambda x: x[1])
    spt_list, __ = zip(*sorted_d)

    path = []
    for crew in range(num_crews):
        path.append([])
    for link in spt_list:
        path[which_crew[link]].append(link)
    elapsed = time.time() - start
    bound, __, __, __ = eval_sequence(test_net, path, after_eq_tstt, before_eq_tstt,
        num_crews=num_crews)
    tap_solved = 0

    return [bound, elapsed, tap_solved, make_span], path


def make_art_links(NETFILE, TRIPFILE, mc_weights, demand_mult):
    """creates artificial links with travel time 10x before eq shortest path"""
    start = time.time()
    art_net = Network(NETFILE, TRIPFILE)
    art_net.netfile = NETFILE
    art_net.tripfile = TRIPFILE
    art_net.not_fixed = set([])
    art_net.art_links = {}
    art_net.mc_weights = mc_weights
    art_net.demand_mult = demand_mult
    art_net.maxruntime=str(10)

    tstt, __, __ = solve_UE(net=art_net, eval_seq=True, flows=True, warm_start=False)

    for ij in art_net.link:
        art_net.link[ij].flow = art_net.linkDict[ij]['flow']
        art_net.link[ij].cost = art_net.linkDict[ij]['cost']

    art_links = {}
    origin_list = []
    backlink = {}
    cost = {}
    for od in art_net.ODpair.values():
        if od.origin not in origin_list:
            origin_list.append(od.origin)
            backlink[od.origin], cost[od.origin] = art_net.shortestPath(od.origin)
            if od.origin != od.destination:
                ODID = str(od.origin) + '->' + str(od.destination)
                art_links[ODID] = 10*cost[od.origin][od.destination]
        else:
            if od.origin != od.destination:
                ODID = str(od.origin) + '->' + str(od.destination)
                art_links[ODID] = 10*cost[od.origin][od.destination]

    art_net.not_fixed = set(damaged_links)
    tstt, __, __ = solve_UE(net=art_net, eval_seq=True, flows=True, warm_start=False)

    for ij in art_net.link:
        art_net.link[ij].flow = art_net.linkDict[ij]['flow']
        art_net.link[ij].cost = art_net.linkDict[ij]['cost']

    origin_list = []
    postbacklink = {}
    postcost = {}
    count = 0
    for od in art_net.ODpair.values():
        if od.origin not in origin_list:
            origin_list.append(od.origin)
            postbacklink[od.origin], postcost[od.origin] = art_net.shortestPath(
                od.origin)
            if postcost[od.origin][od.destination] >= 99999:
                count += 1
            if (od.origin != od.destination and postcost[od.origin][od.destination]
                    < 10*cost[od.origin][od.destination]):
                ODID = str(od.origin) + '->' + str(od.destination)
                del art_links[ODID]
        else:
            if postcost[od.origin][od.destination] >= 99999:
                count += 1
            if (od.origin != od.destination and postcost[od.origin][od.destination]
                    < 10*cost[od.origin][od.destination]):
                ODID = str(od.origin) + '->' + str(od.destination)
                del art_links[ODID]
    print('Created {} artificial links.'.format(len(art_links)))
    if count > len(art_links):
        print('There are {} more paths exceeding cost of 99999 than artificial links \
            created'.format(count-len(art_links)))

    art_net.art_links = art_links
    record_art_net(art_net)
    tstt, __, __ = solve_UE(net=art_net, eval_seq=True, flows=True, warm_start=False)

    elapsed = time.time()-start
    print('Runtime to create artificial links: ', elapsed)
    return art_links


def plot_nodes_links(save_dir, net, damaged_links, coord_dict, names = False,
                     num_crews = 1, which_crew = None):
    """function to map all links and nodes, highlighting damaged links"""
    xMax = max(coord_dict.values())[0]
    xMin = min(coord_dict.values())[0]
    yMax = coord_dict[1][1]
    yMin = coord_dict[1][1]

    for i in coord_dict:
        if coord_dict[i][1] < yMin:
            yMin = coord_dict[i][1]
        if coord_dict[i][1] > yMax:
            yMax = coord_dict[i][1]

    scale = 10**(math.floor(min(math.log(xMax-xMin, 10), math.log(yMax-yMin, 10))) - 1)

    fig, ax = plt.subplots(figsize = (12,9))
    plt.xlim(math.floor(xMin/scale) * scale, math.ceil(xMax/scale) * scale)
    plt.ylim(math.floor(yMin/scale) * scale, math.ceil(yMax/scale+0.2) * scale)
    plt.rcParams["font.size"] = 6

    # Plot nodes
    nodesx = list()
    nodesy = list()
    for i in range(1, len(coord_dict)+1):
        nodesx.append(coord_dict[i][0])
        nodesy.append(coord_dict[i][1])
    if names:
        if NETWORK.find('ChicagoSketch') >= 0:
            for i in range(388, len(nodesx)):
                plt.annotate(i+1, (nodesx[i], nodesy[i]))
        else:
            for i in range(len(nodesx)):
                plt.annotate(i+1, (nodesx[i], nodesy[i]))
    plt.scatter(nodesx, nodesy, s=4)

    # Plot links
    segments = list()
    damaged_segments = list()
    if num_crews != 1:
        for crew in range(num_crews):
            damaged_segments.append([])

    if NETWORK.find('Berlin') >= 0:
        line_width = 0.0025
    else:
        line_width = 0.00025
    for ij in [ij for ij in net.link if ij not in damaged_links]:
        line = mpatch.FancyArrow(coord_dict[net.link[ij].tail][0],
            coord_dict[net.link[ij].tail][1],
            coord_dict[net.link[ij].head][0]-coord_dict[net.link[ij].tail][0],
            coord_dict[net.link[ij].head][1]-coord_dict[net.link[ij].tail][1],
            length_includes_head = True, width = line_width)
        segments.append(line)
    for ij in damaged_links:
        line = mpatch.FancyArrow(coord_dict[net.link[ij].tail][0],
            coord_dict[net.link[ij].tail][1],
            coord_dict[net.link[ij].head][0]-coord_dict[net.link[ij].tail][0],
            coord_dict[net.link[ij].head][1]-coord_dict[net.link[ij].tail][1],
            length_includes_head = True, width = line_width)
        if num_crews == 1:
            damaged_segments.append(line)
        else:
            damaged_segments[which_crew[ij]].append(line)

    lc = mc.PatchCollection(segments)
    ax.add_collection(lc)
    if num_crews == 1:
        lc_damaged = mc.PatchCollection(damaged_segments, color = 'tab:red')
        ax.add_collection(lc_damaged)
    else:
        jet = cm.get_cmap('jet', num_crews)
        lc_damaged = list()
        for crew in range(num_crews):
            lc_damaged.append(mc.PatchCollection(damaged_segments[crew],
                              color = jet(crew/num_crews)))
            ax.add_collection(lc_damaged[crew])
    ax.set_axis_off()
    plt.title('Map of ' + NETWORK.split('/')[-1] + ' with ' + str(len(damaged_links))
        + ' damaged links', fontsize=12)

    save_fig(save_dir, 'map', tight_layout=True)
    plt.close(fig)


def plot_time_OBJ(save_dir, start, end, bs_time_list=None, bs_OBJ_list=None, sa_time_list=None,
                  sa_OBJ_list=None):
    """function to plot running time vs OBJ progression"""
    """1,5 should plot for rep in range(5), 6,10 should plot for rep in range(5,10)"""

    fig, ax = plt.subplots(figsize=(8,6))
    #jet = cm.get_cmap('jet', reps)
    cmap = plt.get_cmap('plasma', end-start+1)

    if bs_time_list is not None:
        bs_indices = [i for i, time in enumerate(bs_time_list) if time == 0]
        bs_indices.append(len(bs_time_list))
        if num_broken < 24:
            for rep in range(start-1,end):
                plt.step(bs_time_list[bs_indices[rep]:bs_indices[rep+1]],
                    (1-np.divide(bs_OBJ_list[bs_indices[rep]:bs_indices[rep+1]],
                    bs_OBJ_list[bs_indices[rep]])) * 100, where='post',
                    color=cmap((rep-start+1)/(end-start+1)), label='Beam Search Run ' + str(rep+2-start))
        else:
            for rep in range(start-1,end):
                plt.step(np.divide(bs_time_list[bs_indices[rep]:bs_indices[rep+1]],60),
                    (1-np.divide(bs_OBJ_list[bs_indices[rep]:bs_indices[rep+1]],
                    bs_OBJ_list[bs_indices[rep]])) * 100, where='post',
                    color=cmap((rep-start+1)/(end-start+1)), label='Beam Search Run ' + str(rep+2-start))

    if sa_time_list is not None:
        sa_indices = dict()
        labels = dict()
        if not sacompare:
            labels[100] = 'Simulated Annealing'
            lines = {100: 'solid'}
        else:
            for el in sacompare:
                labels[el] = str(el)

            if sacompare == [100,200,300]:
                labels[100], labels[200] = '1-Neighborhood', '2-Neighborhood'
                labels[300] = '3-Neighborhood'
                lines = {100:'solid',200:'dashed',300:'dotted'}
            elif sacompare == [100,101,102]:
                labels[100], labels[101], labels[102] = 'Default','Fail-Min','Fail-Random'
                lines = {100:'solid',101:'dashed',102:'dotted'}
            else:
                lineset = [(0,(3,3,1,3,1,3)),'dashdot', 'dotted', 'dashed', 'solid']
                lines = dict()
                for el in sacompare:
                    lines[el] = lineset.pop()

        for i in sa_time_list:
            sa_indices[i] = [j for j, time in enumerate(sa_time_list[i]) if time == 0]
            sa_indices[i].append(len(sa_time_list[i]))
            if num_broken < 24:
                for rep in range(start-1,end):
                    plt.step(sa_time_list[i][sa_indices[i][rep]:sa_indices[i][rep+1]],
                        (1-np.divide(sa_OBJ_list[i][sa_indices[i][rep]:sa_indices[i][rep+1]],
                        sa_OBJ_list[i][sa_indices[i][rep]])) * 100, where='post',
                        color=cmap((rep-start+1)/(end-start+1)), linestyle=lines[i],
                        label=labels[i] +' Run '+ str(rep+2-start))
            else:
                for rep in range(start-1,end):
                    plt.step(np.divide(sa_time_list[i][sa_indices[i][rep]:sa_indices[i][rep+1]], 60),
                        (1 - np.divide(sa_OBJ_list[i][sa_indices[i][rep]:sa_indices[i][rep+1]],
                        sa_OBJ_list[i][sa_indices[i][rep]])) * 100, where='post',
                        color=cmap((rep-start+1)/(end-start+1)), linestyle=lines[i],
                        label=labels[i] +' Run '+ str(rep+2-start))

    if num_broken < 24:
        plt.title('Runtime elapsed (seconds) vs OBJ function improvement (%) for '
            + NETWORK.split('/')[-1] + ' with ' + str(len(damaged_links))
            + ' damaged links', fontsize=10)
        ax.set_xlabel('Runtime elapsed (seconds)', fontsize=8)
    else:
        plt.title('Runtime elapsed (minutes) vs OBJ function improvement (%) for '
            + NETWORK.split('/')[-1] + ' with ' + str(len(damaged_links))
            + ' damaged links', fontsize=10)
        ax.set_xlabel('Runtime elapsed (minutes)', fontsize=8)
    if bs_time_list is not None and sa_time_list is not None:
        plt.legend(ncol=1+len(sa_time_list))
    elif bs_time_list is None:
        plt.legend(ncol=len(sa_time_list))
    ax.set_ylabel('OBJ function improvement over initial seed (%)',fontsize=8)

    save_fig(save_dir, 'timevsOBJ'+'-'+str(start)+'-'+str(end), tight_layout=True)
    plt.close(fig)


if __name__ == '__main__':

    net_name = args.net_name
    num_broken = args.num_broken
    approx = args.approx
    arc = args.arc
    if arc and not approx:
        arc = False
        print('arc reset to False because approx not enabled')
    reps = args.reps
    beam_search = args.beamsearch
    beta = args.beta
    gamma = args.gamma
    if isinstance(args.num_crews, int):
        num_crews = args.num_crews
        alt_crews = None
    elif len(args.num_crews) == 1:
        num_crews = int(args.num_crews[0])
        alt_crews = None
    else:
        num_crews = int(args.num_crews[0])
        alt_crews = list(args.num_crews[1:])
    if isinstance(args.mdecomp, int):
        if args.mdecomp: mdecomp = [args.mdecomp]
        else: mdecomp = args.mdecomp
    else:
        mdecomp = list(args.mdecomp)
    if isinstance(args.idecomp, int):
        if args.idecomp: idecomp = [args.idecomp]
        else: idecomp = args.idecomp
    else:
        idecomp = list(args.idecomp)
    tables = args.tables
    graphing = args.graphing
    before_after = args.onlybeforeafter
    output_sequences = args.output_sequences
    full = args.full
    rand_gen = args.random
    location = args.loc
    calc_ff = args.ff
    try: bf = list(args.bf)
    except: bf = [args.bf]
    opt = bf[0]
    try: sp = list(args.sp)
    except: sp = [args.sp]
    try: mip = list(args.mip)
    except: mip = [args.mip]
    sa = args.sa
    if isinstance(args.sacompare, int):
        if args.sacompare: sacompare = [args.sacompare]
        else: sacompare = args.sacompare
    else:
        sacompare = list(args.sacompare)
    damaged_dict_preset = args.damaged
    multiclass = args.mc
    if damaged_dict_preset != '':
        print('Using preset damaged_dict: ', damaged_dict_preset)
    if isinstance(args.mc_weights, int):
        mc_weights = args.mc_weights
    elif len(args.mc_weights)==1:
        mc_weights = args.mc_weights[0]
    else:
        mc_weights = list(args.mc_weights[:])
    if not isinstance(mc_weights, int):
        print('Class weights are: ',mc_weights)
    demand_mult = args.demand

    # Compile list of methods
    methods = []
    if num_crews == 1 and alt_crews is None:
        methods.extend(['Simple UB', 'HLB'])
    if calc_ff: methods.append('FF LB')
    temp = {1:'OPT BF', 2:'ML BF'}
    for el in bf:
        if el != 0: methods.append(temp[el])
    temp = {1:'OPT SP', 2:'ML SP'}
    for el in sp:
        if el != 0: methods.append(temp[el])
    temp = {1:'OPT IP', 2:'ML IP', 3:'DeltaTSTT'}
    for el in mip:
        if el != 0: methods.append(temp[el])
    if arc: methods.append('Arc Flow')
    temp = {1:'(global opt)', 2:'(local opt)', 3:'(greedy)', 4:'(IF)', 5:'(SPT)'}
    if mdecomp:
        for el in mdecomp:
            methods.append('Minspan '+temp[el])
    if idecomp:
        for el in idecomp:
            methods.append('IC Assign '+temp[el])
    if approx:
        methods.append('LASR')
        methods.append('LAFO')
        if approx==3: methods.append('AltLASR')
    methods.extend(['SQG','LZG','IF','SPT'])
    if sa:
        if not sacompare:
            if sa==1:
                methods.append('Sim Anneal')
            else:
                for i in range(sa):
                    methods.append('Sim Anneal '+str(i+1))
        else:
            for el in sacompare:
                if sa==1:
                    methods.append('SA Method '+str(el))
                else:
                    for i in range(sa):
                        methods.append('SA Method '+str(el)+' '+str(i+1))

    if beam_search: methods.append('Beam Search')
    print(methods)

    NETWORK = os.path.join(FOLDER, net_name)
    if net_name == 'Chicago-Sketch':
        net_name = 'ChicagoSketch'
    if net_name == 'Berlin-Mitte-Center':
        net_name = 'berlin-mitte-center'
    if net_name == 'Berlin-Mitte-Center2':
        net_name = 'berlin-mitte-center2'
    if net_name == 'Berlin-Mitte-Center3':
        net_name = 'berlin-mitte-center3'
    if net_name == 'Berlin-Center':
        net_name = 'berlin-center'
    if net_name == 'Berlin-Mitte-Prenzlauerberg-Friedrichshain-Center':
        net_name = 'berlin-mitte-prenzlauerberg-friedrichshain-center'
    NETFILE = os.path.join(NETWORK, net_name + "_net.tntp")

    TRIPFILE = os.path.join(NETWORK, net_name + "_trips.tntp")
    if not os.path.exists(TRIPFILE):
        TRIPFILE = list()
        i=1
        while True:
            if os.path.exists(os.path.join(NETWORK, net_name+"_trips"+str(i)+".tntp")):
                TRIPFILE.append(os.path.join(NETWORK, net_name+"_trips"+str(i)+".tntp"))
                i+=1
            else:
                if TRIPFILE == []:
                    print('Improper tripfile naming.')
                    raise utils.BadFileFormatException
                break

    SAVED_FOLDER_NAME = "saved"
    PROJECT_ROOT_DIR = "."
    SAVED_DIR = os.path.join(PROJECT_ROOT_DIR, SAVED_FOLDER_NAME)
    os.makedirs(SAVED_DIR, exist_ok=True)

    NETWORK_DIR = os.path.join(SAVED_DIR, NETWORK)
    os.makedirs(NETWORK_DIR, exist_ok=True)

    if tables:
        get_common_numbers()
        get_tables(NETWORK_DIR)
    else:
        if graphing:
            if beam_search:
                bs_time_list = list()
                bs_OBJ_list = list()
            else:
                bs_time_list = None
                bs_OBJ_list = None
            if sa:
                sa_time_list = dict()
                sa_OBJ_list = dict()
                if sacompare:
                    for el in sacompare:
                        sa_time_list[el] = list()
                        sa_OBJ_list[el] = list()
                else:
                    sa_time_list[100] = list()
                    sa_OBJ_list[100] = list()
            elif sa:
                sa_time_list = list()
                sa_OBJ_list = list()
            else:
                sa_time_list = None
                sa_OBJ_list = None
        global approx_params
        approx_params = None

        lb_gap = []
        lb_percent = []
        num_lb_fail = 0
        for rep in range(reps):
            if damaged_dict_preset != '':
                damaged_dict = load(damaged_dict_preset + '/' + 'damaged_dict')

                SCENARIO_DIR = NETWORK_DIR
                ULT_SCENARIO_DIR = os.path.join(SCENARIO_DIR, str(num_broken))

                i = ord('a')
                while True:
                    try:
                        ULT_SCENARIO_REP_DIR = damaged_dict_preset+chr(i)
                        os.makedirs(ULT_SCENARIO_REP_DIR)
                        break
                    except FileExistsError:
                        print(chr(i) + ' is already taken, trying ' + chr(i+1))
                        i += 1
                    if i > 121:
                        break


            elif rand_gen and damaged_dict_preset=='':
                memory = {}

                net = create_network(NETFILE, TRIPFILE, mc_weights=mc_weights,
                                     demand_mult=demand_mult)
                net.not_fixed = set([])
                net.art_links = {}
                net.maxruntime=str(10)
                solve_UE(net=net, warm_start=False, eval_seq=True)

                f = "flows.txt"
                file_created = False
                st = time.time()
                while not file_created:
                    if os.path.exists(f):
                        file_created = True

                    if time.time() - st > 10:
                        popen = subprocess.call(args, stdout=subprocess.DEVNULL)

                    netflows = {}
                    if file_created:
                        with open(f, "r") as flow_file:
                            for line in flow_file.readlines():
                                if line.find('(') == -1:
                                    continue
                                try:
                                    ij = str(line[:line.find(' ')])
                                    line = line[line.find(' '):].strip()
                                    flow = float(line[:line.find(' ')])
                                    line = line[line.find(' '):].strip()
                                    cost = float(line.strip())

                                    if cost != 99999.0:
                                        netflows[ij] = {}
                                        netflows[ij] = flow
                                except:
                                    break
                        os.remove('flows.txt')

                sorted_d = sorted(netflows.items(), key=op.itemgetter(1))[::-1]

                cutind = len(netflows)*0.7
                try:
                    sorted_d = sorted_d[:int(cutind)]
                except:
                    sorted_d = sorted_d

                all_links = [lnk[0] for lnk in sorted_d]
                flow_on_links = [lnk[1] for lnk in sorted_d]

                np.random.seed((rep)*42+int(num_broken))
                random.seed((rep)*42+int(num_broken))
                netg = Network(NETFILE, TRIPFILE)
                G = nx.DiGraph()

                G.add_nodes_from(np.arange(len(netg.node)) + 1)
                edge_list = []
                for alink in netg.link:
                    edge_list.append((int(netg.link[alink].tail),
                                      int(netg.link[alink].head)))
                G.add_edges_from(edge_list)

                damaged_links = []
                art_links = []
                decoy = deepcopy(all_links)
                i = 0

                dG = deepcopy(G)
                import geopy.distance

                if NETWORK.find('SiouxFalls') >= 0:
                    nodetntp = pd.read_csv(os.path.join(NETWORK, net_name+"_node.tntp"),
                        delimiter='\t')
                    coord_dict = {}
                    for index, row in nodetntp.iterrows():
                        lon = row['X']
                        lat = row['Y']
                        coord_dict[row['Node']] = (lon, lat)

                if NETWORK.find('Chicago-Sketch') >=0:
                    nodetntp = pd.read_csv(os.path.join(NETWORK, net_name+"_node.tntp"),
                        delimiter='\t')
                    coord_dict = {}
                    for index, row in nodetntp.iterrows():
                        lon = row['X']
                        lat = row['Y']
                        coord_dict[row['node']] = (lon, lat)

                if NETWORK.find('Anaheim') >= 0:
                    nodetntp = pd.read_json(os.path.join(NETWORK,
                                                         'anaheim_nodes.geojson'))
                    coord_dict = {}
                    for index, row in nodetntp.iterrows():
                        coord_dict[row['features']['properties']['id']] = row[
                            'features']['geometry']['coordinates']

                if NETWORK.find('Berlin') >= 0:
                    nodetntp = pd.read_csv(os.path.join(NETWORK, net_name+"_node.tntp"),
                        delim_whitespace = True, skipinitialspace=True)
                    coord_dict = {}
                    for index, row in nodetntp.iterrows():
                        lon = row['X']
                        lat = row['Y']
                        coord_dict[row['Node']] = (lon, lat)

                if location:
                    # Pick a center node at random:
                    nodedecoy = deepcopy(list(netg.node.keys()))
                    nodedecoy = sorted(nodedecoy)
                    if NETWORK.find('Chicago-Sketch') >= 0:
                        nodedecoy = nodedecoy[387:]
                    center_node = np.random.choice(nodedecoy, 1, replace=False)[0]
                    nodedecoy.remove(center_node)

                    #find distances from center node:
                    dist_dict = {}
                    for anode in nodedecoy:
                        distance = (np.linalg.norm(np.array(coord_dict[center_node])
                                    - np.array(coord_dict[anode])))
                        dist_dict[anode] = distance

                    #sort dist_dict by distances
                    sorted_dist = sorted(dist_dict.items(), key=op.itemgetter(1))
                    all_nodes = [nodes[0] for nodes in sorted_dist]
                    distances = [nodes[1] for nodes in sorted_dist]

                    selected_nodes = [center_node]

                    distance_original = deepcopy(distances)
                    decoy = deepcopy(all_nodes)

                    i = 0
                    while i <= int(math.floor(num_broken*2/3.0)) and len(distances)>0:
                        another_node = random.choices(decoy,
                            1.0/np.array(distances)**location, k=1)[0]
                        idx = decoy.index(another_node) # default location parameter: 3
                        del distances[idx]
                        del decoy[idx]
                        selected_nodes.append(another_node)
                        i += 1

                    selected_indices = [all_nodes.index(i) for i in selected_nodes[1:]]
                    print('Selected nodes: ', selected_nodes)

                    #create linkset
                    linkset = []
                    for anode in selected_nodes:
                        links_rev = netg.node[anode].reverseStar
                        links_fw = netg.node[anode].forwardStar

                        for alink in links_rev:
                            if alink not in linkset:
                                if NETWORK.find('Chicago-Sketch') >= 0:
                                    if netg.link[alink].tail > 387:
                                        linkset.append(alink)
                                else:
                                    linkset.append(alink)

                        for alink in links_fw:
                            if alink not in linkset:
                                if NETWORK.find('Chicago-Sketch') >= 0:
                                    if netg.link[alink].head > 387:
                                        linkset.append(alink)
                                else:
                                    linkset.append(alink)

                    cop_linkset = deepcopy(linkset)
                    i = 0

                    flow_on_links = []
                    for ij in linkset:
                        flow_on_links.append(netflows[ij])
                    safe = deepcopy(flow_on_links)
                    fail = False
                    iterno = 0
                    while i < int(num_broken):
                        curlen = len(linkset)

                        if not fail:
                            ij = random.choices(linkset, weights=np.exp(np.power(
                                flow_on_links, 1 / 3.0)), k=1)[0]
                        else:
                            ij = random.choices(linkset, weights=np.exp(np.power(
                                flow_on_links, 1 / 4.0)), k=1)[0]

                        u = netg.link[ij].tail
                        v = netg.link[ij].head
                        dG.remove_edge(u, v)

                        for od in netg.ODpair.values():
                            if not nx.has_path(dG, od.origin, od.destination):
                                art_links.append(ij)
                                break

                        i += 1
                        damaged_links.append(ij)
                        ind_ij = linkset.index(ij)
                        del flow_on_links[ind_ij]
                        linkset.remove(ij)

                        if iterno % 100 == 0 and iterno != 0:
                            print(iterno, damaged_links)
                        if iterno % 2000 == 0 and iterno != 0:
                            print(damaged_links)
                            damaged_links = []
                            flow_on_links = safe
                            linkset = cop_linkset
                            i = 0
                            fail = True
                            iterno = 0
                            dG = deepcopy(G)
                        iterno +=1
                        if iterno > 5000:
                            pdb.set_trace()


                else:
                    weights = flow_on_links
                    while i < int(num_broken):
                        curlen = len(decoy)
                        ij = random.choices(decoy, weights=np.exp(
                                np.power(weights, 1/3.0)), k=1)[0]
                        u = netg.link[ij].tail
                        v = netg.link[ij].head
                        if NETWORK.find('Chicago-Sketch') >= 0:
                            if u > 387 and v > 387:
                                dG.remove_edge(u, v)
                                for od in netg.ODpair.values():
                                    if not nx.has_path(dG, od.origin, od.destination):
                                        art_links.append(ij)
                                        break
                                i += 1
                                damaged_links.append(ij)
                        else:
                            dG.remove_edge(u, v)
                            for od in netg.ODpair.values():
                                if not nx.has_path(dG, od.origin, od.destination):
                                    art_links.append(ij)
                                    break
                            i += 1
                            damaged_links.append(ij)

                        ind_ij = decoy.index(ij)
                        del weights[ind_ij]
                        decoy.remove(ij)

                print('Damaged_links are created:', damaged_links)
                damaged_dict = {}
                net.linkDict = {}

                with open(NETFILE, "r") as networkFile:

                    fileLines = networkFile.read().splitlines()
                    metadata = utils.readMetadata(fileLines)
                    for line in fileLines[metadata['END OF METADATA']:]:
                        # Ignore comments and blank lines
                        line = line.strip()
                        commentPos = line.find("~")
                        if commentPos >= 0:  # Strip comments
                            line = line[:commentPos]
                        if len(line) == 0:
                            continue

                        data = line.split()
                        if len(data) < 11 or data[10] != ';':
                            print("Link data line not formatted properly:\n '%s'"% line)
                            raise utils.BadFileFormatException

                        # Create link
                        linkID = '('+str(data[0]).strip()+","+str(data[1]).strip()+')'
                        net.linkDict[linkID] = {}
                        net.linkDict[linkID]['length-cap'] = (float(data[2])
                            * np.cbrt(float(data[3])))

                len_links = [net.linkDict[lnk]['length-cap'] for lnk in damaged_links]

                for lnk in damaged_links:
                    if net.linkDict[lnk]['length-cap'] > np.quantile(len_links, 0.95):
                        damaged_dict[lnk] = np.random.gamma(4*2, 7*2, 1)[0]

                    elif net.linkDict[lnk]['length-cap'] > np.quantile(len_links, 0.75):
                        damaged_dict[lnk] = np.random.gamma(7*np.sqrt(2), 7*np.sqrt(2),
                                                            1)[0]

                    elif net.linkDict[lnk]['length-cap'] > np.quantile(len_links, 0.5):
                        damaged_dict[lnk] = np.random.gamma(12, 7, 1)[0]

                    elif net.linkDict[lnk]['length-cap'] > np.quantile(len_links, 0.25):
                        damaged_dict[lnk] = np.random.gamma(10, 7, 1)[0]

                    else:
                        damaged_dict[lnk] = np.random.gamma(6, 7, 1)[0]

                print(f'Experiment number: {rep+1}')

                SCENARIO_DIR = NETWORK_DIR
                ULT_SCENARIO_DIR = os.path.join(SCENARIO_DIR, str(num_broken))
                os.makedirs(ULT_SCENARIO_DIR, exist_ok=True)

                repetitions = get_folders(ULT_SCENARIO_DIR)

                if len(repetitions) == 0:
                    max_rep = -1
                else:
                    num_scenario = [int(i) for i in repetitions]
                    max_rep = max(num_scenario)
                cur_scenario_num = max_rep + 1

                ULT_SCENARIO_REP_DIR = os.path.join(
                    ULT_SCENARIO_DIR, str(cur_scenario_num))

                os.makedirs(ULT_SCENARIO_REP_DIR, exist_ok=True)

            else:
                print('No scenario file available.')
                raise Exception

            # Finalize damaged links into a list to maintain order of links
            damaged_links = list(damaged_dict.keys())

            print('damaged_dict: ', damaged_dict)
            save_dir = ULT_SCENARIO_REP_DIR
            save(save_dir + '/damaged_dict', damaged_dict)


            if before_after:
                net_after, after_eq_tstt, _ = state_after(damaged_links, save_dir)
                net_before, before_eq_tstt, _ = state_before(damaged_links, save_dir)
                plot_nodes_links(save_dir, netg, damaged_links, coord_dict, names=True)
                print(net_name + ' with ' + str(num_broken) + ' broken bridges')
                print('TSTT before disruption: {}, TSTT after disruption: {}'.format(
                    before_eq_tstt,after_eq_tstt))


            elif output_sequences:
                """ builds a dict (link names are keys) that contains attributes:
                duration, importance factor, immediate benefit. Then solves to
                optimality, adds optimal order of repair ahead of attributes (ie. '1'
                if repaired first) """

                memory = {}
                art_link_dict = make_art_links(NETFILE,TRIPFILE,mc_weights,demand_mult)

                net_before, before_eq_tstt, time_net_before = state_before(
                    damaged_links, save_dir, real=True)
                net_after, after_eq_tstt, time_net_after = state_after(damaged_links,
                    save_dir, real=True)
                first_b, bb_time = first_benefit(net_after,damaged_links,after_eq_tstt)
                last_b = last_benefit(net_before, damaged_links, before_eq_tstt)
                wb, bb, swapped_links = safety(last_b, first_b)
                if num_crews==1:
                    upper_bound = (after_eq_tstt - before_eq_tstt) * sum(
                        net_before.damaged_dict.values())
                save(save_dir + '/upper_bound', upper_bound)

                if damaged_dict_preset=='':
                    plot_nodes_links(save_dir, netg, damaged_links, coord_dict,
                                     names=True)

                damaged_attributes = get_attributes(net_before, first_b, last_b,
                        swapped_links)
                opt_obj, opt_soln, opt_elapsed, opt_num_tap, opt_times = brute_force(
                        net_after, after_eq_tstt, before_eq_tstt, num_crews=num_crews)
                for el in range(num_broken):
                    link = opt_soln[el]
                    damaged_attributes[link].insert(0,el+1)

                print('Damaged attributes', damaged_attributes)
                print('Swapped links', swapped_links)

                fname = save_dir + '/damaged_attributes.csv'
                with open(fname, 'w', newline='') as f:
                    writer = csv.writer(f)
                    for link in damaged_attributes:
                        writer.writerow([link] + damaged_attributes[link])

            else:
                memory = {}
                art_link_dict = make_art_links(NETFILE,TRIPFILE,mc_weights,demand_mult)

                net_before, before_eq_tstt, time_net_before = state_before(
                    damaged_links, save_dir, real=True)
                net_after, after_eq_tstt, time_net_after = state_after(damaged_links,
                    save_dir, real=True)
                first_b, bb_time = first_benefit(net_after,damaged_links,after_eq_tstt)
                last_b = last_benefit(net_before, damaged_links, before_eq_tstt)
                wb, bb, swapped_links = safety(last_b, first_b)
                net_before.maxruntime = str(min(math.ceil(bb_time), 2*time_net_after
                                                * len(damaged_dict)))
                net_after.maxruntime = net_before.maxruntime
                if damaged_dict_preset=='':
                    plot_nodes_links(save_dir, netg, damaged_links, coord_dict,
                                     names=True)

                if num_crews==1:
                    start = time.time()
                    upper_bound = (after_eq_tstt - before_eq_tstt) * sum(
                        net_before.damaged_dict.values())
                    UB_res = [upper_bound, time.time()-start, 2]
                    save(save_dir + '/upper_bound', upper_bound)

                    start = time.time()
                    sortedbb = sorted(bb.items(), key=lambda x: x[1], reverse=True)
                    hlower_seq = []
                    hlower_bound = upper_bound
                    curdisc = 0
                    for k, v in sortedbb:
                        hlower_seq.append(k)
                        hlower_bound -= (curdisc * damaged_dict[k])
                        curdisc += v
                        if curdisc > (after_eq_tstt - before_eq_tstt):
                            curdisc = (after_eq_tstt - before_eq_tstt)
                    HLB_res = [hlower_bound, time.time()-start, 2*num_broken+2]

                print('before tstt: {}, after tstt: {}, total (single crew) \
                      duration: {}'.format(before_eq_tstt, after_eq_tstt,
                                           sum(damaged_dict.values())))
                print('Benefit from repairing each link first: ',first_b)
                print('Benefit from repairing each link last: ',last_b)
                print('Swapped links: ',swapped_links)
                if num_crews==1:
                    print('Simple upper bound on obj function (total travel delay): ',
                          upper_bound)
                    print('Heuristic lower bound on obj function (total travel \
                          delay): ', hlower_bound)
                    start = time.time()
                    __, __, __, __ = eval_sequence(net_after, hlower_seq, after_eq_tstt,
                        before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
                    evaluation_time = time.time() - start
                    print('Time to evaluate a sequence: ', evaluation_time)
                else:
                    start = time.time()
                    __, __, __, __ = eval_sequence(net_after, damaged_links, after_eq_tstt,
                        before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
                    evaluation_time = time.time() - start


                #['Simple UB','HLB','FF LB','OPT BF','ML BF','OPT SP','ML SP','OPT IP',
                # 'ML IP','DeltaTSTT','Arc Flow','Minspan','IC Assign','LASR','LAFO',
                # 'AltLASR','SQG','LZG','IF','SPT','Sim Anneal','Beam Search']

                if num_crews == 1:
                    results = pd.DataFrame(columns=['Objective','Run Time','# TAP'])
                    res_seqs = pd.DataFrame(columns=range(1,num_broken+1))
                else:
                    results = pd.DataFrame(columns=['Objective','Run Time','# TAP',
                                                    'Make Span'])
                    res_seqs = pd.DataFrame(columns=range(1,num_crews+1))

                pointer = 0
                if num_crews==1 and alt_crews is None:
                    results.loc[methods[0]] = UB_res
                    results.loc[methods[1]] = HLB_res
                    pointer += 2
                if methods[pointer]=='FF LB':
                    record_ff_net(net_after)
                    test_net = deepcopy(net_after)
                    test_net.free_flow = True
                    vec_FF, FF_TSTTs_time, TSTT_num_tap = calc_all_TSTT(test_net)
                    ff_sp_res, res_seqs.loc[methods[pointer]], ff_sp_times = opt_sp(3,
                        test_net, after_eq_tstt, before_eq_tstt, FF_TSTTs_time, vec_FF)
                    ff_sp_res[2] += TSTT_num_tap
                    results.loc[methods[pointer]] = ff_sp_res
                    pointer += 1

                memory1 = deepcopy(memory) # takes a snapshot of memory
                if methods[pointer]=='OPT BF': # find optimal sequence by enumeration
                    opt_res, res_seqs.loc[methods[pointer]], opt_times = brute_force(
                        net_after, after_eq_tstt, before_eq_tstt, num_crews=num_crews)
                    results.loc[methods[pointer]] = opt_res
                    #if opt_res[0] < hlower_bound:
                    #    num_lb_fail += 1
                    #lb_gap.append(opt_res[0] - hlower_bound)
                    #lb_percent.append((opt_res[0] - hlower_bound)/opt_res[0])
                    memory = deepcopy(memory1)
                    pointer += 1
                if methods[pointer]=='ML BF': # enumeration using values from ML model
                    ML_mem = {}
                    ML_start = time.time()
                    model, meany, stdy, Z_bar, ML_num_tap = ML_preprocess(damaged_links,
                                                                          net_after)
                    approx_params = (model, meany, stdy)
                    ML_time = time.time() - ML_start + 2*bb_time
                    print('Time to train ML model: '+str(ML_time))
                    opt_res, opt_seq, opt_times = brute_force(net_after, after_eq_tstt,
                        before_eq_tstt, is_approx=True, num_crews=num_crews)
                    opt_res[1] += ML_time
                    opt_res[2] += ML_num_tap
                    memory = deepcopy(memory1)
                    results.loc[methods[pointer]] = opt_res
                    res_seqs.loc[methods[pointer]] = opt_seq
                    pointer += 1

                if methods[pointer]=='OPT SP': # SP formulation using calculated TSTTs
                    vecTSTT, TSTTs_time, TSTT_num_tap = calc_all_TSTT(net_after)
                    opt_sp_res, res_seqs.loc[methods[pointer]], opt_sp_times = opt_sp(1,
                        net_after, after_eq_tstt, before_eq_tstt, TSTTs_time, vecTSTT)
                    opt_sp_res[2] += TSTT_num_tap
                    if opt_sp_res[0] < hlower_bound:
                        num_lb_fail += 1
                    lb_gap.append(opt_sp_res[0] - hlower_bound)
                    lb_percent.append((opt_sp_res[0] - hlower_bound)/opt_sp_res[0])
                    results.loc[methods[pointer]] = opt_sp_res
                    pointer += 1
                if methods[pointer]=='ML SP': # SP formulation using ML TSTT values
                    if 'ML BF' not in methods:
                        ML_start = time.time()
                        model, meany, stdy, Z_bar, ML_num_tap = ML_preprocess(
                            damaged_links, net_after)
                        approx_params = (model, meany, stdy)
                        ML_time = time.time() - ML_start + 2*bb_time
                        print('Time to train ML model: '+str(ML_time))
                    vec_ML, ML_TSTTs_time = calc_all_ML(approx_params, damaged_links,
                                                        memory)
                    print('Time to find all ML TSTTs after training: '
                          + str(ML_TSTTs_time - ML_time))
                    ml_sp_res, res_seqs.loc[methods[pointer]], ml_sp_times = opt_sp(2,
                        net_after, after_eq_tstt, before_eq_tstt, ML_TSTTs_time, vec_ML)
                    ml_sp_res[2] += ML_num_tap
                    memory = deepcopy(memory1)
                    results.loc[methods[pointer]] = ml_sp_res
                    pointer += 1

                if methods[pointer]=='OPT IP': # IP formulation using calculated TSTTs
                    if 'OPT SP' not in methods:
                        vecTSTT, TSTTs_time, TSTT_num_tap = calc_all_TSTT(net_after)
                    mip1_res, res_seqs.loc[methods[pointer]] = mip_bf(1, net_before,
                        after_eq_tstt, before_eq_tstt, TSTTs_time, vecTSTT)
                    mip1_res[2] += TSTT_num_tap
                    results.loc[methods[pointer]] = mip1_res
                    pointer += 1
                if methods[pointer]=='ML IP': # IP formulation using ML TSTT values
                    if 'ML BF' not in methods and 'ML SP' not in methods:
                        ML_start = time.time()
                        model, meany, stdy, Z_bar, ML_num_tap = ML_preprocess(
                            damaged_links, net_after)
                        approx_params = (model, meany, stdy)
                        ML_time = time.time() - ML_start + 2*bb_time
                        print('Time to train ML model: '+str(ML_time))
                    if 'ML SP' not in methods:
                        vec_ML, ML_TSTTs_time = calc_all_ML(approx_params,
                            damaged_links, memory)
                    print('Time to find all ML TSTTs after training: '
                          + str(ML_TSTTs_time - ML_time))
                    mip2_res, res_seqs.loc[methods[pointer]] = mip_bf(2, net_before,
                        after_eq_tstt, before_eq_tstt, ML_TSTTs_time, vec_ML)
                    mip2_res[2] += ML_num_tap
                    memory = deepcopy(memory1)
                    results.loc[methods[pointer]] = mip2_res
                    pointer += 1
                if methods[pointer]=='DeltaTSTT':
                    # alternate IP formulation using estimated deltaTSTT[t,b]
                    preprocess_st = time.time()
                    deltaTSTT, pre_num_tap = find_deltaTSTT(damaged_links, net_after,
                        after_eq_tstt, before_eq_tstt, last_b, first_b)
                    time_mip_before = time.time() - preprocess_st + 2*bb_time
                    mip3_res, res_seqs.loc[methods[pointer]] = mip_delta(net_before,
                        after_eq_tstt, before_eq_tstt, time_mip_before, deltaTSTT)
                    mip3_res[2] += pre_num_tap
                    memory = deepcopy(memory1)
                    results.loc[methods[pointer]] = mip3_res
                    pointer += 1

                if methods[pointer].find('Arc Flow') >= 0:
                    preprocess_st = time.time()
                    Z_bar, alt_Z_bar, preprocessing_num_tap = find_approx(approx,
                        damaged_links, net_after, last_b, first_b)
                    time_before = time.time() - preprocess_st + 2*bb_time
                    arc_res, arc_seq = arc_flow(net_before, after_eq_tstt, before_eq_tstt,
                                              time_before, Z_bar, first_b)
                    arc_res[2] += preprocessing_num_tap
                    results.loc[methods[pointer]] = arc_res
                    res_seqs.loc[methods[pointer]] = arc_seq
                    pointer += 1

                if methods[pointer].find('Minspan') >= 0:
                    minspan_res, minspan_seq = find_decomp('minmake', mdecomp,
                        net_before, net_after, after_eq_tstt, before_eq_tstt,
                        time_net_before, num_crews)
                    for i in range(len(mdecomp)):
                        results.loc[methods[pointer]] = minspan_res[i]
                        res_seqs.loc[methods[pointer]] = minspan_seq[i]
                        pointer += 1
                    memory = deepcopy(memory1)
                if methods[pointer].find('IC Assign') >= 0:
                    ic_res, ic_seq = find_decomp('icassign', idecomp, net_before,
                        net_after, after_eq_tstt, before_eq_tstt, time_net_before,
                        num_crews)
                    for i in range(len(idecomp)):
                        results.loc[methods[pointer]] = ic_res[i]
                        res_seqs.loc[methods[pointer]] = ic_seq[i]
                        pointer += 1
                    memory = deepcopy(memory1)

                if methods[pointer]=='LASR': # Largest Average Smith Ratio
                    if not arc:
                        preprocess_st = time.time()
                        Z_bar, alt_Z_bar, preprocessing_num_tap = find_approx(approx,
                            damaged_links, net_after, last_b, first_b)
                        time_before = time.time() - preprocess_st + 2*bb_time
                    LASR_res, LASR_seq = LASR(net_before, after_eq_tstt, before_eq_tstt,
                                              time_before, Z_bar)
                    LASR_res[2] += preprocessing_num_tap
                    results.loc[methods[pointer]] = LASR_res
                    res_seqs.loc[methods[pointer]] = LASR_seq
                    pointer += 1
                if methods[pointer]=='LAFO': # Largest Average First Order
                    LAFO_res, LAFO_seq = LAFO(net_before, after_eq_tstt, before_eq_tstt,
                                              time_before, Z_bar)
                    LAFO_res[2] += preprocessing_num_tap
                    results.loc[methods[pointer]] = LAFO_res
                    res_seqs.loc[methods[pointer]] = LAFO_seq
                    pointer += 1
                    bfs_LA = BestSoln()
                    if LAFO_res[0] < LASR_res[0]:
                        bfs_LA.cost, bfs_LA.path = LAFO_res[0], LAFO_seq
                    else:
                        bfs_LA.cost, bfs_LA.path = LASR_res[0], LASR_seq

                if methods[pointer]=='AltLASR': # Modified LASR
                    altLASR_res, altLASR_seq, = altLASR(net_before, after_eq_tstt,
                        before_eq_tstt, time_before, alt_Z_bar)
                    altLASR_res[2] += preprocessing_num_tap
                    memory = deepcopy(memory1)
                    results.loc[methods[pointer]] = altLASR_res
                    res_seqs.loc[methods[pointer]] = altLASR_seq
                    pointer += 1

                if methods[pointer]=='SQG': # Sequential greedy method
                    wb_update, bb_update = deepcopy(wb), deepcopy(bb)
                    if num_crews == 1:
                        greedy_res, greedy_seq = greedy_heuristic(net_after,
                            after_eq_tstt, before_eq_tstt, time_net_before,
                            time_net_after)
                    else:
                        greedy_res, greedy_seq = greedy_heuristic_mult(net_after,
                            after_eq_tstt, before_eq_tstt, time_net_before,
                            time_net_after, num_crews)
                    bfs = BestSoln()
                    bfs.cost, bfs.path = greedy_res[0], greedy_seq
                    wb, bb = deepcopy(wb_update), deepcopy(bb_update)
                    wb_orig, bb_orig = deepcopy(wb), deepcopy(bb)
                    results.loc[methods[pointer]] = greedy_res
                    res_seqs.loc[methods[pointer]] = greedy_seq
                    pointer += 1
                if methods[pointer]=='LZG': # Lazy greedy method
                    (results.loc[methods[pointer]], res_seqs.loc[methods[pointer]]
                     ) = lazy_greedy_heuristic(net_after, after_eq_tstt, before_eq_tstt,
                        first_b, bb_time)
                    pointer += 1

                if methods[pointer]=='IF': # Importance factor
                    importance_res, importance_seq = importance_factor_solution(
                        net_before, after_eq_tstt, before_eq_tstt, time_net_before)
                    if importance_res[0] < bfs.cost:
                        bfs.cost = importance_res[0]
                        bfs.path = importance_seq
                    results.loc[methods[pointer]] = importance_res
                    res_seqs.loc[methods[pointer]] = importance_seq
                    try:
                        if importance_res[0] < bfs_LA.cost:
                            bfs_LA.cost = importance_res[0]
                            bfs_LA.path = importance_seq
                    except:
                        pass
                    pointer += 1
                if methods[pointer]=='SPT': # Shortest processing time
                    (results.loc[methods[pointer]], res_seqs.loc[methods[pointer]]
                     ) = SPT_solution(net_before, after_eq_tstt, before_eq_tstt,
                        time_net_before)
                    pointer += 1

                memory2 = deepcopy(memory) # memory snapshot including states from SQG
                if pointer < len(methods) and methods[pointer].find('Sim Anneal') >= 0:
                    for i in range(sa):
                        (results.loc[methods[pointer]],
                            res_seqs.loc[methods[pointer]]) = sim_anneal(100, i,
                            bfs, net_after, after_eq_tstt, before_eq_tstt,
                            num_crews=num_crews)
                        memory = deepcopy(memory2)
                        pointer += 1
                if pointer < len(methods) and methods[pointer].find('SA Method') >= 0:
                    for el in sacompare:
                        for i in range(sa):
                            if num_crews != 1 and str(el)[2] == 1:
                                memory = deepcopy(memory1)
                            (results.loc[methods[pointer]],
                                res_seqs.loc[methods[pointer]]) = sim_anneal(el,
                                i, bfs, net_after, after_eq_tstt, before_eq_tstt,
                                num_crews=num_crews)
                            memory = deepcopy(memory2)
                            pointer += 1

                if pointer < len(methods) and methods[pointer]=='Beam Search':
                    k, gamma, beta = 2, 128, 128
                    r_algo_num_tap = greedy_res[2] + num_broken - 1
                    search_start = time.time()
                    start_node, end_node = get_se_nodes(damaged_dict, after_eq_tstt,
                        before_eq_tstt, relax=True)
                    if graphing:
                        (r_algo_path, r_algo_obj, r_search_tap_solved, tot_childr,
                            uncommon_numberr, common_numberr, num_purger, wb, bb,
                            wb_update, bb_update, memory) = search(net_after,
                            after_eq_tstt, before_eq_tstt, start_node, end_node, bfs,
                            wb, bb, wb_update, bb_update, memory, beam_search=
                            beam_search, beam_k=k, beta=beta, gamma=gamma, graphing=
                            graphing, bs_time_list=bs_time_list,bs_OBJ_list=bs_OBJ_list)
                    else:
                        (r_algo_path, r_algo_obj, r_search_tap_solved, tot_childr,
                            uncommon_numberr, common_numberr, num_purger, wb, bb,
                            wb_update, bb_update, memory) = search(net_after,
                            after_eq_tstt, before_eq_tstt, start_node, end_node, bfs,
                            wb, bb, wb_update, bb_update, memory, beam_search=
                            beam_search, beam_k=k, beta=beta, gamma=gamma)
                    search_elapsed = time.time() - search_start
                    if graphing:
                        bs_time_list.append(search_elapsed)
                        bs_OBJ_list.append(r_algo_obj)
                    r_algo_num_tap += r_search_tap_solved
                    r_algo_elapsed = (search_elapsed + greedy_res[1] + importance_res[1]
                                      + 2*evaluation_time)
                    if num_crews==1:
                        results.loc[methods[pointer]] = [r_algo_obj, r_algo_elapsed,
                                                     r_algo_num_tap,]
                        res_seqs.loc[methods[pointer]] = r_algo_path
                    else:
                        bs_crew_seqs, bs_make_span = gen_crew_seqs(r_algo_path, damaged_dict,
                                                             num_crews)
                        results.loc[methods[pointer]] = [r_algo_obj, r_algo_elapsed,
                                                     r_algo_num_tap, bs_make_span]
                        res_seqs.loc[methods[pointer]] = bs_crew_seqs
                    pointer += 1


                if mdecomp:
                    temp = {1:'(global opt)', 2:'(local opt)', 3:'(greedy)', 4:'(IF)',
                            5:'(SPT)'}
                    moveloc = list(results.index).index('Minspan '+temp[mdecomp[0]])
                elif idecomp:
                    temp = {1:'(global opt)', 2:'(local opt)', 3:'(greedy)', 4:'(IF)',
                            5:'(SPT)'}
                    moveloc = list(results.index).index('Minmake '+temp[idecomp[0]])
                elif approx:
                    moveloc = list(results.index).index('LASR')
                else:
                    moveloc = list(results.index).index('SQG')
                if beam_search:
                    line = results.loc['Beam Search':'Beam Search']
                    results = pd.concat([results[:moveloc],line[:],
                                         results[moveloc:len(results)-1]])
                    line = res_seqs.loc['Beam Search':'Beam Search']
                    if num_crews==1 and alt_crews is None:
                        res_seqs = pd.concat([res_seqs[:moveloc-2],line[:],
                                             res_seqs[moveloc-2:len(res_seqs)-1]])
                    else:
                        res_seqs = pd.concat([res_seqs[:moveloc],line[:],
                                             res_seqs[moveloc:len(res_seqs)-1]])
                    moveloc += 1
                if sa==1 and not sacompare:
                    line = results.loc['Sim Anneal':'Sim Anneal']
                    results = pd.concat([results[:moveloc],line[:],
                                         results[moveloc:len(results)-1]])
                    line = res_seqs.loc['Sim Anneal':'Sim Anneal']
                    if num_crews==1 and alt_crews is None:
                        res_seqs = pd.concat([res_seqs[:moveloc-2],line[:],
                                          res_seqs[moveloc-2:len(res_seqs)-1]])
                    else:
                        res_seqs = pd.concat([res_seqs[:moveloc],line[:],
                                          res_seqs[moveloc:len(res_seqs)-1]])

                if alt_crews is None and not multiclass:
                    print(results)
                    print(res_seqs)
                    results.to_pickle(save_dir + '/results' + extension)
                    res_seqs.to_pickle(save_dir + '/res_seqs' + extension)
                    fname = save_dir + '/results.csv'
                    with open(fname, 'w', newline='') as f:
                        f.write(results.to_csv(header=False))
                    fname = save_dir + '/res_seqs.csv'
                    with open(fname, 'w', newline='') as f:
                        f.write(res_seqs.to_csv(header=False))
                elif multiclass and isinstance(net_after.tripfile, list):
                    class_results = results.copy()
                    test_net = deepcopy(net_after)
                    num_classes = len(net_after.tripfile)
                    for i in range(1,num_classes+1):
                        class_results.insert(i,'Class '+str(i),np.nan)
                    for method in methods:
                        if method=='Simple UB' or method=='HLB':
                            continue
                        temp, __, __, __ = eval_sequence(test_net,
                            list(res_seqs.loc[method]), after_eq_tstt, before_eq_tstt,
                            num_crews=num_crews, multiclass=multiclass)
                        class_results.at[method,'Objective'] = temp[0]
                        for i in range(1,len(temp)):
                            class_results.at[method,'Class '+str(i)] = temp[i]
                    print(class_results)
                    print(res_seqs)
                    class_results.to_pickle(save_dir + '/results' + extension)
                    res_seqs.to_pickle(save_dir + '/res_seqs' + extension)
                    fname = save_dir + '/results.csv'
                    with open(fname, 'w', newline='') as f:
                        f.write(class_results.to_csv(header=False))
                    fname = save_dir + '/res_seqs.csv'
                    with open(fname, 'w', newline='') as f:
                        f.write(res_seqs.to_csv(header=False))
                else:
                    crew_results = results.copy()
                    crew_res_seqs = res_seqs.copy()
                    test_net = deepcopy(net_after)
                    for i in range(len(alt_crews)):
                        crew_results.insert(i+1,'PP'+str(alt_crews[i]),np.nan)
                    crew_results.insert(1,'PP'+str(num_crews),np.nan)
                    for i in range(len(alt_crews)):
                        crew_results.insert(i+1,'C'+str(alt_crews[i]),np.nan)
                    crew_results.insert(len(crew_results.columns),'Make Span',
                                        np.nan)
                    for method in methods:
                        if method=='Simple UB' or method=='HLB':
                            continue
                        elif method.find('OPT') >= 0:
                            # df numbering starts at 1
                            locopt = list(crew_results.index).index(method)+1
                            for i in range(len(alt_crews)):
                                memory = deepcopy(memory1)
                                match method:
                                    case 'OPT BF':
                                        temp_res, temp_seq, temp_times = brute_force(
                                            net_after, after_eq_tstt, before_eq_tstt,
                                            num_crews=alt_crews[i])
                                    case 'OPT SP':
                                        temp_res, temp_seq, temp_times = opt_sp(1,
                                            net_after, after_eq_tstt, before_eq_tstt,
                                            TSTTs_time, vec_TSTT)
                                    case 'OPT IP':
                                        temp_res, temp_seq, temp_times = mip_bf(1,
                                            net_after, after_eq_tstt, before_eq_tstt,
                                            TSTTs_time, vec_TSTT)
                                for j in range(len(alt_crews)+1):
                                    if j < i+1:
                                        temp_res.insert(0,np.nan)
                                    if j > i+1:
                                        temp_res.insert(j,np.nan)
                                for j in range(len(alt_crews)+1):
                                    temp_res.insert(len(alt_crews)+1,np.nan)
                                line = pd.DataFrame(columns=crew_results.columns)
                                line.loc[method+str(alt_crews[i])] = temp_res
                                crew_results = pd.concat([crew_results[:locopt+i],
                                    line.loc[:], crew_results[locopt+i:]])
                                #line = pd.DataFrame(columns=res_seqs.columns)
                                #line.loc[method+str(alt_crews[i])] = temp_seq
                                #crew_res_seqs = pd.concat([crew_res_seqs[:locopt+i],
                                #    line.loc[:], crew_res_seqs[locopt+i:]])
                        else:
                            for crew in alt_crews:
                                (crew_results.at[method,'PP'+str(crew)], __, __, __
                                 ) = eval_sequence(test_net, list(res_seqs.loc[method]),
                                    after_eq_tstt, before_eq_tstt, num_crews=crew)
                    print(crew_results)
                    print(res_seqs)
                    crew_results.to_pickle(save_dir + '/results' + extension)
                    res_seqs.to_pickle(save_dir + '/res_seqs' + extension)
                    fname = save_dir + '/results.csv'
                    with open(fname, 'w', newline='') as f:
                        f.write(crew_results.to_csv(header=False))
                    fname = save_dir + '/res_seqs.csv'
                    with open(fname, 'w', newline='') as f:
                        f.write(res_seqs.to_csv(header=False))

        if multiclass and damaged_dict_preset != '':
            # Read unequal class weighting soln paths, and calc obj for same paths,
            # but equally weighted
            seq_ineq = pd.read_pickle(damaged_dict_preset + '/res_seqs' + extension)
            compare = class_results.drop(['Run Time','# TAP'],axis=1)
            compare = compare.merge(compare, left_index=True, right_index=True,
                               suffixes=(' (EqualPri)', ' (UneqPri)'))
            compare.insert(len(compare.columns),'% Overall Change',[0]*len(compare))
            for i in range(1,num_classes+1):
                compare.insert(len(compare.columns),'% Change Class '+str(i),
                               [0]*len(compare))

            for method in methods:
                if method=='Simple UB' or method=='HLB':
                    continue
                if method=='FF LB':
                    test_net = deepcopy(net_after)
                    test_net.free_flow = True
                    temp, __, __, __ = eval_sequence(test_net,
                        list(seq_ineq.loc[method]), after_eq_tstt, before_eq_tstt,
                        num_crews=num_crews, multiclass=multiclass)
                else:
                    temp, __, __, __ = eval_sequence(net_after,
                        list(seq_ineq.loc[method]), after_eq_tstt, before_eq_tstt,
                        num_crews=num_crews, multiclass=multiclass)
                compare.at[method,'Objective'+' (UneqPri)'] = temp[0]
                compare.at[method,'% Overall Change'] = percentChange(
                    compare.at[method,'Objective'+' (EqualPri)'], temp[0])
                for i in range(1,len(temp)):
                    compare.at[method,'Class '+str(i)+' (UneqPri)'] = temp[i]
                    compare.at[method,'% Change Class '+str(i)] = percentChange(
                        compare.at[method,'Class '+str(i)+' (EqualPri)'], temp[i])
            print(compare)
            compare.to_pickle(save_dir + '/compare' + extension)
            fname = save_dir + '/compare.csv'
            with open(fname, 'w', newline='') as f:
                f.write(compare.to_csv(header=False))

            if graphing:
                pass
                #get_sequence_graphs(NETWORK_DIR, str(num_broken), alt_dir=
                #    ULT_SCENARIO_REP_DIR, multiclass=True, mc_weights=mc_weights)

        elif graphing:
            pass
            #get_sequence_graphs(NETWORK_DIR, str(num_broken), mc_weights=mc_weights)

        if graphing:
            if reps < 6:
                plot_time_OBJ(save_dir, 1, reps, bs_time_list, bs_OBJ_list,sa_time_list,
                              sa_OBJ_list)
            else:
                plot_time_OBJ(save_dir, 1, math.ceil(reps/2), bs_time_list, bs_OBJ_list,
                              sa_time_list, sa_OBJ_list)
                plot_time_OBJ(save_dir, math.ceil(reps/2)+1, reps, bs_time_list,
                              bs_OBJ_list, sa_time_list, sa_OBJ_list)

        #print('hlower_bound was invalid {} times out of {}'.format(num_lb_fail, reps))
        #print(lb_gap)
        #print(lb_percent)
        #save(save_dir + '/hlowerinvalid', [num_lb_fail,reps])
