from graphing import *
from correspondence import *
from search import search
from search import get_se_nodes

import os.path
from ctypes import *
import random
import operator as op
from functools import reduce
import itertools
from scipy.special import comb

import csv
from prettytable import PrettyTable
from prettytable import MSWORD_FRIENDLY
from matplotlib import collections as mc
from matplotlib import patches as mpatch

import math
import argparse
import cProfile
import pstats
import networkx as nx
import progressbar
# import caffeine
from network import *


extension = '.pickle'

parser = argparse.ArgumentParser(description='find an order for repairing bridges')
parser.add_argument('-n', '--net_name', type=str, help='network name')
parser.add_argument('-b', '--num_broken', type=int, help='number of broken bridges')
parser.add_argument('-a', '--approx', type=int, help=('approximation methods enabled - LAFO/LASR'
                    +'1: display only min of LAFO/LASR, 2: display both, 3: also display altLASR'),
                    default=0)
parser.add_argument('-r', '--reps', type=int, help='number of scenarios with the given parameters',
                    default=5)
parser.add_argument('-t', '--tables', type=bool, help='table output mode', default=False)
parser.add_argument('-g', '--graphing', type=bool,
                    help='save results graphs and solution quality vs runtime graph', default=False)
parser.add_argument('-l', '--loc', type=int, help='location based sampling', default=3)
parser.add_argument('-o', '--scenario', type=str, help='scenario file', default='scenario.csv')
parser.add_argument('-e', '--strength', type=str, help='strength of the earthquake')
parser.add_argument('-y', '--onlybeforeafter', type=bool, help='to get before and after tstt',
                    default=False)
parser.add_argument('-z', '--output_sequences', type=bool,
                    help='to get sequences and params to csv from net/#', default=False)
parser.add_argument('-f', '--full', type=bool, help='to use full algorithm not beam search',
                    default=False)
parser.add_argument('-s', '--beamsearch',
                    help='use beam search for speed, to disable, enter -s without arguments',
                    action='store_false')
parser.add_argument('-j', '--random',
                    help='generate scenarios randomly, to disable, enter -j without arguments',
                    action='store_false')
parser.add_argument('-c', '--gamma', type=int, help='hyperparameter to expand the search',
                    default=128)
parser.add_argument('-v', '--beta', type=int,
                    help='hyperparameter that decides the frequency of purge', default=128)
parser.add_argument('-d', '--num_crews', nargs='+', type=int, help='number of work crews available',
                    default=1)
""" when multiple values are given for num_crews, order is found based
on first number, and postprocessing is performed to find OBJ for other
crew numbers """
parser.add_argument('--opt', type=int,
                    help='1: brute force (opt), 2: brute force using ML values', default=0)
parser.add_argument('--mip', type=int,
                    help=('1: opt w/ precalced TSTTs, 2: ML TSTT values, 3: estimated ' +
                    'deltaTSTT[t,b] values, 4: deltaTSTT[t,b] values'),
                    default=0) # uses Girobi solver, need license to use, coded for single-crew
parser.add_argument('--sa', type=bool, help='solve using simulated annealing starting at bfs',
                    default=False)
parser.add_argument('--damaged', type=str, help='set damaged_dict to previously stored values',
                    default='')
parser.add_argument('--mc', type=bool, help='display separate TSTTs for each class of demand',
                    default=False)
parser.add_argument('-w', '--mc_weights', nargs='+', type=int,
                    help='TSTT weights for each class of demand', default=1)
parser.add_argument('--demand', type=float, help='demand multiplier', default=1)
args = parser.parse_args()

SEED = 42
FOLDER = 'TransportationNetworks'
MAX_DAYS = 180
MIN_DAYS = 21
memory = {}

CORES = min(mp.cpu_count(),4)

class BestSoln():
    """A node class for bi-directional search for pathfinding"""
    def __init__(self):
        self.cost = None
        self.path = None


def eval_working_sequence(
        net, order_list, after_eq_tstt, before_eq_tstt, is_approx=False, num_crews=1,
        approx_params=None):
    """evaluates the total tstt for a repair sequence, using memory to minimize runtime"""
    tap_solved = 0
    days_list = []
    tstt_list = []
    global memory
    global ML_mem

    to_visit = order_list
    added = []

    # Crew order list is the order in which projects complete
    crew_order_list, which_crew, days_list = gen_crew_order(
        order_list, damaged_dict=net.damaged_dict, num_crews=num_crews)

    for link_id in crew_order_list:
        added.append(link_id)
        not_fixed = set(to_visit).difference(set(added))
        net.not_fixed = set(not_fixed)

        if is_approx:
            if frozenset(net.not_fixed) in ML_mem.keys():
                tstt_after = ML_mem[frozenset(net.not_fixed)]
            else:
                damaged_links = list(net.damaged_dict.keys())
                state = list(set(damaged_links).difference(net.not_fixed))
                state = [damaged_links.index(i) for i in state]
                pattern = np.zeros(len(damaged_links))
                pattern[(state)] = 1
                tstt_after = approx_params[0].predict(pattern.reshape(1, -1),
                    verbose=0) * approx_params[2] + approx_params[1]
                tstt_after = tstt_after[0][0]
                ML_mem[frozenset(net.not_fixed)] = tstt_after
        else:
            if frozenset(net.not_fixed) in memory.keys():
                tstt_after = memory[frozenset(net.not_fixed)]
            else:
                tstt_after = solve_UE(net=net, eval_seq=True)
                memory[frozenset(net.not_fixed)] = tstt_after
                tap_solved += 1

        tstt_list.append(tstt_after)

    tot_area = 0
    for i in range(len(days_list)):
        if i == 0:
            tstt = after_eq_tstt
        else:
            tstt = tstt_list[i - 1]
        tot_area += (tstt - before_eq_tstt) * days_list[i]

    return tot_area, tap_solved, tstt_list


def last_benefit(
        before, links_to_remove, before_eq_tstt, relax=False, bsearch=False, ext_name=''):
    """builds last benefits dict by finding benefit of repairing each link last"""
    if relax:
        ext_name = '_relax'
    elif bsearch:
        ext_name = ext_name
    fname = before.save_dir + '/last_benefit_dict' + ext_name

    # For each bridge, find the effect on TSTT when that bridge is removed while keeping others
    if not os.path.exists(fname + extension):
        last_b = {}
        not_fixed = []
        for link in links_to_remove:
            test_net = deepcopy(before)
            not_fixed = [link]
            test_net.not_fixed = set(not_fixed)

            tstt = solve_UE(net=test_net, eval_seq=True)
            global memory
            memory[frozenset(test_net.not_fixed)] = tstt
            last_b[link] = tstt - before_eq_tstt
        save(fname, last_b)
    else:
        last_b = load(fname)

    return last_b


def first_benefit(after, links_to_remove, after_eq_tstt, relax=False, bsearch=False, ext_name=''):
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

            tstt_after = solve_UE(net=test_net, eval_seq=True)
            global memory
            memory[frozenset(test_net.not_fixed)] = tstt_after
            first_b[link] = after_eq_tstt - tstt_after
        elapsed = time.time() - start
        save(fname, first_b)
    else:
        first_b = load(fname)

    return first_b, elapsed


def state_after(damaged_links, save_dir, relax=False, real=False, bsearch=False, ext_name=''):
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
        net_after = create_network(NETFILE, TRIPFILE, mc_weights=mc_weights, demand_mult=demand_mult)
        net_after.not_fixed = set(damaged_links)
        net_after.art_links = art_link_dict
        net_after.damaged_dict = damaged_dict

        after_eq_tstt = solve_UE(net=net_after, eval_seq=True, warm_start=False, initial=True)
        if isinstance(mc_weights, list):
            test_net = deepcopy(net_after)
            test_net.mc_weights = 1
            after_eq_tstt_mcunw = solve_UE(net=test_net, eval_seq=True, warm_start=False,
                multiclass=True)

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
        damaged_links, save_dir, relax=False, real=False, bsearch=False, ext_name='',
        mc_weights=1):
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
        net_before = create_network(NETFILE, TRIPFILE, mc_weights=mc_weights, demand_mult=demand_mult)
        net_before.not_fixed = set([])
        net_before.art_links = art_link_dict
        net_before.damaged_dict = damaged_dict

        if isinstance(mc_weights, list):
            test_net = deepcopy(net_before)
            test_net.mc_weights = 1
            before_eq_tstt_mcunw = solve_UE(net=test_net, eval_seq=True, warm_start=False,
                multiclass=True)
        before_eq_tstt = solve_UE(net=net_before, eval_seq=True, warm_start=False, flows=True,
            initial=True)

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
            swapped_links[a_link] = first_b[a_link] - last_b[a_link]
        else:
            wb[a_link] = last_b[a_link]
            bb[a_link] = first_b[a_link]
    return wb, bb, swapped_links


def eval_state(state, after, damaged_links, eval_seq=False):
    """finds tstt for a repair state (checks memory first for cached solution)"""
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
        tstt_after = solve_UE(net=test_net, eval_seq=eval_seq)
        memory[frozenset(test_net.not_fixed)] = tstt_after
        num_tap += 1

    return tstt_after, num_tap


def find_approx(approx, damaged_links, net_after, last_b, first_b):
    """uses sampling as described in Rey et al. 2019 to find approximated
    average first-order effects. If approx==3, uses exact first and last
    effects, and samples for middle values only"""
    X_train = []
    y_train = []
    Z_train = [0]*len(damaged_links)
    for i in range(len(damaged_links)):
        Z_train[i] = []
    if approx==3:
        alt_Z_train = deepcopy(Z_train)

    print('first benefits: {}, last benefits: {}'.format(first_b,last_b))

    preprocessing_num_tap = 0
    damaged_links = [i for i in damaged_links]

    for k, v in memory.items():
        pattern = np.ones(len(damaged_links))
        state = [damaged_links.index(i) for i in k]
        pattern[(state)] = 0
        X_train.append(pattern)
        y_train.append(v)
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
                TSTT, tap = eval_state(temp_state, net_after, damaged_links, eval_seq=True)
                preprocessing_num_tap += tap
                X_train.append(pattern)
                y_train.append(TSTT)

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
                    new_TSTT, tap = eval_state(new_state, net_after, damaged_links,
                        eval_seq=True)
                    preprocessing_num_tap += tap
                    X_train.append(new_pattern)
                    y_train.append(new_TSTT)
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


def find_deltaTSTT(damaged_links, net_after, after_eq_tstt, before_eq_tstt, last_b, first_b):
    """uses sampling adapted from that described in Rey et al. 2019 to find
    approximated average first-order effects for EACH position in the repair
    sequence"""
    X_train = []
    y_train = []
    Z_train = [0]*len(damaged_links)
    for i in range(len(damaged_links)):
        Z_train[i] = []
    alt_Z_train = [0]*(len(damaged_links)-2)
    for i in range(len(damaged_links)-2):
        alt_Z_train[i] = deepcopy(Z_train)
    print('best benefits: {}, worst benefits: {}'.format(bb,wb))

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

            TSTT, tap = eval_state(temp_state, net_after, damaged_links, eval_seq=True)
            preprocessing_num_tap += tap
            X_train.append(pattern)
            y_train.append(TSTT)

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
                    if any((new_pattern is test) or (new_pattern == test).all() for test in
                            X_train):
                        try:
                            new_TSTT = Y_train[X_train.index(new_pattern)]
                            Z_train[el].append(abs(new_TSTT - TSTT))
                            alt_Z_train[stage-1][el].append(abs(new_TSTT - TSTT))
                        except:
                            pass
                    else:
                        new_TSTT, tap = eval_state(new_state, net_after, damaged_links,
                            eval_seq=True)
                        preprocessing_num_tap += tap
                        X_train.append(new_pattern)
                        y_train.append(new_TSTT)
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

                if any((pattern is test) or (pattern == test).all() for test in X_train):
                    try:
                        TSTT = Y_train[X_train.index(pattern)]
                    except:
                        TSTT, tap = eval_state(temp_state, net_after, damaged_links, eval_seq=True)
                        preprocessing_num_tap += tap
                        X_train.append(pattern)
                        y_train.append(TSTT)
                else:
                    TSTT, tap = eval_state(temp_state, net_after, damaged_links, eval_seq=True)
                    preprocessing_num_tap += tap
                    X_train.append(pattern)
                    y_train.append(TSTT)

                new_pattern = np.zeros(card_P)
                new_state = deepcopy(temp_state)
                new_state.append(damaged_links[el])
                state = [damaged_links.index(i) for i in new_state]
                new_pattern[(state)] = 1
                if any((new_pattern is test) or (new_pattern == test).all() for test in X_train):
                    try:
                        new_TSTT = Y_train[X_train.index(new_pattern)]
                    except:
                        new_TSTT, tap = eval_state(new_state, net_after, damaged_links, eval_seq=True)
                        preprocessing_num_tap += tap
                        X_train.append(new_pattern)
                        y_train.append(new_TSTT)
                else:
                    new_TSTT, tap = eval_state(new_state, net_after, damaged_links, eval_seq=True)
                    preprocessing_num_tap += tap
                    X_train.append(new_pattern)
                    y_train.append(new_TSTT)
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
    y_train = []
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
        y_train.append(v)
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
                TSTT, tap = eval_state(temp_state, net_after, damaged_links, eval_seq=True)
                preprocessing_num_tap += tap
                X_train.append(pattern)
                y_train.append(TSTT)

                for el in range(len(pattern)):
                    new_pattern = np.zeros(len(damaged_links))
                    new_state = deepcopy(temp_state)
                    if pattern[el] == 1:
                        new_state.remove(damaged_links[el])
                    else:
                        new_state.append(damaged_links[el])
                    state = [damaged_links.index(i) for i in new_state]
                    new_pattern[(state)] = 1

                    if any((new_pattern is test) or (new_pattern == test).all() for test in
                            X_train):
                        pass
                    else:
                        new_TSTT, tap = eval_state(new_state, net_after, damaged_links,
                            eval_seq=True)
                        preprocessing_num_tap += tap
                        X_train.append(new_pattern)
                        y_train.append(new_TSTT)
                        Z_train[el].append(abs(new_TSTT - TSTT))

    Z_bar = np.zeros(len(damaged_links))
    for i in range(len(damaged_links)):
        Z_bar[i] = np.mean(Z_train[i])

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from tensorflow import keras

    c = list(zip(X_train, y_train))
    random.shuffle(c)
    X_train, y_train = zip(*c)

    X_train_full = np.array(X_train)
    y_train_full = np.array(y_train)

    meany = np.mean(y_train_full)
    stdy = np.std(y_train_full)
    y_train_full = (y_train_full - meany) / stdy

    cutt = int(X_train_full.shape[0] * 0.1)
    X_train = X_train_full[cutt:]
    y_train = y_train_full[cutt:]
    X_valid = X_train_full[:cutt]
    y_valid = y_train_full[:cutt]

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


    history = model.fit(X_train, y_train, validation_data=(
        X_valid, y_valid), epochs=1000, verbose=0, callbacks=[early_stopping_cb])

    # Tests
    state = random.sample(damaged_links, 1)
    TSTT, tap = eval_state(state, net_after, damaged_links)
    state = [damaged_links.index(i) for i in state]
    pattern = np.zeros(len(damaged_links))
    pattern[(state)] = 1
    predicted_TSTT = model.predict(pattern.reshape(1, -1)) * stdy + meany
    print('Predicted tstt vs real tstt, percent error:', predicted_TSTT[0][0], TSTT,
        (predicted_TSTT[0][0]-TSTT)/TSTT*100)

    state = random.sample(damaged_links, 4)
    TSTT, tap = eval_state(state, net_after, damaged_links)
    state = [damaged_links.index(i) for i in state]
    pattern = np.zeros(len(damaged_links))
    pattern[(state)] = 1
    predicted_TSTT = model.predict(pattern.reshape(1, -1)) * stdy + meany
    print('Predicted tstt vs real tstt, percent error:', predicted_TSTT[0][0], TSTT,
        (predicted_TSTT[0][0]-TSTT)/TSTT*100)

    return model, meany, stdy, Z_bar, preprocessing_num_tap


def sim_anneal(bfs, net_after, after_eq_tstt, before_eq_tstt, damaged_links, num_crews=1):
    """starts at bfs (greedy or importance) and conducts simulated annealling to find solution"""
    start = time.time()
    fname = net_after.save_dir + '/sim_anneal_solution'
    num_runs = 1

    if not os.path.exists(fname + extension):
        print('Finding the simulated annealing solution ...')
        tap_solved = 0
        current = list(deepcopy(bfs.path))
        best_soln = list(deepcopy(bfs.path))
        curcost = deepcopy(bfs.cost)
        best_cost = deepcopy(bfs.cost)
        curnet = deepcopy(net_after)
        if graphing:
            global sa_time_list
            global sa_OBJ_list
            sa_time_list.append(0)
            sa_OBJ_list.append(deepcopy(bfs.cost))

        global memory
        t = 0
        fail = 0
        ratio = 0
        lastMvmt = 0

        if num_crews==1:
            for run in range(num_runs):
                if num_runs > 1:
                    print('Starting simulated annealing run number: ', run+1)
                    if run > 0:
                        current = list(deepcopy(bfs.path))
                        best_soln = list(deepcopy(bfs.path))
                        curcost = deepcopy(bfs.cost)
                        best_cost = deepcopy(bfs.cost)
                        t = 0
                        fail = 0
                        ratio = 0
                        lastMvmt = 0
                    else:
                        final_soln = deepcopy(best_soln)
                        final_cost = deepcopy(best_cost)
                else:
                    final_soln = deepcopy(best_soln)
                    final_cost = deepcopy(best_cost)

                while t < 1.2 * len(current)**3:
                    t += 1
                    idx = random.randrange(0,len(current)-1)
                    nextord = deepcopy(current)
                    el = nextord.pop(idx)
                    swap = nextord[idx]
                    nextord.insert(idx+1,el)

                    # Get tstt before fixing el or swap, then tstt if fixing each first to find
                    # difference in total area
                    startstate = current[:idx]
                    startTSTT, tap = eval_state(startstate, curnet, damaged_links, eval_seq=True)
                    tap_solved += tap

                    endstate = current[:idx+2]
                    endTSTT, tap = eval_state(endstate, curnet, damaged_links, eval_seq=True)
                    tap_solved += tap

                    elstate = current[:idx+1]
                    elTSTT, tap = eval_state(elstate, curnet, damaged_links, eval_seq=True)
                    tap_solved += tap

                    swapstate = nextord[:idx+1]
                    swapTSTT, tap = eval_state(swapstate, curnet, damaged_links, eval_seq=True)
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
                        prob = math.exp(negdelta/curcost*(t**(2/3)))
                        if random.random() <= prob:
                            current = deepcopy(nextord)
                            curcost = deepcopy(nextcost)
                        else:
                            fail += 1

                    if curcost < best_cost:
                        best_soln = deepcopy(current)
                        best_cost = deepcopy(curcost)
                        if run == 0 and graphing:
                            sa_time_list.append(time.time()-start)
                            sa_OBJ_list.append(deepcopy(best_cost))
                        lastMvmt = t
                        print('On iteration {}, new best solution cost is {} with sequence \
                               {}'.format(t,best_cost,best_soln))

                    if run > 0 and t >= 1.2 * len(current)**2.5:
                        ratio = fail/t
                        print('Finished run {} on iteration {} with ratio {} with best solution \
                              found on iteration {}'.format(run+1,t,ratio,lastMvmt))
                        if run != num_runs-1:
                            if best_cost < final_cost:
                                final_soln = deepcopy(best_soln)
                                final_cost = deepcopy(best_cost)
                        else:
                            if best_cost > final_cost:
                                best_soln = deepcopy(final_soln)
                                best_cost = deepcopy(final_cost)
                        break
                if run == 0:
                    ratio = fail/t
                    print('Finished on iteration {} with ratio {} with best solution found on \
                          iteration {}'.format(t,ratio,lastMvmt))
                    if best_cost < final_cost:
                        final_soln = deepcopy(best_soln)
                        final_cost = deepcopy(best_cost)

        else:
            while t < 1.2 * (len(current)-num_crews+1)**3:
                t += 1
                idx = random.randrange(0,len(current)-1)
                nextord = deepcopy(current)
                el = nextord.pop(idx)
                if idx+1 >= num_crews:
                    nextord.insert(idx+1,el)
                else:
                    nextord.insert(num_crews,el)

                nextcost, tap, __ = eval_working_sequence(
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
                    test1, __, __ = eval_sequence(curnet, current, after_eq_tstt, before_eq_tstt,
                        num_crews=num_crews)
                    if best_cost < test1:
                        print('Inaccuracy in new best soln: ' + str(test1-curcost) + ', test1 = '
                              + str(test1) + ', curcost = ' + str(curcost) + '. Do not use.')
                    else:
                        best_soln = deepcopy(current)
                        best_cost = deepcopy(curcost)
                        if graphing:
                            sa_time_list.append(time.time()-start)
                            sa_OBJ_list.append(deepcopy(best_cost))
                        lastMvmt = t
                        print('New best solution cost is ' + str(best_cost) + ' with sequence '
                              + str(best_soln))

                if t % 128 == 0:
                    ratio = fail/t
                    print('Iteration ' + str(t) + ', ratio ' + str(ratio))
            ratio = fail/t
            print('Finished on iteration {} with ratio {} with best solution found on iteration \
                  {}'.format(t,ratio,lastMvmt))

        elapsed = time.time() - start
        path = best_soln
        bound = best_cost
        if graphing:
            sa_time_list.append(elapsed)
            sa_OBJ_list.append(deepcopy(best_cost))

        test2, __, __ = eval_sequence(curnet, best_soln, after_eq_tstt, before_eq_tstt,
            num_crews=num_crews)

        if abs(test2-bound)> 5:
            print('Inaccuracy in best soln: ' + str(test2-bound) + ', test2 = ' + str(test2)
                  + ', bound = ' + str(bound))
        bound = test2

        save(fname + '_obj', bound)
        save(fname + '_path', path)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', tap_solved)
    else:
        bound = load(fname + '_obj')
        path = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    return bound, path, elapsed, tap_solved


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
        bound, eval_taps, __ = eval_sequence(
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

    return bound, path, elapsed, tap_solved


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
        bound, eval_taps, __ = eval_sequence(
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

    return bound, path, elapsed, tap_solved


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
        bound, eval_taps, __ = eval_sequence(
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

    return bound, path, elapsed, tap_solved


def oldaltLASR(net_before, after_eq_tstt, before_eq_tstt, time_before, Z_bar, last_b,
    first_b, bb_time):
    """approx solution method based on estimating Largest Average Smith Ratio
    using the preprocessing function, then supplementing with greedy choices
    for the first k slots"""
    start = time.time()
    tap_solved = 0

    fname = net_before.save_dir + '/altLASR_bound'
    if not os.path.exists(fname + extension):
        LASR_net = deepcopy(net_before)

        order = np.zeros(len(damaged_links))
        for i in range(len(damaged_links)):
            order[i] = Z_bar[i]/list(damaged_dict.values())[i]

        c = list(zip(order, damaged_links))
        sorted_c = sorted(c,reverse=True)
        LASR_benefits, LASR_path = zip(*sorted_c)
        print('Sorted LASR benefits and path: ', sorted_c)

        # Get best immediate benefit
        ordered_days, orderedb_benefits = [], []
        sorted_d = sorted(damaged_dict.items(), key=lambda x: x[1])
        for key, value in sorted_d:
            ordered_days.append(value)
            orderedb_benefits.append(first_b[key])
        ob, od, lzg_order = orderlists(orderedb_benefits, ordered_days, rem_keys=sorted_d)
        lzg_order = [i[0] for i in lzg_order]
        bdratio = [ob[i]/od[i] for i in range(len(ob))]
        print('Sorted LZG benefits/duration: {} and path: {}'.format(bdratio, lzg_order))

        # Improve pure LASR solution
        to_move = []
        for i in range(num_crews):
            if bdratio[i] > LASR_benefits[num_crews-i-1]:
                to_move.append(lzg_order[i])
        path = to_move
        for ij in LASR_path:
            if ij not in path:
                path.append(ij)

        print('altLASR path: ', path)

        elapsed = time.time() - start + time_before
        bound, eval_taps, __ = eval_sequence(
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

    return bound, path, elapsed, tap_solved


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
        N = len(damaged_links)
        delta = deltaTSTT

        # create model and add vars and constraints
        m = gp.Model('mip_delta')
        y = m.addMVar(shape=(N,N),vtype=GRB.BINARY)
        m.addConstrs((y[i,:].sum()==1 for i in range(N)), 'stages')
        m.addConstrs((y[:,i].sum()==1 for i in range(N)), 'links')

        obj = y[0,:] @ delta @ y[1,:].T
        for i in range(2,N):
            for j in range(i):
                obj += y[j,:] @ delta @ y[i,:].T
        m.setObjective(obj, GRB.MAXIMIZE)

        # optimize and print solution
        m.optimize()
        if m.Status == GRB.OPTIMAL:
            y_sol = y.getAttr('X')
            path = []
            for i in range(N):
                path.append(list(damaged_links)[int(y_sol[i].nonzero()[0])])
            print('path found using Gurobi: ' + str(path))
        else:
            print('Gurobi solver status not optimal...')

        elapsed = time.time() - start + time_before
        bound, eval_taps, __ = eval_sequence(
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

    return bound, path, elapsed, tap_solved


def SPT_solution(net_before, after_eq_tstt, before_eq_tstt, time_net_before):
    """simple heuristic which orders link for repair based on shortest repair time"""
    start = time.time()
    tap_solved = 0

    fname = net_before.save_dir + '/SPT_bound'
    if not os.path.exists(fname + extension):
        SPT_net = deepcopy(net_before)
        sorted_d = sorted(damaged_dict.items(), key=lambda x: x[1])
        path, __ = zip(*sorted_d)

        elapsed = time.time() - start + time_net_before
        bound, eval_taps, __ = eval_sequence(
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

    return bound, path, elapsed, tap_solved


def importance_factor_solution(net_before, after_eq_tstt, before_eq_tstt, time_net_before):
    """simple heuristic which orders links based on predamage flow"""
    start = time.time()

    fname = net_before.save_dir + '/importance_factor_bound'
    if not os.path.exists(fname + extension):
        print('Finding the importance factor solution ...')
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
        bound, eval_taps, __ = eval_sequence(if_net, path, after_eq_tstt, before_eq_tstt,
            if_dict, importance=True, num_crews=num_crews)
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

    return bound, path, elapsed, tap_solved


def linear_combo_solution(
        net_before, after_eq_tstt, before_eq_tstt, time_net_before, last_b, first_b, bb_time):
    """simple heuristic which orders links based a linear combination of pre-
    disruption flow, repair time, and immediate benefit to repairing each link"""
    start = time.time()

    fname = net_before.save_dir + '/linear_combo_bound'
    if not os.path.exists(fname + extension):
        print('Finding the linear combination solution ...')

        # Get importance factors
        tot_flow = 0
        if_net = deepcopy(net_before)
        for ij in if_net.linkDict:
            tot_flow += if_net.linkDict[ij]['flow']
        damaged_links = damaged_dict.keys()
        if_dict = {}
        for link_id in damaged_links:
            link_flow = if_net.linkDict[link_id]['flow']
            if_dict[link_id] = link_flow / tot_flow

        # Build linear combo dictionary to be sorted
        lc_dict = {}
        for link in damaged_links:
            lc_dict[link] = if_dict[link] + first_b[link] - damaged_dict[link]

        sorted_d = sorted(lc_dict.items(), key=lambda x: x[1])
        path, __ = zip(*sorted_d)
        path = path[::-1]

        elapsed = time.time() - start + time_net_before + bb_time
        start = time.time()
        bound, eval_taps, __ = eval_sequence(
            if_net, path, after_eq_tstt, before_eq_tstt, num_crews=num_crews)
        eval_time = time.time() - start
        print('Time to evaluate sequence: {}, number of TAPs to evaluate sequence: {}'.format(
              eval_time, eval_taps))
        tap_solved = len(damaged_links)+2

        save(fname + '_obj', bound)
        save(fname + '_path', path)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', 1)
    else:
        bound = load(fname + '_obj')
        path = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    return bound, path, elapsed, tap_solved


def brute_force(net_after, after_eq_tstt, before_eq_tstt, is_approx=False, num_crews=1):
    """enumerates all possible sequences to find the lowest overall tstt impacts"""
    start = time.time()
    tap_solved = 0
    damaged_links = damaged_dict.keys()
    approx_ext = ''
    if is_approx:
        approx_ext = '_approx'
        global approx_params

    fname = net_after.save_dir + '/min_seq' + approx_ext
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

            seq_net = deepcopy(net_after)
            cost, eval_taps, __ = eval_working_sequence(seq_net, sequence, after_eq_tstt,
                before_eq_tstt, is_approx=is_approx, num_crews=num_crews,
                approx_params=approx_params)
            tap_solved += eval_taps

            if cost < min_cost or min_cost == 1e+80:
                min_cost = cost
                min_seq = sequence
            i += 1

        elapsed = time.time() - start
        bound, __, __ = eval_sequence(
            seq_net, min_seq, after_eq_tstt, before_eq_tstt, num_crews=num_crews)

        save(fname + '_obj', bound)
        save(fname + '_path', min_seq)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', tap_solved)
    else:
        min_cost = load(fname + '_obj')
        min_seq = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')

    return min_cost, min_seq, elapsed, tap_solved


def lazy_greedy_heuristic(net_after, after_eq_tstt, before_eq_tstt, first_b, bb_time):
    """heuristic which orders links for repair based on effect on tstt if repaired first"""
    start = time.time()
    ordered_days = []
    orderedb_benefits = []
    sorted_d = sorted(damaged_dict.items(), key=lambda x: x[1])
    for key, value in sorted_d:
        ordered_days.append(value)
        orderedb_benefits.append(first_b[key])

    ob, od, lzg_order = orderlists(orderedb_benefits, ordered_days, rem_keys=sorted_d)
    lzg_order = [i[0] for i in lzg_order]
    elapsed = time.time() - start + bb_time

    test_net = deepcopy(net_after)
    bound, eval_taps, _ = eval_sequence(
        test_net, lzg_order, after_eq_tstt, before_eq_tstt, num_crews=num_crews)
    tap_solved = len(damaged_dict)+1

    fname = save_dir + '/lazygreedy_solution'
    save(fname + '_obj', bound)
    save(fname + '_path', lzg_order)
    save(fname + '_elapsed', elapsed)
    save(fname + '_num_tap', tap_solved)

    return bound, lzg_order, elapsed, tap_solved


def greedy_heuristic(net_after, after_eq_tstt, before_eq_tstt, time_net_before, time_net_after):
    """heuristic which orders links for repair at each step based on immediate effect on tstt"""
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

                after_fix_tstt = solve_UE(net=test_net, eval_seq=True)
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

        bound, __, __ = eval_sequence(
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

    return bound, path, elapsed, tap_solved


def greedy_heuristic_mult(
        net_after, after_eq_tstt, before_eq_tstt, time_net_before, time_net_after, num_crews):
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
                not_fixed = set(damaged_dict).difference(set(crew_order_list[:baseidx+1]))
                not_fixed.difference_update(set(added))
                test_net.not_fixed = set(not_fixed)

                after_fix_tstt = solve_UE(net=test_net, eval_seq=True)
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

            __, __, order = orderlists(orderedb_benefits, ordered_days, rem_keys=sorted_d)

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
                        sorted(crews).index(crews[which_crew[link_to_add]]) + 1 ,link_to_add)
                path.append(link_to_add)

                if completion[link_to_add] == min(crews):
                    min_index = eligible_to_add.index(link_to_add)
                    after_ = new_tstts[min_index]
                    baseidx = crew_order_list.index(link_to_add)
                else:
                    base = [k for k, v in completion.items() if v==min(crews)][0]
                    baseidx = crew_order_list.index(base)
                    not_fixed = set(damaged_dict).difference(set(crew_order_list[:baseidx+1]))
                    test_net.not_fixed = set(not_fixed)
                    after_ = solve_UE(net=test_net, eval_seq=True)
                    memory[frozenset(test_net.not_fixed)] = after_
                    tap_solved += 1

                eligible_to_add.remove(link_to_add)
                decoy_dd = deepcopy(decoy_dd)
                del decoy_dd[link_to_add]

        net = deepcopy(net_after)

        tap_solved += 1
        elapsed = time.time() - start + time_net_before + time_net_after

        bound, __, __ = eval_sequence(
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

    return bound, path, elapsed, tap_solved


def make_art_links(NETFILE, TRIPFILE, mc_weights, demand_mult):
    """creates artificial links with travel time and length 10x before eq shortest path"""
    start = time.time()
    art_net = Network(NETFILE, TRIPFILE)
    art_net.netfile = NETFILE
    art_net.tripfile = TRIPFILE
    art_net.not_fixed = set([])
    art_net.art_links = {}
    art_net.mc_weights = mc_weights
    art_net.demand_mult = demand_mult

    tstt = solve_UE(net=art_net, eval_seq=True, flows=True, warm_start=False)

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
    tstt = solve_UE(net=art_net, eval_seq=True, flows=True, warm_start=False)

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
            postbacklink[od.origin], postcost[od.origin] = art_net.shortestPath(od.origin)
            if postcost[od.origin][od.destination] >= 99999:
                count += 1
            if (od.origin != od.destination and
                    postcost[od.origin][od.destination] < 10*cost[od.origin][od.destination]):
                ODID = str(od.origin) + '->' + str(od.destination)
                del art_links[ODID]
        else:
            if postcost[od.origin][od.destination] >= 99999:
                count += 1
            if (od.origin != od.destination and
                    postcost[od.origin][od.destination] < 10*cost[od.origin][od.destination]):
                ODID = str(od.origin) + '->' + str(od.destination)
                del art_links[ODID]
    print('Created {} artificial links.'.format(len(art_links)))
    if count > len(art_links):
        print('There are {} more paths exceeding cost of 99999 than artificial links \
            created'.format(count-len(art_links)))

    art_net.art_links = art_links
    tstt = solve_UE(net=art_net, eval_seq=True, flows=True, warm_start=False)

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
            lc_damaged.append(mc.PatchCollection(damaged_segments[crew], color = jet(crew/num_crews)))
            ax.add_collection(lc_damaged[crew])
    ax.set_axis_off()
    plt.title('Map of ' + NETWORK.split('/')[-1] + ' with ' + str(len(damaged_links))
        + ' damaged links', fontsize=12)

    save_fig(save_dir, 'map', tight_layout=True)
    plt.close(fig)


def plot_time_OBJ(
        save_dir, bs_time_list=None, bs_OBJ_list=None, sa_time_list=None, sa_OBJ_list=None):
    """function to plot running time vs OBJ progression"""

    fig, ax = plt.subplots(figsize=(8,6))
    jet = cm.get_cmap('jet', reps)
    if bs_time_list != None:
        bs_indices = [i for i, time in enumerate(bs_time_list) if time == 0]
        bs_indices.append(len(bs_time_list))
        if num_broken < 16:
            for rep in range(reps):
                plt.step(bs_time_list[bs_indices[rep]:bs_indices[rep+1]],
                    (1-np.divide(bs_OBJ_list[bs_indices[rep] : bs_indices[rep+1]],
                    bs_OBJ_list[bs_indices[rep]])) * 100, where='post', color=jet(rep/reps),
                    label='beam search ' + str(rep+1))
        else:
            for rep in range(reps):
                plt.step(np.divide(bs_time_list[bs_indices[rep] : bs_indices[rep+1]],60),
                    (1-np.divide(bs_OBJ_list[bs_indices[rep] : bs_indices[rep+1]],
                    bs_OBJ_list[bs_indices[rep]])) * 100, where='post', color=jet(rep/reps),
                    label='beam search ' + str(rep+1))

    if sa_time_list != None:
        sa_indices = [i for i, time in enumerate(sa_time_list) if time == 0]
        sa_indices.append(len(sa_time_list))
        if num_broken < 16:
            for rep in range(reps):
                plt.step(sa_time_list[sa_indices[rep] : sa_indices[rep+1]],
                    (1-np.divide(sa_OBJ_list[sa_indices[rep] : sa_indices[rep+1]],
                    sa_OBJ_list[sa_indices[rep]])) * 100, where='post', color=jet(rep/reps),
                    linestyle='dashed', label='sim anneal ' + str(rep+1))
        else:
            for rep in range(reps):
                plt.step(np.divide(sa_time_list[sa_indices[rep] : sa_indices[rep+1]], 60),
                    (1 - np.divide(sa_OBJ_list[sa_indices[rep] : sa_indices[rep+1]],
                    sa_OBJ_list[sa_indices[rep]])) * 100, where='post', color=jet(rep/reps),
                    linestyle='dashed', label='sim anneal ' + str(rep+1))

    if num_broken < 16:
        plt.title('Runtime (seconds) vs OBJ function improvement (%) for '
            + NETWORK.split('/')[-1] + ' with ' + str(len(damaged_links)) + ' damaged links',
            fontsize=10)
    else:
        plt.title('Runtime (minutes) vs OBJ function improvement (%) for '
            + NETWORK.split('/')[-1] + ' with ' + str(len(damaged_links)) + ' damaged links',
            fontsize=10)
    plt.legend(ncol=2)

    save_fig(save_dir, 'timevsOBJ', tight_layout=True)
    plt.close(fig)


if __name__ == '__main__':

    net_name = args.net_name
    num_broken = args.num_broken
    approx = args.approx
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
    tables = args.tables
    graphing = args.graphing
    scenario_file = args.scenario
    before_after = args.onlybeforeafter
    output_sequences = args.output_sequences
    strength = args.strength
    full = args.full
    rand_gen = args.random
    location = args.loc
    opt = args.opt
    mip = args.mip
    sa = args.sa
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
    demand_mult = args.demand

    NETWORK = os.path.join(FOLDER, net_name)
    if net_name == 'Chicago-Sketch':
        net_name = 'ChicagoSketch'
    if net_name == 'Berlin-Mitte-Center':
        net_name = 'berlin-mitte-center'
    if net_name == 'Berlin-Mitte-Center2':
        net_name = 'berlin-mitte-center2'
    if net_name == 'Berlin-Mitte-Center3':
        net_name = 'berlin-mitte-center3'
    JSONFILE = os.path.join(NETWORK, net_name.lower() + '.geojson')
    NETFILE = os.path.join(NETWORK, net_name + "_net.tntp")

    TRIPFILE = os.path.join(NETWORK, net_name + "_trips.tntp")
    if not os.path.exists(TRIPFILE):
        TRIPFILE = list()
        i=1
        while True:
            if os.path.exists(os.path.join(NETWORK, net_name + "_trips" + str(i) + ".tntp")):
                TRIPFILE.append(os.path.join(NETWORK, net_name + "_trips" + str(i) + ".tntp"))
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
                sa_time_list = list()
                sa_OBJ_list = list()
            else:
                sa_time_list = None
                sa_OBJ_list = None
        global approx_params
        approx_params = None

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

                net = create_network(NETFILE, TRIPFILE, mc_weights=mc_weights, demand_mult=demand_mult)
                net.not_fixed = set([])
                net.art_links = {}
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
                    edge_list.append((int(netg.link[alink].tail), int(netg.link[alink].head)))
                G.add_edges_from(edge_list)

                damaged_links = []
                art_links = []
                decoy = deepcopy(all_links)
                i = 0

                dG = deepcopy(G)
                import geopy.distance

                if NETWORK.find('SiouxFalls') >= 0:
                    nodetntp = pd.read_csv(os.path.join(NETWORK, net_name + "_node.tntp"),
                        delimiter='\t')
                    coord_dict = {}
                    for index, row in nodetntp.iterrows():
                        lon = row['X']
                        lat = row['Y']
                        coord_dict[row['Node']] = (lon, lat)

                if NETWORK.find('Chicago-Sketch') >=0:
                    nodetntp = pd.read_csv(os.path.join(NETWORK, net_name + "_node.tntp"),
                        delimiter='\t')
                    coord_dict = {}
                    for index, row in nodetntp.iterrows():
                        lon = row['X']
                        lat = row['Y']
                        coord_dict[row['node']] = (lon, lat)

                if NETWORK.find('Anaheim') >= 0:
                    nodetntp = pd.read_json(os.path.join(NETWORK, 'anaheim_nodes.geojson'))
                    coord_dict = {}
                    for index, row in nodetntp.iterrows():
                        coord_dict[row['features']['properties']['id']] = row[
                            'features']['geometry']['coordinates']

                if NETWORK.find('Berlin-Mitte-Center') >= 0:
                    nodetntp = pd.read_csv(os.path.join(NETWORK, net_name + "_node.tntp"),
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
                        another_node = random.choices(decoy, 1.0/np.array(distances)**location, k=1)[0]
                        idx = decoy.index(another_node) # default location is 3
                        del distances[idx]
                        del decoy[idx]
                        selected_nodes.append(another_node)
                        i += 1

                    selected_indices = [all_nodes.index(ind) for ind in selected_nodes[1:]]
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
                            ij = random.choices(linkset, weights=np.exp(np.power(flow_on_links,
                                                1 / 3.0)), k=1)[0]
                        else:
                            ij = random.choices(linkset, weights=np.exp(np.power(flow_on_links,
                                                1 / 4.0)), k=1)[0]

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
                        ij = random.choices(decoy, weights=np.exp(np.power(weights, 1/3.0)),
                                            k=1)[0]

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
                            print("Link data line not formatted properly:\n '%s'" % line)
                            raise utils.BadFileFormatException

                        # Create link
                        linkID = '(' + str(data[0]).strip() + "," + str(data[1]).strip() + ')'
                        net.linkDict[linkID] = {}
                        net.linkDict[linkID]['length-cap'] = (float(data[2])
                            * np.cbrt(float(data[3])))

                len_links = [net.linkDict[lnk]['length-cap'] for lnk in damaged_links]

                for lnk in damaged_links:
                    if net.linkDict[lnk]['length-cap'] > np.quantile(len_links, 0.95):
                        damaged_dict[lnk] = np.random.gamma(4*2, 7*2, 1)[0]

                    elif net.linkDict[lnk]['length-cap'] > np.quantile(len_links, 0.75):
                        damaged_dict[lnk] = np.random.gamma(7*np.sqrt(2), 7*np.sqrt(2), 1)[0]

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
                cur_scnario_num = max_rep + 1

                ULT_SCENARIO_REP_DIR = os.path.join(
                    ULT_SCENARIO_DIR, str(cur_scnario_num))

                os.makedirs(ULT_SCENARIO_REP_DIR, exist_ok=True)


            else:
                damaged_dict_ = get_broken_links(JSONFILE, scenario_file)

                memory = {}
                net = create_network(NETFILE, TRIPFILE, demand_mult=demand_mult)

                all_links = damaged_dict_.keys()
                damaged_links = np.random.choice(list(all_links), num_broken, replace=False)
                removals = []

                damaged_dict = {}

                for a_link in damaged_links:
                    if a_link in removals:
                        continue
                    damaged_dict[a_link] = damaged_dict_[a_link]

                del damaged_dict_
                num_broken = len(damaged_dict)

                SCENARIO_DIR = os.path.join(NETWORK_DIR, strength)
                ULT_SCENARIO_DIR = os.path.join(SCENARIO_DIR, str(num_broken))
                os.makedirs(ULT_SCENARIO_DIR, exist_ok=True)

                repetitions = get_folders(ULT_SCENARIO_DIR)

                if len(repetitions) == 0:
                    max_rep = -1
                else:
                    num_scenario = [int(i) for i in repetitions]
                    max_rep = max(num_scenario)
                cur_scnario_num = max_rep + 1

                ULT_SCENARIO_REP_DIR = os.path.join(
                    ULT_SCENARIO_DIR, str(cur_scnario_num))

                os.makedirs(ULT_SCENARIO_REP_DIR, exist_ok=True)


            # Finalize damaged links
            damaged_links = damaged_dict.keys()

            print('damaged_dict: ', damaged_dict)
            save_dir = ULT_SCENARIO_REP_DIR
            save(save_dir + '/damaged_dict', damaged_dict)


            if before_after:
                net_after, after_eq_tstt, _ = state_after(damaged_links, save_dir)
                net_before, before_eq_tstt, _ = state_before(damaged_links, save_dir)
                plot_nodes_links(save_dir, netg, damaged_links, coord_dict, names=True)
                t = PrettyTable()
                t.title = net_name + ' with ' + str(num_broken) + ' broken bridges'
                t.field_names = ['Before EQ TSTT', 'After EQ TSTT']
                t.add_row([before_eq_tstt, after_eq_tstt])
                print(t)

            elif output_sequences:
                """ builds a dict (link names are keys) that contains attributes: duration,
                importance factor, immediate benefit. Then solves to optimality, adds optimal
                order of repair ahead of attributes (ie. '1' if repaired first) """

                memory = {}
                art_link_dict = make_art_links(NETFILE, TRIPFILE, mc_weights, demand_mult)

                net_before, before_eq_tstt, time_net_before = state_before(damaged_links, save_dir,
                    real=True)
                net_after, after_eq_tstt, time_net_after = state_after(damaged_links, save_dir,
                    real=True)
                net_before.save_dir = save_dir
                net_after.save_dir = save_dir
                first_b, bb_time = first_benefit(net_before, damaged_links, before_eq_tstt)
                last_b = last_benefit(net_after, damaged_links, after_eq_tstt)
                wb, bb, swapped_links = safety(last_b, first_b)
                if num_crews==1:
                    upper_bound = (after_eq_tstt - before_eq_tstt)
                                   * sum(net_before.damaged_dict.values())
                save(save_dir + '/upper_bound', upper_bound)

                if damaged_dict_preset=='':
                    plot_nodes_links(save_dir, netg, damaged_links, coord_dict, names=True)

                # Get repair durations
                damaged_attributes = dict()
                for link in damaged_links:
                    damaged_attributes[link] = list()
                    damaged_attributes[link].append(damaged_dict[link])

                # Get importance factors
                tot_flow = 0
                if_net = deepcopy(net_before)
                for ij in if_net.linkDict:
                    tot_flow += if_net.linkDict[ij]['flow']
                for link in damaged_links:
                    link_flow = if_net.linkDict[link]['flow']
                    damaged_attributes[link].append(link_flow / tot_flow)

                # Get immediate benefit
                for link in damaged_links:
                    damaged_attributes[link].append(first_b[link])
                    if link in swapped_links:
                        damaged_attributes[link].append('swapped')
                    else:
                        damaged_attributes[link].append('not swapped')
                    damaged_attributes[link].append(abs(first_b[link]-last_b[link]))

                opt_obj, opt_soln, opt_elapsed, opt_num_tap = brute_force(
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
                art_link_dict = make_art_links(NETFILE, TRIPFILE, mc_weights, demand_mult)

                net_before, before_eq_tstt, time_net_before = state_before(damaged_links, save_dir,
                    real=True)
                net_after, after_eq_tstt, time_net_after = state_after(damaged_links, save_dir,
                    real=True)
                net_before.save_dir = save_dir
                net_after.save_dir = save_dir
                first_b, bb_time = first_benefit(net_before, damaged_links, before_eq_tstt)
                last_b = last_benefit(net_after, damaged_links, after_eq_tstt)
                wb, bb, swapped_links = safety(last_b, first_b)
                if num_crews==1:
                    upper_bound = (after_eq_tstt - before_eq_tstt)
                                   * sum(net_before.damaged_dict.values())
                save(save_dir + '/upper_bound', upper_bound)

                if damaged_dict_preset=='':
                    plot_nodes_links(save_dir, netg, damaged_links, coord_dict, names=True)
                print('before tstt: {}, after tstt: {}, total (single crew) duration: {}'.format(
                      before_eq_tstt, after_eq_tstt, sum(damaged_dict.values())))
                if num_crews==1:
                    print('Simple upper bound on obj function (total travel delay): ', upper_bound)

                # Approx solution methods
                if approx:
                    memory1 = deepcopy(memory)
                    preprocess_st = time.time()
                    Z_bar, alt_Z_bar, preprocessing_num_tap = find_approx(approx, damaged_links,
                         net_after, last_b, first_b)
                    preprocess_elapsed = time.time() - preprocess_st
                    time_before = preprocess_elapsed + time_net_before + bb_time*2

                    # Largest Average First Order
                    LAFO_obj, LAFO_soln, LAFO_elapsed, LAFO_num_tap = LAFO(
                        net_before, after_eq_tstt, before_eq_tstt, time_before, Z_bar)
                    LAFO_num_tap += preprocessing_num_tap
                    if alt_crews == None and not multiclass:
                        print('LAFO_obj: ', LAFO_obj)
                    elif multiclass and isinstance(net_after.tripfile, list):
                        test_net = deepcopy(net_after)
                        LAFO_obj_mc, __, __ = eval_sequence(test_net, LAFO_soln, after_eq_tstt,
                            before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
                        print('LAFO_obj: ', LAFO_obj_mc)
                    else:
                        LAFO_obj_mult = [0]*(len(alt_crews)+1)
                        LAFO_obj_mult[0] = LAFO_obj
                        for num in range(len(alt_crews)):
                            test_net = deepcopy(net_before)
                            LAFO_obj_mult[num+1], __, __ = eval_sequence(test_net, LAFO_soln,
                                after_eq_tstt, before_eq_tstt, num_crews=alt_crews[num])
                        print('LAFO_obj: ', LAFO_obj_mult)


                    # Largest Average Smith Ratio
                    LASR_obj, LASR_soln, LASR_elapsed, LASR_num_tap = LASR(
                        net_before, after_eq_tstt, before_eq_tstt, time_before, Z_bar)
                    LASR_num_tap += preprocessing_num_tap
                    if alt_crews == None and not multiclass:
                        print('LASR_obj: ', LASR_obj)
                    elif multiclass and isinstance(net_after.tripfile, list):
                        test_net = deepcopy(net_after)
                        LASR_obj_mc, __, __ = eval_sequence(test_net, LASR_soln, after_eq_tstt,
                            before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
                        print('LASR_obj: ', LASR_obj_mc)
                    else:
                        LASR_obj_mult = [0]*(len(alt_crews)+1)
                        LASR_obj_mult[0] = LAFO_obj
                        for num in range(len(alt_crews)):
                            test_net = deepcopy(net_before)
                            LASR_obj_mult[num+1], __, __ = eval_sequence(test_net, LASR_soln,
                                after_eq_tstt, before_eq_tstt, num_crews=alt_crews[num])
                        print('LASR_obj: ', LASR_obj_mult)


                    # Modified LASR
                    if approx==3:
                        altLASR_obj, altLASR_soln, altLASR_elapsed, altLASR_num_tap = altLASR(
                            net_before, after_eq_tstt, before_eq_tstt, time_before, alt_Z_bar)
                        altLASR_num_tap += preprocessing_num_tap
                        if alt_crews == None and not multiclass:
                            print('altLASR_obj: ', altLASR_obj)
                        elif multiclass and isinstance(net_after.tripfile, list):
                            test_net = deepcopy(net_after)
                            altLASR_obj_mc, __, __ = eval_sequence(test_net, altLASR_soln,
                                after_eq_tstt, before_eq_tstt, num_crews=num_crews,
                                multiclass=multiclass)
                            print('altLASR_obj: ', altLASR_obj_mc)
                        else:
                            altLASR_obj_mult = [0]*(len(alt_crews)+1)
                            altLASR_obj_mult[0] = LAFO_obj
                            for num in range(len(alt_crews)):
                                test_net = deepcopy(net_before)
                                altLASR_obj_mult[num+1], __, __ = eval_sequence(test_net, altLASR_soln,
                                    after_eq_tstt, before_eq_tstt, num_crews=alt_crews[num])
                            print('altLASR_obj: ', altLASR_obj_mult)

                    memory = deepcopy(memory1)


                if mip==3:
                    memory1 = deepcopy(memory)
                    preprocess_st = time.time()
                    deltaTSTT, preprocessing_num_tap = find_deltaTSTT(damaged_links, net_after,
                        after_eq_tstt, before_eq_tstt, last_b, first_b)
                    preprocess_elapsed = time.time() - preprocess_st
                    time_mip_before = preprocess_elapsed + time_net_before + bb_time*2

                    # MIP alternate formulation using estimated deltaTSTT[t,b] values
                    mip3_obj, mip3_soln, mip3_elapsed, mip3_num_tap = mip_delta(
                        net_before, after_eq_tstt, before_eq_tstt, time_mip_before, deltaTSTT)
                    mip3_num_tap += preprocessing_num_tap
                    if alt_crews == None and not multiclass:
                        print('MIP_delta_obj: ', mip3_obj)
                    elif multiclass and isinstance(net_after.tripfile, list):
                        test_net = deepcopy(net_after)
                        mip3_obj_mc, __, __ = eval_sequence(test_net, mip3_soln, after_eq_tstt,
                            before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
                        print('MIP_delta_obj: ', mip3_obj_mc)
                    else:
                        mip3_obj_mult = [0]*(len(alt_crews)+1)
                        mip3_obj_mult[0] = mip3_obj
                        for num in range(len(alt_crews)):
                            test_net = deepcopy(net_before)
                            mip3_obj_mult[num+1], __, __ = eval_sequence(test_net, mip3_soln,
                                after_eq_tstt, before_eq_tstt, num_crews=alt_crews[num])
                        print('MIP_delta_obj: ', mip3_obj_mult)
                    memory = deepcopy(memory1)


                # Shortest processing time solution
                SPT_obj, SPT_soln, SPT_elapsed, SPT_num_tap = SPT_solution(
                    net_before, after_eq_tstt, before_eq_tstt, time_net_before)
                test_net = deepcopy(net_after)
                start = time.time()
                __, __, __ = eval_sequence(test_net, SPT_soln, after_eq_tstt, before_eq_tstt,
                    num_crews=num_crews, multiclass=multiclass)
                evaluation_time = time.time() - start
                print('Time to evaluate a sequence: ', evaluation_time)
                if alt_crews == None and not multiclass:
                    print('SPT_obj: ', SPT_obj)
                elif multiclass and isinstance(net_after.tripfile, list):
                    test_net = deepcopy(net_after)
                    SPT_obj_mc, __, __ = eval_sequence(test_net, SPT_soln, after_eq_tstt,
                        before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
                    print('SPT_obj: ', SPT_obj_mc)
                else:
                    SPT_obj_mult = [0]*(len(alt_crews)+1)
                    SPT_obj_mult[0] = SPT_obj
                    for num in range(len(alt_crews)):
                        test_net = deepcopy(net_before)
                        SPT_obj_mult[num+1], __, __ = eval_sequence(test_net, SPT_soln,
                            after_eq_tstt, before_eq_tstt, num_crews=alt_crews[num])
                    print('SPT_obj: ', SPT_obj_mult)


                # Lazy greedy solution
                lg_obj, lg_soln, lg_elapsed, lg_num_tap = lazy_greedy_heuristic(
                   net_after, after_eq_tstt, before_eq_tstt, first_b, bb_time)
                if alt_crews == None and not multiclass:
                    print('lazy_greedy_obj: ', lg_obj)
                elif multiclass and isinstance(net_after.tripfile, list):
                    test_net = deepcopy(net_after)
                    lg_obj_mc, __, __ = eval_sequence(test_net, lg_soln, after_eq_tstt,
                        before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
                    print('lazy_greedy_obj: ', lg_obj_mc)
                else:
                    lg_obj_mult = [0]*(len(alt_crews)+1)
                    lg_obj_mult[0] = lg_obj
                    for num in range(len(alt_crews)):
                        test_net = deepcopy(net_after)
                        lg_obj_mult[num+1], __, __ = eval_sequence(test_net, lg_soln,
                            after_eq_tstt, before_eq_tstt, num_crews=alt_crews[num])
                    print('lazy_greedy_obj: ', lg_obj_mult)

                wb_update = deepcopy(wb)
                bb_update = deepcopy(bb)


                # Get greedy solution
                if num_crews == 1:
                    greedy_obj, greedy_soln, greedy_elapsed, greedy_num_tap = greedy_heuristic(
                        net_after, after_eq_tstt, before_eq_tstt, time_net_before, time_net_after)
                else:
                    (greedy_obj, greedy_soln, greedy_elapsed, greedy_num_tap
                        ) = greedy_heuristic_mult(net_after, after_eq_tstt, before_eq_tstt,
                        time_net_before, time_net_after, num_crews)
                if alt_crews == None and not multiclass:
                    print('greedy_obj: ', greedy_obj)
                elif multiclass and isinstance(net_after.tripfile, list):
                    test_net = deepcopy(net_after)
                    greedy_obj_mc, __, __ = eval_sequence(test_net, greedy_soln, after_eq_tstt,
                        before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
                    print('greedy_obj: ', greedy_obj_mc)
                else:
                    greedy_obj_mult = [0]*(len(alt_crews)+1)
                    greedy_obj_mult[0] = greedy_obj
                    for num in range(len(alt_crews)):
                        test_net = deepcopy(net_after)
                        greedy_obj_mult[num+1], __, __ = eval_sequence(test_net, greedy_soln,
                            after_eq_tstt, before_eq_tstt, num_crews=alt_crews[num])
                    print('greedy_obj: ', greedy_obj_mult)

                bfs = BestSoln()
                bfs.cost = greedy_obj
                bfs.path = greedy_soln

                wb = deepcopy(wb_update)
                bb = deepcopy(bb_update)
                wb_orig = deepcopy(wb)
                bb_orig = deepcopy(bb)


                # Get feasible solution using importance factors
                (importance_obj, importance_soln, importance_elapsed, importance_num_tap
                    ) = importance_factor_solution(net_before, after_eq_tstt, before_eq_tstt,
                    time_net_before)
                if alt_crews == None and not multiclass:
                    print('importance_obj: ', importance_obj)
                elif multiclass and isinstance(net_after.tripfile, list):
                    test_net = deepcopy(net_after)
                    importance_obj_mc, __, __ = eval_sequence(test_net, importance_soln,
                        after_eq_tstt, before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
                    print('importance_obj: ', importance_obj_mc)
                else:
                    importance_obj_mult = [0]*(len(alt_crews)+1)
                    importance_obj_mult[0] = importance_obj
                    for num in range(len(alt_crews)):
                        test_net = deepcopy(net_after)
                        importance_obj_mult[num+1], __, __ = eval_sequence(test_net,
                            importance_soln, after_eq_tstt, before_eq_tstt,
                            num_crews=alt_crews[num])
                    print('importance_obj: ', importance_obj_mult)

                if importance_obj < bfs.cost:
                    bfs.cost = importance_obj
                    bfs.path = importance_soln


                # Get feasible solution using linear combination of factors
                #lc_obj, lc_soln, lc_elapsed, lc_num_tap = linear_combo_solution(net_before,
                #    after_eq_tstt, before_eq_tstt, time_net_before, wb, bb, bb_time, swapped_links)
                #if alt_crews == None and not multiclass:
                #    print('Linear combination obj: ', lc_obj)
                #elif multiclass and isinstance(net_after.tripfile, list):
                #    test_net = deepcopy(net_after)
                #    lc_obj_mc, __, __ = eval_sequence(test_net, lc_soln, after_eq_tstt,
                #        before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
                #    print('Linear combination obj: ', lc_obj_mc)
                #else:
                #    lc_obj_mult = [0]*(len(alt_crews)+1)
                #    lc_obj_mult[0] = importance_obj
                #    for num in range(len(alt_crews)):
                #        test_net = deepcopy(net_after)
                #        lc_obj_mult[num+1], __, __ = eval_sequence(test_net, lc_soln,
                #            after_eq_tstt, before_eq_tstt, num_crews=alt_crews[num])
                #    print('Linear combination obj: ', lc_obj_mult)

                memory1 = deepcopy(memory)


                # Get optimal solution via brute force
                if opt:
                    if opt==1:
                        is_approx = False
                    if opt==2:
                        is_approx = True
                        ML_mem = {}
                        ML_start = time.time()
                        model, meany, stdy, Z_bar, ML_num_tap = ML_preprocess(
                            damaged_links, net_after)
                        approx_params = (model, meany, stdy)
                        ML_time = time.time() - ML_start
                        print('Time to train ML model: '+str(ML_time))
                    opt_obj, opt_soln, opt_elapsed, opt_num_tap = brute_force(
                        net_after, after_eq_tstt, before_eq_tstt, is_approx=is_approx, num_crews=num_crews)
                    if opt==1:
                        opt_elapsed += greedy_elapsed
                        opt_num_tap += greedy_num_tap + len(damaged_links) - 2
                        print('Optimal objective with {} crew(s): {}, optimal path: {}'.format(
                              num_crews, opt_obj, opt_soln))
                    if opt==2:
                        opt_elapsed += ML_time
                        opt_num_tap += ML_num_tap
                        print('ML brute force objective with {} crew(s): {}, path: {}'.format(
                              num_crews, opt_obj, opt_soln))
                    if multiclass and isinstance(net_after.tripfile, list):
                        test_net = deepcopy(net_after)
                        opt_obj_mc, __, __ = eval_sequence(test_net, opt_soln, after_eq_tstt,
                            before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
                        print('Optimal objective and demand class breakdown with {} crew(s): \
                              {}'.format(num_crews, opt_obj_mc))
                    if alt_crews != None:
                        opt_obj_mult = [0]*(len(alt_crews)+1)
                        opt_obj_mult[0] = opt_obj
                        opt_soln_mult = [0]*(len(alt_crews)+1)
                        opt_soln_mult[0] = opt_soln
                        opt_elapsed_mult = [0]*(len(alt_crews)+1)
                        opt_elapsed_mult[0] = opt_elapsed
                        opt_num_tap_mult = [0]*(len(alt_crews)+1)
                        opt_num_tap_mult[0] = opt_num_tap
                        for num in range(len(alt_crews)):
                            memory = deepcopy(memory1)
                            (opt_obj_mult[num+1], opt_soln_mult[num+1], opt_elapsed_mult[num+1],
                                opt_num_tap_mult[num+1]) = brute_force(net_after, after_eq_tstt,
                                before_eq_tstt, is_approx=is_approx, num_crews=alt_crews[num])
                            if opt==1:
                                opt_elapsed_mult[num+1] += greedy_elapsed
                                opt_num_tap_mult[num+1] += greedy_num_tap + len(damaged_links) - 2
                                print('Optimal objective with {} crew(s): {}, optimal path: {}'.format(
                                      alt_crews[num], opt_obj_mult[num+1], opt_soln_mult[num+1]))
                            if opt==2:
                                opt_elapsed_mult[num+1] += ML_elapsed
                                opt_num_tap_mult[num+1] += ML_num_tap
                                print('ML brute force objective with {} crew(s): {}, path: {}'.format(
                                      alt_crews[num], opt_obj_mult[num+1], opt_soln_mult[num+1]))
                            if multiclass and isinstance(net_after.tripfile, list):
                                test_net = deepcopy(net_after)
                                temp, __, __ = eval_sequence(test_net, opt_soln_mult[num+1],
                                    after_eq_tstt, before_eq_tstt, num_crews=alt_crews[num],
                                    multiclass=multiclass)
                                opt_obj_mc.append(temp)
                                opt_obj_mc[num+1].insert(0, opt_obj_mult[num+1])
                                print('Optimal objective and demand class breakdown with {} \
                                      crew(s): {}'.format(num_crews, opt_obj_mc[num+1]))
                    if opt==2:
                        del ML_mem

                best_benefit_taps = num_broken
                worst_benefit_taps = num_broken

                memory = deepcopy(memory1)


                # Get simulated annealing solution
                if sa:
                    sa_obj, sa_soln, sa_elapsed, sa_num_tap = sim_anneal(bfs, net_after,
                        after_eq_tstt, before_eq_tstt, damaged_links, num_crews=num_crews)
                    sa_elapsed += greedy_elapsed + importance_elapsed + 2*evaluation_time
                    sa_num_tap += greedy_num_tap + 2*num_broken
                    if alt_crews == None and not multiclass:
                        print('Simulated annealing obj: ', sa_obj)
                    elif multiclass and isinstance(net_after.tripfile, list):
                        test_net = deepcopy(net_after)
                        sa_obj_mc, __, __ = eval_sequence(test_net, sa_soln, after_eq_tstt,
                            before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
                        print('Simulated annealing obj: ', sa_obj_mc)
                    else:
                        sa_obj_mult = [0]*(len(alt_crews)+1)
                        sa_obj_mult[0] = sa_obj
                        for num in range(len(alt_crews)):
                            test_net = deepcopy(net_after)
                            sa_obj_mult[num+1], _, _ = eval_sequence(test_net, sa_soln,
                                after_eq_tstt, before_eq_tstt, num_crews=alt_crews[num])
                        print('Simulated annealing obj: ', sa_obj_mult)


                # Use full search algorthm to find solution
                if full:
                    print('Running full algorithm ...')
                    algo_num_tap = best_benefit_taps + worst_benefit_taps
                    fname = save_dir + '/algo_solution'

                    if not os.path.exists(fname + extension):
                        search_start = time.time()
                        if graphing:
                            (algo_path, algo_obj, search_tap_solved, tot_child, uncommon_number,
                                common_number, __, wb, bb, wb_update, bb_update, memory) = search(
                                NETFILE, TRIPFILE, net_after, after_eq_tstt, before_eq_tstt,
                                start_node, end_node, bfs, wb, bb, wb_update, bb_update, memory,
                                mc_weights=mc_weights, demand_mult=demand_mult, graphing=graphing,
                                bs_time_list=bs_time_list, bs_OBJ_list=bs_OBJ_list)
                        else:
                            (algo_path, algo_obj, search_tap_solved, tot_child, uncommon_number,
                                common_number, __, wb, bb, wb_update, bb_update, memory) = search(
                                NETFILE, TRIPFILE, net_after, after_eq_tstt, before_eq_tstt,
                                start_node, end_node, bfs, wb, bb, wb_update, bb_update, memory,
                                mc_weights=mc_weights, demand_mult=demand_mult)
                        search_elapsed = (time.time() - search_start + greedy_elapsed
                                          + importance_elapsed + 2*evaluation_time)

                        net_after, after_eq_tstt = state_after(damaged_links, save_dir, real=True)
                        net_before, before_eq_tstt = state_before(damaged_links, save_dir,
                            real=True)

                        first_net = deepcopy(net_after)
                        first_net.relax = False
                        algo_obj, __, __ = eval_sequence(first_net, algo_path, after_eq_tstt,
                            before_eq_tstt, num_crews=num_crews)

                        algo_num_tap += search_tap_solved + greedy_num_tap + 2*(num_broken-1)
                        algo_elapsed = search_elapsed + benefit_analysis_elapsed

                        save(fname + '_obj', algo_obj)
                        save(fname + '_path', algo_path)
                        save(fname + '_num_tap', algo_num_tap)
                        save(fname + '_elapsed', algo_elapsed)
                        save(fname + '_totchild', tot_child)
                        save(fname + '_uncommon', uncommon_number)
                        save(fname + '_common', common_number)
                    else:
                        algo_obj = load(fname + '_obj')
                        algo_path = load(fname + '_path')
                        algo_num_tap = load(fname + '_num_tap')
                        algo_elapsed = load(fname + '_elapsed')

                    print('---------------FULLOBJ')
                    if alt_crews == None and not multiclass:
                        print('Full obj: ', algo_obj)
                    elif multiclass and isinstance(net_after.tripfile, list):
                        test_net = deepcopy(net_after)
                        algo_obj_mc, _, _ = eval_sequence(test_net, algo_path, after_eq_tstt,
                            before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
                        print('Full obj: ', algo_obj_mc)
                    else:
                        algo_obj_mult = [0]*(len(alt_crews)+1)
                        algo_obj_mult[0] = algo_obj
                        for num in range(len(alt_crews)):
                            test_net = deepcopy(net_after)
                            algo_obj_mult[num+1], __, __ = eval_sequence(test_net, algo_path,
                                after_eq_tstt, before_eq_tstt, num_crews=alt_crews[num])
                        print('Full obj: ', algo_obj_mult)


                # Use beamsearch algorithm to find solution
                if beam_search:
                    runs = ['normal'] # sa_seed must be run first because of memory from sa
                    for run in runs:
                        k = 2
                        betas = [128]
                        gammas = [128]
                        beta_gamma = list(itertools.product(gammas, betas))

                        experiment_dict = {}
                        for a_pair in beta_gamma:
                            gamma = a_pair[0]
                            beta = a_pair[1]

                            # Beamsearch using gap 1e-4
                            print('===', gamma, beta , '===')

                            fname = save_dir + '/r_algo_solution' + '_k' + str(k)
                            ext_name = 'r_k' + str(k)

                            if run != 'sa_seed':
                                del memory
                                memory = deepcopy(memory1)

                            wb = wb_orig
                            bb = bb_orig
                            wb_update = wb_orig
                            bb_update = bb_orig

                            if run == 'sa_seed':
                                if sa_obj < bfs.cost:
                                    bfs.cost = sa_obj
                                    bfs.path = sa_soln
                            else:
                                if greedy_obj <= importance_obj:
                                    bfs.cost = greedy_obj
                                    bfs.path = greedy_soln
                                else:
                                    bfs.cost = importance_obj
                                    bfs.path = importance_soln

                            if run == 'sa_seed':
                                sa_r_algo_num_tap = sa_num_tap
                            else:
                                r_algo_num_tap = (worst_benefit_taps - 1 + greedy_num_tap
                                                  + 2*(num_broken-1))

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
                                    graphing, bs_time_list=bs_time_list, bs_OBJ_list=bs_OBJ_list)
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

                            first_net = deepcopy(net_after)
                            first_net.relax = False
                            r_algo_obj, __, __ = eval_sequence(first_net, r_algo_path,
                                after_eq_tstt, before_eq_tstt, num_crews=num_crews)

                            if run == 'sa_seed':
                                sa_r_algo_num_tap += r_search_tap_solved
                                sa_r_algo_elapsed = search_elapsed + sa_elapsed
                                sa_r_algo_obj = deepcopy(r_algo_obj)
                                sa_r_algo_path = deepcopy(r_algo_path)
                                save(fname + '_obj_sa', sa_r_algo_obj)
                                save(fname + '_path_sa', sa_r_algo_path)
                                save(fname + '_num_tap_sa', sa_r_algo_num_tap)
                                save(fname + '_elapsed_sa', sa_r_algo_elapsed)
                                save(fname + '_totchild_sa', tot_childr)
                                save(fname + '_uncommon_sa', uncommon_numberr)
                                save(fname + '_common_sa', common_numberr)
                                save(fname + '_num_purge_sa', num_purger)

                                print(f'Beam_search k: {k}, relaxed: True, type of run {run}')
                                if alt_crews == None and not multiclass:
                                    print('SA seeded beam search obj: ', sa_r_algo_obj)
                                elif multiclass and isinstance(net_after.tripfile, list):
                                    test_net = deepcopy(net_after)
                                    sa_r_algo_obj_mc, __, __ = eval_sequence(test_net,
                                        sa_r_algo_path, after_eq_tstt, before_eq_tstt,
                                        num_crews=num_crews, multiclass=multiclass)
                                    print('SA seeded beam search obj: ', sa_r_algo_obj_mc)
                                else:
                                    sa_r_algo_obj_mult = [0]*(len(alt_crews)+1)
                                    sa_r_algo_obj_mult[0] = sa_r_algo_obj
                                    for num in range(len(alt_crews)):
                                        test_net = deepcopy(net_after)
                                        sa_r_algo_obj_mult[num+1], __, __ = eval_sequence(
                                            test_net, sa_r_algo_path, after_eq_tstt,
                                            before_eq_tstt, num_crews=alt_crews[num])
                                    print('SA seeded beam search obj: ', sa_r_algo_obj_mult)

                            else:
                                r_algo_num_tap += r_search_tap_solved
                                r_algo_elapsed = (search_elapsed + greedy_elapsed +
                                                  importance_elapsed + 2*evaluation_time)
                                save(fname + '_obj', r_algo_obj)
                                save(fname + '_path', r_algo_path)
                                save(fname + '_num_tap', r_algo_num_tap)
                                save(fname + '_elapsed', r_algo_elapsed)
                                save(fname + '_totchild', tot_childr)
                                save(fname + '_uncommon', uncommon_numberr)
                                save(fname + '_common', common_numberr)
                                save(fname + '_num_purge', num_purger)

                                print(f'Beam_search k: {k}, relaxed: True, type of run {run}')
                                if alt_crews == None and not multiclass:
                                    print('Beam search obj: ', r_algo_obj)
                                elif multiclass and isinstance(net_after.tripfile, list):
                                    test_net = deepcopy(net_after)
                                    r_algo_obj_mc, __, __ = eval_sequence(test_net, r_algo_path,
                                        after_eq_tstt, before_eq_tstt, num_crews=num_crews,
                                        multiclass=multiclass)
                                    print('Beam search obj: ', (r_algo_obj_mc))
                                else:
                                    r_algo_obj_mult = [0]*(len(alt_crews)+1)
                                    r_algo_obj_mult[0] = r_algo_obj
                                    for num in range(len(alt_crews)):
                                        test_net = deepcopy(net_after)
                                        r_algo_obj_mult[num+1], __, __ = eval_sequence(test_net,
                                            r_algo_path, after_eq_tstt, before_eq_tstt,
                                            num_crews=alt_crews[num])
                                    print('Beam search obj: ', r_algo_obj_mult)

                                experiment_dict[a_pair] = [r_algo_obj, r_algo_num_tap,
                                    r_algo_elapsed]

                        for k,v in experiment_dict.items():
                            print('Gamma - Beta: {}, obj-tap-elapsed: {}'.format(k,v))


                if multiclass and isinstance(net_after.tripfile, list):
                    t = PrettyTable()
                    if damaged_dict_preset=='':
                        t.title = (net_name + ' with ' + str(num_broken)
                            + ' broken bridges (class priorities)')
                    else:
                        t.title = (net_name + ' with ' + str(num_broken)
                            + ' broken bridges (equal priority)')
                    t.field_names = ['Method', 'Objective', 'Run Time', '# TAP']
                    if opt==1:
                        t.add_row(['OPTIMAL', opt_obj_mc, opt_elapsed, opt_num_tap])
                    if opt==2:
                        t.add_row(['ML BF', opt_obj_mc, opt_elapsed, opt_num_tap])
                    if approx:
                        if approx>=2 or (approx==1 and LAFO_obj_mc[0]<LASR_obj_mc[0]):
                            t.add_row(['approx-LAFO', LAFO_obj_mc, LAFO_elapsed, LAFO_num_tap])
                        if approx>=2 or (approx==1 and LAFO_obj_mc[0]>=LASR_obj_mc[0]):
                            t.add_row(['approx-LASR', LASR_obj_mc, LASR_elapsed, LASR_num_tap])
                        if approx==3:
                            t.add_row(['alt-LASR', altLASR_obj_mc, altLASR_elapsed, altLASR_num_tap])
                    if mip==3:
                        t.add_row(['MIP_delta', mip3_obj_mc, mip3_elapsed, mip3_num_tap])
                    if full:
                        t.add_row(['FULL ALGO', algo_obj_mc, algo_elapsed, algo_num_tap])
                    if beam_search:
                        t.add_row(['BeamSearch_relaxed', r_algo_obj_mc, r_algo_elapsed,
                            r_algo_num_tap])
                        try:
                            t.add_row(['BeamSearch_sa_seed', sa_r_algo_obj_mc, sa_r_algo_elapsed,
                                sa_r_algo_num_tap])
                        except:
                            pass
                    if sa:
                        t.add_row(['Simulated Annealing', sa_obj_mc, sa_elapsed, sa_num_tap])
                    t.add_row(['GREEDY', greedy_obj_mc, greedy_elapsed, greedy_num_tap])
                    t.add_row(['LG', lg_obj_mc, lg_elapsed, lg_num_tap])
                    #t.add_row(['Linear Combination', lc_obj_mc, lc_elapsed, lc_num_tap])
                    t.add_row(['IMPORTANCE', importance_obj_mc, importance_elapsed,
                        importance_num_tap])
                    t.add_row(['SPT', SPT_obj_mc, SPT_elapsed, SPT_num_tap])
                    t.set_style(MSWORD_FRIENDLY)
                    print(t)

                elif alt_crews == None:
                    t = PrettyTable()
                    t.title = net_name + ' with ' + str(num_broken) + ' broken bridges'
                    t.field_names = ['Method', 'Objective', 'Run Time', '# TAP']
                    if opt==1:
                        t.add_row(['OPTIMAL', opt_obj, opt_elapsed, opt_num_tap])
                    if opt==2:
                        t.add_row(['ML BF', opt_obj, opt_elapsed, opt_num_tap])
                    if approx:
                        if approx>=2 or (approx==1 and LAFO_obj<LASR_obj):
                            t.add_row(['approx-LAFO', LAFO_obj, LAFO_elapsed, LAFO_num_tap])
                        if approx>=2 or (approx==1 and LAFO_obj>=LASR_obj):
                            t.add_row(['approx-LASR', LASR_obj, LASR_elapsed, LASR_num_tap])
                        if approx==3:
                            t.add_row(['alt-LASR', altLASR_obj, altLASR_elapsed, altLASR_num_tap])
                    if mip==3:
                        t.add_row(['MIP_delta', mip3_obj, mip3_elapsed, mip3_num_tap])
                    if full:
                        t.add_row(['FULL ALGO', algo_obj, algo_elapsed, algo_num_tap])
                    if beam_search:
                        t.add_row(['BeamSearch_relaxed', r_algo_obj, r_algo_elapsed,
                            r_algo_num_tap])
                        try:
                            t.add_row(['BeamSearch_sa_seed', sa_r_algo_obj, sa_r_algo_elapsed,
                                sa_r_algo_num_tap])
                        except:
                            pass
                    if sa:
                        t.add_row(['Simulated Annealing', sa_obj, sa_elapsed, sa_num_tap])
                    t.add_row(['GREEDY', greedy_obj, greedy_elapsed, greedy_num_tap])
                    t.add_row(['LG', lg_obj, lg_elapsed, lg_num_tap])
                    #t.add_row(['Linear Combination', lc_obj, lc_elapsed, lc_num_tap])
                    t.add_row(['IMPORTANCE', importance_obj, importance_elapsed,
                        importance_num_tap])
                    t.add_row(['SPT', SPT_obj, SPT_elapsed, SPT_num_tap])
                    t.set_style(MSWORD_FRIENDLY)
                    print(t)
                    if num_crews != 1:
                        if opt:
                            order_list = opt_soln
                            print('Mapping crew assignments from the brute force solution')
                        elif sa and beam_search and r_algo_obj <= sa_obj:
                            order_list = r_algo_path
                            print('Mapping crew assignments from the beam search solution')
                        elif sa:
                            order_list = sa_soln
                            print('Mapping crew assignments from the simulated annealing solution')
                        else:
                            order_list = r_algo_path
                            print('Mapping crew assignments from the beam search solution')
                        __, which_crew, __ = gen_crew_order(
                            order_list, damaged_dict=damaged_dict, num_crews=num_crews)
                        plot_nodes_links(save_dir, netg, damaged_links, coord_dict, names=True,
                            num_crews=num_crews, which_crew=which_crew)

                else:
                    t = PrettyTable()
                    t.title = net_name + ' with ' + str(num_broken) + ' broken bridges'
                    t.field_names = ['Method', 'Objectives', 'Run Time', '# TAP']
                    if opt:
                        for num in range(len(alt_crews)+1):
                            if num == 0:
                                if opt==1:
                                    t.add_row(['OPTIMAL '+str(num_crews), opt_obj_mult[num],
                                        opt_elapsed_mult[num], opt_num_tap])
                                if opt==2:
                                    t.add_row(['ML BF '+str(num_crews), opt_obj_mult[num],
                                        opt_elapsed_mult[num], opt_num_tap])
                            else:
                                if opt==1:
                                    t.add_row(['OPTIMAL '+str(alt_crews[num-1]), opt_obj_mult[num],
                                        opt_elapsed_mult[num], opt_num_tap])
                                if opt==2:
                                    t.add_row(['ML BF '+str(alt_crews[num-1]), opt_obj_mult[num],
                                        opt_elapsed_mult[num], opt_num_tap])
                    if approx:
                        if approx>=2 or (approx==1 and LAFO_obj_mult[0]<LASR_obj_mult[0]):
                            t.add_row(['approx-LAFO', LAFO_obj_mult, LAFO_elapsed, LAFO_num_tap])
                        if approx>=2 or (approx==1 and LAFO_obj_mult[0]>=LASR_obj_mult[0]):
                            t.add_row(['approx-LASR', LASR_obj_mult, LASR_elapsed, LASR_num_tap])
                        if approx==3:
                            t.add_row(['alt-LASR', altLASR_obj_mult, altLASR_elapsed, altLASR_num_tap])
                    if mip==3:
                        t.add_row(['MIP_delta', mip3_obj_mult, mip3_elapsed, mip3_num_tap])
                    if full:
                        t.add_row(['FULL ALGO', algo_obj_mult, algo_elapsed, algo_num_tap])
                    if beam_search:
                        t.add_row(['BeamSearch_relaxed', r_algo_obj_mult, r_algo_elapsed,
                            r_algo_num_tap])
                        try:
                            t.add_row(['BeamSearch_sa_seed', sa_r_algo_obj_mult,
                                sa_r_algo_elapsed, sa_r_algo_num_tap])
                        except:
                            pass
                    if sa:
                        t.add_row(['Simulated Annealing', sa_obj_mult, sa_elapsed, sa_num_tap])
                    t.add_row(['GREEDY', greedy_obj_mult, greedy_elapsed, greedy_num_tap])
                    t.add_row(['LG', lg_obj_mult, lg_elapsed, lg_num_tap])
                    #t.add_row(['Linear Combination', lc_obj_mult, lc_elapsed, lc_num_tap])
                    t.add_row(['IMPORTANCE', importance_obj_mult, importance_elapsed,
                        importance_num_tap])
                    t.add_row(['SPT', SPT_obj_mult, SPT_elapsed, SPT_num_tap])
                    t.set_style(MSWORD_FRIENDLY)
                    print(t)

                fname = save_dir + '/results.csv'
                with open(fname, 'w', newline='') as f:
                    f.write(t.get_csv_string(header=False))

                print('Swapped links: ', swapped_links)

                if beam_search:
                    print('r totchild vs uncommon vs common vs purged')
                    print(tot_childr, uncommon_numberr, common_numberr, num_purger)

                print('PATHS')
                if approx:
                    print('approx-LAFO: ', LAFO_soln)
                    print('---------------------------')
                    print('approx-LASR: ', LASR_soln)
                    print('---------------------------')
                    if approx==3:
                        print('alt-LASR: ', altLASR_soln)
                        print('---------------------------')
                if mip==3:
                    print('MIP_delta: ', mip3_soln)
                    print('---------------------------')
                if opt==1:
                    print('optimal by brute force: ', opt_soln)
                    print('---------------------------')
                if opt==2:
                    print('ML brute force: ', opt_soln)
                    print('---------------------------')
                if full:
                    print('full algorithm: ', algo_path)
                    print('---------------------------')
                if beam_search:
                    print('beamsearch: ', r_algo_path)
                    print('---------------------------')
                    try:
                        print('beamsearch with SA seed: ', sa_r_algo_path)
                        print('---------------------------')
                    except:
                        pass
                if sa:
                    print('simulated annealing: ', sa_soln)
                    print('---------------------------')
                print('greedy: ', greedy_soln)
                print('---------------------------')
                print('lazy greedy: ', lg_soln)
                print('---------------------------')
                #print('Linear Combination: ', lc_soln)
                #print('---------------------------')
                print('importance factors: ', importance_soln)
                print('---------------------------')
                print('shortest processing time: ', SPT_soln)

                # Make sequences dict --> csv file
                damaged_seqs = dict()

                for link in damaged_links:
                    damaged_seqs[link] = list()
                temp_dict = {}
                temp_dict['header'] = list()
                if opt==1:
                    temp_dict['header'].append('OPT')
                if opt==2:
                    temp_dict['header'].append('ML BF')
                if approx:
                    temp_dict['header'].append('LAFO')
                    temp_dict['header'].append('LASR')
                    if approx==3:
                        temp_dict['header'].append('altLASR')
                if mip==3:
                    temp_dict['header'].append('MIP_delta')
                if full:
                    temp_dict['header'].append('algo')
                if beam_search:
                    temp_dict['header'].append('BS')
                if sa:
                    temp_dict['header'].append('SA')
                temp_dict['header'].append('Greedy')
                temp_dict['header'].append('LG')
                #temp_dict['header'].append('LinComb')
                temp_dict['header'].append('IF')
                temp_dict['header'].append('SPT')
                temp_dict['header'].append('Dur')
                temp_dict['header'].append('IF')
                temp_dict['header'].append('BB')
                temp_dict['header'].append('swap')

                for link in damaged_links:
                    if opt:
                        el = opt_soln.index(link)
                        damaged_seqs[link].append(el+1)
                    if approx:
                        el = LAFO_soln.index(link)
                        damaged_seqs[link].append(el+1)
                        el = LASR_soln.index(link)
                        damaged_seqs[link].append(el+1)
                        if approx==3:
                            el = altLASR_soln.index(link)
                            damaged_seqs[link].append(el+1)
                    if mip==3:
                        el = mip3_soln.index(link)
                        damaged_seqs[link].append(el+1)
                    if full:
                        el = algo_path.index(link)
                        damaged_seqs[link].append(el+1)
                    if beam_search:
                        el = r_algo_path.index(link)
                        damaged_seqs[link].append(el+1)
                    if sa:
                        el = sa_soln.index(link)
                        damaged_seqs[link].append(el+1)
                    el = greedy_soln.index(link)
                    damaged_seqs[link].append(el+1)
                    el = lg_soln.index(link)
                    damaged_seqs[link].append(el+1)
                    #el = lc_soln.index(link)
                    #damaged_seqs[link].append(el+1)
                    el = importance_soln.index(link)
                    damaged_seqs[link].append(el+1)
                    el = SPT_soln.index(link)
                    damaged_seqs[link].append(el+1)

                # Get repair durations
                for link in damaged_links:
                    damaged_seqs[link].append(damaged_dict[link])

                # Get importance factors
                tot_flow = 0
                if_net = deepcopy(net_before)
                for ij in if_net.linkDict:
                    tot_flow += if_net.linkDict[ij]['flow']
                for link in damaged_links:
                    link_flow = if_net.linkDict[link]['flow']
                    damaged_seqs[link].append(link_flow / tot_flow)

                # Get immediate benefit
                for link in damaged_links:
                    damaged_seqs[link].append(first_b[link])
                    if link in swapped_links:
                        damaged_seqs[link].append('swapped')
                    else:
                        damaged_seqs[link].append('not swapped')

                print('Repair priorities ', damaged_seqs)
                print('Swapped links', swapped_links)

                fname = save_dir + '/damaged_seqs.csv'
                with open(fname, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([str(rep+1)] + temp_dict['header'])
                    for link in damaged_seqs:
                        writer.writerow([link] + damaged_seqs[link])


        # Read soln paths and get obj values for weighted classes after the unweighted class run
        if multiclass and damaged_dict_preset != '':
            # Read soln paths
            if approx:
                LAFO_soln_ineq = load(damaged_dict_preset + '/' + 'LAFO_bound_path')
                LASR_soln_ineq = load(damaged_dict_preset + '/' + 'LASR_bound_path')
                if approx==3:
                    altLASR_soln_ineq = load(damaged_dict_preset + '/' + 'altLASR_bound_path')
            if mip==3:
                mip3_soln_ineq = load(damaged_dict_preset + '/' + 'mip_delta_path')
            if opt==1:
                opt_soln_ineq = load(damaged_dict_preset + '/' + 'min_seq_path')
            if opt==2:
                opt_soln_ineq = load(damaged_dict_preset + '/' + 'min_seq_approx_path')
            if full:
                algo_path_ineq = load(damaged_dict_preset + '/' + 'algo_solution_path')
            if beam_search:
                r_algo_path_ineq = load(damaged_dict_preset + '/' + 'r_algo_solution_k2_path')
                try:
                    sa_r_algo_path_ineq = load(damaged_dict_preset + '/'
                        + 'sa_r_algo_solution_k2_path')
                except:
                    pass
            if sa:
                sa_soln_ineq = load(damaged_dict_preset + '/' + 'sim_anneal_solution_path')
            greedy_soln_ineq = load(damaged_dict_preset + '/' + 'greedy_solution_path')
            lg_soln_ineq = load(damaged_dict_preset + '/' + 'lazygreedy_solution_path')
            #lc_soln_ineq = load(damaged_dict_preset + '/' + 'linear_combination_bound_path')
            importance_soln_ineq = load(damaged_dict_preset + '/' + 'importance_factor_bound_path')
            SPT_soln_ineq = load(damaged_dict_preset + '/' + 'SPT_bound_path')

            # Get obj values
            if approx:
                LAFO_obj_ineq, __, __ = eval_sequence(net_after, LAFO_soln_ineq, after_eq_tstt,
                    before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
                LASR_obj_ineq, __, __ = eval_sequence(net_after, LASR_soln_ineq, after_eq_tstt,
                    before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
                if approx==3:
                    altLASR_obj_ineq, __, __ = eval_sequence(net_after, altLASR_soln_ineq, after_eq_tstt,
                        before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
            if mip==3:
                mip3_obj_ineq, __, __ = eval_sequence(net_after, mip3_soln_ineq, after_eq_tstt,
                    before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
            if opt:
                opt_obj_ineq, __, __ = eval_sequence(net_after, opt_soln_ineq, after_eq_tstt,
                    before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
            if full:
                algo_obj_ineq, __, __ = eval_sequence(net_after, algo_path_ineq, after_eq_tstt,
                    before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
            if beam_search:
                r_algo_obj_ineq, __, __ = eval_sequence(net_after, r_algo_path_ineq,
                    after_eq_tstt, before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
                try:
                    sa_r_algo_obj_ineq, __, __ = eval_sequence(net_after, sa_r_algo_path_ineq,
                        after_eq_tstt, before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
                except:
                    pass
            if sa:
                sa_obj_ineq, __, __ = eval_sequence(net_after, sa_soln_ineq, after_eq_tstt,
                    before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
            greedy_obj_ineq, __, __ = eval_sequence(net_after, greedy_soln_ineq, after_eq_tstt,
                before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
            lg_obj_ineq, __, __ = eval_sequence(net_after, lg_soln_ineq, after_eq_tstt,
                before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
            #lc_obj_ineq, __, __ = eval_sequence(net_after, lc_soln_ineq, after_eq_tstt,
            #    before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
            importance_obj_ineq, __, __ = eval_sequence(net_after, importance_soln_ineq,
                 after_eq_tstt, before_eq_tstt, num_crews=num_crews, multiclass=multiclass)
            SPT_obj_ineq, __, __ = eval_sequence(net_after, SPT_soln_ineq, after_eq_tstt,
                 before_eq_tstt, num_crews=num_crews, multiclass=multiclass)

            # Display results for weighted vs unweighted classes
            t = PrettyTable()
            t.title = (net_name + ' with ' + str(num_broken)
                + ' broken bridges (equal vs unequal priorities)')
            t.field_names = ['Method', 'Equal Class Priority OBJ', 'Unequal Class Priority OBJ',
                '% overall change, % change by class']
            if opt==1:
                t.add_row(['OPTIMAL', opt_obj_mc, opt_obj_ineq, percentChange(opt_obj_mc,
                    opt_obj_ineq)])
            if opt==2:
                t.add_row(['ML BF', opt_obj_mc, opt_obj_ineq, percentChange(opt_obj_mc,
                    opt_obj_ineq)])
            if approx:
                t.add_row(['approx-LAFO', LAFO_obj_mc, LAFO_obj_ineq, percentChange(LAFO_obj_mc,
                    LAFO_obj_ineq)])
                t.add_row(['approx-LASR', LASR_obj_mc, LASR_obj_ineq], percentChange(LASR_obj_mc,
                    LASR_obj_ineq))
                if approx==3:
                    t.add_row(['alt-LASR', altLASR_obj_mc, altLASR_obj_ineq], percentChange(altLASR_obj_mc,
                        altLASR_obj_ineq))
            if mip==3:
                t.add_row(['MIP_delta', mip3_obj_mc, mip3_obj_ineq], percentChange(mip3_obj_mc,
                    mip3_obj_ineq))
            if full:
                t.add_row(['FULL ALGO', algo_obj_mc, algo_obj_ineq, percentChange(algo_obj_mc,
                    algo_obj_ineq)])
            if beam_search:
                t.add_row(['BeamSearch_relaxed', r_algo_obj_mc, r_algo_obj_ineq, percentChange(
                    r_algo_obj_mc, r_algo_obj_ineq)])
                try:
                    t.add_row(['BeamSearch_sa_seed', sa_r_algo_obj_mc, sa_r_algo_obj_ineq,
                        percentChange(sa_r_algo_obj_mc, sa_r_algo_obj_ineq)])
                except:
                    pass
            if sa:
                t.add_row(['Simulated Annealing', sa_obj_mc, sa_obj_ineq, percentChange(
                    sa_obj_mc, sa_obj_ineq)])
            t.add_row(['GREEDY', greedy_obj_mc, greedy_obj_ineq, percentChange(greedy_obj_mc,
                greedy_obj_ineq)])
            t.add_row(['LG', lg_obj_mc, lg_obj_ineq, percentChange(lg_obj_mc, lg_obj_ineq)])
            #t.add_row(['Linear Combination', lc_obj_mc, lc_obj_ineq, percentChange(lc_obj_mc,
            #    lc_obj_ineq)])
            t.add_row(['IMPORTANCE', importance_obj_mc, importance_obj_ineq, percentChange(
                importance_obj_mc, importance_obj_ineq)])
            t.add_row(['SPT', SPT_obj_mc, SPT_obj_ineq, percentChange(SPT_obj_mc, SPT_obj_ineq)])
            t.set_style(MSWORD_FRIENDLY)
            print(t)

            fname = save_dir + '/compare.csv'
            with open(fname, 'w', newline='') as f:
                f.write(t.get_csv_string(header=False))
            if graphing:
                get_sequence_graphs(NETWORK_DIR, str(num_broken), alt_dir=ULT_SCENARIO_REP_DIR,
                    multiclass=True, mc_weights=mc_weights)

        elif graphing:
            get_sequence_graphs(NETWORK_DIR, str(num_broken), mc_weights=mc_weights)

        if graphing:
            plot_time_OBJ(save_dir,bs_time_list,bs_OBJ_list,sa_time_list,sa_OBJ_list)
