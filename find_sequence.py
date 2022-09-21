from sequence_utils import *
import random
import operator as op
from functools import reduce
import itertools
from prettytable import PrettyTable
from prettytable import MSWORD_FRIENDLY
import multiprocessing as mp
from graphing import *
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from scipy.special import comb
import math
import argparse
import numpy as np
import cProfile
import pstats
import networkx as nx
# import caffeine
import os.path
from ctypes import *
import time
from scipy import stats
from correspondence import *
import progressbar
from network import *

extension = '.pickle'

parser = argparse.ArgumentParser(
    description='find an order for repairing bridges')
parser.add_argument('-n', '--net_name', type=str, help='network name')
parser.add_argument('-b', '--num_broken', type=int, help='number of broken bridges')
parser.add_argument('-a', '--approx', type=bool,
                    help='approximation methods enabled - LAFO and LASR', default=False)
parser.add_argument('-r', '--reps', type=int,
                    help='number of scenarios with the given parameters', default=5)
parser.add_argument('-t', '--tables', type=bool, help='table output mode', default=False)
parser.add_argument('-g', '--graphing', type=bool, help='graph solution quality vs runtime', default=False)
parser.add_argument('-l', '--loc', type=int, help='location based sampling', default=0)
parser.add_argument('-o', '--scenario', type=str, help='scenario file', default='scenario.csv')
parser.add_argument('-e', '--strength', type=str, help='strength of the earthquake')
parser.add_argument('-y', '--onlybeforeafter', type=bool, help='to get before and after tstt', default=False)
parser.add_argument('-f', '--full', type=bool, help='to use full algo not beam search', default=False)
parser.add_argument('-s', '--beamsearch', help='use beam search for speed, in order to disable, enter -s without arguments',
                    action='store_false')
parser.add_argument('-j', '--random', help='generate scenarios randomly, in order to disable, enter -j without arguments',
                    action='store_false')
parser.add_argument('-c', '--gamma', type=int, help='hyperparameter to expand the search', default=128)
parser.add_argument('-v', '--beta', type=int,
                    help='hyperparameter that decides the frequency of purge', default=128)
parser.add_argument('-d', '--num_crews', nargs='+', type=int, help='number of work crews available', default=1)
""" when multiple values are given for num_crews, order is found based on first number,
and postprocessing is performed to find OBJ for other crew numbers """
parser.add_argument('--opt', type=bool, help='solve to optimality by brute force', default=False)
parser.add_argument('--sa', type=bool, help='solve using simulated annealing starting at bfs', default=False)
parser.add_argument('--damaged', type=str, help='set damaged_dict to known values to explore various parameters', default='')
parser.add_argument('--mc', type=bool, help='display separate TSTTs for each class of demand', default=False)

args = parser.parse_args()

SEED = 42
FOLDER = "TransportationNetworks"
MAX_DAYS = 180
MIN_DAYS = 21
memory = {}

CORES = min(mp.cpu_count(),4)

class BestSoln():
    """A node class for bi-directional search for pathfinding"""
    def __init__(self):
        self.cost = None
        self.path = None

class Node():
    """A node class for bi-directional search for pathfinding"""

    def __init__(self, visited=None, link_id=None, parent=None, tstt_after=None, tstt_before=None, level=None, forward=True, relax=False, not_fixed=None):

        self.relax = relax
        self.forward = forward
        self.parent = parent
        self.level = level
        self.link_id = link_id
        self.path = []
        self.tstt_before = tstt_before
        self.tstt_after = tstt_after
        self.g = 0
        self.h = 0
        self.f = 0
        if relax:
            self.err_rate = 0.01
        else:
            self.err_rate = 0

        self.assign_char()

    def __str__(self):
        return str(self.path)

    def assign_char(self):

        if self.parent is not None:
            self.benefit = self.tstt_after - self.tstt_before
            self.days = damaged_dict[self.link_id]
            prev_path = deepcopy(self.parent.path)
            prev_path.append(self.link_id)
            self.path = prev_path
            self.visited = set(self.path)
            self.days_past = self.parent.days_past + self.days
            self.before_eq_tstt = self.parent.before_eq_tstt
            self.after_eq_tstt = self.parent.after_eq_tstt

            self.realized = self.parent.realized + \
                (self.tstt_before - self.before_eq_tstt) * self.days

            self.realized_u = self.parent.realized_u + \
                (self.tstt_before - self.before_eq_tstt) * \
                self.days * (1 + self.err_rate)

            self.realized_l = self.parent.realized_l + \
                (self.tstt_before - self.before_eq_tstt) * \
                self.days * (1 - self.err_rate)

            self.not_visited = set(damaged_dict.keys()
                                   ).difference(self.visited)

            self.forward = self.parent.forward
            if self.forward:
                self.not_fixed = self.not_visited
                self.fixed = self.visited

            else:
                self.not_fixed = self.visited
                self.fixed = self.not_visited

        else:
            if self.link_id is not None:
                self.path = [self.link_id]
                self.realized = (self.tstt_before -
                                 self.before_eq_tstt) * self.days
                self.realized_u = (self.tstt_before -
                                   self.before_eq_tstt) * self.days * (1 + self.err_rate)
                self.realized_l = (self.tstt_before -
                                   self.before_eq_tstt) * self.days * (1 - self.err_rate)
                self.days_past = self.days

            else:
                self.realized = 0
                self.realized_u = 0
                self.realized_l = 0
                self.days = 0
                self.days_past = self.days

    def __eq__(self, other):
        return self.fixed == other.fixed


def eval_working_sequence(net, order_list, after_eq_tstt, before_eq_tstt, is_approx=False, damaged_dict=None, num_crews=1, approx_params=None):
    """evaluates the total tstt for a repair sequence, using memory to minimize runtime"""
    tap_solved = 0
    days_list = []
    tstt_list = []
    global memory

    to_visit = order_list
    added = []

    ### crew order list is the order in which projects complete ###
    crew_order_list, which_crew, days_list = gen_crew_order(
        order_list, damaged_dict=damaged_dict, num_crews=num_crews)

    for link_id in crew_order_list:
        added.append(link_id)
        not_fixed = set(to_visit).difference(set(added))
        net.not_fixed = set(not_fixed)

        if is_approx:
            damaged_links = list(damaged_dict.keys())
            state = list(set(damaged_links).difference(net.not_fixed))
            state = [damaged_links.index(i) for i in state]
            pattern = np.zeros(len(damaged_links))
            pattern[(state)] = 1
            tstt_after = approx_params[0].predict(pattern.reshape(1, -1), verbose=0) * approx_params[2] + approx_params[1]
            tstt_after = tstt_after[0][0]
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


def get_successors_f(node, iter_num):
    """given a state, returns list of bridges that have not yet been fixed"""
    not_visited = node.not_visited


    successors = []

    if node.level != 0:
        tail = node.path[-1]
        for a_link in not_visited:
            if iter_num >= 1000:
                if wb[a_link]/damaged_dict[a_link] - bb[tail]/damaged_dict[tail] > 0:
                    continue
            successors.append(a_link)
    else:
        successors = not_visited

    return successors


def get_successors_b(node, iter_num):
    """given a state, returns list of bridges that have not yet been removed"""
    not_visited = node.not_visited
    successors = []

    if node.level != 0:
        tail = node.path[-1]
        for a_link in not_visited:
            if iter_num >= 1000:
                if -wb[tail] * damaged_dict[a_link] + bb[a_link] * damaged_dict[tail] < 0:
                    continue
            successors.append(a_link)
    else:
        successors = not_visited

    return successors


def expand_sequence_f(node, a_link, level):
    """given a link and a node, expands the sequence"""
    solved = 0
    tstt_before = node.tstt_after

    net = create_network(NETFILE, TRIPFILE)

    net.not_fixed = set(node.not_fixed).difference(set([a_link]))
    net.art_links = artLinkDict
    global memory

    if frozenset(net.not_fixed) in memory.keys():
        tstt_after = memory[frozenset(net.not_fixed)]

    else:
        tstt_after = solve_UE(net=net, relax=node.relax)
        memory[frozenset(net.not_fixed)] = (tstt_after)
        solved = 1

    global wb_update
    global bb_update

    diff = tstt_before - tstt_after
    if wb_update[a_link] > diff:
        wb_update[a_link] = diff
    if bb_update[a_link] < diff:
        bb_update[a_link] = diff

    global wb
    global bb
    if wb[a_link] > diff:
        wb[a_link] = diff
    if bb[a_link] < diff:
        bb[a_link] = diff

    node = Node(link_id=a_link, parent=node, tstt_after=tstt_after,
                tstt_before=tstt_before, level=level, relax=node.relax)



    del net
    return node, solved


def expand_sequence_b(node, a_link, level):
    """given a link and a node, expands the sequence"""
    solved = 0
    tstt_after = node.tstt_before

    net = create_network(NETFILE, TRIPFILE)
    net.not_fixed = node.not_fixed.union(set([a_link]))
    net.art_links = artLinkDict

    global memory
    if frozenset(net.not_fixed) in memory.keys():
        tstt_before = memory[frozenset(net.not_fixed)]
    else:
        tstt_before = solve_UE(net=net, relax=node.relax)
        memory[frozenset(net.not_fixed)] = tstt_before
        solved = 1

    global wb_update
    global bb_update

    diff = tstt_before - tstt_after
    if wb_update[a_link] > diff:
        wb_update[a_link] = diff
    if bb_update[a_link] < diff:
        bb_update[a_link] = diff

    global wb
    global bb
    if wb[a_link] > diff:
        wb[a_link] = diff
    if bb[a_link] < diff:
        bb[a_link] = diff

    node = Node(link_id=a_link, parent=node, tstt_after=tstt_after,
                tstt_before=tstt_before, level=level, relax=node.relax)

    del net
    return node, solved


def check(fwd_node, bwd_node, relax):
    """obsolete function"""
    fwdnet = create_network(NETFILE, TRIPFILE)

    fwdnet.not_fixed = fwd_node.not_fixed
    fwdnet.art_links = artLinkDict

    fwd_tstt = solve_UE(net=fwdnet, relax=relax)

    bwdnet = create_network(NETFILE, TRIPFILE)
    bwdnet.not_fixed = bwd_node.not_fixed
    net.art_links = artLinkDict

    bwd_tstt = solve_UE(net=bwdnet, relax=relax)

    return fwd_tstt, bwd_tstt, fwdnet, bwdnet


def get_minlb(node, fwd_node, bwd_node, orderedb_benefits, orderedw_benefits, ordered_days, forward_tstt, backward_tstt, bfs=None, uncommon_number=0, common_number=0):
    """finds upper and lower bounds on tstt between a forward and backward node"""
    slack = forward_tstt - backward_tstt
    if slack < 0:
        backward_tstt_orig = backward_tstt
        backward_tstt = forward_tstt
        forward_tstt = backward_tstt_orig
        slack = forward_tstt - backward_tstt

    if len(ordered_days) == 0:
        node.ub = fwd_node.realized_u + bwd_node.realized_u
        node.lb = fwd_node.realized_l + bwd_node.realized_l
        cur_obj = fwd_node.realized + bwd_node.realized
        new_feasible_path = fwd_node.path + bwd_node.path[::-1]

        if cur_obj < bfs.cost:
            bfs.cost = cur_obj
            bfs.path = new_feasible_path

        return uncommon_number, common_number

    elif len(ordered_days) == 1:
        node.ub = fwd_node.realized_u + bwd_node.realized_u + \
            (fwd_node.tstt_after - node.before_eq_tstt) * ordered_days[0]
        node.lb = fwd_node.realized_l + bwd_node.realized_l + \
            (fwd_node.tstt_after - node.before_eq_tstt) * ordered_days[0]
        cur_obj = fwd_node.realized + bwd_node.realized + \
            (fwd_node.tstt_after - node.before_eq_tstt) * ordered_days[0]

        lo_link = list(set(damaged_dict.keys()).difference(
            set(fwd_node.path).union(set(bwd_node.path))))
        new_feasible_path = fwd_node.path + \
            [str(lo_link[0])] + bwd_node.path[::-1]

        if cur_obj < bfs.cost:
            bfs.cost = cur_obj
            bfs.path = new_feasible_path

        return uncommon_number, common_number

    b, days_b = orderlists(orderedb_benefits, ordered_days, slack)
    w, days_w = orderlists(orderedw_benefits, ordered_days, slack)
    sumw = sum(w)
    sumb = sum(b)

    orig_lb = node.lb
    orig_ub = node.ub


    ### FIND LB FROM FORWARDS ###
    if sumw < slack:

        for i in range(len(days_w)):

            if i == 0:
                fwd_w = deepcopy(w)
                fwd_days_w = deepcopy(days_w)
                b_tstt = sumw

            else:
                b_tstt = b_tstt - fwd_w[0]
                fwd_w = fwd_w[1:]
                fwd_days_w = fwd_days_w[1:]


            node.lb += b_tstt * fwd_days_w[0] + (bwd_node.tstt_before - node.before_eq_tstt) * fwd_days_w[0]
        common_number+=1
    else:
        node.lb += (fwd_node.tstt_after-node.before_eq_tstt)*min(days_w) + (bwd_node.tstt_before - node.before_eq_tstt)*(sum(days_w) - min(days_w))
        uncommon_number+=1

    lb2 = (fwd_node.tstt_after-node.before_eq_tstt)*min(days_w) + (bwd_node.tstt_before - node.before_eq_tstt)*(sum(days_w) - min(days_w))
    node.lb = max(node.lb, lb2 + orig_lb)

    ### FIND UB FROM FORWARDS ###
    if sumb > slack:
        for i in range(len(days_b)):

            if i == 0:
                fwd_b = deepcopy(b)
                fwd_days_b = deepcopy(days_b)
                fwd_b, fwd_days_b = orderlists(fwd_b, fwd_days_b, reverse=False)
                b_tstt = sumb

            else:
                b_tstt = b_tstt - fwd_b[0]
                fwd_b = fwd_b[1:]
                fwd_days_b = fwd_days_b[1:]

            node.ub += b_tstt * fwd_days_b[0] + (bwd_node.tstt_before - node.before_eq_tstt) * fwd_days_b[0]

        common_number += 1
    else:
        node.ub += sum(days_b) * (forward_tstt - node.before_eq_tstt)
        uncommon_number += 1

    node.ub = min(node.ub, sum(days_b) * (forward_tstt - node.before_eq_tstt) + orig_ub)

    if node.lb > node.ub:

        if node.relax == False:
            pass

        if abs(node.lb - node.ub) < 5:
            tempub = node.ub
            node.ub = node.lb
            node.lb = tempub

        else:
            print('node.lb > node.ub for node : ' + str(node) + ' and bwd node : ' + str(bwd_node) + '. Forward TSTT = ' + str(forward_tstt) + ' and backward TSTT = ' + str(backward_tstt))
            print('Worst benefits : ' + str(w) + ' and best benefits : ' + str(b))
            pass

    return uncommon_number, common_number


def set_bounds_bif(node, open_list_b, end_node, front_to_end=True, bfs=None, uncommon_number=0, common_number=0):
    """given a forward child node, set upper and lower bounds on remaining path"""
    sorted_d = sorted(damaged_dict.items(), key=lambda x: x[1])

    remaining = []
    eligible_backward_connects = []

    if front_to_end:
        eligible_backward_connects = [end_node]
    else:
        for other_end in sum(open_list_b.values(),[]):
            if len(set(node.visited).intersection(other_end.visited)) == 0:
                eligible_backward_connects.append(other_end)

    minlb = np.inf
    maxub = np.inf

    if len(eligible_backward_connects) == 0:
        eligible_backward_connects = [end_node]
    # if min(open_list_b.keys()) >= 5:
    #     eligible_backward_connects.append(end_node)


    minother_end = None
    for other_end in eligible_backward_connects:
        ordered_days = []
        orderedw_benefits = []
        orderedb_benefits = []

        node.ub = node.realized_u + other_end.realized_u
        node.lb = node.realized + other_end.realized

        union = node.visited.union(other_end.visited)
        remaining = set(damaged_dict.keys()).difference(union)

        for key, value in sorted_d:
            # print("%s: %s" % (key, value))
            if key in remaining:
                ordered_days.append(value)
                orderedw_benefits.append(wb[key])
                orderedb_benefits.append(bb[key])

        forward_tstt = node.tstt_after
        backward_tstt = other_end.tstt_before


        uncommon_number, common_number = get_minlb(node, node, other_end, orderedb_benefits,
                  orderedw_benefits, ordered_days, forward_tstt, backward_tstt, bfs=bfs, uncommon_number=uncommon_number, common_number=common_number)

        if node.lb < minlb:
            minlb = node.lb
            minother_end = other_end

        if node.ub < maxub:
            maxub = node.ub

    node.lb = minlb
    node.ub = maxub


    if node.lb > node.ub:
        print('node.lb > node.ub for node : ' + str(node) + ' and bwd node : ' + str(bwd_node) + '. Forward TSTT = ' + str(forward_tstt) + ' and backward TSTT = ' + str(backward_tstt))
        print('Worst benefits : ' + str(w) + ' and best benefits : ' + str(b))

    return uncommon_number, common_number


def set_bounds_bib(node, open_list_f, start_node, front_to_end=True, bfs=None, uncommon_number=0, common_number=0):
    """given a backward child node, set upper and lower bounds on remaining path"""
    sorted_d = sorted(damaged_dict.items(), key=lambda x: x[1])

    remaining = []
    eligible_backward_connects = []

    if front_to_end:
        eligible_backward_connects = [start_node]
    else:
        for other_end in sum(open_list_f.values(), []):
            if len(set(node.visited).intersection(other_end.visited)) == 0:
                eligible_backward_connects.append(other_end)

    minlb = np.inf
    maxub = np.inf

    if len(eligible_backward_connects) == 0:
        eligible_backward_connects = [start_node]

    # if min(open_list_f.keys()) >= 5:
    #     eligible_backward_connects.append(start_node)

    minother_end = None
    for other_end in eligible_backward_connects:
        ordered_days = []
        orderedw_benefits = []
        orderedb_benefits = []

        node.ub = node.realized_u + other_end.realized_u
        node.lb = node.realized + other_end.realized

        union = node.visited.union(other_end.visited)
        remaining = set(damaged_dict.keys()).difference(union)

        for key, value in sorted_d:
            if key in remaining:
                ordered_days.append(value)
                orderedw_benefits.append(wb[key])
                orderedb_benefits.append(bb[key])
        forward_tstt = other_end.tstt_after
        backward_tstt = node.tstt_before



        uncommon_number, common_number = get_minlb(node, other_end, node, orderedb_benefits, orderedw_benefits,
                  ordered_days, forward_tstt, backward_tstt, bfs=bfs, uncommon_number=uncommon_number, common_number=common_number)


        if node.lb < minlb:
            minlb = node.lb
            minother_end = other_end
        if node.ub < maxub:
            maxub = node.ub

    node.lb = minlb
    node.ub = maxub


    if node.lb > node.ub:
        print('node.lb > node.ub for node : ' + str(node) + ' and bwd node : ' + str(bwd_node) + '. Forward TSTT = ' + str(forward_tstt) + ' and backward TSTT = ' + str(backward_tstt))
        print('Worst benefits : ' + str(w) + ' and best benefits : ' + str(b))
        pass

    return uncommon_number, common_number


def expand_forward(start_node, end_node, minimum_ff_n, open_list_b, open_list_f, bfs, num_tap_solved, max_level_b, closed_list_f, closed_list_b, iter_num, front_to_end=False, uncommon_number=0, tot_child=0, common_number=0):
    """expand the search tree forwards"""
    fvals = [node.f for node in sum(open_list_f.values(),[])]

    update_bfs = False
    minfind = np.argmin(fvals)
    minimum_ff_n = sum(open_list_f.values(),[])[minfind]
    current_node = minimum_ff_n
    open_list_f[minimum_ff_n.level].remove(minimum_ff_n)

    if len(open_list_f[minimum_ff_n.level]) == 0:
        del open_list_f[minimum_ff_n.level]

    cur_visited = current_node.visited

    can1, can2, can3, can4 = False, False, False, False
    if len(damaged_dict)-len(cur_visited) in open_list_b.keys():
        can1 = True
    if len(damaged_dict)-len(cur_visited) - 1 in open_list_b.keys():
        can2 = True
    # if len(damaged_dict)-len(cur_visited) in closed_list_b.keys():
    #     can3 = True
    # if len(damaged_dict)-len(cur_visited) - 1 in closed_list_b.keys():
    #     can4 = True

    go_through = []
    if can1:
        go_through.extend(open_list_b[len(damaged_dict) - len(cur_visited)])
    if can2:
        go_through.extend(open_list_b[len(damaged_dict) - len(cur_visited) - 1])
    # if can3:
    #     go_through.extend(closed_list_b[len(damaged_dict) - len(cur_visited)])
    # if can4:
    #     go_through.extend(closed_list_b[len(damaged_dict) - len(cur_visited) - 1])

    for other_end in go_through:

        if len(set(other_end.visited).intersection(set(cur_visited))) == 0 and len(damaged_dict) - len(set(other_end.visited).union(set(cur_visited))) == 1:
            lo_link = set(damaged_dict.keys()).difference(
                set(other_end.visited).union(set(cur_visited)))
            lo_link = lo_link.pop()
            cur_soln = current_node.g + other_end.g + \
                (current_node.tstt_after - current_node.before_eq_tstt) * \
                damaged_dict[lo_link]
            cur_path = current_node.path + \
                [str(lo_link)] + other_end.path[::-1]

            if cur_soln <= bfs.cost:
                bfs.cost = cur_soln
                bfs.path = cur_path
                update_bfs = True

        if set(other_end.not_visited) == set(cur_visited):
            cur_soln = current_node.g + other_end.g
            cur_path = current_node.path + other_end.path[::-1]
            if cur_soln <= bfs.cost:

                bfs.cost = cur_soln
                bfs.path = cur_path
                update_bfs = True

    # Found the goal
    if current_node == end_node:
        return open_list_f, num_tap_solved, current_node.level, minimum_ff_n, tot_child, uncommon_number, common_number, update_bfs

    if iter_num > 3*len(damaged_dict):
        if current_node.f > bfs.cost:
            print('Iter ' + str(iter_num) + ' current node ' + str(current_node) + ' is pruned.')
            return open_list_f, num_tap_solved, current_node.level, minimum_ff_n, tot_child, uncommon_number, common_number, update_bfs

    # Generate children
    eligible_expansions = get_successors_f(current_node, iter_num)
    children = []

    current_level = current_node.level
    for a_link in eligible_expansions:

        # Create new node
        current_level = current_node.level + 1
        new_node, solved = expand_sequence_f(
            current_node, a_link, level=current_level)
        num_tap_solved += solved
        new_node.g = new_node.realized

        # Append
        append = True
        removal = False

        if current_level in open_list_f.keys():
            for open_node in open_list_f[current_level]:
                if open_node == new_node:
                    if new_node.g >= open_node.g:
                        append = False
                        break
                    else:
                        removal = True
                        break

        if removal:
            open_list_f[current_level].remove(open_node)
        if append:
            children.append(new_node)

    del current_node

    # Loop through children
    for child in children:

        # set upper and lower bounds
        tot_child += 1

        uncommon_number, common_number = set_bounds_bif(child, open_list_b,
                       front_to_end=front_to_end, end_node=end_node, bfs=bfs, uncommon_number=uncommon_number, common_number=common_number)


        if child.ub <= bfs.cost:
            # best_ub = child.ub
            if len(child.path) == len(damaged_dict):
                if child.g < bfs.cost:
                    bfs.cost = child.g
                    bfs.path = child.path
                    update_bfs = True

        if child.lb == child.ub:
            continue

        if child.lb > child.ub:
            print('node.lb > node.ub for node : ' + str(node) + ' and bwd node : ' + str(bwd_node) + '. Forward TSTT = ' + str(forward_tstt) + ' and backward TSTT = ' + str(backward_tstt))
            print('Worst benefits : ' + str(w) + ' and best benefits : ' + str(b))
            pass

        child.g = child.realized
        child.f = child.lb

        if iter_num > 3*len(damaged_dict):
            if child.f > bfs.cost:
                #print('Iter ' + str(iter_num) + ' Child node ' + str(child) + ' not added because child.f < bfs.cost.')
                continue

        if current_level not in open_list_f.keys():
            open_list_f[current_level] = []

        open_list_f[current_level].append(child)
        del child

    return open_list_f, num_tap_solved, current_level, minimum_ff_n, tot_child, uncommon_number, common_number, update_bfs


def expand_backward(start_node, end_node, minimum_bf_n, open_list_b, open_list_f, bfs, num_tap_solved, max_level_f, closed_list_f, closed_list_b, iter_num, front_to_end=False, uncommon_number=0, tot_child=0, common_number=0):
    """expand the search tree forwards"""
    fvals = [node.f for node in sum(open_list_b.values(),[])]

    update_bfs = False
    minfind = np.argmin(fvals)
    minimum_bf_n = sum(open_list_b.values(),[])[minfind]
    current_node = minimum_bf_n
    open_list_b[minimum_bf_n.level].remove(minimum_bf_n)


    if len(open_list_b[minimum_bf_n.level]) == 0:
        del open_list_b[minimum_bf_n.level]

    cur_visited = current_node.visited

    can1, can2, can3, can4 = False, False, False, False
    if len(damaged_dict)-len(cur_visited) in open_list_f.keys():
        can1 = True
    if len(damaged_dict)-len(cur_visited) - 1 in open_list_f.keys():
        can2 = True
    # if len(damaged_dict)-len(cur_visited) in closed_list_f.keys():
    #     can3 = True
    # if len(damaged_dict)-len(cur_visited) - 1 in closed_list_f.keys():
    #     can4 = True

    go_through = []
    if can1:
        go_through.extend(open_list_f[len(damaged_dict)-len(cur_visited)])
    if can2:
        go_through.extend(open_list_f[len(damaged_dict)-len(cur_visited)-1])
    # if can3:
    #     go_through.extend(closed_list_f[len(damaged_dict) - len(cur_visited)])
    # if can4:
    #     go_through.extend(closed_list_f[len(damaged_dict) - len(cur_visited) - 1])

    for other_end in go_through:

        if len(set(other_end.visited).intersection(set(cur_visited))) == 0 and len(damaged_dict) - len(set(other_end.visited).union(set(cur_visited))) == 1:
            lo_link = set(damaged_dict.keys()).difference(
                set(other_end.visited).union(set(cur_visited)))
            lo_link = lo_link.pop()
            cur_soln = current_node.g + other_end.g + \
                (other_end.tstt_after - other_end.before_eq_tstt) * \
                damaged_dict[lo_link]

            if cur_soln <= bfs.cost:
                bfs.cost = cur_soln
                bfs.path = other_end.path + \
                    [str(lo_link)] + current_node.path[::-1]
                update_bfs = True

        elif set(other_end.not_visited) == set(cur_visited):
            cur_soln = current_node.g + other_end.g
            if cur_soln <= bfs.cost:
                bfs.cost = cur_soln
                bfs.path = other_end.path + \
                    current_node.path[::-1]
                update_bfs = True

    # Found the goal
    if current_node == start_node:
        return open_list_b, num_tap_solved, current_node.level, minimum_bf_n, tot_child, uncommon_number, common_number, update_bfs

    if iter_num > 3*len(damaged_dict):
        if current_node.f > bfs.cost:
            print('Iter ' + str(iter_num) + ' current node ' + str(current_node) + ' is pruned.')
            return open_list_b, num_tap_solved, current_node.level, minimum_bf_n, tot_child, uncommon_number, common_number, update_bfs

    # Generate children
    eligible_expansions = get_successors_b(current_node, iter_num)
    children = []
    current_level = current_node.level

    for a_link in eligible_expansions:

        # Create new node
        current_level = current_node.level + 1
        new_node, solved = expand_sequence_b(
            current_node, a_link, level=current_level)
        num_tap_solved += solved

        new_node.g = new_node.realized

        append = True
        removal = False

        if current_level in open_list_b.keys():

            for open_node in open_list_b[current_level]:
                if open_node == new_node:
                    if new_node.g >= open_node.g:
                        append = False
                        break
                    else:
                        removal = True
                        break

        if removal:
            open_list_b[current_level].remove(open_node)
        if append:
            children.append(new_node)

    del current_node

    # Loop through children
    for child in children:

        # set upper and lower bounds
        tot_child += 1
        uncommon_number, common_number = set_bounds_bib(child, open_list_f,
                       front_to_end=front_to_end, start_node=start_node, bfs=bfs, uncommon_number=uncommon_number, common_number=common_number)

        if child.ub <= bfs.cost:
            # best_ub = child.ub
            if len(child.path) == len(damaged_dict):
                if child.g < bfs.cost:
                    bfs.cost = child.g
                    bfs.path = child.path[::-1]
                    update_bfs = True

        if child.lb == child.ub:
            continue

        if child.lb > child.ub:
            print('node.lb > node.ub for node : ' + str(node) + ' and bwd node : ' + str(bwd_node) + '. Forward TSTT = ' + str(forward_tstt) + ' and backward TSTT = ' + str(backward_tstt))
            print('Worst benefits : ' + str(w) + ' and best benefits : ' + str(b))
            pass

        child.g = child.realized
        child.f = child.lb

        if iter_num > 3*len(damaged_dict):
            if child.f > bfs.cost:
                #print('Iter ' + str(iter_num) + ' Child node ' + str(child) + ' not added because child.f < bfs.cost.')
                continue

        if current_level not in open_list_b.keys():
            open_list_b[current_level] = []
        open_list_b[current_level].append(child)
        del child

    return open_list_b, num_tap_solved, current_level, minimum_bf_n, tot_child, uncommon_number, common_number, update_bfs


def purge(open_list_b, open_list_f, beam_k, num_purged, len_f, len_b, closed_list_f, closed_list_b, f_activated, b_activated, minimum_ff_n, minimum_bf_n, iter_num, beta=256, n_0=0.1):
    """purge nodes as part of beam search"""
    if f_activated:
        for i in range(2, len(damaged_dict) + 1):

            if i in open_list_f.keys():
                values_ofn = np.ones((beam_k)) * np.inf
                indices_ofn = np.ones((beam_k)) * np.inf
                keep_f = []
                if len(open_list_f[i]) == 0:
                    del open_list_f[i]
                    continue

                for idx, ofn in enumerate(open_list_f[i]):
                    cur_lev = ofn.level

                    try:
                        cur_max = np.max(values_ofn[:])
                        max_idx = np.argmax(values_ofn[:])

                    except:
                        pdb.set_trace()

                    if ofn.f < cur_max:
                        indices_ofn[max_idx] = idx
                        values_ofn[max_idx] = ofn.f
                    # else:
                    #     keep_f.append(idx)

                indices_ofn = indices_ofn.ravel()
                indices_ofn = indices_ofn[indices_ofn < 1000]
                keepinds = np.concatenate((np.array(keep_f), indices_ofn), axis=None).astype(int)

                if len(indices_ofn) > 0:
                    num_purged += len(open_list_f[i]) - len(indices_ofn)
                closed_list_f[i] = list(
                    np.array(open_list_f[i])[list(set(range(len(open_list_f[i]))).difference(set(keepinds)))])
                open_list_f[i] = list(np.array(open_list_f[i])[keepinds])

                try:
                    olf_node = open_list_f[i][np.argmin([node.f for node in open_list_f[i]])]
                    olf_min = olf_node.f
                    olf_ub = olf_node.ub
                except ValueError:
                    continue
                removed_from_closed = []


                for a_node in closed_list_f[i]:

                    if a_node.lb <= olf_min * (1 + n_0*0.5**(math.floor(iter_num/beta))):
                        open_list_f[i].append(a_node)
                        removed_from_closed.append(a_node)
                    elif a_node.ub <= olf_ub:
                        open_list_f[i].append(a_node)
                        removed_from_closed.append(a_node)

                for a_node in removed_from_closed:
                    closed_list_f[i].remove(a_node)
                del removed_from_closed


    if b_activated:
        for i in range(2, len(damaged_dict)+1):

            if i in open_list_b.keys():
                keep_b = []
                values_ofn = np.ones((beam_k)) * np.inf
                indices_ofn = np.ones((beam_k)) * np.inf

                if len(open_list_b[i]) == 0:
                    del open_list_b[i]
                    continue

                for idx, ofn in enumerate(open_list_b[i]):

                    cur_lev = ofn.level
                    try:
                        cur_max = np.max(values_ofn[:])
                        max_idx = np.argmax(values_ofn[:])
                    except:
                        pass

                    if ofn.f < cur_max:
                        indices_ofn[max_idx] = idx
                        values_ofn[max_idx] = ofn.f
                    # else:
                    #     keep_b.append(idx)

                indices_ofn = indices_ofn.ravel()
                indices_ofn = indices_ofn[indices_ofn < 1000]
                keepinds = np.concatenate((np.array(keep_b), indices_ofn), axis=None).astype(int)

                if len(indices_ofn) > 0:
                    num_purged += len(open_list_b[i]) - len(indices_ofn)
                closed_list_b[i] = list(np.array(open_list_b[i])[list(set(range(len(open_list_b[i]))).difference(set(keepinds)))])
                open_list_b[i] = list(np.array(open_list_b[i])[keepinds])

                try:
                    olb_node = open_list_b[i][np.argmin([node.f for node in open_list_b[i]])]
                    olb_min = olb_node.f
                    olb_ub = olb_node.ub
                except ValueError:
                    continue
                removed_from_closed = []


                for a_node in closed_list_b[i]:
                        if a_node.lb <= olb_min * (1 + n_0*0.5**(math.floor(iter_num/beta))):
                            open_list_b[i].append(a_node)
                            removed_from_closed.append(a_node)
                        elif a_node.ub <= olb_ub:
                            open_list_b[i].append(a_node)
                            removed_from_closed.append(a_node)

                for a_node in removed_from_closed:
                    closed_list_b[i].remove(a_node)
                del removed_from_closed

    return open_list_b, open_list_f, num_purged,  f_activated, b_activated, {}, {}
    # return open_list_b, open_list_f, num_purged,  f_activated, b_activated, closed_list_f, closed_list_b


def search(start_node, end_node, bfs, beam_search=False, beam_k=None, get_feas=True, beta=128, gamma=128):
    """Returns the best order to visit the set of nodes"""

    # ideas in Holte: (search that meets in the middle)
    # another idea is to expand the min priority, considering both backward and forward open list
    # pr(n) = max(f(n), 2g(n)) - min priority expands
    # U is best solution found so far - stops when - U <=max(C,fminF,fminB,gminF +gminB + eps)
    # this is for front to end

    # bfs comes from the greedy heuristic or importance solution
    print('starting search with a feasible solution: ', bfs.cost)
    max_level_b = 0
    max_level_f = 0

    iter_count = 0
    kb, kf = 0, 0
    damaged_links = damaged_dict.keys()

    # Initialize both open and closed list for forward and backward directions
    open_list_f = {}
    open_list_f[max_level_f]=[start_node]
    closed_list_f = {}
    open_list_b = {}
    open_list_b[max_level_b]=[end_node]
    closed_list_b = {}

    minimum_ff_n = start_node
    minimum_bf_n = end_node

    num_tap_solved = 0
    f_activated, b_activated = False, False
    tot_child = 0
    uncommon_number = 0
    common_number = 0
    num_purged = 0

    len_f = len(sum(open_list_f.values(), []))
    len_b = len(sum(open_list_b.values(), []))

    if graphing:
        global bs_time_list
        global bs_OBJ_list
        bs_time_list.append(0)
        bs_OBJ_list.append(deepcopy(bfs.cost))
    search_timer = time.time()
    while len_f > 0 or len_b > 0:
        iter_count += 1
        search_direction = 'Forward'

        len_f = len(sum(open_list_f.values(), []))
        len_b = len(sum(open_list_b.values(), []))

        if len_f <= len_b and len_f != 0:
            search_direction = 'Forward'
        else:
            if len_f != 0:
                search_direction = 'Backward'
            else:
                search_direction = 'Forward'

        if search_direction == 'Forward':
            open_list_f, num_tap_solved, level_f, minimum_ff_n, tot_child, uncommon_number, common_number, update_bfs = expand_forward(
                start_node, end_node, minimum_ff_n, open_list_b, open_list_f, bfs, num_tap_solved, max_level_b, closed_list_f, closed_list_b, iter_count, front_to_end=False, uncommon_number=uncommon_number, tot_child=tot_child, common_number=common_number)
            max_level_f = max(max_level_f, level_f)

        else:
            open_list_b, num_tap_solved, level_b, minimum_bf_n, tot_child, uncommon_number, common_number, update_bfs = expand_backward(
                start_node, end_node, minimum_bf_n , open_list_b, open_list_f, bfs, num_tap_solved, max_level_f, closed_list_f, closed_list_b, iter_count, front_to_end=False, uncommon_number=uncommon_number, tot_child=tot_child, common_number=common_number)
            max_level_b = max(max_level_b, level_b)

        if update_bfs and graphing:
            bs_time_list.append(time.time() - search_timer)
            bs_OBJ_list.append(deepcopy(bfs.cost))

        all_open_f = sum(open_list_f.values(), [])
        all_open_b = sum(open_list_b.values(), [])

        len_f = len(all_open_f)
        len_b = len(all_open_b)

        if (len_f == 0 or len_b == 0) and bfs.path is not None:
            print('one of them is 0 length')
            return bfs.path, bfs.cost, num_tap_solved, tot_child, uncommon_number, common_number, num_purged

        if max(minimum_bf_n.f, minimum_ff_n.f) >= bfs.cost and bfs.path is not None:
            print('min lb bigger than feasible upper bound')
            return bfs.path, bfs.cost, num_tap_solved, tot_child, uncommon_number, common_number, num_purged

        #stop after 2000 iterations
        if iter_count >= 2000:
            return bfs.path, bfs.cost, num_tap_solved, tot_child, uncommon_number, common_number, num_purged

        fvals = [node.f for node in all_open_f]
        minfind = np.argmin(fvals)
        minimum_ff_n = all_open_f[minfind]

        bvals = [node.f for node in all_open_b]
        minbind = np.argmin(bvals)
        minimum_bf_n = all_open_b[minbind]

        #every 64 (128) iter, update bounds
        if iter_count<512:
            up_freq = 64
        else:
            up_freq = 128

        if iter_count % up_freq==0 and iter_count!=0:
            print('----------')
            print(f'search time elapsed: {(time.time() - search_timer)/60}')
            print(f'max_level_f: {max_level_f}, max_level_b: {max_level_b}')
            print('iteration: {}, best_found: {}'.format(iter_count, bfs.cost))
            if graphing:
                bs_time_list.append(time.time() - search_timer)
                bs_OBJ_list.append(deepcopy(bfs.cost))

            global wb
            global bb
            global wb_update
            global bb_update

            wb = deepcopy(wb_update)
            bb = deepcopy(bb_update)
            for k,v in open_list_f.items():
                for a_node in v:
                    _, _ = set_bounds_bif(a_node, open_list_b, front_to_end=False, end_node=end_node, bfs=bfs,uncommon_number=uncommon_number, common_number=common_number)

                    a_node.f = a_node.lb

            for k, v in open_list_b.items():
                for a_node in v:

                    _, _ = set_bounds_bib(a_node, open_list_f, front_to_end=False, start_node=start_node,
                                          bfs=bfs,
                                          uncommon_number=uncommon_number,
                                          common_number=common_number)
                    a_node.f = a_node.lb

        # #every gamma iters get feasible solution
        if iter_count % gamma==0 and iter_count!=0:

            if get_feas:
                # forwards
                # accrued = minimum_ff_n.realized
                remaining = set(damaged_dict.keys()).difference(minimum_ff_n.visited)
                tot_days = 0
                for ll in remaining:
                    tot_days += damaged_dict[ll]

                eligible_to_add = list(deepcopy(remaining))
                decoy_dd = deepcopy(damaged_dict)

                for visited_links in minimum_ff_n.visited:
                    del decoy_dd[visited_links]

                after_ = minimum_ff_n.tstt_after
                test_net = create_network(NETFILE, TRIPFILE)
                test_net.art_links = artLinkDict
                path = minimum_ff_n.path.copy()
                new_bb = {}
                for i in range(len(remaining)):

                    new_tstts = []
                    for link in eligible_to_add:

                        added = [link]

                        not_fixed = set(eligible_to_add).difference(set(added))
                        test_net.not_fixed = set(not_fixed)

                        after_fix_tstt = solve_UE(net=test_net)

                        diff = after_ - after_fix_tstt
                        new_bb[link] = diff


                        if wb_update[link] > diff:
                            wb_update[link] = diff
                        if bb_update[link] < diff:
                            bb_update[link] = diff

                        if wb[link] > diff:
                            wb[link] = diff
                        if bb[link] < diff:
                            bb[link] = diff

                        new_tstts.append(after_fix_tstt)

                    ordered_days = []
                    orderedb_benefits = []

                    sorted_d = sorted(decoy_dd.items(), key=lambda x: x[1])
                    for key, value in sorted_d:
                        ordered_days.append(value)
                        orderedb_benefits.append(new_bb[key])

                    llll, lll, ord = orderlists(orderedb_benefits, ordered_days, rem_keys=sorted_d)

                    link_to_add = ord[0][0]
                    path.append(link_to_add)
                    min_index = eligible_to_add.index(link_to_add)
                    after_ = new_tstts[min_index]
                    eligible_to_add.remove(link_to_add)
                    decoy_dd = deepcopy(decoy_dd)
                    del decoy_dd[link_to_add]

                net = deepcopy(net_after)
                bound, _, _ = eval_sequence(net, path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews)
                if bound < bfs.cost:

                    bfs.cost = bound
                    bfs.path = path


                # backwards
                remaining = set(damaged_dict.keys()).difference(minimum_bf_n.visited)
                tot_days = 0
                for ll in remaining:
                    tot_days += damaged_dict[ll]

                eligible_to_add = list(deepcopy(remaining))
                after_ = after_eq_tstt
                test_net = create_network(NETFILE, TRIPFILE)
                test_net.art_links = artLinkDict

                decoy_dd = deepcopy(damaged_dict)

                for visited_links in minimum_bf_n.visited:
                    del decoy_dd[
                        visited_links]

                path = []
                new_bb = {}

                for i in range(len(remaining)):
                    improvements = []
                    new_tstts = []
                    for link in eligible_to_add:

                        added = [link]

                        not_fixed = set(eligible_to_add).difference(set(added))
                        test_net.not_fixed = set(not_fixed)

                        after_fix_tstt = solve_UE(net=test_net)

                        diff = after_ - after_fix_tstt
                        new_bb[link] = diff


                        if wb_update[link] > diff:
                            wb_update[link] = diff
                        if bb_update[link] < diff:
                            bb_update[link] = diff

                        if wb[link] > diff:
                            wb[link] = diff
                        if bb[link] < diff:
                            bb[link] = diff

                        new_tstts.append(after_fix_tstt)

                    ordered_days = []
                    orderedb_benefits = []

                    sorted_d = sorted(decoy_dd.items(), key=lambda x: x[1])
                    for key, value in sorted_d:
                        ordered_days.append(value)
                        orderedb_benefits.append(new_bb[key])

                    llll, lll, ord = orderlists(orderedb_benefits, ordered_days, rem_keys=sorted_d)


                    link_to_add = ord[0][0]
                    path.append(link_to_add)
                    min_index = eligible_to_add.index(link_to_add)
                    after_ = new_tstts[min_index]
                    eligible_to_add.remove(link_to_add)
                    decoy_dd = deepcopy(decoy_dd)
                    del decoy_dd[link_to_add]


                path.extend(minimum_bf_n.path[::-1])
                net = deepcopy(net_after)
                bound, _, _ = eval_sequence(net, path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews)

                if bound < bfs.cost:
                    bfs.cost = bound
                    bfs.path = path

            if graphing:
                bs_time_list.append(time.time() - search_timer)
                bs_OBJ_list.append(deepcopy(bfs.cost))

            seq_length = len(damaged_links)
            threshold_start = seq_length + int(seq_length**2/int(math.log(seq_length,2)))

            if len_f >= threshold_start:
                f_activated = True
            if len_b >= threshold_start:
                b_activated = True

            if f_activated or b_activated:
                print('Iter ' + str(iter_count) + ': len_f ' + str(len_f) + ' and len_b ' + str(len_b) + ' about to purge')

                if beam_search:
                    open_list_b, open_list_f, num_purged, f_activated, b_activated, closed_list_f, closed_list_b = purge(
                    open_list_b, open_list_f, beam_k, num_purged, len_f, len_b, closed_list_f, closed_list_b, f_activated, b_activated, minimum_ff_n, minimum_bf_n, iter_count, beta=beta)

    #check in case something weird happened
    if len(bfs.path) != len(damaged_links):
        print('{} is not {}'.format('bestpath', 'full length'))

    if graphing:
        bs_time_list.append(time.time() - search_timer)
        bs_OBJ_list.append(deepcopy(bfs.cost))

    return bfs.path, bfs.cost, num_tap_solved, tot_child, uncommon_number, common_number, num_purged


def worst_benefit(before, links_to_remove, before_eq_tstt, relax=False, bsearch=False, ext_name=''):
    """builds worst benefits dict by finding benefit of repairing each link last"""
    if relax:
        ext_name = '_relax'
    elif bsearch:
        ext_name = ext_name
    fname = before.save_dir + '/worst_benefit_dict' + ext_name
    if not os.path.exists(fname + extension):
        # print('worst benefit analysis is running ...')
        # for each bridge, find the effect on TSTT when that bridge is removed
        # while keeping others

        wb = {}
        not_fixed = []
        for link in links_to_remove:
            test_net = deepcopy(before)
            not_fixed = [link]
            test_net.not_fixed = set(not_fixed)

            tstt = solve_UE(net=test_net, eval_seq=True)
            global memory
            memory[frozenset(test_net.not_fixed)] = tstt
            wb[link] = tstt - before_eq_tstt
            # print(link, wb[link])

        save(fname, wb)
    else:
        wb = load(fname)

    return wb


def best_benefit(after, links_to_remove, after_eq_tstt, relax=False, bsearch=False, ext_name=''):
    """builds best benefits dict by finding benefit of repairing each link first"""
    if relax:
        ext_name = '_relax'
    elif bsearch:
        ext_name = ext_name
    fname = after.save_dir + '/best_benefit_dict' + ext_name

    if not os.path.exists(fname + extension):
        start = time.time()
        # print('best benefit analysis is running ...')

        # for each bridge, find the effect on TSTT when that bridge is removed
        # while keeping others
        bb = {}
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
            bb[link] = after_eq_tstt - tstt_after
            # print(link, bb[link])

        elapsed = time.time() - start
        save(fname, bb)

    else:
        bb = load(fname)

    return bb, elapsed


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
        net_after = create_network(NETFILE, TRIPFILE)
        net_after.not_fixed = set(damaged_links)
        net_after.art_links = artLinkDict
        after_eq_tstt = solve_UE(net=net_after, eval_seq=True, wu=False)
        global memory
        memory[frozenset(net_after.not_fixed)] = after_eq_tstt
        elapsed = time.time() - start
        save(fname, net_after)
        save(fname + '_tstt', after_eq_tstt)
        # shutil.copy('batch0.bin', 'after/batch0.bin')
        return net_after, after_eq_tstt, elapsed
    else:
        net_after = load(fname)
        after_eq_tstt = load(fname + '_tstt')

    # shutil.copy('batch0.bin', 'after/batch0.bin')
    return net_after, after_eq_tstt


def state_before(damaged_links, save_dir, relax=False, real=False, bsearch=False, ext_name=''):
    """creates network and solves for tstt before damage occured (also after all repairs are complete)"""
    if relax:
        ext_name = '_relax'
    elif real:
        ext_name = '_real'

    elif bsearch:
        ext_name = ext_name

    fname = save_dir + '/net_before' + ext_name
    if not os.path.exists(fname + extension):
        start = time.time()
        net_before = create_network(NETFILE, TRIPFILE)
        net_before.not_fixed = set([])
        net_before.art_links = artLinkDict

        before_eq_tstt = solve_UE(net=net_before, eval_seq=True, wu=False, flows=True)
        global memory
        memory[frozenset(net_before.not_fixed)] = before_eq_tstt
        elapsed = time.time() - start

        save(fname, net_before)
        save(fname + '_tstt', before_eq_tstt)
        # shutil.copy('batch0.bin', 'before/batch0.bin')
        return net_before, before_eq_tstt, elapsed
    else:
        net_before = load(fname)
        before_eq_tstt = load(fname + '_tstt')

    # shutil.copy('batch0.bin', 'before/batch0.bin')
    return net_before, before_eq_tstt


def safety(wb, bb):
    swapped_links = {}
    for a_link in wb:
        if bb[a_link] < wb[a_link]:
            worst = bb[a_link]
            bb[a_link] = wb[a_link]
            wb[a_link] = worst
            swapped_links[a_link] = bb[a_link] - wb[a_link]
    return wb, bb, swapped_links

def get_wb(damaged_links, save_dir, relax=False, bsearch=False, ext_name=''):
    """get worst and best benefits of repairing links, and initiate start and end nodes"""
    net_before, before_eq_tstt, time_net_before = state_before(damaged_links, save_dir, real=True, relax=relax, bsearch=bsearch, ext_name=ext_name)
    net_after, after_eq_tstt, time_net_after = state_after(damaged_links, save_dir, real=True, relax=relax,  bsearch=bsearch, ext_name=ext_name)

    save(save_dir + '/damaged_dict', damaged_dict)

    net_before.save_dir = save_dir
    net_after.save_dir = save_dir

    wb = worst_benefit(net_before, damaged_links, before_eq_tstt, relax=relax, bsearch=bsearch, ext_name=ext_name)
    bb, bb_time = best_benefit(net_after, damaged_links, after_eq_tstt, relax=relax,  bsearch=bsearch, ext_name=ext_name)
    wb, bb, swapped_links = safety(wb, bb)

    start_node = Node(tstt_after=after_eq_tstt)
    start_node.before_eq_tstt = before_eq_tstt
    start_node.after_eq_tstt = after_eq_tstt
    start_node.realized = 0
    start_node.realized_u = 0
    start_node.level = 0
    start_node.visited, start_node.not_visited = set(
        []), set(damaged_links)
    start_node.fixed, start_node.not_fixed = set([]), set(damaged_links)

    end_node = Node(tstt_before=before_eq_tstt, forward=False)
    end_node.before_eq_tstt = before_eq_tstt
    end_node.after_eq_tstt = after_eq_tstt
    end_node.level = 0

    end_node.visited, end_node.not_visited = set([]), set(damaged_links)

    end_node.fixed, end_node.not_fixed = set(
        damaged_links), set([])

    start_node.relax = relax
    end_node.relax = relax

    return wb, bb, start_node, end_node, net_after, net_before, after_eq_tstt, before_eq_tstt, time_net_before, time_net_after, bb_time, swapped_links


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

def simple_preprocess(damaged_links, net_after):
    """uses sampling as described in Rey et al. 2019 to find approximated average first-order effects"""
    samples = []
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

    ns = 1
    card_P = len(damaged_links)
    denom = 2 ** card_P

    # a value of 1 means the link has already been repaired in that state
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

                    if any((new_pattern is test) or (new_pattern == test).all() for test in X_train):
                        pass
                    else:
                        new_TSTT, tap = eval_state(new_state, net_after, damaged_links, eval_seq=True)
                        preprocessing_num_tap += tap
                        X_train.append(new_pattern)
                        y_train.append(new_TSTT)
                        Z_train[el].append(abs(new_TSTT - TSTT))

    Z_bar = np.zeros(len(damaged_links))
    for i in range(len(damaged_links)):
        Z_bar[i] = np.mean(Z_train[i])
    print(Z_bar)

    return Z_bar, preprocessing_num_tap

def preprocessing(damaged_links, net_after):
    """trains a TensorFlow model to predict tstt from the binary repair state"""
    samples = []
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

    ns = 1
    card_P = len(damaged_links)
    denom = 2 ** card_P

    # a value of 1 means the link has already been repaired in that state
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

                    if any((new_pattern is test) or (new_pattern == test).all() for test in X_train):
                        pass
                    else:
                        new_TSTT, tap = eval_state(new_state, net_after, damaged_links, eval_seq=True)
                        preprocessing_num_tap += tap
                        X_train.append(new_pattern)
                        y_train.append(new_TSTT)
                        Z_train[el].append(abs(new_TSTT - TSTT))

    Z_bar = np.zeros(len(damaged_links))
    for i in range(len(damaged_links)):
        Z_bar[i] = np.mean(Z_train[i])

    # from sklearn.preprocessing import PolynomialFeatures
    # from sklearn.linear_model import LinearRegression
    # from sklearn.linear_model import SGDRegressor
    # from sklearn.metrics import mean_squared_error
    # from sklearn.ensemble import RandomForestRegressor
    # from sklearn.neural_network import MLPRegressor
    from tensorflow import keras
    # poly_features = PolynomialFeatures(degree=2, include_bias=False)
    # X_train = poly_features.fit_transform(X_train)
    # reg = RandomForestRegressor(n_estimators=50)
    # reg = MLPRegressor((100,75,50),early_stopping=True, verbose=True)

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

    ### Tests ###
    state = random.sample(damaged_links, 1)
    TSTT, tap = eval_state(state, net_after, damaged_links)
    state = [damaged_links.index(i) for i in state]
    pattern = np.zeros(len(damaged_links))
    pattern[(state)] = 1
    # pattern = poly_features.transform(pattern.reshape(1,-1))
    predicted_TSTT = model.predict(pattern.reshape(1, -1)) * stdy + meany
    print('predicted tstt vs real tstt, percent error:', predicted_TSTT[0][0], TSTT, (predicted_TSTT[0][0]-TSTT)/TSTT*100)

    state = random.sample(damaged_links, 4)
    TSTT, tap = eval_state(state, net_after, damaged_links)
    state = [damaged_links.index(i) for i in state]
    pattern = np.zeros(len(damaged_links))
    pattern[(state)] = 1
    predicted_TSTT = model.predict(pattern.reshape(1, -1)) * stdy + meany
    print('predicted tstt vs real tstt, percent error:', predicted_TSTT[0][0], TSTT, (predicted_TSTT[0][0]-TSTT)/TSTT*100)

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
                    print('Starting simulated annealing run number:', run+1)
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

                    #get tstt before fixing el or swap, then tstt if fixing each first to find difference in total area
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
                        #print('Iter ' +str(t) + ' positive movement')
                        #print('MOVED TO NEW STATE ' + str(current) + ' on iteration ' + str(t))

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
                        print('On iteration {}, new best solution cost {} with sequence {}'.format(t,best_cost,best_soln))

                    ratio = fail/t
                    #if t % 128 == 0:
                    #    print('Iteration ' + str(t) + ', ratio ' + str(ratio))
                    if run > 0 and t >= 1.2 * len(current)**2.5:
                        print('Finished run {} on iteration {} with ratio {} with best solution found on iteration {}'.format(run+1,t,ratio,lastMvmt))
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
                    print('Finished on iteration {} with ratio {} with best solution found on iteration {}'.format(t,ratio,lastMvmt))
                    if best_cost < final_cost:
                        final_soln = deepcopy(best_soln)
                        final_cost = deepcopy(best_cost)

        else:
            while t < 1.2 * len(current)**3:
                t += 1
                idx = random.randrange(0,len(current)-1)
                nextord = deepcopy(current)
                el = nextord.pop(idx)
                nextord.insert(idx+1,el)

                nextcost, tap, _ = eval_working_sequence(
                    curnet, nextord, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews)
                tap_solved += tap

                negdelta = curcost - nextcost

                if negdelta > 0:
                    current = deepcopy(nextord)
                    curcost = deepcopy(nextcost)
                    #print('Iter ' +str(t) + ' positive movement')
                    #print('MOVED TO NEW STATE ' + str(current) + ' on iteration ' + str(t))

                else:
                    prob = math.exp(negdelta/curcost*(t**(2/3)))
                    if random.random() <= prob:
                        current = deepcopy(nextord)
                        curcost = deepcopy(nextcost)
                    else:
                        fail += 1

                if curcost < best_cost:
                    test1, _, _ = eval_sequence(curnet, current, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews)
                    if best_cost < test1:
                        print('inaccuracy in new best soln ' + str(test1-curcost) + ' test1= ' + str(test1) + ' curcost= ' + str(curcost) + '. Do not use.')
                    else:
                        best_soln = deepcopy(current)
                        best_cost = deepcopy(curcost)
                        if graphing:
                            sa_time_list.append(time.time()-start)
                            sa_OBJ_list.append(deepcopy(best_cost))
                        lastMvmt = t
                        print('New best solution cost ' + str(best_cost) + ' with sequence ' + str(best_soln))

                ratio = fail/t
                if t % 128 == 0:
                    print('Iteration ' + str(t) + ', ratio ' + str(ratio))
            print('Finished on iteration {} with ratio {} with best solution found on iteration {}'.format(t,ratio,lastMvmt))

        elapsed = time.time() - start
        path = best_soln
        bound = best_cost
        if graphing:
            sa_time_list.append(elapsed)
            sa_OBJ_list.append(deepcopy(best_cost))

        test2, _, _ = eval_sequence(curnet, best_soln, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews)

        if abs(test2-bound)> 5:
            print('inaccuracy in best soln ' + str(test2-bound) + ' test2= ' + str(test2) + ' bound= ' + str(bound))
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
        _, path = zip(*sorted_c)

        elapsed = time.time() - start + time_before
        bound, eval_taps, _ = eval_sequence(
            LAFO_net, path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews)

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
        _, path = zip(*sorted_c)

        elapsed = time.time() - start + time_before
        bound, eval_taps, _ = eval_sequence(
            LASR_net, path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews)

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
        path, _ = zip(*sorted_d)

        elapsed = time.time() - start + time_net_before
        bound, eval_taps, _ = eval_sequence(
            SPT_net, path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews)

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
        bound, eval_taps, _ = eval_sequence(
            if_net, path, after_eq_tstt, before_eq_tstt, if_dict, importance=True, damaged_dict=damaged_dict, num_crews=num_crews)
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


def brute_force(net_after, after_eq_tstt, before_eq_tstt, is_approx=False, num_crews=1):
    """enumerates all possible sequences to find the lowest overall tstt impacts"""
    start = time.time()
    tap_solved = 0
    damaged_links = damaged_dict.keys()
    approx_ext = ''
    verbose = False
    if is_approx:
        approx_ext = '_approx'
        # verbose = True
        global approx_params

    fname = net_after.save_dir + '/min_seq' + approx_ext

    if not os.path.exists(fname + extension):

        print('Finding the optimal sequence...')
        all_sequences = itertools.permutations(damaged_links)

        # seq_dict = {}
        i = 0
        min_cost = 1e+80
        min_seq = None

        for sequence in all_sequences:

            seq_net = deepcopy(net_after)
            cost, eval_taps, _ = eval_working_sequence(
                seq_net, sequence, after_eq_tstt, before_eq_tstt, is_approx=is_approx, damaged_dict=damaged_dict, num_crews=num_crews, approx_params=approx_params)
            tap_solved += eval_taps
            # seq_dict[sequence] = cost

            if cost < min_cost or min_cost == 1e+80:
                min_cost = cost
                min_seq = sequence

            i += 1
            if verbose:
                print(i)
                print('min cost found so far: ', min_cost)
                print('min seq found so far: ', min_seq)

        # def compute(*args):
        #     seq_net = args[1]
        #     sequence = args[0]
        #     cost, eval_taps, _ = eval_sequence(
        #         seq_net, sequence, after_eq_tstt, before_eq_tstt)
        #     # print('sequence: {}, cost: {}'.format(sequence, cost))
        #     return cost, sequence, eval_taps

        # def all_seq_generator():
        #     for seq in itertools.permutations(damaged_links):
        #         yield seq, deepcopy(net_after)

        # with mp.Pool(processes=4) as p:
        #     data = p.map(compute, all_seq_generator())

        # min_cost, min_seq, tap_solved = min(data)
        # pdb.set_trace()

        elapsed = time.time() - start
        bound, _, _ = eval_sequence(net, min_seq, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews)
        save(fname + '_obj', bound)
        save(fname + '_path', min_seq)
        save(fname + '_elapsed', elapsed)
        save(fname + '_num_tap', tap_solved)
    else:
        min_cost = load(fname + '_obj')
        min_seq = load(fname + '_path')
        tap_solved = load(fname + '_num_tap')
        elapsed = load(fname + '_elapsed')
    if not is_approx:
        print('brute_force solution: ', min_seq)
    return min_cost, min_seq, elapsed, tap_solved


def orderlists(benefits, days, slack=0, rem_keys=None, reverse=True):
    """helper function for sorting/reversing lists"""
    # if sum(benefits) > slack:
    #     benefits = np.array(benefits)
    #     benefits = [min(slack, x) for x in benefits]

    bang4buck = np.array(benefits) / np.array(days)

    days = [x for _, x in sorted(
        zip(bang4buck, days), reverse=reverse)]
    benefits = [x for _, x in sorted(
        zip(bang4buck, benefits), reverse=reverse)]

    if rem_keys is not None:
        rem_keys = [x for _, x in sorted(
            zip(bang4buck, rem_keys), reverse=reverse)]
        return benefits, days, rem_keys

    return benefits, days

def lazy_greedy_heuristic():
    """heuristic which orders links for repair based on effect on tstt if repaired first"""
    start = time.time()
    net_b = create_network(NETFILE, TRIPFILE)
    net_b.not_fixed = set([])
    net_b.art_links = artLinkDict
    b_eq_tstt = solve_UE(net=net_b, eval_seq=True, flows=True)

    damaged_links = damaged_dict.keys()
    net_a = create_network(NETFILE, TRIPFILE)
    net_a.not_fixed = set(damaged_links)
    net_a.art_links = artLinkDict
    a_eq_tstt = solve_UE(net=net_a, eval_seq=True)

    lzg_bb = {}
    links_to_remove = deepcopy(list(damaged_links))
    to_visit = links_to_remove
    added = []
    for link in links_to_remove:
        test_net = deepcopy(net_a)
        added = [link]
        not_fixed = set(to_visit).difference(set(added))
        test_net.not_fixed = set(not_fixed)
        tstt_after = solve_UE(net=test_net)
        lzg_bb[link] = a_eq_tstt - tstt_after

    ordered_days = []
    orderedb_benefits = []

    sorted_d = sorted(damaged_dict.items(), key=lambda x: x[1])
    for key, value in sorted_d:
        ordered_days.append(value)
        orderedb_benefits.append(lzg_bb[key])

    ob, od, lzg_order = orderlists(orderedb_benefits, ordered_days, rem_keys=sorted_d)
    lzg_order = [i[0] for i in lzg_order]
    elapsed = time.time() - start

    bound, eval_taps, _ = eval_sequence(net_b, lzg_order, a_eq_tstt, b_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews)
    tap_solved = len(damaged_dict)

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
        print('Finding the greedy_solution ...')
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

            llll, lll, ord = orderlists(orderedb_benefits, ordered_days, rem_keys=sorted_d)

            link_to_add = ord[0][0]
            path.append(link_to_add)
            min_index = eligible_to_add.index(link_to_add)
            after_ = new_tstts[min_index]
            eligible_to_add.remove(link_to_add)
            decoy_dd = deepcopy(decoy_dd)
            del decoy_dd[link_to_add]

        net = deepcopy(net_after)

        elapsed = time.time() - start + time_net_before + time_net_after

        bound, _, _ = eval_sequence(net, path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews)

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


def greedy_heuristic_mult(net_after, after_eq_tstt, before_eq_tstt, time_net_before, time_net_after, num_crews):
    """multicrew heuristic which orders links for repair at each step based on
    immediate effect on tstt if that link completed before any additional links"""
    start = time.time()
    tap_solved = 0
    fname = net_after.save_dir + '/greedy_solution'
    global memory

    if not os.path.exists(fname + extension):
        print('Finding the greedy_solution ...')
        tap_solved = 0

        damaged_links = [link for link in damaged_dict.keys()]

        eligible_to_add = deepcopy(damaged_links)

        test_net = deepcopy(net_after)
        after_ = after_eq_tstt

        decoy_dd = deepcopy(damaged_dict)
        path = []
        crew_order_list = []
        crews = [0]*num_crews # crews is the current finish time for that crew
        which_crew = dict()
        completion = dict() # completion time of each link
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

            llll, lll, order = orderlists(orderedb_benefits, ordered_days, rem_keys=sorted_d)

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

        elapsed = time.time() - start + time_net_before + time_net_after

        bound, _, _ = eval_sequence(net, path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews)

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


def makeArtLinks():
    """creates artificial links with travel time and length 10x before eq shortest path"""
    start = time.time()
    artNet = Network(NETFILE, TRIPFILE)
    artNet.netfile = NETFILE
    artNet.tripfile = TRIPFILE
    artNet.not_fixed = set([])
    artNet.art_links = {}

    tstt = solve_UE(net=artNet, eval_seq=True, flows=True)

    for ij in artNet.link:
        artNet.link[ij].flow = artNet.linkDict[ij]['flow']
        artNet.link[ij].cost = artNet.linkDict[ij]['cost']

    art_links = {}
    originList = []
    backlink = {}
    cost = {}
    for od in artNet.ODpair.values():
        if od.origin not in originList:
            originList.append(od.origin)
            backlink[od.origin], cost[od.origin] = artNet.shortestPath(od.origin)
            if od.origin != od.destination:
                ODID = str(od.origin) + '->' + str(od.destination)
                art_links[ODID] = 10*cost[od.origin][od.destination]
        else:
            if od.origin != od.destination:
                ODID = str(od.origin) + '->' + str(od.destination)
                art_links[ODID] = 10*cost[od.origin][od.destination]

    for ij in damaged_links:
        artNet.link[ij].capacity = SMALL
        artNet.link[ij].freeFlowTime = SEQ_INFINITY
        artNet.link[ij].cost = SEQ_INFINITY

    artNet.not_fixed = set(damaged_links)
    artNet.art_links = {}
    tstt = solve_UE(net=artNet, eval_seq=True, flows=True)

    for ij in artNet.link:
        artNet.link[ij].flow = artNet.linkDict[ij]['flow']
        artNet.link[ij].cost = artNet.linkDict[ij]['cost']

    originList = []
    postbacklink = {}
    postcost = {}
    count = 0
    for od in artNet.ODpair.values():
        if od.origin not in originList:
            originList.append(od.origin)
            postbacklink[od.origin], postcost[od.origin] = artNet.shortestPath(od.origin)
            if postcost[od.origin][od.destination] >= 99999:
                count += 1
            if od.origin != od.destination and postcost[od.origin][od.destination] < 10*cost[od.origin][od.destination]:
                ODID = str(od.origin) + '->' + str(od.destination)
                del art_links[ODID]
        else:
            if postcost[od.origin][od.destination] >= 99999:
                count += 1
            if od.origin != od.destination and postcost[od.origin][od.destination] < 10*cost[od.origin][od.destination]:
                ODID = str(od.origin) + '->' + str(od.destination)
                del art_links[ODID]
    print('Created {} artificial links.'.format(len(art_links)))
    if count > len(art_links):
        print('There are {} more paths exceeding cost of 99999 than artificial links created'.format(count-len(art_links)))

    artNet.art_links = art_links
    tstt = solve_UE(net=artNet, eval_seq=True, flows=True)

    elapsed = time.time()-start
    print('Runtime to create artificial links: ',elapsed)
    return art_links


def plotNodesLinks(save_dir, net, damaged_links, coord_dict, names = False):
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

    scale = 10**(math.floor(min(math.log(xMax-xMin,10),math.log(yMax-yMin,10)))-1)

    fig, ax = plt.subplots(figsize = (12,9))
    plt.xlim(math.floor(xMin/scale)*scale,math.ceil(xMax/scale)*scale)
    plt.ylim(math.floor(yMin/scale)*scale,math.ceil(yMax/scale+0.2)*scale)
    plt.rcParams["font.size"] = 6

    # plot nodes
    nodesx = list()
    nodesy = list()
    for i in range(1,len(coord_dict)+1):
        nodesx.append(coord_dict[i][0])
        nodesy.append(coord_dict[i][1])
    if names == True:
        if NETWORK.find('ChicagoSketch') >= 0:
            for i in range(388,len(nodesx)):
                plt.annotate(i+1,(nodesx[i],nodesy[i]))
        else:
            for i in range(len(nodesx)):
                plt.annotate(i+1,(nodesx[i],nodesy[i]))
    plt.scatter(nodesx,nodesy, s=4)

    # plot links
    segments = list()
    damaged_segments = list()
    #for ij in [ij for ij in Network.link if Network.link[ij].flow > 0]:
    for ij in [ij for ij in net.link if ij not in damaged_links]:
        line = [coord_dict[net.link[ij].tail], coord_dict[net.link[ij].head]]
        segments.append(line)
    for ij in damaged_links:
        line = [coord_dict[net.link[ij].tail], coord_dict[net.link[ij].head]]
        damaged_segments.append(line)
    lc = mc.LineCollection(segments)
    lc_damaged = mc.LineCollection(damaged_segments, color = 'tab:red')
    ax.add_collection(lc)
    ax.add_collection(lc_damaged)
    ax.set_axis_off()
    plt.title('Map of ' + NETWORK.split('/')[-1] + ' with ' + str(len(damaged_links)) + ' damaged links', fontsize=12)

    save_fig(save_dir, 'map', tight_layout=True)


def plotTimeOBJ(save_dir, bs_time_list=None, bs_OBJ_list=None, sa_time_list=None, sa_OBJ_list=None):
    """function to plot running time vs OBJ progression"""

    fig, ax = plt.subplots(figsize = (8,6))
    jet = cm.get_cmap('jet', reps)
    if bs_time_list != None:
        bs_indices = [i for i, time in enumerate(bs_time_list) if time == 0]
        bs_indices.append(len(bs_time_list))
        if num_broken < 16:
            for rep in range(reps):
                plt.step(bs_time_list[bs_indices[rep]:bs_indices[rep+1]],
                   (1-np.divide(bs_OBJ_list[bs_indices[rep]:bs_indices[rep+1]],bs_OBJ_list[bs_indices[rep]]))*100,
                    where='post', color = jet(rep/reps), label = 'beam search '+str(rep+1))
        else:
            for rep in range(reps):
                plt.step(np.divide(bs_time_list[bs_indices[rep]:bs_indices[rep+1]],60),
                   (1-np.divide(bs_OBJ_list[bs_indices[rep]:bs_indices[rep+1]],bs_OBJ_list[bs_indices[rep]]))*100,
                    where='post', color = jet(rep/reps), label = 'beam search '+str(rep+1))
    if sa_time_list != None:
        sa_indices = [i for i, time in enumerate(sa_time_list) if time == 0]
        sa_indices.append(len(sa_time_list))
        if num_broken < 16:
            for rep in range(reps):
                plt.step(sa_time_list[sa_indices[rep]:sa_indices[rep+1]],
                    (1-np.divide(sa_OBJ_list[sa_indices[rep]:sa_indices[rep+1]],sa_OBJ_list[sa_indices[rep]]))*100,
                    where='post', color = jet(rep/reps), linestyle='dashed', label = 'sim anneal '+str(rep+1))
        else:
            for rep in range(reps):
                plt.step(np.divide(sa_time_list[sa_indices[rep]:sa_indices[rep+1]],60),
                    (1-np.divide(sa_OBJ_list[sa_indices[rep]:sa_indices[rep+1]],sa_OBJ_list[sa_indices[rep]]))*100,
                    where='post', color = jet(rep/reps), linestyle='dashed', label = 'sim anneal '+str(rep+1))
    if num_broken < 16:
        plt.title('Runtime (seconds) vs OBJ function improvement (%) for ' + NETWORK.split('/')[-1] +
            ' with ' + str(len(damaged_links)) + ' damaged links', fontsize=10)
    else:
        plt.title('Runtime (minutes) vs OBJ function improvement (%) for ' + NETWORK.split('/')[-1] +
            ' with ' + str(len(damaged_links)) + ' damaged links', fontsize=10)
    plt.legend(ncol=2)

    save_fig(save_dir, 'timevsOBJ', tight_layout=True)


if __name__ == '__main__':

    net_name = args.net_name
    num_broken = args.num_broken
    approx = args.approx
    reps = args.reps
    beam_search = args.beamsearch
    beta = args.beta
    gamma = args.gamma
    if type(args.num_crews) == int:
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
    strength = args.strength
    full = args.full
    rand_gen = args.random
    location_based = args.loc
    opt = args.opt
    sa = args.sa
    damaged_dict_preset = args.damaged
    if damaged_dict_preset != '':
        print(damaged_dict_preset)
    multiClass = args.mc

    NETWORK = os.path.join(FOLDER, net_name)
    if net_name == 'Chicago-Sketch':
        net_name = 'ChicagoSketch'
    if net_name == 'Berlin-Mitte-Center':
        net_name = 'berlin-mitte-center'
    if net_name == 'Berlin-Mitte-Center2':
        net_name = 'berlin-mitte-center2'
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
    if multiClass == True and damaged_dict_preset != '':
        TRIPFILE = list()
        i=1
        while True:
            if os.path.exists(os.path.join(NETWORK, net_name + "_trips" + str(i) + "a.tntp")):
                TRIPFILE.append(os.path.join(NETWORK, net_name + "_trips" + str(i) + "a.tntp"))
                i+=1
            else:
                if TRIPFILE == []:
                    print('Improper tripfile naming.')
                    raise utils.BadFileFormatException
                print(TRIPFILE)
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
                #print(num_broken)
                memory = {}

                net = create_network(NETFILE, TRIPFILE)
                net.not_fixed = set([])
                net.art_links = {}
                solve_UE(net=net, wu=False, eval_seq=True)

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
                # all_links = sorted(all_links)
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
                if location_based:
                    import geopy.distance

                    if NETWORK.find('SiouxFalls') >= 0:
                        nodetntp = pd.read_csv(os.path.join(NETWORK, net_name + "_node.tntp"), delimiter='\t')
                        coord_dict = {}

                        for index, row in nodetntp.iterrows():
                            lon = row['X']
                            lat = row['Y']
                            coord_dict[row['Node']] = (lon, lat)

                    if NETWORK.find('Chicago-Sketch') >=0:
                        nodetntp = pd.read_csv(os.path.join(NETWORK, net_name + "_node.tntp"), delimiter='\t')
                        coord_dict = {}

                        for index, row in nodetntp.iterrows():
                            lon = row['X']
                            lat = row['Y']
                            coord_dict[row['node']] = (lon, lat)

                    if NETWORK.find('Anaheim') >= 0:
                        nodetntp = pd.read_json(os.path.join(NETWORK, 'anaheim_nodes.geojson'))
                        coord_dict = {}

                        for index, row in nodetntp.iterrows():
                            coord_dict[row['features']['properties']['id']] = row['features']['geometry']['coordinates']

                    if NETWORK.find('Berlin-Mitte-Center') >= 0:
                        nodetntp = pd.read_csv(os.path.join(NETWORK, net_name + "_node.tntp"), delim_whitespace = True, skipinitialspace=True)
                        coord_dict = {}

                        for index, row in nodetntp.iterrows():
                            lon = row['X']
                            lat = row['Y']
                            coord_dict[row['Node']] = (lon, lat)

                    #pick a center node at random:
                    nodedecoy = deepcopy(list(netg.node.keys()))
                    nodedecoy = sorted(nodedecoy)
                    if NETWORK.find('Chicago-Sketch') >= 0:
                        nodedecoy = nodedecoy[387:]
                    center_node = np.random.choice(nodedecoy, 1, replace=False)[0]

                    nodedecoy.remove(center_node)
                    #find distances from center node:
                    dist_dict = {}

                    for anode in nodedecoy:
                        distance = np.linalg.norm(np.array(coord_dict[center_node]) - np.array(coord_dict[anode]))
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
                        another_node = random.choices(decoy, 1.0/np.array(distances)**3, k=1)[0]

                        # weights = 1.0/np.array(distances)
                        # weights = weights/weights[0]
                        # weights = weights**3.5
                        # weights = list(weights)
                        # another_node = random.choices(decoy, weights, k=1)[0]
                        # another_node = random.choices(decoy, np.exp(np.cbrt(1.0/np.array(distances))), k=1)[0]

                        idx = decoy.index(another_node)
                        del distances[idx]
                        del decoy[idx]
                        # del weights[idx]
                        selected_nodes.append(another_node)
                        i += 1

                    selected_indices = [all_nodes.index(ind) for ind in selected_nodes[1:]]
                    print(selected_nodes)


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
                        fol = np.array(flow_on_links)
                        fol = fol/np.max(fol)

                        # ij = random.choice(linkset)
                        if not fail:
                            ij = random.choices(linkset, weights=np.exp(np.power(flow_on_links, 1 / 3.0)), k=1)[0]
                            # ij = random.choices(linkset, weights=fol, k=1)[0]

                        else:
                            ij = random.choices(linkset, weights=np.exp(np.power(flow_on_links, 1 / 4.0)), k=1)[0]
                            # ij = random.choices(linkset, weights=fol, k=1)[0]


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

                        if iterno % 100 == 0:
                            print(iterno, damaged_links)
                        if iterno % 2000==0 and iterno!=0:
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
                        # ij = random.choices(decoy, weights=np.exp(np.cbrt(weights)), k=1)[0]
                        # ij = random.choices(decoy, weights=np.exp(np.power(weights, 1/3.0)), k=1)[0]
                        ij = random.choice(decoy)

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

                print('damaged_links are created:', damaged_links)
                damaged_dict = {}
                net.linkDict = {}
                # pdb.set_trace()

                with open(NETFILE, "r") as networkFile:

                    fileLines = networkFile.read().splitlines()
                    metadata = utils.readMetadata(fileLines)
                    for line in fileLines[metadata['END OF METADATA']:]:
                        # Ignore comments and blank lines
                        line = line.strip()
                        commentPos = line.find("~")
                        if commentPos >= 0:  # strip comments
                            line = line[:commentPos]

                        if len(line) == 0:
                            continue

                        data = line.split()
                        if len(data) < 11 or data[10] != ';':
                            print("Link data line not formatted properly:\n '%s'" % line)
                            raise utils.BadFileFormatException

                        # Create link
                        linkID = '(' + str(data[0]).strip() + \
                                 "," + str(data[1]).strip() + ')'

                        net.linkDict[linkID] = {}
                        net.linkDict[linkID]['length-cap'] = float(data[2]) * np.cbrt(float(data[3]))

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

                print(f'experiment number: {rep+1}')

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
                # num_broken = len(damaged_dict)

                opt = True
                if num_broken > 8:
                    opt = False

                memory = {}
                net = create_network(NETFILE, TRIPFILE)

                all_links = damaged_dict_.keys()
                damaged_links = np.random.choice(list(all_links), num_broken, replace=False)
                # repair_days = [net.link[a_link].length for a_link in damaged_links]
                removals = []

                # net.not_fixed = set([])
                # solve_UE(net=net)
                # file_created = False
                # f = 'flows.txt'
                # st = time.time()
                # while not file_created:
                #     if os.path.exists(f):
                #         file_created = True

                #     if time.time()-st >10:
                #         popen = subprocess.call(args, stdout=subprocess.DEVNULL)

                #     if file_created:
                #         with open(f, "r") as flow_file:
                #             for line in flow_file.readlines():
                #                 try:
                #                     ij = str(line[:line.find(' ')])
                #                     line = line[line.find(' '):].strip()
                #                     flow = float(line[:line.find(' ')])
                #                     line = line[line.find(' '):].strip()
                #                     cost = float(line.strip())
                #                     net.link[ij].flow = flow
                #                     net.link[ij].cost = cost
                #                 except:
                #                     break

                #         os.remove('flows.txt')

                # for ij in damaged_links:
                #     if net.link[ij].flow == 0:
                #         removals.append(ij)

                # min_rd = min(repair_days)
                # max_rd = max(repair_days)
                # med_rd = np.median(repair_days)
                damaged_dict = {}

                for a_link in damaged_links:
                    if a_link in removals:
                        continue
                    damaged_dict[a_link] = damaged_dict_[a_link]

                    # repair_days = net.link[a_link].length
                    # if repair_days > med_rd:
                    #     repair_days += (max_rd - repair_days) * 0.3
                    # y = ((repair_days - min_rd) / (min_rd - max_rd)
                    #      * (MIN_DAYS - MAX_DAYS) + MIN_DAYS)
                    # mu = y
                    # std = y * 0.3
                    # damaged_dict[a_link] = np.random.normal(mu, std, 1)[0]

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


            ### finalize damaged links ###
            damaged_links = damaged_dict.keys()
            num_damaged = len(damaged_links)

            print(damaged_dict)
            save_dir = ULT_SCENARIO_REP_DIR


            if before_after:
                net_after, after_eq_tstt, _ = state_after(damaged_links, save_dir)
                net_before, before_eq_tstt, _ = state_before(damaged_links, save_dir)
                plotNodesLinks(save_dir, netg, damaged_links, coord_dict, names=True)
                t = PrettyTable()
                t.title = net_name + ' with ' + str(num_broken) + ' broken bridges'
                t.field_names = ['Before EQ TSTT', 'After EQ TSTT']
                t.add_row([before_eq_tstt, after_eq_tstt])
                print(t)
            else:
                memory = {}
                artLinkDict = makeArtLinks()
                benefit_analysis_st = time.time()
                wb, bb, start_node, end_node, net_after, net_before, after_eq_tstt, before_eq_tstt, time_net_before, time_net_after, bb_time, swapped_links = get_wb(damaged_links, save_dir)
                benefit_analysis_elapsed = time.time() - benefit_analysis_st
                plotNodesLinks(save_dir, netg, damaged_links, coord_dict, names=True)

                ### approx solution methods ###
                if approx:
                    memory1 = deepcopy(memory)

                    preprocess_st = time.time()
                    #model, meany, stdy, Z_bar, preprocessing_num_tap = preprocessing(damaged_links, net_after)
                    #approx_params = (model, meany, stdy)
                    Z_bar, preprocessing_num_tap = simple_preprocess(damaged_links, net_after)
                    preprocess_elapsed = time.time() - preprocess_st
                    time_before = preprocess_elapsed + time_net_before

                    ### Largest Averge First Order ###
                    LAFO_obj, LAFO_soln, LAFO_elapsed, LAFO_num_tap = LAFO(
                        net_before, after_eq_tstt, before_eq_tstt, time_before, Z_bar)
                    LAFO_num_tap += preprocessing_num_tap
                    if alt_crews == None and not multiClass:
                        print('LAFO_obj: ', LAFO_obj)
                    elif multiClass and type(net_after.tripfile) == list:
                        test_net = deepcopy(net_after)
                        LAFO_obj_mc, _, _ = eval_sequence(
                            test_net, LAFO_soln, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews, multiClass=multiClass)
                        LAFO_obj_mc.insert(0, LAFO_obj)
                        print('LAFO_obj: ', LAFO_obj_mc)
                    else:
                        LAFO_obj_mult = [0]*(len(alt_crews)+1)
                        LAFO_obj_mult[0] = LAFO_obj
                        for num in range(len(alt_crews)):
                            test_net = deepcopy(net_before)
                            LAFO_obj_mult[num+1], _, _ = eval_sequence(
                                test_net, LAFO_soln, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=alt_crews[num])
                        print('LAFO_obj: ', LAFO_obj_mult)


                    ### Largest Averge Smith Ratio ###
                    LASR_obj, LASR_soln, LASR_elapsed, LASR_num_tap = LASR(
                        net_before, after_eq_tstt, before_eq_tstt, time_before, Z_bar)
                    LASR_num_tap += preprocessing_num_tap
                    if alt_crews == None and not multiClass:
                        print('LASR_obj: ', LASR_obj)
                    elif multiClass and type(net_after.tripfile) == list:
                        test_net = deepcopy(net_after)
                        LASR_obj_mc, _, _ = eval_sequence(
                            test_net, LASR_soln, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews, multiClass=multiClass)
                        LASR_obj_mc.insert(0, LASR_obj)
                        print('LASR_obj: ', LASR_obj_mc)
                    else:
                        LASR_obj_mult = [0]*(len(alt_crews)+1)
                        LASR_obj_mult[0] = LAFO_obj
                        for num in range(len(alt_crews)):
                            test_net = deepcopy(net_before)
                            LASR_obj_mult[num+1], _, _ = eval_sequence(
                                test_net, LASR_soln, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=alt_crews[num])
                        print('LASR_obj: ', LASR_obj_mult)

                    # approx_obj, approx_soln, approx_elapsed, approx_num_tap = brute_force(net_after, after_eq_tstt, before_eq_tstt, is_approx=True)
                    # approx_num_tap += preprocessing_num_tap
                    # print('approx obj: {}, approx path: {}'.format(approx_obj, approx_soln))
                    memory = deepcopy(memory1)

                ### Shortest processing time solution ###
                SPT_obj, SPT_soln, SPT_elapsed, SPT_num_tap = SPT_solution(
                    net_before, after_eq_tstt, before_eq_tstt, time_net_before)
                if alt_crews == None and not multiClass:
                    print('incorrectly entered  alt_crews == None and not multiClass if for SPT')
                    print('SPT_obj: ', SPT_obj)
                elif multiClass and type(net_after.tripfile) == list:
                    print('correctly entered multiclass elif for SPT')
                    test_net = deepcopy(net_after)
                    SPT_obj_mc, _, _ = eval_sequence(
                        test_net, SPT_soln, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews, multiClass=multiClass)
                    SPT_obj_mc.insert(0, SPT_obj)
                    print('SPT_obj: ', SPT_obj_mc)
                else:
                    print('incorrectly entered multiCREW else for SPT')
                    SPT_obj_mult = [0]*(len(alt_crews)+1)
                    SPT_obj_mult[0] = SPT_obj
                    for num in range(len(alt_crews)):
                        test_net = deepcopy(net_before)
                        SPT_obj_mult[num+1], _, _ = eval_sequence(
                            test_net, SPT_soln, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=alt_crews[num])
                    print('SPT_obj: ', SPT_obj_mult)


                ### Lazy greedy solution ###
                lg_obj, lg_soln, lg_elapsed, lg_num_tap = lazy_greedy_heuristic()
                    #net_after, after_eq_tstt, before_eq_tstt, time_net_before, bb_time)
                if alt_crews == None and not multiClass:
                    print('lazy_greedy_obj: ', lg_obj)
                elif multiClass and type(net_after.tripfile) == list:
                    test_net = deepcopy(net_after)
                    lg_obj_mc, _, _ = eval_sequence(
                        test_net, lg_soln, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews, multiClass=multiClass)
                    lg_obj_mc.insert(0, lg_obj)
                    print('lazy_greedy_obj: ', lg_obj_mc)
                else:
                    lg_obj_mult = [0]*(len(alt_crews)+1)
                    lg_obj_mult[0] = lg_obj
                    for num in range(len(alt_crews)):
                        test_net = deepcopy(net_after)
                        lg_obj_mult[num+1], _, _ = eval_sequence(
                            test_net, lg_soln, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=alt_crews[num])
                    print('lazy_greedy_obj: ', lg_obj_mult)

                wb_update = deepcopy(wb)
                bb_update = deepcopy(bb)


                ### Get greedy solution ###
                if num_crews == 1:
                    greedy_obj, greedy_soln, greedy_elapsed, greedy_num_tap = greedy_heuristic(
                        net_after, after_eq_tstt, before_eq_tstt, time_net_before, time_net_after)
                else:
                    greedy_obj, greedy_soln, greedy_elapsed, greedy_num_tap = greedy_heuristic_mult(
                        net_after, after_eq_tstt, before_eq_tstt, time_net_before, time_net_after, num_crews)
                if alt_crews == None and not multiClass:
                    print('greedy_obj: ', greedy_obj)
                elif multiClass and type(net_after.tripfile) == list:
                    test_net = deepcopy(net_after)
                    greedy_obj_mc, _, _ = eval_sequence(
                        test_net, greedy_soln, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews, multiClass=multiClass)
                    greedy_obj_mc.insert(0, greedy_obj)
                    print('greedy_obj: ', greedy_obj_mc)
                else:
                    greedy_obj_mult = [0]*(len(alt_crews)+1)
                    greedy_obj_mult[0] = greedy_obj
                    for num in range(len(alt_crews)):
                        test_net = deepcopy(net_after)
                        greedy_obj_mult[num+1], _, _ = eval_sequence(
                            test_net, greedy_soln, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=alt_crews[num])
                    print('greedy_obj: ', greedy_obj_mult)

                bfs = BestSoln()
                bfs.cost = greedy_obj
                bfs.path = greedy_soln

                wb = deepcopy(wb_update)
                bb = deepcopy(bb_update)

                wb_orig = deepcopy(wb)
                bb_orig = deepcopy(bb)


                ### Get feasible solution using importance factor ###
                importance_obj, importance_soln, importance_elapsed, importance_num_tap = importance_factor_solution(
                    net_before, after_eq_tstt, before_eq_tstt, time_net_before)
                if alt_crews == None and not multiClass:
                    print('importance_obj: ', importance_obj)
                elif multiClass and type(net_after.tripfile) == list:
                    test_net = deepcopy(net_after)
                    importance_obj_mc, _, _ = eval_sequence(
                        test_net, importance_soln, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews, multiClass=multiClass)
                    importance_obj_mc.insert(0, importance_obj)
                    print('importance_obj: ', importance_obj_mc)
                else:
                    importance_obj_mult = [0]*(len(alt_crews)+1)
                    importance_obj_mult[0] = importance_obj
                    for num in range(len(alt_crews)):
                        test_net = deepcopy(net_after)
                        importance_obj_mult[num+1], _, _ = eval_sequence(
                            test_net, importance_soln, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=alt_crews[num])
                    print('importance_obj: ', importance_obj_mult)

                if importance_obj < bfs.cost:
                    bfs.cost = importance_obj
                    bfs.path = importance_soln

                memory1 = deepcopy(memory)


                ### Get optimal solution via brute force ###
                if opt:
                    opt_obj, opt_soln, opt_elapsed, opt_num_tap = brute_force(
                        net_after, after_eq_tstt, before_eq_tstt, num_crews=num_crews)
                    opt_elapsed += greedy_elapsed
                    opt_num_tap += greedy_num_tap + len(damaged_links) - 1
                    print('optimal obj with {} crew(s): {}, optimal path: {}'.format(num_crews, opt_obj, opt_soln))
                    if multiClass and type(net_after.tripfile) == list:
                        test_net = deepcopy(net_after)
                        opt_obj_mc, _, _ = eval_sequence(
                            test_net, opt_soln, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews, multiClass=multiClass)
                        opt_obj_mc.insert(0, opt_obj)
                        print('optimal obj and demand class breakdown with {} crew(s): {}'.format(num_crews, opt_obj_mc))
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
                            opt_obj_mult[num+1], opt_soln_mult[num+1], opt_elapsed_mult[num+1], opt_num_tap_mult[num+1] = brute_force(
                                net_after, after_eq_tstt, before_eq_tstt, num_crews=alt_crews[num])
                            opt_elapsed_mult[num+1] += greedy_elapsed
                            opt_num_tap_mult[num+1] += greedy_num_tap + len(damaged_links) - 1
                            print('optimal obj with {} crew(s): {}, optimal path: {}'.format(alt_crews[num], opt_obj_mult[num+1], opt_soln_mult[num+1]))
                            if multiClass and type(net_after.tripfile) == list:
                                test_net = deepcopy(net_after)
                                temp, _, _ = eval_sequence(
                                    test_net, opt_soln_mult[num+1], after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=alt_crews[num], multiClass=multiClass)
                                opt_obj_mc.append(temp)
                                opt_obj_mc[num+1].insert(0, opt_obj_mult[num+1])
                                print('optimal obj and demand class breakdown with {} crew(s): {}'.format(num_crews, opt_obj_mc[num+1]))

                best_benefit_taps = num_damaged
                worst_benefit_taps = num_damaged

                memory = deepcopy(memory1)


                ### Get simulated annealing solution ###
                if sa:
                    sa_obj, sa_soln, sa_elapsed, sa_num_tap = sim_anneal(
                        bfs, net_after, after_eq_tstt, before_eq_tstt, damaged_links, num_crews=num_crews)
                    sa_elapsed += greedy_elapsed + importance_elapsed
                    sa_num_tap += greedy_num_tap
                    if alt_crews == None and not multiClass:
                        print('simulated annealing obj: ', sa_obj)
                    elif multiClass and type(net_after.tripfile) == list:
                        test_net = deepcopy(net_after)
                        sa_obj_mc, _, _ = eval_sequence(
                            test_net, sa_soln, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews, multiClass=multiClass)
                        sa_obj_mc.insert(0, sa_obj)
                        print('simulated annealing obj: ', sa_obj_mc)
                    else:
                        sa_obj_mult = [0]*(len(alt_crews)+1)
                        sa_obj_mult[0] = sa_obj
                        for num in range(len(alt_crews)):
                            test_net = deepcopy(net_after)
                            sa_obj_mult[num+1], _, _ = eval_sequence(
                                test_net, sa_soln, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=alt_crews[num])
                        print('simulated annealing obj: ', sa_obj_mult)


                ### Use full search algorthm to find solution ###
                if full:
                    print('full')
                    algo_num_tap = best_benefit_taps + worst_benefit_taps

                    fname = save_dir + '/algo_solution'

                    if not os.path.exists(fname + extension):
                        search_start = time.time()
                        algo_path, algo_obj, search_tap_solved, tot_child, uncommon_number, common_number, _ = search(
                            start_node, end_node, bfs)
                        search_elapsed = time.time() - search_start + greedy_elapsed

                        net_after, after_eq_tstt = state_after(damaged_links, save_dir, real=True)
                        net_before, before_eq_tstt = state_before(damaged_links, save_dir, real=True)

                        first_net = deepcopy(net_after)
                        first_net.relax = False
                        algo_obj, _, _ = eval_sequence(
                            first_net, algo_path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews)

                        algo_num_tap += search_tap_solved + greedy_num_tap
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
                    if alt_crews == None and not multiClass:
                        print('full obj: ', algo_obj)
                    elif multiClass and type(net_after.tripfile) == list:
                        test_net = deepcopy(net_after)
                        algo_obj_mc, _, _ = eval_sequence(
                            test_net, algo_path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews, multiClass=multiClass)
                        algo_obj_mc.insert(0, algo_obj)
                        print('full obj: ', algo_obj_mc)
                    else:
                        algo_obj_mult = [0]*(len(alt_crews)+1)
                        algo_obj_mult[0] = algo_obj
                        for num in range(len(alt_crews)):
                            test_net = deepcopy(net_after)
                            algo_obj_mult[num+1], _, _ = eval_sequence(
                                test_net, algo_path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=alt_crews[num])
                        print('full obj: ', algo_obj_mult)


                ### Use beamsearch algorithm to find solution ###
                if beam_search:

                    runs = ['normal'] # sa_seed must be run first because of memory from sa

                    for run in runs:

                        # betas = [64, 128, 256, 512]
                        # gammas = [32, 64, 128]

                        k = 2
                        betas = [128]
                        gammas = [128]

                        beta_gamma = list(itertools.product(gammas, betas))

                        experiment_dict = {}
                        for a_pair in beta_gamma:
                            gamma = a_pair[0]
                            beta = a_pair[1]

                            # beamsearch with gap 1e-4
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


                            if greedy_obj <= importance_obj:
                                bfs.cost = greedy_obj
                                bfs.path = greedy_soln
                            else:
                                bfs.cost = importance_obj
                                bfs.path = importance_soln
                            if run == 'sa_seed':
                                if sa_obj < bfs.cost:
                                    bfs.cost = sa_obj
                                    bfs.path = sa_soln

                            if run == 'sa_seed':
                                sa_r_algo_num_tap = sa_num_tap
                            else:
                                r_algo_num_tap = worst_benefit_taps - 1 + greedy_num_tap

                            start_node.relax = True
                            end_node.relax = True

                            # if not os.path.exists(fname + extension):
                            search_start = time.time()
                            r_algo_path, r_algo_obj, r_search_tap_solved, tot_childr, uncommon_numberr, common_numberr, num_purger = search(
                                start_node, end_node, bfs, beam_search=beam_search, beam_k=k, beta=beta, gamma=gamma)
                            search_elapsed = time.time() - search_start

                            if graphing:
                                bs_time_list.append(search_elapsed)
                                bs_OBJ_list.append(r_algo_obj)

                            net_after, after_eq_tstt = state_after(damaged_links, save_dir, real=True)
                            net_before, before_eq_tstt = state_before(damaged_links, save_dir, real=True)

                            first_net = deepcopy(net_after)
                            first_net.relax = False
                            r_algo_obj, _, _ = eval_sequence(
                                first_net, r_algo_path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews)

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

                                print(f'beam_search k: {k}, relaxed: True, type of run {run}')
                                if alt_crews == None and not multiClass:
                                    print('sa seeded beam search obj: ', sa_r_algo_obj)
                                elif multiClass and type(net_after.tripfile) == list:
                                    test_net = deepcopy(net_after)
                                    sa_r_algo_obj_mc, _, _ = eval_sequence(
                                        test_net, sa_r_algo_path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews, multiClass=multiClass)
                                    sa_r_algo_obj_mc.insert(0, sa_r_algo_obj)
                                    print('sa seeded beam search obj: ', sa_r_algo_obj_mc)
                                else:
                                    sa_r_algo_obj_mult = [0]*(len(alt_crews)+1)
                                    sa_r_algo_obj_mult[0] = sa_r_algo_obj
                                    for num in range(len(alt_crews)):
                                        test_net = deepcopy(net_after)
                                        sa_r_algo_obj_mult[num+1], _, _ = eval_sequence(
                                            test_net, sa_r_algo_path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=alt_crews[num])
                                    print('sa seeded beam search obj: ', sa_r_algo_obj_mult)

                            else:
                                r_algo_num_tap += r_search_tap_solved
                                r_algo_elapsed = search_elapsed + greedy_elapsed + importance_elapsed
                                save(fname + '_obj', r_algo_obj)
                                save(fname + '_path', r_algo_path)
                                save(fname + '_num_tap', r_algo_num_tap)
                                save(fname + '_elapsed', r_algo_elapsed)
                                save(fname + '_totchild', tot_childr)
                                save(fname + '_uncommon', uncommon_numberr)
                                save(fname + '_common', common_numberr)
                                save(fname + '_num_purge', num_purger)

                            # else:
                            #     r_algo_obj = load(fname + '_obj')
                            #     r_algo_path = load(fname + '_path')
                            #     r_algo_num_tap = load(fname + '_num_tap')
                            #     r_algo_elapsed = load(fname + '_elapsed')

                                print(f'beam_search k: {k}, relaxed: True, type of run {run}')
                                if alt_crews == None and not multiClass:
                                    print('beam search obj: ', r_algo_obj)
                                elif multiClass and type(net_after.tripfile) == list:
                                    test_net = deepcopy(net_after)
                                    r_algo_obj_mc, _, _ = eval_sequence(
                                        test_net, r_algo_path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=num_crews, multiClass=multiClass)
                                    r_algo_obj_mc.insert(0, r_algo_obj)
                                    print('beam search obj: ', (r_algo_obj_mc))
                                else:
                                    r_algo_obj_mult = [0]*(len(alt_crews)+1)
                                    r_algo_obj_mult[0] = r_algo_obj
                                    for num in range(len(alt_crews)):
                                        test_net = deepcopy(net_after)
                                        r_algo_obj_mult[num+1], _, _ = eval_sequence(
                                            test_net, r_algo_path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict, num_crews=alt_crews[num])
                                    print('beam search obj: ', r_algo_obj_mult)

                                experiment_dict[a_pair] = [r_algo_obj, r_algo_num_tap, r_algo_elapsed]

                        for k,v in experiment_dict.items():
                            print('Gamma - Beta: {}, obj-tap-elapsed: {}'.format(k,v))


                if multiClass and type(net_after.tripfile) == list:
                    t = PrettyTable()
                    if damaged_dict_preset=='':
                        t.title = net_name + ' with ' + str(num_broken) + ' broken bridges (class priorities)'
                    else:
                        t.title = net_name + ' with ' + str(num_broken) + ' broken bridges (equal priority)'
                    t.field_names = ['Method', 'Objective', 'Run Time', '# TAP']
                    if opt:
                        t.add_row(['OPTIMAL', opt_obj_mc, opt_elapsed, opt_num_tap])
                    if approx:
                        t.add_row(['approx-LAFO', LAFO_obj_mc, LAFO_elapsed, LAFO_num_tap])
                        t.add_row(['approx-LASR', LASR_obj_mc, LASR_elapsed, LASR_num_tap])
                    # if len(damaged_links) < 32:
                    #     t.add_row(['BeamSearch', beamsearch_obj, beamsearch_elapsed, beamsearch_num_tap])
                    if full:
                        t.add_row(['FULL ALGO', algo_obj_mc, algo_elapsed, algo_num_tap])
                    if beam_search:
                        t.add_row(['BeamSearch_relaxed', r_algo_obj_mc, r_algo_elapsed, r_algo_num_tap])
                        try:
                            t.add_row(['BeamSearch_sa_seed', sa_r_algo_obj_mc, sa_r_algo_elapsed, sa_r_algo_num_tap])
                        except:
                            pass
                    if sa:
                        t.add_row(['Simulated Annealing', sa_obj_mc, sa_elapsed, sa_num_tap])
                    t.add_row(['GREEDY', greedy_obj_mc, greedy_elapsed, greedy_num_tap])
                    t.add_row(['LG', lg_obj_mc, lg_elapsed, lg_num_tap])
                    t.add_row(['IMPORTANCE', importance_obj_mc, importance_elapsed, importance_num_tap])
                    t.add_row(['SPT', SPT_obj_mc, SPT_elapsed, SPT_num_tap])
                    t.set_style(MSWORD_FRIENDLY)
                    print(t)

                elif alt_crews == None:
                    t = PrettyTable()
                    t.title = net_name + ' with ' + str(num_broken) + ' broken bridges'
                    t.field_names = ['Method', 'Objective', 'Run Time', '# TAP']
                    if opt:
                        t.add_row(['OPTIMAL', opt_obj, opt_elapsed, opt_num_tap])
                    if approx:
                        t.add_row(['approx-LAFO', LAFO_obj, LAFO_elapsed, LAFO_num_tap])
                        t.add_row(['approx-LASR', LASR_obj, LASR_elapsed, LASR_num_tap])
                    # if len(damaged_links) < 32:
                    #     t.add_row(['BeamSearch', beamsearch_obj, beamsearch_elapsed, beamsearch_num_tap])
                    if full:
                        t.add_row(['FULL ALGO', algo_obj, algo_elapsed, algo_num_tap])
                    if beam_search:
                        t.add_row(['BeamSearch_relaxed', r_algo_obj, r_algo_elapsed, r_algo_num_tap])
                        try:
                            t.add_row(['BeamSearch_sa_seed', sa_r_algo_obj, sa_r_algo_elapsed, sa_r_algo_num_tap])
                        except:
                            pass
                    if sa:
                        t.add_row(['Simulated Annealing', sa_obj, sa_elapsed, sa_num_tap])
                    t.add_row(['GREEDY', greedy_obj, greedy_elapsed, greedy_num_tap])
                    t.add_row(['LG', lg_obj, lg_elapsed, lg_num_tap])
                    t.add_row(['IMPORTANCE', importance_obj, importance_elapsed, importance_num_tap])
                    t.add_row(['SPT', SPT_obj, SPT_elapsed, SPT_num_tap])
                    t.set_style(MSWORD_FRIENDLY)
                    print(t)

                else:
                    t = PrettyTable()
                    t.title = net_name + ' with ' + str(num_broken) + ' broken bridges'
                    t.field_names = ['Method', 'Objectives', 'Run Time', '# TAP']
                    if opt:
                        for num in range(len(alt_crews)+1):
                            if num == 0:
                                t.add_row(['OPTIMAL '+str(num_crews), opt_obj_mult[num], opt_elapsed_mult[num], opt_num_tap])
                            else:
                                t.add_row(['OPTIMAL '+str(alt_crews[num-1]), opt_obj_mult[num], opt_elapsed_mult[num], opt_num_tap])
                    if approx:
                        t.add_row(['approx-LAFO', LAFO_obj_mult, LAFO_elapsed, LAFO_num_tap])
                        t.add_row(['approx-LASR', LASR_obj_mult, LASR_elapsed, LASR_num_tap])
                    # if len(damaged_links) < 32:
                    #     t.add_row(['BeamSearch', beamsearch_obj, beamsearch_elapsed, beamsearch_num_tap])
                    if full:
                        t.add_row(['FULL ALGO', algo_obj_mult, algo_elapsed, algo_num_tap])
                    if beam_search:
                        t.add_row(['BeamSearch_relaxed', r_algo_obj_mult, r_algo_elapsed, r_algo_num_tap])
                        try:
                            t.add_row(['BeamSearch_sa_seed', sa_r_algo_obj_mult, sa_r_algo_elapsed, sa_r_algo_num_tap])
                        except:
                            pass
                    if sa:
                        t.add_row(['Simulated Annealing', sa_obj_mult, sa_elapsed, sa_num_tap])
                    t.add_row(['GREEDY', greedy_obj_mult, greedy_elapsed, greedy_num_tap])
                    t.add_row(['LG', lg_obj_mult, lg_elapsed, lg_num_tap])
                    t.add_row(['IMPORTANCE', importance_obj_mult, importance_elapsed, importance_num_tap])
                    t.add_row(['SPT', SPT_obj_mult, SPT_elapsed, SPT_num_tap])
                    t.set_style(MSWORD_FRIENDLY)
                    print(t)

                fname = save_dir + '/results.csv'
                with open(fname, 'w', newline='') as f:
                    f.write(t.get_csv_string(header=False))

                print(swapped_links)
                # pdb.set_trace()

                # print('bs totchild vs uncommon vs common vs purged')
                # print(tot_childbs, uncommon_numberbs, common_numberbs, num_purgebs)
                if beam_search:
                    print('r totchild vs uncommon vs common vs purged')
                    print(tot_childr, uncommon_numberr, common_numberr, num_purger)

                print('PATHS')
                if approx:
                    print('approx-LAFO: ', LAFO_soln)
                    print('---------------------------')
                    print('approx-LASR: ', LASR_soln)
                    print('---------------------------')
                if opt:
                    print('optimal by brute force: ', opt_soln)
                    print('---------------------------')
                if full:
                    print('algorithm: ', algo_path)
                    print('---------------------------')
                if beam_search:
                    print('beamsearch: ', r_algo_path)
                    print('---------------------------')
                    try:
                        print('beamsearch with sa seed: ', sa_r_algo_path)
                        print('---------------------------')
                    except:
                        pass
                if sa:
                    print('simulated annealing: ', sa_soln)
                    print('---------------------------')
                print('greedy: ', greedy_soln)
                print('---------------------------')
                print('lazy_greedy: ', lg_soln)
                print('---------------------------')
                print('importance factors: ', importance_soln)
                print('---------------------------')
                print('shortest processing time: ', SPT_soln)

        if graphing:
            plotTimeOBJ(save_dir,bs_time_list,bs_OBJ_list,sa_time_list,sa_OBJ_list)

