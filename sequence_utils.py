import pickle
import os
from copy import deepcopy
import operator as op
from functools import reduce
import shlex
import subprocess
import shutil
import pdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import multiprocessing as mp

SEQ_INFINITY = 99999
ALPHA = 0.15
BETA = 4.0
ALPHAff = 0
BETAff = 1
CORES = min(mp.cpu_count(),4)

class Network:
    def __init__(self, networkFile="", demandFile="", mc_weights=1, demand_mult=1):
        """Class initializer; if both a network file and demand file are specified,
        will read these files to fill the network data structure."""
        self.netfile = networkFile
        self.tripfile = demandFile
        self.mc_weights = mc_weights
        self.demand_mult = demand_mult


def save(fname, data, extension='pickle'):
    path = fname + "." + extension
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load(fname, extension='pickle'):
    path = fname + "." + extension
    with open(path, 'rb') as f:
        item = pickle.load(f)
    return item


def save_fig(plt_path, algo, tight_layout=True, fig_extension="png", resolution=300):
    plt_path = os.path.join(plt_path, "figures")
    os.makedirs(plt_path, exist_ok=True)
    path = os.path.join(plt_path, algo + "." + fig_extension)
    print("Saving figure", algo)

    if tight_layout:
        plt.tight_layout(pad=1)
    plt.savefig(path, format=fig_extension, dpi=resolution)


def create_network(netfile=None, tripfile=None, mc_weights=1, demand_mult=1):
    net = Network(netfile, tripfile, mc_weights=mc_weights, demand_mult=demand_mult)
    return net


def read_scenario(fname='ScenarioAnalysis.xlsx', sname='Moderate_1'):
    scenario_pd = pd.read_excel(fname, sname)
    dlinks = scenario_pd[scenario_pd['Link Condition'] == 1]['Link'].tolist()
    cdays = scenario_pd[scenario_pd['Link Condition'] == 1][
        'Closure day (day)'].tolist()

    damage_dict = {}
    for i in range(len(dlinks)):
        damage_dict[dlinks[i]] = cdays[i]
    return damage_dict


def percentChange(a,b):
    """returns percent change from a to b"""
    res = 100*(np.array(b)-np.array(a)) / np.array(a)
    return np.round(res,3)


def int_to_state(c, N):
    """returns z vector binary representation given integer representation where c
    is the integer and N is the number of broken links"""
    temp = np.binary_repr(c, width=N)
    z = np.array([int(s) for s in temp])
    return z


def orderlists(benefits, days, slack=0, rem_keys=None, reverse=True):
    """helper function for sorting/reversing lists"""
    bang4buck = np.array(benefits) / np.array(days)
    days = [x for __, x in sorted(zip(bang4buck, days), reverse=reverse)]
    benefits = [x for __, x in sorted(zip(bang4buck, benefits), reverse=reverse)]

    if rem_keys is not None:
        rem_keys = [x for __, x in sorted(zip(bang4buck, rem_keys), reverse=reverse)]
        return benefits, days, rem_keys
    return benefits, days


def write_tui(net, relax, eval_seq, warm_start, initial=False):
    """write tui file to be read by tap-b"""
    if eval_seq:
        prec = '1e-7'
    elif relax:
        prec = '1e-4'
    else:
        prec = '1e-6'

    with open('current_params.txt','w') as f2:
        f2.write('<NETWORK FILE> ')
        f2.write('current_net.tntp')
        f2.write('\n')
        if isinstance(net.tripfile,list):
            for item in net.tripfile:
                f2.write('<TRIPS FILE> ')
                f2.write(item)
                f2.write('\n')
        else:
            f2.write('<TRIPS FILE> ')
            f2.write(net.tripfile)
            f2.write('\n')
        f2.write('<CONVERGENCE GAP> ')
        f2.write(prec)
        f2.write('\n')
        f2.write('<MAX RUN TIME> ')
        f2.write(net.maxruntime)
        f2.write('\n')
        if isinstance(net.tripfile, list):
            f2.write('<NUMBER OF THREADS> 1')
            f2.write('\n')
            f2.write('<NUMBER OF BATCHES> ')
            f2.write(str(len(net.tripfile)))
            f2.write('\n')
        else:
            f2.write('<NUMBER OF THREADS> ')
            f2.write(str(CORES))
            f2.write('\n')
        f2.write('<FILE PATH> ')
        f2.write('./')
        f2.write('\n')
        f2.write('<DATA PATH> ')
        f2.write('./')
        f2.write('\n')
        if warm_start:
            f2.write('<WARM START>')
            f2.write('\n')
        if initial:
            f2.write('<STORE MATRICES>')
            f2.write('\n')
            f2.write('<STORE BUSHES>')
            f2.write('\n')
        if net.demand_mult != 1:
            f2.write('<DEMAND MULTIPLIER> ')
            f2.write(str(net.demand_mult))
            f2.write('\n')


def find_class_tstt(net, args, flows=False):
    """ returns a list of total TSTT, class 1 TSTT, class 2 TSTT, etc. """
    if flows:
        f = "flows.txt"
        file_created = False
        st = time.time()
        while not file_created:
            if os.path.exists(f):
                file_created = True
            if time.time()-st >10:
                popen = subprocess.call(args, stdout=subprocess.DEVNULL)

            net.linkDict = {}
            tstt = 0
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
                            net.linkDict[ij] = {}
                            net.linkDict[ij]['flow'] = flow
                            net.linkDict[ij]['cost'] = cost
                            tstt += flow*cost
                        except:
                            break
                os.remove('flows.txt')

    try_again = False
    f = "full_log.txt"
    file_created = False
    while not file_created:
        if os.path.exists(f):
            file_created = True

        if file_created:
            with open(f, "r") as log_file:
                last_line = log_file.readlines()[-1]
                if last_line.find('TSTT:') >= 0:
                    obj = last_line[last_line.find('TSTT:') + 5:].strip()
                    try:
                        tstt = float(obj)
                    except:
                        try_again = True
                else:
                    try_again = True

            idx_wanted = None
            if try_again:
                with open(f, "r") as log_file:
                    lines = log_file.readlines()
                    for idx, line in enumerate(lines):
                        if line[:4] == 'next':
                            idx_wanted = idx-1
                            break
                    last_line = lines[idx_wanted]
                    obj = last_line[last_line.find('TSTT:') + 5:].strip()
                    try:
                        tstt = float(obj)
                    except:
                        try_again = True

            if isinstance(net.tripfile, list):
                num_classes = len(net.tripfile)
                class_tstt = [0]*(num_classes)
                with open(f, "r") as log_file:
                    temp = log_file.readlines()[-num_classes-1:]
                for i in range(num_classes):
                    active_line = temp[i]
                    obj = active_line[active_line.find('TSTT:') + 5:].strip()
                    try:
                        class_tstt[i] = float(obj)
                    except:
                        print('Error encountered in find_class_tstt for demand class '
                              + str(i+1))
                        return tstt
                class_tstt.insert(0,tstt)
            else:
                print('Find_class_tstt function called with only one class of demand \
                      present')
                return tstt

            os.remove('full_log.txt')

    os.remove('current_net.tntp')

    return class_tstt


def net_update(net, args, flows=False):
    if flows:
        f = "flows.txt"
        file_created = False
        st = time.time()
        while not file_created:
            if os.path.exists(f):
                file_created = True
            if time.time()-st >10:
                popen = subprocess.call(args, stdout=subprocess.DEVNULL)

            net.linkDict = {}
            tstt = 0
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
                            net.linkDict[ij] = {}
                            net.linkDict[ij]['flow'] = flow
                            net.linkDict[ij]['cost'] = cost
                            tstt += flow*cost
                        except:
                            break
                os.remove('flows.txt')

    try_again = False
    f = "full_log.txt"
    file_created = False
    while not file_created:
        if os.path.exists(f):
            file_created = True

        if file_created:
            with open(f, "r") as log_file:
                last_line = log_file.readlines()[-1]
                if last_line.find('TSTT:') >= 0:
                    obj = last_line[last_line.find('TSTT:') + 5:].strip()
                    try:
                        tstt = float(obj)
                    except:
                        try_again = True
                else:
                    try_again = True

            idx_wanted = None
            if try_again:
                with open(f, "r") as log_file:
                    lines = log_file.readlines()
                    for idx, line in enumerate(lines):
                        if line[:4] == 'next':
                            idx_wanted = idx-1
                            break
                    last_line = lines[idx_wanted]
                    obj = last_line[last_line.find('TSTT:') + 5:].strip()
                    try:
                        tstt = float(obj)
                    except:
                        try_again = True

            os.remove('full_log.txt')

    os.remove('current_net.tntp')

    return tstt


def solve_UE(
        net=None, relax=False, eval_seq=False, flows=False, warm_start=True, rev=False,
        multiclass=False, initial=False):
    """If mc_weights is a list, then finds TSTT for each class separately and weights to
    find overall TSTT. If multiclass, then reports TSTT for each class separately"""
    # modify the net.txt file to send to c code and create parameters file
    prep_st = time.time()
    try:
        if net.free_flow:
            free_flow = True
        else:
            free_flow = False
    except:
        free_flow = False

    if free_flow:
        shutil.copy('ff_net.tntp', 'current_net.tntp')
    elif len(net.art_links) > 0:
        shutil.copy('art_net.tntp', 'current_net.tntp')
    else:
        shutil.copy(net.netfile, 'current_net.tntp')
    networkFileName = "current_net.tntp"
    try:
        if net.free_flow:
            free_flow = True
        else:
            free_flow = False
    except:
        free_flow = False

    if len(net.not_fixed) > 0:
        df = pd.read_csv(networkFileName, delimiter='\t', skipinitialspace=True)
        for a_link in net.not_fixed:
            home = a_link[a_link.find("'(") + 2:a_link.find(",")]
            to = a_link[a_link.find(",") + 1:]
            to = to[:to.find(")")]
            try:
                ind = df[(df['Unnamed: 1'] == str(home)) & (df['Unnamed: 2'] == str(to))
                         ].index.tolist()[0]
            except:
                for col in df.columns: # If cols contain str type, strip()
                    if pd.api.types.is_string_dtype(df[col]):
                        df[col] = df[col].str.strip()
                    else:
                        try:
                            df[col] = df[col].str.strip()
                        except:
                            pass
                df = df.replace({"":np.nan}) # Replace any empty strings with nan
                ind = df[(df['Unnamed: 1'] == str(home)) & (df['Unnamed: 2'] == str(to))
                         ].index.tolist()[0]
            df.loc[ind, 'Unnamed: 5'] = SEQ_INFINITY
            if free_flow:
                df.loc[ind, 'Unnamed: 6'] = ALPHA
                df.loc[ind, 'Unnamed: 7'] = BETA

        df.to_csv('current_net.tntp', index=False, sep="\t")

    f = 'current_net.tntp'
    file_created = False
    while not file_created:
        if os.path.exists(f):
            file_created = True
    folder_loc = "tap-b/bin/tap "

    start = time.time()
    write_tui(net, relax, eval_seq, False, initial=initial)
    args = shlex.split(folder_loc + "current_params.txt")
    popen = subprocess.run(args, stdout=subprocess.DEVNULL)
    elapsed = time.time() - start

    if multiclass or isinstance(net.mc_weights,list):
        try:
            class_tstt = find_class_tstt(net, args, flows)
        except:
            print('error in executing net_update from solve_ue, retrying')
            shutil.copy('full_log.txt', 'full_log_error.txt')
            shutil.copy('current_net.tntp', 'current_net_error.tntp')
            shutil.copy('current_params.txt', 'current_params_error.txt')

            temp = 1
            if isinstance(net.tripfile, list):
                temp = len(net.tripfile)

            if rev:
                bush_loc = 'after-batch'
                mat_loc = 'after-matrix'
            else:
                bush_loc = 'before-batch'
                mat_loc = 'before-matrix'
            for i in range(temp):
                shutil.copy(bush_loc+str(i)+'.bin', 'batch'+str(i)+'.bin')
                shutil.copy(mat_loc+str(i)+'.bin', 'matrix'+str(i)+'.bin')

            write_tui(net, relax, eval_seq, True, initial=initial)
            args = shlex.split(folder_loc + "current_params.txt")
            popen = subprocess.run(args, stdout=subprocess.DEVNULL)
            shutil.copy('full_log.txt', 'full_log_error2.txt')
            class_tstt = find_class_tstt(net, args, flows)

        if isinstance(net.mc_weights, list):
            if len(net.mc_weights)==len(class_tstt)-1:
                class_tstt[0] = 0
                for i in range(len(net.mc_weights)):
                    class_tstt[0] += net.mc_weights[i]*class_tstt[i+1]
            else:
                print('User has provided {} mc_weights, and there are {} classes of \
                      demand. Returning UNWEIGHTED TSTT'.format(len(net.mc_weights),
                                                                len(class_tstt)-1))
        if multiclass:
            prep_time = time.time() - prep_st - elapsed
            tap_time = elapsed
            return class_tstt, prep_time, tap_time
        else:
            prep_time = time.time() - prep_st - elapsed
            tap_time = elapsed
            return class_tstt[0], prep_time, tap_time

    else:
        try:
            tstt = net_update(net, args, flows)
        except:
            try:
                print('error in executing net_update from solve_ue, retrying')
                shutil.copy('full_log.txt', 'full_log_error.txt')
                shutil.copy('current_net.tntp', 'current_net_error.tntp')
                shutil.copy('current_params.txt', 'current_params_error.txt')

                temp = 1
                if isinstance(net.tripfile, list):
                    temp = len(net.tripfile)

                if rev:
                    bush_loc = 'after-batch'
                    mat_loc = 'after-matrix'
                else:
                    bush_loc = 'before-batch'
                    mat_loc = 'before-matrix'
                for i in range(temp):
                    shutil.copy(bush_loc+str(i)+'.bin', 'batch'+str(i)+'.bin')
                    shutil.copy(mat_loc+str(i)+'.bin', 'matrix'+str(i)+'.bin')

                write_tui(net, relax, eval_seq, True, initial=initial)
                args = shlex.split(folder_loc + "current_params.txt")

                popen = subprocess.run(args, stdout=subprocess.DEVNULL)
                shutil.copy('full_log.txt', 'full_log_error2.txt')
                tstt = net_update(net, args, flows)
            except:
                print('second error in executing net_update from solve_ue, retrying')
                shutil.copy('full_log.txt', 'full_log_error2.txt')
                shutil.copy('current_net.tntp', 'current_net_error2.tntp')
                shutil.copy('current_params.txt', 'current_params_error2.txt')

                temp = 1
                if isinstance(net.tripfile, list):
                    temp = len(net.tripfile)

                if rev:
                    bush_loc = 'before-batch'
                    mat_loc = 'before-matrix'
                else:
                    bush_loc = 'after-batch'
                    mat_loc = 'after-matrix'
                for i in range(temp):
                    shutil.copy(bush_loc+str(i)+'.bin', 'batch'+str(i)+'.bin')
                    shutil.copy(mat_loc+str(i)+'.bin', 'matrix'+str(i)+'.bin')

                write_tui(net, relax, eval_seq, True, initial=initial)
                args = shlex.split(folder_loc + "current_params.txt")

                popen = subprocess.run(args, stdout=subprocess.DEVNULL)
                shutil.copy('full_log.txt', 'full_log_error3.txt')
                tstt = net_update(net, args, flows)
                
    prep_time = time.time() - prep_st - elapsed
    tap_time = elapsed
    return tstt, prep_time, tap_time


def gen_crew_seqs(order_list, damaged_dict, num_crews):
    """takes in the order in which projects start, the damaged dict, and the number of
    crews, and returns the order in which projects are completed within crews and the
    makespan"""
    if num_crews == 1:
        crew_seqs = order_list
        makespan = sum(damaged_dict.values())
    elif not isinstance(order_list[0],str):
        crews = [0]*num_crews
        for crew in range(num_crews):
            for link in order_list[crew]:
                crews[crew] += damaged_dict[link]
        makespan = max(crews)
        crew_seqs = order_list
    else:
        crew_seqs = [[] for i in range(num_crews)]
        crew_order_list = []
        crews = [0]*num_crews
        which_crew = dict()

        temp = damaged_dict[order_list[0]]
        crew_order_list.append(order_list[0])
        for ij in order_list[1:num_crews]:
            if damaged_dict[ij] < temp:
                temp = damaged_dict[ij]
                crew_order_list[0] = ij

        crews[0]+=damaged_dict[crew_order_list[0]]
        which_crew[crew_order_list[0]] = 0
        crew_seqs[0].append(crew_order_list[0])

        for link in order_list:
            if link not in crew_order_list:
                which_crew[link] = crews.index(min(crews))
                crews[which_crew[link]] += damaged_dict[link]
                crew_seqs[which_crew[link]].append(link)
                if crews[which_crew[link]] == max(crews):
                    crew_order_list.append(link)
                else:
                    crew_order_list.insert(len(crew_order_list) - num_crews +
                        sorted(crews).index(crews[which_crew[link]]) + 1 ,link)
        makespan = max(crews)

    return crew_seqs, makespan


def gen_single_seq(crew_seqs, damaged_dict, num_crews):
    if isinstance(crew_seqs[0],str):
        return crew_seqs
    else:
        order_list = []
        crews = [0]*num_crews
        pointer = [0]*num_crews

        for i in range(num_crews):
            order_list.append(crew_seqs[i][0])
            crews[i] += damaged_dict[crew_seqs[i][0]]
            pointer[i] += 1

        while len(order_list) < len(damaged_dict):
            i = np.argmin(crews)
            link = crew_seqs[i][pointer[i]]
            order_list.append(link)
            crews[i] += damaged_dict[link]
            pointer[i] += 1
        return order_list


def gen_crew_order(order_list, damaged_dict=None, num_crews=1):
    """takes in the order in which projects start, the damaged dict, and the
    number of crews, and returns the order in which projects finish, which crew
    completes each project in that ordered list, and the days list"""
    if num_crews == 1:
        crew_order_list = order_list
        which_crew = None
    else:
        crew_order_list = []
        crews = [0]*num_crews
        which_crew = dict()

        temp = damaged_dict[order_list[0]]
        crew_order_list.append(order_list[0])
        for ij in order_list[1:num_crews]:
            if damaged_dict[ij] < temp:
                temp = damaged_dict[ij]
                crew_order_list[0] = ij

        crews[0]+=damaged_dict[crew_order_list[0]]
        which_crew[crew_order_list[0]] = 0

        for link in order_list:
            if link not in crew_order_list:
                which_crew[link] = crews.index(min(crews))
                crews[which_crew[link]] += damaged_dict[link]
                if crews[which_crew[link]] == max(crews):
                    crew_order_list.append(link)
                else:
                    crew_order_list.insert(len(crew_order_list) - num_crews +
                        sorted(crews).index(crews[which_crew[link]]) + 1 ,link)

    days_list = []
    crews = [0]*num_crews
    total_days = 0
    for link_id in crew_order_list:
        if num_crews == 1:
            days_list.append(damaged_dict[link_id])
        else:
            crews[which_crew[link_id]] += damaged_dict[link_id]
            days_list.append(crews[which_crew[link_id]] - total_days)
            total_days = max(crews)

    return crew_order_list, which_crew, days_list


def gen_decomp_crew_order(order_list, damaged_dict=None, num_crews=1):
    """takes in the order in which projects start by crew, the damaged dict, and the
    number of crews, and returns the order in which projects finish, which crew
    completes each project in that ordered list, and the days list"""
    if num_crews == 1:
        crew_order_list = order_list
        which_crew = None
    else:
        crew_order_list = []
        crews = [0]*num_crews
        which_crew = dict()
        pointer = [0]*num_crews

        for i in range(num_crews):
            for link in order_list[i]:
                which_crew[link] = i
            link = order_list[i][0]
            crews[i] += damaged_dict[link]
            if crews[i] == max(crews):
                crew_order_list.append(link)
            else:
                crew_order_list.insert(len(crew_order_list) - num_crews +
                        sorted(crews).index(crews[i]) + 1 ,link)
            pointer[i] += 1

        for j in range(len(damaged_dict)-num_crews):
            i = crews.index(min(crews))
            try:
                link = order_list[i][pointer[i]]
                pointer[i] += 1
                crews[i] += damaged_dict[link]
            except:
                for val in sorted(crews, reverse=True):
                    try:
                        k = crews.index(val)
                        link = order_list[k][pointer[k]]
                        pointer[k] += 1
                        crews[i] += damaged_dict[link]
                        which_crew[link] = i
                    except:
                        pass
            if crews[i] == max(crews):
                crew_order_list.append(link)
            else:
                crew_order_list.insert(len(crew_order_list) - num_crews +
                    sorted(crews).index(crews[i]) + 1 ,link)

    days_list = []
    crews = [0]*num_crews
    total_days = 0
    for link_id in crew_order_list:
        if num_crews == 1:
            days_list.append(damaged_dict[link_id])
        else:
            crews[which_crew[link_id]] += damaged_dict[link_id]
            days_list.append(crews[which_crew[link_id]] - total_days)
            total_days = max(crews)

    return crew_order_list, which_crew, days_list


def eval_sequence(
        net, order_list, after_eq_tstt, before_eq_tstt, if_list=None, importance=False,
        is_approx=False, num_crews=1, approx_params=None, multiclass=False):
    """evaluates the total tstt for a repair sequence, does not write to memory
    if multiclass=True, then evaluates the total area for each class separately
    approx and multiclass cannot be active simultaneously"""
    damaged_links = list(net.damaged_dict.keys())
    times = [0]*3 # eval combinations, prep for tap, eval taps
    tap_solved = 0
    days_list = []
    tstt_list = []
    fp = None

    if importance:
        fp = []
        firstfp = 1
        for link_id in order_list:
            firstfp -= if_list[link_id]
        fp.append(firstfp * 100)
        curfp = firstfp

    if isinstance(order_list[0], str):
        to_visit = order_list
    else:
        to_visit = list(reduce(op.concat, order_list))
    added = []

    # Crew order list is the order in which projects complete
    if isinstance(order_list[0], str):
        crew_order_list, which_crew, days_list = gen_crew_order(
            order_list, damaged_dict=net.damaged_dict, num_crews=num_crews)
    else:
        crew_order_list, which_crew, days_list = gen_decomp_crew_order(
            order_list, damaged_dict=net.damaged_dict, num_crews=num_crews)

    if multiclass and isinstance(net.tripfile, list):
        net.not_fixed = set(to_visit)
        after_eq_tstt_mc, prep_time, tap_time = solve_UE(net=net, eval_seq=True,
            multiclass=multiclass)
        times[1] += prep_time
        times[2] += tap_time
        net.not_fixed = set([])
        before_eq_tstt_mc, prep_time, tap_time = solve_UE(net=net, eval_seq=True,
            multiclass=multiclass)
        times[1] += prep_time
        times[2] += tap_time

    for link_id in crew_order_list:
        added.append(link_id)
        not_fixed = set(to_visit).difference(set(added))
        net.not_fixed = set(not_fixed)

        if is_approx:
            prep_st = time.time()
            state = list(set(damaged_links).difference(net.not_fixed))
            state = [damaged_links.index(i) for i in state]
            pattern = np.zeros(len(damaged_links))
            pattern[(state)] = 1
            prep_time = time.time() - prep_st
            tap_st = time.time()
            tstt_after = (approx_params[0].predict(pattern.reshape(1, -1), verbose=0)
                          * approx_params[2] + approx_params[1])
            tap_time = time.time() - tap_st
            tstt_after = tstt_after[0][0]
        else:
            tap_solved += 1
            tstt_after, prep_time, tap_time = solve_UE(net=net, eval_seq=True,
                multiclass=multiclass)

        times[1] += prep_time
        times[2] += tap_time
        tstt_list.append(tstt_after)

        # Check for tstt's less than before eq tstt's by greater than 1%
        if multiclass:
            for i in range(len(tstt_after)):
                if (before_eq_tstt_mc[i] - tstt_after[i]) / before_eq_tstt_mc[i] > 0.01:
                    print('tstt after repairing link {} is lower than tstt before eq \
                          by {} ({} percent) for class {}'.format(str(link_id),
                          round(before_eq_tstt_mc[i] - tstt_after[i], 2),
                          round((before_eq_tstt_mc[i] - tstt_after[i])
                                / before_eq_tstt_mc[i]*100, 5), i))
                    f = 'troubleflows'+str(i)+'.txt'
                    count = 0
                    while os.path.exists(f):
                        count += 1
                        f = 'troubleflows'+str(i)+'-'+str(count)+'.txt'
                    shutil.copy('flows.txt', f)

        if importance:
            curfp += if_list[link_id]
            fp.append(curfp * 100)

    if multiclass and isinstance(net.tripfile, list):
        tot_area = [0]*(len(net.tripfile)+1)
        for j in range(len(net.tripfile)+1):
            for i in range(len(days_list)):
                if i == 0:
                    tstt = after_eq_tstt_mc[j]
                else:
                    tstt = tstt_list[i - 1][j]
                tot_area[j] += (tstt - before_eq_tstt_mc[j]) * days_list[i]
    else:
        tot_area = 0
        for i in range(len(days_list)):
            if i == 0:
                tstt = after_eq_tstt
            else:
                tstt = tstt_list[i - 1]
            tot_area += (tstt - before_eq_tstt) * days_list[i]

    return tot_area, tap_solved, tstt_list, times


def get_attributes(net_before, first_b, last_b, swapped_links):
    """builds a dictionary of damaged links' attributes"""
    # Get repair durations
    damaged_attributes = dict()
    damaged_links = list(net_before.damaged_dict.keys())
    for link in damaged_links:
        damaged_attributes[link] = list()
        damaged_attributes[link].append(net_before.damaged_dict[link])

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
    return damaged_attributes


def calc_all_TSTT(net_after):
    """calculates TSTT for every possible repair state and saves as a vector"""
    test_net = deepcopy(net_after)
    damaged_links = list(test_net.damaged_dict.keys())
    N = len(damaged_links)
    start = time.time()
    vecTSTT = np.zeros(2**N)
    TSTT_num_tap = 0
    for i in range(len(vecTSTT)):
        not_fixed = []
        pattern = int_to_state(i, N)
        for el in range(len(pattern)):
            if not pattern[el]:
                not_fixed.append(damaged_links[el])
        test_net.not_fixed = not_fixed
        vecTSTT[i], __, __ = solve_UE(net=test_net, eval_seq=True)
        TSTT_num_tap += 1
    TSTTs_time = time.time() - start
    print('Time to find all TSTTs: '+str(TSTTs_time))
    return vecTSTT, TSTTs_time, TSTT_num_tap


def calc_all_ML(approx_params, damaged_links, memory):
    """calculates ML estimation of TSTT for every possible repair state and saves as
    a vector"""
    N = len(damaged_links)
    ML_start = time.time()
    vecML = np.zeros(2**N)
    for i in range(len(vecML)):
        pattern = int_to_state(i, N)
        vecML[i] = (approx_params[0].predict(pattern.reshape(1, -1),
                      verbose=0) * approx_params[2] + approx_params[1])[0][0]
    for k, v in memory.items():
        pattern = np.ones(len(damaged_links),dtype=int)
        state = [damaged_links.index(i) for i in k]
        pattern[(state)] = 0
        num = int("".join(str(x) for x in pattern),2)
        vecML[num] = v
    ML_TSTTs_time = time.time() - ML_start
    return vecML, ML_TSTTs_time


def record_art_net(art_net):
    """builds a net file which includes artificial links"""
    shutil.copy(art_net.netfile, 'art_net.tntp')
    if len(art_net.art_links) > 0:
        df = pd.read_csv('art_net.tntp', delimiter='\t', skipinitialspace=True)
        for col in df.columns: # If cols contain str type, strip()
            if pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].str.strip()
        df = df.replace({"":np.nan}) # Replace any empty strings with nan
        for link in art_net.art_links.keys():
            df.loc[len(df.index)] = [np.nan,link[:link.find('-')],
                link[link.find('>')+1:], SEQ_INFINITY, art_net.art_links[link],
                art_net.art_links[link], ALPHAff, BETAff, 0.0, 0.0, 1, ';']
        if len(art_net.art_links) > 0:
            for i in range(1,10):
                if df.iloc[i,0].find('NUMBER OF LINKS') >= 0:
                    idx = i
                    break
            else:
                idx = -1
            if idx == -1:
                print('cannot find NUMBER OF LINKS to update')
            else:
                temp = df.iloc[idx,0]
                numLinks = temp[temp.find('> ')+1:]
                numLinks.strip()
                numLinks = int(numLinks)
                numLinks += len(art_net.art_links)
                temp = temp[:temp.find('> ')+2] + str(numLinks)
                df.iloc[idx,0] = temp
        df.to_csv('art_net.tntp', index=False, sep="\t")


def record_ff_net(net_after):
    """builds a net file with modified link attributes; sets alpha=1 and beta=0 for
    all links"""
    if len(net_after.art_links) > 0:
        shutil.copy('art_net.tntp', 'ff_net.tntp')
    else:
        shutil.copy(net_after.netfile, 'ff_net.tntp')
    df = pd.read_csv('ff_net.tntp', delimiter='\t', skipinitialspace=True)
    for i in df.index:
        try:
            if isinstance(df.loc[i][0],float):
                df.loc[i, 'Unnamed: 6'] = ALPHAff
                df.loc[i, 'Unnamed: 7'] = BETAff
        except:
            for col in df.columns: # If cols contain str type, strip()
                if pd.api.types.is_string_dtype(df[col]):
                    df[col] = df[col].str.strip()
            df = df.replace({"":np.nan}) # Replace any empty strings with nan
            if isinstance(df.loc[i][0],float):
                df.loc[i, 'Unnamed: 6'] = ALPHAff
                df.loc[i, 'Unnamed: 7'] = BETAff
    df.to_csv('ff_net.tntp', index=False, sep="\t")


def get_marginal_tstts(net, path, after_eq_tstt, before_eq_tstt, multiclass=False):
    """function not currently used"""
    __, __, tstt_list, __ = eval_sequence(deepcopy(net), path, after_eq_tstt,
        before_eq_tstt, multiclass=multiclass)

    # tstt_list.insert(0, after_eq_tstt)
    days_list = []
    for link in path:
        days_list.append(net.damaged_dict[link])

    return tstt_list, days_list
