import pickle
import os
from copy import deepcopy
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

class Network:
    def __init__(self, networkFile="", demandFile=""):
        """
        Class initializer; if both a network file and demand file are specified,
        will read these files to fill the network data structure.
        """
        self.netfile = networkFile
        self.tripfile = demandFile

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


def create_network(netfile=None, tripfile=None):
    net = Network(netfile, tripfile)
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

            net.link = {}
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
                            net.link[ij] = {}

                            net.link[ij]['flow'] = flow
                            net.link[ij]['cost'] = cost
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
                if last_line.find('obj') >= 0:
                    obj = last_line[last_line.find('obj') + 3:].strip()
                    try:
                        tstt = float(obj[:obj.find(',')])
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
                    obj = last_line[last_line.find('obj') + 3:].strip()
                    try:
                        tstt = float(obj[:obj.find(',')])
                    except:
                        try_again = True

            os.remove('full_log.txt')

    os.remove('current_net.tntp')

    return tstt


def solve_UE(net=None, relax=False, eval_seq=False, flows=False, wu=True, rev=False):
    # modify the net.txt file to send to c code
    shutil.copy(net.netfile, 'current_net.tntp')
    networkFileName = "current_net.tntp"

    if len(net.not_fixed) > 0:
        df = pd.read_csv(networkFileName, delimiter='\t', skipinitialspace=True)

        for a_link in net.not_fixed:
            home = a_link[a_link.find("'(") + 2:a_link.find(",")]
            to = a_link[a_link.find(",") + 1:]
            to = to[:to.find(")")]
            try:
                ind = df[(df['Unnamed: 1'] == str(home)) & (df['Unnamed: 2'] == str(to))].index.tolist()[0]
            except:
                for col in df.columns:
                    if pd.api.types.is_string_dtype(df[col]): # if cols contain str type, strip()
                        df[col] = df[col].str.strip()
                df = df.replace({"":np.nan}) # replace any empty strings with nan
                ind = df[(df['Unnamed: 1'] == str(home)) & (df['Unnamed: 2'] == str(to))].index.tolist()[0]

            df.loc[ind, 'Unnamed: 3'] = 1e-10
            df.loc[ind, 'Unnamed: 5'] = 1e10

        df.to_csv('current_net.tntp', index=False, sep="\t")

    f = 'current_net.tntp'
    file_created = False
    while not file_created:
        if os.path.exists(f):
            file_created = True

    if rev:
        bush_loc = 'after/batch0.bin'
    else:
        bush_loc = 'before/batch0.bin'

    # if wu:
    #     folder_loc = "tap-b_wu/bin/tap "
    #     shutil.copy(bush_loc, 'batch0.bin')
    # else:
    folder_loc = "tap-b/bin/tap "
        # print('non_wu')

    start = time.time()
    cores = max(mp.cpu_count(),4)
    if type(net.tripfile) == list:
        temp = ""
        for item in net.tripfile:
            temp = temp + item + " "

        if eval_seq:
            args = shlex.split(folder_loc + "1e-7 " + str(len(net.tripfile)) + " " +"current_net.tntp " + temp + str(cores))
        else:
            if relax:
                args = shlex.split(folder_loc + "1e-4 " + str(len(net.tripfile)) + " " +"current_net.tntp " + temp + str(cores))
            else:
                args = shlex.split(folder_loc + "1e-6 " + str(len(net.tripfile)) + " " +"current_net.tntp " + temp + str(cores))

    else:
        if eval_seq:
            args = shlex.split(folder_loc + "1e-7 1 " +"current_net.tntp " + net.tripfile + " " + str(cores))
        else:
            if relax:
                args = shlex.split(folder_loc + "1e-4 1 " + "current_net.tntp " + net.tripfile + " " + str(cores))
            else:
                args = shlex.split(folder_loc + "1e-6 1 " + "current_net.tntp " + net.tripfile + " " + str(cores))

    popen = subprocess.run(args, stdout=subprocess.DEVNULL)
    elapsed = time.time() - start
    # print('tap elapsed: {}, wu: {}, eval_seq: {}, relax: {} '.format(elapsed, wu, eval_seq, relax))
    tstt = net_update(net, args, flows)

    return tstt


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


def eval_sequence(net, order_list, after_eq_tstt, before_eq_tstt, if_list=None, importance=False, is_approx=False, damaged_dict=None, num_crews=1, approx_params=None):
    """evaluates the total tstt for a repair sequence, does not write to memory"""
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
            tap_solved += 1
            tstt_after = solve_UE(net=net, eval_seq=True)

        tstt_list.append(tstt_after)

        if importance:
            curfp += if_list[link_id]
            fp.append(curfp * 100)

    tot_area = 0
    for i in range(len(days_list)):
        if i == 0:
            tstt = after_eq_tstt
        else:
            tstt = tstt_list[i - 1]

        tot_area += (tstt - before_eq_tstt) * days_list[i]

    return tot_area, tap_solved, tstt_list


def get_marginal_tstts(net, path, after_eq_tstt, before_eq_tstt, damaged_dict):
    _, _, tstt_list = eval_sequence(
        deepcopy(net), path, after_eq_tstt, before_eq_tstt, damaged_dict=damaged_dict)

    # tstt_list.insert(0, after_eq_tstt)

    days_list = []
    for link in path:
        days_list.append(damaged_dict[link])

    return tstt_list, days_list
