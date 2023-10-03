# file contains beam search 'search' function and subfunctions
from sequence_utils import *
import math

class sNode():
    """A node class for bi-directional search for pathfinding"""

    def __init__(self, damaged_dict, visited=None, link_id=None, parent=None,
                 tstt_after=None, tstt_before=None, level=None, forward=True,
                 relax=False, not_fixed=None):

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

        self.assign_char(damaged_dict)

    def __str__(self):
        return str(self.path)

    def assign_char(self, damaged_dict):

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

            self.realized = self.parent.realized + (self.tstt_before
                - self.before_eq_tstt) * self.days
            self.realized_u = self.parent.realized_u + (self.tstt_before
                - self.before_eq_tstt) * self.days * (1 + self.err_rate)
            self.realized_l = self.parent.realized_l + (self.tstt_before
                - self.before_eq_tstt) * self.days * (1 - self.err_rate)
            self.not_visited = set(damaged_dict.keys()).difference(self.visited)

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
                self.realized = (self.tstt_before-self.before_eq_tstt) * self.days
                self.realized_u = (self.tstt_before-self.before_eq_tstt) * self.days *(1
                    + self.err_rate)
                self.realized_l = (self.tstt_before-self.before_eq_tstt) * self.days *(1
                    - self.err_rate)
                self.days_past = self.days

            else:
                self.realized = 0
                self.realized_u = 0
                self.realized_l = 0
                self.days = 0
                self.days_past = self.days

    def __eq__(self, other):
        return self.fixed == other.fixed


def get_se_nodes(damaged_dict, after_eq_tstt, before_eq_tstt, relax=False):
    """initiate start and end nodes for beam search"""
    damaged_links = damaged_dict.keys()
    start_node = sNode(damaged_dict, tstt_after=after_eq_tstt)
    start_node.before_eq_tstt = before_eq_tstt
    start_node.after_eq_tstt = after_eq_tstt
    start_node.realized = 0
    start_node.realized_u = 0
    start_node.level = 0
    start_node.visited, start_node.not_visited = set([]), set(damaged_links)
    start_node.fixed, start_node.not_fixed = set([]), set(damaged_links)

    end_node = sNode(damaged_dict, tstt_before=before_eq_tstt, forward=False)
    end_node.before_eq_tstt = before_eq_tstt
    end_node.after_eq_tstt = after_eq_tstt
    end_node.level = 0
    end_node.visited, end_node.not_visited = set([]), set(damaged_links)
    end_node.fixed, end_node.not_fixed = set(damaged_links), set([])

    start_node.relax = relax
    end_node.relax = relax

    return start_node, end_node


def get_successors_f(node, iter_num, wb, bb, damaged_dict):
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


def get_successors_b(node, iter_num, wb, bb, damaged_dict):
    """given a state, returns list of bridges that have not yet been removed"""
    not_visited = node.not_visited
    successors = []

    if node.level != 0:
        tail = node.path[-1]
        for a_link in not_visited:
            if iter_num >= 1000:
                if - wb[tail]*damaged_dict[a_link] + bb[a_link]*damaged_dict[tail] < 0:
                    continue
            successors.append(a_link)
    else:
        successors = not_visited

    return successors


def expand_sequence_f(node, a_link, level, net_after, wb, bb, wb_update, bb_update,
                      memory):
    """given a link and a node, expands the sequence"""
    solved = 0
    tstt_before = node.tstt_after
    damaged_dict = net_after.damaged_dict

    net = create_network(net_after.netfile, net_after.tripfile,
        mc_weights=net_after.mc_weights, demand_mult=net_after.demand_mult)
    net.not_fixed = set(node.not_fixed).difference(set([a_link]))
    net.art_links = net_after.art_links
    net.maxruntime = net_after.maxruntime
    if frozenset(net.not_fixed) in memory.keys():
        tstt_after = memory[frozenset(net.not_fixed)]
    else:
        tstt_after, __, __ = solve_UE(net=net, relax=node.relax)
        memory[frozenset(net.not_fixed)] = (tstt_after)
        solved = 1

    diff = tstt_before - tstt_after
    if wb_update[a_link] > diff:
        wb_update[a_link] = diff
    if bb_update[a_link] < diff:
        bb_update[a_link] = diff

    if wb[a_link] > diff:
        wb[a_link] = diff
    if bb[a_link] < diff:
        bb[a_link] = diff

    node = sNode(damaged_dict, link_id=a_link, parent=node, tstt_after=tstt_after,
                 tstt_before=tstt_before, level=level, relax=node.relax)
    del net

    return node, solved, wb, bb, wb_update, bb_update, memory


def expand_sequence_b(node, a_link, level, net_after, wb, bb, wb_update, bb_update,
                      memory):
    """given a link and a node, expands the sequence"""
    solved = 0
    tstt_after = node.tstt_before
    damaged_dict = net_after.damaged_dict

    net = create_network(net_after.netfile, net_after.tripfile,
        mc_weights=net_after.mc_weights, demand_mult=net_after.demand_mult)
    net.not_fixed = node.not_fixed.union(set([a_link]))
    net.art_links = net_after.art_links
    net.maxruntime = net_after.maxruntime

    if frozenset(net.not_fixed) in memory.keys():
        tstt_before = memory[frozenset(net.not_fixed)]
    else:
        tstt_before, __, __ = solve_UE(net=net, relax=node.relax)
        memory[frozenset(net.not_fixed)] = tstt_before
        solved = 1

    diff = tstt_before - tstt_after
    if wb_update[a_link] > diff:
        wb_update[a_link] = diff
    if bb_update[a_link] < diff:
        bb_update[a_link] = diff

    if wb[a_link] > diff:
        wb[a_link] = diff
    if bb[a_link] < diff:
        bb[a_link] = diff

    node = sNode(damaged_dict, link_id=a_link, parent=node, tstt_after=tstt_after,
                 tstt_before=tstt_before, level=level, relax=node.relax)
    del net

    return node, solved, wb, bb, wb_update, bb_update, memory


def get_minlb(damaged_dict, node, fwd_node, bwd_node, orderedb_benefits,
              orderedw_benefits, ordered_days, forward_tstt, backward_tstt, bfs=None,
              uncommon_number=0, common_number=0):
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
        node.ub = fwd_node.realized_u + bwd_node.realized_u + (fwd_node.tstt_after
            - node.before_eq_tstt) * ordered_days[0]
        node.lb = fwd_node.realized_l + bwd_node.realized_l + (fwd_node.tstt_after
            - node.before_eq_tstt) * ordered_days[0]
        cur_obj = fwd_node.realized + bwd_node.realized + (fwd_node.tstt_after
            - node.before_eq_tstt) * ordered_days[0]

        lo_link = list(set(damaged_dict.keys()).difference(set(fwd_node.path).union(
                       set(bwd_node.path))))
        new_feasible_path = fwd_node.path + [str(lo_link[0])] + bwd_node.path[::-1]

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

    # Find lower bound from forwards
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
            node.lb += (b_tstt*fwd_days_w[0] +(bwd_node.tstt_before-node.before_eq_tstt)
                * fwd_days_w[0])
        common_number += 1
    else:
        node.lb += ((fwd_node.tstt_after-node.before_eq_tstt)*min(days_w)
            + (bwd_node.tstt_before-node.before_eq_tstt) * (sum(days_w)-min(days_w)))
        uncommon_number += 1

    lb2 = (fwd_node.tstt_after-node.before_eq_tstt)*min(days_w) + (bwd_node.tstt_before
           - node.before_eq_tstt) * (sum(days_w)-min(days_w))
    node.lb = max(node.lb, lb2 + orig_lb)

    # Find upper bound from forwards
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
            node.ub += b_tstt*fwd_days_b[0] + (bwd_node.tstt_before
                       - node.before_eq_tstt) * fwd_days_b[0]
        common_number += 1
    else:
        node.ub += sum(days_b) * (forward_tstt - node.before_eq_tstt)
        uncommon_number += 1

    node.ub = min(node.ub, sum(days_b) * (forward_tstt - node.before_eq_tstt) + orig_ub)
    if node.lb > node.ub:
        if not node.relax:
            pass
        if abs(node.lb - node.ub) < 5:
            node.ub, node.lb = node.lb, node.ub
        else:
            print('node.lb > node.ub for node : ' + str(node) + ' and bwd node : '
                  + str(bwd_node) + '. Forward TSTT = ' + str(forward_tstt)
                  + ' and backward TSTT = ' + str(backward_tstt))
            print('Worst benefits : ' + str(w) + ' and best benefits : ' + str(b))
            pass

    return uncommon_number, common_number


def set_bounds_bif(
        damaged_dict, wb, bb, node, open_list_b, end_node, front_to_end=True,
        bfs=None, uncommon_number=0, common_number=0):
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

        forward_tstt = node.tstt_after
        backward_tstt = other_end.tstt_before
        uncommon_number, common_number = get_minlb(damaged_dict, node, node, other_end,
            orderedb_benefits, orderedw_benefits, ordered_days, forward_tstt,
            backward_tstt, bfs=bfs, uncommon_number=uncommon_number,
            common_number=common_number)

        if node.lb < minlb:
            minlb = node.lb
            minother_end = other_end
        if node.ub < maxub:
            maxub = node.ub

    node.lb = minlb
    node.ub = maxub

    if node.lb > node.ub:
        print('node.lb > node.ub for node : ' + str(node) + ' and bwd node : '
              + str(node) + '. Forward TSTT = ' + str(forward_tstt)
              + ' and backward TSTT = ' + str(backward_tstt))

    return uncommon_number, common_number


def set_bounds_bib(
        damaged_dict, wb, bb, node, open_list_f, start_node, front_to_end=True,
        bfs=None, uncommon_number=0, common_number=0):
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
        uncommon_number, common_number = get_minlb(damaged_dict, node, other_end, node,
            orderedb_benefits, orderedw_benefits, ordered_days, forward_tstt,
            backward_tstt, bfs=bfs, uncommon_number=uncommon_number,
            common_number=common_number)

        if node.lb < minlb:
            minlb = node.lb
        if node.ub < maxub:
            maxub = node.ub

    node.lb = minlb
    node.ub = maxub

    if node.lb > node.ub:
        print('node.lb > node.ub for node : ' + str(node) + ' and bwd node : '
              + str(node) + '. Forward TSTT = ' + str(forward_tstt)
              + ' and backward TSTT = ' + str(backward_tstt))

    return uncommon_number, common_number


def expand_forward(
        net_after, wb, bb, wb_update, bb_update, memory, start_node, end_node,
        minimum_ff_n, open_list_b, open_list_f, bfs, num_tap_solved, iter_num,
        front_to_end=False, uncommon_number=0, tot_child=0, common_number=0):
    """expand the search tree forwards"""

    damaged_dict = net_after.damaged_dict
    damaged_links = damaged_dict.keys()
    fvals = [node.f for node in sum(open_list_f.values(),[])]
    update_bfs = False
    minfind = np.argmin(fvals)
    minimum_ff_n = sum(open_list_f.values(),[])[minfind]
    current_node = minimum_ff_n
    open_list_f[minimum_ff_n.level].remove(minimum_ff_n)
    if len(open_list_f[minimum_ff_n.level]) == 0:
        del open_list_f[minimum_ff_n.level]
    cur_visited = current_node.visited

    can1, can2 = False, False
    if len(damaged_links)-len(cur_visited) in open_list_b.keys():
        can1 = True
    if len(damaged_links)-len(cur_visited) - 1 in open_list_b.keys():
        can2 = True

    go_through = []
    if can1:
        go_through.extend(open_list_b[len(damaged_links) - len(cur_visited)])
    if can2:
        go_through.extend(open_list_b[len(damaged_links) - len(cur_visited) - 1])

    for other_end in go_through:
        if (len(set(other_end.visited).intersection(set(cur_visited))) == 0
                and (len(damaged_links)
                     - len(set(other_end.visited).union(set(cur_visited)))) == 1):
            lo_link = set(damaged_links).difference(
                set(other_end.visited).union(set(cur_visited)))
            lo_link = lo_link.pop()
            cur_soln = current_node.g + other_end.g + (current_node.tstt_after
                - current_node.before_eq_tstt) * damaged_dict[lo_link]
            cur_path = current_node.path + [str(lo_link)] + other_end.path[::-1]
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
        return (open_list_f, num_tap_solved, current_node.level, minimum_ff_n,
                tot_child, uncommon_number, common_number, update_bfs, wb, bb,
                wb_update, bb_update, memory)
    if iter_num > 3*len(damaged_links):
        if current_node.f > bfs.cost:
            print('Iteration ' + str(iter_num) + ', current node ' + str(current_node)
                  + ' is pruned.')
            return (open_list_f, num_tap_solved, current_node.level, minimum_ff_n,
                    tot_child, uncommon_number, common_number, update_bfs, wb, bb,
                    wb_update, bb_update, memory)

    # Generate children
    eligible_expansions = get_successors_f(current_node, iter_num, wb, bb, damaged_dict)
    children = []
    current_level = current_node.level
    for a_link in eligible_expansions:

        # Create new node
        current_level = current_node.level + 1
        new_node, solved, wb, bb, wb_update, bb_update, memory = expand_sequence_f(
            current_node, a_link, current_level, net_after, wb, bb, wb_update,
            bb_update, memory)
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
        # Set upper and lower bounds
        tot_child += 1
        uncommon_number, common_number = set_bounds_bif(damaged_dict, wb, bb, child,
            open_list_b, front_to_end=front_to_end, end_node=end_node, bfs=bfs,
            uncommon_number=uncommon_number, common_number=common_number)

        if child.ub <= bfs.cost:
            if len(child.path) == len(damaged_links):
                if child.g < bfs.cost:
                    bfs.cost = child.g
                    bfs.path = child.path
                    update_bfs = True
        if child.lb == child.ub:
            continue
        if child.lb > child.ub:
            print('node.lb > node.ub for node : ' + str(child) + ' and bwd node : '
                  + str(child) + '. Lower bound = ' + str(child.lb)
                  + ' and upper bound = ' + str(child.ub))

        child.g = child.realized
        child.f = child.lb

        if iter_num > 3*len(damaged_links):
            if child.f > bfs.cost:
                # print('Iter ' + str(iter_num) + ' Child node ' + str(child)
                #       + ' not added because child.f < bfs.cost.')
                continue

        if current_level not in open_list_f.keys():
            open_list_f[current_level] = []
        open_list_f[current_level].append(child)
        del child

    return (open_list_f, num_tap_solved, current_level, minimum_ff_n, tot_child,
            uncommon_number, common_number, update_bfs, wb, bb, wb_update, bb_update,
            memory)


def expand_backward(
        net_after, wb, bb, wb_update, bb_update, memory, start_node, end_node,
        minimum_bf_n, open_list_b, open_list_f, bfs, num_tap_solved, iter_num,
        front_to_end=False, uncommon_number=0, tot_child=0, common_number=0):
    """expand the search tree forwards"""

    damaged_dict = net_after.damaged_dict
    damaged_links = damaged_dict.keys()
    fvals = [node.f for node in sum(open_list_b.values(),[])]
    update_bfs = False
    minfind = np.argmin(fvals)
    minimum_bf_n = sum(open_list_b.values(),[])[minfind]
    current_node = minimum_bf_n
    open_list_b[minimum_bf_n.level].remove(minimum_bf_n)

    if len(open_list_b[minimum_bf_n.level]) == 0:
        del open_list_b[minimum_bf_n.level]
    cur_visited = current_node.visited

    can1, can2 = False, False
    if len(damaged_dict)-len(cur_visited) in open_list_f.keys():
        can1 = True
    if len(damaged_dict)-len(cur_visited) - 1 in open_list_f.keys():
        can2 = True

    go_through = []
    if can1:
        go_through.extend(open_list_f[len(damaged_dict)-len(cur_visited)])
    if can2:
        go_through.extend(open_list_f[len(damaged_dict)-len(cur_visited)-1])

    for other_end in go_through:
        if (len(set(other_end.visited).intersection(set(cur_visited))) == 0
                and (len(damaged_links)
                     - len(set(other_end.visited).union(set(cur_visited)))) == 1):
            lo_link = set(damaged_links).difference(
                set(other_end.visited).union(set(cur_visited)))
            lo_link = lo_link.pop()
            cur_soln = current_node.g + other_end.g + (other_end.tstt_after
                - other_end.before_eq_tstt) * damaged_dict[lo_link]
            if cur_soln <= bfs.cost:
                bfs.cost = cur_soln
                bfs.path = other_end.path + [str(lo_link)] + current_node.path[::-1]
                update_bfs = True

        elif set(other_end.not_visited) == set(cur_visited):
            cur_soln = current_node.g + other_end.g
            if cur_soln <= bfs.cost:
                bfs.cost = cur_soln
                bfs.path = other_end.path + current_node.path[::-1]
                update_bfs = True

    # Found the goal
    if current_node == start_node:
        return (open_list_b, num_tap_solved, current_node.level, minimum_bf_n,
                tot_child, uncommon_number, common_number, update_bfs, wb, bb,
                wb_update, bb_update, memory)

    if iter_num > 3*len(damaged_links):
        if current_node.f > bfs.cost:
            print('Iteration ' + str(iter_num) + ', current node ' + str(current_node)
                  + ' is pruned.')
            return (open_list_b, num_tap_solved, current_node.level, minimum_bf_n,
                    tot_child, uncommon_number, common_number, update_bfs, wb, bb,
                    wb_update, bb_update, memory)

    # Generate children
    eligible_expansions = get_successors_b(current_node, iter_num, wb, bb, damaged_dict)
    children = []
    current_level = current_node.level

    for a_link in eligible_expansions:
        # Create new node
        current_level = current_node.level + 1
        new_node, solved, wb, bb, wb_update, bb_update, memory = expand_sequence_b(
            current_node, a_link, current_level, net_after, wb, bb, wb_update,
            bb_update, memory)
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
        uncommon_number, common_number = set_bounds_bib(damaged_dict, wb, bb, child,
            open_list_f, front_to_end=front_to_end, start_node=start_node, bfs=bfs,
            uncommon_number=uncommon_number, common_number=common_number)

        if child.ub <= bfs.cost:
            # best_ub = child.ub
            if len(child.path) == len(damaged_links):
                if child.g < bfs.cost:
                    bfs.cost = child.g
                    bfs.path = child.path[::-1]
                    update_bfs = True
        if child.lb == child.ub:
            continue
        if child.lb > child.ub:
            print('node.lb > node.ub for node : ' + str(child) + ' and bwd node : '
                  + str(child) + '. Lower bound = ' + str(child.lb)
                  + ' and upper bound = ' + str(child.ub))
            pass

        child.g = child.realized
        child.f = child.lb

        if iter_num > 3*len(damaged_links):
            if child.f > bfs.cost:
                # print('Iter ' + str(iter_num) + ' Child node ' + str(child)
                #       + ' not added because child.f < bfs.cost.')
                continue

        if current_level not in open_list_b.keys():
            open_list_b[current_level] = []
        open_list_b[current_level].append(child)
        del child

    return (open_list_b, num_tap_solved, current_level, minimum_bf_n, tot_child,
            uncommon_number, common_number, update_bfs, wb, bb, wb_update, bb_update,
            memory)


def purge(damaged_links, open_list_b, open_list_f, beam_k, num_purged, closed_list_f,
          closed_list_b, f_activated, b_activated, iter_num, beta=256, n_0=0.1):
    """purge nodes as part of beam search"""
    if f_activated:
        for i in range(2, len(damaged_links) + 1):
            if i in open_list_f.keys():
                values_ofn = np.ones((beam_k)) * np.inf
                indices_ofn = np.ones((beam_k)) * np.inf
                keep_f = []
                if len(open_list_f[i]) == 0:
                    del open_list_f[i]
                    continue

                for idx, ofn in enumerate(open_list_f[i]):
                    try:
                        cur_max = np.max(values_ofn[:])
                        max_idx = np.argmax(values_ofn[:])
                    except:
                        pdb.set_trace()
                    if ofn.f < cur_max:
                        indices_ofn[max_idx] = idx
                        values_ofn[max_idx] = ofn.f

                indices_ofn = indices_ofn.ravel()
                indices_ofn = indices_ofn[indices_ofn < 1000]
                keepinds = np.concatenate((np.array(keep_f), indices_ofn),
                                          axis=None).astype(int)

                if len(indices_ofn) > 0:
                    num_purged += len(open_list_f[i]) - len(indices_ofn)
                closed_list_f[i] = list(np.array(open_list_f[i])[
                    list(set(range(len(open_list_f[i]))).difference(set(keepinds)))])
                open_list_f[i] = list(np.array(open_list_f[i])[keepinds])

                try:
                    olf_node = open_list_f[i][np.argmin([node.f for node in
                                                         open_list_f[i]])]
                    olf_min = olf_node.f
                    olf_ub = olf_node.ub
                except ValueError:
                    continue
                removed_from_closed = []

                for a_node in closed_list_f[i]:
                    if a_node.lb <= olf_min * (1 + n_0 * 0.5**(iter_num//beta)):
                        open_list_f[i].append(a_node)
                        removed_from_closed.append(a_node)
                    elif a_node.ub <= olf_ub:
                        open_list_f[i].append(a_node)
                        removed_from_closed.append(a_node)

                for a_node in removed_from_closed:
                    closed_list_f[i].remove(a_node)
                del removed_from_closed

    if b_activated:
        for i in range(2, len(damaged_links)+1):
            if i in open_list_b.keys():
                keep_b = []
                values_ofn = np.ones((beam_k)) * np.inf
                indices_ofn = np.ones((beam_k)) * np.inf
                if len(open_list_b[i]) == 0:
                    del open_list_b[i]
                    continue

                for idx, ofn in enumerate(open_list_b[i]):
                    try:
                        cur_max = np.max(values_ofn[:])
                        max_idx = np.argmax(values_ofn[:])
                    except:
                        pass

                    if ofn.f < cur_max:
                        indices_ofn[max_idx] = idx
                        values_ofn[max_idx] = ofn.f

                indices_ofn = indices_ofn.ravel()
                indices_ofn = indices_ofn[indices_ofn < 1000]
                keepinds = np.concatenate((np.array(keep_b), indices_ofn),
                                          axis=None).astype(int)

                if len(indices_ofn) > 0:
                    num_purged += len(open_list_b[i]) - len(indices_ofn)
                closed_list_b[i] = list(np.array(open_list_b[i])[
                    list(set(range(len(open_list_b[i]))).difference(set(keepinds)))])
                open_list_b[i] = list(np.array(open_list_b[i])[keepinds])

                try:
                    olb_node = open_list_b[i][np.argmin([node.f for node in
                                                         open_list_b[i]])]
                    olb_min = olb_node.f
                    olb_ub = olb_node.ub
                except ValueError:
                    continue
                removed_from_closed = []

                for a_node in closed_list_b[i]:
                        if a_node.lb <= olb_min * (1 + n_0 * 0.5**(iter_num//beta)):
                            open_list_b[i].append(a_node)
                            removed_from_closed.append(a_node)
                        elif a_node.ub <= olb_ub:
                            open_list_b[i].append(a_node)
                            removed_from_closed.append(a_node)

                for a_node in removed_from_closed:
                    closed_list_b[i].remove(a_node)
                del removed_from_closed

    return open_list_b, open_list_f, num_purged, f_activated, b_activated, {}, {}


def search(
        net_after, after_eq_tstt, before_eq_tstt, start_node, end_node, bfs, wb, bb,
        wb_update, bb_update, memory, beam_search=False, beam_k=None, get_feas=True,
        beta=128, gamma=128, graphing=False, bs_time_list=None, bs_OBJ_list=None):
    """Returns the best order to visit the set of nodes"""

    # ideas in Holte: (search that meets in the middle)
    # another idea is to expand the min priority, considering both backward and forward
    # open list pr(n) = max(f(n), 2g(n)) - min priority expands
    # U is best solution found so far - stop when:
    # U <= max(C, fminF, fminB, gminF + gminB + eps)
    # this is for front to end

    # bfs comes from the greedy heuristic or importance solution
    print('Starting search with a feasible solution with cost: ', bfs.cost)
    max_level_b = 0
    max_level_f = 0

    iter_count = 0
    art_link_dict = net_after.art_links
    damaged_dict = net_after.damaged_dict
    damaged_links = list(damaged_dict.keys())

    # Initialize both open and closed list for forward and backward directions
    open_list_f = {}
    open_list_f[max_level_f] = [start_node]
    closed_list_f = {}
    open_list_b = {}
    open_list_b[max_level_b] = [end_node]
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
            (open_list_f, num_tap_solved, level_f, minimum_ff_n, tot_child,
             uncommon_number, common_number, update_bfs, wb, bb, wb_update, bb_update,
             memory) = expand_forward(net_after, wb, bb, wb_update, bb_update, memory,
                start_node, end_node, minimum_ff_n, open_list_b, open_list_f, bfs,
                num_tap_solved, iter_count, front_to_end=False, uncommon_number=
                uncommon_number, tot_child=tot_child, common_number=common_number)
            max_level_f = max(max_level_f, level_f)
        else:
            (open_list_b, num_tap_solved, level_b, minimum_bf_n, tot_child,
             uncommon_number, common_number, update_bfs, wb, bb, wb_update, bb_update,
             memory) = expand_backward(net_after, wb, bb, wb_update, bb_update, memory,
                start_node, end_node, minimum_bf_n, open_list_b, open_list_f, bfs,
                num_tap_solved, iter_count, front_to_end=False, uncommon_number=
                uncommon_number, tot_child=tot_child, common_number=common_number)
            max_level_b = max(max_level_b, level_b)

        if update_bfs and graphing:
            bs_time_list.append(time.time() - search_timer)
            bs_OBJ_list.append(deepcopy(bfs.cost))

        all_open_f = sum(open_list_f.values(), [])
        all_open_b = sum(open_list_b.values(), [])
        len_f = len(all_open_f)
        len_b = len(all_open_b)

        if (len_f == 0 or len_b == 0) and bfs.path is not None:
            print('Either forward or backward open set has length 0')
            return (bfs.path, bfs.cost, num_tap_solved, tot_child, uncommon_number,
                    common_number, num_purged, wb, bb, wb_update, bb_update, memory)

        if max(minimum_bf_n.f, minimum_ff_n.f) >= bfs.cost and bfs.path is not None:
            print('Minimum lower bound is larger than feasible upper bound')
            return (bfs.path, bfs.cost, num_tap_solved, tot_child, uncommon_number,
                    common_number, num_purged, wb, bb, wb_update, bb_update, memory)

        # Stop after 2000 iterations
        if iter_count >= 2000:
            return (bfs.path, bfs.cost, num_tap_solved, tot_child, uncommon_number,
                    common_number, num_purged, wb, bb, wb_update, bb_update, memory)

        fvals = [node.f for node in all_open_f]
        minfind = np.argmin(fvals)
        minimum_ff_n = all_open_f[minfind]

        bvals = [node.f for node in all_open_b]
        minbind = np.argmin(bvals)
        minimum_bf_n = all_open_b[minbind]

        # Every 64 (128) iter, update bounds
        if iter_count < 512:
            up_freq = 64
        else:
            up_freq = 128

        if iter_count % up_freq==0 and iter_count!=0:
            print('----------')
            print(f'Search time elapsed: {(time.time() - search_timer) / 60}')
            print(f'max_level_f: {max_level_f}, max_level_b: {max_level_b}')
            print('Iteration: {}, best_found: {}'.format(iter_count, bfs.cost))
            if graphing:
                bs_time_list.append(time.time() - search_timer)
                bs_OBJ_list.append(deepcopy(bfs.cost))

            wb = deepcopy(wb_update)
            bb = deepcopy(bb_update)
            for k,v in open_list_f.items():
                for a_node in v:
                    __, __ = set_bounds_bif(damaged_dict, wb, bb, a_node, open_list_b,
                        front_to_end=False, end_node=end_node, bfs=bfs, uncommon_number=
                        uncommon_number, common_number=common_number)
                    a_node.f = a_node.lb

            for k, v in open_list_b.items():
                for a_node in v:
                    __, __ = set_bounds_bib(damaged_dict, wb, bb, a_node, open_list_f,
                        front_to_end=False, start_node=start_node, bfs=bfs,
                        uncommon_number=uncommon_number, common_number=common_number)
                    a_node.f = a_node.lb

        # every gamma iters, get feasible solution
        if iter_count % gamma==0 and iter_count!=0:
            if get_feas:
                remaining = set(damaged_links).difference(minimum_ff_n.visited)
                tot_days = 0
                for el in remaining:
                    tot_days += damaged_dict[el]

                eligible_to_add = list(deepcopy(remaining))
                decoy_dd = deepcopy(damaged_dict)
                for visited_links in minimum_ff_n.visited:
                    del decoy_dd[visited_links]

                after_ = minimum_ff_n.tstt_after
                test_net = create_network(net_after.netfile, net_after.tripfile,
                    mc_weights=net_after.mc_weights, demand_mult=net_after.demand_mult)
                test_net.art_links = art_link_dict
                test_net.maxruntime = net_after.maxruntime
                path = minimum_ff_n.path.copy()
                new_bb = {}

                for i in range(len(remaining)):
                    new_tstts = []
                    for link in eligible_to_add:
                        added = [link]
                        not_fixed = set(eligible_to_add).difference(set(added))
                        test_net.not_fixed = set(not_fixed)

                        after_fix_tstt, __, __ = solve_UE(net=test_net)
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
                    __, __, ord = orderlists(orderedb_benefits, ordered_days,
                                             rem_keys=sorted_d)

                    link_to_add = ord[0][0]
                    path.append(link_to_add)
                    min_index = eligible_to_add.index(link_to_add)
                    after_ = new_tstts[min_index]
                    eligible_to_add.remove(link_to_add)
                    decoy_dd = deepcopy(decoy_dd)
                    del decoy_dd[link_to_add]

                net = deepcopy(net_after)
                bound, __, __, __ = eval_sequence(net, path, after_eq_tstt,
                    before_eq_tstt)

                if bound < bfs.cost:
                    bfs.cost = bound
                    bfs.path = path

                # Backwards pass
                remaining = set(damaged_dict.keys()).difference(minimum_bf_n.visited)
                tot_days = 0
                for el in remaining:
                    tot_days += damaged_dict[el]

                eligible_to_add = list(deepcopy(remaining))
                after_ = after_eq_tstt
                test_net = create_network(net_after.netfile, net_after.tripfile,
                    mc_weights=net_after.mc_weights, demand_mult=net_after.demand_mult)
                test_net.art_links = art_link_dict
                test_net.maxruntime = net_after.maxruntime

                decoy_dd = deepcopy(damaged_dict)
                for visited_links in minimum_bf_n.visited:
                    del decoy_dd[
                        visited_links]

                path = []
                new_bb = {}

                for i in range(len(remaining)):
                    new_tstts = []
                    for link in eligible_to_add:
                        added = [link]
                        not_fixed = set(eligible_to_add).difference(set(added))
                        test_net.not_fixed = set(not_fixed)

                        after_fix_tstt, __, __ = solve_UE(net=test_net)
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

                    __, __, ord = orderlists(orderedb_benefits, ordered_days,
                                             rem_keys=sorted_d)

                    link_to_add = ord[0][0]
                    path.append(link_to_add)
                    min_index = eligible_to_add.index(link_to_add)
                    after_ = new_tstts[min_index]
                    eligible_to_add.remove(link_to_add)
                    decoy_dd = deepcopy(decoy_dd)
                    del decoy_dd[link_to_add]

                path.extend(minimum_bf_n.path[::-1])
                net = deepcopy(net_after)
                bound, __, __, __ = eval_sequence(net,path,after_eq_tstt,before_eq_tstt)

                if bound < bfs.cost:
                    bfs.cost = bound
                    bfs.path = path

            if graphing:
                bs_time_list.append(time.time() - search_timer)
                bs_OBJ_list.append(deepcopy(bfs.cost))

            seq_length = len(damaged_links)
            threshold_start = seq_length+ int(seq_length**2/int(math.log(seq_length,2)))

            if len_f >= threshold_start:
                f_activated = True
            if len_b >= threshold_start:
                b_activated = True

            if f_activated or b_activated:
                print('Iteration ' + str(iter_count) + ': len_f is ' + str(len_f)
                      + ' and len_b is ' + str(len_b) + ', about to purge')

                if beam_search:
                    (open_list_b, open_list_f, num_purged, f_activated, b_activated,
                     closed_list_f, closed_list_b) = purge(damaged_links, open_list_b,
                        open_list_f, beam_k, num_purged, closed_list_f, closed_list_b,
                        f_activated, b_activated, iter_count, beta=beta)

    # Check in case something weird happened
    if len(bfs.path) != len(damaged_links):
        print('Best path found is not full length')

    if graphing:
        bs_time_list.append(time.time() - search_timer)
        bs_OBJ_list.append(deepcopy(bfs.cost))

    print('Beam search objective value: ',bfs.cost)
    return (bfs.path, bfs.cost, num_tap_solved, tot_child, uncommon_number,
            common_number, num_purged, wb, bb, wb_update, bb_update, memory)

