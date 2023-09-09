from matplotlib import cm
import pdb
import numpy as np
import os
from sequence_utils import *
import matplotlib._color_data as mcd
from scipy import stats


def graph_current(tstt_state, days_state, before_eq_tstt, after_eq_tstt, path,
                  plt_path, algo, together, place, color_dict):
    N = sum(days_state)

    cur_tstt = after_eq_tstt

    tstt_g = []
    tot_area = 0
    days = 0

    for i in range(len(days_state)):
        days += days_state[i]
        if i == 0:
            tot_area += days_state[i] * (after_eq_tstt - before_eq_tstt)
            tstt_g.append(after_eq_tstt)
        else:
            tot_area += days_state[i] * (tstt_state[i - 1] - before_eq_tstt)
            tstt_g.append(tstt_state[i - 1])

    tstt_g.append(before_eq_tstt)
    y = tstt_g
    x = np.zeros(len(y))

    for j in range(len(y)):
        x[j] = sum(days_state[:j])

    if together:
        plt.subplot(place)
    else:
        plt.figure(figsize=(16, 8))

    plt.fill_between(x, y, before_eq_tstt, step="post", color='#087efe', alpha=0.2,
                     label='TOTAL AREA:' + '{0:1.3e}'.format(tot_area))
    # plt.step(x, y, label='tstt', where="post")
    plt.xlabel('DAYS')
    plt.ylabel('TSTT (in minutes)')

    plt.ylim((before_eq_tstt, (after_eq_tstt + after_eq_tstt * 0.07)))

    for i in range(len(tstt_state)):
        color = color_dict[path[i]]

        start = sum(days_state[:i])

        bbox_props = dict(boxstyle="round,pad=0.3",
                          fc="w", ec=color, alpha=0.8, lw=1)

        ha = 'center'
        # if i >= len(tstt_state) - 3:
        #     ha = 'right'

        t = plt.text(start + N * 0.01, y[i] + before_eq_tstt * 0.02,
            "link: " + path[i] + '\n' + "time: " + str(round(days_state[i], 2)), ha=ha,
            size=8, bbox=bbox_props)

        bb = t.get_bbox_patch()

        # start + days_state[i]/2
        # plt.annotate("link: " + path[i], (start, y[i]), textcoords='offset points',
        #              xytext=(5,15), ha='left', size='smaller')
        # plt.annotate("time: " + str(round(days_state[i],2)), (start, y[i]),
        # textcoords='offset points', xytext=(5,5), ha='left', size='smaller')
        # #, arrowprops=dict(width= days_state[i]))

        # plt.annotate("", xy=(start + days_state[i], y[i]), xytext=(start, y[i]),
        #              textcoords='offset points', ha='center',
        #              arrowprops=dict(arrowstyle='<->', connectionstyle="arc3"))
        # plt.arrow(0.85, 0.5, dx = -0.70, dy = 0, head_width=0.05, head_length=0.03,
        #           linewidth=4, color='g', length_includes_head=True)

        plt.annotate(s='', xy=(start + days_state[i], y[i]), xytext=(
            start, y[i]), arrowprops=dict(arrowstyle='<->', color=color))

        # plt.annotate(s='', xy=(start + days_state[i], y[i]),
        #              xytext=(start + days_state[i], y[i + 1]),
        #              arrowprops=dict(arrowstyle='<->', color='indigo'))

        # plt.annotate("" + str(round(y[i] - y[i+1], 2)), xy=(start +
        # days_state[i], (y[i] - y[i+1])/2), xytext=(10,10), textcoords='offset
        # points', ha='left') #, arrowprops=dict(width= days_state[i]))

        # animation example:
        # import numpy as np
        # import matplotlib.pyplot as plt
        # import matplotlib.animation as animation

        # fig, ax = plt.subplots()
        # ax.axis([-2,2,-2,2])

        # arrowprops=dict(arrowstyle='<-', color='blue', linewidth=10, mutation_scale=150)
        # an = ax.annotate('Blah', xy=(1, 1), xytext=(-1.5, -1.5), xycoords='data',
        #                  textcoords='data', arrowprops=arrowprops)

        # colors=["crimson", "limegreen", "gold", "indigo"]
        # def update(i):
        #     c = colors[i%len(colors)]
        #     an.arrow_patch.set_color(c)

        # ani = animation.FuncAnimation(fig, update, 10, interval=1000, repeat=True)
        # plt.show()

        # plt.text(0, 0.1, r'$\delta$',
        #  {'color': 'black', 'fontsize': 24, 'ha': 'center', 'va': 'center',
        #   'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
        # xticks(np.arange(5), ('Tom', 'Dick', 'Harry', 'Sally', 'Sue'))

        # using tex in labels
        # Use tex in labels
        # plt.xticks((-1, 0, 1), ('$-1$', r'$\pm 0$', '$+1$'), color='k', size=20)

        # Left Y-axis labels, combine math mode and text mode
        # plt.ylabel(r'\bf{phase field} $\phi$', {'color': 'C0', 'fontsize': 20})
        # plt.yticks((0, 0.5, 1), (r'\bf{0}', r'\bf{.5}', r'\bf{1}'), color='k', size=20)

    ###
    plt.title(algo, fontsize=16)
    plt.legend(loc='upper right', fontsize=14)

    if not together:
        save_fig(plt_path, algo)
        plt.clf()


def mean_std_lists(metric_values, sample_size):
    t_critical = stats.t.ppf(q=0.9, df=sample_size)

    mean = np.mean(metric_values)
    stdev = np.std(metric_values)  # Get the sample standard deviation
    sigma = stdev / np.sqrt(sample_size)  # Standard deviation estimate
    error = t_critical * sigma

    return mean, error, stdev


def prep_dictionaries(method_dict, key_list):
    for i in range(len(key_list)):
        method_dict[key_list[i]] = []

def get_common_numbers():

    netdirs = []
    FOLDER = 'TransportationNetworks'
    for net_name in ['SiouxFalls', 'ChicagoSketch']:
        NETWORK = os.path.join(FOLDER, net_name)
        JSONFILE = os.path.join(NETWORK, net_name.lower() + '.geojson')
        NETFILE = os.path.join(NETWORK, net_name + "_net.tntp")
        TRIPFILE = os.path.join(NETWORK, net_name + "_trips.tntp")

        SAVED_FOLDER_NAME = "saved"
        PROJECT_ROOT_DIR = "."
        SAVED_DIR = os.path.join(PROJECT_ROOT_DIR, SAVED_FOLDER_NAME)
        os.makedirs(SAVED_DIR, exist_ok=True)

        NETWORK_DIR = os.path.join(SAVED_DIR, NETWORK)
        netdirs.append(NETWORK_DIR)

    plt.figure(figsize=(16, 8))
    i = 0
    reps=4
    for netdir in netdirs:

        brokens = ['8', '16']
        for broken in brokens:
            figure_grid = 220
            i += 1
            file_path = os.path.join(netdir, broken)
            plt.subplot(figure_grid + i)

            bs_param_2 = str(int(int(broken) / 2.0))

            filenames = ['r_algo_solution_k2']

            heuristic32 = {}

            dict_list = [heuristic32]
            key_list = ['_common', '_uncommon']

            for method_dict in dict_list:
                prep_dictionaries(method_dict, key_list)

            for rep in range(reps + 1):
                for method_dict in dict_list:
                    for key in key_list:
                        method_dict[key].append(load(os.path.join(file_path, str(rep))
                            + '/' + filenames[dict_list.index(method_dict)] + key))

            sample_size = reps + 1

            common_means = []
            uncommon_means = []

            for method_dict in dict_list:
                cn = np.array(method_dict['_common'])
                un = np.array(method_dict['_uncommon'])

                common_means.append(cn)
                uncommon_means.append(un)

            cm = np.mean(common_means)
            um = np.mean(uncommon_means)


            tot = cm + um
            cm_perc = cm/tot
            um_perc = um/tot

            barWidth = 0.1

            plt.rcParams["font.size"] = 8
            # hatch='///', hatch='\\\\\\', hatch='xxx'
            plt.bar(0, cm_perc, width=barWidth, edgecolor='#087efe', color='#087efe',
                ecolor='#c6ccce', alpha=0.8, capsize=5, label='Case 1', align='center')
            plt.bar(0.1, um_perc, width=barWidth, edgecolor='#b7fe00', color='#b7fe00',
                ecolor='#c6ccce', alpha=0.8, capsize=5, label='Case 2', align='center')

            plt.annotate('{:1.3f}'.format(cm_perc) + '%', (0, 0), textcoords=
                'offset points', xytext=(0, 20), ha='center', va='bottom', rotation=70,
                size=10)
            plt.annotate('{:1.3f}'.format(um_perc) + '%', (0 + barWidth, 0), textcoords=
                'offset points', xytext=(0, 20), ha='center', va='bottom', rotation=70,
                size=10)

            plt.ylabel('Percentage', size=10)
            if netdir.find('SiouxFalls') >= 0:
                plt.xlabel('SiouxFalls - ' + broken, fontsize=12)
            else:
                plt.xlabel('Chicago-Sketch - ' + broken, fontsize=12)
            plt.tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False)
            plt.legend(fontsize=8)
    save_fig(FOLDER, 'common_uncommon', tight_layout=False)



def result_table(reps, file_path, broken, ks, biginstance=True):
    bs_param_2 = str(int(int(broken) / 2.0))
    if int(broken) >= 32:
        biginstance = True


    fullinc = True
    try:
        load(os.path.join(file_path, str(0)) + '/' + 'algo_solution' + '_obj')
    except:
        fullinc = False

    filenames = []
    for k in ks[::-1]:
        if not biginstance:
            filenames.extend(['beamsearch_solution_k' + str(k),
                              'r_algo_solution_k' + str(k)])
        else:
            filenames.extend(['r_algo_solution_k' + str(k)])

    filenames.extend(['greedy_solution','layzgreedy_solution','importance_factor_bound'])

    if fullinc:
        filenames.insert(0,'algo_solution')

    heuristic2 = {}
    r_heuristic2 = {}
    greedy = {}
    lgreedy = {}
    importance_factor = {}
    algo = {}


    if biginstance:
        dict_list = [r_heuristic2, greedy, lgreedy, importance_factor]

    else:
        dict_list = [heuristic2, r_heuristic2, greedy, lgreedy, importance_factor]

    if fullinc:
        dict_list.insert(0,algo)

    key_list = ['_obj', '_num_tap', '_elapsed']
    key_list2 = ['_common', '_uncommon']


    for method_dict in dict_list:
        prep_dictionaries(method_dict, key_list)
        prep_dictionaries(method_dict, key_list2)

    for rep in range(reps + 1):
        for method_dict in dict_list:
            for key in key_list + key_list2:
                try:
                    method_dict[key].append(load(os.path.join(file_path, str(rep))
                        + '/' + filenames[dict_list.index(method_dict)] + key))
                except:
                    print(method_dict)
                    if key == '_num_tap':
                        method_dict[key].append((int(broken) + 1) * int(broken) / 2)


    sample_size = reps + 1

    obj_means = []
    tap_means = []
    elapsed_means = []

    obj_stds = []
    tap_stds =[]
    elapsed_stds =[]


    obj_err = []
    tap_err = []
    elapsed_err = []

    obj_range_above = []
    obj_range_below = []
    tap_range_above = []
    tap_range_below = []
    elapsed_range_above = []
    elapsed_range_below = []

    percgapmax = []
    numtapmax = []
    elapsedmax = []

    if int(broken) <= 6:
        optimal = []
        for rep in range(reps + 1):
             optimal.append(load(os.path.join(file_path, str(rep)) + '/' + 'min_seq_obj'))

    else:
        # optimal = np.minimum(np.array(r_heuristic4['_obj']),
        #     np.minimum(np.array(heuristic4['_obj']),
        #     np.minimum(np.array(r_heuristic8['_obj']),
        #     np.minimum(np.array(r_heuristic2['_obj']), np.minimum(
        #     np.minimum(np.array(heuristic2['_obj']),
        #     np.array(heuristic8['_obj'])), np.array(greedy['_obj']))))))
        if biginstance:
            optimal = np.minimum(np.array(r_heuristic2['_obj']),
                                 np.array(greedy['_obj']))

        else:
            optimal = np.minimum(np.array(r_heuristic2['_obj']),
                np.minimum(np.array(heuristic2['_obj']), np.array(greedy['_obj'])))
    if fullinc:
        optimal = np.minimum(optimal, np.array(algo['_obj']))

    data = np.zeros((len(dict_list), 6))
    r = 0

    for method_dict in dict_list:
        objs = np.array(method_dict['_obj'])
        objs = ((objs - optimal) / optimal) * 100
        objs = np.maximum(0, objs)

        mean, error, stdev = mean_std_lists(objs, sample_size)

        obj_range_above.append(max(objs))
        obj_range_below.append(min(objs))
        obj_means.append(mean)
        obj_stds.append(stdev)
        obj_err.append(error)
        data[r, 0] = mean
        data[r, 1] = error


        taps = method_dict['_num_tap']
        mean, error,stdev = mean_std_lists(taps, sample_size)
        tap_means.append(mean)
        tap_err.append(error)
        tap_range_above.append(max(taps))
        tap_range_below.append(min(taps))
        tap_stds.append(stdev)
        data[r, 2] = mean
        data[r, 3] = error

        elapsed_values = method_dict['_elapsed']
        mean, error, stdev= mean_std_lists(elapsed_values, sample_size)
        elapsed_means.append(mean)
        elapsed_err.append(error)
        elapsed_stds.append(stdev)
        elapsed_range_above.append(max(elapsed_values))
        elapsed_range_below.append(min(elapsed_values))
        data[r, 4] = mean
        data[r, 5] = error

        r += 1


    print(obj_means, obj_stds, tap_means, tap_stds, elapsed_means, elapsed_stds)
    # pdb.set_trace()

    obj_means_scaled = obj_means / max(obj_means)

    obj_range_above = obj_range_above / max(obj_means)
    obj_range_below = obj_range_below / max(obj_means)

    tap_range_above = np.array(tap_range_above)
    tap_range_below  = np.array(tap_range_below)
    tap_range_above = tap_range_above/max(tap_means)
    tap_range_below = tap_range_below/max(tap_means)

    elapsed_range_above = np.array(elapsed_range_above)
    elapsed_range_below  = np.array(elapsed_range_below)
    elapsed_range_above = elapsed_range_above/max(elapsed_means)
    elapsed_range_below = elapsed_range_below/max(elapsed_means)

    obj_err_scaled = np.nan_to_num(np.array(obj_err)/np.array(obj_means)
                                   *np.array(obj_means_scaled))
    # obj_err_scaled = obj_err * obj_means_scaled /obj_means

    tap_means_scaled = tap_means / max(tap_means)
    tap_err_scaled = np.nan_to_num(np.array(tap_err)/np.array(tap_means)
                                   *np.array(tap_means_scaled))
    # tap_err_scaled = tap_err / max(tap_means)

    elapsed_means_scaled = elapsed_means / max(elapsed_means)
    elapsed_err_scaled = np.nan_to_num(np.array(elapsed_err)/np.array(elapsed_means)
                                       *np.array(elapsed_means_scaled))
    # elapsed_err_scaled = elapsed_err / max(elapsed_means)


    maxdiff = np.max((np.array(greedy['_obj']) - optimal)/optimal*100)
    argmaxdiff = np.argmax((np.array(greedy['_obj']) - optimal)/optimal*100)


    percgap = maxdiff #/ optimal[argmaxdiff] * 100

    # pdb.set_trace()
    # which = np.argmin([np.array(heuristic2['_obj'])[argmaxdiff],
    #    np.array(heuristic32['_obj'])[argmaxdiff], np.array(algo['_obj'])[argmaxdiff]])
    # percgapmax.append(0)


    if fullinc:
        percgapmax.append(np.max((np.array(algo['_obj'])[argmaxdiff]
                                  - optimal[argmaxdiff])/optimal[argmaxdiff]*100))
        numtapmax.append(np.array(algo['_num_tap'])[argmaxdiff])
        elapsedmax.append(np.array(algo['_elapsed'])[argmaxdiff])


    if not biginstance:
        percgapmax.append(np.max((np.array(heuristic2['_obj'])[argmaxdiff]
                                  - optimal[argmaxdiff]) / optimal[argmaxdiff] * 100))
        numtapmax.append(np.array(heuristic2['_num_tap'])[argmaxdiff])
        elapsedmax.append(np.array(heuristic2['_elapsed'])[argmaxdiff])

    percgapmax.append(np.max((np.array(r_heuristic2['_obj'])[argmaxdiff]
                              - optimal[argmaxdiff]) / optimal[argmaxdiff] * 100))
    numtapmax.append(np.array(r_heuristic2['_num_tap'])[argmaxdiff])
    elapsedmax.append(np.array(r_heuristic2['_elapsed'])[argmaxdiff])

    percgapmax.append(percgap)
    numtapmax.append(np.array(greedy['_num_tap'])[argmaxdiff])
    elapsedmax.append(np.array(greedy['_elapsed'])[argmaxdiff])

    print(argmaxdiff)
    percgapmax_scaled = percgapmax / max(percgapmax)
    numtapmax_scaled = numtapmax / max(numtapmax)
    elapsedmax_scaled = elapsedmax / max(elapsedmax)

    # plt.figure(figsize=(10, 6))
    #
    # barWidth = 0.2
    # r_obj = np.arange(len(percgapmax))
    # r_tap = [x + barWidth for x in r_obj]
    # r_elapsed = [x + 2 * barWidth for x in r_obj]
    #
    # plt.rcParams["font.size"] = 8
    # # hatch='///', hatch='\\\\\\', hatch='xxx'
    #
    # plt.bar(r_obj, percgapmax_scaled, width=barWidth, edgecolor='#087efe',
    #     color='#087efe', ecolor='#c6ccce', alpha=0.8, capsize=5, label='Rel Gap',
    #     hatch='///')
    # plt.bar(r_tap, numtapmax_scaled, width=barWidth, edgecolor='#00b33c',
    #     color='#00b33c', ecolor='#c6ccce', alpha=0.8, capsize=5, label='Tap Solved',
    #     hatch='\\\\\\')
    # plt.bar(r_elapsed, elapsedmax_scaled, width=barWidth, edgecolor='#FF4500',
    #     color='#FF4500', ecolor='#c6ccce', alpha=0.8, capsize=5,
    #     label='Time Elapsed (s)', hatch='xxx')
    #
    # # tap_means_scaled[i]/2
    # for i in range(len(percgapmax)):
    #     plt.annotate('{0:1.1f}'.format(percgapmax[i]) + '%', (i, 0),
    #         textcoords='offset points', xytext=(0, 20), ha='center', va='bottom',
    #         rotation=70, size=10)
    #     plt.annotate('{0:1.1f}'.format(numtapmax[i]), (i + barWidth, 0),
    #         textcoords='offset points', xytext=(0, 20), ha='center', va='bottom',
    #         rotation=70, size=10)
    #     plt.annotate('{0:1.1f}'.format(elapsedmax[i]), (i + 2 * barWidth, 0),
    #         textcoords='offset points', xytext=(0, 20), ha='center', va='bottom',
    #         rotation=70, size=10)
    #
    # plt.ylabel('Normalized Metric Value', size=10)
    #
    # if fullinc:
    #     xticks_str = ['MF', 'M' + bs_param_2, 'M2', 'GM']
    # else:
    #     xticks_str = ['M2', 'RM2', 'GM']
    #
    #
    #
    # plt.xticks([(r + barWidth) for r in range(len(elapsedmax))],
    #                xticks_str, size=10)
    # # plt.title('Performance Comparison - ' + broken, fontsize=7)
    #
    # txt = "# Broken Links: " + \
    #       str(broken) + ".\n worst within: " + \
    #       str(reps + 1) + ' different instances.'
    #
    # # plt.figtext(0.5, 0.01, txt, wrap=True,
    # #             ha='center', va="bottom", fontsize=7)
    # plt.legend(fontsize=8)
    # save_fig(file_path, 'worst gap possible' +
    #          'w_bridge_' + str(broken), tight_layout=False)
    #
    # plt.close()

    plt.figure(figsize=(10, 6))

    barWidth = 0.2
    r_obj = np.arange(len(obj_means))
    r_tap = [x + barWidth for x in r_obj]
    r_elapsed = [x + 2 * barWidth for x in r_obj]

    plt.rcParams["font.size"] = 8
    # hatch='///', hatch='\\\\\\', hatch='xxx'
    if len(objs) <= 2:
        plt.bar(r_obj, obj_means_scaled, width=barWidth, edgecolor='#087efe',
                color='#087efe', ecolor='#c6ccce', alpha=0.8, capsize=5,
                label='Avg Rel Gap', hatch='///')
        plt.bar(r_tap, tap_means_scaled, width=barWidth, edgecolor='#b7fe00',
                color='#b7fe00', ecolor='#c6ccce', alpha=0.8, capsize=5,
                label='Avg Tap Solved', hatch='\\\\\\')
        plt.bar(r_elapsed, elapsed_means_scaled, width=barWidth, edgecolor='#ff9700',
                color='#FF4500', ecolor='#c6ccce', alpha=0.8, capsize=5,
                label='Avg Time Elapsed (s)', hatch='xxx')

    else:
        lower_err = np.array(obj_means_scaled) - np.array(obj_err_scaled)
        lower_err = np.where(lower_err < 0, obj_means_scaled, obj_err_scaled)

        plt.bar(r_obj, obj_means_scaled, width=barWidth, edgecolor='#087efe',
                color='#087efe', ecolor='#c6ccce', alpha=0.8, yerr=(obj_range_below,
                obj_range_above), capsize=5, label='Avg Rel Gap')
        plt.bar(r_tap, tap_means_scaled, width=barWidth, edgecolor='#00b33c',
                color='#00b33c', ecolor='#c6ccce', alpha=0.8, yerr=(tap_range_below,
                tap_range_above), capsize=5, label='Avg Tap Solved')
        plt.bar(r_elapsed, elapsed_means_scaled, width=barWidth, edgecolor='#FF4500',
                color='#FF4500', ecolor='#c6ccce', alpha=0.8, yerr=(elapsed_range_below,
                elapsed_range_above), capsize=5, label='Avg Time Elapsed (s)')
        # plt.bar(r_obj, obj_means_scaled, width=barWidth, edgecolor='#087efe',
        #         color='#087efe', ecolor='#c6ccce', alpha=0.8, capsize=5,
        #         label='Avg Rel Gap')
        # plt.bar(r_tap, tap_means_scaled, width=barWidth, edgecolor='#00b33c',
        #         color='#00b33c', ecolor='#c6ccce', alpha=0.8, capsize=5,
        #         label='Avg Tap Solved')
        # plt.bar(r_elapsed, elapsed_means_scaled, width=barWidth, edgecolor='#FF4500',
        #         color='#FF4500', ecolor='#c6ccce', alpha=0.8, capsize=5,
        #         label='Avg Time Elapsed (s)')


    # tap_means_scaled[i]/2
    for i in range(len(dict_list)):
        plt.annotate('{0:1.1f}'.format(obj_means[i]) + '%', (i, 0),
                     textcoords='offset points', xytext=(0, 20), ha='center',
                     va='bottom', rotation=70, size=10)
        plt.annotate('{0:1.1f}'.format(tap_means[i]), (i + barWidth, 0),
                     textcoords='offset points', xytext=(0, 20), ha='center',
                     va='bottom', rotation=70, size=10)
        plt.annotate('{0:1.1f}'.format(elapsed_means[i]), (i + 2 * barWidth, 0),
                     textcoords='offset points', xytext=(0, 20), ha='center',
                     va='bottom', rotation=70, size=10)

    plt.ylabel('Normalized Metric Value', size=10)
    if fullinc:
        xticks_str = ['MF', 'M'+bs_param_2, 'RM'+bs_param_2, 'M2', 'RM2', 'GM', 'IF']
    else:
        if biginstance:
            xticks_str = ['RM', 'GM', 'LGM', 'IF']
        else:
            xticks_str = ['M', 'RM', 'GM', 'LGM', 'IF']

    plt.xticks([(r + barWidth) for r in range(len(obj_means))],
               xticks_str, size=10)


    plt.title(str(broken) + ' damaged links' , fontsize=10)
    if broken != 10:
        txt = "# Broken Links: " + \
              str(broken) + ".\n Averaged over: " + \
              str(reps + 1) + ' different instances.'
    else:
        txt = "# Broken Links: " + str(broken) + ".\n Averaged over: " + str(
            reps + 1) + ' different instances.\n Heuristic solution was taken as best \
            - since not possible to solve to optimality'
    # plt.figtext(0.5, 0.01, '', wrap=True,
    #             ha='center', va="bottom", fontsize=9)
    # plt.ylim(0,)

    print('objs')
    print(obj_range_below)
    print(obj_range_above)
    print('\n taps')
    print(tap_range_below)
    print(tap_range_above)
    print('\n elapsed')
    print(elapsed_range_below)
    print(elapsed_range_above)

    plt.legend(fontsize=8)
    save_fig(file_path, 'performance_graph_' +
             'w_bridge_' + str(broken), tight_layout=False)

    # columns = ('Avg Rel Gap', 'Delta', 'Avg Tap Solved',
    #            'Delta', 'Avg Elapsed(s)', 'Delta')
    # rows = ['M2', 'RM2', 'M' + bs_param_2, 'RM' + bs_param_2, 'GM', 'IF']
    #
    # plt.close()
    #
    # fig, ax = plt.subplots()
    #
    # cell_text = []
    # for row in range(len(data)):
    #     cell_text.append(['%1.2f' % x + '%' if (i == 0 or i == 1)
    #                       else '%1.2f' % x for i, x in enumerate(data[row])])
    #
    # ax.axis('off')
    # ax.axis('tight')
    #
    # # # Add a table at the bottom of the axes
    # the_table = ax.table(cellText=cell_text,
    #                      rowLabels=rows,
    #                      # rowColours=colors,
    #                      colLabels=columns,
    #                      loc='center')
    #
    # if broken != 10:
    #     txt = "# Broken Links: " + \
    #           str(broken) + ". \n Averaged over: " + \
    #           str(reps + 1) + ' different instances.'
    # else:
    #     txt = "# Broken Links: " + str(broken) + ". \n Averaged over: " + str(
    #         reps + 1) + ' different instances.\n Heuristic solution was taken as \
    #         best - since not possible to solve to optimality'
    #
    # plt.figtext(0.5, 0.01, txt, wrap=True,
    #             horizontalalignment='center', fontsize=7)
    #
    # save_fig(file_path, 'performance_table_' + 'w_bridge_' + str(broken))


def result_sequence_graphs(rep, save_dir, alt_dir=None, mc_weights=1):
    if alt_dir!=None:
        path_pre = alt_dir + '/'
    else:
        path_pre = os.path.join(save_dir, str(rep)) + '/'
    print(path_pre)

    net_after = load(path_pre + 'net_after_real')
    net_after.damaged_dict = load(path_pre + 'damaged_dict')

    if len(net_after.damaged_dict) > 8:
        opt = False
    else:
        opt = True

    if opt:
        opt_soln = load(path_pre + 'min_seq_path')

    r_algo_path = load(path_pre + 'r_algo_solution_k2_path')
    sa_path = load(path_pre + 'sim_anneal_solution_path')

    greedy_soln = load(path_pre + 'greedy_solution_path')
    importance_soln = load(path_pre + 'importance_factor_bound_path')
    after_eq_tstt = load(path_pre + 'net_after_real_tstt')
    before_eq_tstt = load(path_pre + 'net_before_real_tstt')

    # GRAPH THE RESULTS
    if opt:
        paths = [opt_soln, r_algo_path, sa_path, greedy_soln, importance_soln]
        names = ['Optimal', 'BeamSearch', 'SimAnneal',
                 'GreedyMethod']
        places = [221, 222, 223, 224]
    else:
        paths = [r_algo_path, sa_path, greedy_soln, importance_soln]

        names = ['BeamSearch', 'SimAnneal', 'GreedyMethod', 'ImportanceFactor']

        places = [221, 222, 223, 224]

    colors = plt.cm.ocean(np.linspace(0, 0.7, len(r_algo_path)))
    color_dict = {}
    color_i = 0
    for alink in r_algo_path:
        color_dict[alink] = colors[color_i]
        color_i += 1

    # colors = ['y', 'r', 'b', 'g']
    for path, name, place in zip(paths, names, places):
        tstt_list, days_list = get_marginal_tstts(net_after, path, after_eq_tstt,
            before_eq_tstt, net_after.damaged_dict, mc_weights=mc_weights)
        graph_current(tstt_list, days_list, before_eq_tstt,
                      after_eq_tstt, path, path_pre, name, False, place, color_dict)

    plt.close()
    # plt.figure(figsize=(16, 8))

    for path, name, place in zip(paths, names, places):
        tstt_list, days_list = get_marginal_tstts(net_after, path, after_eq_tstt,
            before_eq_tstt, net_after.damaged_dict, mc_weights=mc_weights)
        graph_current(tstt_list, days_list, before_eq_tstt,
                      after_eq_tstt, path, path_pre, name, True, place, color_dict)

    save_fig(path_pre, 'Comparison')


def multiClass_result_sequence_graphs(rep, save_dir, alt_dir=None, mc_weights=1):
    print('alt_dir is: ',alt_dir)
    if alt_dir is not None:
        path_pre = alt_dir + '/'
    else:
        path_pre = os.path.join(save_dir, str(rep)) + '/'
    path_pre_alt = path_pre[:-2] + '/'
    print(path_pre)
    print(path_pre_alt)

    net_after = load(path_pre + 'net_after_real')
    net_after.damaged_dict = load(path_pre + 'damaged_dict')

    if len(net_after.damaged_dict) > 8:
        opt = False
    else:
        opt = True

    if opt:
        soln_path = load(path_pre + 'min_seq_path')
        soln_path_alt = load(path_pre_alt + 'min_seq_path')
    else:
        soln_path = load(path_pre + 'sim_anneal_solution_path')
        soln_path_alt = load(path_pre_alt + 'sim_anneal_solution_path')

    after_eq_tstt = load(path_pre + 'net_after_real_tstt')
    before_eq_tstt = load(path_pre + 'net_before_real_tstt')

    after_eq_tstt_mcunw = load(path_pre_alt + 'net_after_real_tstt_mcunw')
    before_eq_tstt_mcunw = load(path_pre_alt + 'net_before_real_tstt_mcunw')

    # GRAPH THE RESULTS
    paths = [soln_path, soln_path, soln_path, soln_path_alt, soln_path_alt, soln_path_alt]
    classes = [0,1,2,0,1,2]
    names = ['Overall-unweighted', 'Class 1-unweighted', 'Class 2-unweighted',
             'Overall-weighted', 'Class 1-weighted', 'Class 2-weighted']
    places = [321, 323, 325, 322, 324, 326]

    colors = plt.cm.ocean(np.linspace(0, 0.7, len(soln_path)))
    color_dict = {}
    color_i = 0
    for alink in soln_path:
        color_dict[alink] = colors[color_i]
        color_i += 1

    # colors = ['y', 'r', 'b', 'g']
    for path, cl, name, place in zip(paths, classes, names, places):
        tstt_list, days_list = get_marginal_tstts(net_after, path, after_eq_tstt,
            before_eq_tstt, net_after.damaged_dict, multiclass=True)
        sub_list = []
        for el in tstt_list:
            sub_list.append(el[cl])
        graph_current(sub_list, days_list, before_eq_tstt_mcunw[cl],
            after_eq_tstt_mcunw[cl], path, path_pre, name, False, place, color_dict)

    plt.close()
    # plt.figure(figsize=(16, 8))

    for path, cl, name, place in zip(paths, classes, names, places):
        tstt_list, days_list = get_marginal_tstts(net_after, path, after_eq_tstt,
            before_eq_tstt, net_after.damaged_dict, multiclass=True)
        sub_list = []
        for el in tstt_list:
            sub_list.append(el[cl])
        graph_current(sub_list, days_list, before_eq_tstt_mcunw[cl],
            after_eq_tstt_mcunw[cl], path, path_pre, name, True, place, color_dict)

    save_fig(path_pre, 'Comparison')


def get_folders(path):
    folders = os.listdir(path)

    try:
        folders.remove('.DS_Store')
    except:
        pass

    try:
        folders.remove('figures')
    except:
        pass

    return folders


def get_tables(directory):
    SCENARIO_DIR = directory

    try:
        brokens = ['16']
    except:
        return
    ks = [2]
    print(brokens)
    for broken in brokens:
        if broken == 'prev':
            continue

        try:
            B_DIR = os.path.join(SCENARIO_DIR, broken)

        except:
            return

        repetitions = get_folders(B_DIR)

        if len(repetitions) == 0:
            return

        reps = [int(i) for i in range(len(repetitions))]
        max_reps = max(reps)
        result_table(max_reps, B_DIR, broken, ks)
        print(reps)


def get_sequence_graphs(directory, broken, alt_dir=None, multiClass=False, mc_weights=1):
    if alt_dir != None:
        if multiClass == False:
            result_sequence_graphs(1, directory, alt_dir=alt_dir, mc_weights=mc_weights)
        else:
            multiClass_result_sequence_graphs(1, directory, alt_dir=alt_dir,
                                              mc_weights=mc_weights)
    else:
        SCENARIO_DIR = directory
        #try:
        #    broken_bridges = get_folders(SCENARIO_DIR)
        #except:
        #    return

        #if len(broken_bridges) == 0:
        #    return

        if type(broken)==str:
            broken_bridges = [broken]
        else:
            broken_bridges = broken

        for broken in broken_bridges:

            ULT_SCENARIO_DIR = os.path.join(SCENARIO_DIR, broken)
            repetitions = get_folders(ULT_SCENARIO_DIR)

            if len(repetitions) == 0:
                return

            reps = []
            for i in repetitions:
                try:
                    reps.append(int(i))
                except:
                    pass

            for rep in reps:
                if multiClass == False:
                    result_sequence_graphs(rep, ULT_SCENARIO_DIR, mc_weights=mc_weights)
                else:
                    multiClass_result_sequence_graphs(rep, ULT_SCENARIO_DIR,
                                                      mc_weights=mc_weights)
