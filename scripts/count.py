import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

FONTSIZE = 20
LINEWIDTH = 4.

def plot_data(data, xaxis='Epoch', value="AverageEpRet",
              condition="Condition1", smooth=1, paper=False,
              hidelegend=False, title=None, savedir=None,
              clear_xticks=False, logy=False, cstrline=False, env_name=None, **kwargs):
    # special handling for plotting a horizontal line
    # fig, ax = plt.subplots()
    splits = value.split(',')
    value = splits[0]
    #if len(splits) > 1:
    #    y_horiz = float(splits[1])
    #else:
    #    y_horiz = None
    y_horiz, ymin, ymax = None, None, None
    if len(splits) > 1:
        for split in splits[1:]:
            if split[0]=='h':
                y_horiz = float(split[1:])
            elif split[0]=='l':
                ymin = float(split[1:])
            elif split[0]=='u':
                ymax = float(split[1:])

    if isinstance(data, list):
        # Seive data so only data with value column is present
        data = [x for x in data if value in x.columns]

    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
            datum[value] = smoothed_x

    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    df_times = []
    tag_list = ['AverageCostVVals','AverageEpRet','AverageEpCost']
    data_dist = data[tag_list]
    for algo in ['FPO','CPO','PPO-L','PPO']: #
        epoch = 149
        mean = data_dist[np.logical_and(data['Condition1'] == algo, data['Epoch'] == epoch)].mean()
        std = data_dist[np.logical_and(data['Condition1'] == algo, data['Epoch'] == epoch)].std()
        for tag in tag_list:
            print('Algo: ' + algo + ' tag: ' + tag + ' mean: {:.3f}'.format(mean[tag]) + 'std: {:.3f}'.format(std[tag]))



def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.

    Assumes that any file "progress.txt" is a valid hit.
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root,'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_table(os.path.join(root,'progress.txt'))
            except:
                print('Could not read from %s'%os.path.join(root,'progress.txt'))
                continue
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns),'Unit',unit)
            exp_data.insert(len(exp_data.columns),'Condition1',condition1)
            exp_data.insert(len(exp_data.columns),'Condition2',condition2)
            exp_data.insert(len(exp_data.columns),'Performance',exp_data[performance])
            datasets.append(exp_data)
    return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           pull data from it;

        2) if not, check to see if the entry is a prefix for a
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1]=='/':
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x : osp.join(basedir, x)
            prefix = logdir.split('/')[-1]
            listdir= os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '='*DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            data += get_datasets(log)
    return data


def make_plots(all_logdirs, legend=None, xaxis=None, values=None, count=False,
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean',
               paper=True, hidelegend=False, title=None, savedir=None, show=True,
               clear_xticks=False, logy=False, cstrline=False):
    data = get_all_datasets(all_logdirs, legend, select, exclude)
    env_name = all_logdirs[0].split('-')[-2]
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
    for value in values:
        plot_data(data, xaxis=xaxis, value=value, condition=condition,
                  smooth=smooth, estimator=estimator,
                  paper=paper, hidelegend=hidelegend,
                  title=title, savedir=savedir,
                  clear_xticks=clear_xticks,logy=logy, cstrline=cstrline, env_name=env_name)

    # if show:
    #     plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('logdir', default=['../data/2021-11-20_ppo_Safexp-CustomPush2-v0',
    #                                       ], nargs='*')
    # parser.add_argument('logdir', default=['../data/2021-11-19_ppo_dual_ascent_Safexp-CustomGoal2-v0',
    #                                        '../data/2021-11-20_cpo_Safexp-CustomGoal2-v0',
    #                                        '../data/2021-11-20_ppo_lagrangian_Safexp-CustomGoal2-v0',
    #                                        '../data/2021-11-20_ppo_Safexp-CustomGoal2-v0',
    #                                        ], nargs='*')
    parser.add_argument('logdir', default=['../data/2021-11-21_ppo_dual_ascent_Safexp-CustomPush2-v0',
                                           '../data/2021-11-21_cpo_Safexp-CustomPush2-v0',
                                           '../data/2021-11-21_ppo_lagrangian_Safexp-CustomPush2-v0',
                                           '../data/2021-11-21_ppo_Safexp-CustomPush2-v0',
                                           ], nargs='*')
    # parser.add_argument('logdir', default=['../data/2021-11-19_ppo_dual_ascent_Safexp-CustomGoalPillar2-v0',
    #                                        '../data/2021-11-20_cpo_Safexp-CustomGoalPillar2-v0',
    #                                        '../data/2021-11-20_ppo_lagrangian_Safexp-CustomGoalPillar2-v0',
    #                                        '../data/2021-11-20_ppo_Safexp-CustomGoalPillar2-v0',
    #                                        ], nargs='*')
    parser.add_argument('--legend', '-l', default=['FPO','CPO','PPO-L','PPO'], nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--value', '-y', default=['AverageEpCost,h10,u100'], nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--cstrline', default=False, action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    parser.add_argument('--paper', default=True, action='store_true')
    parser.add_argument('--hidelegend', '-hl', default=True, action='store_true')
    parser.add_argument('--title', type=str, default='Performance')
    parser.add_argument('--savedir', type=str, default='data/figure')
    parser.add_argument('--dont_show', default=False, action='store_true')
    parser.add_argument('--clearx', action='store_true')
    parser.add_argument('--logy', default=False)
    args = parser.parse_args()
    """

    Args: 
        logdir (strings): As many log directories (or prefixes to log 
            directories, which the plotter will autocomplete internally) as 
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The 
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one 
            match for a given logdir prefix, and you will need to provide a 
            legend string for each one of those matches---unless you have 
            removed some of them as candidates via selection or exclusion 
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis. 
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the 
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the 
            off-policy algorithms. The plotter will automatically figure out 
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for 
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``, 
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves 
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This 
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show 
            curves from logdirs that do not contain these substrings.

    """

    make_plots(args.logdir, args.legend, args.xaxis, args.value, args.count,
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est, paper=args.paper, hidelegend=args.hidelegend,
               title=args.title, savedir=args.savedir, show=not(args.dont_show),
               clear_xticks=args.clearx, logy=args.logy, cstrline=args.cstrline)

if __name__ == "__main__":
    main()