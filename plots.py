from datetime import datetime, timedelta
from os.path import isfile

import seaborn as sns
from matplotlib import use
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, gca, xlim, ylim, savefig, tight_layout, legend
from numpy import array, nan
from progress.bar import Bar

from almanac import equation_of_time, meridian_passage


def eqot_img(year, filename=None, size=(16, 9)):
    sns.set_theme(context="notebook", style="whitegrid")
    if filename:
        if isfile(filename): return filename
        use("Agg")
    eqot = []
    xticks = []
    xlabels = []
    for d in Bar(filename or "Equation of Time Graph", max=365, suffix="%(percent)d%% %(eta_td)s").iter(range(365)):
        t = datetime(year, 1, 1) + timedelta(days=d)
        if t.day in [1, 10, 20]:
            xticks.append(d)
            xlabels.append(f"{t:%b}" if t.day == 1 else t.day)
        eqot.append(equation_of_time(t) * 60)

    figure(figsize=size)
    plot(eqot)
    title(f"Equation of Time {year}")
    xlabel("month of year")
    ylabel("solar time - mean time (UT1) [minutes]")
    xlim(0, 364)
    gca().set_xticks(xticks, xlabels)
    tight_layout()
    if filename:
        savefig(filename, pad_inches=0)
    return filename


def mp_img(year, filename=None, size=(16, 9)):
    sns.set_theme(context="notebook", style="whitegrid")
    if filename:
        if isfile(filename): return filename
        use("Agg")
    dashes = {"Sun": [],
              "Jupiter": [6, 2],
              "Saturn": [1, 2],
              "Venus": [1, 2, 1, 2, 6, 2],
              "Mars": [1, 2, 6, 2, 6, 2],
              "Mercury": [1, 2, 6, 2], }
    mp = {p: [] for p in dashes.keys()}
    xticks = []
    xlabels = []
    for d in Bar(filename or "Meridian Passages Graph", max=365, suffix="%(percent)d%% %(eta_td)s").iter(range(365)):
        t = datetime(year, 1, 1) + timedelta(days=d)
        if t.day in [1, 10, 20]:
            xticks.append(d)
            xlabels.append(f"{t:%b}" if t.day == 1 else t.day)
        for p in mp.keys():
            mp[p].append(meridian_passage(t, p))

    figure(figsize=size)
    for p, d in mp.items():
        d = array(d)
        d[d > 23.8] = nan
        plot(d, label=p, dashes=dashes[p])
    title(f"Meridian Passages {year}")
    xlabel("month of year")
    ylabel("MP [hours]")
    xlim(0, 364)
    ylim(0, 24)
    legend()
    gca().set_xticks(xticks, xlabels)
    yticks = 0, 3, 6, 9, 12, 15, 18, 21
    gca().set_yticks(yticks, yticks)
    tight_layout()
    if filename:
        savefig(filename, pad_inches=0)
    return filename
