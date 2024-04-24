import argparse
import os

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy


BLACK = 'k'
GREEN = '#2ca02c'
BLUE = '#1f77b4'
RED = '#d62728'
ORANGE = '#ff7f0e'
PURPLE = '#9467bd'
TURQUOIS = '#17becf'


def compute_avg_reward(reward):
    avg_reward = np.zeros_like(reward)

    for i in range(len(reward)):
        avg_reward[i] = np.sum(reward[:(i + 1)]) / (i + 1)

    return avg_reward


def get_results(figure_num, results_dir):
    x_ticks = [0, 2000, 4000, 6000, 8000, 10000]
    x_tick_vals = x_ticks

    y_ticks = None
    marker = None

    x_label = "Steps"
    y_label = "Average rewards"

    if figure_num == 4:
        results = []
        legend = []
        fig_dir = f"{results_dir}/sum_rate_power"

        values = [8, 32]

        for val in values:
            results.append(np.load(f"{fig_dir}/{val}.npy").squeeze())
            legend.append(f"M = {val}, N = {val}, K = {val}")

        legend_loc = 'upper left'
        colors = [RED, BLUE]

        x_ticks = np.arange(-20, 35, 5)
        x_tick_vals = x_ticks
        y_ticks = np.arange(0, 40, 5)

        marker = ['o', '<']

        x_label = "$P_{t}$ (dB)"
        y_label = "Sum rate (bps/Hz)"
    if figure_num == 5:
        legend = []
        fig_dir = f"{results_dir}/sum_rate_ris"

        results = [np.load(f"{fig_dir}/result.npy")]
        legend.append("Proposed DRL Method")
        legend_loc = 'upper left'
        colors = [RED]

        x_ticks = np.arange(10, 210, 10)
        x_tick_vals = x_ticks
        y_ticks = np.arange(12, 34, 2)

        x_label = "Number of elements in RIS"
        y_label = "Sum rate (bps/Hz)"
    elif figure_num == 6:
        fig_dir = f"{results_dir}/power"

        for file_name in os.listdir(fig_dir):
            if "5" in file_name:
                five = np.load(f"{fig_dir}/{file_name}").squeeze()
            elif "30" in file_name:
                thirty = np.load(f"{fig_dir}/{file_name}").squeeze()

        avg_five = compute_avg_reward(five)
        avg_thirty = compute_avg_reward(thirty)

        results = [thirty, five, avg_thirty, avg_five]
        legend = ["Instant Rewards, $P_{t}$ = 30dB", "Instant Rewards, $P_{t}$ = 5dB", "Average Rewards, $P_{t}$ = 30dB", "Average Rewards, $P_{t}$ = 5 dB"]
        legend_loc = 'upper left'
        colors = [BLUE, GREEN, RED, PURPLE]

        y_ticks = [1, 10]

        y_label = "Rewards"
    elif figure_num == 7:
        results = []
        legend = []

        fig_dir = f"{results_dir}/power"

        power_levels = [-10, 0, 10, 20, 30]

        for p_t in power_levels:
            reward = np.load(f"{fig_dir}/{p_t}.npy").squeeze()
            avg_reward = compute_avg_reward(reward)
            results.append(avg_reward)

            legend.append(f"$P_t$ = {p_t}dB")

        legend_loc = 'best'
        colors = [RED, BLUE, TURQUOIS, PURPLE, BLACK]

        y_ticks = [1, 2, 3, 4, 5, 6, 7, 8]
    elif figure_num == 8:
        results = []
        legend = []

        fig_dir = f"{results_dir}/rsi_elements"

        rsi_N = [30, 20, 10, 4]

        for N in rsi_N:
            reward = np.load(f"{fig_dir}/{N}.npy").squeeze()
            avg_reward = compute_avg_reward(reward)
            results.append(avg_reward)

            legend.append(f"M = 4, N = {N}, K = 4")

        legend_loc = 'lower right'
        colors = [RED, BLUE, TURQUOIS, PURPLE]

        y_ticks = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    save_name = f"{figure_num}_reproduced.jpg"

    return results, legend, legend_loc, colors, x_ticks, x_tick_vals, y_ticks, marker, x_label, y_label, save_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Choose the type of the experiment
    parser.add_argument('--figure_num', default=4, type=int, choices=[4, 5, 6, 7, 8],
                        help='Choose one of figures from the paper to reproduce')

    args = parser.parse_args()

    results_dir = "./DRL_Implementation/Learning Curves"
    fig_dir = f"./DRL_Implementation/Learning Figures"

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    results, legend, legend_loc, colors, x_ticks, x_tick_vals, y_ticks, markers, x_label, y_label, save_name = get_results(args.figure_num, results_dir)

    plt.rcParams['figure.figsize'] = [12, 10]

    linewidth = 3
    legend_size = 30

    font_size = 15 if args.figure_num == 5 else 25
    legend_font_size = 15 if args.figure_num == 6 or args.figure_num == 7 or args.figure_num == 10 or args.figure_num == 11 else 25

    if markers is None:
        for res, color in zip(results, colors):
            if args.figure_num == 5:
                plt.plot(x_ticks, res, linewidth=linewidth, color=color)
            else:
                plt.plot(res, linewidth=linewidth, color=color)
    else:
        if args.figure_num == 4 or args.figure_num == 9:
            for res, color, marker in zip(results, colors, markers):
                # plt.scatter(x_ticks, res, s=150, marker=marker, facecolors='none', edgecolors=color, linewidth=linewidth)
                plt.plot(x_ticks, res, color=color, marker=marker, linewidth=linewidth)

    y_ticks_vals = y_ticks

    plt.xticks(x_tick_vals, x_ticks, fontsize=font_size)

    if y_ticks is not None:
        plt.yticks(y_ticks_vals, y_ticks, fontsize=font_size)

    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)

    plt.legend(legend, loc=legend_loc, fontsize=legend_font_size, ncol=1)

    plt.grid(True)

    plt.savefig(f"{fig_dir}/{save_name}", bbox_inches='tight')
    plt.show()
