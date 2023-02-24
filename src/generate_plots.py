import json
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

def generate_plots(params):

    colors = {
        'brute_force': 'blue',
        'random_search': 'orange',
        'method_no_pruning_no_exploitation': 'green',
        'method_no_exploitation': 'red',
        'mcts': 'purple'
    }

    legend_names = {
        'brute_force': 'BF-DT',
        'random_search': 'RANDOM',
        'method_no_pruning_no_exploitation': 'BF-ST',
        'method_no_exploitation': 'BF-ST-PRUN',
        'mcts': 'RA-MCTS'
    }

    # threshold_markers = ["s", "o", "D"]
    
    print("Begin Generating Plots\n")

    # Load Experiment Parameters
    # experiment mode
    ground_truth = params['ground_truth']
    uncertainty = params['uncertainty']
    # environment configuration
    lst_env_size = params['lst_env_size']
    # evaluation parameters
    lst_tot_env_steps = params['lst_tot_env_steps']
    frq_env_steps = params['frq_env_steps']
    d = params['d']
    if uncertainty:
        lst_d_unc = params['lst_d_unc']
    # responsibility attribution method
    methods = params['methods']

    #
    if not uncertainty:
        d_list = [d]
    else:
        d_list = lst_d_unc

    for env_size in lst_env_size:
        tot_env_steps = lst_tot_env_steps[lst_env_size.index(env_size)]
        fig, ax = plt.subplots()
        for threshold in d_list:
            # Load Results
            with open(f'results/threshold={threshold}.json', 'r') as f:
                resuts = json.load(f)
            #
                for method in methods:
                    lst_mean = [resuts[f'env_size={env_size}'][f'method={method}'][f'env_steps={env_steps}']['mean'] for env_steps in range(0, tot_env_steps + 1, frq_env_steps)]
                    lst_std = [resuts[f'env_size={env_size}'][f'method={method}'][f'env_steps={env_steps}']['std'] for env_steps in range(0, tot_env_steps + 1, frq_env_steps)]
                    mean_max = np.max(lst_mean)
                    if mean_max == 1:
                        # converged within budget
                        conv_point = np.argmax(lst_mean)
                        lst_mean = lst_mean[:conv_point + 1]
                        lst_std = lst_std[:conv_point + 1]
                        conv_steps = conv_point * frq_env_steps
                        plt.plot([conv_steps], [1.0], marker="*", markersize=13, color=colors[method])
                    else:
                        # did not converge within budget
                        conv_steps = tot_env_steps
                    plt.plot(list(range(0, conv_steps + 1, frq_env_steps)), lst_mean, linewidth=2, label=legend_names[method], color=colors[method])
                    #
                    # if uncertainty:
                    #     plt.plot(list(range(0, conv_steps + 1, conv_steps // 10)), [lst_mean[i] for i in range(len(lst_mean)) if (i* frq_env_steps)%(conv_steps // 10)==0], marker=threshold_markers[d_list.index(threshold)], markersize=11, color=colors[method], linestyle = 'None')
                    # 
                    plt.fill_between(list(range(0, conv_steps + 1, frq_env_steps)), [max(0, lst_mean[i] - lst_std[i]) for i in range(0, len(lst_mean))], [min(1.0, lst_mean[i] + lst_std[i]) for i in range(0, len(lst_mean))], alpha=0.5, color=colors[method])
        plt.xlabel('Number of Steps', fontsize=25)
        if threshold == 0:
            threshold = int(threshold)
        if ground_truth == 'full' and not uncertainty:
            plt.ylabel("Frac. of runs with $\epsilon_{max} \leq $" + f"{threshold}", fontsize= 25)
        elif not uncertainty:
            plt.ylabel("Frac. of runs with $\epsilon^{lo}_{max} \leq $" + f"{threshold}", fontsize=25)
        elif ground_truth == 'full':
            plt.ylabel("Frac. of runs with $\epsilon_{max} \leq D$", fontsize=25)
        else:
            plt.ylabel("Frac. of runs with $\epsilon^{lo}_{max} \leq D$", fontsize=25)
        plt.tick_params(labelsize=20)
        plt.gca().get_xaxis().set_major_formatter(ScalarFormatter(useMathText=True))
        plt.gca().ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.xaxis.set_ticks(np.arange(tot_env_steps // 5, tot_env_steps + 1, tot_env_steps // 5))
        ax.xaxis.get_offset_text().set_fontsize(20)
        plt.xlim(left=tot_env_steps // 200)
        ax.yaxis.set_ticks(np.arange(.2, 1.2, .2))
        plt.ylim(bottom=0, top=1.05)
        plt.tight_layout()
        fig.savefig(f"plots/env_size={env_size}.pdf", bbox_inches = 'tight')
    # Legend
    legend = plt.legend(
            ncol=len(methods), fancybox=True, facecolor="white",
            shadow=True, fontsize=20
        )
    fig.canvas.draw()
    legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
    legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
    legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))
    legend_squared = legend_ax.legend(
        *ax.get_legend_handles_labels(), 
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=legend_fig.transFigure,
        frameon=True,
        facecolor="white",
        fancybox=True,
        shadow=True,
        ncol=len(methods),
        fontsize=20,
    )
    legend_ax.axis('off')
    legend_fig.savefig("plots/legend.pdf",  
                        bbox_inches='tight',
                        bbox_extra_artists=[legend_squared]
    )
    plt.close(legend_fig)
    plt.close(fig)    



# colors = {
#         'brute_force': 'blue',
#         'random_search': ['#ff9933', '#ff8000', '#cc6600'],
#         'method_no_pruning_no_exploitation': 'green',
#         'method_no_exploitation': 'red',
#         'mcts': ['#cc99ff', '#9933ff', '#6600cc']
#     }