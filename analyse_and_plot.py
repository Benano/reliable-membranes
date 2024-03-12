"""Analysing and plotting the results."""
from scipy.stats import norm
import yaml
import matplotlib.pyplot as plt
import numpy as np
import main as sbi_main
import os
import pickle
import scipy.stats as sps
import scipy.stats as scs
import copy
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker


def gaussian(x, mu, sigma):
    """Gaussian pdf."""
    y = ((1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x-mu)**2)/sigma))

    return y


def plot_theory_membrane(plot_params, save_folder):
    """Plot the bayesian integration of information in structural neurons."""
    colors = {
        "prior": {
            "pre": "darkgrey",
            "post": "darkgrey"},
        "posterior": {
            "pre": "olivedrab",
            "post": "teal"},
        "likelihood": {
            "pre": "darkorange",
            "post": "orangered"}}

    params = {
        "prior": {
            "pre": [-68, 3],
            "post": [-68, 3]},
        "posterior": {
            "pre": [-67, 2.2],
            "post": [-63, 1.2]},
        "likelihood": {
            "pre": [-62, 3],
            "post": [-62, 1.5]}}

    class lil_neuron():
        def __init__(self, mu, std, tau):
            self.mu = mu
            self.std = std
            self.tau = tau
            self.mem_trace = []

        def simulate(self, time, dt):
            noise = np.random.randn(int(time/dt)) * self.std
            c_mem = copy.deepcopy(self.mu)
            mem_trace = []

            # Add dt
            for st, ct in enumerate(np.arange(0, time, dt)):
                change_mem = (self.mu - c_mem) * dt/self.tau + noise[st]*dt
                c_mem += change_mem
                mem_trace.append(c_mem)

            return mem_trace

    np.random.seed(69)
    time = 500
    dt = 0.1
    steps = int(time/dt)
    tau = 15
    fig, axs = plt.subplots(nrows=3,
                            ncols=2,
                            dpi=plot_params['dpi'],
                            figsize=(4, 4),
                            gridspec_kw={'width_ratios': [6, 1.5]})

    for dn, dist in enumerate(['prior', 'posterior', 'likelihood']):

        # Current axis
        cax = axs[dn, 0]
        cax.set_ylim(-80, -54)
        cax.set_xlim(-time, time)
        if dn == 2:
            cax.spines[['right', 'top']].set_visible(False)
        else:
            cax.spines[['right', 'top', 'bottom']].set_visible(False)
            cax.set_xticks([])

        if dn == 1:
            cax.set_ylabel('membrane potential (mV)')

        for pn, pp in enumerate(['pre', 'post']):

            # Defining noise
            std = params[dist][pp][1]
            mu = params[dist][pp][0]
            c_neuron = lil_neuron(mu, std, tau)
            mem = c_neuron.simulate(time, dt)

            # Link
            if not pn:
                last_point = mem[-1]
            if pp:
                mem[0] = last_point

            # X & Y
            x = np.linspace(-time + (pn*time), pn * time, steps)
            color = colors[dist][pp]
            y = np.ones(len(x))*mu
            y_up = y + std
            y_down = y - std

            # Plot
            cax.plot(x, mem, color='k', alpha=0.8, lw=0.7)
            cax.fill_between(x, y_up, y_down, color=color, alpha=0.7)

        cax.axvline(0, color='red', ls='--', label='cue change', lw=1)
        if dn == 2:
            cax.legend(fontsize=8, frameon=False, loc='lower left')

        # PDF plot adjustments
        pdf_ax = axs[dn, 1]
        pdf_ax.spines['top'].set_visible(True)
        pdf_ax.spines['bottom'].set_visible(False)
        pdf_ax.spines[['right']].set_visible(False)
        pdf_ax.set_xticks([])
        pdf_ax.set_ylim(-80, -54)
        pdf_ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        pdf_ax.set_title(dist, fontsize=10, pad=3)

        x_pdf = np.linspace(-80, -54, 100)
        for pp in ['pre', 'post']:
            mu, std = params[dist][pp]
            pdf = norm.pdf(x_pdf, mu, std)
            pdf_ax.plot(pdf, x_pdf, color=colors[dist][pp], lw=2)
            pdf_ax.set_xlim(0, 0.35)
            pdf_ax.fill_betweenx(x_pdf, 0,
                                 pdf,
                                 color=colors[dist][pp],
                                 alpha=0.3)

    cax.set_xlabel('time (ms)')
    fig.tight_layout(w_pad=0.5)

    fig.savefig(plot_folder + '/Fig1_membrane_theory' +
                plot_params['fig_format'])


def plot_raw_spikes(sim_params, data_params, plot_params,
                    trial_data, save_folder):
    """Raster plot with a random selection of trial activity."""
    # Seed for selection
    np.random.seed(99)

    # Sort trial
    idxs = pd.Index([])
    for cn, condition in enumerate(['thresh', 'max']):
        for mn, modality in enumerate(['auditory', 'visual', 'audio-visual']):
            ct = trial_data[trial_data['fit_trial'] == True]
            ct = ct[ct['condition'] == condition]
            ct = ct[ct['modality'] == modality]
            sample_ct = ct.sample(2).index
            idxs = idxs.union(sample_ct)

    shuffled_idxs = idxs.to_series().sample(frac=1).index
    df_to_plot = trial_data.loc[shuffled_idxs]

    gap = 0

    colors = {
        "catch": 'grey',
        "auditory_thresh": 'lightblue',
        "auditory_max": 'royalblue',
        "visual_thresh": 'lightcoral',
        "visual_max": 'firebrick',
        "audio-visual_thresh": 'plum',
        "audio-visual_max": 'purple'
    }

    # Iterate through trials
    fig, ax = plt.subplots(dpi=plot_params['dpi'], figsize=(4, 4))
    for tn in range(len(df_to_plot) + gap):

        if tn < len(df_to_plot):
            trial = df_to_plot.iloc[tn]
        t_spikes = np.array([])

        # Pre
        pre_spikes = trial['spikes_pre']
        t_spikes = np.hstack((t_spikes, pre_spikes)) - sim_params['simtime']

        # Transient
        transient_spikes = trial['spikes_transient']
        transient_spikes = transient_spikes
        t_spikes = np.hstack((t_spikes, transient_spikes))

        # Post spikes
        post_spikes = trial['spikes_post']
        post_spikes = post_spikes + data_params['skip_transient']
        t_spikes = np.hstack((t_spikes, post_spikes))

        cond = trial['condition']
        mod = trial['modality']
        color = colors[f'{mod}_{cond}']

        if tn >= len(df_to_plot):
            alpha = 0
        else:
            alpha = 1

        spx = t_spikes
        y = (np.ones(len(spx)) * tn) + 1
        ax.scatter(spx, y, marker='|', color=color, alpha=alpha, s=70)

    ax.set_xlabel('time (ms)', fontsize=15)
    ax.set_ylabel('trial', fontsize=15)
    ax.set_ylim(0.5, 16)
    ax.set_xlim(-1500, 1700)
    ax.spines[['right', 'top']].set_visible(False)

    fig.savefig(save_folder + '/Fig3_example_spike_trains' +
                plot_params['fig_format'])

    return df_to_plot


def plotting_inference(sbi_params, sim_params, neuron_params,
                       df_to_plot,
                       posterior,
                       plot_folder, plot_params):
    """Plot the ISI distribution and inferred parameters of spike train."""
    trial_to_plot = df_to_plot.iloc[plot_params['trial_to_plot']-1]
    ts = trial_to_plot['spikes_post']
    isi = sbi_main.compute_isi_stats(ts)
    isi.append(len(ts))

    # Recover Parameters
    x = [isi]
    posterior.set_default_x(x)
    temp_samples = posterior.sample((50000,),
                                    show_progress_bars=False)
    prob = posterior.log_prob(temp_samples)
    top_prob = np.argmax(prob)
    c_rec_I = temp_samples[top_prob].numpy()
    isi = np.diff(ts)
    log_isi = np.log2(isi)

    # Bins & Ticks
    width = 0.5
    min_x = 0
    max_x = 10
    bins = np.arange(min_x, max_x, width)
    tick_labels = np.power(2, bins).astype(int)
    counts, _ = np.histogram(log_isi, bins=bins, density=False)

    # Raster plot
    fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2,
                                   height_ratios=(1, 9),
                                   figsize=(4, 4))
    fig.subplots_adjust(hspace=0.32)
    # Raster
    y = np.ones(len(ts))
    x = ts
    ax1.scatter(x, y,
                c='k',
                marker='|',
                s=100)
    ax1.set_ylim(0.97, 1.03)
    ax1.set_xlim(0, 1505)
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.set_xticks(np.arange(0, 1500, 200))
    ax1.set_yticks([])
    ax1.set_xlabel('time (ms)')

    # Log ISI
    isi = np.diff(ts)
    log_isi = np.log2(isi)

    # Bins & Ticks
    width = 1
    min_x = 0
    max_x = 10
    bins = np.arange(min_x, max_x, width)
    tick_labels = np.power(2, bins).astype(int)
    counts, _ = np.histogram(log_isi, bins=bins, density=False)

    # Barplot
    ax2.bar(bins[:-1], counts, width=width,
            color='lightgrey', edgecolor='k', align='edge',
            linewidth=1)

    # Gaussian
    mu = np.mean(log_isi)
    sigma = np.std(log_isi)
    x = np.linspace(min_x, max_x, 1000)
    y = gaussian(x, mu, sigma) * len(log_isi)
    ax2.plot(x, y, color='black',
             linewidth=2.5)
    ax2.spines[['right', 'top']].set_visible(False)

    # Ticks and Labels
    ax2.set_ylabel('count', fontsize=15, labelpad=5)
    ax2.set_xlabel('ISI (ms)', fontsize=15)
    every = 1
    ax2.set_xticks(bins[::every])
    ax2.set_xticklabels(tick_labels[::every])
    ax2.set_yticks(np.arange(min_x, 20.1, 5))
    ax2.set_xlim(1, 9)
    ax2.set_ylim(0, 10)
    ax2.tick_params(axis='both')

    fig.savefig(plot_folder + '/Fig3_spikes_and_distribution' +
                plot_params['fig_format'])

    # Recover Parameters
    # Density
    fig, ax = plt.subplots(dpi=plot_params['dpi'],
                           figsize=(4.5, 4))
    ax.set_xlabel(r"$\mu$ current (pA)", fontsize=15)
    ax.set_ylabel(r"$\sigma$ current (pA)", fontsize=15)

    samples_kde = temp_samples.numpy().T
    kernel = scs.gaussian_kde(samples_kde)
    xmin = -1.5
    xmax = 1.5
    ymin = 0
    ymax = 21
    ax.set_xticks(np.arange(xmin, xmax+0.1, 0.5))
    ax.set_yticks(np.arange(ymin, ymax+0.1, 5))
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    Y, X = np.mgrid[ymin:ymax:200j, xmin:xmax:200j]

    positions = np.vstack([Y.ravel(), X.ravel()])
    Z = np.reshape(kernel(positions), X.shape)
    Z[Z < 0.00001] = 0

    # Distribution
    im = ax.imshow(Z, cmap=plt.cm.CMRmap_r,
                   extent=[xmin, xmax, ymin, ymax], aspect='auto',
                   origin='lower')
    cb = fig.colorbar(im, orientation='vertical')
    cb.set_label(r'posterior',
                 fontsize=15)
    ax.spines[['right', 'top']].set_visible(False)
    ax.scatter(c_rec_I[1],
               c_rec_I[0],
               marker='+',
               s=80,
               color='grey',
               label=r'inferred ($\mu_I$, $\sigma_I$)')
    ax.axvline(c_rec_I[1], c='grey', ls='--', lw='1', alpha=0.7)
    ax.axhline(c_rec_I[0], c='grey', ls='--', lw='1', alpha=0.7)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(plot_folder + '/Fig3_conditioned_posterior' +
                plot_params['fig_format'])


def plot_results(plot_params, df, plot_folder):
    colors = {
        "no_change": 'white',
        "thresh": 'lightgrey',
        "max": 'grey',
        "auditory": 'royalblue',
        "visual": 'firebrick',
        "audio-visual": 'purple'
    }

    # Variables
    result_types = ['isi', 'voltages']
    modalities = ["auditory", "visual", "audio-visual"]
    conditions = ["no_change", "thresh", "max"]

    # Adjust bar width
    bar_width = 0.15  # Make bars thinner
    bar_space = 0.02
    lw = 1
    error_kw = dict(lw=1,
                    capsize=5,
                    capthick=1)

    for res_type in result_types:
        for stat in ['mu', 'std']:
            fig, ax = plt.subplots(ncols=2, dpi=plot_params['dpi'],
                                   figsize=(3, 3.5))
            fig.subplots_adjust(wspace=0.4)

            ax[0].axhline(y=0, c='k', linewidth=lw)
            ax[1].axhline(y=0, c='k', linewidth=lw)

            for cn, condition in enumerate(conditions):
                condition_data = np.array([])

                if condition == 'no_change':
                    c_mod = df['modality'] == 'catch'
                    c_cond = df['condition'] == condition
                    c_lick = df['Lick'] == 0
                    stat_label = f"{res_type}_{stat}_diff"
                    condition_data = \
                        np.append(condition_data,
                                  df[c_mod & c_lick & c_cond][stat_label])
                else:
                    for modality in modalities:
                        c_mod = df['modality'] == modality
                        c_cond = df['condition'] == condition
                        stat_label = f"{res_type}_{stat}_diff"
                        condition_data = \
                            np.append(condition_data,
                                      df[c_mod & c_cond][stat_label])

                mean = np.mean(condition_data)
                e_bar = sps.sem(condition_data)
                ax[0].bar(cn * bar_width,
                          mean,
                          yerr=e_bar,
                          width=bar_width - bar_space,
                          color=colors[condition],
                          alpha=1,
                          edgecolor='k',
                          linewidth=lw,
                          error_kw=error_kw)

            x_ticks_cond = [i * bar_width for i in range(len(conditions))]
            ax[0].set_xticks(x_ticks_cond)
            ax[0].set_xticklabels(['Catch', 'Threshold', 'Max'])

            # Modality Bars
            for mn, modality in enumerate(modalities):
                modality_data = np.array([])
                for condition in conditions[1:]:
                    c_mod = df['modality'] == modality
                    c_cond = df['condition'] == condition
                    stat_label = f"{res_type}_{stat}_diff"
                    modality_data = \
                        np.append(modality_data,
                                  df[c_mod & c_cond][stat_label])

                mean = np.mean(modality_data)
                e_bar = sps.sem(modality_data)
                ax[1].bar(mn * bar_width,
                          mean,
                          yerr=e_bar,
                          width=bar_width - bar_space,
                          color=colors[modality],
                          alpha=1,
                          edgecolor='k',
                          linewidth=lw,
                          error_kw=error_kw)

            x_ticks_mod = [i * bar_width for i in range(len(modalities))]
            ax[1].set_xticks(x_ticks_mod)
            ax[1].set_xticklabels(['Auditory', 'Visual', 'Audio-Visual'])

            if res_type == 'isi':
                y_min, y_max = -9, 2.5
                ax[0].yaxis.set_major_locator(ticker.MultipleLocator(base=2))
                ax[1].yaxis.set_major_locator(ticker.MultipleLocator(base=2))
            elif res_type == 'voltages':
                y_min, y_max = -1.2, 2.5
                ax[0].yaxis.set_major_locator(ticker.MultipleLocator(base=1))
                ax[1].yaxis.set_major_locator(ticker.MultipleLocator(base=1))
            ax[0].set_ylim(y_min, y_max)
            ax[1].set_ylim(y_min, y_max)

            # Setting x
            x_min_mod = min(x_ticks_mod) - (bar_width * 0.6)
            x_max_mod = max(x_ticks_mod) + (bar_width * 0.6)
            x_min_cond = min(x_ticks_cond) - (bar_width * 0.6)
            x_max_cond = max(x_ticks_cond) + (bar_width * 0.6)

            ax[0].set_xlim(x_min_cond, x_max_cond)
            ax[1].set_xlim(x_min_mod, x_max_mod)

            ax[0].spines[['right', 'top']].set_visible(False)
            ax[1].spines[['right', 'top']].set_visible(False)

            # Set common labels and titles based on the stat and res_type
            for a in ax:
                a.tick_params(axis='both', labelsize=12)

            if res_type == 'isi':
                y_label = \
                    r"post - pre ISI $\mu$ (ms)" if stat == 'mu' else r"post - pre ISI $\sigma$ (ms)"
            else:  # voltages
                y_label = r"post - pre voltage $\mu$ (mV)" if stat == 'mu' else r" post - pre voltage $\sigma$ (mV)"
            ax[0].set_ylabel(y_label, fontsize=12)

            # Set y-axis labels and ticks for the left plot only
            ax[0].tick_params(axis='y', labelsize=12)
            ax[1].set_yticklabels([])

            for axis in ['top', 'bottom', 'left', 'right']:
                ax[0].spines[axis].set_linewidth(lw)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax[1].spines[axis].set_linewidth(lw)

            # Adjust x-axis tick labels
            ax[0].set_xticks([i * bar_width for i in range(3)])
            ax[0].set_xticklabels(['C', 'T', 'M'])
            ax[1].set_xticks([i * bar_width for i in range(3)])
            ax[1].set_xticklabels(['A', 'V', 'AV'])

            fig.tight_layout()

            # Save each figure
            fig.savefig(plot_folder + f"/Fig4_{res_type}_{stat}{plot_params['fig_format']}",
                        bbox_inches='tight')


def plot_bursting_histogram(unf_df, plot_folder):
    """Histogram for the burst metric in all trials."""
    bursting_pre = unf_df['burst_pre']
    bursting_post = unf_df['burst_post']

    # Calculate means
    bins = np.arange(0, 0.75, 0.04)

    fig, ax = plt.subplots()
    # Create the histogram
    ax.hist(bursting_pre, bins=bins, alpha=0.5, label='Pre-change')
    ax.hist(bursting_post, bins=bins, alpha=0.5, label='Post-change')

    # Plot the means
    ax.axvline(0.5, color='red',
               linestyle='dashed',
               linewidth=2,
               label='cutoff')

    # Add labels and title
    ax.set_xlim(0, 0.8)
    ax.set_xlabel('bursting ratio')
    ax.set_ylabel('frequency')
    ax.set_title('bursting pre- and post-change')
    ax.legend()

    fig.savefig(plot_folder + '/FigSup_burst_hist' +
                plot_params['fig_format'])


def plot_isi_log_distribution(ts, ax1, ax2, xlabel, ylabel):
    """Plot single spike train isi log distribution."""
    # Raster plot
    # Raster
    y = np.ones(len(ts))
    x = ts

    ax1.scatter(x, y,
                c='k',
                marker='|',
                s=100)
    ax1.set_ylim(0.97, 1.03)
    ax1.set_xlim(0, 1505)
    ax1.spines[['right', 'top']].set_visible(False)
    ax1.set_xticks(np.arange(0, 1500, 200))
    ax1.set_yticks([])
    ax1.set_xlabel('time [ms]')

    # Log ISI
    # Simulate real_I to obtain real_v
    isi = np.diff(ts)
    log_isi = np.log2(isi)

    # Bins & Ticks
    width = 1
    min_x = 0
    max_x = 10
    bins = np.arange(min_x, max_x, width)
    tick_labels = np.power(2, bins).astype(int)
    counts, _ = np.histogram(log_isi, bins=bins, density=False)

    # Barplot
    ax2.bar(bins[:-1], counts, width=width,
            color='lightgrey', edgecolor='k', align='edge',
            linewidth=1)

    # Gaussian
    mu = np.mean(log_isi)
    sigma = np.std(log_isi)
    x = np.linspace(min_x, max_x, 1000)
    y = gaussian(x, mu, sigma) * len(log_isi)
    ax2.plot(x, y, color='black',
             linewidth=2.5)
    ax2.spines[['right', 'top']].set_visible(False)

    # Ticks and Labels
    if xlabel:
        ax2.set_ylabel('Count', fontsize=15)
    if ylabel:
        ax2.set_xlabel('ISI (ms)', fontsize=15)
    every = 1
    ax2.set_xticks(bins[::every])
    ax2.set_xticklabels(tick_labels[::every])
    ax2.set_yticks(np.arange(min_x, 20.1, 5))
    ax2.set_xlim(1, 9)
    ax2.tick_params(axis='both')


def plot_validation(validation_results, plot_folder):
    fig, axs = plt.subplots(2, 1, sharex='col', sharey='row', figsize=(5, 5))
    c_val = validation_results

    spikes = c_val['spikes']
    mu_diff = c_val['diff_I'][:, 1]
    std_diff = c_val['diff_I'][:, 0]

    step = 1
    fr = np.arange(3, 40, step)
    mu_errors = []
    std_errors = []

    lower_mu = []
    upper_mu = []
    lower_std = []
    upper_std = []

    for c_fr in fr:
        min_val = c_fr
        max_val = c_fr + step

        mask = (spikes > min_val) & (spikes <= max_val)
        c_mu_errors = mu_diff[mask]
        c_std_errors = std_diff[mask]

        mu_errors.append(np.median(c_mu_errors))
        std_errors.append(np.median(c_std_errors))
        lower_mu.append(np.percentile(mu_diff[mask], 25))
        upper_mu.append(np.percentile(mu_diff[mask], 75))
        lower_std.append(np.percentile(std_diff[mask], 25))
        upper_std.append(np.percentile(std_diff[mask], 75))

    # Plotting errors
    color = 'black'
    axs[0].plot(fr, mu_errors, color=color)
    axs[1].plot(fr, std_errors, color=color)

    axs[0].axhline(0, c='black', ls='--')
    axs[1].axhline(0, c='black', ls='--')

    # Plotting IQR
    axs[0].fill_between(fr, lower_mu, upper_mu, color=color, alpha=0.1)
    axs[1].fill_between(fr, lower_std, upper_std, color=color, alpha=0.1)

    # Adjust layout, add legends, and set common labels
    axs[0].set_title("recovered - real parameters")
    axs[0].set_ylabel(f'error $\mu_I$', fontsize=12)
    axs[1].set_ylabel(f'error $\sigma_I$', fontsize=12)
    axs[1].set_xlabel(r'N$_s$', fontsize=15)

    # FIGURE 2
    fig2, axs2 = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(5, 5))
    mu_diff_inc = mu_diff[spikes > 15]
    std_diff_inc = std_diff[spikes > 15]
    mu_diff_ex = mu_diff[spikes <= 15]
    std_diff_ex = std_diff[spikes <= 15]

    # Excluded
    axs2.scatter(mu_diff_ex, std_diff_ex,
                 color='darkorange', alpha=0.2, label='excluded', s=6)
    axs2.axvline(np.mean(mu_diff_ex), color='red', ls='--',
                 label=r'mean error $(N_{s} > 15)$')
    axs2.axhline(np.mean(std_diff_ex), color='red', ls='--')

    # Included
    axs2.scatter(mu_diff_inc, std_diff_inc, color='green',
                 alpha=0.2, label='included', s=6)
    axs2.axvline(np.mean(mu_diff_inc), color='darkgreen', ls='--',
                 label=r'mean error $(N_{s} \leq 15)$')
    axs2.axhline(np.mean(std_diff_inc), color='darkgreen', ls='--')
    axs2.legend(loc='lower left')

    # Labels
    axs2.set_xlabel(r'error $\mu_I$', fontsize=15)
    axs2.set_ylabel(r'error $\sigma_I$', fontsize=15)
    plt.tight_layout()

    fig.savefig(plot_folder + '/FigSup_validation_spikes' +
                plot_params['fig_format'])
    fig2.savefig(plot_folder + '/FigSup_validation_inclusion' +
                 plot_params['fig_format'])


def plot_bursting_raster(unf_df, plot_folder):
    """Plot a raster of bursting activity examples at different fr."""
    nr_spikes = [15, 30, 45]
    bursting = [0.3, 0.40, 0.50]

    # Create a figure
    fig = plt.figure(figsize=(16, 16))  # You can adjust the size as needed
    y_titles = ['10 - 15 Hz', '20 - 25 Hz', '30 - 35 Hz']
    x_titles = ['Burst ratio < 0.35',
                ' 0.4 < Burst ratio > 0.45',
                'Burst ratio > 0.5 (excluded)']

    # Create a 3x3 grid layout
    outer_grid = gridspec.GridSpec(4, 4, wspace=0.2, hspace=0.2)

    for nn, nr in enumerate(nr_spikes):
        for nb, b in enumerate(bursting):

            c_df = unf_df[((unf_df['nr_spikes_pre'] >= nr) &
                           (unf_df['nr_spikes_pre'] < nr + 5) &
                           (unf_df['burst_pre'] >= b) &
                           (unf_df['burst_pre'] < (b + 0.05)))]

            ts = np.array(c_df.sample(n=1, random_state=3)['spikes_pre'])[0]

            co_grid = outer_grid[nn, nb]

            inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                          subplot_spec=co_grid,
                                                          height_ratios=[1, 4])

            ax1 = plt.Subplot(fig, inner_grid[0])
            ax2 = plt.Subplot(fig, inner_grid[1])

            xlabel = False
            ylabel = False
            if nb == 0:
                xlabel = True
            if nn == 2:
                ylabel = True

            plot_isi_log_distribution(ts, ax1, ax2, xlabel, ylabel)

            fig.add_subplot(ax1)
            fig.add_subplot(ax2)

            if nn == 0:
                x_title = x_titles[nb]
                fig.text(0.20 + nb * 0.21, 0.9,  x_title,
                         va='center',
                         ha='center',
                         rotation='horizontal',
                         fontsize=15)

            y_title = y_titles[nn]
            fig.text(0.06, 0.78 - nn * 0.20, y_title,
                     va='center',
                     ha='center',
                     rotation='vertical',
                     fontsize=15)
    fig.savefig(plot_folder + '/FigSup_burst_grid' +
                plot_params['fig_format'])


def filter_for_nr_spikes(sim_params, min_spikes, max_spikes, df):
    """Filter out trials based on nr spikes (not Hz!)."""
    pre_filter = len(df)

    df = df[df['nr_spikes_pre'] >= min_spikes]
    df = df[df['nr_spikes_post'] >= min_spikes]

    df = df[df['nr_spikes_pre'] <= max_spikes]
    df = df[df['nr_spikes_post'] <= max_spikes]

    post_filter = len(df)
    nr = pre_filter - post_filter

    text = \
        f'removed {nr} trials with fewer than {min_spikes} and above {max_spikes} spikes.'
    print(text)

    return df


def filter_bursting(df, cutoff, show=False):
    """Filter out bursting trials."""

    pre_nr = len(df)

    df = df[(df['burst_pre'] <= cutoff) &
            (df['burst_post'] <= cutoff)]

    post_nr = len(df)
    nr = pre_nr - post_nr

    text = \
        f'removed {nr} trials with bursting ratio above {cutoff}.'

    print(text)

    return df


def save_thing(name, thing, folder):
    """Save the thing."""
    f_name = folder + "/" + name + ".pkl"
    with open(f_name, "wb") as handle:
        pickle.dump(thing, handle)


def load_result(data_folder, name):
    """Load and unpickle the saved results."""
    file_name = data_folder + '/' + name + '.pkl'
    unpickle_result = open(file_name, 'rb')
    result = pickle.load(unpickle_result)

    return result


def save_data_for_stats_analysis(df, data_folder):
    """Save data."""
    vars = [
            'animal_id',
            'session_ID',
            'trial_ID',
            'modality',
            'condition',
            'isi_mu_diff',
            'isi_std_diff',
            'voltages_mu_diff',
            'voltages_std_diff',
            'trialType',
            'visualOriPreChange',
            'visualOriPostChange',
            'visualOriChange',
            'audioFreqPreChange',
            'audioFreqPostChange',
            'audioFreqChange',
            'correctResponse',
            'noResponse',
            'audioFreqChangeNorm',
            'visualOriChangeNorm',
            'responseLatency',
            'responseModality',
            'unique_nidx',
            'Lick',
            'nr_spikes_pre',
            'isi_std_pre',
            'isi_mu_pre',
            'isi_std_post',
            'isi_mu_post',
            'log_isi_std_pre',
            'log_isi_mu_pre',
            'log_isi_std_post',
            'log_isi_mu_post',
            'voltages_std_pre',
            'voltages_mu_pre',
            'voltages_std_post',
            'voltages_mu_post',
            ]

    # Data for Analysis in SPSS
    # Main Results
    df = df[vars]
    no_catch = df['modality'] != 'catch'
    no_mix = df['condition'] != 'mix'
    df_analysis = df[no_mix & no_catch]
    df_analysis.to_csv(data_folder + '/data.csv')

    # Catch Results
    catch = df['modality'] == 'catch'
    no_response = df['noResponse'] == 1
    df_catch = df[no_mix & catch & no_response]
    df_catch.to_csv(data_folder + '/catch_data.csv')


def analyse_and_plot(sbi_params, sim_params, neuron_params,
                     data_params, plot_params, data_folder, plot_folder):
    """Plot all of the sbi results."""
    # Load Results
    df = load_result(data_folder, 'fit_trial_data')
    validation_results = load_result(data_folder, 'validation_results')
    posterior = load_result(data_folder, 'posterior')

    # Filtering Data
    fit_trial = df['fit_trial'] == True
    area = df['area'] == 'PPC'
    df = df[fit_trial & area]
    unf_df = sbi_main.add_burstyness_measure(df)

    # Filtering for firing rate
    min_spikes = plot_params['analysis_min_spikes']
    max_spikes = plot_params['analysis_max_spikes']
    spk_df = filter_for_nr_spikes(sim_params, min_spikes, max_spikes, unf_df)

    # Remove Bursting Trials
    burst_cutoff = plot_params['analysis_burst_cutoff']
    df = filter_bursting(spk_df, cutoff=burst_cutoff, show=True)

    # Remove mixed and false catch trials
    no_mix = df['condition'] != 'mix'
    df = df[no_mix]

    save_thing('filtered_data', df, data_folder)

    # Export data to CSV
    save_data_for_stats_analysis(df, data_folder)

    # # Figure 1: Theory
    plot_theory_membrane(plot_params, plot_folder)

    # Figure 3: Inference
    df_to_plot = plot_raw_spikes(sim_params, data_params, plot_params,
                                 df, plot_folder)
    plotting_inference(sbi_params, sim_params, neuron_params,
                       df_to_plot, posterior, plot_folder,
                       plot_params)

    # Figure 4: Results
    plot_results(plot_params, df, plot_folder)

    if plot_params['show_figures']:
        plt.show()

    # Figures Validation
    plot_bursting_raster(unf_df, plot_folder)
    plot_bursting_histogram(spk_df, plot_folder)
    plot_validation(validation_results, plot_folder)


if __name__ == '__main__':
    cwd = os.getcwd()
    data_folder = os.path.join(cwd, 'results_current')
    plot_folder = os.path.join(cwd, 'figures')

    if not os.path.isdir(plot_folder):
        os.mkdir(figure_folder)

    # General Params
    with open(data_folder + '/config.yml', 'r') as f:
        params = \
            yaml.load(f, Loader=yaml.FullLoader)

    sim_params = params['sim_params']
    sbi_params = params['sbi_params']
    plot_params = params['plot_params']
    data_params = params['data_params']
    neuron_params = params['neuron_params']

    analyse_and_plot(sbi_params, sim_params, neuron_params,
                     data_params, plot_params, data_folder,
                     plot_folder)
    plt.close('all')
