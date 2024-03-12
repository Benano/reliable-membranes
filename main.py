"""Performing sbi on simulated and real data."""
from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils import RestrictionEstimator, BoxUniform
import torch
import copy
import numpy as np
from utils import session_simple
from utils.session_simple import Session, per_neuron_trial_spikes
import pickle
import pandas as pd
from tqdm import tqdm
import func_timeout
import nest


# %% Simulation
def simulate_neuron(sim_params, neuron_params, simulate_voltage=False):
    """Simulate the firing rate of a neuron using the nest simulator."""
    # Set parameters
    sigma = abs(sim_params['std_I'])
    mu = sim_params['mu_I']

    # Simulation
    nest.set_verbosity("M_WARNING")
    nest.ResetKernel()
    nest.rng_seed = sim_params['seed']
    nest.resolution = sim_params['sim_res']

    # Neuron
    neurons = nest.Create("iaf_psc_exp", 1,
                          params=neuron_params)

    # Noise
    noise = nest.Create("noise_generator")
    noise.set({"mean": mu, "std": sigma,
               "dt": sim_params['dt_noise']})

    # Spike Detector
    spikedetector = nest.Create("spike_recorder")

    # Connections
    nest.Connect(noise, neurons)

    # Run without recording
    nest.Simulate(sim_params['dry_simtime'])

    # Connect spike & voltage recorder
    nest.Connect(neurons, spikedetector)
    if simulate_voltage:
        vm = nest.Create('voltmeter')
        nest.SetStatus(vm, {'interval': 1})
        nest.Connect(vm, neurons)

    # Run the simulation
    nest.Simulate(sim_params['simtime'])

    # Get Voltage Data
    if simulate_voltage:
        dmm = vm.get()
        Vms = dmm['events']['V_m']
        single_mem = np.array(Vms)
    else:
        single_mem = []

    # Get Spike Data
    dSD = spikedetector.get("events")
    ts_n = dSD["times"]
    single_ts = np.array(ts_n)

    return single_ts, single_mem


def compute_isi_stats(single_ts):
    """Calculate moments of isi distribution."""
    if len(single_ts) < 3:
        std = 0
        mu = 0
    elif len(single_ts) > 75:
        std = 0
        mu = 0
    else:
        isi = np.diff(single_ts)
        log_isi = np.log2(isi)
        mu = np.mean(log_isi)
        std = np.std(log_isi)

    isi_stats = [std, mu]

    return isi_stats


def recover_voltages_from_params(sim_params, neuron_params, in_params):
    """Simulate neurons to measure voltage stats from input params."""
    # Create array
    curr_sim_params = copy.deepcopy(sim_params)
    curr_sim_params['mu_I'] = in_params[1]
    curr_sim_params['std_I'] = in_params[0]
    curr_sim_params['simtime'] = 5_000
    curr_sim_params['seed'] = np.random.randint(1, 50000)
    ts, voltage = simulate_neuron(curr_sim_params,
                                  neuron_params,
                                  simulate_voltage=True)

    mean_mem = np.mean(voltage)
    std_mem = np.std(voltage)

    isi = np.diff(ts)
    isi_mean = np.mean(isi)
    isi_std = np.std(isi)

    voltage_stats = np.array([std_mem, mean_mem])
    isi_stats = np.array([isi_std, isi_mean])

    return voltage_stats, isi_stats


class sbi_simulator:
    """SBI simulator class."""

    def __init__(self, sbi_params, sim_params, neuron_params):
        """Init."""
        self.sbi_params = sbi_params
        self.sim_params = sim_params
        self.neuron_params = neuron_params

    def __call__(self, theta):
        """Run simulation."""
        theta = np.array(theta)[0]
        new_sim_params = copy.deepcopy(self.sim_params)
        new_sim_params['std_I'] = theta[0]
        new_sim_params['mu_I'] = theta[1]
        new_sim_params['seed'] = np.random.randint(1, 50000)
        neuron_params = copy.deepcopy(self.neuron_params)

        # Firing
        ts, _ = simulate_neuron(new_sim_params, neuron_params)

        # Stats
        isi_stats = compute_isi_stats(ts)
        isi_stats.append(len(ts))

        # Valid trial
        if isi_stats[1] == 0:
            observation = np.full(len(isi_stats), np.nan)
        else:
            observation = np.array(isi_stats)

        observation = np.array([observation.astype('float32')])

        return torch.tensor(observation)


def create_prior(sbi_params):
    """Create prior to be used for sbi."""
    low = []
    high = []
    for prior_param in ['std', 'mu']:
        bounds_name = 'prior_bounds_' + prior_param
        low.append(sbi_params[bounds_name][0])
        high.append(sbi_params[bounds_name][1])

    prior = BoxUniform(low=torch.asarray(low),
                       high=torch.asarray(high))

    return prior


def create_posterior(sbi_params, sim_params, neuron_params):
    """Create posterior via simulation based inference."""
    # Create simulator
    simulator = sbi_simulator(sbi_params,
                              sim_params, neuron_params)
    # Create Prior
    prior = create_prior(sbi_params)
    restriction_estimator = RestrictionEstimator(prior=prior)
    proposals = [prior]

    num_rounds = sbi_params['prior_nr_rounds']
    for i in range(num_rounds):
        # Run sbi
        theta, x = simulate_for_sbi(simulator, proposals[-1],
                                    sbi_params['prior_nr_sims'],
                                    num_workers=sbi_params['nr_workers'],
                                    show_progress_bar=True)

        restriction_estimator.append_simulations(theta, x)
        restriction_estimator.train()
        restricted_prior = restriction_estimator.restrict_prior()
        proposals.append(restricted_prior)

    # Main simulation
    theta, x = simulate_for_sbi(simulator,
                                proposals[-1],
                                sbi_params['nr_train_sims'],
                                num_workers=sbi_params['nr_workers'])
    restriction_estimator.append_simulations(theta, x)

    # Get all simulations
    all_theta, all_x, _ = restriction_estimator.get_simulations()

    # Simulate
    prior_samples, prior_obs = \
        simulate_for_sbi(simulator, proposals[-1],
                         sbi_params['validation_samples'],
                         num_workers=sbi_params['nr_workers'],
                         show_progress_bar=False)

    # Train posterior
    inference = SNPE(prior=prior)
    inference.append_simulations(all_theta, all_x).train()
    posterior = inference.build_posterior()

    return posterior, simulator, prior_samples, prior_obs


def sample_posterior(sbi_params, posterior, obs):
    """Sample the posterior with the observations from data."""
    sample_obs = posterior.sample((sbi_params['nr_posterior_samples'],),
                                  x=obs, show_progress_bars=False)
    return sample_obs


def run_w_timeout(f, args):
    """Timeout a function if it takes too long."""
    try:
        return func_timeout.func_timeout(2, f, args=args)
    except func_timeout.FunctionTimedOut:
        pass
    return []


def get_max_likelihood_from_posterior(sbi_params, posterior, isi_stats):
    """Sample the posterior and then pick likeliest parameters."""
    obs = isi_stats
    c_posterior = copy.deepcopy(posterior)
    c_posterior.set_default_x(obs)

    # Run with timeout in case there are some unsupported observations
    # These are given input parameters 0.5, 0.5 and can be filtered out
    f = sample_posterior
    args = (sbi_params,
            c_posterior,
            obs)
    temp_samples = run_w_timeout(f, args)

    if len(temp_samples) == 0:
        rec_params = np.array([0.5, 0.5], dtype='float32')
    else:
        prob = c_posterior.log_prob(temp_samples)
        top_prob = np.argmax(prob)
        rec_params = temp_samples[top_prob].numpy()

    return rec_params


def get_stats_from_obs(sbi_params, sim_params, neuron_params,
                       posterior, isi_stats):
    """Get all relevant stats."""
    in_params = get_max_likelihood_from_posterior(sbi_params, posterior,
                                                  isi_stats)
    voltage_stats, isi_stats = recover_voltages_from_params(sim_params,
                                                            neuron_params,
                                                            in_params)

    return in_params, voltage_stats, isi_stats


def create_trial_dataset(sim_params, data_params):
    """Create trial data set from all selected sessions."""
    # Loading Data

    areas = data_params['areas_to_fit']

    # Initialize a list to store rows
    rows = []
    # Iterate over areas
    for area in areas:
        sessions = session_simple.select_sessions(dataset='ChangeDetectionConflict',
                                            area=area,
                                            min_units=0,
                                            min_performance=None,
                                            filter_performance=False,
                                            exclude_muscimol=True,
                                            exclude_opto=True)

        # Iterate over sessions
        for sn, c_session in sessions.iterrows():
            print(f"Loading trials from session number {sn}")

            # Setting session parameters
            c_data_params = copy.deepcopy(data_params)
            animal_id = c_session['animal_id']
            session_id = c_session['session_id']
            c_data_params['area'] = area

            c_session = Session(dataset=data_params['dataset'],
                                animal_id=animal_id,
                                session_id=session_id,
                                subfolder='')

            c_session.load_data(load_lfp=False, use_newtrials=False)

            trial_numbers = c_session.select_trials()
            sel_trial_ind = np.isin(c_session.trial_data['trialNum'],
                                    trial_numbers)
            c_trial_data = c_session.trial_data.loc[sel_trial_ind, :]

            # Trial times
            sec_pre = sim_params['simtime']/1000
            sec_post = \
                (sim_params['simtime'] + data_params['skip_transient'])/1000

            trial_times = c_session.get_aligned_times(trial_numbers,
                                                      time_before_in_s=sec_pre,
                                                      time_after_in_s=sec_post,
                                                      event='stimChange')
            # Selecting neurons
            neuron_index = c_session.select_units(area=c_data_params['area'],
                                                  layer=None,
                                                  min_isolation_distance=10)

            spikes = c_session.spike_time_stamps[neuron_index]
            neuron_ts = per_neuron_trial_spikes(c_data_params, spikes,
                                                trial_times, neuron_index)

            # Iterate over the rows in session.trial_data (each trial)
            for trial_idx, trial_row in tqdm(c_trial_data.iterrows()):
                for neuron_idx in range(len(neuron_index)):

                    # Make a copy of the trial_row (Series)
                    new_row = trial_row.copy()
                    neuron_id = neuron_index[neuron_idx]

                    # Append neuron-specific information
                    new_row['spikes_pre'] = \
                        neuron_ts[neuron_idx][trial_idx][0]
                    new_row['spikes_transient'] = \
                        neuron_ts[neuron_idx][trial_idx][1]
                    new_row['spikes_post'] = \
                        neuron_ts[neuron_idx][trial_idx][2]
                    new_row['animal_id'] = animal_id
                    new_row['session_id'] = session_id
                    new_row['neuron_id'] = neuron_id
                    new_row['unique_nidx'] = \
                        f"{session_id}_{animal_id}__{neuron_id}"
                    new_row['unique_tidx'] = \
                        f"{session_id}_{animal_id}_{trial_idx}"
                    new_row['area'] = area

                    rows.append(new_row)

    # Convert the list of Series to a DataFrame
    df = pd.DataFrame(rows).reset_index(drop=True)

    return df


def perform_inference(sbi_params, sim_params,
                      data_params, neuron_params,
                      posterior, trial_dataset):
    """Infer parameters from all trials pre and post stimulus change."""
    # Perform inference
    # trials_to_fit = trial_dataset[trial_dataset['fit_trial']]
    # trials_to_fit.reset_index(drop=True, inplace=True)
    nr_trials = len(trial_dataset)
    for t_nr in tqdm(range(nr_trials)):

        c_trial = trial_dataset.loc[t_nr]

        # Test pre and post
        if c_trial['fit_trial']:

            # Perform Inference Pre
            isi_mu_pre = c_trial['log_isi_mu_pre']
            isi_std_pre = c_trial['log_isi_std_pre']
            spikes_pre = c_trial['nr_spikes_pre']
            isi_stats_pre = [isi_std_pre, isi_mu_pre]
            isi_stats_pre.append(spikes_pre)
            in_stats, voltage_stats, isi_stats = \
                get_stats_from_obs(sbi_params,
                                   sim_params,
                                   neuron_params,
                                   posterior,
                                   isi_stats_pre)
            # Saving
            # Inputs
            trial_dataset.at[t_nr, 'input_std_pre'] = in_stats[0]
            trial_dataset.at[t_nr, 'input_mu_pre'] = in_stats[1]
            # Voltages
            trial_dataset.at[t_nr, 'voltages_std_pre'] = voltage_stats[0]
            trial_dataset.at[t_nr, 'voltages_mu_pre'] = voltage_stats[1]
            # ISI
            trial_dataset.at[t_nr, 'rec_isi_std_pre'] = isi_stats[0]
            trial_dataset.at[t_nr, 'rec_isi_mu_pre'] = isi_stats[1]

            # Perform Inference Post
            isi_mu_post = c_trial['log_isi_mu_post']
            isi_std_post = c_trial['log_isi_std_post']
            spikes_post = c_trial['nr_spikes_post']
            isi_stats_post = [isi_std_post, isi_mu_post]
            isi_stats_post.append(spikes_post)
            in_stats, voltage_stats, isi_stats = \
                get_stats_from_obs(sbi_params,
                                   sim_params,
                                   neuron_params,
                                   posterior,
                                   isi_stats_post)
            # Saving
            # Inputs
            trial_dataset.at[t_nr, 'input_std_post'] = in_stats[0]
            trial_dataset.at[t_nr, 'input_mu_post'] = in_stats[1]
            # Voltages
            trial_dataset.at[t_nr, 'voltages_std_post'] = voltage_stats[0]
            trial_dataset.at[t_nr, 'voltages_mu_post'] = voltage_stats[1]
            # ISI
            trial_dataset.at[t_nr, 'rec_isi_std_post'] = isi_stats[0]
            trial_dataset.at[t_nr, 'rec_isi_mu_post'] = isi_stats[1]

    return trial_dataset


def prior_val(sbi_params, sim_params, neuron_params,
              posterior, prior_samples, prior_obs):
    """Validate the parameter recovery of parameters sampled from the prior."""
    nr_samples = sbi_params['validation_samples']
    prior_I_samples = np.array(prior_samples[0:nr_samples])
    prior_isi = np.array(prior_obs[0:nr_samples])

    real_I = []
    rec_I = []
    spikes = []

    for sample in tqdm(range(len(prior_I_samples))):

        # Simulate real_I
        c_real_I = prior_I_samples[sample]
        c_real_isi = prior_isi[sample]

        if c_real_isi[0] > 0:

            c_spikes = c_real_isi[2]
            x = np.array([c_real_isi])
            posterior.set_default_x(x)
            temp_samples = \
                posterior.sample((sbi_params['nr_posterior_samples'],),
                                 show_progress_bars=False)
            prob = posterior.log_prob(temp_samples)
            top_prob = np.argmax(prob)
            c_rec_I = temp_samples[top_prob].numpy()

            # Append everything to list
            real_I.append(c_real_I)
            rec_I.append(c_rec_I)
            spikes.append(c_spikes)

    spikes = np.array(spikes)
    real_I = np.array(real_I)
    rec_I = np.array(rec_I)
    diff_I = real_I - rec_I

    # Results
    results_validation = {
        'real_I': real_I,
        'rec_I': rec_I,
        'spikes': spikes,
        'diff_I': diff_I}

    return results_validation


def compute_full_isi_stats(single_ts):
    """Calculate moments of isi distribution."""
    nr_spikes = len(single_ts)
    if nr_spikes < 3:
        std = 0
        mu = 0
        log_std = 0
        log_mu = 0
        cv = 0
    else:
        isi = np.diff(single_ts)
        std = np.std(isi)
        mu = np.mean(isi)
        stats = compute_isi_stats(single_ts)
        log_std = stats[0]
        log_mu = stats[1]
        cv = std/mu
    return pd.Series([nr_spikes, std, mu, log_std, log_mu, cv])


def add_isi_stats(trial_dataset):
    """Add pre and post ISI stats for each trial in dataset."""
    # Add isi stats pre
    columns_pre = ['nr_spikes_pre', 'isi_std_pre',
                   'isi_mu_pre', 'log_isi_std_pre',
                   'log_isi_mu_pre', 'cv_pre']
    trial_dataset[columns_pre] = \
        trial_dataset['spikes_pre'].apply(compute_full_isi_stats)

    # Add isi stats post
    columns_post = ['nr_spikes_post', 'isi_std_post',
                    'isi_mu_post', 'log_isi_std_post',
                    'log_isi_mu_post', 'cv_post']
    trial_dataset[columns_post] = \
        trial_dataset['spikes_post'].apply(compute_full_isi_stats)

    # Add difference isi_stats
    trial_dataset['isi_std_diff'] = \
        trial_dataset['isi_std_post'] - trial_dataset['isi_std_pre']
    trial_dataset['isi_mu_diff'] = \
        trial_dataset['isi_mu_post'] - trial_dataset['isi_mu_pre']

    return trial_dataset


def identify_bursts(spike_train):
    """Identify bursting ISI in single spike train."""
    if len(spike_train) < 5:
        burstyness = 0

    else:
        isi = np.diff(spike_train)
        mean_isi = np.mean(isi)

        low_isi = isi[isi < mean_isi]
        ML = np.mean(low_isi)

        # Burstyness
        burst_isi = low_isi[low_isi < ML]
        nr_bursts = len(burst_isi)
        burstyness = nr_bursts/len(isi)

    return burstyness


def add_burstyness_measure(df):
    """Compute the burst measure for each trial period."""
    nr_trials = len(df)

    for pre_post in ["pre", "post"]:

        burstyness_list = []
        for n in range(nr_trials):
            test_train = df[f'spikes_{pre_post}'].iloc[n]
            burstyness = identify_bursts(test_train)
            burstyness_list.append(burstyness)
        df[f'burst_{pre_post}'] = burstyness_list

    return df


def add_change_scores_to_fit_trials(df):
    """Add change scores for fit trials."""
    # Inputs
    df['input_mu_diff'] = df['input_mu_post'] - df['input_mu_pre']
    df['input_std_diff'] = df['input_std_post'] - df['input_std_pre']
    # Voltages
    df['voltages_mu_diff'] = df['voltages_mu_post'] - df['voltages_mu_pre']
    df['voltages_std_diff'] = df['voltages_std_post'] - df['voltages_std_pre']

    return df


def add_condition_labels(df):
    """Add condition labels (modality & intensity) to df."""
    # Modality
    df['modality'] = np.zeros(len(df))
    aud = df['trialType'] == 'Y'
    vis = df['trialType'] == 'X'
    aud_vis = df['trialType'] == 'C'
    catch = df['trialType'] == 'P'

    df.loc[aud, 'modality'] = 'auditory'
    df.loc[vis, 'modality'] = 'visual'
    df.loc[aud_vis, 'modality'] = 'audio-visual'
    df.loc[catch, 'modality'] = 'catch'

    df['condition'] = np.zeros(len(df))

    AC = df['audioFreqChangeNorm']
    VC = df['visualOriChangeNorm']

    no_change = (AC == 1) & (VC == 1)
    thresh = ((AC == 2) & (VC == 1)) | ((AC == 1) & (VC == 2)) |\
        ((AC == 2) & (VC == 2)) | ((AC == 2) & (VC == 2))
    maximum = ((AC == 3) & (VC == 1)) | ((AC == 1) & (VC == 3)) |\
        ((AC == 3) & (VC == 3)) | ((AC == 3) & (VC == 3))
    both = ((AC == 3) & (VC == 2)) | ((AC == 2) & (VC == 3))

    df.loc[no_change, 'condition'] = 'no_change'
    df.loc[thresh, 'condition'] = 'thresh'
    df.loc[maximum, 'condition'] = 'max'
    df.loc[both, 'condition'] = 'mix'

    return df


def filter_and_flag_trials(sim_params, data_params, trial_dataset):
    """Flag trials for fitting."""
    # Good Spikes
    min_cutoff = (sim_params['simtime']/1000) * \
        data_params['min_fr_in_trial']
    max_cutoff = (sim_params['simtime']/1000) * \
        data_params['max_fr_in_trial']
    good_spikes_pre = \
        (trial_dataset['nr_spikes_pre'] > min_cutoff) & \
        (trial_dataset['nr_spikes_pre'] < max_cutoff)
    good_spikes_post = \
        (trial_dataset['nr_spikes_post'] > min_cutoff) & \
        (trial_dataset['nr_spikes_post'] < max_cutoff)
    trial_dataset['good_spikes'] = good_spikes_pre & good_spikes_post

    # Good trial
    trial_dataset['fit_trial'] = \
        trial_dataset['good_spikes']

    return trial_dataset


def save_thing(name, thing, save_folder, plot_folder):
    """Save something results."""
    for folder in [save_folder, plot_folder]:
        # Validation
        f_name = folder + "/" + name + ".pkl"
        with open(f_name, "wb") as handle:
            pickle.dump(thing, handle)


def load_or_create_posterior(run_params, sbi_params, sim_params,
                             data_params, neuron_params,
                             plot_params, save_folder,
                             data_folder):
    """Load or create posterior and sample prior parameters for validation."""
    if run_params['fresh_posterior']:
        posterior, simulator, prior_samples, prior_obs = \
            create_posterior(sbi_params, sim_params, neuron_params)
        save_thing('posterior', posterior, save_folder, data_folder)
        save_thing('simulator', simulator, save_folder, data_folder)
        save_thing('prior_obs', prior_obs, save_folder, data_folder)
        save_thing('prior_samples', prior_samples, save_folder, data_folder)
    else:
        try:
            # Simulator
            file_name = data_folder + '/simulator.pkl'
            unpickle_sim = open(file_name, 'rb')
            simulator = pickle.load(unpickle_sim)

            # Posterior
            file_name = data_folder + '/posterior.pkl'
            unpickle_post = open(file_name, 'rb')
            posterior = pickle.load(unpickle_post)

            # Prior_obs
            file_name = data_folder + '/prior_obs.pkl'
            unpickle_df = open(file_name, 'rb')
            prior_obs = pickle.load(unpickle_df)

            # Prior_samples
            file_name = data_folder + '/prior_samples.pkl'
            unpickle_df = open(file_name, 'rb')
            prior_samples = pickle.load(unpickle_df)

        except FileNotFoundError:
            print("""Did not find posterior ... Creating new one.""")
            posterior, simulator, prior_samples, prior_obs = \
                create_posterior(sbi_params, sim_params, neuron_params)
            save_thing('prior_obs', prior_obs, save_folder, data_folder)
            save_thing('prior_samples', prior_samples, save_folder, data_folder)
            save_thing('posterior', posterior, save_folder, data_folder)
            save_thing('simulator', simulator, save_folder, data_folder)

        else:
            print("Loading previous Posterior, Simulator & Prior")

    return posterior, simulator, prior_samples, prior_obs


def load_or_create_trial_dataset(run_params, sbi_params, sim_params,
                                 data_params, neuron_params,
                                 plot_params, save_folder,
                                 data_folder):
    """Load or create trial dataset."""
    # Create new trial dataset
    if run_params['fresh_trial_data']:
        print("creating new trial database")
        trial_dataset = create_trial_dataset(sim_params, data_params)
        trial_dataset = add_isi_stats(trial_dataset)
        trial_dataset = add_condition_labels(trial_dataset)
        save_thing('trial_data', trial_dataset, save_folder, data_folder)
    else:
        try:
            # Data
            file_name = data_folder + '/trial_data.pkl'
            unpickle_df = open(file_name, 'rb')
            trial_dataset = pickle.load(unpickle_df)
        except FileNotFoundError:
            print("""Did not find Trial Database ... Creating new one.""")
            trial_dataset = create_trial_dataset(sim_params,
                                                 data_params)
            trial_dataset = add_isi_stats(trial_dataset)
            trial_dataset = add_condition_labels(trial_dataset)
            save_thing('trial_data', trial_dataset, save_folder, data_folder)
        else:
            print("loaded already existing trial database")

    return trial_dataset


def infer_parameters_for_all_trials(run_params, sbi_params, sim_params,
                                    data_params, neuron_params,
                                    plot_params, save_folder,
                                    data_folder, posterior, trial_dataset):
    """Load or create posterior and sample prior parameters for validation."""
    if run_params['fresh_fitting']:
        print("Fitting all trials")
        fit_trial_dataset = perform_inference(sbi_params,
                                              sim_params,
                                              data_params,
                                              neuron_params,
                                              posterior,
                                              trial_dataset)
        fit_trial_dataset = add_change_scores_to_fit_trials(fit_trial_dataset)
        save_thing('fit_trial_data', fit_trial_dataset,
                   save_folder, data_folder)
    else:
        try:
            # Fit Data
            file_name = data_folder + '/fit_trial_data.pkl'
            unpickle_df = open(file_name, 'rb')
            fit_trial_dataset = pickle.load(unpickle_df)
        except FileNotFoundError:
            print("""Did not find Fitted Trials. Fitting new trials.""")
            fit_trial_dataset = perform_inference(sbi_params,
                                                  sim_params,
                                                  data_params,
                                                  neuron_params,
                                                  posterior,
                                                  trial_dataset)
            fit_trial_dataset = \
                add_change_scores_to_fit_trials(fit_trial_dataset)
            save_thing('fit_trial_data', fit_trial_dataset,
                       save_folder, data_folder)
        else:
            print("loaded already existing fit trial database")


def perform_validation(run_params, sbi_params, sim_params,
                       data_params, neuron_params,
                       plot_params, save_folder,
                       data_folder, posterior, prior_samples,
                       prior_obs):
    """Run validation procedure on prior parameters and observations."""
    if run_params['fresh_validation']:
        validation_results = prior_val(sbi_params, sim_params,
                                       neuron_params, posterior,
                                       prior_samples, prior_obs)

        save_thing('validation_results', validation_results,
                   save_folder, data_folder)
    else:
        pass


def main(run_params, sbi_params, sim_params,
         data_params, neuron_params,
         plot_params, save_folder, data_folder):
    """Perform Sbi."""
    # Setting seed
    np.random.seed(sim_params['numpy_seed'])
    _ = torch.manual_seed(sbi_params['sbi_seed'])

    posterior, simulator, prior_samples, prior_obs = \
        load_or_create_posterior(run_params, sbi_params, sim_params,
                                 data_params, neuron_params,
                                 plot_params, save_folder,
                                 data_folder)

    trial_dataset = \
        load_or_create_trial_dataset(run_params, sbi_params, sim_params,
                                     data_params, neuron_params,
                                     plot_params, save_folder,
                                     data_folder)

    trial_dataset = \
        filter_and_flag_trials(sim_params, data_params, trial_dataset)

    infer_parameters_for_all_trials(run_params, sbi_params, sim_params,
                                    data_params, neuron_params,
                                    plot_params, save_folder, data_folder,
                                    posterior, trial_dataset)

    perform_validation(run_params, sbi_params, sim_params,
                        data_params, neuron_params,
                        plot_params, save_folder,
                        data_folder, posterior,
                        prior_samples, prior_obs)
