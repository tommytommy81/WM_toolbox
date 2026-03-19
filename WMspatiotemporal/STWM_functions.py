"""
# FUNCTIONS File
# This file contain all functions relevant to sensor and source analysis
@author: Nikita Otstavnov, 2023
"""

# Standard library imports
import os
import os.path as op

# Third-party imports
import numpy as np
import scipy
import matplotlib.pyplot as plt
import mne

# MNE-specific imports
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs, corrmap
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc

# Scipy imports
from scipy import stats as stats

# FOOOF imports
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
from fooof.plts.spectra import plot_spectrum, plot_spectra


################################# SENSOR SPACE ANALYSIS
def perform_initial_analysis(n_components, random_state, max_iter, 
                             stim_channel,list_channels,
                             meg, exclude_channels, folder, 
                             file_name, subject_name, min_freqs, 
                             max_freqs, notch_freq, n_jobs):
    
    ### DOWNLOADING THE DATA 
    file_to_read     = os.path.join(folder, file_name) 
    raw_data         = mne.io.read_raw_fif(file_to_read, 
                                             allow_maxshield=False, 
                                             preload=False, 
                                             on_split_missing='raise', 
                                             verbose=None) 
    info             = raw_data.info
    raw_data.plot(title = 'Raw data')

    ### FILTERING
    raw_data.load_data() 
    list_channels    = list_channels

    #BAND-PASS FILTERING
   
    raw_data         = raw_data.filter(l_freq=min_freqs, 
                                       h_freq=max_freqs,
                                       n_jobs = n_jobs) 
  
    #NOTCH FILTERING
    raw_data         = raw_data.notch_filter(freqs=notch_freq, 
                                            # picks=list_channels, 
                                             method='spectrum_fit', 
                                             filter_length='10s',
                                             n_jobs=n_jobs)

    fig_1            = raw_data.plot_psd(fmax=max_freqs, 
                                         average=True,
                                         n_jobs=n_jobs)
    raw_data.plot(title = 'Notch + Bandpass data')

    ### ICA 
    ica              = mne.preprocessing.ICA(n_components=n_components, 
                                             random_state=random_state, max_iter=max_iter)
    ica_filt_raw     = ica.fit(raw_data)  
    # ica.plot_sources(raw_data, show_scrollbars=False)
    ica.plot_components(sensors = True, colorbar = True, 
                        title = 'ICA components', 
                        outlines = 'head')
        
    ica.exclude      = []
    eog_indices, eog_scores = ica.find_bads_eog(raw_data)
    ica.exclude      = eog_indices

    ecg_indices, ecg_scores = ica.find_bads_ecg(raw_data, 
                                                method='correlation', 
                                                threshold='auto')
    ica.exclude      = ica.exclude + ecg_indices
    
    #SAVER
    ica.save(op.join(folder, '{}-ica.fif'.format(subject_name)), overwrite=True)
    
    return raw_data, ica, ica.exclude 


# #### ICA APPLY
def ica_apply(data, ica, subject_name):
    ica.apply(data )
    data.plot()
    
    return data


def event_renaming(data, stim_channel, list_channels, 
                   subject_name, meg):

    a = 0
    i = 0
    
    event            = mne.find_events(data, 
                                       shortest_event=1, 
                                       stim_channel=stim_channel)
    
    data.plot(events=event, title = 'Original events')
    
    while i < len(event[:,2]):     
        if event[i,2] == 155:
            a = i    
            if  event[a-1, 2] > 180 and event[a+1, 2] > 180:
                event[a,2] = 255
            else: 
                event[a,2] = 155
        i += 1 
    data.plot(events=event, title = 'New events')
    
    return data, event

def raw_data_saver(folder, data, subject_name, meg):
    
    #### TAKING ONLY PARTICULAR CHANNELS FOR FUTURE ANALYSIS
    data_2           = data.copy().pick_types(meg="grad", exclude=[])

    
    data_2.save(op.join(folder, '{}_filtered.fif'.format(subject_name)), overwrite=True)
    data_2.annotations.save(op.join(folder, '{}_-annotations.csv'.format(subject_name)),
                            overwrite = True)
    return data_2

# not used for now, but can be used for future analysis
def epochs_initialization(data, event, main_event_1, main_event_2, 
                          tmin_epo, tmax_epo, 
                          reject_criteria, flat_criteria, subject_name, 
                          condition_1, condition_2, ch_type):
    epochs_1         = mne.Epochs(data, event, event_id=main_event_1,             #155 - Delay Spatial
                                  tmin=tmin_epo, tmax=tmax_epo, 
                                  reject = reject_criteria, 
                                  flat=flat_criteria, 
                                  preload=True,
                                  picks = ch_type,
                                  baseline=(None,None))
    epochs_2         = mne.Epochs(data, event, event_id=main_event_2,             #255 - Delay Temporal
                                  tmin=tmin_epo, tmax=tmax_epo, 
                                  reject = reject_criteria, 
                                  flat=flat_criteria, 
                                  preload=True,
                                  picks = ch_type,
                                  baseline=(None,None))
    epochs_full      = mne.Epochs(data, event, event_id=[main_event_1, main_event_2],             #255 - Delay Temporal
                                  tmin=tmin_epo, tmax=tmax_epo, 
                                  reject = reject_criteria, 
                                  flat=flat_criteria, 
                                  preload=True,
                                  picks = ch_type,
                                  baseline=(None,None))
     
    epochs_1.save(op.join(folder, '{}_{}_epochs-epo.fif'.format(subject_name, condition_1)),  overwrite=True)
    epochs_2.save(op.join(folder, '{}_{}_epochs-epo.fif'.format(subject_name, condition_2)),  overwrite=True)
    epochs_full.save(op.join(folder, '{}_ave_epochs-epo.fif'.format(subject_name)),  overwrite=True)

    return epochs_1, epochs_2, epochs_full    

def evoked_data(epochs_1, epochs_2, subject_name, condition_1, condition_2): 
   
    evoked_1         = epochs_1.average()
    evoked_2         = epochs_2.average()

    fig_1            = evoked_1.plot(titles = 'Evoked data of {} for condition {}'.format(subject_name, condition_1))
    fig_2            = evoked_2.plot(titles = 'Evoked data of {} for condition {}'.format(subject_name, condition_2))
    
    fig_1.savefig(op.join(folder, 'Evoked data of {} for condition {}'.format(subject_name, condition_1))) 
    fig_2.savefig(op.join(folder, 'Evoked data of {} for condition {}'.format(subject_name, condition_2)))
  
    return evoked_1, evoked_2
                  


def fooof_merger(num_subjects, folder, condition_1, condition_2): 
    
    list_1_ped_ordered        =  []
    list_1_aper_ordered       =  []
    list_2_ped_ordered        =  []
    list_2_aper_ordered       =  []

    i = 0 
    for i in range(num_subjects):
        list_1_ped_ordered.append(np.load(op.join(folder,'S{}_{}_ped_crop.npy'.format(i+1,condition_1))))  
        list_1_aper_ordered.append(np.load(op.join(folder,'S{}_{}_aper_crop.npy'.format(i+1,condition_1))))
        list_2_ped_ordered.append(np.load(op.join(folder,'S{}_{}_ped_crop.npy'.format(i+1,condition_2))))
        list_2_aper_ordered.append(np.load(op.join(folder,'S{}_{}_aper_crop.npy'.format(i+1,condition_2))))
        
        
    list_1_ped_ordered_array  = np.array(list_1_ped_ordered)
    list_1_aper_ordered_array = np.array(list_1_aper_ordered)
    list_2_ped_ordered_array  = np.array(list_2_ped_ordered)
    list_2_aper_ordered_array = np.array(list_2_aper_ordered)
   
    np.save(op.join(folder,'list_{}_ped_crop.npy'.format(condition_1)), arr=list_1_ped_ordered_array)
    np.save(op.join(folder,'list_{}_aper_crop.npy'.format(condition_1)), arr=list_1_aper_ordered_array)
    np.save(op.join(folder,'list_{}_ped_crop.npy'.format(condition_2)), arr=list_2_ped_ordered_array)
    np.save(op.join(folder,'list_{}_aper_crop.npy'.format(condition_2)), arr=list_2_aper_ordered_array)

    return list_1_ped_ordered_array, list_1_aper_ordered_array, list_2_ped_ordered_array, list_2_aper_ordered_array


def sensor_statistics(folder, subject_name, condition_1, ch_type,
                      list_1_ped, list_2_ped, alpha,
                      threshold, n_permutations, tail, out_type):
    
    epochs = mne.read_epochs(op.join(folder,'{}_{}_epochs-epo.fif'.format(subject_name, 
                                                            condition_1)), 
                                                            preload=False)
    info                                   = epochs.info
    adj, ch_names                          = mne.channels.find_ch_adjacency(info, 
                                                                            ch_type= ch_type)

    #### DATA PREPARATION
    p_1                                    = list_1_ped 
    p_2                                    = list_2_ped

    print(p_1.shape)
    print(p_2.shape)
    a_1                                    = np.transpose(p_1, (0,2,1))
    a_2                                    = np.transpose(p_2, (0,2,1))
    print(a_1.shape)
    print(a_2.shape)

    #### STATISTICS
    p_threshold                            = threshold
    n_permutations                         = n_permutations
    obj                                    = a_2 - a_1
    #Non-parametric cluster-level paired t-test for spatio-temporal data.
    df                                     = len(a_1) - 1
    threshold                              = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)                        
    T_obs, clusters, cluster_p_values, H0  = mne.stats.spatio_temporal_cluster_1samp_test(obj, 
                                                     out_type=out_type, adjacency=adj, 
                                                     n_permutations=n_permutations,
                                                     threshold=threshold, tail=tail)

    #### STAT OUTCOMES FEATURES
    print(T_obs) 
    print(clusters)
    print(cluster_p_values) 
    print(np.shape(T_obs)) 
    print(np.shape(clusters))
    print(np.shape(cluster_p_values)) 
    print(np.shape(H0))

    ####### CHECKING WITH SIGNIFICANT LEVEL
    p_accept                              = alpha
    good_cluster_inds                     = np.where(cluster_p_values < p_accept)[0]
    print(good_cluster_inds)
    len(good_cluster_inds)
    print(cluster_p_values)
    
    T_obs_plot                            = 0  * np.ones_like(T_obs)
    for c, p_val                          in zip(clusters, cluster_p_values):
        if p_val                          <= p_accept:
             T_obs_plot[c]                = T_obs[c]
           
             
    fig, ax                               = plt.subplots(2, 1)
    ax[0].imshow(T_obs, aspect='auto', origin='lower', cmap='RdBu_r', 
            vmin=-5, vmax=5)     
    ax[1].imshow(T_obs_plot, aspect='auto', origin='lower', cmap='RdBu_r', 
            vmin=-5, vmax=5)     

    return T_obs, T_obs_plot, clusters, cluster_p_values 



def csd_calc(min_freq, max_freq, freq_res, folder, subject_name, 
             condition_1, condition_2, t_interest_min, t_interest_max, 
             n_cycles, decim, n_jobs): 
    
    frequencies = np.logspace(min_freq,max_freq, num=freq_res)
  
    epochs_1    = mne.read_epochs(op.join(folder, '{}_{}_epochs-epo.fif'.format(subject_name, 
                                                                condition_1)), preload=True)
    epochs_2    = mne.read_epochs(op.join(folder, '{}_{}_epochs-epo.fif'.format(subject_name, 
                                                                condition_2)), preload=True)

    csd_1       = mne.time_frequency.csd_morlet(epochs_1, frequencies, tmin=t_interest_min, 
                                                tmax=t_interest_max, 
                                                n_cycles=n_cycles, decim=decim, n_jobs=n_jobs)   
    
    csd_1.save(op.join(folder, '{}_{}_csd.h5'.format(subject_name, condition_1)), 
               overwrite=True, verbose=None)
    
    csd_2       = mne.time_frequency.csd_morlet(epochs_2, frequencies, 
                                                tmin=t_interest_min, 
                                                tmax=t_interest_max, n_cycles=n_cycles,
                                                decim=decim, n_jobs=n_jobs)
    csd_2.save(op.join(folder, '{}_{}_csd.h5'.format(subject_name, condition_2)), 
               overwrite=True, verbose=None)

    return csd_1, csd_2
   
    
def csd_average(min_freq, max_freq, freq_res, subject_name, 
                   condition_1, condition_2, folder, 
                   tbase_min, tbase_max, picks, n_cycles, 
                   decim, n_jobs, ch_type, flat_criteria,reject_criteria, 
                   main_event_1, main_event_2, event):
    
    frequencies = np.logspace(min_freq,max_freq, num=freq_res)
    
     
    epochs      = mne.read_epochs(op.join(folder, '{}_ave_epochs-epo.fif'.format(subject_name)),
                                  preload=True)

    csd_av      = mne.time_frequency.csd_morlet(epochs, frequencies, 
                                                tmin=tbase_min, tmax=tbase_max, 
                                                n_cycles=n_cycles, decim=decim, 
                                                n_jobs=n_jobs)

    csd_av.save(op.join(folder, '{}_average_base_csd.h5'.format(subject_name)), 
                 overwrite=True, verbose=None)   
    
    return csd_av
    

################################# SOURCE SPACE ANALYSIS
def visual_freesurfer_check(subject_name, folder, 
                            file_name, subjects_dir): 
    
    # raw data
    file_to_read       = os.path.join(folder, file_name) 
    raw_data           = mne.io.read_raw_fif(file_to_read, 
                                             allow_maxshield=False, 
                                             preload=False, 
                                             on_split_missing='raise', 
                                             verbose=None) 
    info               = raw_data.info
    
    # visualization
    trans              = op.join(folder, '{}-trans.fif'.format(subject_name))
    mne.viz.plot_alignment(info, trans, subject=subject_name, dig=True,
                           meg=['helmet', 'sensors'], 
                           subjects_dir=subjects_dir,
                           surfaces='head-dense')
    
def creating_source_space_object(subject_name, subjects_dir,
                                 spacing, brain_surfaces, folder, n_jobs, 
                                 orientation): 
    #### SURFACE SOURCE SPACE
    src             = mne.setup_source_space(subject_name, spacing=spacing, 
                                             subjects_dir = subjects_dir, 
                                             n_jobs=n_jobs) 
    print(src)
    plot_bem_kwargs = dict(subject=subject_name, subjects_dir=subjects_dir, 
                           brain_surfaces=brain_surfaces, orientation='coronal',
                           slices=[50, 100, 150, 200])
    
    mne.viz.plot_bem(src=src, **plot_bem_kwargs)
    mne.write_source_spaces(os.path.join(folder, '{}-{}-src.fif'.format(subject_name,spacing)), src, 
                            overwrite = True)
      
    return src

def creating_average_source_space(spacing, subjects_dir, folder): 
    
    src_avg                 = mne.setup_source_space('fsaverage', spacing = spacing,  
                                                     subjects_dir = subjects_dir)
    mne.write_source_spaces(os.path.join(folder, 'Av-{}-src.fif'.format(spacing)), src_avg, 
                            overwrite = True)
   
    return src_avg

def creating_forward_model(folder, subject_name, spacing, conductivity, 
                           subjects_dir, mindist, n_jobs, ico, surfaces,
                           coord_frame): 
    
    src_surf            = mne.read_source_spaces(op.join(folder, '{}-{}-src.fif'.format(subject_name,spacing)))
    trans               = op.join(folder, '{}-trans.fif'.format(subject_name))
   
    SDFR                = op.join(folder, 'S1_filtered.fif')
    data                = mne.io.read_raw_fif(SDFR, allow_maxshield=False, 
                                              preload=False, on_split_missing='raise', 
                                              verbose=None)
    info                = data.info
 
    conductivity        = conductivity
    model               = mne.make_bem_model(subject='{}'.format(subject_name), ico=ico,
                                             conductivity=conductivity,
                                             subjects_dir=subjects_dir)
    bem                 = mne.make_bem_solution(model)
    
    mne.write_bem_solution(op.join(folder, '{}-ind-bem-sol.fif'.format(subject_name)), bem, 
                                overwrite=True, verbose=None)
       
    fwd                 = mne.make_forward_solution(info, trans=trans,
                                                    src=src_surf, bem=bem,
                                                    meg=True, eeg=False, mindist=mindist,
                                                    verbose=True, n_jobs=n_jobs)
   
    mne.write_forward_solution(op.join(folder, '{}-{}-surf-fwd.fif'.format(subject_name, spacing)), 
                               fwd, overwrite = True) 
    fig                = mne.viz.plot_alignment(subject=subject_name, 
                                                   subjects_dir=subjects_dir,
                                                   surfaces=surfaces, coord_frame=coord_frame,
                                                   src=src_surf)
    
    mne.viz.set_3d_view(fig, azimuth=173.78, elevation=101.75,
                        distance=0.40, focalpoint=(-0.03, -0.01, 0.03))
    mne.viz.set_3d_title(fig, title = '{}'.format(subject_name))
    return bem, fwd

def creating_average_forward_model(folder, subject_name, spacing, conductivity, 
                           subjects_dir, mindist, n_jobs, ico, 
                           surfaces, coord_frame): 
  
    src_surf            = mne.read_source_spaces(op.join(folder, '{}-{}-src.fif'.format(subject_name,spacing)))
    trans               = op.join(folder, 'Av-trans.fif')
   
    SDFR                = op.join(folder, 'S1_filtered.fif')
    data                = mne.io.read_raw_fif(SDFR, allow_maxshield=False, 
                                              preload=False, on_split_missing='raise', 
                                              verbose=None)
    info                = data.info
   
    src_avg             = mne.read_source_spaces(op.join(folder, 'Av-{}-src.fif'.format(spacing)))
    conductivity        = conductivity            
    model               = mne.make_bem_model(subject='fsaverage', ico=ico,
                                             conductivity=conductivity, 
                                             subjects_dir=subjects_dir)
    bem                 = mne.make_bem_solution(model)
    trans               = op.join(folder, 'fsaverage-trans.fif')
    
    ## SURFACE SRC
    fwd_av              = mne.make_forward_solution(info, trans=trans, 
                                                    src=src_avg, bem=bem,  
                                                    meg=True, eeg=False, mindist=mindist,
                                                    verbose=True, n_jobs = n_jobs )
    mne.write_forward_solution(op.join(folder, 'Av-{}-surf-fwd.fif'.format(spacing)), 
                               fwd_av, overwrite = True) 
    
    fig                = mne.viz.plot_alignment(subject='Average', 
                                                   subjects_dir=subjects_dir,
                                                   surfaces=surfaces, coord_frame=coord_frame,
                                                   src=src_avg)
    
    mne.viz.set_3d_view(fig, azimuth=173.78, elevation=101.75,
                        distance=0.40, focalpoint=(-0.03, -0.01, 0.03))
    mne.viz.set_3d_title(fig, title = 'Average')

    return fwd_av

def creating_source_estimate_object(folder, subject_name, spacing,
                                    condition_1, condition_2, condition_3,
                                    freq_min, freq_max, orientation, 
                                    surf_ori, force_fixed, reg, depth, 
                                    inversion):

    trans               = op.join(folder, '{}-trans.fif'.format(subject_name))
     
    SDFR                = op.join(folder, 'S1_filtered.fif')
    data                = mne.io.read_raw_fif(SDFR, allow_maxshield=False, 
                                              preload=False, on_split_missing='raise', 
                                              verbose=None)
    info                = data.info
        
    fwd_ind             = mne.read_forward_solution(op.join(folder, '{}-{}-surf-fwd.fif'.format(subject_name,spacing))) 
    src_surf            = mne.read_source_spaces(op.join(folder, '{}-{}-src.fif'.format(subject_name,spacing)))
        
    csd_1               = mne.time_frequency.read_csd(op.join(folder, '{}_{}_csd.h5'.format(subject_name, condition_1)))
    csd_2               = mne.time_frequency.read_csd(op.join(folder, '{}_{}_csd.h5'.format(subject_name, condition_2)))      
    csd_Ab              = mne.time_frequency.read_csd(op.join(folder, '{}_average_base_csd.h5'.format(subject_name)))        
    freq_min            = freq_min
    freq_max            = freq_max   
    orientation         = orientation                                                    #free, fixed, tang
  
    csd_to_use          = csd_1.copy()
    csd_to_use._data   += csd_2._data
    csd_to_use._data   /= 2 
    csd_dics            = csd_to_use.mean(fmin=freq_min, fmax=freq_max)  
        

    csd_1               = csd_1.mean(fmin=freq_min, fmax=freq_max)    
    csd_2               = csd_2.mean(fmin=freq_min, fmax=freq_max)    
    csd_Ab              = csd_Ab.mean(fmin=freq_min, fmax=freq_max)
 
    fwd                 = mne.convert_forward_solution(fwd_ind, surf_ori=surf_ori         ,        
                                                       force_fixed=force_fixed, 
                                                       copy=True, 
                                                       use_cps=True, 
                                                       verbose=None)
    dics_filter         = mne.beamformer.make_dics(info,fwd_ind, csd_dics, reg=0.05, 
                                                         #pick_ori='normal',           
                                                         inversion=inversion,           
                                                         weight_norm=None, 
                                                         real_filter=True, depth = depth) 
        
    print(dics_filter)
    dics_filter.save(op.join(folder, '{}_from_{}_to_{}_{}_ind-dics.h5'.format(subject_name, 
                                                                 freq_min, 
                                                                 freq_max,
                                                                 orientation)), 
                     overwrite = True)
    
    stc_1,  freq_1      = mne.beamformer.apply_dics_csd(csd_1,  dics_filter)
    stc_2,  freq_2      = mne.beamformer.apply_dics_csd(csd_2,  dics_filter)
    stc_Ab, freq_Ab     = mne.beamformer.apply_dics_csd(csd_Ab, dics_filter)
    
    stc_1.save(op.join(folder, '{}_from_{}_to_{}_{}_ind_surf_{}_stc'.format(subject_name, 
                                                                     freq_min, 
                                                                     freq_max, 
                                                                     orientation,
                                                                     condition_1)),
               overwrite=True) 
    stc_2.save(op.join(folder, '{}_from_{}_to_{}_{}_ind_surf_{}_stc'.format(subject_name, 
                                                                      freq_min, 
                                                                      freq_max, 
                                                                      orientation,
                                                                      condition_2)),
               overwrite=True) 
    stc_Ab.save(op.join(folder, '{}_from_{}_to_{}_{}_ind_surf_{}_stc'.format(  subject_name, 
                                                                    freq_min, 
                                                                    freq_max, 
                                                                    orientation,
                                                                    condition_3)),
                overwrite=True)
    
    return stc_1, stc_2, stc_Ab, src_surf


def source_estimate_morphing_to_average(folder, subject_name, spacing, freq_min, 
                                        freq_max, orientation,
                                        condition_1, condition_2, condition_3, subjects_dir): 

    stc_1        = mne.read_source_estimate(op.join(folder, '{}_from_{}_to_{}_{}_ind_surf_{}_stc'.format(subject_name, 
                                                                                         freq_min, 
                                                                                         freq_max, 
                                                                                         orientation,
                                                                                         condition_1))) 
    stc_2        = mne.read_source_estimate(op.join(folder, '{}_from_{}_to_{}_{}_ind_surf_{}_stc'.format(subject_name, 
                                                                                         freq_min, 
                                                                                         freq_max, 
                                                                                         orientation,
                                                                                         condition_2)))
    stc_A        = mne.read_source_estimate(op.join(folder, '{}_from_{}_to_{}_{}_ind_surf_{}_stc'.format(subject_name,
                                                                                         freq_min, 
                                                                                         freq_max, 
                                                                                         orientation,
                                                                                         condition_3)))
                                        
    src_surf     = mne.read_source_spaces(op.join(folder, '{}-{}-src.fif'.format(subject_name,spacing)))
    src_surf_av  = mne.read_source_spaces(op.join(folder, 'Av-{}-src.fif'.format(spacing)))

    stc_1_to_morph   = stc_1      
    stc_2_to_morph   = stc_2
    stc_A_to_morph   = stc_A

    #PLOTTING WHAT WAS BEFORE THE MORPING - REQUIRE STC.EXPAND!!!
    stc_before       = (stc_1_to_morph - stc_2_to_morph)/stc_A_to_morph
    brain_before     = stc_before.plot(subject=subject_name, surface='inflated', 
                                        hemi='both',         colormap='auto', 
                                        time_label='auto',   smoothing_steps=10, 
                                        transparent=True,    alpha=1.0, time_viewer='auto', 
                                        subjects_dir=subjects_dir,    
                                        figure=None, 
                                        views=['dorsal', 'lateral', 'medial','ventral'], 
                                        colorbar=True, clim='auto', cortex='classic', 
                                        size=800, background='black', 
                                        foreground=None, initial_time=None, 
                                        time_unit='s', backend='auto', 
                                        spacing='oct6', title=None, 
                                        show_traces='auto', src=src_surf, 
                                        volume_options=1.0, view_layout='vertical', 
                                        add_data_kwargs=None, 
                                        brain_kwargs=None, verbose=None)

    brain_before.save_image(op.join(folder, '{}_from_{}_to_{}_{}_cont_before.png'.format(subject_name, 
                                                                     freq_min, 
                                                                     freq_max, 
                                                                     orientation)))
    
    morph_surf_1     = mne.compute_source_morph(stc_1_to_morph, 
                                     subject_from=subject_name, 
                                     subject_to = 'fsaverage', 
                                     src_to=src_surf_av, #src_to=src_fs, 
                                     subjects_dir=subjects_dir, 
                                     smooth = 20,
                                     # spacing = fsave_vertices, #verbose
                                     verbose=True)
    morph_surf_2     = mne.compute_source_morph(stc_2_to_morph, 
                                     subject_from=subject_name, 
                                     subject_to = 'fsaverage', 
                                     src_to=src_surf_av, #src_to=src_fs, 
                                     subjects_dir=subjects_dir, smooth = 20,
                                     # spacing = fsave_vertices, #verbose
                                     verbose=True)
    morph_surf_A     = mne.compute_source_morph(stc_A_to_morph, 
                                     subject_from=subject_name, 
                                     subject_to = 'fsaverage', 
                                     src_to=src_surf_av, #src_to=src_fs, 
                                     subjects_dir=subjects_dir, smooth = 20,
                                     # spacing = fsave_vertices, #verbose
                                     verbose=True)
    
    stc_fs_1         = morph_surf_1.apply(stc_1_to_morph) 
    stc_fs_2         = morph_surf_2.apply(stc_2_to_morph)
    stc_fs_A         = morph_surf_A.apply(stc_A_to_morph)    
   
    stc_fs_1.save(op.join(folder, '{}_from_{}_to_{}_{}_morph_surf_{}_stc'.format(subject_name,
                                                                 freq_min, 
                                                                 freq_max, 
                                                                 orientation,
                                                                 condition_1)), 
                                                                 overwrite=True)
    stc_fs_2.save(op.join(folder, '{}_from_{}_to_{}_{}_morph_surf_{}_stc'.format(subject_name,
                                                                 freq_min, 
                                                                 freq_max, 
                                                                 orientation,
                                                                 condition_2)), 
                                                                 overwrite=True)
    stc_fs_A.save(op.join(folder, '{}_from_{}_to_{}_{}_morph_surf_{}_stc'.format(subject_name,
                                                                 freq_min, 
                                                                 freq_max, 
                                                                 orientation,
                                                                 condition_3)), 
                                                                 overwrite=True)              
    morph_surf_1.save(op.join(folder, '{}_from_{}_to_{}_{}_morph_surf_{}.h5'.format(subject_name,
                                                                 freq_min, 
                                                                 freq_max, 
                                                                 orientation,
                                                                 condition_1)), 
                                                                 overwrite=True)
    morph_surf_2.save(op.join(folder, '{}_from_{}_to_{}_{}_morph_surf_{}.h5'.format(subject_name,
                                                                 freq_min, 
                                                                 freq_max, 
                                                                 orientation,
                                                                 condition_1)), 
                                                                 overwrite=True)
    morph_surf_A.save(op.join(folder, '{}_from_{}_to_{}_{}_morph_surf_{}.h5'.format(subject_name,
                                                                 freq_min, 
                                                                 freq_max, 
                                                                 orientation,
                                                                 condition_1)), 
                                                                 overwrite=True)
    
    return stc_fs_1, stc_fs_2, stc_fs_A, src_surf, src_surf_av


def stc_merger(folder, num_subject, 
               subject_name,freq_min, freq_max, 
               orientation, condition_1, condition_2, condition_3,
               subjects_dir, spacing):
 
    src_fs      = mne.read_source_spaces(op.join(folder, 'Av-{}-src.fif'.format(spacing)))
    stc_surf_1  = []
    stc_surf_2  = [] 
    stc_surf_a  = []
    
    index       = np.linspace(1, num_subject, num_subject, dtype = int)
    orientation = orientation

    i           = 1
    for i in index:
        stc_surf_1.append(mne.read_source_estimate(op.join(folder, 'S{}_from_{}_to_{}_{}_morph_surf_{}_stc'.format(i,
                                                                                              freq_min, 
                                                                                              freq_max, 
                                                                                              orientation,
                                                                                              condition_1)))) 
        stc_surf_2.append( mne.read_source_estimate(op.join(folder, 'S{}_from_{}_to_{}_{}_morph_surf_{}_stc'.format(i,
                                                                                              freq_min, 
                                                                                              freq_max, 
                                                                                              orientation,
                                                                                              condition_2))) )
        stc_surf_a.append(mne.read_source_estimate(op.join(folder, 'S{}_from_{}_to_{}_{}_morph_surf_{}_stc'.format(i,
                                                                                              freq_min, 
                                                                                              freq_max, 
                                                                                              orientation,
                                                                                              condition_3)))) 
        
                          
     
        
        
    return stc_surf_1, stc_surf_2, stc_surf_a


def source_estimate_visualization(stc, subject_name,
                                  freq_min, freq_max, spacing,
                                  hemi, colormap, time_label, alpha, time_viewer, 
                                  views, volume_options, view_layout, surface,
                                  annotation, mode, subjects_dir, backend,condition): 
 
    mne.viz.set_3d_options(depth_peeling=False, antialias=False, multi_samples=1)   

    src       = mne.read_source_spaces('{}-{}-src.fif'.format(subject_name,spacing))
 
    brain_ind = mne.viz.plot_source_estimates(stc, subject =subject_name, surface=surface, 
                                              hemi=hemi, colormap=colormap, 
                                              time_label = time_label, alpha=alpha, 
                                              time_viewer = time_viewer, 
                                              subjects_dir = subjects_dir, 
                                              figure=None, views=views,
                                              backend=backend, 
                                              spacing=spacing, title=subject_name, 
                                              show_traces=False, 
                                              src = src, 
                                              volume_options=volume_options, 
                                              view_layout=view_layout, 
                                              add_data_kwargs=None, 
                                              brain_kwargs=None, 
                                              verbose=None)
    brain_ind.add_annotation(annotation, borders=True)
    
    mne.viz.Brain.save_image(brain_ind, filename='{}_from_{}_to_{}_{}_{}.png'.format(subject_name,
                                                                      freq_min,freq_max, spacing,condition), 
                             mode=mode)

    return brain_ind

def source_estimate_visualization_morph(stc, subject_name,
                                  freq_min, freq_max, spacing,
                                  hemi, colormap, time_label, alpha, time_viewer, 
                                  views, volume_options, view_layout, surface,
                                  annotation, mode, subjects_dir, backend,condition): 
 
    mne.viz.set_3d_options(depth_peeling=False, antialias=False, multi_samples=1)   

    src_fs    = mne.read_source_spaces('Av-{}-src.fif'.format(spacing))

    brain_ind = mne.viz.plot_source_estimates(stc, subject ='fsaverage', surface=surface, 
                                              hemi=hemi, colormap=colormap, 
                                              time_label = time_label, alpha=alpha, 
                                              time_viewer = time_viewer, 
                                              subjects_dir = subjects_dir, 
                                              figure=None, views=views,
                                              backend=backend, 
                                              spacing=spacing, title=subject_name, 
                                              show_traces=False, 
                                              src = src_fs, 
                                              volume_options=volume_options, 
                                              view_layout=view_layout, 
                                              add_data_kwargs=None, 
                                              brain_kwargs=None, 
                                              verbose=None)
    brain_ind.add_annotation(annotation, borders=True)
    # brain_ind.add_volume_labels(labels=labels, legend = True)
    mne.viz.Brain.save_image(brain_ind, filename='{}_from_{}_to_{}_{}_{}.png'.format(subject_name,
                                                                      freq_min,freq_max, spacing,condition), 
                             mode=mode)

    return brain_ind


def source_estimate_average_visual_checher(stc_surf_1, stc_surf_2, stc_surf_a, 
                                   subject_to_visualize,freq_min,freq_max,spacing,
                                   hemi, colormap, time_label, alpha, time_viewer, 
                                   views, volume_options, view_layout, surface,
                                   annotation, mode, subjects_dir, backend): 
   
    mne.viz.set_3d_options(depth_peeling=False, antialias=False, multi_samples=1)   

    src_fs             = mne.read_source_spaces('Av-{}-src.fif'.format(spacing))

    stc_s_to_visualize = stc_surf_1[subject_to_visualize]
    stc_t_to_visualize = stc_surf_2[subject_to_visualize]
    stc_a_to_visualize = stc_surf_a[subject_to_visualize]

    stc_after          = (stc_s_to_visualize - stc_t_to_visualize) / stc_a_to_visualize
    brain_ind          = mne.viz.plot_source_estimates(stc_after, subject='fsaverage', surface=surface, 
                                                       hemi=hemi, colormap=colormap, 
                                                       time_label=time_label, alpha=alpha, 
                                                       time_viewer=time_viewer, subjects_dir=subjects_dir, figure=None, 
                                                       views=views,
                                                       backend=backend, spacing=spacing, title=subject_to_visualize, 
                                                       show_traces=False, 
                                                       src = src_fs, 
                                                       volume_options=volume_options, 
                                                       view_layout=view_layout, 
                                                       add_data_kwargs=None, 
                                                       brain_kwargs=None, 
                                                       verbose=None)
    brain_ind.add_annotation(annotation, borders=True)
 
    mne.viz.Brain.save_image(brain_ind, filename='{}_from_{}_to_{}.png'.format(subject_to_visualize,
                                                                      freq_min,freq_max), 
                             mode=mode)

    return brain_ind


def statistical_inference(num_subject, stc_s, 
                          stc_t, stc_a, 
                          spacing, folder, subjects_dir,
                          p_threshold, n_permutations, tstep,
                          n_jobs, out_type, buffer_size, alpha_level):
    
    src_fs   = mne.read_source_spaces(op.join(folder, 'Av-{}-src.fif'.format(spacing)))
   
    group_1 = []
    group_2 = []
    group_a = []
    
    i=1
    for i in range(len(stc_surf_2)): 
        group_1.append(stc_surf_1[i].data) 
        group_2.append(stc_surf_2[i].data)
        group_a.append(stc_surf_a[i].data)
      

    group_1 = np.array(group_1)
    group_2 = np.array(group_2)
    group_a = np.array(group_a)
   
    diff    = (group_2 - group_1)/group_a 
    STAT    = np.transpose(diff, [0, 2, 1])
    np.shape(STAT)                                                             #Sub / Freq band / source
        
    print('Computing adjacency.')
          
    adjacency      = mne.spatial_src_adjacency(src_fs)
    np.shape(adjacency)
    fsave_vertices = [s['vertno'] for s in src_fs]
    
    #### STATISTICS
    p_threshold    = p_threshold
    n_permutations = n_permutations
    n_subjects     = len(stc_surf_2)
    df             = n_subjects - 1  # degrees of freedom for the test
    t_threshold    = stats.distributions.t.ppf(1 - p_threshold / 2, df=df) 
    
    T_obs, clusters, cluster_p_values, H0 = clu = \
        spatio_temporal_cluster_1samp_test(STAT, adjacency=adjacency, n_jobs=n_jobs,
                                           threshold=t_threshold, buffer_size=buffer_size, 
                                           verbose=True, n_permutations=n_permutations, out_type=out_type)


    #SIGNIFICANCE
    good_clusters_idx   = np.where(cluster_p_values < alpha_level)[0]
    good_clusters       = [clusters[idx] for idx in good_clusters_idx]
    len(good_clusters)

    #FOR VISUALIZATION
    stc_all_cluster_vis = summarize_clusters_stc(clu, p_thresh =alpha_level, 
                                                 tmin=0,      
                                                 vertices=fsave_vertices, tstep=tstep,
                                                 subject='fsaverage')
    stc_new             = stc_all_cluster_vis
    stc_new             = stc_all_cluster_vis.crop(tmin=0, tmax=0)  
   
    diff_2              = np.mean(diff, axis = 0)
    stc_new.data        = stc_all_cluster_vis.data * diff_2

    type(stc_all_cluster_vis)

    return stc_all_cluster_vis, stc_new, clu

def stat_visualization(stc_new, freq_min, freq_max, spacing,
                       hemi, colormap, time_label, transparency, time_viewer, 
                       views, volume_options, view_layout, surface,
                       annotation, mode, subjects_dir, backend): 
   
    mne.viz.set_3d_options(depth_peeling=False, antialias=False, multi_samples=1)   

    src_fs    = mne.read_source_spaces('Av-{}-src.fif'.format(spacing))

    brain_ind = mne.viz.plot_source_estimates(stc_new, subject='fsaverage', 
                                              surface=surface, 
                                              hemi=hemi, colormap=colormap, 
                                              time_label=time_label, 
                                              alpha=transparency, 
                                              time_viewer=time_viewer, subjects_dir=subjects_dir, figure=None, 
                                              views=views,
                                              backend=backend, spacing=spacing, 
                                              title='{}-{} frequency range'.format(freq_min,freq_max), 
                                              show_traces=False, 
                                              src = src_fs, 
                                              volume_options=volume_options, 
                                              view_layout=view_layout, 
                                              add_data_kwargs=None, 
                                              brain_kwargs=None, 
                                              verbose=None,
                                              background = 'white')
                         
    brain_ind.add_annotation(annotation, borders=True)
  
    mne.viz.Brain.save_image(brain_ind, 
                             filename='Average_statistics_from_{}_to_{}.png'.format(freq_min,
                                                                                    freq_max), 
                             mode=mode)

    return brain_ind

