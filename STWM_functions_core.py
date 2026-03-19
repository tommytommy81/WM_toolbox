"""
# FUNCTIONS File
# This file contains all functions relevant to source space analysis
# Only functions used in main path scripts are included
@author: Nikita Otstavnov, 2023
"""

# Standard library imports
import os
import os.path as op

# Third-party imports
import numpy as np
import mne

# MNE-specific imports
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc

# Scipy imports
from scipy import stats as stats


################################# SOURCE SPACE ANALYSIS

def creating_source_space_object(config): 
    """
    Create source space on cortical surface.
    
    Parameters
    ----------
    subject_name : str
        Subject name
    subjects_dir : str
        FreeSurfer subjects directory
    spacing : str
        Source space spacing (e.g., 'ico4', 'ico5', 'oct6')
    brain_surfaces : str
        Surface type
    folder : str
        Output folder
    n_jobs : int
        Number of parallel jobs
    orientation : str
        Source orientation constraint
        
    Returns
    -------
    src : SourceSpaces
        Source space object
    """
    
    folder          = config['paths']['data_folder']
    subjects_dir    = config['paths']['subjects_dir']
    folder_output   = config['paths']['output_folder']

    subject_name    = config['subject']['subject_name']
    spacing         = config['source_space']['spacing']
    brain_surfaces  = config['source_space']['surfaces']
    orientation     = config['source_space']['orientation']
    n_jobs          = config['processing']['n_jobs']
    
    #### SURFACE SOURCE SPACE
    src             = mne.setup_source_space(subject_name, spacing=spacing, 
                                             subjects_dir = subjects_dir, 
                                             n_jobs=n_jobs) 
    print(src)
    plot_bem_kwargs = dict(subject=subject_name, subjects_dir=subjects_dir, 
                           brain_surfaces=brain_surfaces, orientation=orientation,
                           slices=[50, 100, 150, 200])
    
    mne.viz.plot_bem(src=src, **plot_bem_kwargs)
    mne.write_source_spaces(os.path.join(folder_output, subject_name, '{}-{}-src.fif'.format(subject_name,spacing)), src, 
                            overwrite = True)
      
    return src


def creating_forward_model(config): 
    """
    Create forward solution (leadfield matrix).
    
    Parameters
    ----------
    folder : str
        Data folder
    subject_name : str
        Subject name
    spacing : str
        Source space spacing
    conductivity : tuple
        BEM conductivity values
    subjects_dir : str
        FreeSurfer subjects directory
    mindist : float
        Minimum distance from inner skull (mm)
    n_jobs : int
        Number of parallel jobs
    ico : int
        BEM surface decimation
    surfaces : str
        Surface type
    coord_frame : str
        Coordinate frame
        
    Returns
    -------
    bem : ConductorModel
        BEM solution
    fwd : Forward
        Forward solution
    """
    
    folder          = config['paths']['data_folder']
    subjects_dir    = config['paths']['subjects_dir']
    folder_output   = config['paths']['output_folder']
    subject_name    = config['subject']['subject_name']
    file_name       = config['subject']['file_name']
    
    spacing         = config['source_space']['spacing']
    n_jobs          = config['processing']['n_jobs']
    folder_output   = config['paths']['output_folder']
    conductivity    = tuple(config['forward_model']['conductivity'])
    ico             = config['forward_model']['ico']
    mindist         = config['forward_model']['mindist']
    surfaces        = config['forward_model']['surfaces']
    coord_frame     = config['forward_model']['coord_frame']


    src_surf            = mne.read_source_spaces(op.join(folder_output, subject_name, '{}-{}-src.fif'.format(subject_name,spacing)))
    trans               = op.join(folder, subject_name, file_name[:7]+'-trans.fif')
   
    SDFR                = op.join(folder, subject_name, file_name) 
    data                = mne.io.read_raw_fif(SDFR, allow_maxshield=False, 
                                              preload=False, on_split_missing='raise', 
                                              verbose=None)
    info                = data.info
 
    conductivity        = conductivity
    model               = mne.make_bem_model(subject='{}'.format(subject_name), ico=ico,
                                             conductivity=conductivity,
                                             subjects_dir=subjects_dir)
    bem                 = mne.make_bem_solution(model)
    
    mne.write_bem_solution(op.join(folder_output, subject_name, '{}-ind-bem-sol.fif'.format(subject_name)), bem, 
                                overwrite=True, verbose=None)
       
    fwd                 = mne.make_forward_solution(info, trans=trans,
                                                    src=src_surf, bem=bem,
                                                    meg=True, eeg=False, mindist=mindist,
                                                    verbose=True, n_jobs=n_jobs)
   
    mne.write_forward_solution(op.join(folder_output, subject_name, '{}-{}-surf-fwd.fif'.format(subject_name, spacing)), 
                               fwd, overwrite = True) 
    fig                = mne.viz.plot_alignment(subject=subject_name, 
                                                   subjects_dir=subjects_dir,
                                                   surfaces=surfaces, coord_frame=coord_frame,
                                                   src=src_surf)
    
    mne.viz.set_3d_view(fig, azimuth=173.78, elevation=101.75,
                        distance=0.40, focalpoint=(-0.03, -0.01, 0.03))
    mne.viz.set_3d_title(fig, title = '{}'.format(subject_name))
    return bem, fwd


def creating_source_estimate_object(config):
    """
    Compute source estimates using beamformer.
    
    Parameters
    ----------
    folder : str
        Data folder
    subject_name : str
        Subject name
    spacing : str
        Source space spacing
    condition_1, condition_2, condition_3 : str
        Condition names
    freq_min, freq_max : float
        Frequency band limits
    orientation : str
        Source orientation
    surf_ori : bool
        Surface-oriented sources
    force_fixed : bool
        Force fixed orientation
    reg : float
        Regularization parameter
    depth : float
        Depth weighting
    inversion : str
        Inversion method
        
    Returns
    -------
    stc_1, stc_2, stc_Ab : SourceEstimate
        Source estimates for each condition
    src_surf : SourceSpaces
        Source space
    """
    
    folder          = config['paths']['data_folder']
    output_folder   = config['paths']['output_folder']
    subject_name    = config['subject']['subject_name']
    file_name       = config['subject']['file_name']
    
    spacing         = config['source_space']['spacing']
    condition_1     = config['conditions']['condition_1']['name']
    condition_2     = config['conditions']['condition_2']['name']
    condition_3     = config['conditions'].get('baseline', {}).get('name', 'Baseline')
    freq_min        = config['source_estimate']['freq_min']
    freq_max        = config['source_estimate']['freq_max']
    orientation     = config['source_estimate']['orientation']
    surf_ori        = config['source_estimate']['surf_ori']
    force_fixed     = config['source_estimate']['force_fixed']
    reg             = config['source_estimate']['reg']
    depth           = config['source_estimate']['depth']
    inversion       = config['source_estimate']['method'] #dSPM, MNE 
    
    
    trans               = op.join(folder, subject_name, file_name[:7]+'-trans.fif')
     
    SDFR                = op.join(folder, subject_name, file_name) 
    data                = mne.io.read_raw_fif(SDFR, allow_maxshield=False, 
                                              preload=False, on_split_missing='raise', 
                                              verbose=None)
    info                = data.info
        
    fwd_ind             = mne.read_forward_solution(op.join(output_folder, subject_name, '{}-{}-surf-fwd.fif'.format(subject_name,spacing))) 
    src_surf            = mne.read_source_spaces(op.join(output_folder, subject_name, '{}-{}-src.fif'.format(subject_name,spacing)))
        
    csd_1               = mne.time_frequency.read_csd(op.join(output_folder,  subject_name, '{}_{}_csd.h5'.format(subject_name, condition_1)))
    csd_2               = mne.time_frequency.read_csd(op.join(output_folder,  subject_name, '{}_{}_csd.h5'.format(subject_name, condition_2)))      
    csd_Ab              = mne.time_frequency.read_csd(op.join(output_folder,  subject_name, '{}_baseline_csd.h5'.format(subject_name)))        
                                                   #free, fixed, tang
  
    # Average across frequencies first
    csd_to_use          = csd_1.copy()
    csd_to_use._data   += csd_2._data
    csd_to_use._data   /= 2 
    csd_dics            = csd_to_use.mean(fmin=freq_min, fmax=freq_max)  
        
    csd_1               = csd_1.mean(fmin=freq_min, fmax=freq_max)    
    csd_2               = csd_2.mean(fmin=freq_min, fmax=freq_max)    
    csd_Ab              = csd_Ab.mean(fmin=freq_min, fmax=freq_max)
    
    # Pick only gradiometers to avoid requiring noise covariance
    picks_grad          = mne.pick_types(info, meg='grad', eeg=False, exclude=[])
    info                = mne.pick_info(info, picks_grad)
    fwd_ind             = mne.pick_channels_forward(fwd_ind, info['ch_names'], ordered=True)
    csd_1               = csd_1.pick_channels(info['ch_names'], ordered=True)
    csd_2               = csd_2.pick_channels(info['ch_names'], ordered=True)
    csd_Ab              = csd_Ab.pick_channels(info['ch_names'], ordered=True)
    csd_dics            = csd_dics.pick_channels(info['ch_names'], ordered=True)
 
    fwd                 = mne.convert_forward_solution(fwd_ind, surf_ori=surf_ori         ,        
                                                       force_fixed=force_fixed, 
                                                       copy=True, 
                                                       use_cps=True, 
                                                       verbose=None)
    dics_filter         = mne.beamformer.make_dics(info,fwd_ind, csd_dics, reg=0.05, 
                                                        #  pick_ori='normal', 
                                                         inversion=inversion,           
                                                         weight_norm=None, 
                                                         real_filter=True, depth = depth,
                                                         rank='info') 
        
    print(dics_filter)
    dics_filter.save(op.join(output_folder, subject_name, '{}_from_{}_to_{}_{}_ind-dics.h5'.format(subject_name, 
                                                                 freq_min, 
                                                                 freq_max,
                                                                 orientation)), 
                     overwrite = True)
    
    stc_1,  freq_1      = mne.beamformer.apply_dics_csd(csd_1,  dics_filter)
    stc_2,  freq_2      = mne.beamformer.apply_dics_csd(csd_2,  dics_filter)
    stc_Ab, freq_Ab     = mne.beamformer.apply_dics_csd(csd_Ab, dics_filter)
    
    stc_1.save(op.join(output_folder, subject_name, '{}_from_{}_to_{}_{}_ind_surf_{}_stc'.format(subject_name, 
                                                                     freq_min, 
                                                                     freq_max, 
                                                                     orientation,
                                                                     condition_1)),
               overwrite=True) 
    stc_2.save(op.join(output_folder, subject_name, '{}_from_{}_to_{}_{}_ind_surf_{}_stc'.format(subject_name, 
                                                                      freq_min, 
                                                                      freq_max, 
                                                                      orientation,
                                                                      condition_2)),
               overwrite=True) 
    stc_Ab.save(op.join(output_folder, subject_name, '{}_from_{}_to_{}_{}_ind_surf_{}_stc'.format(  subject_name, 
                                                                    freq_min, 
                                                                    freq_max, 
                                                                    orientation,
                                                                    condition_3)),
                overwrite=True)
    
    return stc_1, stc_2, stc_Ab, src_surf


def source_estimate_morphing_to_average(config): 
    """
    Morph individual source estimates to fsaverage brain.
    
    Parameters
    ----------
    folder : str
        Data folder
    subject_name : str
        Subject name
    spacing : str
        Source space spacing
    freq_min, freq_max : float
        Frequency band limits
    orientation : str
        Source orientation
    condition_1, condition_2, condition_3 : str
        Condition names
    subjects_dir : str
        FreeSurfer subjects directory
        
    Returns
    -------
    stc_fs_1, stc_fs_2, stc_fs_A : SourceEstimate
        Morphed source estimates
    src_surf, src_surf_av : SourceSpaces
        Individual and average source spaces
    """
    
    folder          = config['paths']['data_folder']
    output_folder   = config['paths']['output_folder']
    subject_name    = config['subject']['subject_name']
    file_name       = config['subject']['file_name']
    subjects_dir    = config['paths']['subjects_dir']

    spacing         = config['source_space']['spacing']
    condition_1     = config['conditions']['condition_1']['name']
    condition_2     = config['conditions']['condition_2']['name']
    condition_3     = config['conditions'].get('baseline', {}).get('name', 'Baseline')
    freq_min        = config['source_estimate']['freq_min']
    freq_max        = config['source_estimate']['freq_max']
    orientation     = config['source_estimate']['orientation']

    
    stc_1        = mne.read_source_estimate(op.join(output_folder, subject_name, '{}_from_{}_to_{}_{}_ind_surf_{}_stc'.format(subject_name, 
                                                                                         freq_min, 
                                                                                         freq_max, 
                                                                                         orientation,
                                                                                         condition_1))) 
    stc_2        = mne.read_source_estimate(op.join(output_folder, subject_name, '{}_from_{}_to_{}_{}_ind_surf_{}_stc'.format(subject_name, 
                                                                                         freq_min, 
                                                                                         freq_max, 
                                                                                         orientation,
                                                                                         condition_2)))
    stc_A        = mne.read_source_estimate(op.join(output_folder, subject_name, '{}_from_{}_to_{}_{}_ind_surf_{}_stc'.format(subject_name,
                                                                                         freq_min, 
                                                                                         freq_max, 
                                                                                         orientation,
                                                                                         condition_3)))
                                        
    src_surf     = mne.read_source_spaces(op.join(output_folder, subject_name, '{}-{}-src.fif'.format(subject_name,spacing)))
    src_surf_av  = mne.read_source_spaces(op.join(output_folder, subject_name, 'Av-{}-src.fif'.format(spacing)))

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

    brain_before.save_image(op.join(output_folder, subject_name, '{}_from_{}_to_{}_{}_cont_before.png'.format(subject_name, 
                                                                     freq_min, 
                                                                     freq_max, 
                                                                     orientation)))
    
    morph_surf_1     = mne.compute_source_morph(stc_1_to_morph, 
                                     subject_from=subject_name, 
                                     subject_to = 'fsaverage', 
                                     src_to=src_surf_av,
                                     subjects_dir=subjects_dir, 
                                     smooth = 20,
                                     verbose=True)
    morph_surf_2     = mne.compute_source_morph(stc_2_to_morph, 
                                     subject_from=subject_name, 
                                     subject_to = 'fsaverage', 
                                     src_to=src_surf_av,
                                     subjects_dir=subjects_dir, smooth = 20,
                                     verbose=True)
    morph_surf_A     = mne.compute_source_morph(stc_A_to_morph, 
                                     subject_from=subject_name, 
                                     subject_to = 'fsaverage', 
                                     src_to=src_surf_av,
                                     subjects_dir=subjects_dir, smooth = 20,
                                     verbose=True)
    
    stc_fs_1         = morph_surf_1.apply(stc_1_to_morph) 
    stc_fs_2         = morph_surf_2.apply(stc_2_to_morph)
    stc_fs_A         = morph_surf_A.apply(stc_A_to_morph)    
   
    stc_fs_1.save(op.join(output_folder, subject_name, '{}_from_{}_to_{}_{}_morph_surf_{}_stc'.format(subject_name,
                                                                 freq_min, 
                                                                 freq_max, 
                                                                 orientation,
                                                                 condition_1)), 
                                                                 overwrite=True)
    stc_fs_2.save(op.join(output_folder, subject_name, '{}_from_{}_to_{}_{}_morph_surf_{}_stc'.format(subject_name,
                                                                 freq_min, 
                                                                 freq_max, 
                                                                 orientation,
                                                                 condition_2)), 
                                                                 overwrite=True)
    stc_fs_A.save(op.join(output_folder, subject_name, '{}_from_{}_to_{}_{}_morph_surf_{}_stc'.format(subject_name,
                                                                 freq_min, 
                                                                 freq_max, 
                                                                 orientation,
                                                                 condition_3)), 
                                                                 overwrite=True)              
    morph_surf_1.save(op.join(output_folder, subject_name, '{}_from_{}_to_{}_{}_morph_surf_{}.h5'.format(subject_name,
                                                                 freq_min, 
                                                                 freq_max, 
                                                                 orientation,
                                                                 condition_1)), 
                                                                 overwrite=True)
    morph_surf_2.save(op.join(output_folder, subject_name, '{}_from_{}_to_{}_{}_morph_surf_{}.h5'.format(subject_name,
                                                                 freq_min, 
                                                                 freq_max, 
                                                                 orientation,
                                                                 condition_2)), 
                                                                 overwrite=True)
    morph_surf_A.save(op.join(output_folder, subject_name, '{}_from_{}_to_{}_{}_morph_surf_{}.h5'.format(subject_name,
                                                                 freq_min, 
                                                                 freq_max, 
                                                                 orientation,
                                                                 condition_3)), 
                                                                 overwrite=True)
    
    return stc_fs_1, stc_fs_2, stc_fs_A, src_surf, src_surf_av


def source_estimate_visualization(config, stc, subject_name,
                                  freq_min, freq_max, spacing,
                                  hemi, colormap, time_label, alpha, time_viewer, 
                                  views, volume_options, view_layout, surface,
                                  annotation, mode, subjects_dir, backend,condition): 
    """
    Visualize source estimates on individual brain.
    
    Parameters
    ----------
    stc : SourceEstimate
        Source estimate
    subject_name : str
        Subject name
    freq_min, freq_max : float
        Frequency band
    spacing : str
        Source space spacing
    hemi : str
        Hemisphere ('lh', 'rh', 'both')
    colormap : str
        Colormap
    time_label : str
        Time label format
    alpha : float
        Transparency
    time_viewer : str or bool
        Time viewer option
    views : list
        View angles
    volume_options : float
        Volume visualization options
    view_layout : str
        Layout of views
    surface : str
        Surface type
    annotation : str
        Cortical parcellation
    mode : str
        Screenshot mode
    subjects_dir : str
        FreeSurfer subjects directory
    backend : str
        Visualization backend
    condition : str
        Condition name
        
    Returns
    -------
    brain_ind : Brain
        Brain visualization object
    """
    
    folder          = config['paths']['data_folder']
    output_folder   = config['paths']['output_folder']
    subject_name    = config['subject']['subject_name']
    file_name       = config['subject']['file_name']
    subjects_dir    = config['paths']['subjects_dir']
    
    mne.viz.set_3d_options(depth_peeling=False, antialias=False, multi_samples=1)   

    src       = mne.read_source_spaces(os.path.join(output_folder, subject_name, '{}-{}-src.fif'.format(subject_name,spacing)))
 
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
    
    mne.viz.Brain.save_image(brain_ind, filename=os.path.join(output_folder, subject_name, '{}_from_{}_to_{}_{}_{}.png'.format(subject_name,
                                                                      freq_min,freq_max, spacing,condition)), 
                             mode=mode)

    return brain_ind


def source_estimate_visualization_morph(stc, subject_name,
                                  freq_min, freq_max, spacing,
                                  hemi, colormap, time_label, alpha, time_viewer, 
                                  views, volume_options, view_layout, surface,
                                  annotation, mode, subjects_dir, backend,condition): 
    """
    Visualize morphed source estimates on fsaverage brain.
    
    Parameters
    ----------
    stc : SourceEstimate
        Morphed source estimate
    subject_name : str
        Original subject name (for labeling)
    [Other parameters same as source_estimate_visualization]
        
    Returns
    -------
    brain_ind : Brain
        Brain visualization object
    """
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
    mne.viz.Brain.save_image(brain_ind, filename='{}_from_{}_to_{}_{}_{}.png'.format(subject_name,
                                                                      freq_min,freq_max, spacing,condition), 
                             mode=mode)

    return brain_ind


def stc_merger(folder, num_subject, 
               subject_name,freq_min, freq_max, 
               orientation, condition_1, condition_2, condition_3,
               subjects_dir, spacing):
    """
    Merge morphed source estimates across subjects for group analysis.
    
    Parameters
    ----------
    folder : str
        Data folder
    num_subject : int
        Number of subjects
    subject_name : str
        Subject name pattern
    freq_min, freq_max : float
        Frequency band
    orientation : str
        Source orientation
    condition_1, condition_2, condition_3 : str
        Condition names
    subjects_dir : str
        FreeSurfer subjects directory
    spacing : str
        Source space spacing
        
    Returns
    -------
    stc_surf_1, stc_surf_2, stc_surf_a : list
        Lists of source estimates for each condition
    """
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


def source_estimate_average_visual_checher(stc_surf_1, stc_surf_2, stc_surf_a, 
                                   subject_to_visualize,freq_min,freq_max,spacing,
                                   hemi, colormap, time_label, alpha, time_viewer, 
                                   views, volume_options, view_layout, surface,
                                   annotation, mode, subjects_dir, backend): 
    """
    Visualize group average source estimate contrast.
    
    Parameters
    ----------
    stc_surf_1, stc_surf_2, stc_surf_a : list
        Lists of source estimates
    subject_to_visualize : int
        Subject index to visualize
    [Other parameters same as source_estimate_visualization]
        
    Returns
    -------
    brain_ind : Brain
        Brain visualization object
    """
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
    """
    Perform cluster-based permutation test on source estimates.
    
    Parameters
    ----------
    num_subject : int
        Number of subjects
    stc_s, stc_t, stc_a : list
        Lists of source estimates for each condition
    spacing : str
        Source space spacing
    folder : str
        Data folder
    subjects_dir : str
        FreeSurfer subjects directory
    p_threshold : float
        Cluster-forming threshold
    n_permutations : int
        Number of permutations
    tstep : float
        Time step
    n_jobs : int
        Number of parallel jobs
    out_type : str
        Output type ('mask' or 'indices')
    buffer_size : int or None
        Buffer size for computation
    alpha_level : float
        Significance level
        
    Returns
    -------
    stc_all_cluster_vis : SourceEstimate
        Visualization of all clusters
    stc_new : SourceEstimate
        Significant clusters only
    clu : tuple
        Cluster test results
    """
    src_fs   = mne.read_source_spaces(op.join(folder, 'Av-{}-src.fif'.format(spacing)))
   
    group_1 = []
    group_2 = []
    group_a = []
    
    i=1
    for i in range(len(stc_t)): 
        group_1.append(stc_s[i].data) 
        group_2.append(stc_t[i].data)
        group_a.append(stc_a[i].data)
      

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
    n_subjects     = len(stc_t)
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
    """
    Visualize statistical results on cortical surface.
    
    Parameters
    ----------
    stc_new : SourceEstimate
        Significant clusters
    freq_min, freq_max : float
        Frequency band
    spacing : str
        Source space spacing
    [Other parameters same as source_estimate_visualization]
        
    Returns
    -------
    brain_ind : Brain
        Brain visualization object
    """
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



################################# SENSOR SPACE ANALYSIS


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_preprocessed_data(file_path, meg_type='grad', exclude_channels=None):
    """
    Load preprocessed MEG data.
    
    Parameters
    ----------
    file_path : str
        Path to FIF file
    meg_type : str
        Type of MEG channels to load ('grad', 'mag', or 'all')
    exclude_channels : list or None
        List of channels to exclude
        
    Returns
    -------
    raw_data : mne.io.Raw
        Loaded raw data
    """
    raw_data = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    
    if exclude_channels is None:
        exclude_channels = []
    
    raw_data = raw_data.pick_types(meg=meg_type, exclude=exclude_channels)
    
    return raw_data


def extract_events(raw_data, stim_channel='STI101', min_duration=0.001):
    """
    Extract events from raw data.
    
    Parameters
    ----------
    raw_data : mne.io.Raw
        Raw MEG data
    stim_channel : str
        Name of stimulus channel
    min_duration : float
        Minimum event duration (seconds)
        
    Returns
    -------
    events : array
        Events array (n_events, 3)
    """
    events = mne.find_events(raw_data, stim_channel=stim_channel,
                            shortest_event=1, verbose=False)
    return events


def create_epochs(raw_data, events, event_id, tmin, tmax,
                 reject=None, flat=None, baseline=None, 
                 picks=None, preload=True):
    """
    Create epochs from raw data.
    
    Parameters
    ----------
    raw_data : mne.io.Raw
        Raw MEG data
    events : array
        Events array
    event_id : int or list
        Event ID(s) to epoch
    tmin, tmax : float
        Start and end time relative to event (seconds)
    reject : dict or None
        Rejection criteria
    flat : dict or None
        Flat detection criteria
    baseline : tuple or None
        Baseline period (tmin, tmax)
    picks : str, list, or None
        Channels to pick
    preload : bool
        Whether to preload data
        
    Returns
    -------
    epochs : mne.Epochs
        Epochs object
    """
    epochs = mne.Epochs(raw_data, events, event_id=event_id,
                       tmin=tmin, tmax=tmax,
                       reject=reject, flat=flat,
                       baseline=baseline, picks=picks,
                       preload=preload, verbose=False)
    return epochs



def compute_time_frequency(epochs, freqs, n_cycles=5, decim=1, 
                          n_jobs=1, return_itc=True):
    """
    Compute time-frequency representation using Morlet wavelets.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    freqs : array
        Frequencies of interest (Hz)
    n_cycles : float or array
        Number of cycles for Morlet wavelets
    decim : int
        Decimation factor
    n_jobs : int
        Number of parallel jobs
    return_itc : bool
        Whether to return inter-trial coherence
        
    Returns
    -------
    power : mne.time_frequency.AverageTFR
        Time-frequency power
    itc : mne.time_frequency.AverageTFR or None
        Inter-trial coherence (if return_itc=True)
    """
    power, itc = mne.time_frequency.tfr_morlet(
        epochs, n_cycles=n_cycles, return_itc=return_itc,
        freqs=freqs, decim=decim, n_jobs=n_jobs, verbose=False)
    
    if return_itc:
        return power, itc
    else:
        return power
    
    
def compute_csd(epochs, freqs, tmin=None, tmax=None, 
               n_cycles=5, decim=1, n_jobs=1):
    """
    Compute cross-spectral density using Morlet wavelets.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object
    freqs : array
        Frequencies of interest (Hz)
    tmin, tmax : float or None
        Time window for CSD computation
    n_cycles : float or array
        Number of cycles for Morlet wavelets
    decim : int
        Decimation factor
    n_jobs : int
        Number of parallel jobs
        
    Returns
    -------
    csd : mne.time_frequency.CrossSpectralDensity
        CSD object
    """
    csd = mne.time_frequency.csd_morlet(
        epochs, freqs, tmin=tmin, tmax=tmax,
        n_cycles=n_cycles, decim=decim, n_jobs=n_jobs, verbose=False)
    return csd



# ============================================================================
# FOOOF DECOMPOSITION
# ============================================================================

def apply_fooof_single_channel(spectrum, freqs, fm_settings=None):
    """
    Apply FOOOF to a single channel spectrum.
    
    Parameters
    ----------
    spectrum : array
        Power spectrum (n_freqs,)
    freqs : array
        Frequency values
    fm_settings : dict or None
        FOOOF settings
        
    Returns
    -------
    periodic : array
        Periodic (peak) component
    aperiodic : array
        Aperiodic component
    fm : FOOOF
        Fitted FOOOF model
    """
    from fooof import FOOOF
    from fooof.sim.gen import gen_aperiodic
    
    # Initialize FOOOF model
    if fm_settings is None:
        fm = FOOOF()
    else:
        fm = FOOOF(**fm_settings)
    
    # Fit model
    fm.fit(freqs, spectrum)
    
    # Extract components
    aperiodic = gen_aperiodic(fm.freqs, 
                             fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
    periodic = fm.power_spectrum - aperiodic
    
    return periodic, aperiodic, fm


def apply_fooof_multi_channel(spectrum, freqs, fm_settings=None):
    """
    Apply FOOOF to multi-channel spectra.
    
    Parameters
    ----------
    spectrum : array
        Power spectra (n_channels, n_freqs)
    freqs : array
        Frequency values
    fm_settings : dict or None
        FOOOF settings
        
    Returns
    -------
    periodic : array
        Periodic components (n_channels, n_freqs)
    aperiodic : array
        Aperiodic components (n_channels, n_freqs)
    """
    n_channels, n_freqs = spectrum.shape
    periodic = np.zeros((n_channels, n_freqs))
    aperiodic = np.zeros((n_channels, n_freqs))
    
    for ch in range(n_channels):
        per, aper, _ = apply_fooof_single_channel(spectrum[ch, :], freqs, 
                                                  fm_settings)
        periodic[ch, :] = per
        aperiodic[ch, :] = aper
    
    return periodic, aperiodic



