"""
Source-space utilities
======================

Used by notebooks/example_usage_source_complete.ipynb and
notebooks/example_usage_source_group_statistics.ipynb, and by
`cli/stwm.py source*`.

Contents
--------
- Individual level: source space, forward model, DICS source estimates,
  morphing to fsaverage, visualization, run_source_analysis
- Group level: stc_merger, statistical_inference, stat_visualization,
  run_source_group_statistics
- Publication figure: fig_5

@author: Nikita Otstavnov, 2023 (refactored 2026)
"""

import os
import os.path as op

import numpy as np
import mne
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc
from scipy import stats


def _stc_name(subject_name, freq_min, freq_max, orientation, kind, condition):
    """File stem of a saved source estimate, e.g. S1_from_8_to_12_fix_morph_surf_SiVe_stc."""
    return '{}_from_{}_to_{}_{}_{}_surf_{}_stc'.format(subject_name, freq_min, freq_max,
                                                       orientation, kind, condition)


def _conditions(config):
    return (config['conditions']['condition_1']['name'],
            config['conditions']['condition_2']['name'],
            config['conditions'].get('baseline', {}).get('name', 'Baseline'))


def _plot_stc(stc, subject, src, spacing, title, hemi, colormap, time_label, alpha,
              time_viewer, views, volume_options, view_layout, surface,
              annotation, mode, subjects_dir, backend, filename, **kwargs):
    """Plot a source estimate with an annotation and save a screenshot."""
    mne.viz.set_3d_options(depth_peeling=False, antialias=False, multi_samples=1)
    brain = mne.viz.plot_source_estimates(stc, subject=subject, surface=surface,
                                          hemi=hemi, colormap=colormap,
                                          time_label=time_label, alpha=alpha,
                                          time_viewer=time_viewer,
                                          subjects_dir=subjects_dir, figure=None,
                                          views=views, backend=backend,
                                          spacing=spacing, title=title,
                                          show_traces=False, src=src,
                                          volume_options=volume_options,
                                          view_layout=view_layout, **kwargs)
    brain.add_annotation(annotation, borders=True)
    mne.viz.Brain.save_image(brain, filename=filename, mode=mode)
    return brain


# ============================================================================
# INDIVIDUAL LEVEL
# ============================================================================

def creating_average_source_space(spacing, subjects_dir, folder):
    """Create the fsaverage source space (Av-<spacing>-src.fif) used for morphing / group stats."""
    src_avg = mne.setup_source_space('fsaverage', spacing=spacing, subjects_dir=subjects_dir)
    mne.write_source_spaces(os.path.join(folder, 'Av-{}-src.fif'.format(spacing)), src_avg,
                            overwrite=True)
    return src_avg


def creating_source_space_object(config):
    """Create and save the cortical-surface source space of the subject."""
    subjects_dir   = config['paths']['subjects_dir']
    folder_output  = config['paths']['output_folder']
    subject_name   = config['subject']['subject_name']
    spacing        = config['source_space']['spacing']
    n_jobs         = config['processing']['n_jobs']

    src = mne.setup_source_space(subject_name, spacing=spacing,
                                 subjects_dir=subjects_dir, n_jobs=n_jobs)
    print(src)
    mne.viz.plot_bem(src=src, subject=subject_name, subjects_dir=subjects_dir,
                     brain_surfaces=config['source_space']['surfaces'],
                     orientation=config['source_space']['orientation'],
                     slices=[50, 100, 150, 200])
    mne.write_source_spaces(os.path.join(folder_output, subject_name,
                                         '{}-{}-src.fif'.format(subject_name, spacing)),
                            src, overwrite=True)
    return src


def creating_forward_model(config):
    """Create BEM solution and forward model (leadfield). Returns (bem, fwd)."""
    folder        = config['paths']['data_folder']
    subjects_dir  = config['paths']['subjects_dir']
    folder_output = config['paths']['output_folder']
    subject_name  = config['subject']['subject_name']
    file_name     = config['subject']['file_name']
    spacing       = config['source_space']['spacing']
    n_jobs        = config['processing']['n_jobs']
    fm            = config['forward_model']
    out_dir       = op.join(folder_output, subject_name)

    src_surf = mne.read_source_spaces(op.join(out_dir, '{}-{}-src.fif'.format(subject_name, spacing)))
    trans    = op.join(folder, subject_name, file_name[:7] + '-trans.fif')
    info     = mne.io.read_raw_fif(op.join(folder, subject_name, file_name), allow_maxshield=False,
                                   preload=False, on_split_missing='raise').info

    model = mne.make_bem_model(subject=subject_name, ico=fm['ico'],
                               conductivity=tuple(fm['conductivity']),
                               subjects_dir=subjects_dir)
    bem   = mne.make_bem_solution(model)
    mne.write_bem_solution(op.join(out_dir, '{}-ind-bem-sol.fif'.format(subject_name)), bem,
                           overwrite=True)

    fwd = mne.make_forward_solution(info, trans=trans, src=src_surf, bem=bem,
                                    meg=True, eeg=False, mindist=fm['mindist'],
                                    verbose=True, n_jobs=n_jobs)
    mne.write_forward_solution(op.join(out_dir, '{}-{}-surf-fwd.fif'.format(subject_name, spacing)),
                               fwd, overwrite=True)

    fig = mne.viz.plot_alignment(subject=subject_name, subjects_dir=subjects_dir,
                                 surfaces=fm['surfaces'], coord_frame=fm['coord_frame'],
                                 src=src_surf)
    mne.viz.set_3d_view(fig, azimuth=173.78, elevation=101.75,
                        distance=0.40, focalpoint=(-0.03, -0.01, 0.03))
    mne.viz.set_3d_title(fig, title=subject_name)
    return bem, fwd


def creating_source_estimate_object(config):
    """
    DICS beamformer source estimates for condition 1, condition 2 and baseline,
    from the CSDs of the sensor-space analysis (gradiometers only).

    Returns
    -------
    stc_1, stc_2, stc_Ab, src_surf
    """
    folder        = config['paths']['data_folder']
    output_folder = config['paths']['output_folder']
    subject_name  = config['subject']['subject_name']
    file_name     = config['subject']['file_name']
    spacing       = config['source_space']['spacing']
    condition_1, condition_2, condition_3 = _conditions(config)
    se            = config['source_estimate']
    freq_min, freq_max, orientation = se['freq_min'], se['freq_max'], se['orientation']
    out_dir       = op.join(output_folder, subject_name)

    info     = mne.io.read_raw_fif(op.join(folder, subject_name, file_name), allow_maxshield=False,
                                   preload=False, on_split_missing='raise').info
    fwd_ind  = mne.read_forward_solution(op.join(out_dir, '{}-{}-surf-fwd.fif'.format(subject_name, spacing)))
    src_surf = mne.read_source_spaces(op.join(out_dir, '{}-{}-src.fif'.format(subject_name, spacing)))

    read_csd = lambda name: mne.time_frequency.read_csd(op.join(out_dir, '{}_{}_csd.h5'.format(subject_name, name)))
    csd_1, csd_2, csd_Ab = read_csd(condition_1), read_csd(condition_2), read_csd('baseline')

    # Common filter from the average of both conditions
    csd_to_use        = csd_1.copy()
    csd_to_use._data += csd_2._data
    csd_to_use._data /= 2
    csd_dics          = csd_to_use.mean(fmin=freq_min, fmax=freq_max)

    csd_1  = csd_1.mean(fmin=freq_min, fmax=freq_max)
    csd_2  = csd_2.mean(fmin=freq_min, fmax=freq_max)
    csd_Ab = csd_Ab.mean(fmin=freq_min, fmax=freq_max)

    # Pick only gradiometers to avoid requiring noise covariance
    info    = mne.pick_info(info, mne.pick_types(info, meg='grad', eeg=False, exclude=[]))
    fwd_ind = mne.pick_channels_forward(fwd_ind, info['ch_names'], ordered=True)
    csd_1, csd_2, csd_Ab, csd_dics = [c.pick_channels(info['ch_names'], ordered=True)
                                      for c in (csd_1, csd_2, csd_Ab, csd_dics)]

    dics_filter = mne.beamformer.make_dics(info, fwd_ind, csd_dics, reg=0.05,
                                           inversion=se['method'], weight_norm=None,
                                           real_filter=True, depth=se['depth'], rank='info')
    print(dics_filter)
    dics_filter.save(op.join(out_dir, '{}_from_{}_to_{}_{}_ind-dics.h5'.format(
        subject_name, freq_min, freq_max, orientation)), overwrite=True)

    stcs = []
    for csd, cond in ((csd_1, condition_1), (csd_2, condition_2), (csd_Ab, condition_3)):
        stc, _ = mne.beamformer.apply_dics_csd(csd, dics_filter)
        stc.save(op.join(out_dir, _stc_name(subject_name, freq_min, freq_max, orientation, 'ind', cond)),
                 overwrite=True)
        stcs.append(stc)

    return stcs[0], stcs[1], stcs[2], src_surf


def source_estimate_morphing_to_average(config):
    """
    Morph individual source estimates to fsaverage.

    Returns
    -------
    stc_fs_1, stc_fs_2, stc_fs_A, src_surf, src_surf_av
    """
    output_folder = config['paths']['output_folder']
    subject_name  = config['subject']['subject_name']
    subjects_dir  = config['paths']['subjects_dir']
    spacing       = config['source_space']['spacing']
    conditions    = _conditions(config)
    se            = config['source_estimate']
    freq_min, freq_max, orientation = se['freq_min'], se['freq_max'], se['orientation']
    out_dir       = op.join(output_folder, subject_name)

    stc_1, stc_2, stc_A = [mne.read_source_estimate(op.join(
        out_dir, _stc_name(subject_name, freq_min, freq_max, orientation, 'ind', c)))
        for c in conditions]
    src_surf    = mne.read_source_spaces(op.join(out_dir, '{}-{}-src.fif'.format(subject_name, spacing)))
    src_surf_av = mne.read_source_spaces(op.join(out_dir, 'Av-{}-src.fif'.format(spacing)))

    # Contrast before morphing
    stc_before   = (stc_1 - stc_2) / stc_A
    brain_before = stc_before.plot(subject=subject_name, surface='inflated', hemi='both',
                                   colormap='auto', time_label='auto', smoothing_steps=10,
                                   transparent=True, alpha=1.0, time_viewer='auto',
                                   subjects_dir=subjects_dir,
                                   views=['dorsal', 'lateral', 'medial', 'ventral'],
                                   colorbar=True, clim='auto', cortex='classic',
                                   size=800, background='black', time_unit='s',
                                   backend='auto', spacing='oct6', show_traces='auto',
                                   src=src_surf, volume_options=1.0, view_layout='vertical')
    brain_before.save_image(op.join(out_dir, '{}_from_{}_to_{}_{}_cont_before.png'.format(
        subject_name, freq_min, freq_max, orientation)))

    morphed = []
    for stc, cond in zip((stc_1, stc_2, stc_A), conditions):
        morph = mne.compute_source_morph(stc, subject_from=subject_name, subject_to='fsaverage',
                                         src_to=src_surf_av, subjects_dir=subjects_dir,
                                         smooth=20, verbose=True)
        stc_fs = morph.apply(stc)
        stem = _stc_name(subject_name, freq_min, freq_max, orientation, 'morph', cond)
        stc_fs.save(op.join(out_dir, stem), overwrite=True)
        morph.save(op.join(out_dir, stem[:-len('_stc')] + '.h5'), overwrite=True)
        morphed.append(stc_fs)

    return morphed[0], morphed[1], morphed[2], src_surf, src_surf_av


def source_estimate_visualization(config, stc, subject_name,
                                  freq_min, freq_max, spacing,
                                  hemi, colormap, time_label, alpha, time_viewer,
                                  views, volume_options, view_layout, surface,
                                  annotation, mode, subjects_dir, backend, condition):
    """Visualize a source estimate on the individual brain and save a PNG."""
    out_dir      = op.join(config['paths']['output_folder'], config['subject']['subject_name'])
    subject_name = config['subject']['subject_name']
    subjects_dir = config['paths']['subjects_dir']
    src = mne.read_source_spaces(op.join(out_dir, '{}-{}-src.fif'.format(subject_name, spacing)))
    return _plot_stc(stc, subject_name, src, spacing, subject_name, hemi, colormap, time_label,
                     alpha, time_viewer, views, volume_options, view_layout, surface,
                     annotation, mode, subjects_dir, backend,
                     filename=op.join(out_dir, '{}_from_{}_to_{}_{}_{}.png'.format(
                         subject_name, freq_min, freq_max, spacing, condition)))


def source_estimate_visualization_morph(stc, subject_name,
                                        freq_min, freq_max, spacing,
                                        hemi, colormap, time_label, alpha, time_viewer,
                                        views, volume_options, view_layout, surface,
                                        annotation, mode, subjects_dir, backend, condition):
    """Visualize a morphed source estimate on fsaverage (reads Av-<spacing>-src.fif from cwd)."""
    src_fs = mne.read_source_spaces('Av-{}-src.fif'.format(spacing))
    return _plot_stc(stc, 'fsaverage', src_fs, spacing, subject_name, hemi, colormap, time_label,
                     alpha, time_viewer, views, volume_options, view_layout, surface,
                     annotation, mode, subjects_dir, backend,
                     filename='{}_from_{}_to_{}_{}_{}.png'.format(
                         subject_name, freq_min, freq_max, spacing, condition))


def run_source_analysis(config):
    """Individual source pipeline: source space -> forward -> DICS -> visualization -> morph."""
    folder        = config['paths']['data_folder']
    output_folder = config['paths'].get('output_folder', folder)
    subjects_dir  = config['paths']['subjects_dir']
    subject_name  = config['subject']['subject_name']
    spacing       = config['source_space']['spacing']
    condition_1, condition_2, _ = _conditions(config)
    freq_min      = config['source_estimate']['freq_min']
    freq_max      = config['source_estimate']['freq_max']
    viz           = config.get('visualization', {})
    hemi          = viz.get('hemi', 'both')
    surface       = viz.get('surface', 'inflated')
    views         = viz.get('views', ['dorsal', 'lateral', 'medial', 'ventral'])
    viz_args      = (hemi, 'auto', 'auto', 0.5, 'auto', views, 1.0, 'vertical', surface,
                     'aparc.a2009s', 'rgb', subjects_dir, 'auto')

    print(f"\n{'='*60}\nStarting Source Space Analysis for Subject: {subject_name}\n{'='*60}\n")
    os.chdir(folder)

    print("Step 1: Creating source space...")
    creating_source_space_object(config)
    print(f"  ✓ Source space created with spacing: {spacing}")

    print("\nStep 2: Creating forward model...")
    creating_forward_model(config)
    print("  ✓ Forward model created")

    print("\nStep 3: Computing source estimates...")
    stc_1, stc_2, _, _ = creating_source_estimate_object(config)
    print(f"  ✓ {condition_1}: {stc_1.data.shape} | {condition_2}: {stc_2.data.shape} "
          f"| {freq_min}-{freq_max} Hz")

    print("\nStep 4: Visualizing source estimates...")
    try:
        source_estimate_visualization(config, stc_1, subject_name, freq_min, freq_max, spacing,
                                      *viz_args, condition_1)
        print("  ✓ Visualization saved")
    except Exception as e:
        print(f"  ⚠ Warning: Visualization failed: {e}")

    print("\nStep 5: Morphing to fsaverage...")
    try:
        stc_1_m, stc_2_m, stc_b_m, _, _ = source_estimate_morphing_to_average(config)
        brain_avg = source_estimate_visualization_morph((stc_1_m - stc_2_m) / stc_b_m,
                                                        subject_name, freq_min, freq_max, spacing,
                                                        *viz_args, 'Contrast')
        brain_avg.save_image(os.path.join(output_folder,
                                          f'{subject_name}_contrast_{freq_min}_to_{freq_max}.png'))
        print("  ✓ Morphed to fsaverage, contrast visualization saved")
    except Exception as e:
        print(f"  ⚠ Warning: Morphing/contrast visualization failed: {e}")

    print(f"\n{'='*60}\nSource Space Analysis Complete for: {subject_name}\n{'='*60}")
    print(f"Output files saved to: {output_folder}")


# ============================================================================
# GROUP LEVEL
# ============================================================================

def stc_merger(folder, num_subject,
               subject_name, freq_min, freq_max,
               orientation, condition_1, condition_2, condition_3,
               subjects_dir, spacing):
    """Load morphed STCs of subjects S1..S<num_subject>. Returns three lists (one per condition)."""
    return tuple([mne.read_source_estimate(op.join(
        folder, _stc_name('S{}'.format(i), freq_min, freq_max, orientation, 'morph', cond)))
        for i in range(1, num_subject + 1)]
        for cond in (condition_1, condition_2, condition_3))


def source_estimate_average_visual_checher(stc_surf_1, stc_surf_2, stc_surf_a,
                                           subject_to_visualize, freq_min, freq_max, spacing,
                                           hemi, colormap, time_label, alpha, time_viewer,
                                           views, volume_options, view_layout, surface,
                                           annotation, mode, subjects_dir, backend):
    """Visualize (cond1 - cond2) / baseline for one subject on fsaverage."""
    src_fs = mne.read_source_spaces('Av-{}-src.fif'.format(spacing))
    i = subject_to_visualize
    stc_after = (stc_surf_1[i] - stc_surf_2[i]) / stc_surf_a[i]
    return _plot_stc(stc_after, 'fsaverage', src_fs, spacing, subject_to_visualize, hemi,
                     colormap, time_label, alpha, time_viewer, views, volume_options,
                     view_layout, surface, annotation, mode, subjects_dir, backend,
                     filename='{}_from_{}_to_{}.png'.format(subject_to_visualize, freq_min, freq_max))


def statistical_inference(num_subject, stc_s, stc_t, stc_a,
                          spacing, folder, subjects_dir,
                          p_threshold, n_permutations, tstep,
                          n_jobs, out_type, buffer_size, alpha_level):
    """
    Spatio-temporal cluster permutation test on (stc_t - stc_s) / stc_a.

    Returns
    -------
    stc_all_cluster_vis, stc_new (clusters weighted by mean contrast), clu
    """
    src_fs = mne.read_source_spaces(op.join(folder, 'Av-{}-src.fif'.format(spacing)))

    group_1 = np.array([s.data for s in stc_s])
    group_2 = np.array([s.data for s in stc_t])
    group_a = np.array([s.data for s in stc_a])
    diff    = (group_2 - group_1) / group_a
    STAT    = np.transpose(diff, [0, 2, 1])   # subject / freq band / source

    print('Computing adjacency.')
    adjacency      = mne.spatial_src_adjacency(src_fs)
    fsave_vertices = [s['vertno'] for s in src_fs]

    df          = len(stc_t) - 1
    t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)

    T_obs, clusters, cluster_p_values, H0 = clu = \
        spatio_temporal_cluster_1samp_test(STAT, adjacency=adjacency, n_jobs=n_jobs,
                                           threshold=t_threshold, buffer_size=buffer_size,
                                           verbose=True, n_permutations=n_permutations,
                                           out_type=out_type)

    stc_all_cluster_vis = summarize_clusters_stc(clu, p_thresh=alpha_level, tmin=0,
                                                 vertices=fsave_vertices, tstep=tstep,
                                                 subject='fsaverage')
    stc_new      = stc_all_cluster_vis.crop(tmin=0, tmax=0)
    stc_new.data = stc_all_cluster_vis.data * np.mean(diff, axis=0)

    return stc_all_cluster_vis, stc_new, clu


def stat_visualization(stc_new, freq_min, freq_max, spacing,
                       hemi, colormap, time_label, transparency, time_viewer,
                       views, volume_options, view_layout, surface,
                       annotation, mode, subjects_dir, backend):
    """Render significant clusters on fsaverage and save Average_statistics_from_<f1>_to_<f2>.png."""
    src_fs = mne.read_source_spaces('Av-{}-src.fif'.format(spacing))
    return _plot_stc(stc_new, 'fsaverage', src_fs, spacing,
                     '{}-{} frequency range'.format(freq_min, freq_max), hemi, colormap,
                     time_label, transparency, time_viewer, views, volume_options,
                     view_layout, surface, annotation, mode, subjects_dir, backend,
                     filename='Average_statistics_from_{}_to_{}.png'.format(freq_min, freq_max),
                     background='white')


def run_source_group_statistics(config):
    """Group source pipeline: merge morphed STCs -> cluster test -> render statistics."""
    folder       = config['paths']['data_folder']
    subjects_dir = config['paths']['subjects_dir']
    spacing      = config['source_space']['spacing']
    se           = config['source_estimate']
    gs           = config['source_group_statistics']
    sv           = config['source_visualization']
    viz          = config['visualization']
    condition_1, condition_2, _ = _conditions(config)

    os.chdir(folder)
    stc_s, stc_t, stc_a = stc_merger(folder, gs['num_subjects'], config['subject']['subject_name'],
                                     se['freq_min'], se['freq_max'], se['orientation'],
                                     condition_1, condition_2, gs['condition_baseline'],
                                     subjects_dir, spacing)
    stc_all_cluster_vis, stc_new, clu = statistical_inference(
        gs['num_subjects'], stc_s, stc_t, stc_a, spacing, folder, subjects_dir,
        gs['p_threshold'], gs['n_permutations'], gs['tstep'], config['processing']['n_jobs'],
        gs['out_type'], gs['buffer_size'], gs['alpha_level'])
    print(f"Significant clusters: {np.sum(clu[2] < gs['alpha_level'])} / {len(clu[1])}")

    stat_visualization(stc_new, se['freq_min'], se['freq_max'], spacing,
                       viz['hemi'], sv['colormap'], sv['time_label'], sv['transparency'],
                       sv['time_viewer'], viz['views'], sv['volume_options'], sv['view_layout'],
                       viz['surface'], sv['annotation'], sv['mode'], subjects_dir, sv['backend'])
    return stc_all_cluster_vis, stc_new, clu


# ============================================================================
# PUBLICATION FIGURE
# ============================================================================

def fig_5(freq_min_1, freq_max_1, freq_min_2, freq_max_2, vmin, vmax):
    """
    Two-panel figure of source statistics images (from stat_visualization).
    Take vmin / vmax from the "Using control points [...]" printed by the brain viewer.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import image as img
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)

    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Theta frequency source estimate', fontdict={'fontsize': 13})
    ax[1].set_title('Beta frequency source estimate', fontdict={'fontsize': 13})
    for a, (f1, f2) in zip(ax, ((freq_min_1, freq_max_1), (freq_min_2, freq_max_2))):
        a.imshow(img.imread('Average_statistics_from_{}_to_{}.png'.format(f1, f2)))
        a.axis('off')

    vmean = vmin + (vmax - vmin) / 2
    norm  = Normalize(vmin=vmin, vmax=vmax)
    for pos in ([0.05, 0.4, 0.01, 0.2], [0.5, 0.4, 0.01, 0.2]):
        cax  = fig.add_axes(pos)
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=plt.colormaps["hot"]),
                            cax=cax, ticks=[vmin, vmax, vmean])
        cbar.outline.set_visible(False)
        cax.set_title("%", y=1.1)
        cax.tick_params(labelsize=10)
        cax.yaxis.set_ticks_position('right')

    return fig
