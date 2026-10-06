"""
Connectivity utilities
======================

Used by notebooks/example_usage_connectivity.ipynb and by
`cli/stwm.py connectivity*`.

All-to-all source-space connectivity with the DICS beamformer. Every function
receives the single ``config`` dictionary (``config.yaml``). Requires the
``conpy`` package (https://aaltoimaginglanguage.github.io/conpy/), imported
lazily so this module can be imported without it.

Workflow
--------
1. src_average                 (group, once)
2. new_morphing + new_morphed_forward_model   (per subject)
3. pairs_identification        (group, once)
4. connectivity_estimation + connectivity_vizualization   (per subject)
5. connectivity_statistics + connectivity_statistics_visualization   (group)

@author: Nikita Otstavnov, 2023 (refactored 2026)
"""

import os
import os.path as op

import numpy as np
import matplotlib.pyplot as plt
import mne


def _con_name(subject_name, freq_min, freq_max, condition):
    return '{}-connectivity for band from {} to {}_{}'.format(subject_name, freq_min,
                                                              freq_max, condition)


def _read_raw_info(path):
    return mne.io.read_raw_fif(path, allow_maxshield=False, preload=False,
                               on_split_missing='raise').info


# ============================================================================
# SOURCE SPACES AND FORWARD MODELS
# ============================================================================

def src_average(config):
    """Create the fsaverage template source space and check vertices are in sensor range."""
    import conpy

    folder = config['paths']['data_folder']
    cn     = config['connectivity']
    spacing = cn['spacing']

    os.chdir(folder)
    fsaverage = mne.setup_source_space('fsaverage', spacing=spacing,
                                       subjects_dir=config['paths']['subjects_dir'],
                                       n_jobs=config['processing']['n_jobs'], add_dist=False)
    mne.write_source_spaces('Sub_for_con_Avg-{}-src.fif'.format(spacing), fsaverage, overwrite=True)

    info  = _read_raw_info(op.join(folder, cn.get('average_ref_file', 'S1_filtered.fif')))
    trans = op.join(folder, cn.get('average_trans_file', 'Av-trans.fif'))
    conpy.select_vertices_in_sensor_range(fsaverage, dist=cn['max_sensor_dist'],
                                          info=info, trans=trans)
    return fsaverage


def new_morphing(config):
    """Morph the fsaverage template source space onto the subject."""
    folder       = config['paths']['data_folder']
    subject_name = config['subject']['subject_name']
    spacing      = config['connectivity']['spacing']

    os.chdir(folder)
    fsaverage   = mne.read_source_spaces('Sub_for_con_Avg-{}-src.fif'.format(spacing))
    subject_src = mne.morph_source_spaces(fsaverage, subject_name,
                                          subjects_dir=config['paths']['subjects_dir'])
    mne.write_source_spaces('{}_for_con-morph-src.fif'.format(subject_name),
                            subject_src, overwrite=True)
    return subject_src


def new_morphed_forward_model(config):
    """Forward model of the morphed subject source space, restricted to vertices in sensor range."""
    import conpy

    folder       = config['paths']['data_folder']
    subject_name = config['subject']['subject_name']
    fm           = config['forward_model']

    os.chdir(folder)
    info  = _read_raw_info(os.path.join(folder, config['subject']['file_name']))
    src   = mne.read_source_spaces('{}_for_con-morph-src.fif'.format(subject_name))
    trans = op.join(folder, '{}-trans.fif'.format(subject_name))

    verts   = conpy.select_vertices_in_sensor_range(src, dist=config['connectivity']['max_sensor_dist'],
                                                    info=info, trans=trans)
    src_sub = conpy.restrict_src_to_vertices(src, verts)

    bem_model = mne.make_bem_model(subject_name, ico=fm['ico'],
                                   subjects_dir=config['paths']['subjects_dir'],
                                   conductivity=tuple(fm['conductivity']))
    bem = mne.make_bem_solution(bem_model)

    fwd = mne.make_forward_solution(info, trans=trans, src=src_sub, bem=bem, meg=True,
                                    eeg=False, mindist=fm['mindist'],
                                    n_jobs=config['processing']['n_jobs'])
    mne.write_forward_solution('{}-for_con-morphed-fwd.fif'.format(subject_name), fwd,
                               overwrite=True)
    return fwd


def pairs_identification(config):
    """
    Shared vertices across subjects and all-to-all connectivity pairs
    (group level, run once). Saves Average-pairs.npy and
    S<i>-commonvertices-surf-fwd.fif.

    Returns
    -------
    pairs : list of two lists (fsaverage vertex indices)
    """
    import conpy

    folder       = config['paths']['data_folder']
    subjects_dir = config['paths']['subjects_dir']
    cn           = config['connectivity']
    index        = range(1, cn['num_subjects'] + 1)

    os.chdir(folder)
    src_surf_fs = mne.read_source_spaces('Sub_for_con_Avg-{}-src.fif'.format(cn['spacing']))

    fwd_ind = [conpy.forward_to_tangential(
        mne.read_forward_solution('S{}-for_con-morphed-fwd.fif'.format(i))) for i in index]
    fwd_ind[0] = conpy.restrict_forward_to_sensor_range(fwd_ind[0], cn['max_sensor_dist'])

    vert_inds = conpy.select_shared_vertices(fwd_ind, ref_src=src_surf_fs,
                                             subjects_dir=subjects_dir)
    fwd_common = [conpy.restrict_forward_to_vertices(fwd, v) for fwd, v in zip(fwd_ind, vert_inds)]
    for fwd_r, i in zip(fwd_common, index):
        mne.write_forward_solution('S{}-commonvertices-surf-fwd.fif'.format(i), fwd_r,
                                   overwrite=True)
    fwd_first = fwd_common[0]

    print('Computing connectivity pairs for all subjects...')
    pairs = conpy.all_to_all_connectivity_pairs(fwd_first, min_dist=cn['min_dist'])

    subj1_to_fsaverage = conpy.utils.get_morph_src_mapping(
        src_surf_fs, fwd_first['src'], indices=True, subjects_dir=subjects_dir)[1]
    pairs = [[subj1_to_fsaverage[v] for v in pairs[0]],
             [subj1_to_fsaverage[v] for v in pairs[1]]]
    np.save('Average-pairs', pairs)
    return pairs


# ============================================================================
# CONNECTIVITY ESTIMATION (per subject)
# ============================================================================

def connectivity_estimation(config):
    """
    DICS connectivity of both conditions for one subject over the configured band.

    Returns
    -------
    connectivity_1, connectivity_2 : list with one Connectivity object each
    """
    import conpy

    folder       = config['paths']['data_folder']
    subjects_dir = config['paths']['subjects_dir']
    subject_name = config['subject']['subject_name']
    cn           = config['connectivity']
    freq_min, freq_max = cn['freq_min'], cn['freq_max']

    os.chdir(folder)
    fsaverage = mne.read_source_spaces('Sub_for_con_Avg-{}-src.fif'.format(cn['spacing']))
    fwd_ind   = mne.read_forward_solution('S{}-commonvertices-surf-fwd.fif'.format(
        int(subject_name.replace('S', ''))))
    fwd_tan   = conpy.forward_to_tangential(fwd_ind)

    fsaverage_to_subj = conpy.utils.get_morph_src_mapping(
        fsaverage, fwd_ind['src'], indices=True, subjects_dir=subjects_dir)[0]
    pairs = np.load('Average-pairs.npy')
    pairs = [[fsaverage_to_subj[v] for v in pairs[0]],
             [fsaverage_to_subj[v] for v in pairs[1]]]

    result = []
    for key in ('condition_1', 'condition_2'):
        cond = config['conditions'][key]['name']
        csd  = mne.time_frequency.read_csd('{}_{}_csd.h5'.format(subject_name, cond))
        csd  = csd.mean(fmin=freq_min, fmax=freq_max)
        con  = conpy.dics_connectivity(vertex_pairs=pairs, fwd=fwd_tan, data_csd=csd,
                                       reg=cn['regularization'],
                                       n_jobs=config['processing']['n_jobs'])
        con.save(_con_name(subject_name, freq_min, freq_max, cond))
        result.append([con])

    return result[0], result[1]


def connectivity_vizualization(config):
    """
    Contrast (condition 1 - condition 2) of one subject: adjacency matrix,
    parcellated circle plot and cortical surface.

    Returns
    -------
    p : parcellated Connectivity, brain : Brain
    """
    import conpy

    subjects_dir = config['paths']['subjects_dir']
    subject_name = config['subject']['subject_name']
    condition_1  = config['conditions']['condition_1']['name']
    condition_2  = config['conditions']['condition_2']['name']
    cn           = config['connectivity']
    atlas        = cn['atlas']
    hemi         = cn.get('hemi', config.get('visualization', {}).get('hemi', 'both'))

    con_1 = conpy.read_connectivity(_con_name(subject_name, cn['freq_min'], cn['freq_max'], condition_1))
    con_2 = conpy.read_connectivity(_con_name(subject_name, cn['freq_min'], cn['freq_max'], condition_2))

    plt.figure()
    plt.imshow((con_1 - con_2).get_adjacency().toarray(), interpolation='nearest')

    labels = mne.read_labels_from_annot(subject_name, atlas, subjects_dir=subjects_dir)
    del labels[-1]
    p = con_1.parcellate(labels, 'degree', weight_by_degree=True)
    p.plot(n_lines=cn['n_lines'], vmin=cn.get('vmin'), vmax=cn.get('vmax'))
    plt.title('Strongest parcel-to-parcel connection', color='white')

    brain = (con_1 - con_2).make_stc('absmax').plot(subject_name, subjects_dir=subjects_dir,
                                                    hemi=hemi, figure=cn.get('figure'),
                                                    size=cn.get('size', 800))
    brain.add_annotation(atlas, borders=cn.get('borders', True))
    return p, brain


def run_connectivity_analysis(config):
    """Individual pipeline: morph source space -> forward model -> connectivity -> visualization."""
    subject_name = config['subject']['subject_name']
    cn           = config['connectivity']

    print(f"\n{'='*60}\nStarting Connectivity Analysis for Subject: {subject_name}\n{'='*60}\n")
    os.chdir(config['paths']['data_folder'])

    print("Step 1: Morphing fsaverage source space to subject...")
    new_morphing(config)
    print("\nStep 2: Creating morphed forward model...")
    new_morphed_forward_model(config)
    print(f"\nStep 3: Estimating DICS connectivity ({cn['freq_min']}-{cn['freq_max']} Hz)...")
    connectivity_estimation(config)
    print("\nStep 4: Visualizing connectivity contrast...")
    try:
        connectivity_vizualization(config)
    except Exception as e:
        print(f"  ⚠ Warning: Visualization failed: {e}")

    print(f"\n{'='*60}\nConnectivity Analysis Complete for: {subject_name}\n{'='*60}")


# ============================================================================
# GROUP STATISTICS
# ============================================================================

def _grand_average_contrast(config):
    """Load all subjects' connectivity, map to fsaverage, return (fsaverage, list_1, list_2, ga_1, ga_2)."""
    import conpy

    subjects_dir = config['paths']['subjects_dir']
    cn           = config['connectivity']

    os.chdir(config['paths']['data_folder'])
    fsaverage = mne.read_source_spaces('Sub_for_con_Avg-{}-src.fif'.format(cn['spacing']))

    per_cond, ga = [], []
    for key in ('condition_1', 'condition_2'):
        cond = config['conditions'][key]['name']
        cons = [conpy.read_connectivity(_con_name('S{}'.format(i), cn['freq_min'], cn['freq_max'], cond))
                .to_original_src(fsaverage, subjects_dir=subjects_dir)
                for i in range(1, cn['num_subjects'] + 1)]
        avg = cons[0].copy()
        for other in cons[1:]:
            avg += other
        avg /= len(cons)
        per_cond.append(cons)
        ga.append(avg)

    print('Averaged connectivity objects.')
    return fsaverage, per_cond[0], per_cond[1], ga[0], ga[1]


def _plot_stat_brain(con_clust, config):
    cn = config['connectivity']
    mne.viz.set_3d_options(depth_peeling=False, antialias=False, multi_samples=1)
    brain = con_clust.make_stc(cn['brain_mode']).plot(
        'fsaverage', subjects_dir=config['paths']['subjects_dir'], hemi=cn['hemi_stat'],
        figure=6, size=400, views=cn['views'])
    brain.add_annotation(cn['atlas'], borders=cn.get('borders', True))
    return brain


def connectivity_statistics(config):
    """
    Group cluster-permutation test between conditions. Saves Con_stat_*,
    Con_statistics-from_* (HDF5) and Con_statistics-*_contr.

    Returns
    -------
    connection_indices, bundles, bundle_ts, bundle_ps, H0, contrast
    """
    import conpy
    from h5io import write_hdf5

    cn = config['connectivity']
    freq_min, freq_max = cn['freq_min'], cn['freq_max']

    fsaverage, con_1_av, con_2_av, ga_1, ga_2 = _grand_average_contrast(config)
    contrast = ga_1 - ga_2

    connection_indices, bundles, bundle_ts, bundle_ps, H0 = conpy.cluster_permutation_test(
        con_1_av, con_2_av, cluster_threshold=cn['cluster_threshold'], src=fsaverage,
        n_permutations=cn['n_permutations'], verbose=True, alpha=cn['alpha'],
        tail=cn['tail'], n_jobs=config['processing']['n_jobs'], seed=cn['seed'],
        return_details=True, max_spread=cn['max_spread'])

    con_clust = contrast[connection_indices]
    con_clust.save('Con_stat_from_{}_to_{}'.format(freq_min, freq_max))
    write_hdf5('Con_statistics-from_{}_to_{}'.format(freq_min, freq_max),
               dict(connection_indices=connection_indices, bundles=bundles,
                    bundle_ts=bundle_ts, bundle_ps=bundle_ps, H0=H0), overwrite=True)

    labels = mne.read_labels_from_annot('fsaverage', cn['atlas'], cn['hemi_stat'],
                                        subjects_dir=config['paths']['subjects_dir'])
    del labels[-1]  # drop 'unknown' label
    con_parc = con_clust.parcellate(labels, summary=cn['summary'], weight_by_degree=False)
    con_parc.save('Con_statistics-{}_{}_contr'.format(freq_min, freq_max))

    _plot_stat_brain(con_clust, config)
    return connection_indices, bundles, bundle_ts, bundle_ps, H0, contrast


def connectivity_statistics_visualization(config):
    """
    Render saved group statistics: parcellated circle plot (optionally
    restricted by `regexp`) and the contrast on the cortical surface.

    Returns
    -------
    brain, con_parc
    """
    from h5io import read_hdf5

    cn = config['connectivity']
    _, _, _, ga_1, ga_2 = _grand_average_contrast(config)
    contrast = ga_2 - ga_1

    connection_indices = read_hdf5('Con_statistics-from_{}_to_{}'.format(
        cn['freq_min'], cn['freq_max'])).get('connection_indices')
    con_clust = contrast[connection_indices]

    selected_label = mne.read_labels_from_annot('fsaverage', hemi=cn.get('hemi', 'both'),
                                                regexp=cn.get('regexp'),
                                                subjects_dir=config['paths']['subjects_dir'])
    con_parc = con_clust.parcellate(selected_label, summary=cn['summary'],
                                    weight_by_degree=cn['weight_by_degree'])
    con_parc.plot(n_lines=cn['n_lines_stat'], vmin=cn.get('vmin_stat'), vmax=cn.get('vmax_stat'),
                  node_colors=[label.color for label in selected_label],
                  fontsize_names=cn['fontsize_names'], fontsize_colorbar=cn['fontsize_colorbar'])

    brain = _plot_stat_brain(con_clust, config)
    return brain, con_parc


# ============================================================================
# PUBLICATION FIGURES
# ============================================================================

def fig_6(data_path, folder_with_files, circ_file_name, brain_file_name, freq, vmin, vmax):
    """Circle plot + brain side by side with a colorbar; saves Circ_plus_brain.png."""
    from matplotlib import image as img
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    fig, ax = plt.subplots(1, 2, layout="constrained")
    fig.set_facecolor('black')
    ax[0].imshow(img.imread('{}.png'.format(circ_file_name)))
    ax[1].imshow(img.imread('{}.png'.format(brain_file_name)))
    ax[0].axis('off')
    ax[1].axis('off')

    cax  = fig.add_axes([0.9, 0.41, 0.005, 0.16])
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax),
                                       cmap=plt.colormaps["hot"]), cax=cax, ticks=[vmin, vmax])
    cbar.outline.set_visible(False)
    cax.yaxis.set_ticks_position('right')
    cbar.set_ticks([-0.1, 0.2])
    cax.tick_params(labelsize=12, colors='w')
    cbar.set_ticklabels(["0", "0.2"])
    plt.subplots_adjust(wspace=0, hspace=0)

    fig.savefig('Circ_plus_brain.png')
    return fig


def fig_merge(file_1, file_2, file_3, file_4, file_5, file_6):
    """Merge six panel images into a 4x2 grid."""
    from matplotlib import image as img

    fig, ax = plt.subplots(4, 2, sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.set_tight_layout(True)
    fig.set_facecolor('white')

    for (r, c), f in zip(((0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (3, 0)),
                         (file_1, file_2, file_3, file_4, file_5, file_6)):
        ax[r, c].imshow(img.imread(f))
        ax[r, c].axis('off')
    return fig
