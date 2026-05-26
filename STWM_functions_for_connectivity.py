"""
Connectivity Functions
======================

All functions used by the connectivity pipeline of the Spatial-Temporal
Working Memory MEG project. Every function receives the single ``config``
dictionary (loaded from ``config.yaml``) and extracts its own parameters from
it, mirroring the convention used in ``STWM_functions_core.py``.

These functions implement all-to-all source-space connectivity with the DICS
beamformer and require the ``conpy`` package
(https://aaltoimaginglanguage.github.io/conpy/).

@author: Nikita Otstavnov, 2023 (refactored 2026)
"""


def src_average(config):
    """
    Create the fsaverage template source space used as the common reference
    for all subjects, and verify that vertices fall within sensor range.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Uses ``paths``, ``processing`` and
        ``connectivity`` (spacing, max_sensor_dist, average_ref_file,
        average_trans_file).

    Returns
    -------
    fsaverage : SourceSpaces
        Template (fsaverage) source space.
    """
    import os
    import os.path as op
    import mne
    import conpy

    folder        = config['paths']['data_folder']
    subjects_dir  = config['paths']['subjects_dir']
    n_jobs        = config['processing']['n_jobs']

    spacing       = config['connectivity']['spacing']
    dist          = config['connectivity']['max_sensor_dist']
    ref_file      = config['connectivity'].get('average_ref_file', 'S1_filtered.fif')
    trans_file    = config['connectivity'].get('average_trans_file', 'Av-trans.fif')

    os.chdir(folder)

    # Average (template) source space
    fsaverage = mne.setup_source_space('fsaverage', spacing=spacing,
                                       subjects_dir=subjects_dir,
                                       n_jobs=n_jobs, add_dist=False)
    mne.write_source_spaces('Sub_for_con_Avg-{}-src.fif'.format(spacing),
                            fsaverage, overwrite=True)

    # Average forward model info / coregistration
    trans = op.join(folder, trans_file)
    SDFR  = op.join(folder, ref_file)
    data  = mne.io.read_raw_fif(SDFR, allow_maxshield=False,
                                preload=False, on_split_missing='raise',
                                verbose=None)
    info  = data.info

    verts = conpy.select_vertices_in_sensor_range(fsaverage,
                                                  dist=dist,
                                                  info=info,
                                                  trans=trans)

    return fsaverage


def new_morphing(config):
    """
    Morph the fsaverage template source space onto an individual subject.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Uses ``paths``, ``subject`` and
        ``connectivity`` (spacing).

    Returns
    -------
    subject_src : SourceSpaces
        Subject-specific morphed source space.
    """
    import os
    import mne

    folder       = config['paths']['data_folder']
    subjects_dir = config['paths']['subjects_dir']
    subject_name = config['subject']['subject_name']
    spacing      = config['connectivity']['spacing']

    os.chdir(folder)

    fsaverage   = mne.read_source_spaces('Sub_for_con_Avg-{}-src.fif'.format(spacing))

    subject_src = mne.morph_source_spaces(fsaverage,
                                          '{}'.format(subject_name),
                                          subjects_dir=subjects_dir)

    mne.write_source_spaces('{}_for_con-morph-src.fif'.format(subject_name),
                            subject_src, overwrite=True)

    return subject_src


def new_morphed_forward_model(config):
    """
    Create the forward model for the morphed subject source space, restricted
    to vertices within sensor range.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Uses ``paths``, ``subject``, ``processing``,
        ``forward_model`` (ico, mindist, conductivity) and ``connectivity``
        (max_sensor_dist).

    Returns
    -------
    fwd : Forward
        Forward solution restricted to the in-range vertices.
    """
    import os
    import os.path as op
    import mne
    import conpy

    folder       = config['paths']['data_folder']
    subjects_dir = config['paths']['subjects_dir']
    subject_name = config['subject']['subject_name']
    file_name    = config['subject']['file_name']

    n_jobs       = config['processing']['n_jobs']
    dist         = config['connectivity']['max_sensor_dist']

    ico          = config['forward_model']['ico']
    mindist      = config['forward_model']['mindist']
    conductivity = tuple(config['forward_model']['conductivity'])

    os.chdir(folder)
    file_to_read = os.path.join(folder, file_name)
    raw_data     = mne.io.read_raw_fif(file_to_read,
                                       allow_maxshield=False,
                                       preload=False,
                                       on_split_missing='raise',
                                       verbose=None)
    info         = raw_data.info

    src          = mne.read_source_spaces('{}_for_con-morph-src.fif'.format(subject_name))
    trans        = op.join(folder, '{}-trans.fif'.format(subject_name))

    verts        = conpy.select_vertices_in_sensor_range(src,
                                                         dist=dist,
                                                         info=info,
                                                         trans=trans)
    src_sub      = conpy.restrict_src_to_vertices(src, verts)

    bem_model    = mne.make_bem_model('{}'.format(subject_name), ico=ico,
                                      subjects_dir=subjects_dir,
                                      conductivity=conductivity)
    bem          = mne.make_bem_solution(bem_model)

    fwd          = mne.make_forward_solution(info, trans=trans, src=src_sub,
                                             bem=bem,
                                             meg=True,
                                             eeg=False,
                                             mindist=mindist,
                                             n_jobs=n_jobs)

    mne.write_forward_solution('{}-for_con-morphed-fwd.fif'.format(subject_name),
                               fwd, overwrite=True)

    return fwd


def pairs_identification(config):
    """
    Identify the common set of vertices shared across all subjects and build
    the all-to-all connectivity pairs (group-level, run once).

    Parameters
    ----------
    config : dict
        Configuration dictionary. Uses ``paths`` and ``connectivity``
        (spacing, num_subjects, max_sensor_dist, min_dist).

    Returns
    -------
    pairs : list of two lists
        Vertex pairs expressed in fsaverage indices.
    """
    import os
    import mne
    import conpy
    import numpy as np

    folder          = config['paths']['data_folder']
    subjects_dir    = config['paths']['subjects_dir']
    spacing         = config['connectivity']['spacing']
    num_subject     = config['connectivity']['num_subjects']
    max_sensor_dist = config['connectivity']['max_sensor_dist']
    min_dist        = config['connectivity']['min_dist']

    os.chdir(folder)

    src_surf_fs = mne.read_source_spaces('Sub_for_con_Avg-{}-src.fif'.format(spacing))
    index       = np.linspace(1, num_subject, num_subject, dtype=int)
    fwd_ind     = []
    fwd1        = []

    for i in index:
        os.chdir(folder)
        fwd = mne.read_forward_solution('S{}-for_con-morphed-fwd.fif'.format(i))
        fwd_ind.append(conpy.forward_to_tangential(fwd))

    fwd_ind_3    = np.array(fwd_ind)
    fwd_first    = conpy.restrict_forward_to_sensor_range(fwd_ind_3[0], max_sensor_dist)
    fwd_ind_3[0] = fwd_first

    vert_inds   = conpy.select_shared_vertices(fwd_ind_3,
                                               ref_src=src_surf_fs,
                                               subjects_dir=subjects_dir)

    for fwd, vert_ind, i in zip(fwd_ind_3, vert_inds, index):
        fwd_r = conpy.restrict_forward_to_vertices(fwd, vert_ind)
        fwd1.append(fwd_r)
        mne.write_forward_solution('S{}-commonvertices-surf-fwd.fif'.format(i), fwd_r,
                                   overwrite=True)
        if i == index[0]:
            fwd_first = fwd_r

    print('Computing connectivity pairs for all subjects...')
    pairs = conpy.all_to_all_connectivity_pairs(fwd_first,
                                                min_dist=min_dist)

    subj1_to_fsaverage = conpy.utils.get_morph_src_mapping(src_surf_fs,
                                                           fwd_first['src'],
                                                           indices=True,
                                                           subjects_dir=subjects_dir)[1]
    pairs = [[subj1_to_fsaverage[v] for v in pairs[0]],
             [subj1_to_fsaverage[v] for v in pairs[1]]]
    np.save('Average-pairs', pairs)

    return pairs


def connectivity_estimation(config):
    """
    Estimate DICS source-space connectivity for both conditions of a single
    subject over the configured frequency band.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Uses ``paths``, ``subject``, ``processing``,
        ``conditions`` and ``connectivity`` (spacing, regularization,
        freq_min, freq_max).

    Returns
    -------
    connectivity_1, connectivity_2 : list
        Connectivity objects for condition 1 and condition 2.
    """
    import os
    import numpy as np
    import mne
    import conpy

    folder       = config['paths']['data_folder']
    subjects_dir = config['paths']['subjects_dir']
    subject_name = config['subject']['subject_name']
    n_jobs       = config['processing']['n_jobs']

    condition_1  = config['conditions']['condition_1']['name']
    condition_2  = config['conditions']['condition_2']['name']

    spacing      = config['connectivity']['spacing']
    reg          = config['connectivity']['regularization']
    freq_min     = config['connectivity']['freq_min']
    freq_max     = config['connectivity']['freq_max']

    subject_index = int(subject_name.replace('S', ''))

    os.chdir(folder)
    fsaverage = mne.read_source_spaces('Sub_for_con_Avg-{}-src.fif'.format(spacing))

    connectivity_1 = []
    connectivity_2 = []

    fwd_ind = mne.read_forward_solution('S{}-commonvertices-surf-fwd.fif'.format(subject_index))
    fwd_tan = conpy.forward_to_tangential(fwd_ind)
    pairs   = np.load('Average-pairs.npy')

    fsaverage_to_subj = conpy.utils.get_morph_src_mapping(fsaverage,
                                                          fwd_ind['src'],
                                                          indices=True,
                                                          subjects_dir=subjects_dir)[0]

    pairs = [[fsaverage_to_subj[v] for v in pairs[0]],
             [fsaverage_to_subj[v] for v in pairs[1]]]

    csd_1 = mne.time_frequency.read_csd('{}_{}_csd.h5'.format(subject_name, condition_1))
    csd_2 = mne.time_frequency.read_csd('{}_{}_csd.h5'.format(subject_name, condition_2))

    csd_1 = csd_1.mean(fmin=freq_min, fmax=freq_max)
    csd_2 = csd_2.mean(fmin=freq_min, fmax=freq_max)

    # Compute connectivity for the frequency band
    con_1 = conpy.dics_connectivity(vertex_pairs=pairs,
                                    fwd=fwd_tan,
                                    data_csd=csd_1,
                                    reg=reg,
                                    n_jobs=n_jobs)
    con_2 = conpy.dics_connectivity(vertex_pairs=pairs,
                                    fwd=fwd_tan,
                                    data_csd=csd_2,
                                    reg=reg,
                                    n_jobs=n_jobs)

    con_1.save('{}-connectivity for band from {} to {}_{}'.format(subject_name, freq_min,
                                                                  freq_max, condition_1))
    con_2.save('{}-connectivity for band from {} to {}_{}'.format(subject_name, freq_min,
                                                                  freq_max, condition_2))

    connectivity_1.append(con_1)
    connectivity_2.append(con_2)

    return connectivity_1, connectivity_2


def connectivity_vizualization(config):
    """
    Visualize the connectivity contrast (condition 1 - condition 2) for a
    single subject as a parcellated circle plot and on the cortical surface.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Uses ``paths``, ``subject``, ``conditions``
        and ``connectivity`` (freq_min, freq_max, atlas, n_lines, vmin, vmax,
        figure, size, borders, hemi).

    Returns
    -------
    p : Connectivity
        Parcellated connectivity object.
    brain : Brain
        Rendered cortical surface.
    """
    import mne
    import conpy
    import matplotlib.pyplot as plt

    subjects_dir = config['paths']['subjects_dir']
    subject_name = config['subject']['subject_name']
    condition_1  = config['conditions']['condition_1']['name']
    condition_2  = config['conditions']['condition_2']['name']

    freq_min     = config['connectivity']['freq_min']
    freq_max     = config['connectivity']['freq_max']
    atlas        = config['connectivity']['atlas']
    n_lines      = config['connectivity']['n_lines']
    vmin         = config['connectivity'].get('vmin', None)
    vmax         = config['connectivity'].get('vmax', None)
    figure       = config['connectivity'].get('figure', None)
    size         = config['connectivity'].get('size', 800)
    borders      = config['connectivity'].get('borders', True)
    hemi         = config['connectivity'].get('hemi', config.get('visualization', {}).get('hemi', 'both'))

    con_1 = conpy.read_connectivity('{}-connectivity for band from {} to {}_{}'.format(
        subject_name, freq_min, freq_max, condition_1))
    con_2 = conpy.read_connectivity('{}-connectivity for band from {} to {}_{}'.format(
        subject_name, freq_min, freq_max, condition_2))

    adj = (con_1 - con_2).get_adjacency()
    plt.figure()
    plt.imshow(adj.toarray(), interpolation='nearest')

    # Parcellated circle plot
    labels = mne.read_labels_from_annot('{}'.format(subject_name), atlas,
                                        subjects_dir=subjects_dir)
    del labels[-1]

    p = con_1.parcellate(labels, 'degree', weight_by_degree=True)

    p.plot(n_lines=n_lines, vmin=vmin, vmax=vmax)
    plt.title('Strongest parcel-to-parcel connection', color='white')

    # Cortical surface
    con_contrast = con_1 - con_2
    all_to_all   = con_contrast.make_stc('absmax')
    brain        = all_to_all.plot('{}'.format(subject_name),
                                   subjects_dir=subjects_dir, hemi=hemi,
                                   figure=figure, size=size)
    brain.add_annotation(atlas, borders=borders)

    return p, brain


def connectivity_statistics(config):
    """
    Group-level cluster-permutation statistics on the connectivity contrast
    between two conditions (run once across subjects).

    Parameters
    ----------
    config : dict
        Configuration dictionary. Uses ``paths``, ``conditions`` and
        ``connectivity`` (spacing, num_subjects, atlas, freq_min, freq_max,
        cluster_threshold, n_permutations, alpha, tail, max_spread, seed,
        summary, brain_mode, hemi_stat, views, borders).

    Returns
    -------
    connection_indices, bundles, bundle_ts, bundle_ps, H0, contrast
        Outputs of the cluster permutation test plus the contrast object.
    """
    import os
    import mne
    import numpy as np
    import conpy
    from h5io import write_hdf5

    folder            = config['paths']['data_folder']
    subjects_dir      = config['paths']['subjects_dir']
    condition_1       = config['conditions']['condition_1']['name']
    condition_2       = config['conditions']['condition_2']['name']

    spacing           = config['connectivity']['spacing']
    num_subject       = config['connectivity']['num_subjects']
    atlas             = config['connectivity']['atlas']
    freq_min          = config['connectivity']['freq_min']
    freq_max          = config['connectivity']['freq_max']

    cluster_threshold = config['connectivity']['cluster_threshold']
    n_perm            = config['connectivity']['n_permutations']
    nj                = config['processing']['n_jobs']
    ms                = config['connectivity']['max_spread']
    tl                = config['connectivity']['tail']
    alpha             = config['connectivity']['alpha']
    seed              = config['connectivity']['seed']
    summary           = config['connectivity']['summary']
    brain_mode        = config['connectivity']['brain_mode']
    hemi_stat         = config['connectivity']['hemi_stat']
    views             = config['connectivity']['views']
    borders           = config['connectivity'].get('borders', True)

    con_surf_1    = []
    con_surf_2    = []
    con_surf_1_av = []
    con_surf_2_av = []

    os.chdir(folder)
    fsaverage = mne.read_source_spaces('Sub_for_con_Avg-{}-src.fif'.format(spacing))
    index     = np.linspace(1, num_subject, num_subject, dtype=int)

    for i in index:
        con_1           = conpy.read_connectivity('S{}-connectivity for band from {} to {}_{}'.format(i, freq_min, freq_max, condition_1))
        con_fsaverage_1 = con_1.to_original_src(fsaverage, subjects_dir=subjects_dir)
        con_surf_1.append(con_1)
        con_surf_1_av.append(con_fsaverage_1)
        con_2           = conpy.read_connectivity('S{}-connectivity for band from {} to {}_{}'.format(i, freq_min, freq_max, condition_2))
        con_fsaverage_2 = con_2.to_original_src(fsaverage, subjects_dir=subjects_dir)
        con_surf_2.append(con_2)
        con_surf_2_av.append(con_fsaverage_2)

    print('Averaging connectivity objects...')
    ga_con = dict()

    con = con_surf_1_av[0].copy()
    for other_con in con_surf_1_av[1:]:
        con += other_con
    con /= len(con_surf_1_av)
    ga_con[1] = con

    con = con_surf_2_av[0].copy()
    for other_con in con_surf_2_av[1:]:
        con += other_con
    con /= len(con_surf_2_av)
    ga_con[2] = con

    contrast = ga_con[1] - ga_con[2]

    # Cluster-permutation statistics
    stats = conpy.cluster_permutation_test(con_surf_1_av, con_surf_2_av,
                                           cluster_threshold=cluster_threshold,
                                           src=fsaverage, n_permutations=n_perm,
                                           verbose=True, alpha=alpha, tail=tl,
                                           n_jobs=nj,
                                           seed=seed,
                                           return_details=True, max_spread=ms)

    connection_indices, bundles, bundle_ts, bundle_ps, H0 = stats

    # Save statistics to disk (connectivity object + HDF5 details)
    con_clust = contrast[connection_indices]
    con_clust.save('Con_stat_from_{}_to_{}'.format(freq_min, freq_max))

    write_hdf5('Con_statistics-from_{}_to_{}'.format(freq_min, freq_max), dict(
        connection_indices=connection_indices,
        bundles=bundles,
        bundle_ts=bundle_ts,
        bundle_ps=bundle_ps,
        H0=H0), overwrite=True)

    # Parcellate the surviving connections
    os.chdir(folder)
    labels = mne.read_labels_from_annot('fsaverage', atlas, hemi_stat,
                                        subjects_dir=subjects_dir)
    del labels[-1]  # drop 'unknown' label
    con_parc = con_clust.parcellate(labels, summary=summary,
                                    weight_by_degree=False)
    con_parc.save('Con_statistics-{}_{}_contr'.format(freq_min, freq_max))

    mne.viz.set_3d_options(depth_peeling=False, antialias=False, multi_samples=1)

    all_to_all = con_clust.make_stc(brain_mode)
    brain      = all_to_all.plot('fsaverage', subjects_dir=subjects_dir, hemi=hemi_stat,
                                 figure=6, size=400, views=views)
    brain.add_annotation(atlas, borders=borders)

    return connection_indices, bundles, bundle_ts, bundle_ps, H0, contrast


def connectivity_statistics_visualization(config):
    """
    Render the group-level statistical connectivity contrast: a parcellated
    circle plot (optionally restricted to selected labels) and the contrast on
    the cortical surface.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Uses ``paths``, ``conditions`` and
        ``connectivity`` (spacing, num_subjects, atlas, freq_min, freq_max,
        hemi, regexp, summary, weight_by_degree, hemi_stat, views, borders,
        n_lines_stat, vmin_stat, vmax_stat, fontsize_names, fontsize_colorbar,
        brain_mode).

    Returns
    -------
    brain : Brain
        Rendered cortical surface.
    con_parc : Connectivity
        Parcellated connectivity object.
    """
    import os
    import mne
    import numpy as np
    import conpy
    from h5io import read_hdf5

    folder            = config['paths']['data_folder']
    subjects_dir      = config['paths']['subjects_dir']
    condition_1       = config['conditions']['condition_1']['name']
    condition_2       = config['conditions']['condition_2']['name']

    spacing           = config['connectivity']['spacing']
    num_subject       = config['connectivity']['num_subjects']
    atlas             = config['connectivity']['atlas']
    freq_min          = config['connectivity']['freq_min']
    freq_max          = config['connectivity']['freq_max']
    hemi              = config['connectivity'].get('hemi', 'both')

    regexp            = config['connectivity'].get('regexp', None)
    summary_stat      = config['connectivity']['summary']
    weight_by_degree  = config['connectivity']['weight_by_degree']
    hemi_stat         = config['connectivity']['hemi_stat']
    views             = config['connectivity']['views']
    borders           = config['connectivity'].get('borders', True)
    n_lines_stat      = config['connectivity']['n_lines_stat']
    vmin_stat         = config['connectivity'].get('vmin_stat', None)
    vmax_stat         = config['connectivity'].get('vmax_stat', None)
    fontsize_names    = config['connectivity']['fontsize_names']
    fontsize_colorbar = config['connectivity']['fontsize_colorbar']
    brain_mode        = config['connectivity']['brain_mode']

    con_surf_1    = []
    con_surf_2    = []
    con_surf_1_av = []
    con_surf_2_av = []

    os.chdir(folder)
    fsaverage = mne.read_source_spaces('Sub_for_con_Avg-{}-src.fif'.format(spacing))
    index     = np.linspace(1, num_subject, num_subject, dtype=int)

    for i in index:
        con_1           = conpy.read_connectivity('S{}-connectivity for band from {} to {}_{}'.format(i, freq_min, freq_max, condition_1))
        con_fsaverage_1 = con_1.to_original_src(fsaverage, subjects_dir=subjects_dir)
        con_surf_1.append(con_1)
        con_surf_1_av.append(con_fsaverage_1)
        con_2           = conpy.read_connectivity('S{}-connectivity for band from {} to {}_{}'.format(i, freq_min, freq_max, condition_2))
        con_fsaverage_2 = con_2.to_original_src(fsaverage, subjects_dir=subjects_dir)
        con_surf_2.append(con_2)
        con_surf_2_av.append(con_fsaverage_2)

    print('Averaging connectivity objects...')
    ga_con = dict()

    con = con_surf_1_av[0].copy()
    for other_con in con_surf_1_av[1:]:
        con += other_con
    con /= len(con_surf_1_av)
    ga_con[1] = con

    con = con_surf_2_av[0].copy()
    for other_con in con_surf_2_av[1:]:
        con += other_con
    con /= len(con_surf_2_av)
    ga_con[2] = con

    contrast = ga_con[2] - ga_con[1]

    os.chdir(folder)

    file = read_hdf5('Con_statistics-from_{}_to_{}'.format(freq_min, freq_max))

    connection_indices = file.get('connection_indices')
    con_clust          = contrast[connection_indices]

    # Optionally restrict to selected labels (unilaterality)
    labels = mne.read_labels_from_annot('fsaverage', atlas, subjects_dir=subjects_dir)

    selected_label = mne.read_labels_from_annot('fsaverage', hemi=hemi,
                                                regexp=regexp,
                                                subjects_dir=subjects_dir)

    label_colors = [label.color for label in selected_label]

    del labels[-1]  # drop 'unknown' label
    con_parc = con_clust.parcellate(selected_label, summary=summary_stat,
                                    weight_by_degree=weight_by_degree)

    con_parc.plot(n_lines=n_lines_stat, vmin=vmin_stat, vmax=vmax_stat,
                  node_colors=label_colors, fontsize_names=fontsize_names,
                  fontsize_colorbar=fontsize_colorbar)

    mne.viz.set_3d_options(depth_peeling=False, antialias=False, multi_samples=1)

    all_to_all = con_clust.make_stc(brain_mode)
    brain      = all_to_all.plot('fsaverage', subjects_dir=subjects_dir, hemi=hemi_stat,
                                 figure=6, size=400, views=views)
    brain.add_annotation(atlas, borders=borders)

    return brain, con_parc
