# -*- coding: utf-8 -*-
"""
### SOURCE LEVEL INDIVIDUAL LEVEL
# This file contain variables, which should be changed according to the analysis goal.
# All functions are in a separate file
@author: Nikita Otstavnov, 2023
"""

### LIBRARIES
%matplotlib qt
n_jobs           = 7

#### FILE NAVIGATION
folder           = 'C:/Users/User/Desktop/For testing'
file_name        = '3_test_1_tsss_mc_trans.fif'
subject_name     = 'S3'
condition_3      = 'AvBase'

#### GUI COREGISTRATION
subjects_dir     = 'C:/Users/User/Desktop/For testing/fmri'

#### SOURCE SPACE PARAMETERS
spacing          = 'ico4'
brain_surfaces   = 'white'
orientation      = 'coronal'

#### FORWARD MODELLING
conductivity     = (0.3,)
ico              = 5
mindist          = 5.0
surfaces         = 'white'
coord_frame      = 'mri'

#### SOURCE ESTIMATE
condition_1      = 'S'
condition_2      = 'T'
freq_min         = 31                                                      
freq_max         = 80                                                      
orientation      = 'fix'     
surf_ori         = True
force_fixed      = True
reg              = 0.05
inversion        = 'single'
depth            = 1.

#### VISUALIZATION OPTIONS
hemi             = 'both'
colormap         = 'auto'
time_label       = 'auto'
alpha            = 0.5
time_viewer      = 'auto'
views            = ['dorsal', 'lateral', 'medial','ventral']
volume_options   = 1.0
view_layout      = 'vertical'
surface          = 'inflated'
annotation       = 'aparc.a2009s'
mode             = 'rgb'
backend          = 'auto'

############## SOURCE SPACE ANALYSIS
#### GUI COREGISTRATION 
from WMspatiotemporal.STWM_functions import visual_freesurfer_check
visual_freesurfer_check(subject_name, folder, file_name,
                            subjects_dir)


# mne.gui.coregistration(subject=subject_name, 
#                         subjects_dir=folder)

#### SOURCE SPACE
from WMspatiotemporal.STWM_functions import creating_source_space_object
from WMspatiotemporal.STWM_functions import creating_average_source_space

source_space = creating_source_space_object(subject_name, subjects_dir,
                                 spacing, brain_surfaces, folder, n_jobs, 
                                 orientation)
# sourve_space_av = creating_average_source_space(spacing, subjects_dir, folder)

#### FORWARD MODELLING
from WMspatiotemporal.STWM_functions import creating_forward_model
from WMspatiotemporal.STWM_functions import creating_average_forward_model

forward_model = creating_forward_model(folder, subject_name, spacing, conductivity, 
                           subjects_dir, mindist, n_jobs, ico, surfaces,
                           coord_frame)
# forward_model_av = creating_average_forward_model(folder, subject_name, 
#                                                   spacing, conductivity, 
#                                                   subjects_dir, mindist, 
#                                                   n_jobs, ico)

#### SOURCE ESTIMATE OBJECT
from WMspatiotemporal.STWM_functions import creating_source_estimate_object
stc_s, stc_t, stc_A, src = creating_source_estimate_object(folder, subject_name, spacing,
                                    condition_1, condition_2, condition_3,
                                    freq_min, freq_max, orientation, 
                                    surf_ori, force_fixed, reg, depth, 
                                    inversion)

#### VISUAL CHECKER
condition = condition_1
stc_to_plot      = stc_s 

from WMspatiotemporal.STWM_functions import source_estimate_visualization
brain_before = source_estimate_visualization(stc_to_plot, subject_name,
                                          freq_min, freq_max, spacing,
                                          hemi, colormap, time_label, alpha, time_viewer, 
                                          views, volume_options, view_layout, surface,
                                          annotation, mode, subjects_dir, backend,
                                          condition)

os.chdir(folder) 
brain_before.save_image('{}_from_{}_to_{}_{}.png'.format(subject_name,
                                                         freq_min, freq_max, 
                                                         condition))

#### MORHPING
from WMspatiotemporal.STWM_functions import source_estimate_morphing_to_average
stc_1, stc_2, stc_a, src, src_a = source_estimate_morphing_to_average(folder, 
                                                                      subject_name, 
                                                                      spacing, 
                                                                      freq_min, 
                                                                      freq_max, 
                                                                      orientation,
                                                                      condition_1, 
                                                                      condition_2, 
                                                                      condition_3,
                                                                      subjects_dir)

#### VISUAL CHECK
condition = 'Contrast'
stc_after       = (stc_1 - stc_2) / stc_a
from WMspatiotemporal.STWM_functions import source_estimate_visualization_morph
brain_after = source_estimate_visualization_morph(stc_after, subject_name,
                                          freq_min, freq_max, spacing,
                                          hemi, colormap, time_label, alpha, time_viewer, 
                                          views, volume_options, view_layout, surface,
                                          annotation, mode, subjects_dir, backend,
                                          condition)

