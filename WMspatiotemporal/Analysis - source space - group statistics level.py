# -*- coding: utf-8 -*-
"""
### SOURCE LEVEL STATISTICS
# This file contain variables, which should be changed according to the analysis goal.
# All functions are in a separate file
@author: Nikita Otstavnov, 2023
"""

##### FEATURES
%matplotlib qt
n_jobs           = 7
num_subject      = 3


#### FILE NAVIGATION
folder           = 'C:/Users/User/Desktop/For testing'
file_name        = '3_test_1_tsss_mc_trans.fif'
subject_name     = 'S3'
condition_3      = 'AvBase'

#### GUI COREGISTRATION
subjects_dir     = 'C:/Users/User/Desktop/For testing/fmri'

#### SOURCE SPACE PARAMETERS
spacing          = 'oct6'
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
transparency     = 0.5
time_viewer      = 'auto'
views            = ['dorsal', 'lateral', 'medial','ventral']
volume_options   = 1.0
view_layout      = 'vertical'
surface          = 'inflated'
annotation       = 'aparc.a2009s'
mode             = 'rgb'
backend          = 'auto'

#### GROUP STATISTICS
p_threshold      = 0.01
n_permutations   = 2000
out_type         = 'indices'
buffer_size      = None
alpha_level      = 0.05
tstep            = 1 

#### MERGER
from WMspatiotemporal.STWM_functions import stc_merger
stc_s, stc_t, stc_a  =    stc_merger(folder, num_subject, 
                                    subject_name,freq_min, freq_max, 
                                    orientation, condition_1, condition_2, condition_3,
                                    subjects_dir, spacing)
        

#### VISUAL CHECKER
subject_to_visualize = 1

from WMspatiotemporal.STWM_functions import source_estimate_average_visual_checher
source_estimate_average_visual_checher(stc_s, stc_t, stc_a, 
                               subject_to_visualize,freq_min,freq_max,spacing,
                               hemi, colormap, time_label, transparency, time_viewer, 
                               views, volume_options, view_layout, surface,
                               annotation, mode, subjects_dir, backend)

#### STATISTICS
from WMspatiotemporal.STWM_functions import statistical_inference
stc_all_cluster_vis, stc_new, clu = statistical_inference(num_subject, stc_s, 
                                                          stc_t, stc_a, 
                                                          spacing, folder, subjects_dir,
                                                          p_threshold, n_permutations, tstep,
                                                          n_jobs, out_type, buffer_size, alpha_level)
#### VISUALIZATION
from WMspatiotemporal.STWM_functions import stat_visualization
brain = stat_visualization(stc_all_cluster_vis, freq_min, freq_max, spacing,
                       hemi, colormap, time_label, alpha, time_viewer, 
                       views, volume_options, view_layout, surface,
                       annotation, mode, subjects_dir, backend)

#### FIGURE FOR PAPER
### WHAT TO PLOT
freq_min_1 = 31
freq_max_1 = 80
freq_min_2 = 31
freq_max_2 = 80
vmin = 0.32
vmax = 0.57

from Script_for_Figure_generating import fig_5
fig = fig_5(freq_min_1, freq_max_1, freq_min_2, freq_max_2, vmin, vmax)
