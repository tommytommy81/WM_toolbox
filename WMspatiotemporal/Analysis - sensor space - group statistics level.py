# -*- coding: utf-8 -*-
"""
### SENSOR LEVEL STATISTICS
# This file contain variables, which should be changed according to the analysis goal.
# All functions are in a separate file
@author: Nikita Otstavnov, 2023
"""

#### FILE NAVIGATION
%matplotlib qt
file_to_read     = 'C:/Users/User/Desktop/For testing/1_test_1_tsss_mc_trans.fif'
folder           = 'C:/Users/User/Desktop/For testing'
file_name        = '1_test_1_tsss_mc_trans.fif'
subject_name     = 'S1'
condition_1      = 'S'
condition_2      = 'T'

#### AVERAGING FOOOF PARAMETERS
t_interest_min   = 0
t_interest_max   = 4
num_subjects     = 3

#### PARAMETERS OF SENSOR STATISTICS
ch_type          = 'grad'
alpha            = 0.05
threshold        = 0.01      #p_value, which will be used to calculate the t_threshold
n_permutations   = 5000
out_type         = 'mask' 
tail             = 0 

#### FIGURE 4: parameters
dpi              = 300
labelsize        = 4
titlesize        = 4
fontsize         = 4
fontfamily       = 'sans-serif'
vmin_4b          = -5
vmax_4b          = 5
vmin_4c          = -3
vmax_4c          = 3
list_channels    = [ 'MEG0113', 'MEG0112', 'MEG0122', 'MEG0123', 'MEG0132', 'MEG0133', 'MEG0143',
 'MEG0142', 'MEG0213', 'MEG0212', 'MEG0222', 'MEG0223', 'MEG0232', 'MEG0233', 'MEG0243',
 'MEG0242', 'MEG0313', 'MEG0312', 'MEG0322', 'MEG0323', 'MEG0333', 'MEG0332', 'MEG0343',
 'MEG0342', 'MEG0413', 'MEG0412', 'MEG0422', 'MEG0423', 'MEG0432', 'MEG0433', 'MEG0443',
 'MEG0442', 'MEG0513', 'MEG0512', 'MEG0523', 'MEG0522', 'MEG0532', 'MEG0533', 'MEG0542',
 'MEG0543', 'MEG0613', 'MEG0612', 'MEG0622', 'MEG0623', 'MEG0633', 'MEG0632', 'MEG0642',
 'MEG0643', 'MEG0713', 'MEG0712', 'MEG0723', 'MEG0722', 'MEG0733', 'MEG0732', 'MEG0743',
 'MEG0742', 'MEG0813', 'MEG0812', 'MEG0822', 'MEG0823', 'MEG0913', 'MEG0912', 'MEG0923',
 'MEG0922', 'MEG0932', 'MEG0933', 'MEG0942', 'MEG0943', 'MEG1013', 'MEG1012', 'MEG1023',
 'MEG1022', 'MEG1032', 'MEG1033', 'MEG1043', 'MEG1042', 'MEG1112', 'MEG1113', 'MEG1123',
 'MEG1122', 'MEG1133', 'MEG1132', 'MEG1142', 'MEG1143', 'MEG1213', 'MEG1212', 'MEG1223',
 'MEG1222', 'MEG1232', 'MEG1233', 'MEG1243', 'MEG1242', 'MEG1312', 'MEG1313', 'MEG1323',
 'MEG1322', 'MEG1333', 'MEG1332', 'MEG1342', 'MEG1343', 'MEG1412', 'MEG1413', 'MEG1423',
 'MEG1422', 'MEG1433', 'MEG1432', 'MEG1442', 'MEG1443', 'MEG1512', 'MEG1513',
 'MEG1522', 'MEG1523', 'MEG1533', 'MEG1532', 'MEG1543', 'MEG1542', 'MEG1613', 'MEG1612',
 'MEG1622', 'MEG1623', 'MEG1632', 'MEG1633', 'MEG1643', 'MEG1642', 'MEG1713', 'MEG1712',
 'MEG1722', 'MEG1723', 'MEG1732', 'MEG1733', 'MEG1743', 'MEG1742', 'MEG1813', 'MEG1812',
 'MEG1822', 'MEG1823', 'MEG1832', 'MEG1833', 'MEG1843', 'MEG1842', 'MEG1912', 'MEG1913',
 'MEG1923', 'MEG1922', 'MEG1932', 'MEG1933', 'MEG1943', 'MEG1942', 'MEG2013', 'MEG2012',
 'MEG2023', 'MEG2022', 'MEG2032', 'MEG2033', 'MEG2042', 'MEG2043', 'MEG2113', 'MEG2112',
 'MEG2122', 'MEG2123', 'MEG2133', 'MEG2132', 'MEG2143', 'MEG2142', 'MEG2212',
 'MEG2213', 'MEG2223', 'MEG2222', 'MEG2233', 'MEG2232', 'MEG2242', 'MEG2243', 'MEG2312',
 'MEG2313', 'MEG2323', 'MEG2322', 'MEG2332', 'MEG2333', 'MEG2343', 'MEG2342', 'MEG2412',
 'MEG2413', 'MEG2423', 'MEG2422', 'MEG2433', 'MEG2432', 'MEG2442', 'MEG2443', 'MEG2512',
 'MEG2513', 'MEG2522', 'MEG2523', 'MEG2533', 'MEG2532', 'MEG2543', 'MEG2542', 'MEG2612',
 'MEG2613', 'MEG2623', 'MEG2622', 'MEG2633', 'MEG2632', 'MEG2642', 'MEG2643']
ch_type        = 'grad'
min_freqs      = 4
max_freqs      = 80

###################################### STEP 2: sensor statistics
#### FOOOF MERGER
from WMspatiotemporal.STWM_functions import fooof_merger
list_1_ped, list_1_aper, list_2_ped, list_2_aper = fooof_merger(num_subjects, 
                                                                folder, 
                                                                condition_1, 
                                                                condition_2) 

#### STATISTICS
from WMspatiotemporal.STWM_functions import sensor_statistics
T_obs, T_obs_plot, clusters, cluster_p_values = sensor_statistics(folder, 
                                                                  subject_name, 
                                                                  condition_1, 
                                                                  ch_type,
                                                                  list_1_ped, 
                                                                  list_2_ped, 
                                                                  alpha,
                                                                  threshold, 
                                                                  n_permutations, 
                                                                  tail, 
                                                                  out_type)
#### FIGURE PARAMETERS
# Figure 3
excel_with_beh_results = 'Result_4'
folder                 = 'C:/Users/User/Desktop'

# Figure 4
vmin_4b                = -5
vmax_4b                = 5
freq_int               = T_obs[9:16]
vmax_4c                = -3
vmin_4c                = 3
Rect                   = False
%matplotlib

#### VISUALISATION 
from Script_for_Figure_generating import fig_3 
from Script_for_Figure_generating import fig_4 

t_test, h, beh = fig_3(folder, excel_with_beh_results)

fig = fig_4(num_subjects, condition_1, condition_2, T_obs, T_obs_plot,
               folder, vmin_4b, vmax_4b, vmax_4c, vmin_4c, freq_int, Rect)


