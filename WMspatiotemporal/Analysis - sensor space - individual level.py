"""
### SENSOR LEVEL INDIVIDUAL
# This file contain variables, which should be changed according to the analysis goal.
# All functions are in a separate file
@author: Nikita Otstavnov, 2023

"""

### LIBRARIES
%matplotlib qt

### FILE NAVIGATION
folder           = 'C:/Users/User/Desktop/For testing'
file_name        = '3_test_1_tsss_mc_trans.fif'
subject_name     = 'S3'

### FILTERING PARAMETERS
# Frequencies of interst
min_freqs        = 4
max_freqs        = 80
#Notch_filter
notch_freq       = 50
#Channel selection
meg              = 'grad'
exclude_channels = []
#ICA parameters
n_components     = 40
random_state     = 97
max_iter         = 800

# Channel with event markers
stim_channel     = 'STI101'
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


# EPOCHING 
reject_criteria  = dict(grad=3000e-13)    # 3000 fT/cm
flat_criteria    = dict(grad=1e-13)         # 1 fT/cm
main_event_1     = 155
main_event_2     = 255
tmin_epo         = -8
tmax_epo         = 8
condition_1      = 'S'
condition_2      = 'T'

# TIME FREQUENCY PARAMETERS
import numpy as np
min_freq         = np.log10(4)
max_freq         = np.log10(80)
freq_res         = 30
decim            = 20
n_jobs           = 7
n_cycles         = 5
mode             = 'logratio'
tb_min           = None
tb_max           = None

# FOOOF 
t_interest_min   = 0
t_interest_max   = 4
num_subjects     = 3

# SENSOR STATISTICS
ch_type          = 'grad'
alpha            = 0.05
threshold        = 3.0
n_permutations   = 5000
out_type         = 'mask' 
tail             = 0 

# CSD
tbase_min        = -8 
tbase_max        = -7
picks            = ['meg']

# Figure 
dpi              = 300
labelsize        = 4
titlesize        = 4
fontsize         = 4
fontfamily       = 'sans-serif'

#### CSD
tbase_min        = -8 
tbase_max        = -7


######## STEP 1: data preparation and initial analysis

#### PERFORMING PRELIMINARY ANALYSIS
from WMspatiotemporal.STWM_functions import perform_initial_analysis 
raw_data, ica, ica.exclude  = perform_initial_analysis(n_components, 
                                                       random_state, max_iter, 
                                                       stim_channel, list_channels,
                                                       meg, exclude_channels, folder, 
                                                       file_name, subject_name, min_freqs, 
                                                       max_freqs, notch_freq, n_jobs)


#### ICA CHECKER
print(ica.exclude)
raw_data.plot()
ica.plot_sources(raw_data, show_scrollbars=False)
#ica.exclude     = []                                                 #enter components for clearing

#### PERFORMING ICA AND CHANNEL SELECTION
from WMspatiotemporal.STWM_functions import ica_apply
raw_data         = ica_apply(raw_data, ica,subject_name)

#### EVENTS
from WMspatiotemporal.STWM_functions import event_renaming
raw_data, event  = event_renaming(raw_data, stim_channel, 
                                 list_channels, subject_name, meg)

#### VISUAL INSPECTATION 
raw_data.plot(events = event, 
              title  = 'Final raw data with events', 
              event_color = {100: 'g', 200: 'g', 
                               101: 'g', 102: 'g',                                    128: 'r', 208: 'r', 152: 'r',  
                               155: 'r', 110: 'g', 
                               120: 'g', 104: 'g',
                               201: 'g', 202: 'g', 
                               203: 'g', 204: 'g',
                               210: 'g', 220: 'g', 
                               255: 'r', 103: 'g',
                               105: 'g', 205: 'g'})

from WMspatiotemporal.STWM_functions import raw_data_saver
data                   = raw_data_saver(folder, raw_data, subject_name,meg)

#### EPOCHING
from WMspatiotemporal.STWM_functions import epochs_initialization
epochs_1, epochs_2, e_full= epochs_initialization(raw_data, event, 
                          main_event_1, main_event_2, 
                          tmin_epo, tmax_epo, 
                          reject_criteria, flat_criteria, 
                          subject_name, 
                          condition_1, condition_2, ch_type)
epochs_1.plot()
epochs_2.plot()
e_full.plot()

#### EVOKED CHECKER 
from WMspatiotemporal.STWM_functions import evoked_data
evoked_1, evoked_2     = evoked_data(epochs_1, epochs_2, 
                                 subject_name, condition_1, 
                                 condition_2) 
    
#### TIME-FREQUENCY ANALYSIS
from WMspatiotemporal.STWM_functions import time_freq
_, _, power_1, power_2 = time_freq(epochs_1, epochs_2, 
                                   min_freq, max_freq, freq_res, 
                                   decim, n_jobs, 
                                   n_cycles, subject_name, 
                                   condition_1, condition_2)

#### VISUAL CHECK
power                  = power_1
power.plot(combine='mean',
           title       = '{}'.format(condition_2))
power.plot_joint(title = '{}'.format(condition_2))
power.plot_topo(baseline=(tb_min,tb_max), 
                mode   = mode, 
                title  ='{}'.format(condition_2))

#### FOOOF FILTERING
from WMspatiotemporal.STWM_functions import fooof_application 
subj_per_1,subj_aper_1, subj_per_2, subj_per_2, fm  = fooof_application(power_1, 
                                                                    power_2, 
                                                                    t_interest_min,
                                                                    t_interest_max,
                                                                    subject_name, 
                                                                    condition_1, 
                                                                    condition_2,
                                                                    min_freq, 
                                                                    max_freq, 
                                                                    freq_res)

#### VISUAL CHECKER
from   fooof.plts.spectra import plot_spectrum, plot_spectra 
power         = power_1
a             = subj_per_1
freqs         = power.freqs
plt_log       = False
plot_spectrum(fm.freqs, a.T)

#### CSD CALCULATION
from WMspatiotemporal.STWM_functions import csd_average
from WMspatiotemporal.STWM_functions import csd_calc

csd_1, csd_2  = csd_calc(min_freq, max_freq, freq_res, folder, subject_name, 
             condition_1, condition_2, t_interest_min, t_interest_max, 
             n_cycles, decim, n_jobs)
csd_av = csd_average(min_freq, max_freq, freq_res, subject_name, 
                   condition_1, condition_2, folder, 
                   tbase_min, tbase_max, picks, n_cycles, 
                   decim, n_jobs, ch_type, flat_criteria,reject_criteria, 
                   main_event_1, main_event_2, event)



