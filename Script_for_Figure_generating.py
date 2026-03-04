# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:48:57 2023

@author: User
"""

def fig_3(folder, excel_with_beh_results):
    """
    This figure displays the results of the behavioral data comparison
    Chang the path and the file according to your data.
    """
    
    import os
    import os.path as op
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy
    
    ### SETTINGS
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20)
    
    ### STATISTICS
    df             = pd.read_excel('{}/{}.xlsx'.format(folder, excel_with_beh_results))
    
    results        = df.to_numpy()
    Accuracy_T     = results[:,1]
    Accuracy_S     = results[:,0]
    Accuracy_full  = np.append(Accuracy_T,Accuracy_S)
    
    t_test = scipy.stats.ttest_rel(Accuracy_T, Accuracy_S,  
                                   nan_policy='propagate', alternative='two-sided')
    print(t_test)
    
    ### PLOT 
    data_df = df.melt(var_name='Condition',value_name='Percentage of correct answers')
    print(data_df.head())
    
    beh = sns.boxplot(data=df, x=None, y=None, hue=None, order=None, 
                    hue_order=None, orient=None, color=None, palette=None, saturation=0.75, 
                    width=0.8, dodge=True, fliersize=5, linewidth=None, whis=1.5)
    beh = sns.stripplot(data = df, color = 'k')
    plt.savefig("Beh_results.png", format='png')
  
    ### EFFECT SIZE
    dof = len(Accuracy_full) - 2
    # Variances
    var1 = np.var(Accuracy_S)
    var2 = np.var(Accuracy_T)
    # Difference of the means
    diff_mean = abs(np.mean(Accuracy_S)-np.mean(Accuracy_T))
    # Pooled standard deviation
    s_pooled_star = np.sqrt((((len(Accuracy_S) - 1) * var1) + ((len(Accuracy_T) - 1) * var2)) / dof)
    # Hedges's g
    hedgess_g = diff_mean / s_pooled_star
    print(f"Hedges's g = {hedgess_g:.3f}")
    
    return t_test, hedgess_g, beh


def fig_4(num_subjects, condition_1, condition_2, T_obs, T_obs_plot,
          folder, vmin_4b, vmax_4b, vmax_4c, vmin_4c, freq_int, Rect):
    """
    This figure displays the results of sensor space analysis 
    Change the path and slace according to your data.
    """
    import os
    import mne
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize  
    from matplotlib.cm import ScalarMappable
    
    #### UPLOADING
    power_1_list = []
    power_2_list = []
    
    ###### UPLOADER
    i = 0
    for i in range(num_subjects): 
        power_1 = mne.time_frequency.read_tfrs('S{}_power_{}-tfr.h5'.format(i+1, condition_1))
        power_1_list.append(power_1[0])
        power_2 = mne.time_frequency.read_tfrs('S{}_power_{}-tfr.h5'.format(i+1, condition_2))
        power_2_list.append(power_2[0])

     
    grand_average_1 = mne.grand_average(power_1_list,interpolate_bads=True, drop_bads=True)
    # grand_average_1.apply_baseline(baseline=(-8,-7), mode='logratio', verbose=None)
    
    fig, ax = plt.subplots(2, 2)
    
    ##### FIGURE 4A
    ax1   = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    grand_average_1.plot(
                        mode='logratio', 
                        axes=ax1,
                        colorbar=True, 
                        combine = 'mean', 
                        yscale='linear')  
    ax1.set_title('Spatial power averaged across subjects', pad=30)
    
    #Rectangle
    if Rect == True:
        ax1.add_patch(plt.Rectangle((0.008, 3.3), 3.985, 77.5, ls="-", lw=1, ec="b", fc="none"))
    
    #Lines
    ax1.axvline(x=-7,   ymin=0, ymax=1, color="black", linestyle="--", linewidth=0.7)
    ax1.axvline(x=-6,   ymin=0, ymax=1, color="black", linestyle="--", linewidth=0.7)
    ax1.axvline(x=-4.5, ymin=0, ymax=1, color="black", linestyle="--", linewidth=0.7) 
    ax1.axvline(x=-3,   ymin=0, ymax=1, color="black", linestyle="--", linewidth=0.7) 
    ax1.axvline(x=-1.5, ymin=0, ymax=1, color="black", linestyle="--", linewidth=0.7) 
    ax1.axvline(x= 5,   ymin=0, ymax=1, color="black", linestyle="--", linewidth=0.7) 
    
    #Y-axis correct labels
    y_label_list = [ '6', '10', '17', '28', '48', '80']
    plt.yticks(    [  13,   26,   39,   52,   66,   80], y_label_list )
    
    #Text
    ax1.text(x=-7.2,  y=87, s='Cue') 
    ax1.text(x=-7.3,  y=82, s='Onset') 
    ax1.text(x=-6.2,  y=87, s='Item')
    ax1.text(x=-6.05, y=82, s='1')
    ax1.text(x=-4.7,  y=87, s='Item')
    ax1.text(x=-4.55, y=82, s='2')
    ax1.text(x=-3.2,  y=87, s='Item')
    ax1.text(x=-3.05, y=82, s='3')
    ax1.text(x=-1.65, y=87, s='Item')
    ax1.text(x=-1.55, y=82, s='4')
    ax1.text(x=-0.45, y=87, s='Retention') 
    ax1.text(x=-0.2,  y=82, s='onset') 
    ax1.text(x= 3.75, y=87, s='Probe') 
    ax1.text(x= 3.75, y=82, s='onset') 
    ax1.text(x= 4.6,  y=87, s='Response') 
    ax1.text(x= 4.8,  y=82, s='onset') 
    
    #### FIGURE 4B
    vmin_4b = vmin_4b
    vmax_4b = vmax_4b
    
    ax2 = plt.subplot2grid((2, 2), (1, 0), rowspan=1)
    ax2.add_patch(plt.Rectangle((130, 10.5), 60, 7, ls="--", lw=0.5, ec="k", fc="none"))
    plt.imshow(T_obs, aspect='auto', origin='lower', cmap='RdBu_r', 
                vmin=vmin_4b, vmax=vmax_4b) 
    plt.colorbar(label = 'T values')
    plt.contour(T_obs_plot, levels = 1, colors='g', alpha =0.5,
                linewidths=[0.5], linestyles='solid', 
                origin = None)
    plt.title('Frequency-spatial plot of T values for contrasting condition (temporal - spatial)')
    
    y_label_list = ['4', '6', '10', '17', '28', '48', '80']
    plt.yticks(    [ 0,   5,   10,   15,   20,   25,  29 ], y_label_list )
    ax2.set_xlabel('Channels')
    ax2.set_ylabel('Frequency (Hz)')
    
    #### FIGURE 4C
    # ADJUCENCY
    os.chdir(folder) 
    SDFR = folder + '/' +'S1_filtered.fif'
    data = mne.io.read_raw_fif(SDFR, allow_maxshield=False, 
                                preload=False, on_split_missing='raise', 
                                verbose=None)
    info            = data.info
     
    what_to_analyze = freq_int
    what_to_analyze = np.mean(what_to_analyze, axis = 0)
    print(np.max(what_to_analyze)) #VMAX
    print(np.min(what_to_analyze)) #VMIN
    vmax       = vmax_4c            
    vmin       = vmin_4c         
    
    ax3   = plt.subplot2grid((2, 2), (1, 1), rowspan=1)                                                                            
    im,cm = mne.viz.plot_topomap(what_to_analyze, 
                                  pos = info, 
                                  ch_type='grad', 
                                  image_interp='cubic',  
                                  cmap = 'Reds',
                                  axes=ax3,  show=True,
                                  vlim=(-vmin,vmax))        
    ax3.set_title('Topomap of T-values for significant frequencies')
 
    cmap       = plt.colormaps["RdBu_r"]
    cax        = fig.add_axes([0.9, 0.1, 0.008, 0.32])
    norm       = Normalize(vmin=vmin, vmax=vmax)
    cbar       = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[vmin, vmax])
    cbar.outline.set_visible(True)
    cax.yaxis.set_ticks_position('right')
    cbar.set_ticks([-3, -2, -1, 0, 1, 2 ,3])
    cbar.set_ticklabels(["-3", "-2", "-1", "0", "1", "2", "3"])
    cbar.set_label('T values', rotation=90)
    
    plt.subplots_adjust(left=0.1, right=1, bottom=0.1, top=0.8, wspace = None)       

    return fig


def fig_5(freq_min_1, freq_max_1, freq_min_2, freq_max_2, vmin, vmax):
    """
    In this part one must be attentive to the scale of the colorbar. 
    It can be present either in percentage or values. 
    The choise of the scale depends on your study goal. 
    Two options are present below, you can choose one of them by commenting another one (ctrl+1)
    """
     
    import mne
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import image as img
    from matplotlib.colors import Normalize  
    from matplotlib.cm import ScalarMappable
    
    ### SETTINGS
    matplotlib.rc('xtick', labelsize=10) 
    matplotlib.rc('ytick', labelsize=10)
    
    ### UPLOADER AND PLOT
    fig, ax    = plt.subplots(1, 2)
    ax[0].set_title('Theta frequency source estimate', fontdict={'fontsize':13})
    ax[1].set_title('Beta frequency source estimate', fontdict={'fontsize': 13})
    image_1    = img.imread('Average_statistics_from_{}_to_{}.png'.format(freq_min_1, freq_max_1))
    image_2    = img.imread('Average_statistics_from_{}_to_{}.png'.format(freq_min_2, freq_max_2))
    
    ax[0].imshow(image_1)
    ax[1].imshow(image_2)
    ax[0].axis('off')
    ax[1].axis('off')
    
    # COLORBAR SETTINGS
    
    # If we want values to be in the plot
    # s_1        = mne.read_source_estimate('Stat_from_{}_to_{}_ind_surf_stc'.format(freq_min_1, freq_max_1))
    # s_2        = mne.read_source_estimate('Stat_from_{}_to_{}_ind_surf_stc'.format(freq_min_2, freq_max_2))
    # s_2        = mne.read_source_estimate('S1_from_31_to_80_fix_morph_surf_AvBase_stc')
    # vmin       = min(s_1.data)
    # vmax       = max(s_1.data)
    # vmean      = (vmax - vmin)/2
    
    # If we want percents to be in the plot: 
        #Plot the brain in the Source space analysis file
        #Take the "Using control points []" as the min, max and mean
   
    vmean= vmin+((vmax - vmin)/2)
    
    cmap       = plt.colormaps["hot"]
    cax        = fig.add_axes([0.05, 0.4, 0.01, 0.2])
    norm       = Normalize(vmin=vmin, vmax=vmax)
    cbar       = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[vmin,vmax,vmean])
    cbar.outline.set_visible(False)
    cax.set_title("%", y =1.1)
    cax.yaxis.set_ticks_position('right')
    
    cmap       = plt.colormaps["hot"]
    cax2       = fig.add_axes([0.5, 0.4, 0.01, 0.2])
    norm       = Normalize(vmin=vmin, vmax=vmax)
    cbar2      = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax2, ticks=[vmin,vmax,vmean])
    cbar2.outline.set_visible(False)
    cax2.set_title("%", y = 1.1)
    cax2.tick_params(labelsize=10)
    cax2.yaxis.set_ticks_position('right')

    return fig


"""
Fig_6 and fig_merger helps to construct figures for connectivity analysis. 
"""

def fig_6(data_path,folder_with_files, 
          circ_file_name, brain_file_name, freq, 
          vmin, vmax):
    
    import matplotlib.pyplot as plt
    from matplotlib import image as img
    from matplotlib.colors import Normalize  
    from matplotlib.cm import ScalarMappable
    
    fig, ax = plt.subplots(1, 2, layout="constrained")
    fig.set_facecolor('black')
    
    image_1_1    = img.imread('{}.png'.format(circ_file_name))
    img_cropped = image_1_1[200:600, 200:600, :]
    image_1_2    = img.imread('{}.png'.format(brain_file_name))
    img_cropped = img_cropped[200:600, 200:600, :]
    
    ax[0].imshow(image_1_1)
    ax[1].imshow(image_1_2)
    ax[0].axis('off')
    ax[1].axis('off')
    
    vmin = vmin
    vmax = vmax
    cmap       = plt.colormaps["hot"]
    cax        = fig.add_axes([0.9, 0.41, 0.005, 0.16])
    norm       = Normalize(vmin=vmin, vmax=vmax)
    cbar       = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, ticks=[vmin, vmax])
    cbar.outline.set_visible(False)
    cax.yaxis.set_ticks_position('right')
    cbar.set_ticks([-0.1, 0.2])
    cax.tick_params(labelsize=12, colors = 'w')
    cbar.set_ticklabels(["0", "0.2"])
    plt.subplots_adjust(wspace=0, hspace=0)
    
    fig.savefig('Circ_plus_brain.png'.format(freq))
    
    return fig

def fig_merge(file_1, file_2, file_3, file_4,
              file_5, file_6):
    import matplotlib
    from matplotlib import image as img
    from matplotlib import plt
    
    fig, ax = plt.subplots(4, 2, sharex = True, sharey = True)
    plt.subplots_adjust(wspace=-0, hspace=0)
    fig.set_tight_layout(True)
    fig.set_facecolor('white')
    
    image_1_1    = img.imread(file_1)
    image_1_2    = img.imread(file_2)
    image_2_1    = img.imread(file_3)
    image_3_1    = img.imread(file_4)
    image_3_2    = img.imread(file_5)
    image_4_1    = img.imread(file_6)
    
    ax[0,0].imshow(image_1_1)
    ax[0,1].imshow(image_1_2)
    ax[1,0].imshow(image_2_1)
    ax[2,0].imshow(image_3_1)
    ax[2,1].imshow(image_3_2)
    ax[3,0].imshow(image_4_1)
    
    ax[0,0].axis('off')
    ax[0,1].axis('off')
    ax[1,0].axis('off')
    ax[2,0].axis('off')
    ax[2,1].axis('off')
    ax[3,0].axis('off')
    
    return fig