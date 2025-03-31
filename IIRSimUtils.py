import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.signal import lfilter
from scipy import signal
import scipy
import math
from iir_biquad import iir_biquad as iir_biquad_patrick
from iir_biquad import test as test_patrick
from IIRSim import *
from FrequencyResponse import *
from FPGATestIO import *
from scipy.special import eval_chebyu

def get_coeffs(filename):
    """ Read in coefficients for biquad from a file"""
    # notch_freq = 350
    # Q_FACTOR = 5
    
    # Assuming 8 samples per clock
    samp_per_clock=8
    coeffs_fixed_point_signed = np.zeros(2*samp_per_clock-3 + 4 + 4, dtype=np.int64) # 21
    b_fixed_point_signed = np.zeros(3, dtype=np.int64)
    a_fixed_point_signed = np.zeros(3, dtype=np.int64)
    b_fixed_point_signed_all = np.zeros(3, dtype=np.int64)
    a_fixed_point_signed_all = np.zeros(3, dtype=np.int64)
    C = np.zeros(4)
    f_fir = np.zeros(6)
    g_fir = np.zeros(7)
    
    #with open("001_files/coefficients/coeff_file_%sMHz_%s.dat"%(notch_freq,Q_FACTOR),"r") as coeff_file:
    with open(filename,"r") as coeff_file:
        coeff_list = []
        for line in coeff_file:
            coeff_value = int(line.strip())
            coeff_list.append(coeff_value)
        
    b_fixed_point_signed[1] = coeff_list[0] # B
    b_fixed_point_signed[0] = coeff_list[1] # A
    b_fixed_point_signed[2] = coeff_list[1] # A (again, we assume)
    
    C[2] = coeff_list[2]#coeff_file.write("%d\n"%(int(C[2])))# C2
    C[3] = coeff_list[3]#coeff_file.write("%d\n"%(int(C[3])))# C3
    C[1] = coeff_list[4]#coeff_file.write("%d\n"%(int(C[1])))# C1
    C[0] = coeff_list[5]#coeff_file.write("%d\n"%(int(C[0])))# C0
    
    a_fixed_point_signed[2] = coeff_list[6]#coeff_file.write("%d\n"%(int(a_fixed_point_signed[2])))# a2'
    a_fixed_point_signed[1] = coeff_list[7]#coeff_file.write("%d\n"%(int(a_fixed_point_signed[1])))# a1'        
    
    D_FF = coeff_list[8]#coeff_file.write("%d\n"%(int(D_FF)))# D_FF
    f_fir[5] = coeff_list[9]# X6
    f_fir[4] = coeff_list[10]# X5
    f_fir[3] = coeff_list[11]# X4
    f_fir[2] = coeff_list[12]# X3
    f_fir[1] = coeff_list[13]# X2
    f_fir[0] = coeff_list[14]# X1
    
    E_GG = coeff_list[15]# E_GG
    g_fir[6] = coeff_list[16]# X7
    g_fir[5] = coeff_list[17]# X6
    g_fir[4] = coeff_list[18]# X5
    g_fir[3] = coeff_list[19]# X4
    g_fir[2] = coeff_list[20]# X3
    g_fir[1] = coeff_list[21]# X2
    g_fir[0] = coeff_list[22]# X1
    
    D_FG = coeff_list[23] # D_FG
    
    E_GF = coeff_list[24] # E_GF
    
    a_fixed_point_signed_all[0] = coeff_list[25]
    a_fixed_point_signed_all[1] = coeff_list[26]
    a_fixed_point_signed_all[2] = coeff_list[27]
    b_fixed_point_signed_all[0] = coeff_list[28]
    b_fixed_point_signed_all[1] = coeff_list[29]
    b_fixed_point_signed_all[2] = coeff_list[30]
    
    coeffs_fixed_point_signed[0:samp_per_clock-2] = f_fir
    coeffs_fixed_point_signed[samp_per_clock-2:2*samp_per_clock-3] = g_fir
    coeffs_fixed_point_signed[2*samp_per_clock-3 + 0] = D_FF
    coeffs_fixed_point_signed[2*samp_per_clock-3 + 1] = D_FG
    coeffs_fixed_point_signed[2*samp_per_clock-3 + 2] = E_GF
    coeffs_fixed_point_signed[2*samp_per_clock-3 + 3] = E_GG
    coeffs_fixed_point_signed[2*samp_per_clock-3 + 4:2*samp_per_clock-3 + 4 + 4] = C
    return coeffs_fixed_point_signed, a_fixed_point_signed_all, b_fixed_point_signed_all


def write_coeffs(notch_freq, q_factor, added_precision=0, file_prefix="001_files/coefficients/coeff_file"):
    """Write out the BQ coefficients to a file"""
    b, a = signal.iirnotch(notch_freq, q_factor, 3000)
    pole = signal.tf2zpk(b,a)[1][0]
    zero = signal.tf2zpk(b,a)[0][0]
    pmag=np.abs(pole)
    pangle=np.angle(pole)
    zmag=np.abs(zero)
    zangle=np.angle(zero)
    
    # Get the coefficients for the quantized biquad
    coeffs = iir_biquad_coeffs(pmag, pangle)
    coeffs_fixed_point = np.zeros(len(coeffs), dtype=np.int64)
    coeffs_fixed_point_signed = np.zeros(len(coeffs), dtype=np.int64)
    b_fixed_point_signed = np.zeros(len(b))
    a_fixed_point_signed = np.zeros(len(a))
    
    coeffs_fixed_point_extended = np.zeros(len(coeffs), dtype=np.int64)
    coeffs_fixed_point_signed_extended = np.zeros(len(coeffs), dtype=np.int64)
    b_fixed_point_signed_extended = np.zeros(len(b))
    a_fixed_point_signed_extended = np.zeros(len(a))
    
    ## Usual Fixed Point
    # For transfer function numerator
    for i in range(len(b)):
        b_fixed_point_signed[i] = np.floor(b[i] * (2**14))
    b_fixed_point_signed = np.array(b_fixed_point_signed, dtype=np.int64)
    # Just in case we want to compare these later
    # For transfer function denominator, after look-ahead
    for i in range(len(a)):
        a_fixed_point_signed[i] = np.floor(a[i] * (2**14))
    a_fixed_point_signed = np.array(a_fixed_point_signed, dtype=np.int64)
    #  For clustered look-ahead
    for i in range(len(coeffs_fixed_point)):
        # Coefficients are in Q4.14, where the sign bit IS counted
        coeffs_fixed_point_signed[i] = np.floor(coeffs[i] * (2**14))
    
    coeffs_fixed_point_signed = np.array(coeffs_fixed_point_signed, dtype=np.int64)
    
    ## Extended Fixed Point
    # For transfer function numerator
    for i in range(len(b)):
        b_fixed_point_signed_extended[i] = np.floor(b[i] * (2**(14+added_precision)))
    # Just in case we want to compare these later
    # For transfer function denominator, after look-ahead
    for i in range(len(a)):
        a_fixed_point_signed_extended[i] = np.floor(a[i] * (2**(14+added_precision)))
    #  For clustered look-ahead
    for i in range(len(coeffs_fixed_point)):
        # Coefficients are in Q4.14, where the sign bit IS counted
        coeffs_fixed_point_signed_extended[i] = np.floor(coeffs[i] * (2**(14+added_precision)))

    
    samp_per_clock=8
    f_fir = coeffs_fixed_point_signed[0:samp_per_clock-2]
    g_fir = coeffs_fixed_point_signed[samp_per_clock-2:2*samp_per_clock-3]
    D_FF = coeffs_fixed_point_signed[2*samp_per_clock-3 + 0]
    D_FG = coeffs_fixed_point_signed[2*samp_per_clock-3 + 1]
    E_GF = coeffs_fixed_point_signed[2*samp_per_clock-3 + 2]
    E_GG = coeffs_fixed_point_signed[2*samp_per_clock-3 + 3]
    C = coeffs_fixed_point_signed[2*samp_per_clock-3 + 4:2*samp_per_clock-3 + 4 + 4]

    
    with open("%s_%sMHz_%s.dat"%(file_prefix, notch_freq,q_factor),"w") as coeff_file:
        coeff_file.write("%d\n"%(int(b_fixed_point_signed[1])))# B
        coeff_file.write("%d\n"%(int(b_fixed_point_signed[0])))# A

        coeff_file.write("%d\n"%(int(C[2])))# C2
        coeff_file.write("%d\n"%(int(C[3])))# C3
        coeff_file.write("%d\n"%(int(C[1])))# C1
        coeff_file.write("%d\n"%(int(C[0])))# C0

        coeff_file.write("%d\n"%(-1*int(a_fixed_point_signed[1])))# a1'
        coeff_file.write("%d\n"%(-1*int(a_fixed_point_signed[2])))# a2'        

        coeff_file.write("%d\n"%(int(D_FF)))# D_FF
        coeff_file.write("%d\n"%(int(f_fir[5])))# X6
        coeff_file.write("%d\n"%(int(f_fir[4])))# X5
        coeff_file.write("%d\n"%(int(f_fir[3])))# X4
        coeff_file.write("%d\n"%(int(f_fir[2])))# X3
        coeff_file.write("%d\n"%(int(f_fir[1])))# X2
        coeff_file.write("%d\n"%(int(f_fir[0])))# X1

        coeff_file.write("%d\n"%(int(E_GG)))# E_GG
        coeff_file.write("%d\n"%(int(g_fir[6])))# X7
        coeff_file.write("%d\n"%(int(g_fir[5])))# X6
        coeff_file.write("%d\n"%(int(g_fir[4])))# X5
        coeff_file.write("%d\n"%(int(g_fir[3])))# X4
        coeff_file.write("%d\n"%(int(g_fir[2])))# X3
        coeff_file.write("%d\n"%(int(g_fir[1])))# X2
        coeff_file.write("%d\n"%(int(g_fir[0])))# X1
        
        coeff_file.write("%d\n"%(int(D_FG)))# D_FG
        
        coeff_file.write("%d\n"%(int(E_GF)))# E_GF

        for a_i in a_fixed_point_signed:
            coeff_file.write("%d\n"%(int(a_i)))
        for b_i in b_fixed_point_signed:
            coeff_file.write("%d\n"%(int(b_i)))


def gaussian_probe_frequency_response(trials, notch, q_factor, smooth=1, savename=None, label_cancellations=True):
    """ Find and plot the frequency response using a windowed gaussian. Note: This has hardcoded paths."""
    TRIALS = trials
    NOTCH = notch
    Q_FACTOR = q_factor
    SAMPLE_FREQ = 3000
    DEBUG=0
    
    b, a = signal.iirnotch(NOTCH,Q_FACTOR, SAMPLE_FREQ)
    pole = signal.tf2zpk(b,a)[1][0]
    zero = signal.tf2zpk(b,a)[0][0]
    pmag=np.abs(pole)
    pangle=np.angle(pole)
    zmag=np.abs(zero)
    zangle=np.angle(zero)
    
    characteristic_poly0 = [1,0,0,0,0,0,0,-1*eval_chebyu(7,np.cos(pangle))*(pmag**7),eval_chebyu(6,np.cos(pangle))*(pmag**8)] # M=7
    roots0 = np.roots(characteristic_poly0)
    cancel_mags0 = np.abs(roots0)
    cancel_angles0 = np.angle(roots0)
        
    characteristic_poly1 = [1,0,0,0,0,0,0,0,-1*eval_chebyu(8,np.cos(pangle))*(pmag**8),eval_chebyu(7,np.cos(pangle))*(pmag**9)] # M=8
    roots1 = np.roots(characteristic_poly1)
    cancel_mags1 = np.abs(roots1)
    cancel_angles1 = np.angle(roots1)
    
    coeffs_fixed_point_signed, a_fixed_point_signed, b_fixed_point_signed = get_coeffs("001_files/coefficients/coeff_file_%sMHz_%s.dat"%(NOTCH,Q_FACTOR))


    # Import the Gaussian input data once to get the lengths
    data_len = len(import_data("001_files/inputs/gauss_input_400_sigma_hanning_clipped_%d.dat"%0))
    fft_input_results = np.zeros((TRIALS, data_len))     
    fft_patrick_results = np.zeros((TRIALS, data_len))     
    fft_python_results = np.zeros((TRIALS, data_len))   
    fft_lfilter_results = np.zeros((TRIALS, data_len))
    
    # Run the trials on the data
    for i in range(TRIALS):
        # Import the trial inputs
        gauss_samps = import_data("001_files/inputs/gauss_input_400_sigma_hanning_clipped_%d.dat"%i)
    
        # Run the filters on the trials
        patrick_data = lfilter(b,[1],iir_biquad_patrick(gauss_samps.copy(), 8, pmag, pangle, ics = None))
        lfilter_data = lfilter(b,a,gauss_samps)
    
        python_data_IIR = iir_biquad_run_fixed_point(gauss_samps.copy(), coeffs_fixed_point_signed, 
                                             decimate=False, a1=a_fixed_point_signed[1], a2=a_fixed_point_signed[2],debug=DEBUG) 
        python_data = np.right_shift(np.array(np.floor(lfilter(np.array(b_fixed_point_signed, dtype=float),[1],python_data_IIR.copy())),dtype=np.int64),14)
        
        fft_input_results[i] = np.abs(np.fft.fft(gauss_samps))**2   
        fft_patrick_results[i] = np.abs(np.fft.fft(patrick_data))**2
        fft_lfilter_results[i] = np.abs(np.fft.fft(lfilter_data))**2
        fft_python_results[i] = np.abs(np.fft.fft(python_data))**2

    
    fft_input_result = np.mean(fft_input_results, axis=0)
    fft_patrick_result = np.mean(fft_patrick_results, axis=0)
    fft_lfilter_result = np.mean(fft_lfilter_results, axis=0)
    fft_python_result = np.mean(fft_python_results, axis=0)

    
    mean_power = np.mean(fft_input_result)

    # Plotting
    
    plot_frequencies = SAMPLE_FREQ*np.fft.fftfreq(len(fft_input_result))[int(np.floor(smooth/2.0)):int(-np.floor(smooth/2.0))]

    fig = plt.figure()
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    ax0 = plt.subplot(gs[0])
    lfilter_line, = ax0.plot(plot_frequencies,np.convolve(np.abs(fft_lfilter_result),  np.ones(smooth)/smooth, mode='valid')/mean_power, 
                            linestyle="None", marker=".", markersize=1, alpha=0.8, color="C0")
    ax0.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C0", label="Theory (lfilter)")

    lfilter_line, = ax0.plot(plot_frequencies,np.convolve(np.abs(fft_patrick_result), np.ones(smooth)/smooth, mode='valid')/mean_power, 
                            linestyle="None", marker=".", markersize=1, alpha=0.8, color="C1")
    ax0.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C1", label="Floating Point CLA")

    lfilter_line, = ax0.plot(plot_frequencies,np.convolve(np.abs(fft_python_result),  np.ones(smooth)/smooth, mode='valid')/mean_power, 
                            linestyle="None", marker=".", markersize=1, alpha=0.8, color="C2")
    ax0.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C2", label="Fixed Point CLA")


    # float_line, = ax0.plot(plot_frequencies,np.convolve(np.abs(fft_patrick_result), np.ones(smooth)/smooth, mode='valid')/mean_power, linestyle="None", 
    #                          marker=".", label="Floating Point CLA", markersize=5, alpha=0.8, color="C1")
    # fixed_line, = ax0.plot(plot_frequencies,np.convolve(np.abs(fft_python_result),  np.ones(smooth)/smooth, mode='valid')/mean_power, linestyle="None",
    #                          marker=".", label="Fixed Point CLA", markersize=5, alpha=0.8, color="C2")
    ax0.set_yscale("log")
    ax0.set_ylim(bottom=10**-1)
    ax0.set_xlim(SAMPLE_FREQ*-0.51,SAMPLE_FREQ*0.51)
    
    plt.setp(ax0.get_xticklabels(), visible=False)
    ax0.set_ylabel("Gain")
    ax0.set_title("Notch: %s MHz, Q: %s"%(NOTCH, Q_FACTOR))
    
    yticks0 = ax0.yaxis.get_major_ticks()
    yticks0[-1].label1.set_visible(False)

    
    ax1 = plt.subplot(gs[1], sharex = ax0)
    diff_line, = ax1.plot(plot_frequencies,(np.convolve(np.abs(fft_python_result), np.ones(smooth)/smooth, mode='valid') - np.convolve(np.abs(fft_patrick_result),  np.ones(smooth)/smooth, mode='valid'))/mean_power, linestyle="None", 
                         marker=".", markersize=5, alpha=0.8, color="C3")
    ax1.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C3", label="Fixed - Floating")
    diff_line_2, = ax1.plot(plot_frequencies,(np.convolve(np.abs(fft_python_result), np.ones(smooth)/smooth, mode='valid') - np.convolve(np.abs(fft_lfilter_result),  np.ones(smooth)/smooth, mode='valid'))/mean_power, linestyle="None", 
                     marker=".", markersize=5, alpha=0.8, color="C4")
    ax1.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C4", label="Fixed - Theory")
    if(SAMPLE_FREQ > 1):
        ax1.set_xlabel("Freq (MHz)")
    else:
        ax1.set_xlabel("Frequency/(Sampling Frequency)")

    if(label_cancellations):
        for a in cancel_angles0:
            ax0.axvline(x=a*SAMPLE_FREQ/(2*np.pi), linestyle=":", color="grey", linewidth=2.0)
            ax1.axvline(x=a*SAMPLE_FREQ/(2*np.pi), linestyle=":", color="grey", linewidth=2.0)
        ax0.plot([],[], linestyle=":", color="grey", linewidth=2.0, label="Cancellation 0")
        ax1.plot([],[], linestyle=":", color="grey", linewidth=2.0, label="Cancellation 0")
        for a in cancel_angles1:
            ax0.axvline(x=a*SAMPLE_FREQ/(2*np.pi), linestyle="--", color="black", linewidth=1.0)
            ax1.axvline(x=a*SAMPLE_FREQ/(2*np.pi), linestyle="--", color="black", linewidth=1.0)
        ax0.plot([],[], linestyle=":", color="black", linewidth=1.0, label="Cancellation 1")
        ax1.plot([],[], linestyle=":", color="black", linewidth=1.0, label="Cancellation 1")
    
    # yticks = ax1.yaxis.get_major_ticks()
    # yticks[0].label1.set_visible(False)
    # ax1.set_xticks(ax1.get_xticks()[0:-1])
    # ax0.legend((lfilter_line, float_line, fixed_line, diff_line, diff_line_2), ("lfilter (theory)", "floating point", "fixed", "fixed - float", "fixed - lfilter"), bbox_to_anchor=(1.04,1))
    ax0.legend(bbox_to_anchor=(1.04,1))
    ax1.legend(bbox_to_anchor=(1.04,1))
    if(not savename is None):
        plt.savefig(savename + ".png", bbox_inches='tight')
    
    plt.subplots_adjust(hspace=.0)
    plt.show()
    
    plt.clf()
    fig = plt.figure(num='Biquad CLA Cancels',figsize=(10,4.8))
    ax=fig.add_subplot(1,10,(1,10),projection='polar')
    ax.set_rlim(0,1.30)

    
    poles_cancel0 = ax.scatter(cancel_angles0,cancel_mags0,c='grey',marker="x", s=75, label="Cancellation Poles 0")
    zeros_cancel0 = ax.scatter(cancel_angles0,cancel_mags0, marker="o", s=75,facecolors='none', edgecolors='grey', label="Cancellation Zeros 0")

    poles_cancel1 = ax.scatter(cancel_angles1,cancel_mags1,c='black',marker="x", s=75, label="Cancellation Poles 1")
    zeros_cancel1 = ax.scatter(cancel_angles1,cancel_mags1, marker="o", s=75,facecolors='none', edgecolors='black', label="Cancellation Zeros 1")
    
    poles = ax.scatter([pangle, -1*pangle],[pmag, pmag],c='red',marker="x", s=75, label="Notch Poles")
    zeros = ax.scatter([zangle, -1*zangle],[zmag, zmag], marker="o", s=75,facecolors='none', edgecolors='red', label="Notch Angles")
    
    unit_circle = ax.plot(np.linspace(0,2*np.pi,1000), np.ones(1000), color="red", linestyle="--")
    ax.legend(bbox_to_anchor=(1.04,1))
    ax.set_title("Notch: %s MHz, Q: %s"%(NOTCH, Q_FACTOR))
    if(not savename is None):
        plt.savefig(savename + "_polezero.png", bbox_inches='tight')
    plt.show()


def gaussian_probe_frequency_response_manual(trials, A, B, pmag, pangle, smooth=1, savename=None, label_cancellations=True):
    """ Find and plot the frequency response using a windowed gaussian. Use manaually generated coefficients (rather than scipy.signal). Note: This has hardcoded paths."""
    TRIALS = trials

    b = [A,B,A]#, a = signal.iirnotch(notch_freq,Q_FACTOR, 3000)
    tf = signal.zpk2tf([],[pmag * np.exp(1j*pangle), pmag * np.exp(-1j*pangle)], 1)
    a = tf[1]
    
    SAMPLE_FREQ = 3000
    DEBUG=0

    # pole = signal.tf2zpk(b,a)[1][0]
    zero = signal.tf2zpk(b,a)[0][0]

    zmag=np.abs(zero)
    zangle=np.angle(zero)
    
    characteristic_poly0 = [1,0,0,0,0,0,0,-1*eval_chebyu(7,np.cos(pangle))*(pmag**7),eval_chebyu(6,np.cos(pangle))*(pmag**8)] # M=7
    roots0 = np.roots(characteristic_poly0)
    cancel_mags0 = np.abs(roots0)
    cancel_angles0 = np.angle(roots0)
        
    characteristic_poly1 = [1,0,0,0,0,0,0,0,-1*eval_chebyu(8,np.cos(pangle))*(pmag**8),eval_chebyu(7,np.cos(pangle))*(pmag**9)] # M=8
    roots1 = np.roots(characteristic_poly1)
    cancel_mags1 = np.abs(roots1)
    cancel_angles1 = np.angle(roots1)


    
    # Get the coefficients for the quantized biquad
    coeffs = iir_biquad_coeffs(pmag, pangle)
    coeffs_fixed_point = np.zeros(len(coeffs), dtype=np.int64)
    coeffs_fixed_point_signed = np.zeros(len(coeffs), dtype=np.int64)
    b_fixed_point_signed = np.zeros(len(b))
    a_fixed_point_signed = np.zeros(len(a))
    
    coeffs_fixed_point_extended = np.zeros(len(coeffs), dtype=np.int64)
    coeffs_fixed_point_signed_extended = np.zeros(len(coeffs), dtype=np.int64)
    b_fixed_point_signed_extended = np.zeros(len(b))
    a_fixed_point_signed_extended = np.zeros(len(a))
    
    ## Usual Fixed Point
    # For transfer function numerator
    for i in range(len(b)):
        b_fixed_point_signed[i] = np.floor(b[i] * (2**14))
    b_fixed_point_signed = np.array(b_fixed_point_signed, dtype=np.int64)
    # Just in case we want to compare these later
    # For transfer function denominator, after look-ahead
    for i in range(len(a)):
        a_fixed_point_signed[i] = np.floor(a[i] * (2**14))
    a_fixed_point_signed = np.array(a_fixed_point_signed, dtype=np.int64)
    #  For clustered look-ahead
    for i in range(len(coeffs_fixed_point)):
        # Coefficients are in Q4.14, where the sign bit IS counted
        coeffs_fixed_point_signed[i] = np.floor(coeffs[i] * (2**14))
    
    coeffs_fixed_point_signed = np.array(coeffs_fixed_point_signed, dtype=np.int64)


    # Import the Gaussian input data once to get the lengths
    data_len = len(import_data("001_files/inputs/gauss_input_400_sigma_hanning_clipped_%d.dat"%0))
    fft_input_results = np.zeros((TRIALS, data_len))     
    fft_patrick_results = np.zeros((TRIALS, data_len))     
    fft_python_results = np.zeros((TRIALS, data_len))   
    fft_lfilter_results = np.zeros((TRIALS, data_len))
    
    # Run the trials on the data
    for i in range(TRIALS):
        # Import the trial inputs
        gauss_samps = import_data("001_files/inputs/gauss_input_400_sigma_hanning_clipped_%d.dat"%i)
    
        # Run the filters on the trials
        patrick_data = lfilter(b,[1],iir_biquad_patrick(gauss_samps.copy(), 8, pmag, pangle, ics = None))
        lfilter_data = lfilter(b,a,gauss_samps)
    
        python_data_IIR = iir_biquad_run_fixed_point(gauss_samps.copy(), coeffs_fixed_point_signed, 
                                             decimate=False, a1=a_fixed_point_signed[1], a2=a_fixed_point_signed[2],debug=DEBUG) 
        python_data = np.right_shift(np.array(np.floor(lfilter(np.array(b_fixed_point_signed, dtype=float),[1],python_data_IIR.copy())),dtype=np.int64),14)
        
        fft_input_results[i] = np.abs(np.fft.fft(gauss_samps))**2   
        fft_patrick_results[i] = np.abs(np.fft.fft(patrick_data))**2
        fft_lfilter_results[i] = np.abs(np.fft.fft(lfilter_data))**2
        fft_python_results[i] = np.abs(np.fft.fft(python_data))**2

    
    fft_input_result = np.mean(fft_input_results, axis=0)
    fft_patrick_result = np.mean(fft_patrick_results, axis=0)
    fft_lfilter_result = np.mean(fft_lfilter_results, axis=0)
    fft_python_result = np.mean(fft_python_results, axis=0)

    
    mean_power = np.mean(fft_input_result)

    # Plotting
    
    plot_frequencies = SAMPLE_FREQ*np.fft.fftfreq(len(fft_input_result))[int(np.floor(smooth/2.0)):int(-np.floor(smooth/2.0))]

    fig = plt.figure()
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    ax0 = plt.subplot(gs[0])
    lfilter_line, = ax0.plot(plot_frequencies,np.convolve(np.abs(fft_lfilter_result),  np.ones(smooth)/smooth, mode='valid')/mean_power, 
                            linestyle="None", marker=".", markersize=1, alpha=0.8, color="C0")
    ax0.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C0", label="Theory (lfilter)")

    lfilter_line, = ax0.plot(plot_frequencies,np.convolve(np.abs(fft_patrick_result), np.ones(smooth)/smooth, mode='valid')/mean_power, 
                            linestyle="None", marker=".", markersize=1, alpha=0.8, color="C1")
    ax0.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C1", label="Floating Point CLA")

    lfilter_line, = ax0.plot(plot_frequencies,np.convolve(np.abs(fft_python_result),  np.ones(smooth)/smooth, mode='valid')/mean_power, 
                            linestyle="None", marker=".", markersize=1, alpha=0.8, color="C2")
    ax0.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C2", label="Fixed Point CLA")


    # float_line, = ax0.plot(plot_frequencies,np.convolve(np.abs(fft_patrick_result), np.ones(smooth)/smooth, mode='valid')/mean_power, linestyle="None", 
    #                          marker=".", label="Floating Point CLA", markersize=5, alpha=0.8, color="C1")
    # fixed_line, = ax0.plot(plot_frequencies,np.convolve(np.abs(fft_python_result),  np.ones(smooth)/smooth, mode='valid')/mean_power, linestyle="None",
    #                          marker=".", label="Fixed Point CLA", markersize=5, alpha=0.8, color="C2")
    ax0.set_yscale("log")
    ax0.set_ylim(bottom=10**-1)
    ax0.set_xlim(SAMPLE_FREQ*-0.51,SAMPLE_FREQ*0.51)
    
    plt.setp(ax0.get_xticklabels(), visible=False)
    ax0.set_ylabel("Gain")
    ax0.set_title("Hugo Email Parameters")
    
    yticks0 = ax0.yaxis.get_major_ticks()
    yticks0[-1].label1.set_visible(False)

    
    ax1 = plt.subplot(gs[1], sharex = ax0)
    diff_line, = ax1.plot(plot_frequencies,(np.convolve(np.abs(fft_python_result), np.ones(smooth)/smooth, mode='valid') - np.convolve(np.abs(fft_patrick_result),  np.ones(smooth)/smooth, mode='valid'))/mean_power, linestyle="None", 
                         marker=".", markersize=5, alpha=0.8, color="C3")
    ax1.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C3", label="Fixed - Floating")
    diff_line_2, = ax1.plot(plot_frequencies,(np.convolve(np.abs(fft_python_result), np.ones(smooth)/smooth, mode='valid') - np.convolve(np.abs(fft_lfilter_result),  np.ones(smooth)/smooth, mode='valid'))/mean_power, linestyle="None", 
                     marker=".", markersize=5, alpha=0.8, color="C4")
    ax1.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C4", label="Fixed - Theory")
    if(SAMPLE_FREQ > 1):
        ax1.set_xlabel("Freq (MHz)")
    else:
        ax1.set_xlabel("Frequency/(Sampling Frequency)")

    if(label_cancellations):
        for a in cancel_angles0:
            ax0.axvline(x=a*SAMPLE_FREQ/(2*np.pi), linestyle=":", color="grey", linewidth=2.0)
            ax1.axvline(x=a*SAMPLE_FREQ/(2*np.pi), linestyle=":", color="grey", linewidth=2.0)
        ax0.plot([],[], linestyle=":", color="grey", linewidth=2.0, label="Cancellation 0")
        ax1.plot([],[], linestyle=":", color="grey", linewidth=2.0, label="Cancellation 0")
        for a in cancel_angles1:
            ax0.axvline(x=a*SAMPLE_FREQ/(2*np.pi), linestyle="--", color="black", linewidth=1.0)
            ax1.axvline(x=a*SAMPLE_FREQ/(2*np.pi), linestyle="--", color="black", linewidth=1.0)
        ax0.plot([],[], linestyle=":", color="black", linewidth=1.0, label="Cancellation 1")
        ax1.plot([],[], linestyle=":", color="black", linewidth=1.0, label="Cancellation 1")
    
    # yticks = ax1.yaxis.get_major_ticks()
    # yticks[0].label1.set_visible(False)
    # ax1.set_xticks(ax1.get_xticks()[0:-1])
    # ax0.legend((lfilter_line, float_line, fixed_line, diff_line, diff_line_2), ("lfilter (theory)", "floating point", "fixed", "fixed - float", "fixed - lfilter"), bbox_to_anchor=(1.04,1))
    ax0.legend(bbox_to_anchor=(1.04,1))
    ax1.legend(bbox_to_anchor=(1.04,1))
    if(not savename is None):
        plt.savefig(savename + ".png", bbox_inches='tight')
    
    plt.subplots_adjust(hspace=.0)
    plt.show()
    
    plt.clf()
    fig = plt.figure(num='Biquad CLA Cancels',figsize=(10,4.8))
    ax=fig.add_subplot(1,10,(1,10),projection='polar')
    ax.set_rlim(0,1.30)

    
    poles_cancel0 = ax.scatter(cancel_angles0,cancel_mags0,c='grey',marker="x", s=75, label="Cancellation Poles 0")
    zeros_cancel0 = ax.scatter(cancel_angles0,cancel_mags0, marker="o", s=75,facecolors='none', edgecolors='grey', label="Cancellation Zeros 0")

    poles_cancel1 = ax.scatter(cancel_angles1,cancel_mags1,c='black',marker="x", s=75, label="Cancellation Poles 1")
    zeros_cancel1 = ax.scatter(cancel_angles1,cancel_mags1, marker="o", s=75,facecolors='none', edgecolors='black', label="Cancellation Zeros 1")
    
    poles = ax.scatter([pangle, -1*pangle],[pmag, pmag],c='red',marker="x", s=75, label="Notch Poles")
    zeros = ax.scatter([zangle, -1*zangle],[zmag, zmag], marker="o", s=75,facecolors='none', edgecolors='red', label="Notch Angles")
    
    unit_circle = ax.plot(np.linspace(0,2*np.pi,1000), np.ones(1000), color="red", linestyle="--")
    ax.legend(bbox_to_anchor=(1.04,1))
    ax.set_title("Hugo's Parameters")#"Notch: %s MHz, Q: %s"%(NOTCH, Q_FACTOR))
    if(not savename is None):
        plt.savefig(savename + "_polezero.png", bbox_inches='tight')
    plt.show()


def frequency_response_manual(clocks, input_data, output_data, smooth=1, savename=None, lfilter_coeffs=None, SAMPLE_FREQ=3000, show=True):
    """Plot the frequency response for (more) arbitrary data"""
    TRIALS = len(input_data)
    data_len = len(input_data[0])
    
    fft_input_results = np.zeros((TRIALS, data_len))     
    fft_output_results = np.zeros((TRIALS, data_len))    
    if(not (lfilter_coeffs is None)):
        fft_lfilter_results = np.zeros((TRIALS, data_len))
        b = lfilter_coeffs[0]
        a = lfilter_coeffs[1]
    
    # Run the trials on the data
    for i in range(TRIALS):
        # Import the trial inputs
        
        if(not (lfilter_coeffs is None)):
            lfilter_data = lfilter(b,a,input_data[i])

        
        fft_input_results[i] = np.abs(np.fft.fft(input_data[i]))**2   
        fft_output_results[i] = np.abs(np.fft.fft(output_data[i]))**2
        if(not (lfilter_coeffs is None)):
            fft_lfilter_results[i] = np.abs(np.fft.fft(lfilter_data))**2

    
    fft_input_result = np.mean(fft_input_results, axis=0)
    fft_output_result = np.mean(fft_output_results, axis=0)
    if(not (lfilter_coeffs is None)):
        fft_lfilter_result = np.mean(fft_lfilter_results, axis=0)

    
    mean_power = np.mean(fft_input_result)
    # print(mean_power)

    # Plotting
    # print(fft_input_result)    
    # print( SAMPLE_FREQ*np.fft.fftfreq(len(fft_input_result)))
    plot_frequencies = SAMPLE_FREQ*np.fft.fftfreq(len(fft_input_result))[int(np.floor(smooth/2.0)):int(-np.floor(smooth/2.0))]
    # print(plot_frequencies)
    
    fig = plt.figure()
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    ax0 = plt.subplot(gs[0])
 
    if(not (lfilter_coeffs is None)):
        lfilter_line, = ax0.plot(plot_frequencies,np.convolve(np.abs(fft_lfilter_result),  np.ones(smooth)/smooth, mode='valid')/mean_power, 
                                linestyle="None", marker=".", markersize=1, alpha=0.8, color="C0")
        ax0.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C0", label="Theory (lfilter)")
    
    output_line, = ax0.plot(plot_frequencies,np.convolve(np.abs(fft_output_result), np.ones(smooth)/smooth, mode='valid')/mean_power, 
                            linestyle="None", marker=".", markersize=1, alpha=0.8, color="C1")
    ax0.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C1", label="Output")
   
    # lfilter_line, = ax0.plot(plot_frequencies,np.convolve(np.abs(fft_python_result),  np.ones(smooth)/smooth, mode='valid')/mean_power, 
    #                         linestyle="None", marker=".", markersize=1, alpha=0.8, color="C2")
    # ax0.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C2", label="Fixed Point CLA")


    # float_line, = ax0.plot(plot_frequencies,np.convolve(np.abs(fft_patrick_result), np.ones(smooth)/smooth, mode='valid')/mean_power, linestyle="None", 
    #                          marker=".", label="Floating Point CLA", markersize=5, alpha=0.8, color="C1")
    # fixed_line, = ax0.plot(plot_frequencies,np.convolve(np.abs(fft_python_result),  np.ones(smooth)/smooth, mode='valid')/mean_power, linestyle="None",
    #                          marker=".", label="Fixed Point CLA", markersize=5, alpha=0.8, color="C2")
    ax0.set_yscale("log")
    ax0.set_ylim(bottom=10**-1)
    ax0.set_xlim(SAMPLE_FREQ*-0.51,SAMPLE_FREQ*0.51)
    
    plt.setp(ax0.get_xticklabels(), visible=False)
    ax0.set_ylabel("Gain")
    ax0.set_title("Frequency Response")
    
    yticks0 = ax0.yaxis.get_major_ticks()
    yticks0[-1].label1.set_visible(False)

    
    if(not (lfilter_coeffs is None)):
        ax1 = plt.subplot(gs[1], sharex = ax0)
        # diff_line, = ax1.plot(plot_frequencies,(np.convolve(np.abs(fft_output_result), np.ones(smooth)/smooth, mode='valid') - np.convolve(np.abs(fft_patrick_result),  np.ones(smooth)/smooth, mode='valid'))/mean_power, linestyle="None", 
        #                      marker=".", markersize=5, alpha=0.8, color="C3")
        # ax1.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C3", label="Fixed - Floating")
        diff_line_2, = ax1.plot(plot_frequencies,(np.convolve(np.abs(fft_output_result), np.ones(smooth)/smooth, mode='valid') - np.convolve(np.abs(fft_lfilter_result),  np.ones(smooth)/smooth, mode='valid'))/mean_power, linestyle="None", 
                         marker=".", markersize=5, alpha=0.8, color="C4")
        ax1.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C4", label="Output - Theory")
        if(SAMPLE_FREQ > 1):
            ax1.set_xlabel("Freq (MHz)")
        else:
            ax1.set_xlabel("Frequency/(Sampling Frequency)")
            
        ax1.legend(bbox_to_anchor=(1.04,1))
        
    # yticks = ax1.yaxis.get_major_ticks()
    # yticks[0].label1.set_visible(False)
    # ax1.set_xticks(ax1.get_xticks()[0:-1])
    # ax0.legend((lfilter_line, float_line, fixed_line, diff_line, diff_line_2), ("lfilter (theory)", "floating point", "fixed", "fixed - float", "fixed - lfilter"), bbox_to_anchor=(1.04,1))
    ax0.legend(bbox_to_anchor=(1.04,1))
    plt.subplots_adjust(hspace=.0)
    if(not savename is None):
        plt.savefig(savename + ".png", bbox_inches='tight', dpi=300)
    
    if(show):
        plt.show()
    
    # plt.clf()
    # fig = plt.figure(num='Biquad CLA Cancels',figsize=(10,4.8))
    # ax=fig.add_subplot(1,10,(1,10),projection='polar')
    # ax.set_rlim(0,1.30)

    
    # poles_cancel0 = ax.scatter(cancel_angles0,cancel_mags0,c='grey',marker="x", s=75, label="Cancellation Poles 0")
    # zeros_cancel0 = ax.scatter(cancel_angles0,cancel_mags0, marker="o", s=75,facecolors='none', edgecolors='grey', label="Cancellation Zeros 0")

    # poles_cancel1 = ax.scatter(cancel_angles1,cancel_mags1,c='black',marker="x", s=75, label="Cancellation Poles 1")
    # zeros_cancel1 = ax.scatter(cancel_angles1,cancel_mags1, marker="o", s=75,facecolors='none', edgecolors='black', label="Cancellation Zeros 1")
    
    # poles = ax.scatter([pangle, -1*pangle],[pmag, pmag],c='red',marker="x", s=75, label="Notch Poles")
    # zeros = ax.scatter([zangle, -1*zangle],[zmag, zmag], marker="o", s=75,facecolors='none', edgecolors='red', label="Notch Angles")
    
    # unit_circle = ax.plot(np.linspace(0,2*np.pi,1000), np.ones(1000), color="red", linestyle="--")
    # ax.legend(bbox_to_anchor=(1.04,1))
    # ax.set_title("Hugo's Parameters")#"Notch: %s MHz, Q: %s"%(NOTCH, Q_FACTOR))
    # if(not savename is None):
    #     plt.savefig(savename + "_polezero.png", bbox_inches='tight')
    # plt.show()

def frequency_response_manual_v2(clocks, input_data, output_data, smooth=1, savename=None, lfilter_coeffs=None, SAMPLE_FREQ=3000, show=True, title=None, diffscale=None, yscale = (-30,30)):
    """Plot the frequency response for (more) arbitrary data"""
    TRIALS = len(input_data)
    data_len = len(input_data[0])
    
    fft_input_results = np.zeros((TRIALS, data_len))     
    fft_output_results = np.zeros((TRIALS, data_len))    
    if(not (lfilter_coeffs is None)):
        fft_lfilter_results = np.zeros((TRIALS, data_len))
        b = lfilter_coeffs[0]
        a = lfilter_coeffs[1]
    
    # Run the trials on the data
    for i in range(TRIALS):
        # Import the trial inputs
        
        if(not (lfilter_coeffs is None)):
            lfilter_data = lfilter(b,a,input_data[i])

        
        fft_input_results[i] = np.abs(np.fft.fft(input_data[i]))**2   
        fft_output_results[i] = np.abs(np.fft.fft(output_data[i]))**2
        if(not (lfilter_coeffs is None)):
            fft_lfilter_results[i] = np.abs(np.fft.fft(lfilter_data))**2

    
    fft_input_result = np.mean(fft_input_results, axis=0)
    fft_output_result = np.mean(fft_output_results, axis=0)
    if(not (lfilter_coeffs is None)):
        fft_lfilter_result = np.mean(fft_lfilter_results, axis=0)

    
    # mean_power = np.mean(fft_input_result)
   
    # Plotting
    # print(fft_input_result)    
    # print( SAMPLE_FREQ*np.fft.fftfreq(len(fft_input_result)))
    plot_frequencies = SAMPLE_FREQ*np.fft.fftfreq(len(fft_input_result))[int(np.floor(smooth/2.0)):int(-np.floor(smooth/2.0))]
    # print(plot_frequencies)
    
    fig = plt.figure()
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    ax0 = plt.subplot(gs[0])
 
    if(not (lfilter_coeffs is None)):
        lfilter_ratios = np.convolve(np.divide(np.abs(fft_lfilter_result), np.abs(fft_input_result)),  np.ones(smooth)/smooth, mode='valid')
        lfilter_db = 10*np.log10(lfilter_ratios)
        lfilter_line, = ax0.plot(plot_frequencies,lfilter_db, 
                                linestyle="None", marker=".", markersize=1, alpha=0.8, color="C0")
        ax0.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C0", label="Theory (lfilter)")

    output_ratios = np.convolve(np.divide(np.abs(fft_output_result), np.abs(fft_input_result)),  np.ones(smooth)/smooth, mode='valid')
    output_db = 10*np.log10(output_ratios)
    output_line, = ax0.plot(plot_frequencies,output_db, 
                            linestyle="None", marker=".", markersize=1, alpha=0.8, color="C1")
    ax0.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C1", label="Output")
   
    # ax0.set_yscale("log")
    ax0.set_ylim(yscale[0],yscale[1])
    ax0.set_xlim(SAMPLE_FREQ*-0.51,SAMPLE_FREQ*0.51)
    
    plt.setp(ax0.get_xticklabels(), visible=False)
    ax0.set_ylabel("Gain (dB)")
    if(title is None):
        ax0.set_title("Frequency Response")
    else:
        ax0.set_title(title)
    yticks0 = ax0.yaxis.get_major_ticks()
    yticks0[-1].label1.set_visible(False)

    ax0.grid(alpha=0.5)

    
    if(not (lfilter_coeffs is None)):
        ax1 = plt.subplot(gs[1], sharex = ax0)
        # diff_line, = ax1.plot(plot_frequencies,(np.convolve(np.abs(fft_output_result), np.ones(smooth)/smooth, mode='valid') - np.convolve(np.abs(fft_patrick_result),  np.ones(smooth)/smooth, mode='valid'))/mean_power, linestyle="None", 
        #                      marker=".", markersize=5, alpha=0.8, color="C3")
        # ax1.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C3", label="Fixed - Floating")
        diff_line_2, = ax1.plot(plot_frequencies,output_db - lfilter_db, linestyle="None", 
                         marker=".", markersize=5, alpha=0.8, color="C4")
        ax1.plot([],[],         linestyle="None", marker=".", markersize=8, alpha=0.8, color="C4", label="Output - Theory")
        ax1.set_ylabel(r'$\Delta$dB')
        if(SAMPLE_FREQ > 1):
            ax1.set_xlabel("Freq (MHz)")
        else:
            ax1.set_xlabel("Frequency/(Sampling Frequency)")
            
        ax1.legend(bbox_to_anchor=(1.35,1), loc="upper right")
        ax1.grid(alpha=0.5)
        if(not diffscale is None):
            ax1.set_ylim(-1,diffscale)
        
    # yticks = ax1.yaxis.get_major_ticks()
    # yticks[0].label1.set_visible(False)
    # ax1.set_xticks(ax1.get_xticks()[0:-1])
    # ax0.legend((lfilter_line, float_line, fixed_line, diff_line, diff_line_2), ("lfilter (theory)", "floating point", "fixed", "fixed - float", "fixed - lfilter"), bbox_to_anchor=(1.04,1))
    ax0.legend(bbox_to_anchor=(1.35,1), loc="upper right")
    plt.subplots_adjust(hspace=.05)
    if(not savename is None):
        plt.savefig(savename + ".png", bbox_inches='tight', dpi=300)
    
    if(show):
        plt.show()
    