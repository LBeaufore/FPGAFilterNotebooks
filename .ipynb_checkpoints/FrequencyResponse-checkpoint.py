import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider,Button
from functools import reduce
from scipy import signal
from IIRSim import *

def sample_generator(frequency, n_samples=8*8, m=8, flatten=False, phase=0, in_amplitude=1.0):
    """Generate n_samples at a given frequency, with m parallelization"""
    indicies = np.arange(n_samples)
    indicies = indicies.reshape(int(n_samples/m), m)
    values = np.cos((np.pi*frequency)*indicies + phase)*in_amplitude
    if flatten:
        values = values.flatten()
        indicies = indicies.flatten()
    return values

def quantized_sample_generator(frequency, n_samples=8*8, m=8, phase=0, in_amplitude=0.9999999999999):
    """Generate n_samples at a given frequency, with m parallelization, and then quantize"""
    if(in_amplitude==1.0):
        in_amplitude =0.9999999999999 
    raw_values = sample_generator(frequency, n_samples=n_samples, m=m, flatten=True, phase=0, in_amplitude=in_amplitude)
    q_values = raw_values *  (2**11)# Cosine goes from -1 to 1, so there is already a factor of 2 here in the sign bitI
    q_values = np.array(np.floor(q_values),dtype=np.int64)
    # q_values = np.left_shift(q_values, 4)
    return q_values

def eqn_from_zeros(zeros=[1J, -1J]):
    """This is for recovering a and b, if you don't have them already """
    if (len(zeros)==0):
        return [1]
    eqns = []
    for zero in zeros:
        eqns.append(np.poly1d([1,-1*zero]))
    full_eqn = reduce(lambda x,y:x*y,eqns)
    return full_eqn.c

def power_ratio(values, values_filtered):
    return (np.sum(np.power(np.abs(values_filtered),2)))/(np.sum(np.power(np.abs(values),2)))

def amp_ratio(values, values_filtered):
    return (np.sum(np.abs(values_filtered)))/(np.sum(np.abs(values)))

def eval_biquad_filter_DFT(zradius, zfreq, pradius, pfreq, sample_freqs, internal_samples=8*100, amplitude=False, phase=0, quantized=True, in_amplitude=1.0):
    n_samples=internal_samples
    topzero=zradius*complex(np.cos(zfreq),np.sin(zfreq))
    botzero=topzero.conjugate()    
    toppole=pradius*complex(np.cos(pfreq),np.sin(pfreq))
    botpole=toppole.conjugate()
    power_ratios = []
    window = np.hanning(n_samples)
    for freq_i in sample_freqs:
        if quantized:
            values = quantized_sample_generator(freq_i, n_samples=n_samples, m=1, phase=phase, in_amplitude=in_amplitude) 
        else:
            values = sample_generator(freq_i, n_samples=n_samples, m=1, flatten=True, phase=phase, in_amplitude=in_amplitude)
        # values = np.multiply(window, values)
        values = np.append(np.multiply(window, values), np.zeros(8*4)) # The padding at the end is to deal with latency
        if quantized:
            values = np.floor(values)
        fft_result = np.fft.fft(values)
        values_filtered = scipy.signal.lfilter(eqn_from_zeros([topzero, botzero]),eqn_from_zeros([toppole,botpole]),values)        
        fft_filter_result = np.fft.fft(values_filtered)
        if amplitude:
            power_ratios.append(amp_ratio(fft_result, fft_filter_result)) 
        else:
            power_ratios.append(power_ratio(fft_result, fft_filter_result))  
            
    power_ratios = power_ratios/np.max(power_ratios)

    return power_ratios

def eval_biquad_quantized_DFT(b, a, sample_freqs, internal_samples=8*100, amplitude=False, phase=0, in_amplitude=1.0):
    """ Evaluate the frequency response of the actual biquad implementation"""

    # First generate the coefficients for the FPGA Biquad
    pole = signal.tf2zpk(b,a)[1][0]
    zero = signal.tf2zpk(b,a)[0][0]
    mag=np.abs(pole)
    angle=np.angle(pole)
    coeffs = iir_biquad_coeffs(mag, angle)
    coeffs_fixed_point = np.zeros(len(coeffs), dtype=np.int64)
    coeffs_fixed_point_signed = np.zeros(len(coeffs), dtype=np.int64)
    b_fixed_point_signed = np.zeros(len(b))
    # For transfer function numerator
    for i in range(len(b)):
        b_fixed_point_signed[i] = np.array(np.floor(b[i] * (2**14)),dtype=np.int64)
    a_fixed_point_signed = np.zeros(len(a))
    # For transfer function denominator, after look-ahead
    for i in range(len(a)):
        a_fixed_point_signed[i] = np.array(np.floor(a[i] * (2**14)),dtype=np.int64)
    # For clustered look-ahead
    for i in range(len(coeffs_fixed_point)):
        # Coefficients are in Q4.14, where the sign bit IS counted
        coeffs_fixed_point_signed[i] = np.array(np.floor(coeffs[i] * (2**14)),dtype=np.int64)
        coeffs_fixed_point[i] = convert_to_fixed_point(np.array(coeffs[i]), 4, 14)
    
    # n_samples=internal_samples
    # topzero=zradius*complex(np.cos(zfreq),np.sin(zfreq))
    # botzero=topzero.conjugate()    
    # toppole=pradius*complex(np.cos(pfreq),np.sin(pfreq))
    # botpole=toppole.conjugate()
    power_ratios = []
    window = np.hanning(internal_samples)
    for freq_i in sample_freqs:
        values = quantized_sample_generator(freq_i, n_samples=internal_samples, m=1, phase=phase, in_amplitude=in_amplitude) 
        values = np.append(np.multiply(window, values), np.zeros(8*4)) # The padding at the end is to deal with latency
        values = np.floor(values)
        fft_result = np.fft.fft(values)

        # Apply the zeros
        #L NOTE: The latency may push some power out of the sample window, if so, increase zeros above
        # Right shift to correct for coefficient size
        values_with_FIR = np.right_shift(np.array(np.floor(signal.lfilter(b_fixed_point_signed, [1], values)),dtype=np.int64),14)
        
        values_filtered = iir_biquad_run_fixed_point(np.array(values_with_FIR), coeffs_fixed_point_signed, 
                                                     decimate=False, a1=a[1] * (1), a2=a[2] * (1),debug=0)      
        fft_filter_result = np.fft.fft(values_filtered)
        if amplitude:
            power_ratios.append(amp_ratio(fft_result, fft_filter_result)) 
        else:
            power_ratios.append(power_ratio(fft_result, fft_filter_result))  
            
    power_ratios = power_ratios/np.max(power_ratios)

    return power_ratios

def plot_filter_polezero(zero_radius=1.0, zero_frequency=0.25, pole_radius=0.5, pole_frequency=0.25):
    """ Make a pole-zero plot, frequencies are in radians/sample"""
    plt.clf()
    fig = plt.figure(num='Biquad Notch Response',figsize=(10,4.8))
    ax=fig.add_subplot(1,10,(1,10),projection='polar')
    ax.set_rlim(0,1.1)
    poles = ax.scatter([pole_frequency,-pole_frequency],[pole_radius,pole_radius],c='red',marker="x", s=75)
    zeros = ax.scatter([zero_frequency,-zero_frequency],[zero_radius,zero_radius],c='red',marker="o", s=75)
    plt.show()

def plot_filter(zero_radius=1.0, zero_frequency=0.25, pole_radius=0.5, pole_frequency=0.25):
    """ Make a frequency response plot, frequencies are in radians/sample"""
    plt.clf()
    fig = plt.figure(num='Biquad Notch Response',figsize=(10,4.8))
    ax=fig.add_subplot(1,10,(1,3),projection='polar')
    ax2=fig.add_subplot(1,10,(5,10))
    ax.set_rlim(0,1.1)
    ax2.set_ylim(0.001,1.1)
    ax2.set_xlabel("Normalized Freq: (xPI radians/sample)")
    zero_theta = np.pi * zero_frequency
    pole_theta = np.pi * pole_frequency
    poles = ax.scatter([pole_theta,-pole_theta],[pole_radius,pole_radius],c='red',marker="x")
    zeros = ax.scatter([zero_theta,-zero_theta],[zero_radius,zero_radius],c='red',marker="o")

    #L REPLACE
    evalfreqs=np.linspace(0,1,10000)
    response=ax2.semilogy(evalfreqs,bqnotchresponse(1,init_radius,init_frequency,evalfreqs))

    
    fig.subplots_adjust(left=0.15, bottom=0.25,wspace=0.25,hspace=0.25)   
    plt.savefig("output.png")
    plt.show()