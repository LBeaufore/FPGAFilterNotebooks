from IIRSim import *
from FrequencyResponse import *
import warnings

def gen_gauss_noise(gauss_sample_num=10000*8, zero_pad=4, noise_size=205, counter=-1, out_format = "scratch/gauss_input_%d_sigma_hanning_clipped_%d.dat"):
    """ Generate Gaussian noise with a Hanning window and optional zero padding on either side"""
    gauss_samps = np.random.normal(loc=0,scale=noise_size,size=gauss_sample_num)
    window = np.hanning(len(gauss_samps))
    gauss_samps = np.concatenate((np.zeros(8*zero_pad), np.multiply(window, gauss_samps), np.zeros(8*zero_pad))) # The padding at the end is to deal with latency
    gauss_samps = np.maximum(gauss_samps, -1*np.ones(len(gauss_samps))*(2**11))
    gauss_samps = np.minimum(gauss_samps, np.ones(len(gauss_samps))*(2**11)-1)
    gauss_samps = np.array(np.round(gauss_samps),dtype=np.int64)
    gauss_samps_old = np.copy(gauss_samps)
    for i in range(len(gauss_samps)):
        gauss_samps[i] = twos_complement_integer(gauss_samps[i],12)
    with open(out_format%(noise_size, counter),"w") as f:
        for samp in gauss_samps:
            f.write("%d\n"%(samp))
    return gauss_samps_old

def gen_pulse(pulse_sample_num=10000*8, zero_pad=4, impulse_size=205, counter=-1, out_format="scratch/pulse_input_height_%d_clipped.dat"):
    """ Generate an impulse"""
    # pulse_samps = {}
    # pulse_samps_old = {}
    # pulse_sample_num = 10000 * 8
    # noise_size=205
    pulse_samps = np.zeros(pulse_sample_num)#np.random.normal(loc=0,scale=noise_size,size=pulse_sample_num)
    pulse_samps[0] = impulse_size
    pulse_samps = np.concatenate((np.zeros(8*zero_pad), pulse_samps, np.zeros(8*zero_pad))) # The padding at the end is to deal with latency
    pulse_samps = np.maximum(pulse_samps, -1*np.ones(len(pulse_samps))*(2**11))
    pulse_samps = np.minimum(pulse_samps, np.ones(len(pulse_samps))*(2**11)-1)
    pulse_samps = np.array(np.round(pulse_samps),dtype=np.int64)
    pulse_samps_old = np.copy(pulse_samps)
    for i in range(len(pulse_samps)):
        pulse_samps[i] = twos_complement_integer(pulse_samps[i],12)
    with open(out_format%(impulse_size),"w") as f:
        for samp in pulse_samps:
            f.write("%d\n"%(samp))
    return pulse_samps_old

def gen_tone(freq_MHz, n_samples=10000*8, zero_pad=4, amplitude=2**11, out_format="scratch/input_%d_MHz_%d_zpclocks_hanning.dat"):
    """ Generate a tone"""
    FREQ_SAMPLE = 3000
    samps=quantized_sample_generator(freq_MHz/FREQ_SAMPLE, n_samples=n_samples, m=8, phase=0, in_amplitude=(amplitude/(2**11)))
    window = np.hanning(len(samps))
    samps = np.multiply(window, samps)
    samps = np.array(np.round(samps),dtype=np.int64)
    samps = np.concatenate((np.zeros(8*zero_pad), samps, np.zeros(8*zero_pad)))
    samps = np.array(np.round(samps),dtype=np.int64)
    oldsamps = np.array(samps, dtype=np.int64)
    for i in range(len(samps)):
        samps[i] = twos_complement_integer(samps[i],12)
    with open(out_format%(freq_MHz,zero_pad),"w") as f:
        for samp in samps:
            f.write("%d\n"%(samp))
    return oldsamps

def import_data(file_name, floating=False, fixed_point=True):
    """ Import FPGA test data from file"""
    data = []
    with open(file_name, "r") as f:
        for line in f:
            if fixed_point:
                data.append(convert_from_fixed_point(int(line), 12, 0))
            if not floating:
                data[-1] = np.round(data[-1])
            else:
                if "." in line and not floating:
                    warnings.warn("Looking for integer but found '.' in '%s'"%(file_name))
                if floating:
                    data.append(float(line))
                else:
                    data.append(np.round(float(line)))
    if floating:
        data = np.array(data)
    else:
        data = np.array(data, dtype=np.int64)
    return data
 
            