def create_lowpass_filter(cutoff_freq, fs, numtaps):
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff_freq / nyquist
    taps = scipy.signal.firwin(numtaps, normalized_cutoff)
    return taps

def impulse_response(filter_coeffs):
    return scipy.signal.unit_impulse(len(filter_coeffs)), filter_coeffs

def apply_filter(signal, filter_coeffs):
    return scipy.signal.lfilter(filter_coeffs, 1.0, signal)