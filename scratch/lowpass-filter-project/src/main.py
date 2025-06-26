import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz

def create_lowpass_filter(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def impulse_response(b, a, num_samples=100):
    impulse = np.zeros(num_samples)
    impulse[0] = 1.0
    response = lfilter(b, a, impulse)
    return response

def main():
    cutoff = 1.5e9  # 1.5 GHz
    fs = 10e9      # Sampling frequency 10 GHz
    order = 5

    b, a = create_lowpass_filter(cutoff, fs, order)
    response = impulse_response(b, a)

    # Plot the impulse response
    plt.figure(figsize=(10, 6))
    plt.plot(response)
    plt.title('Impulse Response of Lowpass Filter')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()