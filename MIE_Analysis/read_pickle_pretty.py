import pickle
import numpy as np
import matplotlib.pyplot as plt
from pypdf import PdfWriter

def load_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data



file_path = './jjbeventsthermal.pkl'
PDF_FILE_SAVE_NAME = 'output2.pdf'



## Load in the data from the pickle file given by it file_path
loaded_data = load_pkl_file(file_path)
print('number of fragments: ', len(loaded_data))

## Data was in 449 fragments, concatenate all the fragments together
data = np.empty((0))
for i in range(len(loaded_data)):
    ## First 8 bytes of each fragments is the header, strip that off (offset = 8) during byte reading and concatenation
    data = np.concat((data, np.frombuffer(loaded_data[i], dtype = np.int16, offset = 8)))

## First 128 bytes of data is more headers, remove it
header = data[:128]
data = data[128:]


## plot entire data stream for quick viewing
plt.plot(data, linestyle = '-', color = "#3F8385")
for i in range(len(data)):
    if (i%(1024*8) == 0):
        plt.axvline(i, linestyle = '--', alpha = 0.35, color = "#820505")
    elif (i%1024 == 0):
        plt.axvline(i, linestyle = '--', alpha = 0.25, color = "#AA5BC7" )
plt.ylim((-2000, 2000))

l = []
#print(np.linspace(512, 1024*27 + 512, 27))
for i in range(28):
    l.append('SURF ' + str(i))
plt.xticks(ticks = np.linspace(1024*4, 1024*8*27 + 1024*4, 28), labels= l, rotation = 45)
plt.ylabel('ADC Count')
plt.show()

## Reshape to be split into SURF#, Channel#
data = np.reshape(data, (28, 8, 1024))



## Plot all the different surfs and channels on seperate graphs
fn = []
for ii in range(28):
    fig, axs = plt.subplots(nrows = 4, ncols = 2, figsize=(10,10))

    plt.suptitle('RF Trigger: SURF ' + str(ii), fontsize = 15, fontweight = 'bold')
    for i in range(8):
        if (i%2==0):
            rms = np.std(data[ii, i][:len(data[ii, i])])
            Vppo2 = (np.max(data[ii, i]) - np.min(data[ii, i]))/2
            axs[i//2, 0].plot(np.linspace(0, 1024, 1024)/3, data[ii, i])
            axs[i//2, 0].set_title('Channel ' + str(i) + ': RMS = ' + str(np.round(rms, 2)))
            axs[i//2, 0].set_xlabel('Time ns')
            axs[i//2, 0].set_ylabel('ADC Count')
            #axs[i//2, 0].set_ylim(ymin=-300, ymax=300)
            axs[i//2, 0].grid()

        else:
            rms = np.std(data[ii, i][:len(data[ii, i])//4])
            Vppo2 = (np.max(data[ii, i]) - np.min(data[ii, i]))/2
            axs[(i-1)//2, 1].plot(np.linspace(0, 1024, 1024)/3, data[ii, i])
            axs[(i-1)//2, 1].set_title('Channel ' + str(i) + ': RMS = ' + str(np.round(rms, 2)))
            axs[(i-1)//2, 1].set_xlabel('Time ns')
            axs[(i-1)//2, 1].set_ylabel('ADC Count')
            #axs[(i-1)//2, 1].set_ylim(ymin=-300, ymax=300)
            axs[(i-1)//2, 1].grid()
        


    plt.tight_layout()
    #plt.ylim((-900, 900))
    #plt.show()
    filename = './SURFdata/SURF_' + str(ii) + 'data.pdf'
    fig.savefig(filename)
    fn.append(filename)


## Merge all the graphs into a single pdf
merger = PdfWriter()

for pdf in fn:
    merger.append(pdf)

merger.write(PDF_FILE_SAVE_NAME)
merger.close()
#plt.show()'''