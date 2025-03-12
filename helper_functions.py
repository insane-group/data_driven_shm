import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error
from scipy import signal
import pywt

def rfecv(X_train,y,X_test):
    from sklearn.feature_selection import RFE
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.tree import DecisionTreeClassifier
    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select = 3 )
    rfe.fit(X_train,y)
    feature_list=[]
    for i,col in zip(range(X_train.shape[1]), X_train.columns):
        if rfe.ranking_[i]<2:
            feature_list.append(col)
    X_train_new = pd.DataFrame()
    X_test_new = pd.DataFrame()
    for i in range(0,len(feature_list)):
            X_train_new[f'feature{i}'] = X_train[feature_list[i]]
            X_test_new[f'feature{i}'] = X_test[feature_list[i]]
    X_test = X_test_new
    X_train = X_train_new
    return X_train,X_test

def pca(X_train,X_test):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50, random_state = 42)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_train = pd.DataFrame(X_train)
    X_test = pca.transform(X_test)
    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    #plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    #plt.ylabel('Percentage of Explained Variance')
    #plt.xlabel('Principal Component')
    #plt.title('Scree Plot')
    #plt.show()
    return X_train,X_test

def res_plot(model_list,min,mid,max,name_list):

    X_axis = np.arange(len(model_list)) 

    plt.bar(X_axis - 0.25, min, 0.2, label = '75 training samples')
    
    #for index, value in enumerate(min):
    #    plt.text(value, index,str(value))
    
    plt.bar(X_axis , mid, 0.2, label = '112 training samples')
    #for index, value in enumerate(mid):
    #    plt.text(value, index,str(value))
    
    plt.bar(X_axis + 0.25 , max, 0.2, label = '225 training samples')
    #for index, value in enumerate(max):
    #    plt.text(value, index,str(value))
    
    plt.xticks(X_axis, name_list)
    plt.xlabel("Models")
    plt.ylabel("Mean absolute Percentage error")
    plt.title(f"MAPE of models with different training sizes ")
    plt.legend() 
    plt.show()

def single_model_result_plot(model,X_train,y,X_test,y_true):
    plt.plot(regression_model_run(model,X_train,y,X_test,y_true)[2],marker = 'o')
    plt.plot(regression_model_run(model,X_train,y,X_test,y_true)[3],linestyle='dashed',marker = 'o')
    plt.xlabel("sample")
    plt.ylabel("y value")
    plt.title(f" Predicted and true value of samples using Linear Regression")
    plt.legend(["y_test", "y_pred"], loc="lower right")
    plt.show()


def parity_plot(y_true,y_pred,model,mode):
    plt.scatter(y_true,y_pred,color='r')
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints)
    plt.xlabel('Test Values')
    plt.ylabel('Predicted Values')
    if model.__name__ =='mlp' : name = 'MLP'
    if model.__name__ =='linear_regression' : name = 'Linear Regression'
    if model.__name__ =='decision_tree_reg' : name = 'Decision Trees'
    plt.title(f'Parity plot of {name}')
    plt.legend(["y_values", "y=x"], loc="lower right")
    if mode=='save':
        plt.savefig(f'{name}_parity_plot.png')
        plt.close('all')
        plt.clf()
    elif mode =='show':
        plt.show()


def regression_model_run(model,X_train,y,X_test,y_true):

    y_pred = model(X_train,y,X_test)
    #print(y_pred)
    mape = 100*mean_absolute_percentage_error(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    return mae,mape,y_true,y_pred

def fourier(sample_sensor):
    fs = 1/1000
    #the sampling frequency is 1/(seconds in a total experiment time)

    fourier = np.fft.fft(sample_sensor)
    #sample sensor is the value of s2 which is the 
    freqs = np.fft.fftfreq(sample_sensor.size,d=fs)
    power_spectrum = np.abs(fourier)
    #
    power_spectrum = np.log(power_spectrum)
    #
    return power_spectrum,freqs


def pwelch(sample_sensor):
    fs = 1000
    (f, S)= signal.welch(sample_sensor, fs, nperseg=1024)
    return S,f
    #plt.semilogy(f, S)
    #plt.xlim([0, 500])
    #plt.xlabel('frequency [Hz]')
    #plt.ylabel('PSD [V**2/Hz]')
    #plt.show()

def psd(sample_sensor):
    fs = 1000
    # f contains the frequency components
    # S is the PSD
    (f, S) = signal.periodogram(sample_sensor, fs, scaling='density')
    return S,f
    #plt.semilogy(f, S)
    #plt.ylim([1e-14, 1e-3])
    #plt.xlim([0,500])
    #plt.xlabel('frequency [Hz]')
    #plt.ylabel('PSD [V**2/Hz]')
    #plt.show()

def spectrogram(sample):
    fs = 1000
    f, t, Sxx = signal.spectrogram(sample, fs)
    #plt.pcolormesh(t, f, Sxx, shading='gouraud')
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    #plt.show()
    return Sxx

def wavelet(sample):
    fs = 1000  # Sampling frequency
    t = np.linspace(0, 1, fs, endpoint=False)  # Time vector
    signal = sample

    # Perform wavelet transform
    wavelet_name = 'db1' # Daubechies wavelet, order 1
    transformed_signal, _ = pywt.dwt(signal, wavelet_name)
    return transformed_signal
    # Plot the original signal
    #plt.subplot(2, 1, 1)
    #plt.plot(signal)
    #plt.title('Original Signal')

    # Plot the transformed signal
    #plt.subplot(2, 1, 2)
    #plt.plot(transformed_signal)
    #plt.title('Transformed Signal')

    #plt.tight_layout()
    #plt.show()


################### den paizei kala thelei ftiaksimo 18/12/2024###################
def signal_data(sample):
    from scipy import signal
    high_peaks, high_peaks_properties = signal.find_peaks(sample,prominence=0.08)
    low_peaks, low_peaks_properties = signal.find_peaks(sample,distance = 500,height=(0.005,0.008))
    dx = low_peaks - high_peaks
    dy = high_peaks_properties['prominences'] - low_peaks_properties['peak_heights']
    signal_props = [high_peaks_properties['prominences'],
                    high_peaks,
                    low_peaks_properties['peak_heights'],
                    low_peaks,
                    dx,
                    dy
                    ]
    return signal_props
###############################################################################################


def signal_props_extract(sample):
    ########## pairnei san input sample apo fourier shma
    ####### dinei output ta signal props tou shmatos
    freq = sample[1]
    amp = sample[0]
    
    #### auta ta bounds allazoun analoga me ta shmeia poy emfanizetai to megisto amp
    ### gia kanoniko shma ta oria einai freq = 0 , freq = 200 kai freq = 400 Khz
    ### gia standardized shma einia freq = 0 , 1.3<=freq<=1.5  kai 2.9<=freq<=3.2 
    for i in range(0,len(freq)):

        if freq[i] >= 1.3 and freq[i] <= 1.5:
            first_bound = i
        if freq[i] >= 2.9 and freq[i] <= 3.2:
            second_bound = i
        if freq[i] ==0:
            zero_bound = i



    first_amp =[]
    for i in range(zero_bound,first_bound):
        first_amp.append(amp[i])

    second_amp =[]
    for i in range(first_bound,second_bound):
        second_amp.append(amp[i])
    
    for i in range(zero_bound,first_bound):
        if amp[i] == max(first_amp):
            first_max_amp = amp[i]
            first_max_freq = freq[i]
            #####
            first_max_i = i
            #####
    

    for i in range(first_bound,second_bound):
        if amp[i] == max(second_amp):
            second_max_amp = amp[i]
            second_max_freq = freq[i]
            #####
            second_max_i = i
            ######

    #####
    #for i in range(zero_bound,first_max_i):
    #    if amp[i] < 0.1 * max(first_amp):
    ##        first_width_first_bound = freq[i]
    #for i in range(first_max_i,first_bound):
    #    if amp[i] < 0.1 * max(first_amp):
    #        first_width_second_bound = freq[i]

    #for i in range(first_bound,second_max_i):
    #    if amp[i] < 0.1 * max(second_amp):
    #        second_width_first_bound = freq[i]
    #for i in range(second_max_i,second_bound):
    #    if amp[i] < 0.1 * max(second_amp):
    #        second_width_second_bound = freq[i]

    #big_width = first_width_second_bound-first_width_first_bound
    #small_width = second_width_second_bound- second_width_first_bound
    #####

    dx = second_max_freq-first_max_freq
    dy = first_max_amp-second_max_amp
    logos_freq = second_max_freq/first_max_freq
    logos_amp = first_max_amp/second_max_amp
    props = first_max_amp,second_max_amp,dx,dy#,logos_freq,logos_amp#,big_width,small_width,first_max_freq,second_max_freq
    
    return props


def fourier_signal_standardization(sample):
    ########## pairnei san input sample apo raw shma
    ####### dinei output to amp kai to freq tou kanonikopoihmenou shmatos
    amp= fourier(sample)[0]
    freq= fourier(sample)[1]

    amp_list =[]
    freq_list =[]
    bound = int(0.5*len(amp))
    #max_amp = max(amp)
    max_amp = -max(amp)
    max_freq = abs(freq[amp.argmax()])

    for i in range(0,bound):
        if i>150 and i<200 :
            amp_list.append(amp[i]/max_amp)
            #amp_list.append(1/(amp[i]/max_amp))
            freq_list.append(freq[i]/max_freq)

    amp = np.array(amp_list)
    freq = np.array(freq_list)
    return amp,freq


def fourier_std_vector(path):
    ########## pairnei san input path
    ####### dinei output to std fourier shma
    from file_opener import X_set
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    vector = np.concatenate(( fourier_vector_maker(s2)[0],fourier_vector_maker(s3)[0],fourier_vector_maker(s4)[0],fourier_vector_maker(s4)[1]),axis=1)
    return vector


def fourier_vector_maker(data):
    ########## pairnei san input data
    ####### dinei output vector kanonikopoihmeno sample me freq
    feature_vector=[]
    freq_vector =[]
    for sample in data:
        feature_vector.append(fourier_signal_standardization(sample)[0])
        freq_vector.append(fourier_signal_standardization(sample)[1])
    return feature_vector,freq_vector


def run_signal_extract(data):
    ########## pairnei san input raw shma
    ####### dinei output ta signal properties tou shmatos
    feature_vector=[]
    for sample in data:
        sample = fourier_signal_standardization(sample)
        feature_vector.append(signal_props_extract(sample))
    return feature_vector

def signal_with_props_vector(path,transformation):
    ########## pairnei san input path
    ####### dinei output ta signal properties tou shmatos me to shma me ton metasxhmatismo
    from file_opener import X_set
    X, s2,s3,s4,freqs = X_set(path,transformation)
    vector = np.concatenate((s2,s3,s4,freqs),axis=1)
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    prop_vector = np.concatenate(( run_signal_extract(s2),run_signal_extract(s3),run_signal_extract(s4)),axis=1)
    vector = np.concatenate((vector,prop_vector),axis=1)
    return vector


def props_vector(path):
    ########## pairnei san input path
    ####### dinei output ta signal properties tou shmatos
    from file_opener import X_set
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    vector = np.concatenate(( run_signal_extract(s2),run_signal_extract(s3),run_signal_extract(s4)),axis=1)
    return vector

def fourier_std_with_props_vector(path):
    ########## pairnei san input path
    ####### dinei output ta signal properties tou shmatos me to shma me to kanonikopoihmeno fourier
    from file_opener import X_set
    vector = fourier_std_vector(path)
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    prop_vector = np.concatenate(( run_signal_extract(s2),run_signal_extract(s3),run_signal_extract(s4)),axis=1)
    vector = np.concatenate((vector,prop_vector),axis=1)
    return vector




def x_y_unwanted_remover(sensor2,sensor3,sensor4,y):
    index_remove_list =[]
    for i in range(0,len(y)):
        if y[i] =='clean' or y[i] =='ola':
            index_remove_list.append(i)
    index_remove_list.reverse()

    for i in index_remove_list:
        del sensor2[i]
        del sensor3[i]
        del sensor4[i]
        y =  y.drop([i])
    X = np.concatenate((sensor2,sensor3,sensor4),axis=1)
    y = np.array(y)
    return X,y


def fourier_std_vector_harmonics(path,min_size):
    ########## pairnei san input path
    ####### dinei output vector kanonikopoihmeno sample me freq
    from file_opener import X_set 
    data= X_set(path,'none')[0]
    feature_vector=[]
    freq_vector =[]
    for sample in data:
        feature_vector.append(fourier_signal_standardization_harmonics(sample)[0])
        freq_vector.append(fourier_signal_standardization_harmonics(sample)[1])
    #### epeidh exw balei thn if sth sunarthsh fourier kathe sample mesa sto feature vector den exei idio megethos
    ### prepei kathe sample na exei idio megethos gia auto stis periptwseis pou exw parapanw times afairw kapoies
    
    min_size_feature_vector =[]
    min_size_freq_vector =[]

    

    for sample in feature_vector:
        sample = np.random.choice(sample, size=min_size, replace=False)
        min_size_feature_vector.append(sample)

    for sample in freq_vector:
        sample = np.random.choice(sample, size=min_size, replace=False)
        min_size_freq_vector.append(sample)
    
    feature_vector = min_size_feature_vector
    freq_vector = min_size_freq_vector

    vector = np.concatenate((feature_vector,freq_vector),axis=1)
    
    return vector


def fourier_signal_standardization_harmonics(sample):
    ########## pairnei san input sample apo raw shma
    ####### dinei output to amp kai to freq tou kanonikopoihmenou shmatos
    amp= fourier(sample)[0]
    freq= fourier(sample)[1]

    amp_list =[]
    freq_list =[]
    bound = int(0.5*len(amp))
    max_amp = -max(amp)
    max_freq = abs(freq[amp.argmax()])

    #
    for i in range(0,bound):
        if freq[i]/max_freq>0.8 and freq[i]/max_freq<1.2 or freq[i]/max_freq>1.8 and freq[i]/max_freq<2.2:
            amp_list.append(amp[i]/max_amp)
            freq_list.append(freq[i]/max_freq)
    
    freq = np.array(freq_list)
    amp = np.array(amp_list)
    from scipy.signal import savgol_filter
    
    amp = savgol_filter(amp,5,3)

    return amp,freq