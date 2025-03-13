
'''
CONTENTS

1)X AND Y SET CREATORS

2)FOURIER SIGNAL NORMALIZATION, SIGNAL PROPERTIES AND HARMONICS EXTRACTION

3)DATA TRANSFORMATIONS

4)FEATURE ENGINEERING TECHNIQUES FOR SIZE REDUCTION

5)PLOTS

'''



########################################################################

########################################################################

########################################################################

########################################################################

########################################################################

'''
1) X AND Y SET CREATORS

---> x set creator (X_set)

to x set pairnei san input path kai to eidos tou transformation
bgazei 5 outputs
to prwto output einai to concatenated amplitude ten shmatwn kai twv triwn sensors
to deutero trito kai tetarto output einai to amplitude tou shmatos tou kathe sensor
to pempto output einai h suxnothta 

---> y set creator (classification and regression) (y_set)

to y_set pairnei san input to path kai to mode(classification/regression)
bgazei san output to eidos tou defect gia kathe sample sto classification
kai to damage percentage tou defect sto regression

'''

def X_set(path,transformation):

    import os
    import glob
    import numpy as np
    import pandas as pd


    sensor_data_list = []
    name_list = []

    # gia kathe filename sto path pou tou exw dwsei afairei to .csv wste meta na mporei na diabasei ton arithmo
    for filename in sorted(glob.glob(os.path.join(path , "data*"))):
        filename = filename.removesuffix('.csv')
        name_list.append(filename)

    #apo kathe filename krataei mono ton arithmo sto telos kai me auton ton arithmo ftiaxeni th nea sthlh index number
    sensor_data = pd.DataFrame({'name':name_list})
    sensor_data['sensor_index_number'] = [int(i.split('_')[-1]) for i in sensor_data['name']]

    #kanw sort th lista basei tou index number
    sensor_data = sensor_data.sort_values(by=['sensor_index_number'])

    suffix='.csv'
    new_names=[]

    #se kathe filename sth lista pou exei ginei sort prosthetei to .csv wste na mporei na to diabasei
    for filename in sensor_data['name']:
        filename = filename+suffix
        new_names.append(filename)

    #anoigei ta arxeia apo kathe path kai ftiaxnei th lista me tis metrhseis

    for filename in new_names:
        df = pd.read_csv(filename,sep=' |,', engine='python').dropna()
        sensor_data_list.append(df)

    freq_list = []
    power_spectrum_list = []
    sensor_names = ['s2','s3','s4']
    for sensor in sensor_names:
        #gia kathe sample sensora dld gia kathe xronoseira (pou prokuptei apo to shma pou lambanei o sensoras efarmozo transformations
        for i in range(0,len(sensor_data_list)):
            sample_sensor =sensor_data_list[i][sensor]
            if transformation == 'fourier':
                power_spectrum = fourier(sample_sensor)[0]
            elif transformation == 'psd':
                power_spectrum = psd(sample_sensor)[0]
            elif transformation == 'pwelch':
                power_spectrum = pwelch(sample_sensor)[0]
            elif transformation == 'wavelet':
                power_spectrum = wavelet(sample_sensor)
            elif transformation == 'none':
                power_spectrum = sample_sensor
            elif transformation == 'spectrogram':
                power_spectrum = spectrogram(sample_sensor)
            power_spectrum_list.append(power_spectrum)  

    sensor2_vector = []
    sensor3_vector = []
    sensor4_vector = []

    bound_1 = int(len(power_spectrum_list)/3)
    bound_2 = int(2*len(power_spectrum_list)/3)
    bound_3 = int(len(power_spectrum_list))

    if transformation == 'fourier':
        for i in range(0,bound_1):
            freq_list.append(fourier(sample_sensor)[1])

    for i in range(0,bound_1):
        sensor2_vector.append(power_spectrum_list[i])
        
    for i in range(bound_1,bound_2):
        sensor3_vector.append(power_spectrum_list[i])
        
    for i in range(bound_2,bound_3):
        sensor4_vector.append(power_spectrum_list[i])
        
    X = np.concatenate((sensor2_vector,sensor3_vector,sensor4_vector),axis=1)
    return X,sensor2_vector,sensor3_vector,sensor4_vector,freq_list


def y_set(path,mode):
    
    import numpy as np
    import pandas as pd
    import os
    import glob

    #### paizei mono gia to balanced data###
    dmg_list = []
    name_list = []
    case_list = []
    defect_list =[]
    # gia kathe file name sto path pou exw dwsei afairei to .csv kai afairei nan values kai kanei mia lista mono me to damage percentage
    for filename in glob.glob(os.path.join(path , "meta*")):
        df = pd.read_csv(filename,sep=' |,', engine='python')
        dmg_perc = df['Damage_percentage']
        case = df['caseStudey'][0]
        dmg_perc = dmg_perc[0]
        dmg_list.append(dmg_perc)
        filename = filename.removesuffix('.csv')
        
        df_defect = df['DamageLayer1'][0] + df['DamageLayer3'][0] + df['DamageLayer5'][0]
        dm_defect = df['DamageLayer1'][1] + df['DamageLayer3'][1] + df['DamageLayer5'][1]
        dd_defect = df['DamageLayer2'][0] + df['DamageLayer4'][0]
        
        if df_defect ==0 and dm_defect ==0 and dd_defect ==0:
            defect_list.append('clean')
        elif df_defect !=0 and dm_defect !=0 and dd_defect !=0:
            defect_list.append('ola')
        elif df_defect !=0 and dm_defect ==0 and dd_defect ==0:
            defect_list.append('df')
        elif df_defect ==0 and dm_defect !=0 and dd_defect ==0:
            defect_list.append('dm')
        elif df_defect ==0 and dm_defect ==0 and dd_defect !=0:
            defect_list.append('dd')
        else:
            defect_list.append('ola')
        
        name_list.append(filename)
        case_list.append(case)

    # ftiaxnei ena dataframe me to damage percentage kai prosthetei to index number kai kanei sort basei autou 
    dmg_data = pd.DataFrame({'dmg':dmg_list,'damage_file_name':name_list,'caseStudey':case_list,'defect':defect_list})
    dmg_data['dmg_index_number'] = [int(i.split('_')[-1]) for i in dmg_data['damage_file_name']]
    dmg_data = dmg_data.sort_values(by=['dmg_index_number'])

    if mode == 'classification':
        return dmg_data['defect'],
    if mode =='regression':
        return dmg_data['dmg']
    
########################################################################

########################################################################

########################################################################

########################################################################

########################################################################


'''

2) FOURIER SIGNAL NORMALIZATION, SIGNAL PROPERTIES AND HARMONICS EXTRACTION

- - - A) FOURIER SIGNAL NORMALIZATION

- - - B) SIGNAL PROPERTIES EXTRACTION WITH NORMALIZATION

- - - C) HARMONICS WITH NORMALIZATION 

'''



############################################################

'''

A) FOURIER SIGNAL NORMALIZATION

ta tria parakatw functions leitourgoun mazi
gia na parw to kanonikopoihmeno shma xrhsimopoiw to fourier std vector
to opoio pairnei san input to path kai to output einai to kanonikopoihmeno shma

---> fourier signal standardization (fourier_signal_standardization)
To input einai ena sample shmatos kai ypologizei to fft kai kanonikopoei ws pros th megisth syxnothta
dhladh th syxnothta diegershs. To amplitude einai kanonikopoihmeno ws pros to amplitude sth megisth syxnothta kai 
h suxnothta einai kanonikopoihmenh ws pros th syxnothta diegershs

---> fourier vector maker (fourier_vector_maker)
Pairnei san input data (mia lista h array apo shmata) kai efarmozei th sunarthsh fourier_signal_standardization kai dinei to kanonikopoihmeno shma
to input einai ta data kai ta output einai mia lista me to kanonikopoihmeno amplitude kai mia lista me thn kanonikopoihmenh syxnothta

---> fourier std vector (fourier_std_vector)
To input einai to path kai efarmozei thn sunarthsh fourier_vector_maker gia olous tous sensors kathe shmatos sto path.
To input einai to path kai to output einai ta concatenated normalized shmata olwn twn samples sto path

'''



def fourier_signal_standardization(sample):
    import numpy as np
    

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
        amp_list.append(amp[i]/max_amp)
        #amp_list.append(1/(amp[i]/max_amp))
        freq_list.append(freq[i]/max_freq)

    amp = np.array(amp_list)
    freq = np.array(freq_list)
    return amp,freq


def fourier_vector_maker(data):
    ########## pairnei san input data
    ####### dinei output vector kanonikopoihmeno sample me freq
    feature_vector=[]
    freq_vector =[]
    for sample in data:
        feature_vector.append(fourier_signal_standardization(sample)[0])
        freq_vector.append(fourier_signal_standardization(sample)[1])
    return feature_vector,freq_vector


def fourier_std_vector(path):

    import numpy as np
    
    ########## pairnei san input path
    ####### dinei output to std fourier shma
    from file_opener import X_set
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    vector = np.concatenate(( fourier_vector_maker(s2)[0],fourier_vector_maker(s3)[0],fourier_vector_maker(s4)[0],fourier_vector_maker(s4)[1]),axis=1)
    return vector



############################################################

############################################################

'''

B) SIGNAL PROPERTIES EXTRACTION WITH NORMALIZATION

ta tria parakatw functions leitourgoun mazi
gia na parw to kanonikopoihmeno shma me ta props xrhsimopoiw to fourier_std_with_props_vector
to opoio pairnei san input to path kai to output einai to kanonikopoihmeno shma me ta props

gia na parw to raw shma me ta props xrhsimopoiw to signal_with_props_vector
to opoio pairnei san input to path kai to output einai to raw shma me ta props

gia na parw ta props xrhsimopoiw to props_vector
to opoio pairnei san input to path kai to output einai ta props


---> signal properties extraction (signal_props_extract)
To input einai to sample tou kanonikopoihmenou shmatos kai to output einai kapoia properties tou shmatos.
T
---> signal properties extraction run (run_signal_extract)

---> raw signal with properties (signal_with_props_vector)

---> properties vector (props_vector)

---> normalized fourier signal with properties (fourier_std_with_props_vector)

'''

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

def run_signal_extract(data):
    ########## pairnei san input raw shma
    ####### dinei output ta signal properties tou shmatos
    feature_vector=[]
    for sample in data:
        sample = fourier_signal_standardization(sample)
        feature_vector.append(signal_props_extract(sample))
    return feature_vector

def signal_with_props_vector(path,transformation):
    
    import numpy as np
    
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

    import numpy as np
    
    
    ########## pairnei san input path
    ####### dinei output ta signal properties tou shmatos
    from file_opener import X_set
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    vector = np.concatenate(( run_signal_extract(s2),run_signal_extract(s3),run_signal_extract(s4)),axis=1)
    return vector

def fourier_std_with_props_vector(path):

    import numpy as np
    
    ########## pairnei san input path
    ####### dinei output ta signal properties tou shmatos me to shma me to kanonikopoihmeno fourier
    from file_opener import X_set
    vector = fourier_std_vector(path)
    X, s2,s3,s4,none_freqs = X_set(path,'none')
    prop_vector = np.concatenate(( run_signal_extract(s2),run_signal_extract(s3),run_signal_extract(s4)),axis=1)
    vector = np.concatenate((vector,prop_vector),axis=1)
    return vector
############################################################

############################################################

'''

C) HARMONICS WITH NORMALIZATION 


---> fourier signal normalization harmonics (fourier_signal_standardization_harmonics)

---> fourier normalized signal with harmonics(fourier_std_vector_harmonics)

'''



def fourier_signal_standardization_harmonics(sample):

    import numpy as np

    
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



def fourier_std_vector_harmonics(path,min_size):

    import numpy as np
    

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



############################################################


########################################################################

########################################################################

########################################################################

########################################################################

########################################################################

'''
3) DATA TRANSFORMATIONS

---> fast fourier transform (fourier)
pairnei san input ena shma kai ypologizei to fft tou shmatos
dinei san output to amplitude tou shmatos kai th suxnothta

---> pwelch (pwelch)
pairnei san input ena shma kai ypologizei to pwelch tou shmatos
dinei san output to amplitude tou shmatos kai th suxnothta

---> psd (psd)
pairnei san input ena shma kai ypologizei to psd tou shmatos
dinei san output to amplitude tou shmatos kai th suxnothta

---> spectrogram (spectrogram)
pairnei san input ena shma kai ypologizei to spectrogram tou shmatos
dinei san output to spectrogram tou shmatos

---> wavelet (wavelet)
pairnei san input ena shma kai ypologizei to wavelet tou shmatos
dinei san output to wavelet

'''

def fourier(sample_sensor):

    import numpy as np
    
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

    from scipy import signal


    fs = 1000
    (f, S)= signal.welch(sample_sensor, fs, nperseg=1024)
    return S,f
    #plt.semilogy(f, S)
    #plt.xlim([0, 500])
    #plt.xlabel('frequency [Hz]')
    #plt.ylabel('PSD [V**2/Hz]')
    #plt.show()

def psd(sample_sensor):

    from scipy import signal
    

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

    from scipy import signal


    fs = 1000
    f, t, Sxx = signal.spectrogram(sample, fs)
    #plt.pcolormesh(t, f, Sxx, shading='gouraud')
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    #plt.show()
    return Sxx

def wavelet(sample):

    import pywt
    import numpy as np

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

########################################################################

########################################################################

########################################################################

########################################################################

########################################################################


'''
4) FEATURE ENGINEERING TECHNIQUES FOR SIZE REDUCTION 

---> random forest feature elimination with cross validation (rfecv)
pairnei san input to X_train to y_train kai to X_test kai kanei rfecv gia na brei ta kalutera features
to output einai to X_train kai to X_test me ta kalutera features 

---> prinicipal component analysis (pca)
pairnei san input to X_train kai to X_test kai kanei pca gia na krathsei tous grammikous sunduasmous me to megalutero variance
to output einai to X_train me to X_test me ta principal components me to megalutero variance

---> kernel principal component analysis (kpca)
pairnei san input to X_train kai to X_test kai prwta xrhsimopoiei enan kernel gia na kanei map ta features se ena allo feature space kai meta
kanei pca gia na krathsei ta features me to megalutero variance
to output einai to X_train me to X_test me ta features me tous sunduasmous me to megalutero variance

'''



def rfecv(X_train,y,X_test):

    
    import pandas as pd

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

    import numpy as np
    import pandas as pd


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

########################################################################

########################################################################

########################################################################

########################################################################

########################################################################

'''
5) PLOTS

---> bar plots specific for regression, pairnei data sizes (bar_res_plot)
einai specific gia ta regression runs
pairnei san input ta modela pou etreksa kai tis times twn mape twn diaforwn megethwn tou dataset pou etreksan (min,mid,max) kai ta onomata twn montelwn pou etreksan
kai dinei san output ta bar plots twn mape gia kathe montelo

---> parity plots it can either save or show the plot (parity_plot)
pairnei san input to y_test to y_pred to montelo kai to mode dhladh an thelw na kanw save h aplws na dw to plot
bgazei to parity plot tou y_test me to y_pred kai eite to kanei save eite to deixnei

'''

def bar_res_plot(model_list,min,mid,max,name_list):
    
    
    import numpy as np
    import matplotlib.pyplot as plt

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



def parity_plot(y_true,y_pred,model,mode):

    import matplotlib.pyplot as plt

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


########################################################################

########################################################################

########################################################################

########################################################################

########################################################################






def single_model_result_plot(model,X_train,y,X_test,y_true):

    import matplotlib.pyplot as plt
    plt.plot(regression_model_run(model,X_train,y,X_test,y_true)[2],marker = 'o')
    plt.plot(regression_model_run(model,X_train,y,X_test,y_true)[3],linestyle='dashed',marker = 'o')
    plt.xlabel("sample")
    plt.ylabel("y value")
    plt.title(f" Predicted and true value of samples using Linear Regression")
    plt.legend(["y_test", "y_pred"], loc="lower right")
    plt.show()


def regression_model_run(model,X_train,y,X_test,y_true):

    from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error

    y_pred = model(X_train,y,X_test)
    #print(y_pred)
    mape = 100*mean_absolute_percentage_error(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    return mae,mape,y_true,y_pred


def x_y_unwanted_remover(sensor2,sensor3,sensor4,y):

    import numpy as np
    
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