from subprocess import call, check_output
from constants import *
import pandas as pd
import numpy as np
import random
import progressbar
from sklearn.metrics import r2_score

import importlib

import preprocess_data
importlib.reload(preprocess_data)
from preprocess_data import *

import cnn_model
importlib.reload(cnn_model)
from cnn_model import create_cnn

from keras.models import load_model


'''
subs_name = 'train' or 'impl'
make_train_data, make_impl_data
'''
def create_subs_data(TARGET, HELPERS, chrom, NAME_EXP, subs_name, quality_percent = 0.25, create_bw = False):
    
    #create data
    print([TARGET] + HELPERS)
    for hist in [TARGET] + HELPERS:
        load(hist, chrom, NAME_EXP, subs_name, quality_percent)
        bed_bedgraph(hist, chrom, NAME_EXP, subs_name, subsample = True)
    if call('test -f %s%s_%s.%s.b%s.bedgraph'%(DATA_PATH, NAME_EXP, TARGET, chrom, BATCH), shell = True) == 1:
        bed_bedgraph(TARGET, chrom, NAME_EXP, subs_name, subsample = False)
    if create_bw == True:
        bedgraph_bw(TARGET, chrom, NAME_EXP, subs_name)
    
    #create names
    X_files = []
    for hist in [TARGET]+HELPERS:
        X_files.append(DATA_PATH + NAME_EXP + '_' + hist + '.' + chrom + '.subs_' + subs_name + '.b' + str(BATCH) + '.bedgraph')
    y_file = DATA_PATH + NAME_EXP+'_' + TARGET + '.' + chrom + '.b' + str(BATCH) + '.bedgraph'
    return X_files, y_file
  

'''
data to df
make_df
'''
def create_df(files, window_len, rows = None, n_rows = None):
    
    data_list = []
    N_max_list = []
    
    for f in files: 
        print(f)
        data = pd.read_csv(f, sep='\t', header = None, dtype={1: int, 2: int, 3: int})
        data.columns = ['chr', 'start', 'end', 'cnt']
        data_list.append(data)
        N_max_list.append(max(data.start))
    
    N_max = min(N_max_list)
    
    if rows == None:
        if n_rows != None:
            rows = random.sample(range(0, int(N_max/BATCH)), n_rows)
        else:
            rows = range(int(min(data_list[0].start)/BATCH),int(max(data_list[0].start)/BATCH))      
    
    window_h_len = int((window_len - 1) / 2)
    npdata = []
    
    t = 0
    bar = progressbar.ProgressBar().start(len(rows))
    
    for n in rows:
        row_vect = []
        for d in data_list:
            d1 = d[['start','cnt']][(d.start >= (n - window_h_len) * BATCH) & (d.start <= (n + window_h_len) * BATCH)]
            try:
                d1['m'] = d1.apply(lambda x: (x.start - n * BATCH) / BATCH + window_h_len, axis = 1).astype(int)
                a = np.zeros(window_len)
                a[d1.m] = d1.cnt               
                a = a.reshape(1, window_len)
            except ValueError:              
                a = np.zeros(window_len).reshape(1, window_len)
            row_vect.append(a)
        npdata.append(np.concatenate(row_vect, axis = 0).transpose())
        t += 1
        bar.update(t)  
    
    bar.finish()    
    return np.array(npdata), rows


def process_model_df(X_files, y_file, W, N_TRAIN = 1000):
    df_X, rows_model = create_df(files = X_files, window_len = W, n_rows = N_TRAIN)
    df_y = create_df(files = [y_file], window_len = 1, rows = rows_model)[0]
    return df_X, df_y


def process_impl_df(X_files_impl, y_file_impl, W):
    df_X_impl, rows_impl = create_df(files = X_files_impl, window_len = W)
    if y_file_impl != None:
        df_y_impl = create_df(files = [y_file_impl], window_len = 1, rows = rows_impl)[0]#.ravel()
    else:
        df_y_impl = None
    return df_X_impl, df_y_impl, rows_impl


def SNR(f):
    data = pd.read_csv(f, sep='\t', header = None, dtype={1: int, 2: int, 3: int})
    data.columns = ['chr', 'start', 'end', 'cnt']
    snr = np.quantile(data[data.cnt>=1].cnt, 0.9)/np.quantile(data[data.cnt>=1].cnt, 0.1)
    #print('signal/noise ratio =', snr)
    return snr


def create_sub_bedgraph(f, start, end ):
    check_output("awk '{if ($2>=%s && $2<%s) print $0}' %s > %ssub_%s_%s.bedgraph" %(start, end, f, f[:-8],start, end), shell = True)
    return f[:-8]+'sub_'+str(start)+'_'+str(end)+'.bedgraph'


'''
subtrack creation

input: model_f – str, model path+filename
input: hist_impl_f – str, implemented histone modification path+filename
input: helpers_impl_f – list of str, helpers histone modification paths+filenames
input: chrom_impl – implemented chromosome
input: sub_impl – None or [int, int], start and end of subtrack
input: check_impl_f – str, implementation check path+filename
'''

def model_implementation(model_name, hist_impl_f, helpers_impl_f, bounds_impl = None, check_impl_f = None, bw_name = 'prediction.bw'): 
    print('!!', bw_name)
    if bounds_impl != None:
        hist_impl_f = create_sub_bedgraph(hist_impl_f, bounds_impl['start'], bounds_impl['end'])
        #bw_name = '.sub_'+str(bounds_impl['start'])+'_'+str(bounds_impl['end'])
    else:
        pass
        #bw_name = '.sub_impl'

    df_X_impl, df_y_impl, rows_impl = process_impl_df([hist_impl_f] + helpers_impl_f, check_impl_f, W)
    
    model = load_model(MODELS_PATH + model_name)
    
    y_impl_prediction = model.predict(df_X_impl, steps = 1)
    
    data = pd.read_csv(hist_impl_f, sep='\t', header = None, dtype={1: int, 2: int, 3: int})
    chrom_impl = data[0][0]
    
    make_prediction_and_bw(chrom_impl, y_impl_prediction, df_y_impl, bw_name, rows_impl)
    
    x_impl_cnt = create_df(files = [hist_impl_f], window_len = 1, rows = rows_impl)[0]
    
    bedgraph_base = pd.DataFrame({'chr': chrom_impl, 'start':np.array(rows_impl)*BATCH, 'end':np.array(rows_impl)*BATCH+BATCH})
    
    x_bedgraph = pd.concat([bedgraph_base, pd.Series(x_impl_cnt.ravel())], axis = 1)
    x_bedgraph.columns = ['chr','start','end','cnt']
    print('low quality signal/noise =', np.quantile(x_bedgraph[x_bedgraph.cnt>=1].cnt, 0.9)/np.quantile(x_bedgraph[x_bedgraph.cnt>=1].cnt, 0.1))
    
    y_bedgraph = pd.concat([bedgraph_base, pd.Series(df_y_impl.ravel())], axis = 1)
    y_bedgraph.columns = ['chr','start','end','cnt']
    print('good quality signal/noise =', np.quantile(y_bedgraph[y_bedgraph.cnt>=1].cnt, 0.9)/np.quantile(y_bedgraph[y_bedgraph.cnt>=1].cnt, 0.1))
    
    y_pred_bedgraph = pd.concat([bedgraph_base, pd.Series(y_impl_prediction.ravel())], axis = 1)
    y_pred_bedgraph.columns = ['chr','start','end','cnt']
    try:
        print('prediction signal/noise =', np.quantile(y_pred_bedgraph[y_pred_bedgraph.cnt>=1].cnt, 0.9)/np.quantile(y_pred_bedgraph[y_pred_bedgraph.cnt>=1].cnt, 0.1))
    except:
        print('All predictid signals less then 0.5')
        
        
def make_prediction_and_bw(chrom, y_impl_prediction, df_y_impl, f_prediction_name, rows_impl):
    print(f_prediction_name)
    r2_impl = r2_score(df_y_impl.reshape(1,-1), y_impl_prediction.reshape(1,-1))
    d = {'col_n': rows_impl}
    df = pd.DataFrame(data = d)
    df['chr'] = chrom
    df['start'] = df.apply(lambda x: x.col_n*BATCH, axis=1)
    df['end'] = df.apply(lambda x: (x.col_n+1)*BATCH, axis=1)
    df_res = pd.concat([df, pd.DataFrame(y_impl_prediction.reshape(-1,1)), pd.DataFrame(df_y_impl.reshape(-1,1))], axis=1)
    df_res.columns = ['col_n', 'chr', 'start', 'end', 'cnt', 'cnt_target']
    df_res[['chr', 'start', 'end', 'cnt']].to_csv(OUTPUT_PATH + f_prediction_name+'.bedgraph', sep='\t', na_rep='', float_format=None, columns=None, header=None, index=False)
    print(f_prediction_name)
    try:
        call('bedGraphToBigWig %s%s.bedgraph %s %s%s.bw' %(OUTPUT_PATH, f_prediction_name, BEDTOOLS_PATH, OUTPUT_PATH, f_prediction_name), shell = True)
        print('called')
    except:
        pass
    return r2_impl


#training CNN steps
def train_cnn(preprocessing = False, 
              X_files = None, y_file = None, 
              histone_target = None, helpers = None, chrom_train = None, name_exp = NAME_EXP, quality_percent = 0.2,
              w = W, n_train = None, model_output_name = 'CNN.h5'):   
    if preprocessing == True:
        X_files, y_file = create_subs_data(histone_target, helpers, chrom_train, name_exp, 'train', quality_percent)
        print(X_files[0], SNR(X_files[0]))
        print(y_file, SNR(y_file))
    X_df, y_df = process_model_df(X_files, y_file, w, n_train)
    model = create_cnn(X_df, y_df , model_output_name)

    
#applying CNN steps
def apply_cnn(preprocessing = False,
              bounds_impl = None, 
              X_files_impl = None, y_file_impl = None, 
              histone_impl = None, helpers = None, chrom_impl = None, name_exp = NAME_EXP, quality_percent = 0.2,
              model_name = 'CNN.h5', output_bw_name = 'prediction.bw'):
    if preprocessing == True:
        X_files_impl, y_file_impl = create_subs_data(histone_impl, helpers, chrom_impl, name_exp, 'impl', quality_percent)
        print(X_files_impl[0], SNR(X_files_impl[0]))
        print(y_file_impl, SNR(y_file_impl))
    model_implementation(model_name, X_files_impl[0], X_files_impl[1:],  bounds_impl, y_file_impl, output_bw_name)
    
    
#traing CNN without data preprocessing
def train_wout_data_preprocessing(X_FILES_IMPL, Y_FILE, N_TRAIN_1, MODEL_NAME_1):    
    train_cnn(preprocessing = False, 
              X_files = X_FILES_IMPL, 
              y_file = Y_FILE, 
              n_train = N_TRAIN_1, 
              model_output_name = MODEL_NAME_1)

#traing CNN with data preprocessing
def train_w_data_preprocessing(HISTONE_TARGET, HELPERS, CHROM_TRAIN, N_TRAIN_2, MODEL_NAME_2):   
    train_cnn(preprocessing = True, 
              histone_target = HISTONE_TARGET, 
              helpers = HELPERS, 
              chrom_train = CHROM_TRAIN, 
              n_train = N_TRAIN_2, 
              model_output_name = MODEL_NAME_2)
    
#applying CNN without data preprocessing    
def apply_wout_data_preprocessing(X_FILES_IMPL, Y_FILE_CHECK, MODEL_IMPL_NAME_1, OUT_BW_NAME_1, bounds = None):      
    apply_cnn(preprocessing = False,
              bounds_impl = bounds, 
              X_files_impl = X_FILES_IMPL, y_file_impl = Y_FILE_CHECK,
              model_name = MODEL_IMPL_NAME_1,
              output_bw_name = OUT_BW_NAME_1)

#applying CNN with data preprocessing        
def apply_w_data_preprocessing(HISTONE_IMPL, HELPERS_IMPL, CHROM_IMPL, MODEL_IMPL_NAME_2, OUT_BW_NAME_2, bounds = None):      
    apply_cnn(preprocessing = True,
              bounds_impl = bounds, 
              histone_impl = HISTONE_IMPL, helpers = HELPERS_IMPL, chrom_impl = CHROM_IMPL,
              model_name = MODEL_IMPL_NAME_2,
              output_bw_name = OUT_BW_NAME_2)