DCNN
=====================

DCNN is an algorithm designed for improving of Ultra Low input ChIP-Seq data quality (increasing the signal-to-noise ratio).

It is possible to use the algorithm in two states: training the convolutional neural network and its applications to improve the quality of the histone modification data (signal-to-noise ratio indicator).

For training, it is possible to launch both on the pre-prepared data in the bedgraph format (a detailed presentation is indicated in the main.ipynb file) and without preliminary preparation. In this case, the target histone modifications are input to the model, then there is an automatic download from the site and further training of the model. As a result of this step of the algorithm, a file is obtained in .h5 format in the directory 'CODA_PATH/output'.

To use the model, it is also possible to run both on the pre-prepared data in the bedgraph format (a detailed presentation is indicated in the main.ipynb file) and without prior preparation. After launching the corresponding step, the files '.bedgraph' and '.bw' are generated in the directory 'CODA_PATH/output'.

## Prerequisites
1. python 3.6 interpretator
2. bedToBam 2.28.0 (https://bedtools.readthedocs.io/en/latest/content/tools/bedtobam.html), 
3. samtools 1.9 (http://samtools.sourceforge.net/), 
4. deepTools 2.0 (https://github.com/deeptools/deepTools/blob/develop/docs/content/tools/bamCoverage.rst)
5. bedGraphToBigWig (https://github.com/ENCODE-DCC/kentUtils)

Before the program is executed, the corresponding directories CODA_PATH and BEDTOOLS_PATH(for pre-processing and final bigwig creation) in the constants.py must be indicated.


Training convolution neural network (CNN)
-----------------------------------

For learning a neural network, there are two options, use the data from https://artyomovlab.wustl.edu/publications/supp_materials/aging/chipseq/Y20O20/bedgz/ or upload your own data (bedgraph format) and train a neural network on it.


### CNN with ArtyomovLab data preprocessing

Preprocessing requires pre-installed and suggested adding installation directory to the PATH variable:

bedToBam (https://bedtools.readthedocs.io/en/latest/content/tools/bedtobam.html)<br/> 
samtools (http://samtools.sourceforge.net/)<br/>
deepTools (https://github.com/deeptools/deepTools/blob/develop/docs/content/tools/bamCoverage.rst)<br/>
bedGraphToBigWig (https://github.com/ENCODE-DCC/kentUtils)<br/>

1) Configure CODA_PATH and BEDTOOLS_PATH in constants.py 

2) specify:  
the target trainig histine modification HISTONE_TARGET<br/> 
several histone modifications for model prediction quality improvement HELPERS<br/> 
chromosome for training CHROM_TRAIN<br/>
the amount of data for training N_TRAIN_2<br/>
the output name for model MODEL_NAME_2

3) run train_w_data_preprocessing(HISTONE_TARGET, HELPERS, CHROM_TRAIN, N_TRAIN_2, MODEL_NAME_2) from main.ipynb


### CNN with your own data

1) Configure CODA_PATH in constants.py and download your data(files in begraph format) to DATA_PATH

2) specify:  
X_FILES: array with directions to data files(.bedggraph), the first one is target for quallity improvement, other are helpers<br/>
Y_FILE: directions to data file(.bedggraph) of good quality track for CNN training<br/>
the amount of data for training N_TRAIN_1<br/>
the output name for model MODEL_NAME_1<br/>

3) run train_wout_data_preprocessing(X_FILES_IMPL, Y_FILE, N_TRAIN_1, MODEL_NAME_1) from main.ipynb


CNN applying
-----------------------------------

For CNN applying there are two options, use the data from https://artyomovlab.wustl.edu/publications/supp_materials/aging/chipseq/Y20O20/bedgz/ or upload your own data (bedgraph format) and train a neural network on it.


### CNN applying with ArtyomovLab data preprocessing

Preprocessing requires pre-installed and suggested adding installation directory to the PATH variable:  
bedToBam (https://bedtools.readthedocs.io/en/latest/content/tools/bedtobam.html)<br/>
samtools (http://samtools.sourceforge.net/)<br/>
deepTools (https://github.com/deeptools/deepTools/blob/develop/docs/content/tools/bamCoverage.rst)<br/>
bedGraphToBigWig (https://github.com/ENCODE-DCC/kentUtils)

1) Configure CODA_PATH and BEDTOOLS_PATH in constants.py 

2) specify:  
the target applying histine modification HISTONE_IMPL<br/> 
histone modifications(same as in trained model) for model prediction quality improvement HELPERS_IMPL<br/> 
chromosome for implementation CHROM_IMPL<br/>
pre-trained model MODEL_IMPL_NAME_2<br/>
the output name for bigwig OUT_BW_NAME_2<br/> 
basepairs bounds for implementation BOUNDS_IMPL_2 = {'start': int, 'end': int} or BOUNDS_IMPL_2 = None

3) run apply_w_data_preprocessing(HISTONE_IMPL, HELPERS_IMPL, CHROM_IMPL, 
                           MODEL_IMPL_NAME_2, OUT_BW_NAME_2, 
                           bounds = BOUNDS_IMPL_2) 
from main.ipynb


### CNN applying with your own data

For BigWig creation requires pre-installed (and suggested adding installation directory to the PATH variable):
bedGraphToBigWig (https://github.com/ENCODE-DCC/kentUtils)

1) Configure CODA_PATH in constants.py and download your data(files in begraph format) to DATA_PATH

2) specify:  
X_FILES_IMPL: array with directions to data files(.bedggraph), the first one is target for quallity improvement, other are helpers<br/>
Y_FILE_CHECK: directions to data file(.bedggraph) of good quality track for the comparison with the result of CNN<br/> 
pre-trained model MODEL_IMPL_NAME_1<br/>
the output name for bigwig OUT_BW_NAME_1<br/>
basepairs bounds for implementation BOUNDS_IMPL_1 = {'start': int, 'end': int} or BOUNDS_IMPL_1 = None

3) run apply_wout_data_preprocessing(X_FILES_IMPL, Y_FILE_CHECK, 
                              MODEL_IMPL_NAME_1, OUT_BW_NAME_1, 
                              bounds = BOUNDS_IMPL_1)
from main.ipynb


### run test data

data to run tests in main can be downloaded from https://drive.google.com/drive/folders/1NAVdy3tu_liG9cqEkYCcT1kgfXoqeo32?usp=sharing and required to be in DATA_PATH direction

References
-----------------------------------
Denoising genome-wide histone ChIP-seq with convolutional neural networks<br/>
Pang Wei Koh, Emma Pierson and Anshul Kundaje<br/>
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5870713/
