DCNN
=====================

DCNN is an algorithm designed for improving of Ultra Low input ChIP-Seq data quality (increasing the signal-to-noise ratio).

The operation of this algorithm includes two states: the construction of a neural network and its application to improve the quality of data. 


## Prerequisites
1. python3.6 interpretator
2. bedToBam (https://bedtools.readthedocs.io/en/latest/content/tools/bedtobam.html), 
3. samtools (http://samtools.sourceforge.net/), 
4. deepTools (https://github.com/deeptools/deepTools/blob/develop/docs/content/tools/bamCoverage.rst)
5. bedGraphToBigWig (https://github.com/ENCODE-DCC/kentUtils)

Before the program is executed, the corresponding directories CODA_PATH and BEDTOOLS_PATH(for pre-processing and final bigwig creation) in the constants.py must be indicated.


Training convolution neural network (CNN)
-----------------------------------

For learning a neural network, there are two options, use the data from https://artyomovlab.wustl.edu/publications/supp_materials/aging/chipseq/Y20O20/bedgz/ or upload your own data (bedgraph format) and train a neural network on it.


### CNN with ArtyomovLab data preprocessing

Preprocessing requires pre-installed and suggested adding installation directory to the PATH variable:

bedToBam (https://bedtools.readthedocs.io/en/latest/content/tools/bedtobam.html), 

samtools (http://samtools.sourceforge.net/), 

deepTools (https://github.com/deeptools/deepTools/blob/develop/docs/content/tools/bamCoverage.rst)

bedGraphToBigWig (https://github.com/ENCODE-DCC/kentUtils)

1) Configure CODA_PATH and BEDTOOLS_PATH in constants.py 

2) specify: 

the target trainig histine modification HISTONE_TARGET

several histone modifications for model prediction quality improvement HELPERS

chromosome for training CHROM_TRAIN

the amount of data for training N_TRAIN_2

the output name for model MODEL_NAME_2

3) run train_w_data_preprocessing(HISTONE_TARGET, HELPERS, CHROM_TRAIN, N_TRAIN_2, MODEL_NAME_2) from main.ipynb


### CNN with your own data

1) Configure CODA_PATH in constants.py and download your data(files in begraph format) to DATA_PATH

2) specify:

X_FILES_IMPL: array with directions to data files(.bedggraph), the first one is target for quallity improvement, other are helpers

Y_FILE: directions to data file(.bedggraph) of good quality track for CNN training

the amount of data for training N_TRAIN_1

the output name for model MODEL_NAME_1

3) run train_wout_data_preprocessing(X_FILES_IMPL, Y_FILE, N_TRAIN_1, MODEL_NAME_1) from main.ipynb


CNN applying
-----------------------------------

For CNN applying there are two options, use the data from https://artyomovlab.wustl.edu/publications/supp_materials/aging/chipseq/Y20O20/bedgz/ or upload your own data (bedgraph format) and train a neural network on it.


### CNN applying with ArtyomovLab data preprocessing

Preprocessing requires pre-installed and suggested adding installation directory to the PATH variable:
bedToBam (https://bedtools.readthedocs.io/en/latest/content/tools/bedtobam.html), 
samtools (http://samtools.sourceforge.net/), 
deepTools (https://github.com/deeptools/deepTools/blob/develop/docs/content/tools/bamCoverage.rst)
bedGraphToBigWig (https://github.com/ENCODE-DCC/kentUtils)

1) Configure CODA_PATH and BEDTOOLS_PATH in constants.py 

2) specify: 
the target applying histine modification HISTONE_IMPL

histone modifications(same as in trained model) for model prediction quality improvement HELPERS_IMPL 

chromosome for implementation CHROM_IMPL

pre-trained model MODEL_IMPL_NAME_2

the output name for bigwig OUT_BW_NAME_2

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
X_FILES_IMPL: array with directions to data files(.bedggraph), the first one is target for quallity improvement, other are helpers

Y_FILE_CHECK: directions to data file(.bedggraph) of good quality track for the comparison with the result of CNN

pre-trained model MODEL_IMPL_NAME_1

the output name for bigwig OUT_BW_NAME_1

basepairs bounds for implementation BOUNDS_IMPL_1 = {'start': int, 'end': int} or BOUNDS_IMPL_1 = None

3) run apply_wout_data_preprocessing(X_FILES_IMPL, Y_FILE_CHECK, 
                              MODEL_IMPL_NAME_1, OUT_BW_NAME_1, 
                              bounds = BOUNDS_IMPL_1)
from main.ipynb


### run test data

data to run tests in main can be downloaded from https://drive.google.com/drive/folders/1NAVdy3tu_liG9cqEkYCcT1kgfXoqeo32?usp=sharing and required to be in DATA_PATH direction

References
-----------------------------------
Denoising genome-wide histone ChIP-seq with convolutional neural networks

Pang Wei Koh, Emma Pierson and Anshul Kundaje

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5870713/
