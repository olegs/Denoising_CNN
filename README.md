DCNN
=====================

DCNN is a tool for improving of Ultra Low input ChIP-Seq data quality (increasing the signal-to-noise ratio).

The operation of this algorithm includes two states: the construction of a neural network and its application to improve the quality of data. 

Before the program is executed, the corresponding directories CODA_PATH and BEDTOOLS_PATH(for pre-processing and final bigwig creation) in the constants.py must be indicated.

Training a convolution neural network (CNN)
-----------------------------------

For learning a neural network, there are 2 options, use the data from https://artyomovlab.wustl.edu/publications/supp_materials/aging/chipseq/Y20O20/bedgz/ or upload your own data (bedgraph format) and train a neural network on them.

### training a CNN with Artyomov Lab data preprocessing

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


### training a CNN with your own data

1) Configure CODA_PATH in constants.py

2) specify:
X_FILES_IMPL: array with directions to data files(.bedggraph), the first one is target for quallity improvement, other are helpers
Y_FILE: directions to data file(.bedggraph) of good quality track for CNN training
the amount of data for training N_TRAIN_1
the output name for model MODEL_NAME_1

3) run train_wout_data_preprocessing(X_FILES_IMPL, Y_FILE, N_TRAIN_1, MODEL_NAME_1) from main.ipynb



Download test pre-prepared data:
data for learning and using the neural network can be downloaded: https://drive.google.com/file/d/1Xr1jWlSKiHWnUmHIHc5LrE2kv-H3Lf_q/view?usp=sharing, then they have to be unzipped and placed in the DATA_PATH directory

Preliminary data preparation with https://artyomovlab.wustl.edu:
1) for the preliminary preparation of the presence of bedToBam, samtools, bamCoverage, bedGraphToBigWig
2) run main (preprocessing = true) in main.ipynb
