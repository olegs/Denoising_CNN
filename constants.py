from subprocess import call

CODA_PATH = '/Users/dashabalashova/0_python_projects/DCNN_2/'
BEDTOOLS_PATH = '/Users/dashabalashova/ngs_2/bedtools2/genomes/human.hg19.genome'

DATA_PATH = CODA_PATH + 'data/'
OUTPUT_PATH = CODA_PATH + 'output/'
MODELS_PATH = CODA_PATH + 'models/'
DATA_LOADING_PATH = DATA_PATH + 'loaded_data/'

call('mkdir -p %s'%(DATA_PATH), shell = True)
call('mkdir -p %s'%(MODELS_PATH), shell = True)
call('mkdir -p %s'%(OUTPUT_PATH), shell = True)
call('mkdir -p %s'%(DATA_LOADING_PATH), shell = True)

NAME_EXP = 'OD8'
BATCH = 25
W = 1001
