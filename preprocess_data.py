from subprocess import call, check_output
from constants2 import *
import pandas as pd
import numpy as np
import random
import progressbar
from sklearn.metrics import r2_score


'''
Load data from artyomovlab.wustl.edu
'''
def load(hist, chrom, NAME_EXP, subs_name, quality_percent = 0.25):
    if call('test -f %sloaded_data/%s_%s.bed'%(DATA_PATH, NAME_EXP, hist), shell = True) == 1:
        print('Loading '+NAME_EXP+'_'+hist+'.bed. ...')
        call('wget https://artyomovlab.wustl.edu/publications/supp_materials/aging/chipseq/Y20O20/bedgz/H3K%s/%s_%s_hg19.bed.gz -O %s%s_%s.bed.gz' %(hist[1:], NAME_EXP, hist, DATA_LOADING_PATH, NAME_EXP, hist), shell = True)
        call('gunzip %sloaded_data/%s_%s.bed.gz'%(DATA_PATH, NAME_EXP, hist), shell = True)

    if call('test -f %s%s_%s.%s.bed'%(DATA_PATH, NAME_EXP, hist, chrom), shell = True) == 1:
        print('Filtering %s_%s.bed: %s'%(NAME_EXP, hist, chrom))
        call('grep -w "%s" %sloaded_data/%s_%s.bed > %s%s_%s.%s.bed'%(chrom, DATA_PATH, NAME_EXP, hist, DATA_PATH, NAME_EXP, hist, chrom), shell = True)
    wc_l = int((check_output('wc -l %s%s_%s.%s.bed'%(DATA_PATH, NAME_EXP, hist, chrom), shell = True)).split()[0])
    n_subsample = int(wc_l * quality_percent)
    call('shuf -n %s %s%s_%s.%s.bed > %s%s_%s.%s.subs_%s.bed'%(n_subsample, DATA_PATH, NAME_EXP, hist, chrom, DATA_PATH, NAME_EXP, hist, chrom, subs_name), shell = True)


'''
bed -> bam -> bedgraph
'''
def bed_bedgraph(hist, chrom, NAME_EXP, subs_name, subsample = False):
    if subsample == True:
        f_name = DATA_PATH + NAME_EXP+'_'+hist+'.'+chrom+'.subs_'+subs_name
        f_name_py = DATA_PATH + NAME_EXP+'_'+hist+'.'+chrom+'.subs_'+subs_name
    else:
        f_name = DATA_PATH + NAME_EXP+'_'+hist+'.'+chrom
        f_name_py = DATA_PATH + NAME_EXP+'_'+hist+'.'+chrom
    #print(f_name, f_name_py)
    call('bedToBam -i %s.bed -g %s > %s.bam' %(f_name, BEDTOOLS_PATH, f_name), shell = True)   
    call('samtools sort %s.bam > %s.sorted.bam' %(f_name, f_name), shell = True)   
    call('samtools index %s.sorted.bam' %f_name, shell = True)   
    print('Creating bedgraph: %s.b%s.bedgraph' %(f_name, BATCH))
    call('bamCoverage -b %s.sorted.bam -bs %s -r %s -o %s.b%s.bedgraph --outFileFormat bedgraph' %(f_name, BATCH, chrom, f_name, BATCH), shell = True) 
    call('rm %s.bam' %f_name, shell = True) 
    call('rm %s.sorted.bam' %f_name, shell = True)
    call('rm %s.sorted.bam.bai' %f_name, shell = True)
    f_name_bedgraph = f_name_py + '.b'+str(BATCH)+'.bedgraph'


'''
bedgraph -> bigwig
'''
def bedgraph_bw(hist, chrom, NAME_EXP, subs_name):
    call('bedGraphToBigWig %s%s_%s.%s.subs_%s.b%s.bedgraph %s %s%s_%s.%s.subs_%s.b%s.bw' %(DATA_PATH, NAME_EXP, hist, chrom, subs_name, BATCH, BEDTOOLS_PATH, DATA_PATH, NAME_EXP, hist, chrom, subs_name, BATCH), shell = True) 
    call('bedGraphToBigWig %s%s_%s.%s.b%s.bedgraph %s %s%s_%s.%s.b%s.bw' %(DATA_PATH, NAME_EXP, hist, chrom, BATCH, BEDTOOLS_PATH, DATA_PATH, NAME_EXP, hist, chrom, BATCH), shell = True)
    