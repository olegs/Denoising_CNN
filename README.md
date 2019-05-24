# DCNN

Run the program on the pre-prepared data (bedgraph):
1) Configure CODA_PATH and BEDTOOLS_PATH in constants2.py (if preprocessing is required)
2) select target / auxiliary histone modifications in main.ipynb and specify the X_files/y_file files for training, X_files_impl/y_file_impl for implementation of the neural network
3) to build .bw, you need to install bedGraphToBigWig (https://github.com/ENCODE-DCC/kentUtils)
4) run main(preprocessing = False) in main.ipynb

Download test pre-prepared data:
data for learning and using the neural network can be downloaded: https://drive.google.com/file/d/1Xr1jWlSKiHWnUmHIHc5LrE2kv-H3Lf_q/view?usp=sharing, then they have to be unzipped and placed in the DATA_PATH directory

Preliminary data preparation with https://artyomovlab.wustl.edu:
1) for the preliminary preparation of the presence of bedToBam, samtools, bamCoverage, bedGraphToBigWig
2) run main (preprocessing = true) in main.ipynb


Запуска программы на предподготовленных данных (bedgraph):
1) в constants2.py настроить CODA_PATH и BEDTOOLS_PATH(если потребуется препроцессинг) 
2) в main.ipynb выбрать целевую/вспомогательные гистоновые модификации и указать файлы X_files/y_file для обучения, X_files_impl/y_file_impl для прменения нейронной сети
3) для построения .bw необходима установка bedGraphToBigWig(https://github.com/ENCODE-DCC/kentUtils)
4) запустить main(preprocessing = False) в main.ipynb

Загрузка тестовых предподготовленных данных:
данные для обучения и применения нейронной сети можно загрузить: https://drive.google.com/file/d/1Xr1jWlSKiHWnUmHIHc5LrE2kv-H3Lf_q/view?usp=sharing, далее их необходимо разархивировать и поместить в директорию DATA_PATH

Предподготовка данных с https://artyomovlab.wustl.edu:
1) для предподготовки предполанается наличие bedToBam, samtools, bamCoverage, bedGraphToBigWig
2) запустить main(preprocessing = True) в main.ipynb
