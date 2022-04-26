# Hyperdimensional Computing
## HDC Classification tasks

In this project I implemented a HDC encoder (allowing for permutation-, association- and hyperbolic tangent kernel-based encodings) along with training anf inference functions, including one-pass training, adaptive one-pass and retraining.

All the results can be found in the `Report.pdf`. The code itself is located in the `driver.py` script. In the `main` function, you can define on which dataset you want to run the experiments. Two options are available: ISOLET and EEG datasets. Also, in `main` you can define a flag to display the plots from the report.

### Dependencies & Important Info
Because HDC is quite costly (encoding data in the hyperdimensional space) and `python` is popular for being "inefficient", I used two methods to boost the time efficiency. First, I parallelized several loops with `joblib`. Using a machine with 4 CPUs, one thread of each was in charge of transforming instances of data and storing them in a list (the order doesn't matter).
Secondly, I adapted the code so that I could compile it with a JIT compiler. I used `Numba` decorators using the `nopython=True` flag, which translated my functions to optimized machine code at runtime using the LLVM compiler library. I also used some experimental features such as `jitclass`.

### Datasets
#### ISOLET
To evaluate the HDC techniques for classification, I used the ISOLET (Isolated Letter Speech Recognition) dataset, which was generated as follows: 150 subjects spoke the name of each letter of the alphabet twice. Thus, there are 52 training examples from each speaker. The speakers have been grouped into sets of 30 speakers each, 4 groups can serve as training set, the last group as the test set. A total of 3 examples are missing, the authors dropped them due to difficulties in recording. Overall, the dataset results in 7797 samples and 617 features.

#### EEG + Alcoholics
To evaluate the HDC techniques for time-series classification, I used the EEG dataset, which comes from a large study to examine how EEG correlates with the genetic predisposition to alcoholism. It contains measurements from 64 electrodes placed on subject's scalps which were sampled at 256 Hz (3.9-msec epoch) for 1 second. There were two groups of subjects: alcoholic and control. Each subject was exposed to either a single stimulus (S1) or to two stimuli (S1 and S2) which were pictures of objects chosen from the 1980 Snodgrass and Vanderwart picture set. 

