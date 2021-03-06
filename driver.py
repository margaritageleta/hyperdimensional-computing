import time
import copy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import squareform, pdist

## PARALLEL dependencies
from joblib import Parallel, delayed

## NUMBA dependencies
import numba
from numba import jit
from numba import int32, float32    # import the types
from numba.experimental import jitclass
from numba.core import types
from numba.typed import Dict, List
from joblib import parallel_backend

@jit(nopython=True)
def Hypervector(D=5000, kind='binary'):
    if kind == 'binary':
        # specify type to avoid type casting to int
        return np.float64(2) * np.random.binomial(1.0, 0.5, D).astype(np.float64) - np.float64(1)
    elif kind == 'gaussian':
        return np.random.randn(D).astype(np.float64)
    elif kind == 'uniform':
        # specify type to avoid type casting to int.
        # Numba methods do not work when called with named arguments. 
        # They do work if passed positional arguments.
        return np.random.uniform(0.0, 2*np.pi, D).astype(np.float64)

# https://stackoverflow.com/questions/67289738/is-it-possible-to-create-a-
# numba-dict-whose-key-type-is-unituple-inside-of-a-fun
value_type = types.float64[:]
key_type = types.int64
spec = [
    ('d', types.int64),
    ('D', types.int64),
    ('levels', types.float64[:]),
    ('fliprate', types.int64),
    ('level_encoding', types.DictType(key_type,value_type)),
    ('position_encoding', types.DictType(key_type,value_type)),
    ('kernel_encoding', types.DictType(key_type,value_type)),
    ('kernel_base', types.float64[:]),
    ('kind', types.unicode_type),
    ('kernel_kind', types.unicode_type),
    ('polarize', types.boolean)
]

@jitclass(spec)
class Encoder():
    def __init__(
        self, d, 
        D=5000, 
        levels=10, 
        fliprate=None, 
        kind='permutation',
        minval=-1,
        maxval=1,
        kernel_kind='rbf',
        polarize=True
    ):
        self.d = d
        self.D = D
        self.fliprate = int(D/2/levels) if fliprate is None else fliprate
        self.kind = kind
        self.kernel_kind = 'none' if kind != 'kernel' else kernel_kind
        self.polarize = polarize
    
        if self.kind != 'kernel':  
            self.levels = self._get_levels(minval=minval, maxval=maxval, levels=levels)
            self.level_encoding = self._get_level_hypervects()
    
        if self.kind == 'association':
            self.position_encoding = Dict.empty(key_type=key_type, value_type=value_type)
            for f in range(d):
                self.position_encoding[f] = Hypervector(D=self.D)
        elif self.kind == 'kernel':
            if self.kernel_kind == 'rbf':
                self.kernel_base = Hypervector(D=self.D, kind='uniform')
            self.kernel_encoding = Dict.empty(key_type=key_type, value_type=value_type)
            for j in range(D):
                self.kernel_encoding[j] = Hypervector(D=self.d, kind='gaussian')
        
    def _get_levels(self, minval, maxval, levels):
        levels = np.unique(np.linspace(minval, maxval, levels + 1)).astype(np.float64)
        print(f'Computing levels...')
        print(levels)
        return levels
    
    def _get_level_hypervects(self):
        print('Generating level hypervectors...')
        level_encoding = Dict.empty(key_type=key_type, value_type=value_type)
        indices =  np.arange(self.D)
        
        base = Hypervector(D=self.D)
        for level in range(len(self.levels)):
            if level > 0:
                flip_idx = np.random.choice(
                    indices, 
                    size=self.fliprate, 
                    replace=False
                )
                for i in flip_idx:
                    base[i] = -1 * base[i]
            level_encoding[level] = base.copy()
        # print(level_encoding)
        print('Level hypervectors generated.')
        return level_encoding
    
    def binary_search(self, v):
        if v == self.levels[-1]: return len(self.levels) - 2
        lower_idx, upper_idx = 0, len(self.levels) - 1
        f_idx = 0
        while upper_idx > lower_idx:
            f_idx = int((upper_idx + lower_idx)/2)
            if self.levels[f_idx] <= v and self.levels[f_idx+1] > v:
                return f_idx
            elif self.levels[f_idx] > v: upper_idx = f_idx
            else: lower_idx = f_idx
            f_idx = int((upper_idx + lower_idx)/2)

        return f_idx 
    
    def _transform_x(self, x):
        hypervector = np.zeros(self.D)
        if self.kind == 'kernel':
            for j in range(self.D):
                kernel = self.kernel_encoding[j]
                activation = kernel @ x
                if self.kernel_kind == 'rbf':
                    cos_value = np.cos(activation + self.kernel_base[j])
                    sin_value = np.sin(activation)
                    hypervector[j] += (cos_value * sin_value)
                elif self.kernel_kind == 'tanh':
                    hypervector[j] += np.tanh(activation)
                else: raise Exception('Unknown kernel.')
        else:
            for f in range(len(x)):
                level = self.binary_search(x[f])
                level_hypervector = self.level_encoding[level]
                if self.kind == 'association':
                    id_hypervector = self.position_encoding[f]
                    hypervector += (level_hypervector * id_hypervector)

                elif self.kind == 'permutation':
                    hypervector += np.roll(level_hypervector, f)

                else: raise Exception('Unknown kind.')
        if self.polarize:
            return np.where(hypervector >= 0, 1, -1).astype(np.float64) 
        else: 
            return hypervector 
    
def transform(X, encoder, parallel=True, n_jobs=2):
    n = X.shape[0]
    print('Starting transformation...')
    if parallel:
        with parallel_backend('threading', n_jobs=n_jobs):
            hyperX = Parallel()(delayed(encoder._transform_x)(X[i]) for i in range(n))
    else:
        hyperX = []
        #hyperX.append(List.empty_list(numba.float64[:]))
        for i in range(n):
            hyperX.append(encoder._transform_x(X[i]))
    return np.asarray(hyperX)
    
def transform(X, encoder, parallel=True, n_jobs=2):
    n = X.shape[0]
    print('Starting transformation...')
    if parallel:
        with parallel_backend('threading', n_jobs=n_jobs):
            hyperX = Parallel()(delayed(encoder._transform_x)(X[i]) for i in range(n))
    else:
        hyperX = []
        for i in range(n):
            hyperX.append(encoder._transform_x(X[i]))
    return np.asarray(hyperX)

def similarity(v1, v2):
    return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_class_hypervectors(hyperX_train, Y_train, polarize=True):
    class_hypervectors = dict()
    for i, c in enumerate(Y_train[:]):
        if c not in class_hypervectors.keys():
            class_hypervectors[c] = np.zeros(hyperX_train.shape[1])
        else:
            class_hypervectors[c] += hyperX_train[i, :]
    if polarize:
        for c in class_hypervectors.keys():
            class_hypervectors[c] = np.where(class_hypervectors[c] >= 0, 1, -1)
    return class_hypervectors

def predict(hypervector, class_hypervectors):
    guess = list(class_hypervectors.keys())[0]
    maximum = float('-inf')
    similarities = {}
    for c in class_hypervectors.keys():
        similarities[c] = similarity(hypervector, class_hypervectors[c])
        if similarities[c] > maximum:
            guess, maximum = c, similarities[c]
    return guess

def train(
    train_hyperX, 
    train_labels, 
    class_hypervectors, 
    lr=None, 
    polarize=False, 
    delay=1
):
    updated_class_hypervectors = copy.deepcopy(class_hypervectors)
    if delay > 1:
        delayed_class_hypervectors = copy.deepcopy(updated_class_hypervectors)
    misclassifications = 0
    n = train_hyperX.shape[0]
    for i in range(n):
        guess = predict(train_hyperX[i,:], delayed_class_hypervectors)
        if train_labels[i] != guess:
            misclassifications += 1
            if lr is None:
                updated_class_hypervectors[guess] = updated_class_hypervectors[guess] - train_hyperX[i,:]
                updated_class_hypervectors[train_labels[i]] = updated_class_hypervectors[train_labels[i]] + train_hyperX[i,:]
            else:
                simneg = similarity(updated_class_hypervectors[guess],train_hyperX[i,:])
                simpos =similarity(updated_class_hypervectors[train_labels[i]],train_hyperX[i,:])
                updated_class_hypervectors[guess] = updated_class_hypervectors[guess] - lr*simneg*train_hyperX[i,:]
                updated_class_hypervectors[train_labels[i]] = updated_class_hypervectors[train_labels[i]] + lr*simpos*train_hyperX[i,:]
        if i % delay == 0:
            delayed_class_hypervectors = updated_class_hypervectors
    error = misclassifications / n
    print(f'Error: {error} -- Misclassifications: {misclassifications}')
    if polarize:
        for c in updated_class_hypervectors.keys():
            updated_class_hypervectors[c] = np.where(updated_class_hypervectors[c] >= 0, 1, -1)
    return error, updated_class_hypervectors

def test(test_hyperX, test_labels, class_hypervectors):
    misclassifications = 0
    n = test_hyperX.shape[0]
    for i in range(n):
        guess = predict(test_hyperX[i,:], class_hypervectors)
        if test_labels[i] != guess:
            misclassifications += 1
    accuracy = (1 - (misclassifications / n)) * 100
    print (f'Accuracy: {accuracy} -- Misclassifications: {misclassifications}')
    return accuracy

def retrain_k_times(k, class_hypervectors, train_hyperX, train_labels, test_hyperX, test_labels):
    accuracy = []
    updated_class_hypervectors = copy.deepcopy(class_hypervectors)
    accuracy.append(test(test_hyperX, test_labels, updated_class_hypervectors))
    for j in range(k):
        print(f'Iteration {j} -- Accuracy: {accuracy[-1]}')
        error, updated_class_hypervectors = train(
            train_hyperX=train_hyperX, 
            train_labels=train_labels, 
            class_hypervectors=updated_class_hypervectors
        )
        accuracy.append(test(test_hyperX, test_labels, updated_class_hypervectors))
    print(f'Iteration {j+1} -- Accuracy: {accuracy[-1]}')
    return accuracy, updated_class_hypervectors

def transform_time_series(X, encoder, wsize=4, parallel=True, n_jobs=4):
    n, m = X.shape
    hyperX = []
    assert m % wsize == 0, 'Error!'
    for ni in tqdm.tqdm(range(n)):
        temporal_patterns = []
        print(f'Sample #{ni}...')
        for j in tqdm.tqdm(range(0,m-wsize)):
            Xin = X[ni,j:j+wsize]
            #H = transform(Xin, encoder, parallel=parallel, n_jobs=n_jobs)
            # del Xin
            # print(j, Xin.shape)
            H = encoder._transform_x(Xin)
            temporal_patterns.append(H)
        print('Adding H...')
        M = np.sum(temporal_patterns, axis=0)
        hyperX.append(M)
        print(M.shape)
    return np.asarray(hyperX)

if __name__ == '__main__':

    dataset = 'isolet'
    plot_data = False

    if dataset == 'isolet':
        isolet = pd.read_csv('isolet/isolet.csv')
        n, d = isolet.shape

        # Shuffle dataset.
        isolet = isolet.sample(frac=1)
        Y = isolet.iloc[:,-1] 
        X = isolet.iloc[:,:-1]

        # Train / Test split:
        tr_prop = 0.80
        X_train = X.iloc[:int(tr_prop * n),:]
        Y_train = Y[:int(tr_prop * n)]
        X_test = X.iloc[int(tr_prop * n):,:]
        Y_test = Y[int(tr_prop * n):]

        # Convert labels to int:
        Y_train = np.asarray(Y_train.map(lambda x: int(x[1:-1])))
        Y_test = np.asarray(Y_test.map(lambda x: int(x[1:-1])))

        # Pickle shuffled labels for later usage.
        with open('isolet_tr_y.pkl', 'wb') as f:
            pickle.dump(Y_train, f, pickle.HIGHEST_PROTOCOL)

        with open('isolet_ts_y.pkl', 'wb') as f:
            pickle.dump(Y_test, f, pickle.HIGHEST_PROTOCOL)

        print(f'TRAIN: {X_train.shape} -- {Y_train.shape}')
        print(f'TEST: {X_test.shape} -- {Y_test.shape}')

        if plot_data:
            fig, ax = plt.subplots(1,1,figsize=(15,5))
            ax.imshow(X, cmap='gray', aspect='auto')
            ax.set_title('ISOLET dataset')
            ax.set_xlabel('Features')
            ax.set_ylabel('Samples')
            plt.show()

        # Encode with permutation encoder:
        enc = Encoder(
            d=617, 
            D=10000, 
            levels=100, 
            fliprate=2000, 
            kind='permutation', 
            polarize=True
        )
        ini = time.time()
        hyperX_train = transform(X=np.asarray(X_train)[:,:], encoder=enc, parallel=True, n_jobs=4)
        print(f'Encoded in {time.time()-ini}s')
        # Encoded in 554.1027100086212s

        with open('isolet_tr_permutation_encoding.pkl', 'wb') as f:
            pickle.dump(hyperX_train, f, pickle.HIGHEST_PROTOCOL)
        
        ini = time.time()
        hyperX_test = transform(X=np.asarray(X_test)[:,:], encoder=enc, parallel=True, n_jobs=4)
        print(f'Encoded in {time.time()-ini}s')

        with open('isolet_ts_permutation_encoding.pkl', 'wb') as f:
            pickle.dump(hyperX_test, f, pickle.HIGHEST_PROTOCOL)

        if plot_data:
            aux = []
            for e in enc.level_encoding.values():
                aux.append(e)
            fig, ax = plt.subplots(1,2,figsize=(15,5))
            ax[0].imshow(np.array(aux), cmap='gray', aspect='auto')
            ax[0].set_title('Level hypervectors')
            ax[0].set_xlabel('Level')
            ax[0].set_xlabel('Dimension')
            ax[1].imshow(hyperX_train, cmap='gray', aspect='auto')
            ax[1].set_title('Permutation encoding of the dataset')
            ax[1].set_xlabel('Sample')
            ax[1].set_xlabel('Dimension')
            plt.show()
        
        # Association encoder:
        enc = Encoder(
            d=617, 
            D=10000, 
            levels=100, 
            fliprate=2000, 
            kind='association', 
            polarize=True
        )
        ini = time.time()
        hyperX_train_a = transform(X=np.asarray(X_train)[:,:], encoder=enc, parallel=True, n_jobs=4)
        print(f'Encoded in {time.time()-ini}s')
        # Association: Encoded in 100.22649216651917s
        with open('isolet_tr_association_encoding.pkl', 'wb') as f:
            pickle.dump(hyperX_train_a, f, pickle.HIGHEST_PROTOCOL)

        ini = time.time()
        hyperX_test_a = transform(X=np.asarray(X_test)[:,:], encoder=enc, parallel=True, n_jobs=4)
        print(f'Encoded in {time.time()-ini}s')
        # Encoded in 25.302175760269165s
        with open('isolet_ts_association_encoding.pkl', 'wb') as f:
            pickle.dump(hyperX_test_a, f, pickle.HIGHEST_PROTOCOL)

        # Kernel encoder:
        enc = Encoder(
            d=617, 
            D=10000, 
            levels=0, 
            fliprate=0, 
            kind='kernel',
            kernel_kind='tanh',
            polarize=False
        )
        ini = time.time()
        hyperX_train_k = transform(X=np.asarray(X_train)[:,:], encoder=enc, parallel=True, n_jobs=4)
        print(f'Encoded in {time.time()-ini}s')
        # Encoded in 221.7497627735138s
        with open('isolet_tr_kernel_good_encoding.pkl', 'wb') as f:
            pickle.dump(hyperX_train_k, f, pickle.HIGHEST_PROTOCOL)
        ini = time.time()
        hyperX_test_k = transform(X=np.asarray(X_test)[:,:], encoder=enc, parallel=True, n_jobs=4)
        print(f'Encoded in {time.time()-ini}s')
        # Encoded in 56.467737913131714s
        with open('isolet_ts_kernel_good_encoding.pkl', 'wb') as f:
            pickle.dump(hyperX_test_k, f, pickle.HIGHEST_PROTOCOL)

        # Get class hypervectors:
        class_hypervectors_p = get_class_hypervectors(hyperX_train, Y_train, polarize=False)
        class_hypervectors_a = get_class_hypervectors(hyperX_train_a, Y_train, polarize=False)
        class_hypervectors_k = get_class_hypervectors(hyperX_train_k, Y_train, polarize=False)

        # Hamming distance between hypervectors:
        if plot_data:
            df = pd.DataFrame(class_hypervectors_k).T
            fig, ax = plt.subplots(1,1,figsize=(18,8))
            img = ax.imshow(squareform(pdist(df.loc[[i for i in range(1,27)]],metric='cityblock')), cmap='brg')
            ax.set_xticks(range(26))
            ax.set_yticks(range(26))
            ax.set_yticklabels([chr(65+i) for i in range(26)],ha='center')
            ax.set_xticklabels([chr(65+i) for i in range(26)],ha='center')
            plt.colorbar(img)
            plt.show()
        
        # One-pass train:
        error, class_hypervectors_k_trained = train(
            train_hyperX=hyperX_train_k, 
            train_labels=Y_train, 
            class_hypervectors=class_hypervectors_k,
            lr=None,
            polarize=False,
            delay=1
        )
        # Inference:
        test(
            test_hyperX=hyperX_test_k, 
            test_labels=Y_test, 
            class_hypervectors=class_hypervectors_k_trained
        )

        ## Batch size test:
        for bz in [1,5,10,15,25,50,100,500]:    
            acc, _ = retrain_k_times(
                k=20, 
                class_hypervectors=class_hypervectors_k, 
                train_hyperX=hyperX_train_k, 
                train_labels=Y_train, 
                test_hyperX=hyperX_test_k, 
                test_labels=Y_test,
                polarize=False,
                delay=bz
            )
            with open(f'isolet_acc_retrain_kernel_bz_{bz}_encoding.pkl', 'wb') as f:
                pickle.dump(acc, f, pickle.HIGHEST_PROTOCOL)
        for bz in [1,5,10,15,25,50,100,500]:    
            acc, _ = retrain_k_times(
                k=20, 
                class_hypervectors=class_hypervectors_p, 
                train_hyperX=hyperX_train, 
                train_labels=Y_train, 
                test_hyperX=hyperX_test, 
                test_labels=Y_test,
                polarize=False,
                delay=bz
            )
            with open(f'isolet_acc_retrain_permutation_bz_{bz}_encoding.pkl', 'wb') as f:
                pickle.dump(acc, f, pickle.HIGHEST_PROTOCOL)
        for bz in [1,5,10,15,25,50,100,500]:    
            acc, _ = retrain_k_times(
                k=20, 
                class_hypervectors=class_hypervectors_a, 
                train_hyperX=hyperX_train_a, 
                train_labels=Y_train, 
                test_hyperX=hyperX_test_a, 
                test_labels=Y_test,
                polarize=False,
                delay=bz
            )
            with open(f'isolet_acc_retrain_association_bz_{bz}_encoding.pkl', 'wb') as f:
                pickle.dump(acc, f, pickle.HIGHEST_PROTOCOL)

        ## Dimensionality test:
        dimensionality = [1000, 2500, 5000, 7500, 10000]
        for D in dimensionality:
            enc = Encoder(
                d=617, 
                D=10000, 
                levels=100, 
                fliprate=2000, 
                kind='association',
                kernel_kind='none'
            )
            ini = time.time()
            hyperX_train = transform(X=np.asarray(X_train)[:,:], encoder=enc, parallel=True, n_jobs=4)
            print(f'Encoded in {time.time()-ini}s')
            with open(f'isolet_tr_association_{D}_encoding.pkl', 'wb') as f:
                pickle.dump(hyperX_train, f, pickle.HIGHEST_PROTOCOL)
            ini = time.time()
            hyperX_test = transform(X=np.asarray(X_test)[:,:], encoder=enc, parallel=True, n_jobs=4)
            print(f'Encoded in {time.time()-ini}s')
            with open(f'isolet_ts_association_{D}_encoding.pkl', 'wb') as f:
                pickle.dump(hyperX_test, f, pickle.HIGHEST_PROTOCOL)
            
            class_hypervectors = get_class_hypervectors(hyperX_train_k, Y_train, polarize=False)
            acc, _ = retrain_k_times(
                k=20, 
                class_hypervectors=class_hypervectors, 
                train_hyperX=hyperX_train, 
                train_labels=Y_train, 
                test_hyperX=hyperX_test, 
                test_labels=Y_test,
                polarize=True
            )
            with open(f'isolet_acc_retrain_association_{D}_encoding.pkl', 'wb') as f:
                pickle.dump(acc, f, pickle.HIGHEST_PROTOCOL)

        if plot_data:
            with open('isolet_acc_retrain_permutation_encoding.pkl', "rb") as f:
                perm_acc = pickle.load(f)
            with open('isolet_acc_retrain_association_encoding.pkl', "rb") as f:
                assc_acc = pickle.load(f)   
            with open('isolet_acc_retrain_kernel_tanh_encoding.pkl', "rb") as f:
                kern_acc = pickle.load(f) 

            fig, ax = plt.subplots(1,1,figsize=(12,5))
            ax.plot(perm_acc)
            ax.scatter(range(len(perm_acc)),perm_acc, label='Permutation encoding')
            ax.plot(assc_acc)
            ax.scatter(range(len(assc_acc)),assc_acc, label='Association encoding')
            ax.plot(kern_acc)
            ax.scatter(range(len(kern_acc)),kern_acc, label='Tanh kernel encoding')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Accuracy % on test data')
            ax.set_xticks(range(21))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.legend()
            plt.grid(color='gainsboro', zorder=-10)
            plt.show()

        ## Print results for batch size:
        for bz in [1,5,10,15,25,50,100,500]:
            for model in ['kernel','permutation','association']:
                with open(f'isolet_acc_retrain_{model}_bz_{bz}_encoding.pkl', "rb") as f:
                    acc = pickle.load(f) 
                print(f'{bz}--{model}: {np.median(acc[-1:])}')

    elif dataset == 'EEG':
        # Join CSVs:
        dfs_list = []
        for csv_filename in tqdm.tqdm(glob.glob('./data/SMNI_CMI_TRAIN/*.csv')):
            dfs_list.append(pd.read_csv(csv_filename))
        eeg = pd.concat(dfs_list)
        del(dfs_list)
        eeg = eeg.drop(['Unnamed: 0'], axis=1)

        # Transpose the table to make the data extraction easier:
        transposed_df_list = []
        for group_df in tqdm.tqdm(eeg.groupby(['name', 'trial number', 'matching condition', 'sensor position', 'subject identifier'])):
            _df = pd.DataFrame(group_df[1]['sensor value']).T
            _df.columns = [f'sample_{idx}' for idx in range(256)]
            _df['name'] = group_df[0][0]
            _df['trial number'] = group_df[0][1]
            _df['matching condition'] = group_df[0][2]
            _df['sensor position'] = group_df[0][3]
            _df['subject identifier'] = group_df[0][4]
            
            transposed_df_list.append(_df)
            
        eeg = pd.concat(transposed_df_list)
        eeg = eeg[[*eeg.columns[-5:],*eeg.columns[0:-5]]]
        eeg = eeg.reset_index(drop=True)

        # Save dataset for later use:
        eeg = eeg.sample(frac=1)
        eeg.iloc[:600,:].to_csv('./eeg_dataset.csv')  
        # eeg = pd.read_csv('./eeg_dataset.csv')

        X = np.asarray(eeg[[f'sample_{i}' for i in range(0,256)]])
        Y = np.asarray(eeg['subject identifier'].map(lambda x: 1 if x == 'a' else 0))
        print(X.shape, Y.shape)

        if plot_data:
            fig, ax = plt.subplots(1,1,figsize=(20,5))
            for i in range(X.shape[0]):
                ax.plot(X[i,:], color='green' if Y[i] == 0 else 'red')
                if i == 10: break
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('EEG sample')
            ax.set_ylabel('Value')
            plt.grid()
            patches = []
            patches.append(mpatches.Patch(color='green', label='Control'))
            patches.append(mpatches.Patch(color='red', label='Alcoholic'))
            ax.legend(handles=patches, bbox_to_anchor=(0. ,0.80 ,1.9,0.1),loc=10,ncol=1,)
            plt.show()

            # Histogram of the feature values.
            plt.hist(np.ravel(X), bins=100)
            plt.xlim((-50,50))
            plt.show()
        
        # Train/Test split:
        X_tr = X[:500,:]
        Y_tr = Y[:500]
        X_ts = X[500:600,:]
        Y_ts = Y[500:600]
        print(X_tr.shape, Y_tr.shape, X_ts.shape, Y_ts.shape)

        dimensionality = [1000, 2500, 5000, 7500, 10000]
        window_sizes = [2,4,8,16,64]

        for wsize in window_sizes:
            for D in dimensionality:
                enc = Encoder(
                    d=wsize, 
                    D=D, 
                    levels=100, 
                    fliprate=25, 
                    kind='permutation', 
                    kernel_kind='none',
                    polarize=False,
                    minval=-50,
                    maxval=50
                )
                # Transform data:
                hyperX_tr = transform_time_series(
                    X=X_tr, 
                    encoder=enc, 
                    wsize=wsize, 
                    parallel=False, 
                    n_jobs=8
                )
                hyperX_ts = transform_time_series(
                    X=X_ts, 
                    encoder=enc, 
                    wsize=wsize, 
                    parallel=False, 
                    n_jobs=8
                )
                # Get class hypervectors:
                class_hypervectors = get_class_hypervectors(hyperX_tr, Y_tr, polarize=False)
                # Adaptive Train:
                error, class_hypervectors_trained = train(
                    train_hyperX=hyperX_tr, 
                    train_labels=Y_tr, 
                    class_hypervectors=class_hypervectors,
                    lr=0.1,
                    polarize=False,
                    delay=1
                )
                # Test:
                acc = test(
                    test_hyperX=hyperX_ts, 
                    test_labels=Y_ts, 
                    class_hypervectors=class_hypervectors_trained
                )
                print(f'Using D={D}, wsize={wsize}: Acc {acc}')

    else: raise Exception('Unknown dataset.')






