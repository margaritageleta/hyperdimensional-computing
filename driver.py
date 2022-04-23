import time
import copy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        # specify type to avoid type casting to int.
        # Numba methods do not work when called with named arguments. 
        # They do work if passed positional arguments.
        return np.float64(2) * np.random.binomial(1.0, 0.5, D).astype(np.float64) - np.float64(1)
    elif kind == 'gaussian':
        return np.random.randn(D).astype(np.float64)
    elif kind == 'uniform':
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
    ('kind', types.unicode_type)
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
    ):
        self.d = d
        self.D = D
        self.fliprate = int(D/2/levels) if fliprate is None else fliprate
        self.kind = kind
    
        if self.kind != 'kernel':  
            self.levels = self._get_levels(minval=minval, maxval=maxval, levels=levels)
            self.level_encoding = self._get_level_hypervects()
    
        if self.kind == 'association':
            self.position_encoding = Dict.empty(key_type=key_type, value_type=value_type)
            for f in range(d):
                self.position_encoding[f] = Hypervector(D=self.D)
        elif self.kind == 'kernel':
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
                cos_value = np.cos(activation + self.kernel_base[j])
                sin_value = np.sin(activation)
                hypervector[j] += (cos_value * sin_value)
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
        return np.where(hypervector >= 0, 1, -1)
    
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
        enc = Encoder(d=617, D=10000, levels=100, fliprate=2000, kind='permutation')
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
        enc = Encoder(d=617, D=10000, levels=100, fliprate=2000, kind='association')
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
        enc = Encoder(d=617, D=10000, levels=0, fliprate=0, kind='kernel')
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





