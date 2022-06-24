import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout

from spektral.layers import GCNConv, GlobalSumPool, GCSConv, GlobalAvgPool, ECCConv, GraphMasking, GlobalMaxPool
from spektral.data import BatchLoader, Graph, DisjointLoader, Dataset
from spektral.datasets import TUDataset
from spektral.transforms import Degree, GCNFilter
from spektral.transforms.normalize_adj import NormalizeAdj

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping
from numpy import sqrt

physical_devices = tf.config.list_physical_devices("GPU")
#print (physical_devices)
#tf.debugging.set_log_device_placement(True)
#input()
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

classlist = [728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2279, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 
2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2644, 2645, 2646, 2647, 2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3291, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3511, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3696, 3697, 3698, 3699, 3700, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 
3709, 3710, 3711, 3712, 3713, 3714, 3715, 3716, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3864, 3865, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987, 3988, 3989, 4073, 4074, 4075, 4076, 4077, 4078, 4142, 4143, 4144, 4145, 4185, 4227]

def load_graph_file(filename):
    r_data = []
    with open(filename, "r") as f:
        for l in f.readlines():
            grep,cl = eval(l)
            vl, edges = grep

            nnodes = len(vl)

            # node features
            X = np.array(vl).astype(int)
            
            # adjacency matrix:
            a = np.zeros((nnodes, nnodes))
            for x,y,_ in edges:
                a[x][y] = 1
                a[y][x] = 1
            
            # edge features:
            e = np.zeros((nnodes, nnodes, 1))
            for x,y,f in edges:
                f = round(f * f)
                e[x][y][0] = f
                e[y][x][0] = f
            
            # label:
            y = np.zeros((len(classlist), ))
            y[classlist.index(cl)] = 1
            #y = np.array(cl)
            r_data.append(Graph(x=X.astype(int), a=a.astype(int), e=e.astype(int), y=y))
    return r_data


test_data = load_graph_file("data/test_graphs_nonperm.txt")
print (test_data[0])
train_data = load_graph_file("data/train_graphs_nonperm.txt")
np.random.shuffle(train_data)
#train_data = train_data[:1000]

test_data = [g for g in test_data if g.n_nodes == 10]
train_data = [g for g in train_data if g.n_nodes == 10]

for g in test_data:
    assert g not in train_data

print (len(test_data), len(train_data))

class TestDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read(self):
        return test_data[::] + train_data[::]


class Net(Model):
    def __init__(self, n_labels):
        super().__init__()
        self.masking = GraphMasking()
        self.conv1 = ECCConv(32, [64, 32])
        self.conv2 = ECCConv(32, [64, 32], activation="relu")
        self.global_pool = GlobalAvgPool()
        self.activ = LeakyReLU(alpha=0.05)
        self.dropout = Dropout(0.01)
        self.dense = Dense(64, activation="relu")
        self.out = Dense(n_labels, activation="softmax")

    def call(self, inputs):
        x, a, e = inputs
        x = self.masking(x)
        x = self.conv1([x, a, e])
        x = self.activ(x)
        x = self.conv2([x, a, e])
        output = self.global_pool(x)
        output = self.dense(output)
        output = self.dropout(output)
        output = self.out(output)

        return output

    def full_save(self, filename):
        self.save_weights("models/ECC network/" + filename + ".h5", overwrite=True)
        #for i,layer in enumerate(self.conv1.kernel_network_layers):
        #    with open("models/ECC network/convfilters/layer1_sub" + str(i) + "_" + filename, "w") as f:
        #        np.save(f, layer.get_weights())
        #for i,layer in enumerate(self.conv2.kernel_network_layers):
        #    with open("models/ECC network/convfilters/layer2_sub" + str(i) + "_" + filename, "w") as f:
        #        np.save(f, layer.get_weights())

class SaveModelCheckpoint(Callback):
    def __init__(self):
        super().__init__()
        self.current_best_score = 100000

    def on_epoch_end(self, epoch, logs=None):
        cscore = logs["loss"]
        if cscore < self.current_best_score:
            self.current_best_score = cscore
            self.model.full_save("current_best")

def evaluate(loader):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        """
        pp = []
        for ppp in pred[0]:
            pp.append(ppp.numpy())
        tt = []
        for ttt in target[0]:
            tt.append(ttt)
        print (pp.index(max(pp)), tt.index(1))
        """
        outs = (
            loss_fn(target, pred),
            tf.reduce_mean(categorical_accuracy(target, pred)),
            len(target),  # Keep track of batch size
        )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])

dataset = TestDataset(transforms=NormalizeAdj())
#dataset_train.filter(lambda g: g.n_nodes == 10)

max_degree = int(dataset.map(lambda g: g.a.sum(-1).max(), reduce=max))
print (max_degree)

dataset.apply(Degree(max_degree))
#dataset.apply(GCNFilter())

dataset_test = dataset[:len(test_data)]
dataset_train = dataset[len(test_data):]

print (dataset.n_labels)
model = Net(dataset.n_labels)
#model.compile('adam', 'categorical_crossentropy')



learning_rate = 0.001 # Learning rate
epochs = 150  # Number of training epochs
es_patience = 400  # Patience for early stopping
batch_size = 32  # Batch size

train_loader = BatchLoader(dataset_train, batch_size=batch_size, epochs=epochs, mask=True)
test_loader = BatchLoader(dataset_test, batch_size=batch_size, mask=True)

optimizer = Adam(learning_rate=learning_rate)
loss_fn = CategoricalCrossentropy()

model.compile(optimizer=optimizer, loss="kl_divergence")

################################################################################
# Fit model
################################################################################
if __name__ == "__main__":
    print ("FITTING")
    mc = SaveModelCheckpoint()
    model.fit(train_loader.load(), steps_per_epoch=train_loader.steps_per_epoch, validation_steps=test_loader.steps_per_epoch, epochs=epochs, validation_data=test_loader.load(), callbacks=[mc])
else:
    model.built = True
    model.fit(train_loader.load(), steps_per_epoch=train_loader.steps_per_epoch, validation_steps=test_loader.steps_per_epoch, epochs=epochs, validation_data=test_loader.load(), callbacks = [EarlyStopping(monitor='loss', patience=1, min_delta=19999999)])
    model.load_weights("current_best.h5")

################################################################################
# Evaluate model
################################################################################
print("Testing model")
loss = model.evaluate(test_loader.load(), steps=test_loader.steps_per_epoch)
print("Done. Test loss: {}".format(loss))
print (evaluate(test_loader))

if __name__ == "__main__":
    model.full_save("final")

    
