from re import X
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import GCNConv, GlobalSumPool, GCSConv, GlobalAvgPool, ECCConv, GraphMasking, GlobalMaxPool
from spektral.data import BatchLoader, Graph, DisjointLoader, Dataset
from spektral.datasets import TUDataset
from spektral.transforms import Degree, GCNFilter
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.models import GeneralGNN

import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from numpy import sqrt

#physical_devices = tf.config.list_physical_devices("GPU")
#print (physical_devices)
#input()
#if len(physical_devices) > 0:
#    tf.config.experimental.set_memory_growth(physical_devices[0], True)

classlist = [728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2279, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 
2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2644, 2645, 2646, 2647, 2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3291, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3511, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3696, 3697, 3698, 3699, 3700, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 
3709, 3710, 3711, 3712, 3713, 3714, 3715, 3716, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3864, 3865, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987, 3988, 3989, 4073, 4074, 4075, 4076, 4077, 4078, 4142, 4143, 4144, 4145, 4185, 4227]


print (classlist[61], classlist[177])
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


test_data = load_graph_file("test_graphs.txt")
test_data = [g for g in test_data if g.n_nodes == 10]

class TestDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read(self):
        return test_data[::]


dataset = TestDataset(transforms=NormalizeAdj())
max_degree = int(dataset.map(lambda g: g.a.sum(-1).max(), reduce=max))
dataset.apply(Degree(max_degree))


print (dataset.n_labels)
test_loader = BatchLoader(dataset, batch_size=32, mask=True)

loss_fn = CategoricalCrossentropy()
model = load_model("OUTPUTMODEL")

################################################################################
# Evaluate model
################################################################################
print("Testing model")
#predictions = model.predict(test_loader.load(), steps=test_loader.steps_per_epoch, verbose=1)
#print (list(zip([list(p).index(max(list(p))) for p in predictions], [list(td.y).index(max(list(td.y))) for td in test_data])))

#loss = model.test_on_batch(test_loader.load(), return_dict=True)
#print("Done. Test loss:", loss)




def evaluate(loader):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        
        pp = []
        for ppp in pred[0]:
            pp.append(ppp.numpy())
        tt = []
        for ttt in target[0]:
            tt.append(ttt)
        print (pp.index(max(pp)), tt.index(1), max(pp))
        
        outs = (
            tf.reduce_mean(categorical_accuracy(target, pred)),
            len(target),  # Keep track of batch size
        )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])

print (evaluate(test_loader))
