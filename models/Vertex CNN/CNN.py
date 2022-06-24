import keras
import tensorflow as tf
import numpy as np
import random
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, LeakyReLU, Dense, AvgPool2D, ReLU, Dropout,Normalization
from keras_preprocessing.sequence import pad_sequences
from keras.losses import mse

from tensorflow.keras.optimizers import Adam

classlist = [728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2279, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 
2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2644, 2645, 2646, 2647, 2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3291, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3511, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3696, 3697, 3698, 3699, 3700, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 
3709, 3710, 3711, 3712, 3713, 3714, 3715, 3716, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3864, 3865, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987, 3988, 3989, 4073, 4074, 4075, 4076, 4077, 4078, 4142, 4143, 4144, 4145, 4185, 4227]


def load_matrix_file(filename):
    items = []
    labels = []
    with open(filename, "r") as f:
        rlns = f.readlines()
        random.shuffle(rlns)
        for l in rlns:
            mrep,cl = eval(l)
            mrep = eval(mrep)
            for i in range(len(mrep)):
                mrep[i][0] = 0
                mrep[i][1] = 0
                mrep[i][2] = 0
            mrep = sorted(mrep)
            
            items.append(np.array(mrep).astype(int))
            labels.append(classlist.index(cl))
    print (items[-1].shape)

    items, labels = np.array(items).astype(np.float64), np.array(labels).astype(np.float64)
    #sample_means = np.mean(items, axis=0)
    #sample_stdev = np.std(items, axis=0)
    #items -= sample_means
    #items /= sample_stdev
    return items, labels

X_train, Y_train_re = load_matrix_file("data/train_matrices.txt")
X_test, Y_test_re = load_matrix_file("data/test_matrices.txt")

#X_train, X_test = X_train.repeat(8, axis = 1).repeat(6, axis = 2), X_test.repeat(8, axis = 1).repeat(6, axis = 2)
def normalize(x):
    layer = Normalization()
    layer.adapt(x)
    return layer(x)

X_train = normalize(X_train)
X_test = normalize(X_test)
trainingX = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
validateX  = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
trainingY=tf.keras.utils.to_categorical(Y_train_re, num_classes=None, dtype='float32')
validateY=tf.keras.utils.to_categorical(Y_test_re, num_classes=None, dtype='float32')

#model = keras.applications.Xception(
#    weights=None, input_shape=(80, 78, 1), classes=405)
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
#model.fit(trainingX, trainingY, batch_size=32, epochs=10, validation_data=(validateX, validateY))


#Y_train_re = np.reshape(Y_train,(Y_train.shape[0],1))
#Y_test_re = np.reshape(Y_test,(Y_test.shape[0],1))
#trainingX = X_train
#validateX = X_test

trainingY=tf.keras.utils.to_categorical(Y_train_re, num_classes=None, dtype='float32')
validateY=tf.keras.utils.to_categorical(Y_test_re, num_classes=None, dtype='float32')

print (trainingY[0])

input_shapeX = trainingX.shape

print (trainingX.shape, input_shapeX, trainingX[0])

model = Sequential()
model.add(Conv2D(32, kernel_size=4, padding='same',input_shape=input_shapeX[1:],data_format="channels_last",activation="relu"))
#model.add(LeakyReLU(alpha=0.05))

model.add(Conv2D(32, kernel_size=4, padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(AvgPool2D(pool_size=(2,2), strides=(1,1), padding='same'))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.01))         
model.add(Dense(405,activation='softmax',input_shape=input_shapeX[1:]))

optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer, loss='categorical_crossentropy')

model.summary()

model.fit(trainingX,trainingY,batch_size=32,epochs=5,validation_data=(validateX,validateY))

predictY = model.predict(validateX,verbose = 1)

print (sum(mse(validateY, predictY).numpy())/11)
input()

for i,p in enumerate(predictY):
    q = validateY[i]
    print (list(p).index(max(p)), list(q).index(max(q)))
#print (predictY)
