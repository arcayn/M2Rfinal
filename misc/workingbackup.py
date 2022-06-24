import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import GCNConv, GlobalSumPool, GCSConv, GlobalAvgPool, ECCConv
from spektral.data import BatchLoader, Graph, DisjointLoader, Dataset
from spektral.datasets import TUDataset
from spektral.transforms import Degree, GCNFilter
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.models import GeneralGNN

import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam
from numpy import sqrt

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_graph_file(filename):
    r_data = []
    with open(filename, "r") as f:
        for l in f.readlines():
            edges,cl = eval(l)

            nnodes = max(max(x,y) for x,y,_ in edges) + 1

            # node features
            X = np.zeros((nnodes, 1))
            
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
            y = np.zeros((4319, ))
            y[cl] = 1
            r_data.append(Graph(x=X.astype(int), a=a.astype(int), e=e.astype(int), y=y.astype(int)))
    return r_data


test_data = load_graph_file("test_graphs.txt")
train_data = load_graph_file("train_graphs.txt")

#test_data = [g for g in test_data_ if g.n_nodes == 10]
#train_data = [g for g in train_data_ if g.n_nodes == 10]

print (len(test_data), len(train_data))

class TestDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read(self):
        return test_data[::] + train_data[::]

class TrainDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read(self):
        return train_data

class Net(Model):
    def __init__(self, n_labels):
        super().__init__()
        self.conv1 = ECCConv(32, activation="relu")
        self.conv2 = ECCConv(32, activation="relu")
        self.global_pool = GlobalSumPool()
        self.dense = Dense(n_labels, activation="softmax")

    def call(self, inputs):
        x, a, e, i = inputs
        X_2 = self.conv1([x, a, e])
        X_3 = self.global_pool([X_2, i])
        output = self.dense(X_3)

        return output

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



learning_rate = 1e-3  # Learning rate
epochs = 400  # Number of training epochs
es_patience = 400  # Patience for early stopping
batch_size = 32  # Batch size

train_loader = DisjointLoader(dataset_train, batch_size=batch_size, epochs=epochs)
test_loader = DisjointLoader(dataset_test, batch_size=batch_size)

optimizer = Adam(learning_rate=learning_rate)
loss_fn = CategoricalCrossentropy()


@tf.function(input_signature=train_loader.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    #print (inputs)
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(categorical_accuracy(target, predictions))
    return loss, acc


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


epoch = step = 0
best_val_loss = np.inf
best_weights = None
patience = es_patience
results = []
for batch in train_loader:
    #print ("\n\n\n")
    #print (batch[0][0])
    #print ("\n\n\n")
    #print (batch[0][1])
    #print ("\n\n\n")
    #print (batch[0][2])
    #print ("\n\n\n")
    #print (batch[0][3])
    #print ("\n\n\n")
    #print (batch[1])
    #print ("\n\n\n")
    step += 1
    loss, acc = train_step(*batch)
    results.append((loss, acc))
    if step == train_loader.steps_per_epoch:
        step = 0
        epoch += 1

        # Compute validation loss and accuracy
        val_loss, val_acc = evaluate(test_loader)
        print(
            "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Val loss: {:.3f} - Val acc: {:.3f}".format(
                epoch, *np.mean(results, 0), val_loss, val_acc
            )
        )

        # Check if loss improved for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = es_patience
            print("New best val_loss {:.3f}".format(val_loss))
            best_weights = model.get_weights()
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping (best val_loss: {})".format(best_val_loss))
                break
        results = []

model.set_weights(best_weights)  # Load best model
test_loss, test_acc = evaluate(test_loader)
print("Done. Test loss: {:.4f}. Test acc: {:.2f}".format(test_loss, test_acc))


#model.compile(optimizer='adam', loss='huber_loss')

#model.fit(train_loader.load(), steps_per_epoch=train_loader.steps_per_epoch, epochs=epochs, validation_data=test_loader.load())

#loss = evaluate(test_loader)
#print (loss)