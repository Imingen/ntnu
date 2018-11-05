from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import SoftmaxLayer

dataset = SupervisedDataSet(1,1)
for sample in range(1,9):
    dataset.addSample((sample,), (sample,))

# Endrer antall hidden noder gjennom 
net = buildNetwork(1,6,1, bias=True, hiddenclass=TanhLayer)

trainer = BackpropTrainer(net, dataset)

x = trainer.trainUntilConvergence(verbose=False, validationProportion=0.15, 
                                maxEpochs=1000, continueEpochs=10)

# Endre hvilket heltall man skal sjekke gjenskapning på 
y = net.activate([6.7])
print(y)



