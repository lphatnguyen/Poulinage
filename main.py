import numpy as np
import torch
import torch.autograd as autograd

data = np.load("trainingData.npy")
s = np.arange(data.shape[0])
np.random.shuffle(s)
data = data[s,:]
numFeats = data.shape[1]-1
label = data[:,numFeats]
data = data[:,:numFeats]
label = torch.LongTensor(label)
data = torch.Tensor(data)
label = label.unsqueeze(1)
data = data.unsqueeze(1)

from neuNet import net2

net = net2().cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(net.parameters(),lr = 0.0001)
lossValueTrain = []

for epoch in range(100):
  runningLoss = 0
  for i in range(7100):
    inputs = autograd.Variable(data[i,:,:].cuda())
    lab = autograd.Variable(label[i].cuda())
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs,lab)
    loss.backward()
    optimizer.step()
    runningLoss = runningLoss + loss.data[0]
    
  print("Perte après l'itération", epoch, " est: ", runningLoss/7100)
  lossValueTrain.append(runningLoss/7100)
  
print("Finish training!!!")

correct = 0
for i in range(7100):
  inputs = autograd.Variable(data[i,:,:].cuda())
  lab = autograd.Variable(label[i].cuda())
  outputs = net(inputs)
  _,outputs = torch.max(outputs.data,1)
  correct += (outputs.cpu()==label[i]).sum()
    
print("Taux de correction:", correct/7100)