import torch
from torch.utils import data
import numpy as np
class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, dataName, embSize, transform=None):
        'Initialization'
        self.dataName = dataName
        self.transform = transform
        self.num_lines = sum(1 for line in open('data/' + dataName + '/PF.txt'))
        if dataName == "EvaluationData":
            self.eval = True
        else:
            self.eval = False
        self.embSize = embSize

  def __len__(self):
        'Denotes the total number of samples'
        return self.num_lines

  def __getitem__(self, index):
        'Generates one sample of data'
        f = 'data/' + self.dataName + '/PF.txt'
        with open(f) as f1:
            for i, line in enumerate(f1):
                if i == index:
                    line = line.split(' ')
                    pf = []
                    for i in line:
                        if len(i) > 0:
                            pf.append(float(i))
                    pf = np.array(pf)
                    break

        f = 'data/' + self.dataName + '/QF.txt'
        with open(f) as f1:
            for i, line in enumerate(f1):
                if i == index:
                    line = line.strip().split(' ')
                    qf = []
                    for i in line:
                        if len(i) > 0:
                            qf.append(float(i))
                    qf = np.array(qf)
                    break

        if self.eval:
            f = 'data/' + self.dataName + '/Id.txt'
        else:
            f = 'data/' + self.dataName + '/Lbl.txt'

        with open(f) as f1:
            for i, line in enumerate(f1):
                if i == index:
                    # line = line.split(' ')
                    Id = int(line)#np.array([float(i) for i in line])
                    break


        # return self.transform(pf.reshape(-1, self.embSize)), self.transform(qf.reshape(-1, self.embSize)), Id
        return pf.reshape(1, -1, self.embSize), qf.reshape(1, -1, self.embSize), Id

# kk = Dataset('TrainData', 50).__getitem__(0)
# import pdb; pdb.set_trace()