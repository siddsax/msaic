import h5py
import torch
from torch.utils import data
import numpy as np
import subprocess

def toNpy(arrs):
    arrs = arrs.split('|')

    arr = arrs[0].split(' ')
    qf = []
    for i in arr:
        if len(i)> 0:
            qf.append(float(i))

    arr = arrs[1].split(' ')
    pf = []
    for i in arr:
        if len(i)> 0:
            pf.append(float(i))

    Id = np.zeros((1, 2))
    Id[0, int(arrs[2])] = 1

    return np.array(pf), np.array(qf), Id

class Dataset(data.Dataset):
#   'Characterizes a dataset for PyTorch'
    def __init__(self, dataName, embSize, transform=None):
        # 'Initialization'
        self.dataName = dataName
        self.transform = transform
        if dataName == "EvaluationData":
            # bashCommand = 'wc -l ' + 'data/traindata.tsv'
            self.num_lines = 524190
            self.eval = True
        elif dataName == "TrainData":
            self.num_lines = 3238000#4717690#45#1194000#4717690
            self.eval = False
        else:
            self.num_lines = 104170
            self.eval = False

        # self.num_lines = int(str(output).split(' ')[0].split('\'')[-1])#sum(1 for line in open('data/' + dataName + '/PF.txt'))
        self.embSize = embSize

    def __len__(self):
            'Denotes the total number of samples'
            return self.num_lines

    def __getitem__(self, index):
            'Generates one sample of data'
            f = 'data/' + self.dataName + '/data_' + str(index//1000) + '.txt'
            fg = 0
            with open(f) as f1:
                for i, line in enumerate(f1):
                    if i == index%1000:
                        pf, qf, Id = toNpy(line)
                        fg = 1
                        break
            if fg == 0:
                   with open(f) as f1:
                      for i, line in enumerate(f1):
                         pf, qf, Id = toNpy(line)
                         print(index)
                         break
            #print(pf.shape, qf.shape, Id.shape)
            #exit() 
            #f = 'data/' + self.dataName + '/QF_' + str(index//1000) + '.0.txt'
            #with open(f) as f1:
            #    for i, line in enumerate(f1):
            #        if i == index%1000:
            #            qf = toNpy(line)
            #            break
            #if self.eval:
            #    f = 'data/' + self.dataName + '/Id_' + str(index//1000) + '.0.txt'
            #else:
            #    f = 'data/' + self.dataName + '/Lbl_' + str(index//1000) + '.0.txt'
            
            #with open(f) as f1:
            #    for i, line in enumerate(f1):
            #        if i == index%1000:
            #            Id = np.zeros((1, 2))
            #            Id[0, int(line)] = 1
                        # if Id != 0 and Id != 1 and self.eval == 0:
                        #     import pdb;pdb.set_trace()
            #            break
            
            return pf.reshape(1, -1, self.embSize), qf.reshape(1, -1, self.embSize), Id

# # kk = Dataset('TrainData', 50).__getitem__(0)
# # import pdb; pdb.set_trace()

# class Dataset(data.Dataset):
#     """Custom Dataset for loading entries from HDF5 databases"""

#     def __init__(self, dataName, embSize, transform=None, test=False):
    
#         # self.num_lines = self.h5f['qf'].shape[0]
#         if dataName == "EvaluationData":
#             self.num_lines = 524190
#             self.eval = True
#         elif dataName == "TrainData":
#             self.num_lines = 4717690
#             self.eval = False
#         else:
#             self.num_lines = 104170
#             self.eval = False

#         self.dataName = dataName
#         self.transform = transform
#         self.test = test
#         self.embSize = embSize

#     def __getitem__(self, index):
        
#         index -= 1
#         while(1):
#             index = (index + 1)%self.num_lines
#             self.h5f = h5py.File('data/' + self.dataName + '_' + str(index//1000) + '.h5', 'r')
#             pf = self.h5f['pf'][index%1000].reshape(-1, self.embSize, 1)
#             qf = self.h5f['qf'][index%1000].reshape(-1, self.embSize, 1)
#             if self.test:
#                 lbl = self.h5f['Id'][index%1000]
#             else:        
#                 lbl = self.h5f['Lbl'][index%1000]
#                 if lbl !=0 and lbl !=1:
#                     # print("Wrong lbl")
#                     continue 

#             # import pdb; pdb.set_trace()
#             if self.transform is not None:
#                 pf = self.transform(pf)
#                 qf = self.transform(qf)
        
#             break
#         return pf, qf, lbl

#     def __len__(self):
#         return self.num_lines
