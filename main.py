import torch
import argparse
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from getData import *
from classifier import *
from trainModel import *
    
if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--qm', dest='q_max_words', type=int, default=12)
    parser.add_argument('--pm', dest='p_max_words', type=int, default=50)
    parser.add_argument('--e', dest='emb_dim', type=int, default=50)
    parser.add_argument('--mb', dest='batchSize', type=int, default=250)
    parser.add_argument('--es', dest='epoch_size', type=int, default=100000)
    parser.add_argument('--ep', dest='total_epochs', type=int, default=200)

    args = parser.parse_args()
    print(args)

    args.loss = nn.BCELoss()

    submissionFileName = "answer.tsv"
   
    trainData = torch.utils.data.DataLoader(
                Dataset('TrainData', 50, 
                transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(
                #     (0.1307,), (0.3081,))
                ])),
                batch_size=args.batchSize, shuffle=True)
    
    valData =   torch.utils.data.DataLoader(
                Dataset('ValidationData', 50, 
                transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(
                #     (0.1307,), (0.3081,))
                ])),
                batch_size=args.batchSize, shuffle=True)

    evalData = torch.utils.data.DataLoader(
                Dataset('EvaluationData', 50,
                transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(
                #     (0.1307,), (0.3081,))
                ])),
                batch_size=args.batchSize, shuffle=True)

    model = classifier1()
    optim = optim.Adam(model.parameters(), lr = 0.001)
    trainModel(model, optim, trainData, valData, args)
    # GetPredictionOnEvalSet(model,testSetFileName,submissionFileName) # Get Predictions on Evaluation Set

    
