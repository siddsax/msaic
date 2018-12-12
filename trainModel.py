
import torch
import argparse
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from getData import *
from classifier import *
import numpy as np

def trainModel(model, optim, trainData, valData, args):

    #*****Hyper-Parameters******
    query_total_dim = args.q_max_words*args.emb_dim
    passage_total_dim = args.p_max_words*args.emb_dim

    for epoch in range(args.total_epochs):
        print("Epoch : ",epoch)
        n = 0.0
        k = 0.0
        for i, (pf, qf, lbl) in enumerate(trainData):
            # print("b")
            if torch.cuda.is_available():
                pf, lbl, qf = pf.type(torch.cuda.FloatTensor), lbl.type(torch.cuda.FloatTensor), qf.type(torch.cuda.FloatTensor)
            else:
                pf, lbl, qf = pf.type(torch.FloatTensor), lbl.type(torch.FloatTensor), qf.type(torch.FloatTensor)
            output = model(qf, pf)
            #import pdb;pdb.set_trace()
            loss = F.binary_cross_entropy(output.squeeze(), lbl.squeeze())
            loss.backward()
            optim.step()
            optim.zero_grad()
            # import pdb
            # pdb.set_trace()
            bc = 1.0*np.sum(np.equal(np.argmax(output.squeeze().data.cpu().numpy(), axis=1), np.argmax(lbl.squeeze().data.cpu().numpy(), axis=1)))
            kk = 1 - (1.0*np.sum(np.argmax(lbl.squeeze().data.cpu().numpy(), axis=1)))/(output.shape[0])
            #import pdb;pdb.set_trace()
            if i % 10 == 0:
                #if i > 1000:
                #    import pdb
                #    pdb.set_trace()
                # if loss.data.cpu().numpy() < 0:
                print("Batch {}/{}; Correct in Batch {}; Loss {} AllZ {}".format(i, len(trainData), bc/args.batchSize, loss.data.cpu().numpy(), bc/args.batchSize - kk))
            n += bc
            k += 1.0*args.batchSize
            # print("a")
        print(n/(k*args.batchSize))

    # ********* Model configuration *******
    # model_output = cnn_network(query_input_var, passage_input_var, num_classes)
    # loss = C.binary_cross_entropy(model_output, output_var)
    # pe = C.classification_error(model_output, output_var)
    # lr_per_minibatch = C.learning_rate_schedule(0.03, C.UnitType.minibatch)   
    # learner = C.adagrad(model_output.parameters, lr=lr_per_minibatch)
    # progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=args.total_epochs)

    # #************Create Trainer with model_output object, learner and loss parameters*************  
    # trainer = C.Trainer(model_output, (loss, pe), learner, progress_printer)
    # C.logging.log_number_of_parameters(model_output) ; print()

    # # **** Train the model in batchwise mode *****
    # for epoch in range(args.total_epochs):       # loop over epochs
    #     print("Epoch : ",epoch)
    #     sample_count = 0
    #     while sample_count < args.epoch_size:  # loop over minibatches in the epoch
    #         data = train_reader.next_minibatch(min(args.minibatch_size, args.epoch_size - sample_count), input_map=input_map) # fetch minibatch.
    #         trainer.train_minibatch(data)        # training step
    #         sample_count += data[output_var].num_samples   # count samples processed so far

    #     trainer.summarize_training_progress()
                
    #     model_output.save("CNN_{}.dnn".format(epoch)) # Save the model for every epoch
    
    #     #*** Find metrics on validation set after every epoch ******#  (Note : you can skip doing this for every epoch instead to optimize the time, do it after every k epochs)
    #     predicted_labels=[]
    #     for i in range(len(validation_query_vectors)):
    #         queryVec   = np.array(validation_query_vectors[i],dtype="float32").reshape(1,q_max_words,args.emb_dim)
    #         passageVec = np.array(validation_passage_vectors[i],dtype="float32").reshape(1,p_max_words,args.emb_dim)
    #         scores = model_output(queryVec,passageVec)[0]   # do forward-prop on model to get score  
    #         predictLabel = 1 if scores[1]>=scores[0] else 0
    #         predicted_labels.append(predictLabel) 
    #     metrics = precision_recall_fscore_support(np.array(validation_labels), np.array(predicted_labels), average='binary')
    #     #print("precision : "+str(metrics[0])+" recall : "+str(metrics[1])+" f1 : "+str(metrics[2])+"\n")



    # return model_output
