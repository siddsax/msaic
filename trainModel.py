
import torch
import argparse
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from getData import *
from classifier import *


def trainModel(model, optim, trainData, valData, args):

    #*****Hyper-Parameters******
    query_total_dim = args.q_max_words*args.emb_dim
    # label_total_dim = num_classes
    passage_total_dim = args.p_max_words*args.emb_dim

    for epoch in range(args.total_epochs):
        print("Epoch : ",epoch)
        n = 0.0
        k = 0.0
        for (pf, qf, lbl) in trainData:
            pf, lbl, qf = pf.type(torch.FloatTensor), lbl.type(torch.FloatTensor), qf.type(torch.FloatTensor)
            output = model(qf, pf)
            loss = args.loss(output, lbl)
            loss.backward()
            optim.step()
            optim.zero_grad()
            n += np.sum(np.equal(np.ceil(output.data.cpu().numpy()), lbl.data.cpu().numpy().reshape(-1, 1)))
            # import pdb; pdb.set_trace()
            k += 1*args.batchSize
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
