import subprocess
import os
import re
import io
import pandas as pd
import numpy as np
import h5py

#Initialize Global variables 
GloveEmbeddings = {}
max_query_words = 12
max_passage_words = 50
emb_dim = 50

def toNpy(arr):
    arr = arr.split(' ')
    out = []
    for i in arr:
        if len(i)> 0:
            out.append(float(i))

    return np.array(out)
#The following method takes Glove Embedding file and stores all words and their embeddings in a dictionary
def loadEmbeddings(embeddingfile):
    global GloveEmbeddings,emb_dim

    fe = io.open(embeddingfile,"r",encoding="utf-8",errors="ignore")
    for line in fe:
        tokens= line.strip().split()
        word = tokens[0]
        vec = tokens[1:]
        vec = " ".join(vec)
        GloveEmbeddings[word]=vec
    #Add Zerovec, this will be useful to pad zeros, it is better to experiment with padding any non-zero constant values also.
    GloveEmbeddings["zerovec"] = "0.0 "*emb_dim
    fe.close()


def TextDataToCTF(inputfile,outputfile,isEvaluation):
    global GloveEmbeddings,emb_dim,max_query_words,max_passage_words

    f = io.open(inputfile,"r",encoding="utf-8",errors="ignore")  # Format of the file : query_id \t query \t passage \t label \t passage_id
    outputfile = 'data/' + outputfile + '_'
    # if not os.path.exists(outputfile):
    #     os.makedirs(outputfile)
    # outputfile += '/'
    # f1 = open(outputfile + "QF.txt","w")#,encoding="utf-8")
    # f2 = open(outputfile + "PF.txt","w")#,encoding="utf-8")
    # if(not isEvaluation):
    #     f3 = open(outputfile + "Lbl.txt","w")#,encoding="utf-8")
    # else:
    #     f3 = open(outputfile + "Id.txt","w")#,encoding="utf-8")

    # bashCommand = 'wc -l ' + inputfile

    # process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()
    # num_lines = int(str(output).split(' ')[0].split('\'')[-1])#sum(1 for line in open('data/' + dataName + '/PF.txt'))


    num_lines = 1000
    for ind, line in enumerate(f):
        # with h5py.File(outputfile, 'w') as h5f:

        if ind%1000 == 0:
            print(ind)
            try:
                h5f.close()
            except:
                print("Initiating")
            h5f = h5py.File(outputfile + str(ind//1000) + ".h5", 'w')
            dset1 = h5f.create_dataset('qf',
                                   shape=(num_lines, emb_dim*max_query_words),
                                   compression=None,
                                   dtype='float32')
            dset2 = h5f.create_dataset('pf',
                                    shape=(num_lines,emb_dim*max_passage_words),
                                    compression=None,
                                    dtype='float32')
            if(not isEvaluation):
                dset3 = h5f.create_dataset('Lbl',
                                        shape=(num_lines,),
                                        compression=None,
                                        dtype='int32')
            else:
                dset3 = h5f.create_dataset('Id',
                                        shape=(num_lines,),
                                        compression=None,
                                        dtype='int32')
        #     f1 = open(outputfile + "QF_" + str(ind/1000) + ".txt","w")#,encoding="utf-8")
        #     f2 = open(outputfile + "PF_"+ str(ind/1000) + ".txt","w")#,encoding="utf-8")
        #     if(not isEvaluation):
        #         f3 = open(outputfile + "Lbl_"+ str(ind/1000) + ".txt","w")#,encoding="utf-8")
        #     else:
        #         f3 = open(outputfile + "Id_"+ str(ind/1000) + ".txt","w")#,encoding="utf-8")

        tokens = line.strip().lower().split("\t")
        query_id,query,passage,label = tokens[0],tokens[1],tokens[2],tokens[3]

        #****Query Processing****
        words = re.split('\W+', query)
        words = [x for x in words if x] # to remove empty words 
        word_count = len(words)
        remaining = max_query_words - word_count  
        if(remaining>0):
            words += ["zerovec"]*remaining # Pad zero vecs if the word count is less than max_query_words
        words = words[:max_query_words] # trim extra words
        #create Query Feature vector 
        query_feature_vector = ""
        for word in words:
            if(word in GloveEmbeddings):
                query_feature_vector += GloveEmbeddings[word]+" "
            else:
                query_feature_vector += GloveEmbeddings["zerovec"]+" "  #Add zerovec for OOV terms
        query_feature_vector = query_feature_vector.strip() 

        #***** Passage Processing **********
        words = re.split('\W+', passage)
        words = [x for x in words if x] # to remove empty words 
        word_count = len(words)
        remaining = max_passage_words - word_count  
        if(remaining>0):
            words += ["zerovec"]*remaining # Pad zero vecs if the word count is less than max_passage_words
        words = words[:max_passage_words] # trim extra words
        #create Passage Feature vector 
        passage_feature_vector = ""
        for word in words:
            if(word in GloveEmbeddings):
                passage_feature_vector += GloveEmbeddings[word]+" "
            else:
                passage_feature_vector += GloveEmbeddings["zerovec"]+" "  #Add zerovec for OOV terms
        passage_feature_vector = passage_feature_vector.strip() 

        #convert label
        # label_str = " 1 0 " if label=="0" else " 0 1 " 
        # import pdb
        # pdb.set_trace()
        if(not isEvaluation):
            dset1[ind%num_lines] = toNpy(query_feature_vector)
            dset2[ind%num_lines] = toNpy(passage_feature_vector)
            dset3[ind%num_lines] = int(label)
            if int(label) != 0 and int(label) != 1:
                import pdb
                pdb.set_trace()
            # f1.write(query_feature_vector + "\n")
            # f2.write(passage_feature_vector + "\n")
            # f3.write(label + "\n")
        # else:
            dset1[ind%num_lines] = toNpy(query_feature_vector)
            dset2[ind%num_lines] = toNpy(passage_feature_vector)
            dset3[ind%num_lines] = int(query_id)
            # f1.write(query_feature_vector + "\n")
            # f2.write(passage_feature_vector + "\n")
            # f3.write(query_id + "\n")

    # with h5py.File(outputfile, 'r') as h5f:
    #     print(h5f['qf'].shape)
    #     print(h5f['pf'].shape)

if __name__ == "__main__":

    trainFileName = "data/traindata.tsv"
    validationFileName = "data/validationdata.tsv"
    EvaluationFileName = "data/eval1_unlabelled.tsv"

    embeddingFileName = "glove/glove.6B.50d.txt"

    loadEmbeddings(embeddingFileName)    

    # Convert Query,Passage Text Data to CNTK Text Format(CTF) using 50-Dimension Glove word embeddings 
    TextDataToCTF(trainFileName,"TrainData",False)
    print("Train Data conversion is done")
    TextDataToCTF(validationFileName,"ValidationData",False)
    print("Validation Data conversion is done")
    TextDataToCTF(EvaluationFileName,"EvaluationData",True)
    print("Evaluation Data conversion is done")





