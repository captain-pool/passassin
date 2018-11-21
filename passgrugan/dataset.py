import tensorflow as tf
import os
import random
import numpy as np
class Data:
    def __init__(self,path,batch_size = 32,window_size = 3,verbose = True,epoch = 2):
        self._filename = os.path.abspath(path)
        self._ws = window_size
        self._iterover = False
        self.batch_size = batch_size
        self._v = verbose
        self._epoch = epoch
    def read(self):
        if self._v:
            print("Opening File: {}".format(os.path.basename(self._filename)),end = " ")
        with open(self._filename,"rb") as f:
            self._raw = f.read()
            self._len = len(self._raw)
        if self._v:
            print("[DONE]")
        return self
    def build_vocab(self):
        if self._v:
            print("Building Vocab",end = " ")
        vocab = dict()
        vocab["<UNK>"] = 0
        vocab["<RET>"] = 1
        count = 2
        st = self._raw
        string = list(set(st))
        for i in string:
            try:
                c = bytes([i]).decode("utf-8")
                if not vocab.get(c,None) and (c!="\n" or c!="\r"):
                    vocab[c] = count;count+=1
            except:
                pass
            self._vocab = vocab
            self._vocab_T = dict(list(zip(self._vocab.values(),self._vocab.keys())))
            self.vocab_size = len(vocab.keys())
        if self._v:
            print("[DONE]")
        return self
    def encode(self,string):
        encoded = []
        for i in string:
            try:
                c = bytes([i]).decode("utf-8")
                if c == "\n" or c == "\r":
                    if encoded[-1]!=self._vocab["<RET>"]:
                        encoded.append(self._vocab["<RET>"])
                elif self._vocab.get(c,None):
                    if c == "\n" or c=="\r":
                        c = "<RET>"
                    encoded.append(self._vocab.get(c))
                else:
                    raise UnicodeDecodeError("utf-8",bytes([i]),0,1,"Error Deliberately thrown")
            except UnicodeDecodeError:
                    encoded.append(self._vocab.get("<UNK>"))
        return self,encoded          
    def decode(self,l):
        for i in l:
            data = [np.argmax(j) for j in i]
            s = "".join([self._vocab_T.get(x,None) for x in data])
            yield self,s
    def __next__(self):
        iterCount = 0
        tensorList = []
        idx = -1
        length = len(self._raw)
        for outerLoop in range(self._epoch):
            while True:
                idx = idx+1
                if (idx+self._ws) >=length:
                    break
                if '\n'.encode("utf-8")[0] in self._raw[idx:idx+self._ws]:
                    index = idx+self._ws-1
                    for i in range(idx,idx+self._ws):
                       if '\n'.encode("utf-8")[0] == self._raw[i]:
                           index = i
                           break
                    #finalBytes = bytes(list(self._raw[idx:idx+index])+(self._ws-index+1)*[255])
                    finalBytes = self._raw[idx-i-1:idx+self._ws]
                else:
                    finalBytes = self._raw[idx:idx+self._ws]
                    _,l = self.encode(finalBytes)
                    # For Test
                    # _,s = self.decode(l)
                    tensorList.append(tf.one_hot(l,self.vocab_size))
                    iterCount += 1
                    if iterCount%self.batch_size == 0:
                        iterCount = 0
                        v = tf.stack(tensorList,axis = 0,name = "data")
                        tensorList = []
                        return v,None
            return None, (outerLoop+1)

