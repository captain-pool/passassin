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
        self.outerLoop = 0
        self.idx = -1
        self.counter = 0
        self.ht = {}
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
        self._length = len(self._raw)
        self.prng_gen = self.prng()
        return self
    def encode(self,string):
        encoded = []
        for i in string:
            try:
                c = bytes([i]).decode("utf-8")
                if (c == "\n" or c == "\r") and len(encoded)>0:
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
    def decode(self,l,test = False):
        if test: 
            s = "".join([self._vocab_T.get(x,None) for x in l])
            print(s)
            return
        lst = []
        for i in l:
            data = [np.argmax(j) for j in i]
            s = "".join([self._vocab_T.get(x,None) for x in data])
            yield s
    #Simulating Rolling a Biased Die    
    def roll(self,massDist):
        randRoll = random.random() # in [0,1)
        sum = 0
        result = 1
        for mass in massDist:
            sum += mass
            if randRoll < sum:
                return result
            result+=1
    #Generting unique numbers by simulating biased die roll
    def fill(self):
        l = np.ones(self.raw.__len__(),dtype = np.float32)
        l[self.ht.keys()] = 0
        f = np.count_nonzero(l>0)
        self._length = f
        l = l/f
        self.l = l
    def biased_die_method(self):
        self.fill()
        while not np.all(self.l==0.0):
            r = self.roll(self.l)-1
            if self._length>1:
                self.l*=float(self._length)/(self._length-1)
                self._length-=1
            self.l[r] = 0
            self.ht[r] = 1
            return r
    def uniform_random_draw(self):
        self.ht = {}
        r = random.randrange(0,self._raw.__len__(),1)
        for _ in range(30):
            if not self.ht.get(r,None):
                self.ht[r] = 1
                return r
        return self.biased_die_method()
    def prng(self):
        while abs(len(self.ht.keys())-self._raw.__len__())>5:
            v = self.uniform_random_draw()
            yield v
    def __next__(self):
        iterCount = 0
        tensorList = []
        while self.outerLoop <self._epoch:
            while True:
                try:
                    self.idx = next(self.prng_gen)
                except StopIteration:
                   break
                finalBytes = self._raw[self.idx:self.idx+self._ws]+self._raw[:max(0,-1*(len(self._raw)-self.idx-self._ws))]
                _,l = self.encode(finalBytes)
                # For Test
                tensorList.append(tf.one_hot(l,self.vocab_size))
                iterCount += 1
                if iterCount%self.batch_size == 0:
                    self.counter+=1
                    v = tf.stack(tensorList,axis = 0,name = "data")
                    tensorList = []
                    return v,None,self.counter
            self.outerLoop += 1
            self.counter = 0
            return None, (self.outerLoop+1)

