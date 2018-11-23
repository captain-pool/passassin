import passgrugan as pgr
from absl import flags,app
import tensorflow as tf
import numpy as np
import os
pgr.settings.init()
FLAGS = flags.FLAGS
BATCH_SIZE = 100
TIMESTEPS = 10
EPOCH = 1
print_freq = 1#00
DEBUG = False
MODELFILE = os.path.abspath("./saved.ckpt")
load = False
if os.path.basename(MODELFILE) in os.listdir(os.path.dirname(MODELFILE)): 
    if os.path.getsize(MODELFILE)>0:
        load = True
    else:
        os.remove(MODELFILE)
if DEBUG:
    print("LOAD:",load)
data = pgr.datapipe("rockyou.txt",window_size = TIMESTEPS,batch_size = BATCH_SIZE,epoch = EPOCH).read().build_vocab()
real = tf.placeholder(tf.float32,[None,TIMESTEPS,data.vocab_size])
disc_input = tf.placeholder(tf.float32,[None,TIMESTEPS,data.vocab_size])
noise = tf.placeholder(tf.float32,[None,TIMESTEPS,data.vocab_size])
print("INITIATE DISCRIMINATOR...",end = "")
disc = pgr.Discriminator(input = disc_input).build()
print("[PASSED]\nINITIATE GENERATOR...",end = "")
generator = pgr.Generator(disc = disc,real = real,noise = noise).build()
print("[PASSED]\nGP LOSS WORKING...",end = "")
discLoss = pgr.losses.wasserstein_gp(real,generator.output,generator.adv_out,disc = disc)
print("[PASSED]\nGENERATOR OPTMIZER...",end = "")
genOptimize = generator.optimize()
print("[PASSED]\nDISCRIMINATOR OPTIMIZER...",end = "")
discOptimize = disc.optimize(discLoss)
print("[PASSED]")
saver = tf.train.Saver()
with tf.Session() as sess:
    print("INITIATE VARIABLES...",end = "")
    if not load:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess,MODELFILE)
    print("[DONE]")
    _epoch = 1
    while True:
        try:
            l = next(data)
            print(len(l))
            if len(l)==3:
                tensor = l[0]
                batch_num = l[2]
                epoch = l[1]
                batch = sess.run(tensor)
            while epoch==None:
                batch = sess.run(tensor)
                Z = np.random.normal(size = batch.shape)
                if DEBUG:
                    print("[DONE]\nOPTMIZING GENERATOR...",end = "")
                _,fake,_gLoss = sess.run([genOptimize,generator.output,generator.loss],feed_dict={real:batch,noise:Z})
                if DEBUG:
                    print("[DONE]\nOPTIMIZING DISCRIMINATOR...",end = "")
                _,_dLoss = sess.run([discOptimize,discLoss],feed_dict={disc_input:fake,real:batch,noise:Z})
                if DEBUG:
                    print("[DONE]\nTEST BUILD PASSED!")
                    print(fake.shape)
                    print("GENERATING OUTPUT...")
                    print(next(data.decode(fake)))
                    sess.close()
                    exit(0)
                if batch_num%print_freq == 0:
                    print("EPOCH: %d\tGENERATOR LOSS: %f\tDISCRIMINATOR LOSS: %f"%(_epoch,_gLoss,_dLoss))
                l = next(data)
                if len(l)==2:
                    _epoch = l[1]
                    break
                tensor = l[0]
                epoch = l[1]
                batch_num = l[2]
        except StopIteration:
            save_path = saver.save(sess,MODELFILE)
            print("CHECKPOINT SAVED AT: %s"%save_path)
