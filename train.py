import passgrugan as pgr
from absl import flags,app
import tensorflow as tf
import numpy as np
pgr.settings.init()
FLAGS = flags.FLAGS
BATCH_SIZE = 100
TIMESTEPS = 10
DEBUG = True
data = pgr.datapipe("rockyou.txt",window_size = TIMESTEPS,batch_size = BATCH_SIZE).read().build_vocab()
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
with tf.Session() as sess:
    print("INITIATE VARIABLES...",end = "")
    sess.run(tf.global_variables_initializer())
    print_freq = 100
    while True:
        try:
            counter = 0
            tensor,epoch = next(data)
            if tensor:
                batch = sess.run(tensor)
            _epoch = epoch
            while epoch==None:
                batch = sess.run(tensor)
                Z = np.random.normal(size = batch.shape)
                if DEBUG:
                    print("[DONE]\nOPTMIZING GENERATOR...",end = "")
                _,fake,_gLoss = sess.run([genOptimize,generator.output,generator.loss],feed_dict={real:batch,noise:Z})
                if DEBUG:
                    print("[DONE]\nOPTIMIZING DISCRIMINATOR...",end = "")
                _,_dLoss = sess.run([discOptimize,dLoss],feed_dict={disc_input:fake,real:batch,noise:Z})
                if DEBUG:
                    print("[DONE]\nTEST BUILD PASSED!")
                    print(fake.shape)
                    print("GENERATING OUTPUT...")
                    print(next(data.decode(fake)))
                    exit(0)
                counter+=1
                if print_freq%counter == 0:
                    print("EPOCH: %d\tGENERATOR LOSS: %f\tDISCRIMINATOR LOSS: %f"%(_epoch,_gLoss,_dLoss))
                tensor,epoch = next(data)
                if not tensor and epoch:
                    _epoch = epoch
                    break
        except StopIteration:
            break
