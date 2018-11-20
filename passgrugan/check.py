from model import *
from losses import losses
import tensorflow as tf
tf.reset_default_graph()
d1,d2,d3 = tf.random_normal([10,21,200]),tf.random_normal([10,21,200]),tf.random_normal([10,21,200])
disc = Discriminator(input = d1).build()
gen = Generator(disc = disc,real = d2,noise = d3).build()
discLoss = losses.wasserstein_gp(d2,gen.output,gen.adv_out,disc = disc)
print(discLoss)

