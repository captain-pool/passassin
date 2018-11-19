from model import *
import tensorflow as tf
tf.reset_default_graph()
d1,d2,d3 = tf.random_normal([10,21,200]),tf.random_normal([10,21,200]),tf.random_normal([10,21,200])
disc = Discriminator(input = d1).build()
gen = Generator(disc = disc,real = d2,noise = d3).build()
