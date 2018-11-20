import passgrugan as pgr
import tensorflow as tf
tf.reset_default_graph()
d1,d2,d3 = tf.random_normal([10,21,200]),tf.random_normal([10,21,200]),tf.random_normal([10,21,200])
disc = pgr.Discriminator(input = d1).build()
print("INITIATE DISCRIMINATOR...[PASSED]")
gen = pgr.Generator(disc = disc,real = d2,noise = d3).build()
print("INITIATE GENERATOR ...[PASSED]")
discLoss = pgr.losses.wasserstein_gp(d2,gen.output,gen.adv_out,disc = disc)
print("GRADIENT PENALTY LOSS WORKING [PASSED]")
genLoss = gen.loss
print("GENERATOR OPTIMIZER ...",end="")
genOptimize = gen.optimize()
print("[PASSED]\nDISCRIMINATOR OPTMIZER...",end = "")
discOptimize = disc.optimize(discLoss)
print("[PASSED]")
print("ALL TESTS PASSED. Go Nuts!")
