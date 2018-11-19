import tensorflow as tf
from .settings import settings
class losses:
    def wasserstein_gp(batch_size,real_data,disc_real,fake_data,disc_fake,**kwargs):
        '''
        Finds Wasserstein Loss with Gradient Penalty between 
        two probabilty distribution.
        Params:
            - batch_size: Size of each batch
            - real_data: Real Data from dataset
            - disc_real: Discriminator output when real data is passed
            - fake_data: Fake Data Generated with which to compare
            - disc_fake: Discriminator output when fake data is passed
            - lamdba: rate of effectiveness of gradient penalty term.
                    Default: 10
            - disc: Discriminator Object
        '''
        lambda_ = kwargs.get("lambda",0.1)
        disc = kwargs["disc"]
        adv_loss = -(tf.reduce_mean(disc_real)-tf.reduce_mean(disc_fake))
        difference = real_data - fake_data
        alpha = tf.random_uniform([batch_size,1],0,1)
        inter = real_data+alpha*difference
        out = disc.set_input(inter).build(reuse = True)
        grad = tf.gradients(out,[inter])[0]
        penalty = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_sum(tf.square(grad),axis = 1))-1))
        adv_loss += lambda_*penalty
        return gen_loss

