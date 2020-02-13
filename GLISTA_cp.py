import numpy as np
import tensorflow as tf
import utils.train

from utils.tf import shrink
from utils.tf import shrink_free
from utils.tf import shrink_ss_return_index
from models.LISTA_base import LISTA_base


### This in the main coding of the GLISTA with couple parameters
class GLISTA_cp (LISTA_base):
    def __init__(self, A, T, B, lam, uf, percent, max_percent, untied, coord, scope, alti, overshoot, gain,gain_fun,over_fun,both_gate,T_combine,T_middle):
        self._A = A.astype (np.float32)
        self._T = T
        self._B = B
        self._lam = lam
        self.uf = uf
        self._overshoot = overshoot
        self._gain = gain
        self._p = percent
        self._maxp = max_percent
        self._M = self._A.shape [0]
        self._N = self._A.shape [1]
        self._scale = 1.001 * np.linalg.norm (A, ord=2)**2
        self._theta = (self._lam / self._scale).astype(np.float32)
        self._alti = alti
        #We set theta as a vector forever
        self._theta = np.ones((self._N, 1), dtype=np.float32) * self._theta
        self._logep = -2.0
        if coord:
            self._theta = np.ones ((self._N, 1), dtype=np.float32) * self._theta
        self.gain_gate = []
        self.over_gate = []
        self.gain_fun = gain_fun
        self.over_fun = over_fun
        self.both_gate = both_gate
        self._T_combine = T_combine
        self._T_middle = T_middle
        if self.both_gate:
            for i in range(0,self._T):
                self.over_gate.append(self.over_fun)
                if self.gain_fun == 'combine':
                    if i>self._T_combine:
                        self.gain_gate.append('inv')
                        #self.over_gate.append('none')
                    else:
                        self.gain_gate.append('relu')##2
                        #self.over_gate.append(self.over_fun)
                else:
                    self.gain_gate.append(self.gain_fun)
        else:
            for i in range(0,self._T):
                if i>self._T_middle:
                    self.gain_gate.append(self.gain_fun)
                    self.over_gate.append('none')
                else:
                    self.gain_gate.append('none')
                    self.over_gate.append(self.over_fun)
        print self.gain_gate
        print self.over_gate
        self._ps = [(t+1) * self._p for t in range (self._T)]
        self._ps = np.clip (self._ps, 0.0, self._maxp)
        self._untied = untied
        self._coord = coord
        self._scope = scope
        if self.uf == 'combine' and self._gain:
            self.combine_function = True
        else:
            self.combine_function = False
        """ Set up layers."""
        self.setup_layers()
    def setup_layers(self):
        Bs_     = []
        Ws_     = []
        thetas_ = []
        D_       = []
        D_over   = []
        W_g_     = []
        B_g_     = []
        b_g_     = []
        log_epsilon_ = []
        alti_    = []
        alti_over = []
        B = (np.transpose(self._A) / self._scale).astype(np.float32)
        W = np.eye(self._N, dtype=np.float32) - np.matmul(B, self._A)
        B_g = (np.transpose(self._A) / self._scale).astype(np.float32)
        W_g = np.eye(self._N, dtype=np.float32) - np.matmul(B, self._A)
        b_g = np.zeros((self._N,1),dtype=np.float32)

        D = np.ones((self._N,1),dtype=np.float32)
        with tf.variable_scope (self._scope, reuse=False) as vs:
            # constant
            self._kA_ = tf.constant (value=self._A, dtype=tf.float32)
            if not self._untied: # tied model
                Ws_.append(tf.get_variable (name='W', dtype=tf.float32,
                                             initializer=B))
                Ws_ = Ws_ * self._T
            for t in range (self._T):
                thetas_.append (tf.get_variable (name="theta_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=self._theta))
                alti_.append (tf.get_variable (name="alti_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=self._alti))
                alti_over.append (tf.get_variable (name="alti_over_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=self._alti))

                if t < 7:
                    _logep = self._logep
                elif t <= 10:
                    _logep = self._logep - 2.0
                else:
                    _logeq = -7.0
                log_epsilon_.append (tf.get_variable(name='log_epsilon_%d'%(t+1),dtype=tf.float32,initializer=_logep))
                if self._untied: # untied model
                    Ws_.append (tf.get_variable (name="W_%d"%(t+1),
                                                 dtype=tf.float32,
                                                 initializer=B))
            for t in range(self._T):
                D_.append(tf.get_variable(name='D_%d'%(t+1), dtype=tf.float32, initializer=D))
                D_over.append(tf.get_variable(name='D_over_%d'%(t+1), dtype=tf.float32, initializer=D))

            if False:
                D_.append(tf.get_variable(name='D',dtype=tf.float32,initializer=D))
                D_ = D_ * self._T
            B_g_.append(tf.get_variable(name='B_g',shape = B_g.shape, dtype=tf.float32,
                                        initializer=tf.glorot_uniform_initializer()))
            B_g_ = B_g_ * self._T

            b_g_.append(tf.get_variable(name='b_g',shape = b_g.shape, dtype=tf.float32,
                                        initializer=tf.glorot_uniform_initializer()))
            b_g_ = b_g_ * self._T

            W_g_.append(tf.get_variable(name='W_g',shape = W_g.shape ,dtype=tf.float32,
                                  initializer=tf.glorot_uniform_initializer()))
            W_g_ = W_g_ * self._T


        # Collection of all trainable variables in the model layer by layer.
        # We name it as `vars_in_layer` because we will use it in the manner:
        # vars_in_layer [t]
        self.vars_in_layer = list (zip (log_epsilon_, Ws_, thetas_,B_g_,W_g_,b_g_,D_,alti_,D_over,alti_over))


    def inference(self, y_, x0_=None):
        def reweight_function(x, D, theta, alti_):
            reweight = 1.0 + alti_*theta*tf.nn.relu(1-tf.nn.relu(D*tf.abs(x)))
            return reweight

        def reweight_inverse(x, D, theta, alti):
            reweight = 1.0 + alti * theta * 0.2/(0.001 + tf.abs(D*x))
            return reweight

        def reweight_exp(x, D, theta, alti):
            reweight = 1.0 + alti * theta * tf.exp(-D * tf.abs(x))
            return reweight

        def reweight_sigmoid(x, D, theta, alti):
            reweight = 1.0 + alti * theta * tf.nn.sigmoid(-D * tf.abs(x))
            return reweight

        def reweight_inverse_variant(x, D, theta, alti, epsilon):
            reweight = 1.0 + alti * theta * 0.2/(epsilon+tf.abs(D*x))
            return reweight

        def gain(x, D, theta, alti_, epsilon,gain_fun):
            if gain_fun == 'relu':
                use_function = reweight_function
            if gain_fun == 'inv':
                use_function = reweight_inverse
            elif gain_fun == 'exp':
                use_function = reweight_exp
            elif gain_fun == 'sigm':
                use_function = reweight_sigmoid
            elif gain_fun == 'inv_v':
                use_function = reweight_inverse_variant
                return  use_function(x, D, theta, alti_, epsilon)
            elif gain_fun == 'none':
                return 1.0 + 0.0*reweight_function(x, D, theta, alti_)+0.0*epsilon
            return use_function(x, D, theta, alti_) + 0.0*epsilon
        def overshoot(alti,Part_1,Part_2):
            if self._overshoot:
                return 1.0 - alti * Part_1 * Part_2
            else:
                return 1.0 + 0.0 * alti * Part_1 * Part_2
        xhs_= [] # collection of the regressed sparse codesnm
        if x0_ is None:
            batch_size = tf.shape(y_)[-1]
            xh_ = tf.zeros(shape=(self._N, batch_size), dtype=tf.float32)
        else:
            xh_ = x0_
        xhs_.append(xh_)
        a=[]
        with tf.variable_scope (self._scope, reuse=True) as vs:
            for t in range (self._T):
                log_epsilon, W_, theta_, B_g, W_g, b_g, D, alti, D_over, alti_over = self.vars_in_layer[t]
                percent = self._ps [t]
                By_ = tf.matmul (W_, y_)
                Part_1_sig = tf.nn.sigmoid(tf.matmul(W_g, xh_) + tf.matmul(B_g, y_) + b_g)
                Part_2_sig = (tf.abs(By_))
                if t == 0:
                    in_ = gain(xh_, D, theta_*0.0+1.0, alti, tf.exp(log_epsilon),self.gain_gate[t])
                    Part_2_inv = theta_
                else:
                    in_ = gain(xh_, D, theta_p*0.0+1.0, alti, tf.exp(log_epsilon),self.gain_gate[t])
                    Part_2_inv = theta_p
                    # This is the bound of layer of ReLU and inverse function.
                res_ = y_ - tf.matmul (self._kA_,in_*xh_)

                xh_title, cindex = shrink_ss_return_index(in_*xh_ + tf.matmul(W_, res_), theta_, percent)
                Part_1_inv = 1.0 / (abs(xh_title - xh_) + 0.1) + 0.0 * Part_1_sig
                if self.over_gate[t] == 'inv':
                    g_ = overshoot(alti_over, Part_1_inv, Part_2_inv)+0.0*D_over
                elif self.over_gate[t] == 'sigm':
                    g_ = overshoot(alti_over, Part_1_sig, Part_2_sig)+0.0*D_over
                elif self.over_gate[t] == 'none':
                    g_ = 1.0+0.0*alti_over*Part_1_sig*Part_2_sig+0.0*D_over
                theta_p = theta_ * cindex
                xh_ = g_ * xh_title + (1 - g_) * xh_
                xhs_.append(xh_)
                a.append(in_)
        a.append(tf.ones_like(in_))
        return xhs_
