import sys
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
from scipy.interpolate import interp2d

class SolidificationPINN:
    def __init__(self, x0, tem0, tb, X_f, X_tem, layers, lb, ub):
        
        ltem = 298.15
        utem = 973.15

        X0 = np.concatenate((x0, 0*x0+5.0), 1) # (x0, 5)
        X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)

        tem_lb = np.concatenate((0*tb + ltem, tb), 1)
        tem_ub = np.concatenate((0*tb + utem, tb), 1)
        
        self.lb = lb
        self.ub = ub

        self.x0 = X0[:,0:1]
        self.t0 = X0[:,1:2]

        self.x_lb = X_lb[:,0:1]
        self.t_lb = X_lb[:,1:2]

        self.x_ub = X_ub[:,0:1]
        self.t_ub = X_ub[:,1:2]

        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]

        self.X_tem = X_tem[:,0:1]

        self.tem0 = tem0
        self.tem_lb = tem_lb[:,0:1]
        self.tem_ub = tem_ub[:,0:1]

        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        self.x0_tf = tf.convert_to_tensor(self.x0, dtype=tf.float64)
        self.t0_tf = tf.convert_to_tensor(self.t0, dtype=tf.float64)

        self.tem0_tf = tf.convert_to_tensor(self.tem0, dtype=tf.float64)

        self.x_lb_tf = tf.convert_to_tensor(self.x_lb, dtype=tf.float64)
        self.t_lb_tf = tf.convert_to_tensor(self.t_lb, dtype=tf.float64)
        self.tem_lb_tf = tf.convert_to_tensor(self.tem_lb, dtype=tf.float64)

        self.x_ub_tf = tf.convert_to_tensor(self.x_ub, dtype=tf.float64)
        self.t_ub_tf = tf.convert_to_tensor(self.t_ub, dtype=tf.float64)
        self.tem_ub_tf = tf.convert_to_tensor(self.tem_ub, dtype=tf.float64)

        self.x_f_tf = tf.convert_to_tensor(self.x_f, dtype=tf.float64)
        self.t_f_tf = tf.convert_to_tensor(self.t_f, dtype=tf.float64)

        self.X_tem_tf = tf.convert_to_tensor(self.X_tem, dtype=tf.float64)

        # tf Graphs
        self.tem0_pred = self.net_uv(self.x0_tf, self.t0_tf)
        self.tem_lb_pred = self.net_uv(self.x_lb_tf, self.t_lb_tf)
        self.tem_ub_pred = self.net_uv(self.x_ub_tf, self.t_ub_tf)
        self.f_tem_pred = self.net_f_uv(self.x_f_tf, self.t_f_tf)
        self.X_tem_pred = self.net_uv(self.x_f_tf, self.t_f_tf)

        self.loss_pre = tf.reduce_mean(tf.square(self.X_tem_pred - self.X_tem_tf))
        self.loss = tf.reduce_mean(tf.square(self.tem0_pred - self.tem0_tf)) + 1.0e-3 * tf.reduce_mean(tf.square(self.f_tem_pred))

        self.global_step = tf.Variable(0, trainable=False, dftype=tf.int64)
        self.decayed_lr = tf.compat.v1.train.exponential_decay(
            learning_rate = 0.1,
            global_step=self.global_step,
            decay_steps=2000,
            decay_rate=0.1,
            staircase=True
        )

        self.decayed_lr2 = tf.compat.v1.train.cosine_decay(
            learning_rate=0.001, 
            global_step=self.global_step, 
            decay_steps=5000,  # adjust as needed
        )

        self.decayed_lr3 = tf.compat.v1.train.polynomial_decay(
            learning_rate=0.001,
            global_step=self.global_step,
            decay_steps=5000,  # adjust as needed
            end_learning_rate=0.0001,
            power=2.0,
            cycle=False
        )

        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        self.sess = tf.compat.v1.Session()
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64), dtype=tf.float64)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def cal_H(self, x):
        eps= 1e-3
        x1 = (-0.4 + eps)*tf.ones_like(x)
        x2 = ( 0.4 - eps)*tf.ones_like(x)
        one = tf.ones_like(x)

        d1 =  x - x1
        d2 = x2 - x
        dist = tf.minimum(d1, d2)

        Hcal = 0.5*( one + dist/eps + 1.0/np.pi*tf.sin(dist*np.pi/eps) )

        #xtmp = tf.where(tf.greater(dist, eps), tf.ones_like(x), Hcal)
        xout = tf.where(tf.less(dist, -eps), tf.zeros_like(x), tf.where(tf.greater(dist, eps), tf.ones_like(x), Hcal))

        return xout

    def net_uv(self, x, t):
        X = tf.concat([x, t], axis=1)

        Hcal = self.cal_H(x)
        one = tf.ones_like(x)

        T1   = 298.15*one
        T2   = 973.15*one
        xlen = 0.8
        dx   = x + 0.4*one
        Tbc  = T1 + (T2-T1)/xlen*dx

        tem = self.neural_net(X, self.weights, self.biases)
        tem = Tbc*(one-Hcal) + tem*Hcal

        return tem

    def net_f_uv(self, x, t):
        tem = self.net_uv(x, t)

        one = tf.ones_like(tem)
        zero = tf.zeros_like(tem)

        rho_Al_liquid = 2555.0 * one
        rho_Al_solid = 2555.0 * one
        rho_grap = 2200.0 * one

        kappa_Al_liquid = 91.0 * one
        kappa_Al_solid = 211.0 * one
        kappa_grap = 100.0 * one

        cp_Al_liquid = 1190.0 * one
        cp_Al_solid = 1190.0 * one
        cp_grap = 1700.0 * one

        cl_Al_liquid = 3.98e5 * one
        cl_Al_solid = 3.98e5 * one
        cl_grap = 3.98e5 * one

        # Value of Ts
        Ts = 913.15 * one
        Tl = 933.15 * one

        tem_t = tf.gradients(tem, t)[0]
        tem_x = tf.gradients(tem, x)[0]
        fL = (tem - Ts) / (Tl - Ts)
        fL = tf.maximum(tf.minimum((tem - Ts) / (Tl - Ts), one), zero)
        fL_t = tf.gradients(fL, t)[0]

        rho = tf.where(tf.greater(x, zero), rho_Al_liquid * fL + rho_Al_solid * (one - fL), rho_grap)
        kappa = tf.where(tf.greater(x, zero), kappa_Al_liquid * fL + kappa_Al_solid * (one - fL), kappa_grap)
        cp = tf.where(tf.greater(x, zero), cp_Al_liquid * fL + cp_Al_solid * (one - fL), cp_grap)
        cl = tf.where(tf.greater(x, zero), cl_Al_liquid * fL + cl_Al_solid * (one - fL), cl_grap)

        lap = tf.gradients(kappa * tem_x, x)[0]

        f_tem = (rho * cp * tem_t + rho * cl * fL_t - lap) / (rho_Al_solid * kappa_Al_solid)

        return f_tem
    
    def callback(self, loss):
        print('Loss:', loss)

    def train(self, nIter_pre, nIter):
        losses = []

        @tf.function
        def train_step_Adam_pre():
            with tf.GradientTape() as tape:
                loss_value = self.loss_pre
            gradients = tape.gradient(loss_value, self.trainable_variables)
            self.optimizer_Adam_pre.apply_gradients(zip(gradients, self.trainable_variables))
            return loss_value
        
        @tf.function
        def train_step_Adam():
            with tf.GradientTape() as tape:
                loss_value = self.loss
            gradients = tape.gradient(loss_value, self.trainable_variables)
            self.optimizer_Adam.apply_gradients(zip(gradients, self.trainable_variables))
            return loss_value
        
        def scipy_lbfgs_optimizer(loss, variables):
            def get_loss_and_grads():
                with tf.GradientTape() as tape:
                    loss_value = loss()
                gradients = tape.gradient(loss_value, variables)
                return loss_value, gradients
        
            tfp.optimizer.lbfgs_minimize(get_loss_and_grads, initial_position=variables)

        it = 0
        tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
               self.tem0_tf: self.tem0,
               self.tem_lb_tf: self.tem_lb, self.tem_ub_tf: self.tem_ub,
               self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
               self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
               self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
               self.X_tem_tf: self.X_tem}
        
        start_time = time.time()
        for it in range(nIter_pre):
            self.sess.run(self.train_op_Adam_pre, tf_dict)

            # Print
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss_pre, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                    (it, loss_value, elapsed))
                sys.stdout.flush()
                start_time = time.time()
                                                                                                        
        # if nIter_pre > 0:
        #     scipy_lbfgs_optimizer(lambda: self.loss_pre, tf_dict)
        threshold = 1
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            loss_value = self.sess.run(self.loss, tf_dict)
            losses.append(loss_value)
            if (loss_value <= threshold) :
                print("Training Completed with threshold = " + threshold)
                break
            # Print
            if it % 1000 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                    (it, loss_value, elapsed))
                sys.stdout.flush()
                start_time = time.time()
        np.savetxt('training_losse_layers=8_lr=0.001_hn=200_epochs=25000.txt', losses)                                                                                  
        # if nIter > 0:
        #     scipy_lbfgs_optimizer(lambda: self.loss, tf_dict)
        return losses


    def predict(self, X_star):
        X_star = tf.convert_to_tensor(X_star, dtype=tf.float64)
        tem_star = self.sess.run(self.net_uv(X_star[:, 0:1], X_star[:, 1:2]))
        ftem_star = self.sess.run(self.net_f_uv(X_star[:, 0:1], X_star[:, 1:2]))
        return tem_star, ftem_star
        
        return tem_star, ftem_star
    
    def save_model(self, path):
        save_path = self.saver.save(self.sess, path)
        print(f"Model saved in path: {save_path}")

    def load_model(self, path):
        self.saver.restore(self.sess, path)
        print(f"Model restored from path: {path}")

if __name__ == "__main__":
    noise = 0.0

    ltem = 298.15
    utem = 973.15
    eps = 0.02

    # Domain bounds
    lb = np.array([-0.4, 5.0])
    ub = np.array([0.4, 10.0])

    lbr = np.array([-0.05, 5.0])
    ubr = np.array([ 0.05, 10.0])

    N0 = 300
    N_b = 100
    N_f = 10000
    num_hidden = 8
    layers = [2] + num_hidden * [200] + [1]

    data = scipy.io.loadmat('project_pinn/thermal_fine.mat')

    x = data['x'].flatten()[:, None]
    t = data['tt'].flatten()[:, None]
    Exact = data['Tem']
    Exact_tem = np.real(Exact)

    ftem = interp2d(x, t, Exact.T)

    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x, :]
    tem0 = Exact_tem[idx_x, 0:1]

    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t, :]

    X_f = lb + (ub - lb) * lhs(2, N_f)

    X_f = X_f[np.argsort(X_f[:, 0])]
    X_tem = ftem(X_f[:, 0], 0).flatten()[:, None]

    model = SolidificationPINN(x0, tem0, tb, X_f, X_tem, layers, lb, ub)

    start_time = time.time()
    losses = model.train(-1, 50000)
    elapsed = time.time() - start_time
    
    print('Training time: %.4f' % (elapsed))

    tem_pred, ftem_pred = model.predict(X_star)
    np.savetxt('predict_xT.txt', X_star)
    np.savetxt('predict_tem.txt', tem_pred)
    np.savetxt('predict_ftem.txt', ftem_pred)

    # Plot training loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig('training_loss_curve.png')

    # model.save_model('model_checkpoint.ckpt')
    # model.load_model('model_checkpoint.ckpt')