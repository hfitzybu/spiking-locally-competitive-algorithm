import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import torch


piecesize = 32

dictionary_size = 3

coding_iter = 3000

bp_iter=100

tao = 0.1

softthresh = 1

import torchvision
from torchvision import datasets, transforms

from torchvision.transforms import ToTensor


def load_mnist(batch_size=1000):
    # # ,transforms.Normalize((0.5,), (0.5,)),
    # Min of input image = 0 -> 0 - 0.5 = -0.5 -> gets divided by 0.5 std -> -1
    # Max of input image = 255 -> toTensor -> 1 -> (1 - 0.5) / 0.5 -> 1
    # so it transforms your data in a ange[-1, 1]

    transform = transforms.Compose([transforms.ToTensor()
                                    # ,transforms.Normalize((0.5,), (0.5,)),
                                    ])

    # change download to True for first timw
    data_set = datasets.MNIST(root='data',
                              transform=transform,
                              download=False)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)

    #
    dataiter = iter(data_loader)
    images, labels = dataiter.next()

    # images, labels = next(iter(train_loader))
    print(images.shape)
    m = images.numpy()
    l = labels.numpy()

    return m, l


def small_thresh(M):
    thresh = 1e-13
    return np.where(M > thresh, M, 0)


def thresh(M):
    return np.where(M > softthresh, M, 0)


def normalize(M):
    # vector l2 norm = 1
    # sigma = np.inner(vector, vector)
    sigma = np.sum(M * M)

    return M / np.sqrt(sigma)


def simple_gamma(M, gamma=1.5):

    return np.where(M > 0, M ** gamma, 0)


class LCA:

    def __init__(self,
                 feature_size=piecesize,
                 dict_size=dictionary_size,
                 bp_iter=bp_iter,
                 rate_coding_iter=coding_iter,
                 max_mp=softthresh,
                 reset_mp=0,
                 lr=0.01,
                 lr_coding=0.01):
        self.feature_size = feature_size
        self.dict_size = dict_size
        self.bp_iter=bp_iter
        self.rate_coding_iter = rate_coding_iter

        # max membrane potential
        self.max_mp = max_mp
        self.reset_mp = reset_mp

        self.lr=lr
        self.lr_coding=lr_coding

        self.dictionary = None
        self.lateral_weight = None
        self.raw_image = None
        self.all_pieces = None

        self.input_feature = None

        # inference parameter

        # activation/ coefficient
        self.infer_a = None
        self.infer_u = None

        self.plot_count=0
        return

    def reconstruct(self):

        # coefficients
        co=self.infer_a.reshape([-1,1,1])
        recon=co*self.dictionary
        recon=np.sum(recon,axis=0)

        # print(recon.shape)

        cv.imshow("recon",np.hstack([self.input_feature,recon]))
        cv.waitKey(0)

        return recon

    def show_dictionary(self):

        cv.imshow('dic',self.dictionary.reshape([self.dict_size*self.feature_size,self.feature_size]))
        cv.waitKey(0)

        return

    def show_input(self):

        cv.imshow("input",self.input_feature)
        cv.waitKey(0)

        return

    def loss(self):
        # only consider the reconstruction loss

        s=self.input_feature.reshape(self.feature_size*self.feature_size,1)
        phi = self.dictionary.reshape(self.dict_size, -1).T
        a=self.infer_a


        residual=s-np.matmul(phi,a)
        from numpy import linalg as LA

        # l2 loss
        loss=LA.norm(residual,'fro')
        print("LOSS",loss)
        return loss

    def update_dict(self):

        # lr=0.01
        # N=784 M=10
        # s (N,1)
        # phi (N,M)
        # a, u, b (M,1)

        # (10, 28, 28) --> (782, 10)
        phi=self.dictionary.reshape(self.dict_size,self.feature_size*self.feature_size).T

        I=self.input_feature.reshape(self.feature_size*self.feature_size,1)

        dphi=np.matmul((I-np.matmul(phi,self.infer_a)),self.infer_a.T)

        ddict=dphi.T.reshape([self.dict_size,self.feature_size,self.feature_size])

        self.dictionary=self.dictionary+ddict*self.lr

        return

    def coding(self):

        # flatten
        dict = self.dictionary.reshape(self.dict_size, -1)
        input = self.input_feature.reshape(-1)

        # (0-1)
        # input = normalize(input)

        for i in range(self.dict_size):
            dict[i] = normalize(dict[i])

        S = input.T
        phi = dict.T

        I = np.identity(self.dict_size)

        # inhibition matrix/ lateral inhibition / lateral weight
        G = np.dot(phi.T, phi) - I

        # bug in numpy (maybe due to accuracy) diagonal should be 0
        G = small_thresh(G)
        # print(G)
        # G=simple_gamma(G,0.05)
        # print(G)

        bT = np.inner(phi.T, S.T)

        # driven forward input, how neurons response to this input feature(based on similarity/inner product)
        # this is a constant with respect to each input feature
        b = bT.T

        # (dict_size, )
        # print(b.shape)

        # soma current
        mu = np.zeros_like(b)

        # average soma current
        u = np.zeros_like(b)

        # spiking rate / encoded coefficients
        a = np.zeros_like(b)

        # membrane potential
        v = np.zeros_like(b)

        # print("--------")

        # control the intensity of soma current/ how fast the membrane potential integrates
        # lr = 0.002

        #
        v_all = np.zeros([0, self.dict_size])

        #
        mu_all = np.zeros([0, self.dict_size])

        # plot use, x axis
        x = np.linspace(0, self.rate_coding_iter, self.rate_coding_iter)

        # decay intensity, will only be activated when one neuron fires and decays exponentially through time
        exp_decay = np.zeros([self.dict_size, self.rate_coding_iter])

        # SPIKE NUM
        spike_num = np.zeros([self.dict_size, self.rate_coding_iter])


        # fire at t0 add an exponential decay sequence
        def update_decay(neuron_index, t0):

            t = -np.arange(self.rate_coding_iter - t0)

            y = np.exp(0.1 * t)

            zeros = np.zeros([t0])

            y = np.hstack([zeros, y])

            # add the effect to the original decay matrix
            exp_decay[neuron_index] = exp_decay[neuron_index] + y

            return

        for i in range(self.rate_coding_iter):

            decay = exp_decay.T[i] - np.sum(exp_decay.T[i])

            sigma_i_no_j = 1 * np.sum(G * decay, axis=0)

            delta_u = b - mu + sigma_i_no_j

            mu = mu + delta_u

            v = v + mu * self.lr_coding


            # check if any neuron reaches its threshold
            # index is neuron index
            for index in range(self.dict_size):
                if v[index] > softthresh:
                    # print("i",i,"  index",index)

                    update_decay(index, i)
                    # print(exp_decay.T[i])

                    v[index] = 0
                    spike_num[index, i] = spike_num[index, i - 1] + 1
                else:
                    spike_num[index, i] = spike_num[index, i - 1]

            v_all = np.vstack([v_all, v])
            mu_all = np.vstack([mu_all, mu])

        # a=spike_num[-1]

        a = spike_num.T[-1] / self.rate_coding_iter / self.lr_coding
        # a=a-self.max_mp
        # print("Spiking Rate", a)

        u = np.sum(mu_all, axis=0) / self.rate_coding_iter
        # print("Average current", u)

        self.infer_a = a.reshape(self.dict_size,1)
        self.infer_u = u.reshape(self.dict_size,1)


        # # plot membrane potential
        # self.plot_(v_all.T,ylabel="membrane potential")
        #
        # # plot decay
        # self.plot_((exp_decay), ylabel="decay")
        #
        # # plot spike num
        # self.plot_(spike_num, ylabel="spike num")
        #
        # # plot soma current
        # self.plot_(mu_all.T,ylabel="soma current")
        #
        # plt.show()

        return

    def inference(self):


        x = np.linspace(0, self.bp_iter, self.bp_iter)

        E = np.zeros([self.bp_iter])

        for i in range(self.bp_iter):

            print("iter ",i," : ")

            self.coding()

            E[i]=self.loss()

            self.update_dict()

        # plot loss through epoch
        # plt.plot(x, E, color='blue')
        # plt.show()

        # self.show_dictionary()
        self.reconstruct()
        # self.show_input()
        # showed=enlarge(self.dictionary.reshape([self.dict_size*self.feature_size,self.feature_size]),3)
        # cv.imshow('dasd',showed)
        #
        # cv.waitKey(0)

        return

    def plot_(self,y,ylabel="---"):

        self.plot_count+=1

        fig=plt.figure(ylabel)

        x = np.linspace(0, self.rate_coding_iter, self.rate_coding_iter)

        plt.plot(x, y[0], color='red')
        plt.plot(x, y[1], color='blue')
        plt.plot(x, y[2], color='green')
        plt.plot(x, y[4], color='orange')
        plt.plot(x, y[5], color='purple')
        plt.plot(x, y[6], color='brown')
        plt.plot(x, y[7], color='pink')
        plt.plot(x, y[8], color='gray')
        plt.plot(x, y[9], color='olive')

        # plt.set_ylabel=(ylabel)
        return


def test_LCA_MNIST():

    # parameters
    model = LCA(feature_size=28,
                dict_size=100,
                bp_iter=50,
                rate_coding_iter=2000,
                max_mp=1.0,
                reset_mp=0.0,
                lr=0.01,
                lr_coding=0.002)

    #  MNIST Dataset
    fn = 1100

    # class num
    class_num = 10
    #
    feature_size = 28

    model.all_pieces, labels = load_mnist(fn)  # .reshape([fn,28,28])

    model.all_pieces = model.all_pieces.reshape([fn, feature_size, feature_size])

    # dataset shape: (1100, 28, 28)
    print("dataset shape:", model.all_pieces.shape)
    # labels shape: (1100,)
    print("labels shape:", labels.shape)

    pieces_with_label = np.zeros([class_num, fn, feature_size, feature_size])

    # count sample num in each class
    count = np.zeros([class_num])

    for i in range(fn):
        pieces_with_label[labels[i], int(count[labels[i]])] = model.all_pieces[i]
        count[labels[i]] += 1

    # count:  [109. 132. 109. 100. 116.  99.  99. 130.  98. 108.]
    print("count: ", count)

    # 100 sampes in each class
    pieces_with_label = pieces_with_label[:, 0:100, :, :]

    # (10, 28, 28)
    model.dictionary = np.zeros([model.dict_size, model.feature_size, model.feature_size])
    # for i in range(model.dict_size):
    for i in range(model.dict_size):
        ww=np.mod(i,10)
        pp=np.remainder(i,10)
        # model.dictionary[i] = normalize(pieces_with_label[i, 0])
        model.dictionary[i] = normalize(pieces_with_label[ww, pp])

    # model.input_feature=model.dictionary[0]
    model.input_feature = pieces_with_label[0, 96]


    # if only do rate coding
    # model.coding()

    # if back-propagation and update dictionary
    # you may want to comment those plot codes at the end of model.coding()
    model.inference()

    # model.show_dictionary()
    # model.show_input()

    return




if __name__ == '__main__':

    test_LCA_MNIST()
