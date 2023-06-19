import pickle
import numpy as np
from Params import args

def reshape(data):
    row,column,time,cat = data.shape
    data_reshape = np.zeros((row*column, time, cat))
    for i in range(row):
        for j in range(column):
            data_reshape[i*column+j,:,:] = data[i,j,:,:]
    return data_reshape

class DataHandler:
    def __init__(self):
        if args.data == 'NYC':
            predir = 'Datasets/NYC_crime/'
        elif args.data == 'CHI':
            predir = 'Datasets/CHI_crime/'
        else:
            predir = None
        self.predir = predir
        with open(predir + 'trn.pkl', 'rb') as fs:
            trnT = pickle.load(fs)
        with open(predir + 'val.pkl', 'rb') as fs:
            valT = pickle.load(fs)
        with open(predir + 'tst.pkl', 'rb') as fs:
            tstT = pickle.load(fs)
        print("self.predir", self.predir)
        args.row, args.col, _, args.offNum = trnT.shape
        args.areaNum = args.row * args.col
        args.trnDays = trnT.shape[2]
        args.valDays = valT.shape[2]
        args.tstDays = tstT.shape[2]
        args.decay_step = args.trnDays//args.batch
        for i in range(5):
            for j in range(5):
                print(trnT[i,j,2,:])
                B = np.reshape(trnT, [args.areaNum, -1, args.offNum])
                print(B[i*5+j,2,:])
                print(trnT[i,j,2,:]==B[i*5+j,2,:])
        self.trnT = reshape(trnT)#np.reshape(trnT, [args.areaNum, -1, args.offNum])
        self.valT = reshape(valT)#np.reshape(valT, [args.areaNum, -1, args.offNum])
        self.tstT = reshape(tstT)#np.reshape(tstT, [args.areaNum, -1, args.offNum])
        self.mean = np.mean(trnT)
        self.std = np.std(trnT)
        self.mask1, self.mask2, self.mask3, self.mask4 = self.getSparsity()
        self.getTestAreas()
        print('Row:', args.row, ', Col:', args.col)
        print('Sparsity:', np.sum(trnT!=0) / np.reshape(trnT, [-1]).shape[0])
    
    def zScore(self, data):
        return (data - self.mean) / self.std

    def zInverse(self, data):
        return data * self.std + self.mean

    def getSparsity(self):
        data = self.tstT
        print(data.shape)
        day = data.shape[1]
        mask = 1 * (data > 0)
        p1 = np.zeros([data.shape[0], data.shape[2]])
        for cate in range(4):
            for region in range(mask.shape[0]):
                p1[region, cate] = np.sum(mask[region, :, cate], axis=0) / day
        mask1 = np.zeros_like(p1)
        mask2 = np.zeros_like(p1)
        mask3 = np.zeros_like(p1)
        mask4 = np.zeros_like(p1)
        for cate1 in range(4):
            for region1 in range(mask.shape[0]):
                if p1[region1, cate1] > 0 and p1[region1, cate1] <= 0.25:
                    mask1[region1, cate1] = 1
                elif p1[region1, cate1] > 0.25 and p1[region1, cate1] <= 0.5:
                    mask2[region1, cate1] = 1
                elif p1[region1, cate1] > 0.5 and p1[region1, cate1] <= 0.75:
                    mask3[region1, cate1] = 1
                elif p1[region1, cate1] > 0.75 and p1[region1, cate1] <= 1:
                    mask4[region1, cate1] = 1
        return mask1, mask2, mask3, mask4

    def getTestAreas(self):
        posTimes = np.sum(1 * (self.trnT!=0), axis=1)
        percent = posTimes / args.trnDays
        self.tstLocs = (percent > -1) * 1