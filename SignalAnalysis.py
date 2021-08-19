from Config import *
from Preprocess.gestureDataLoader import *
import numpy as np
from numpy.fft import fft, fftfreq, ifft
from numpy import real, imag, sqrt, absolute
import matplotlib.pyplot as plt
import math
def toCIR(CFR):
	index = np.asarray([-28,-26,-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,-1,1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21,
						23, 25, 27, 28]) + 32
	shape = (CFR.shape[0],CFR.shape[1],64,CFR.shape[3])
	CFR_expanded = np.zeros(shape, dtype= complex)
	CFR_expanded[:,:,index,:] = CFR
	cir  = np.zeros(CFR_expanded.shape, dtype= complex)
	for i in range(len(CFR)):
		for j in range(CFR.shape[3]):
			cir[i,:,:,j] = ifft(CFR_expanded[i,:,:,j],axis = 1,n=64)
	return cir
def computeK(signal_pow):
	signal_pow_merged = np.expand_dims(np.mean(np.sum( signal_pow, axis = 1 ),axis = 1),axis = 1)
	sigma_sq = np.expand_dims(np.var(signal_pow_merged,axis = 0),axis=0)
	avg_signal_pow = np.mean( signal_pow_merged, axis = 0 )
	P_sq = (avg_signal_pow) ** 2
	gamma = sigma_sq/P_sq
	K = np.sqrt(1 - gamma)/(1 - np.sqrt(1 - gamma))
	return 10*np.log10(K)
if __name__ == '__main__':
	config = getConfig( )
	config.nshots = 5
	# config.train_dir = 'E:/Cross_dataset/20181109/User1'
	config.train_dir = 'E:/test/20181109/User1'
	config.num_finetune_classes = 6
	config.lr = 1e-4
	domain_k = [ ]
	Rx = 6
	for x in range(1,6):
		domain = [ (x,i,Rx) for i in range(1,6)]
		K = [ ]
		for d in domain:
			config.domain_selection = d
			'''location,orientation,receiver'''
			WidarDataLoaderObj = WidarDataloader(dataDir = config.train_dir,selection = config.domain_selection,
						config = config)
			selected_gesture_samples_path = WidarDataLoaderObj.selectPositions(config.domain_selection)
			gesture_Rx3,x_all,_ = WidarDataLoaderObj._mapClassToDataNLabels(selected_gesture_samples_path,ampOnly=True)
			cir = toCIR(x_all)
			signal_pow = np.real( (cir * np.conjugate( cir )) )
			for i in range(len(cir)):
				a = np.mean(computeK( signal_pow[i] ))
				if math.isnan(a):
					print('NaN')
					continue
				K.append( a )
		domain_k.append(np.mean(K))
	print(f'Avgerage K factor for receiver-{Rx} is {np.mean(domain_k)}')