import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Preprocess.gestureDataLoader import *
from Config import *
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import pandas as pd
def visualize_scatter_domain( data_2d, label_ids, perplexity,n_iter,figsize = (10, 10) ):
	plt.figure( figsize = figsize )
	# plt.grid( )
	# nb_classes = len( np.unique( label_ids ) )
	# id_to_label_dict = ['Push&Pull','Sweep','Clap','Slide','Draw-Zigzag(Vertical)','Draw-N(Vertical)']
	# id_to_label_dict = [f'domain_{i + 1}' for i in range(nb_classes)]
	# marker = ['o','v','^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
	# for label_id in np.unique( label_ids ):
		# plt.scatter(
		# 		data_2d[ np.where( label_ids == label_id ), 0 ],
		# 		data_2d[ np.where( label_ids == label_id ), 1 ],
		# 		marker = marker[label_id],
		# 		color = plt.cm.Set1( label_id / float( nb_classes + 1 ) ),
		# 		linewidth = 1,
		# 		alpha = 0.8,
		# 		label = id_to_label_dict[ label_id ]
		# 		)
	domain_labels = []
	for i in range(len(label_ids)):
		domain_labels.append(f'domain_{label_ids[i]+1}')
	tsne_df = pd.DataFrame(
			{
					't-SNE_1'    : data_2d[ :, 0 ],
					't-SNE_2'    : data_2d[ :, 1 ],
					'labels': domain_labels
					}
			)
	sns.scatterplot(
			x = "t-SNE_1", y = "t-SNE_2",
			hue = "labels",
			style = 'labels',
			data = tsne_df
			)
	plt.legend( loc = 'best' )
	plt.title(f'perplexity is {perplexity}, number of iterations is {n_iter}', fontsize=18)
	plt.xlabel("t-SNE_1", fontsize=15)
	plt.ylabel( "t-SNE_2", fontsize = 15 )
def visualize_scatter_classes( data, label_id, label, perplexity,n_iter,figsize = (12,10) ,domain = None,):
	plt.figure( figsize = figsize )
	plt.tick_params( axis = 'x', label1On = False )
	plt.tick_params( axis = 'y', label1On = False )
	# plt.grid( )
	# nb_classes = len( np.unique( label_ids ) )
	# id_to_label_dict = ['Push&Pull','Sweep','Clap','Slide','Draw-Zigzag(Vertical)','Draw-N(Vertical)']
	# id_to_label_dict = list(str(np.unique( label_ids )))
	marker = ['X', '*', 'd', 'h', 'H', 'D', 'P', 'o','v','^', '<', '>', '8', 's', 'p', '*',  ]
	color = np.random.choice([1,2,3],2,replace = False )
	domains = ['lab','home']
	for i in range(len(data)):
		data_2d = data[i]
		label_ids = label_id[ i ]
		domain = domains[i]
		for idx, id in enumerate(np.unique( label_ids ) ):
			p = np.where( label_ids == id )[0]
			if id == 7:
				o = 2
			elif id == 28:
				o = 3
			elif id == 1:
				o = 1
			plt.scatter(
					data_2d[ p, 0 ],
					data_2d[ p, 1 ],
					marker = marker[idx],
					s = 500,
					# color = plt.cm.Set1( id / float( nb_classes + 1 ) ),
					color = plt.cm.Set1( color[i] ),
					linewidth = 1,
					alpha = 0.7,
					label = 'sign' + str(o) + f'({domain})'
					)
	plt.legend( fontsize = 22,loc = 'best' )
def domain_t_sne(data,n_components:int = 2,random_state = 0,perplexity:int = 6,n_iter:int = 5000):
	# data = data.reshape(len(data),-1)
	n = len(data)
	domain_label = []
	data_con = []
	for i in range(n):
		buff = data[ i ].reshape( len( data[ i ] ), -1 )
		data_con.append( buff )
		domain_label.append( [i for _ in range( len( buff ) )] )
	data_con = np.concatenate(data_con,axis = 0)
	domain_label = np.concatenate(domain_label,axis = 0)
	model = TSNE(n_components=n_components, random_state=random_state,perplexity = perplexity,n_iter=n_iter)
	tsne_data = model.fit_transform(data_con)
	visualize_scatter_domain( data_2d = tsne_data, label_ids = domain_label, perplexity = perplexity, n_iter = n_iter )
def class_t_sne(data,label_id,label,n_components:int = 2,random_state = 0,perplexity:int = 6,n_iter:int = 5000):
	data = [d.reshape( len( d ), -1 ) for d in data]
	model = TSNE(n_components=n_components, random_state=random_state,perplexity = perplexity,n_iter=n_iter)
	tsne_data = [model.fit_transform(t_d) for t_d in data]
	visualize_scatter_classes( data = tsne_data,label_id = label_id, label = label, perplexity = perplexity, n_iter = n_iter,
			domain = 'lab' )
if __name__ == '__main__':

	config = getConfig( )
	# config.domain_selection = (2, 2, 1)
	# config.train_dir = 'E:/Sensing_project/Cross_dataset/20181109/User1'
	# WidarDataLoaderObjMulti = WidarDataloader(
	# 		 isMultiDomain = False,
	# 		config = config
	# 		)
	# data = WidarDataLoaderObjMulti.getSQDataForTest(
	# 		nshots = 1, mode = 'fix',
	# 		isTest = False, Best = None
	# 		)
	# data_val_1 = data[ 'Val_data' ]
	# label_val_1 = data['Val_label']
	# domain_label_1 = [0 for i in range(len(data_val_1))]
	# config.domain_selection = (2, 2, 3)
	# config.train_dir = 'E:/Cross_dataset/20181109/User1'
	# WidarDataLoaderObjMulti = WidarDataloader(
	# 		dataDir = config.train_dir, selection = config.domain_selection, isMultiDomain = False,
	# 		config = config
	# 		)
	# data = WidarDataLoaderObjMulti.getSQDataForTest(
	# 		nshots = 1, mode = 'fix',
	# 		isTest = False, Best = None
	# 		)
	# data_val_2 = data[ 'Val_data' ].reshape( 114, -1 )
	# label_val_2 = data['Val_label']
	# domain_label_2 = [1 for i in range(len(data_val_1))]

	# data_val = np.concatenate((data_val_1,data_val_2),axis = 0)
	# domain_label = np.concatenate((domain_label_1,domain_label_2),axis = 0)
	# for j in [2,10,20,30,40,50,60,70,80,90,100,110]:
	# for j in [500,1000,2000,3000,4000,5000]:
	# 	t_sne(data_val,domain_label,perplexity = 10,n_iter = 2000)
	config.train_dir = 'D:\Matlab\SignFi\Dataset'
	config.N_base_classes = 276
	signDataObj = signDataLoader( config = config )

	# signFiData_lab = a[ 0 ][ 0:1500 ]
	# domain_label_3 = [ 1 for i in range( len( signFiData_lab ) ) ]
	# data_val = np.concatenate( (data_val_1, signFiData_lab), axis = 0 )
	# domain_label = np.concatenate( (domain_label_1, domain_label_3), axis = 0 )
	# domain_t_sne( (data_val_1, signFiData_lab), perplexity = 20, n_iter = 2000 )
	selection = [1,7,28]
	signFidata = signDataObj.getFormatedData( source = 'lab',isZscore = True )
	idx = np.where(signFidata[1] == selection)[0]
	data_lab = signFidata[0][idx]
	label_lab = signFidata[1][idx].tolist()


	signFidata = signDataObj.getFormatedData( source = 'home',isZscore = True  )
	idx = np.where(signFidata[1] == selection)[0]
	data_home = signFidata[0][idx]
	label_home = signFidata[1][idx].tolist()


	data = [data_lab,data_home]
	label_id = [label_lab,label_home]
	label = [label_lab_str,label_home_str]
	class_t_sne(data = data,label_id=label_id, label=label,perplexity = 7,n_iter = 2000)