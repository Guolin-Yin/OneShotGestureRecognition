from LearningModel.MODEL import *
from Preprocess.gestureDataLoader import  *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from LearningModel.MODEL import *
from tensorflow.keras.models import load_model
def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]
def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]
class DANNtrain( object ):
	def __init__( self, config ):
		self.config = config
		self.grl_lambd = 1.0
		self.checkpoint_dir = 'Saved_Model'
		self.advObj = AdversarialNetwork( config )
		self.build_DANN( )
		# self.adv_model = self.advObj.buildAdvModel( )
		self.loss = tf.keras.losses.categorical_crossentropy
		self.acc = tf.keras.metrics.categorical_accuracy

		self.train_loss = tf.keras.metrics.Mean( "train_loss", dtype = tf.float32 )
		self.train_act_cls_loss = tf.keras.metrics.Mean( "train_act_cls_loss", dtype = tf.float32 )
		self.train_domain_cls_loss = tf.keras.metrics.Mean( "train_domain_cls_loss", dtype = tf.float32 )
		self.train_act_cls_acc = tf.keras.metrics.Mean( "train_act_cls_acc", dtype = tf.float32 )
		self.train_domain_cls_acc = tf.keras.metrics.Mean( "train_domain_cls_acc", dtype = tf.float32 )

		self.val_loss = tf.keras.metrics.Mean( "val_loss", dtype = tf.float32 )
		self.val_act_cls_loss = tf.keras.metrics.Mean( "val_act_cls_loss", dtype = tf.float32 )
		self.val_domain_cls_loss = tf.keras.metrics.Mean( "val_domain_cls_loss", dtype = tf.float32 )
		self.val_act_cls_acc = tf.keras.metrics.Mean( "val_act_cls_acc", dtype = tf.float32 )
		self.val_domain_cls_acc = tf.keras.metrics.Mean( "val_domain_cls_acc", dtype = tf.float32 )

		self.optimizer = tf.keras.optimizers.Adamax( learning_rate = self.config.lr, beta_1 = 0.95, beta_2 = 0.99,
				epsilon = 1e-09, name = 'Adamax')
	def build_DANN( self ):
		# input = Input(shape = config.input_shape)
		#
		# # Build the DANN model
		# self.feature_extractor = self.advObj.buildFeatureExtractor( mode = 'Alexnet' )(input )
		#
		# '''gesture_classifier'''
		# gesture_classifier = Dense(
		# 		units = self.config.N_base_classes,
		# 		bias_regularizer = regularizers.l2( 4e-4 ), name = 'gesture_classifier'
		# 		)( feature_extractor )
		# gesture_classifier_out = Softmax( name = 'gesture_classifier_out' )( gesture_classifier )
		# '''domain_discriminator'''
		# domain_discriminator = Dense(
		# 		units = self.config.N_base_classes,
		# 		bias_regularizer = regularizers.l2( 4e-4 ), name = 'domain_discriminator'
		# 		)( feature_extractor )
		# domain_discriminator_out = Softmax( name = 'domain_discriminator_out' )( domain_discriminator )
		# adv_model = Model( inputs = input, outputs = [ domain_discriminator_out, gesture_classifier_out ] )
		self.dataInput = tf.keras.layers.Input( shape = self.config.input_shape )
		# Build the DANN model
		self.featureEncoder = self.advObj.buildFeatureExtractor( )
		self.actClsEncoder = self.advObj.buildSignClassifier( )
		self.domainClsEncoder = self.advObj.buildDomainClassifier( )
		self.grl = GradientReversalLayer( )
		# Share the feature
		feature = self.featureEncoder( self.dataInput )
		actCls = self.actClsEncoder( feature )
		domainCls = self.domainClsEncoder( self.grl(feature) )
		self.dannModel = tf.keras.models.Model( self.dataInput, [ actCls, domainCls ] )
		self.dannModel = tf.keras.models.Model( self.dataInput, actCls )
		self.dannModel.summary( )
	def train( self, train_source_datagen,
			# train_target_datagen,
			val_target_datagen, train_iter_num, val_iter_num,):
		"""
			   This is the training for the DANN model
			   :param train_source_datagen: source domain dataset generator
			   :param train_target_datagen: target domain dataset generator
			   :param val_target_datagen: validation dataset generator
			   :param train_iter_num: number of training time for each loop
			   :param val_iter_num: number of validation time for each loop
	   """
		print( '\n----------- start to train -----------\n' )
		best_val_loss = np.Inf
		best_val_act_cls_acc = 0.0

		self.out_train_loss = [ ]
		self.out_train_act_cls_loss = [ ]
		self.out_train_domain_cls_loss = [ ]
		self.out_train_act_cls_acc = [ ]
		self.out_train_domain_cls_acc = [ ]

		self.out_val_loss = [ ]
		self.out_val_act_cls_loss = [ ]
		self.out_val_domain_cls_loss = [ ]
		self.out_val_act_acc = [ ]
		self.out_val_domain_cls_acc = [ ]
		for ep in np.arange( 1, self.config.epoch + 1, 1 ):
			# training for one loop
			train_loss, train_act_cls_loss, train_domain_cls_loss, train_act_cls_acc, train_domain_cls_acc = self.fit_one_epoch(
					train_source_datagen,
					# train_target_datagen,
					train_iter_num,
					ep,
					)
			# train_loss, train_act_cls_loss, train_act_cls_acc = self.fit_one_epoch(
			# 		train_source_datagen,
			# 		train_target_datagen,
			# 		train_iter_num,
			# 		ep,
			# 		)
			self.out_train_loss.append( train_loss )
			self.out_train_act_cls_loss.append( train_act_cls_loss )
			self.out_train_domain_cls_loss.append( train_domain_cls_loss )
			self.out_train_act_cls_acc.append( train_act_cls_acc )
			self.out_train_domain_cls_acc.append( train_domain_cls_acc )
			# validation
			# val_loss, val_act_cls_loss, val_domain_cls_loss, val_act_cls_acc, val_domain_cls_acc = self.eval_one_epoch(
			# 		val_target_datagen,
			# 		val_iter_num,
			# 		ep,
			# 		)
			val_loss, val_act_cls_loss, val_act_cls_acc = self.eval_one_epoch(
					val_target_datagen,
					val_iter_num,
					ep,
					)

			self.out_val_loss.append( val_loss )
			self.out_val_act_cls_loss.append( val_act_cls_loss )
			# self.out_val_domain_cls_loss.append( val_domain_cls_loss )
			self.out_val_act_acc.append( val_act_cls_acc )
			# self.out_val_domain_cls_acc.append( val_domain_cls_acc )

			self.train_loss.reset_states( )
			self.train_act_cls_loss.reset_states( )
			self.train_domain_cls_loss.reset_states( )
			self.train_act_cls_acc.reset_states( )
			self.train_domain_cls_acc.reset_states( )

			self.val_loss.reset_states( )
			self.val_act_cls_loss.reset_states( )
			self.val_domain_cls_loss.reset_states( )
			self.val_act_cls_acc.reset_states( )
			self.val_domain_cls_acc.reset_states( )
			str = "Epoch {:03d}-train_loss-{:.3f}-val_loss-{:.3f}-train_act_cls_acc-{:.3f}-val_act_cls_acc-{:.3f}" \
				.format( ep, train_loss, val_loss, train_act_cls_acc, val_act_cls_acc )
			print( str )

			# val_loss, val_act_cls_loss, val_domain_cls_loss, val_act_cls_acc, val_domain_cls_acc = self.eval_one_epoch(
			# 		test_gen,
			# 		test_iter_num,
			# 		ep,
			# 		)
			# print( f'\ntest Accuracy is {val_domain_cls_acc:0.3f}\n' )
			# self.val_loss.reset_states( )
			# self.val_act_cls_loss.reset_states( )
			# self.val_domain_cls_loss.reset_states( )
			# self.val_act_cls_acc.reset_states( )
			# self.val_domain_cls_acc.reset_states( )
			if val_act_cls_loss <= 1:
				break
			if val_loss <= best_val_loss or val_act_cls_acc >= best_val_act_cls_acc:
				self.dannModel.save( os.path.join( self.checkpoint_dir, str + ".h5" ) )
				if val_loss <= best_val_loss:
					best_val_loss = val_loss
				if val_act_cls_acc >= best_val_act_cls_acc:
					best_val_act_cls_acc = val_act_cls_acc
		self.dannModel.save( os.path.join( self.checkpoint_dir, "trained_dann.h5" ) )
		print( '\n----------- end to train -----------\n' )

		trainResults = {
				"train_loss"                : self.out_train_loss,
				"Act_classification_loss"   : self.out_train_act_cls_loss,
				"Domain_classification_loss": self.out_train_domain_cls_loss,
				"Act_classification_acc"    : self.out_train_act_cls_acc,
				"Domain_classification_acc" : self.out_train_domain_cls_acc
				}
		valResults = {
				"train_loss"                : self.out_val_loss,
				"Act_classification_loss"   : self.out_val_act_cls_loss,
				# "Domain_classification_loss": self.out_val_domain_cls_loss,
				"Act_classification_acc"    : self.out_val_act_acc,
				# "Domain_classification_acc" : self.out_val_domain_cls_acc
				}
		return trainResults, valResults
	def fit_one_epoch( self, train_source_datagen,
			# train_target_datagen,
			train_iter_num, ep ):
		'''
		Training for every Epoch
		train_source_datagen:   source dataset generator
		train_target_datagen:   target dataset generator
		train_iter_num:         number of iteration for one training step
		ep:                     Current epoch
		'''
		progbar = tf.keras.utils.Progbar( train_iter_num )
		print( f'Epoch{ep}/{self.config.epoch}' )
		for i in np.arange( 1, train_iter_num + 1 ):
			batch_act_source_data, batch_act_source_labels, batch_domain_label = train_source_datagen.__next__( )  #
			# train_source_datagen.next_batch()
			# batch_act_target_data, batch_act_target_labels = train_target_datagen.__next__( )  # train_target_datagen.next_batch()
			# batch_domain_label = np.vstack(
			# 		[
			# 				np.tile( [ 1, 0 ], [ len( batch_act_source_labels ), 1 ] ),
			# 				np.tile( [ 0, 1 ], [ len( batch_act_target_labels ), 1 ] )
			# 				]
			# 		)
			# batch_domain_train_data = np.concatenate(
			# 		[ batch_act_source_data, batch_act_target_data ],
			# 		axis = 0
			# 		)
			# batch_domain_cls_label = np.concatenate([batch_act_source_labels,batch_act_target_labels],axis = 0 )
			# update and visualise the trainig
			iter = (ep - 1) * train_iter_num + i
			process = iter * 1.0 / (self.config.epoch * train_iter_num)
			self.grl_lambd = grl_lambda_schedule( process )
			learning_rate = learning_rate_schedule( ep, init_learning_rate = self.config.lr )

			tf.keras.backend.set_value( self.optimizer.lr, learning_rate )
			with tf.GradientTape( ) as tape:
				# calculate the activity classification's output loss and accuracy
				act_cls_feature = self.featureEncoder( batch_act_source_data, training = True)
				act_cls_pred = self.actClsEncoder( act_cls_feature, training = True )
				act_cls_loss = self.loss( batch_act_source_labels, act_cls_pred )
				act_cls_acc = self.acc( batch_act_source_labels, act_cls_pred )

				# calculate the output, loss and accuracy of the domain classifer
				domain_cls_feature = self.featureEncoder( batch_act_source_data )
				domain_cls_pred = self.domainClsEncoder(
						self.grl(domain_cls_feature,self.grl_lambd),
						training = True,
						)
				domain_cls_loss = self.loss( batch_domain_label, domain_cls_pred )
				domain_cls_acc = self.acc( batch_domain_label, domain_cls_pred )

				loss = tf.reduce_mean( act_cls_loss ) + tf.reduce_mean( domain_cls_loss )
			# Optimasation process
			vars = tape.watched_variables( )
			grads = tape.gradient( loss, vars )
			self.optimizer.apply_gradients( zip( grads, vars ) )

			# calculate the average loss and accuracy
			self.train_loss( loss )
			self.train_act_cls_loss( act_cls_loss )
			self.train_domain_cls_loss( domain_cls_loss )
			self.train_act_cls_acc( act_cls_acc )
			self.train_domain_cls_acc( domain_cls_acc )
			progbar.update(
					i,
					[ ('loss', loss),
					  ('Act_cls_loss', act_cls_loss),
					  ('domain_cls_loss', domain_cls_loss),
					  ('Act_cls_acc', act_cls_acc),
					  ('domain_cls_acc', domain_cls_acc)
					  ]
					)
			'''
				return:
						Total training loss
						Domain classification loss
			'''
		print( f'\ncurrent learning rate:{learning_rate}' )
		return self.train_loss.result( ), self.train_act_cls_loss.result( ), self.train_domain_cls_loss.result( ), self.train_act_cls_acc.result( ), self.train_domain_cls_acc.result( )
		# return self.train_loss.result( ), self.train_act_cls_loss.result( ), self.train_act_cls_acc.result( )
	def eval_one_epoch( self, val_target_data_gen, val_iter_num, ep ):
		for i in np.arange( 1, val_iter_num + 1 ):
			batch_act_target_data, batch_act_target_labels = val_target_data_gen.__next__( )
			# batch_target_domain_labels = np.tile( [ 0, 1 ], [ len( batch_act_target_labels ), 1 ] ).astype( np.float32 )
			# compute the target domain activity classification and domain classification output
			target_act_feature = self.featureEncoder( batch_act_target_data )
			target_act_cls_pred = self.actClsEncoder( target_act_feature, training = False )
			# target_domain_cls_pred = self.domainClsEncoder( target_act_feature, training = False )
			# compute the target domain prediction losses
			target_act_cls_loss = self.loss( batch_act_target_labels, target_act_cls_pred )
			# target_domain_cls_loss = self.loss( batch_target_domain_labels, target_domain_cls_pred )
			target_loss = tf.reduce_mean( target_act_cls_loss ) #+ tf.reduce_mean( target_domain_cls_loss )
			# Target domain classification accuracy
			act_cls_acc = self.acc( batch_act_target_labels, target_act_cls_pred )
			# domain_cls_acc = self.acc( batch_target_domain_labels, target_domain_cls_pred )
			# update the validation loss and accuracy
			self.val_loss( target_loss )
			self.val_act_cls_loss( target_act_cls_loss )
			# self.val_domain_cls_loss( target_domain_cls_loss )
			self.val_act_cls_acc( act_cls_acc )
			# self.val_domain_cls_acc( domain_cls_acc )
			'''
			return: 
					Total loss
					Activity classification accuracy

			'''
		# return self.val_loss.result( ), self.val_act_cls_loss.result( ), self.val_domain_cls_loss.result( ), self.val_act_cls_acc.result( ), self.val_domain_cls_acc.result( )
		#
		return self.val_loss.result( ), self.val_act_cls_loss.result( ), self.val_act_cls_acc.result( )

def getAdvData( dataLoadObj,isTraining=True):
	source = ['lab','home']
	domain_data = []
	domain_label = []
	sign_data = []
	sign_label = []
	for i in source:
		train_data, train_labels, test_data, test_labels = dataLoadObj.getFormatedData(
				source = i,
				isZscore = False
				)
		# length = len(train_labels) #+ len(test_labels)
		if isTraining:
			if i == 'lab':
				sign_data.append(train_data)
				sign_label.append(train_labels)
				# domain_data.append( train_data )
				domain_label.append(np.zeros((len(train_labels),1)))
			elif i == 'home':
				test_data = test_data[0:76]
				sign_data.append(test_data)
				sign_label.append(test_labels)
				# domain_data.append( test_data )
				domain_label.append(np.ones((len(test_labels),1)))
		else:
			if i == 'home':
				test_data = test_data[ 76:len(test_data) ]
				test_labels = test_labels[ 76:len( test_labels ) ]
				sign_data.append( test_data )
				sign_label.append( test_labels )
				# domain_data.append( test_data )
				domain_label.append( np.ones( (len( test_labels ), 1) ) )
	sign_data = np.concatenate(sign_data,axis=0)
	sign_label = to_categorical(np.concatenate(sign_label,axis=0) - 1,num_classes = 276)
	# domain_data = np.concatenate(domain_data,axis=0)
	domain_label = to_categorical(np.concatenate(domain_label,axis=0),num_classes=2)
	train_set = {
			'sign_data':sign_data,
			'sign_label':sign_label,
			# 'domain_data': domain_data,
			'domain_label': domain_label,
			}
	return train_set
def getAdvDomainData(dataLoadObj,domain:str ):
	def getSelectedData(dataNlabel,selected_sign):
		data,label = dataNlabel
		idx = np.where( label == selected_sign )[ 0 ]
		data = data[ idx ]
		label = label[ idx ]
		return data, label
	def getData(domain):
		train_data, train_labels, test_data, test_labels = dataLoadObj.getFormatedData(
				source = domain,
				isZscore = False
				)
		data = np.concatenate( (train_data, test_data), axis = 0 )
		label = np.concatenate( (train_labels, test_labels), axis = 0 )
		return [data, label]
	if domain == 'lab':
		selected_sign = np.arange( 1, 277 )
		data, label = getData(domain)
		data, label = getSelectedData( [ data, label ], selected_sign )
		data_2, label_2 = getData( [1,2,3,4,5] )
		domain_label = np.vstack(
					[
							np.tile( 0, [ len( data ), 1 ] ),
							np.tile( 1, [ 1500, 1 ] ),
							np.tile( 2, [ 1500, 1 ] ),
							np.tile( 3, [ 1500, 1 ] ),
							np.tile( 4, [ 1500, 1 ] ),
							np.tile( 5, [ 1500, 1 ] )
							]
					)
		train_data = np.concatenate([data,data_2[np.where(label_2 <= 276)[0]]])
		train_labels = np.concatenate([label,label_2[np.where(label_2 <= 276)[0]]])
		train_data,train_labels = shuffle_aligned_list([train_data,train_labels])
		return [train_data, to_categorical(train_labels - 1,num_classes=config.N_base_classes),to_categorical(
				domain_label,num_classes=6)]
	elif domain == 'home':
		data, label = getData( domain )
		selected_sign = np.random.choice( np.arange( 1, 277 ),276,replace = False )
		data, label = getSelectedData( [ data, label ], selected_sign )
		return [data,to_categorical( label - 1, num_classes = config.N_base_classes )]
		# num = int( 0.1 * len( data ) )
		# train_data = data[ 0:num ]
		# train_label = label[ 0:num ]
		# train_data, train_label = shuffle_aligned_list( [ train_data, train_label ] )
		# val_data = data[ num:len( data ) ]
		# val_label = label[ num:len( label ) ]
		# val_data, val_label = shuffle_aligned_list( [ val_data, val_label ] )
		# return [train_data,to_categorical(train_label - 1,num_classes= config.N_base_classes)],[ val_data, to_categorical( val_label - 1, num_classes = config.N_base_classes ) ]
def train_adv():
	config = getConfig( )
	config.lr = 1e-4
	config.N_novel_classes = 76
	config.train_dir = 'D:\Matlab\SignFi\Dataset'
	config.N_base_classes = 200
	dataLoadObj = signDataLoader( config = config )
	train_set = getAdvData( dataLoadObj )
	test_set = getAdvData(dataLoadObj,isTraining=False)
	advObj = AdversarialNetwork( config )
	adv_net = advObj.buildAdvModel( )

	optimizer =tf.keras.optimizers.Adamax(
                    learning_rate = config.lr, beta_1 = 0.95, beta_2 = 0.99, epsilon = 1e-09,
                    name = 'Adamax'
                    )
	# optimizer = tf.keras.optimizers.SGD(
	# 		        learning_rate =config.lr,
	# 		        momentum = 0.99
	# 		        )
	adv_net.compile(loss = ['categorical_crossentropy','categorical_crossentropy'], loss_weights = [1,1], optimizer =optimizer
	,metrics = 'acc' )
	early_stop = EarlyStopping( 'val_gesture_classifier_out_loss', min_delta = 0, patience = 30,restore_best_weights
        =True )
	reduce_lr = ReduceLROnPlateau( 'val_loss', min_delta = 0, factor = 0.2, patience = 10, verbose = 1 )
	idx = np.random.permutation( len( train_set['sign_data'] ) )
	adv_net.fit(
			train_set['sign_data'][idx],

					# train_set['domain_label'][idx],
					train_set['sign_label'][idx]
					,
			epochs = 400,
			shuffle = True,
			# validation_data = (test_set['sign_data'],[test_set['domain_label'],test_set['sign_label']]),
			validation_data = (test_set[ 'sign_data' ], test_set[ 'sign_label' ] ),
			callbacks = [ early_stop,  reduce_lr]
			)
	adv_net.save( 'adversarial_network.h5' )
	return adv_net
def test_adv():
	config = getConfig( )
	config.N_novel_classes = 76
	config.train_dir = 'D:\Matlab\SignFi\Dataset'
	config.N_base_classes = 200
	adv_net = load_model(
			'adversarial_network.h5', compile = True,
			custom_objects = { 'GradientReversalLayer': GradientReversalLayer }
			)
	dataLoadObj = signDataLoader( config = config )
	test_set = getAdvData(dataLoadObj,isTraining=False)
	# adv_net.evaluate(train_set['sign_data'],[train_set['domain_label'],train_set['sign_label']])
	domain_prediction,cls_prediction = adv_net.predict( test_set['sign_data'])
	# domain_prediction = domain_prediction.argmax(axis=-1)
	# true_domain_label = test_set[ 'domain_label' ].argmax( axis = -1 )

	cls_prediction = cls_prediction.argmax(axis=-1)
	true_sign_label = test_set['sign_label'].argmax(axis=-1)

	N_correct = np.sum((true_sign_label == cls_prediction))
	print(f'Sign language prediction accuracy is {(N_correct/len(true_sign_label))*100: 0.2f}% ')
if __name__ == '__main__':
	# test_adv()
	config = getConfig( )
	config.lr = 1e-4
	# config.N_novel_classes = 76
	config.train_dir = 'D:\Matlab\SignFi\Dataset'
	config.N_base_classes = 276
	dataLoadObj = signDataLoader( config = config )
	source_domain = getAdvDomainData( dataLoadObj, domain = 'lab' )
	val_domain = getAdvDomainData( dataLoadObj, domain = 'home' )
	# val_target_domain = getAdvDomainData( dataLoadObj, domain ='home',iftest=True)
	# test_set = getAdvData( dataLoadObj, isTraining = False )
	train_source_datagen = batch_generator(source_domain,config.batch_size//2)
	# train_target_datagen = batch_generator( target_domain, config.batch_size//2 )
	val_target_datagen = batch_generator( val_domain, config.batch_size )
	# test_gen = batch_generator(test_domain,config.batch_size)
	# Training the model
	train_source_batch_num = int( len( source_domain[0] ) // (config.batch_size )//2 )
	# train_target_batch_num = int( len( target_domain[0] ) // (config.batch_size)//2 )
	train_iter_num = int( np.max( [ train_source_batch_num, train_source_batch_num ] ) )
	val_iter_num = int( len( val_domain[0] ) // config.batch_size )
	# test_iter_num = int( len( test_domain[ 0 ] ) // config.batch_size )
	dann = DANNtrain( config )
	trainResults, valResults = dann.train(
											train_source_datagen,
											# train_target_datagen,
											val_target_datagen,
											train_iter_num,
											val_iter_num,
			# 								test_gen,
			# test_iter_num
											)

