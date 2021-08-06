from modelPreTraining import *
from Preprocess.gestureDataLoader import signDataLoder
from tensorflow.keras.models import Model
from Config import getConfig
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

# from methodTesting.plotResults import pltResults

config = getConfig()
def plot_barchart():
    N = 2
    id = np.arange(N)
    width = 0.35
    oneshot_accuracy = [ 70.0, 88.6 ]
    fiveshot_accuracy = [ 84.4, 98.0 ]
    plt.bar( id, oneshot_accuracy, width, label = 'one shot results' )
    plt.bar( id+width, fiveshot_accuracy, width, label = 'five shot results' )
    plt.ylabel( "Accuracy" )
    plt.title( "Train model on user 1 to 4, Test model on user 5" )
    plt.legend()
    plt.xticks( id + width / 2, ('Without fine tuning', 'With fine tuning') )
    for index, data in enumerate( oneshot_accuracy ):
        plt.text( x = index-0.1, y = data + 1, s = f"{data}%", fontdict = dict( fontsize = 10 ) )

    for index, data in enumerate( fiveshot_accuracy ):
        plt.text( x = index+0.3, y = data + 1, s = f"{data}%", fontdict = dict( fontsize = 10 ) )
    plt.show()
def pltResults(acc):
    ways = np.arange( 2, 27, 1 )
    ways2 = np.arange( 2, 26, 1 )
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot( ways2, acc[ 0 ], linestyle='dashed', marker='o', label='Train on user 1 to 4, test on user 5' )
    ax.plot( ways2, acc[ 1 ], linestyle='dashed', marker='o', label='Train on user 1     , test on user 5' )
    ax.plot( ways2, acc[ 2 ],linestyle='dashed', marker='o', label='Train on user 1 to 4, test on user 5 using softmax' )
    ax.plot( ways2, acc[ 3 ], linestyle='dashed', marker='o',
             label='Train on user 1 to 4, test on user 5 using softmax with fixed support set' )
    # ax.plot( ways, acc[ 1 ], linestyle='dashed', marker='o', label='test on different environment same user' )
    # ax.plot( ways, acc[ 2 ], linestyle='dashed', marker='o', label='test on different environment same user (New sign)' )
    # ax.plot( ways, acc[ 3 ], linestyle='dashed', marker='o', label='test on different environment different user ' )
    # ax.plot( ways2, acc[ 4 ], linestyle='dashed', marker='o', label='test on different environment different user (New sign)' )
    ax.set_ylim( 0, 102 )
    ax.set_title( 'Feature extractor trained on lab environment with 125 classes' )
    ax.set_xlabel( 'Number of new classes' )
    ax.set_ylabel( 'Accuracy' )
    ax.legend( )
def record():
    test_on_lab = [ 100.0, 100.0, 100.0, 100.0, 99.90, 100.0,100.0,100.0,99.9,100.0,99.8,100.0,100.0,99.9,100.0,99.7,99.6,99.8,99.6,100.0,99.9,99.9,99.9,100.0,100.0 ]
    test_on_home = [ 99.2,
                     98.8,
                     97.7,
                     96.39999999999999,
                     96.39999999999999,
                     95.8,
                     96.0,
                     95.0,
                     92.9,
                     93.7,
                     92.9,
                     92.4,
                     92.2,
                     90.9,
                     91.9,
                     91.8,
                     88.7,
                     89.0,
                     87.9,
                     87.0,
                     89.1,
                     87.9,
                     88.9,
                     87.6,
                     87.0 ]
    test_on_user1_trained_sign = [ 84.6,
                                   80.30000000000001,
                                   82.69999999999999,
                                   81.6,
                                   79.5,
                                   79.9,
                                   80.30000000000001,
                                   80.4,
                                   78.4,
                                   82.69999999999999,
                                   80.30000000000001,
                                   80.10000000000001,
                                   78.0,
                                   79.4,
                                   80.4,
                                   78.5,
                                   77.5,
                                   79.80000000000001,
                                   77.8,
                                   77.3,
                                   78.7,
                                   79.10000000000001,
                                   78.60000000000001,
                                   78.8,
                                   80.0 ]
    test_on_user1_unseen_sign = [ 64.9, 47, 37.9, 34, 29.5, 27.4, 25.1, 22.5, 19.1, 18.4, 17.1, 15.3, 14.3, 14.7, 13.8,
                                  14.5, 15.2, 11.5,
                                  12.4, 10.3, 8.5, 10, 11.6, 7.7 ]
    test_on_user5_trained_sign = [ 99.3,
                                   99.5,
                                   99.2,
                                   98.2,
                                   98.7,
                                   97.89999999999999,
                                   98.0,
                                   98.5,
                                   98.0,
                                   97.89999999999999,
                                   97.2,
                                   97.1,
                                   97.0,
                                   97.1,
                                   97.5,
                                   97.2,
                                   95.6,
                                   96.39999999999999,
                                   94.69999999999999,
                                   96.8,
                                   96.7,
                                   97.2,
                                   95.3,
                                   96.0,
                                   96.89999999999999 ]
    train_on_user1to4_test_on5 = [93.60000000000001,
 88.4,
 84.89999999999999,
 83.3,
 82.5,
 80.60000000000001,
 77.3,
 74.5,
 77.10000000000001,
 73.0,
 76.9,
 75.2,
 72.8,
 73.5,
 73.4,
 71.5,
 75.5,
 71.5,
 71.1,
 70.3,
 73.4,
 72.6,
 70.19999999999999,
 69.89999999999999]
    train_on_user1to4_test_on5_softmax = [ 93.89999999999999,
      89.0,
      86.1,
      83.0,
      81.0,
      80.2,
      79.0,
      75.1,
      77.60000000000001,
      77.2,
      73.6,
      71.6,
      75.6,
      76.4,
      73.4,
      72.6,
      72.2,
      73.4,
      70.3,
      70.3,
      73.0,
      72.8,
      70.39999999999999,
      69.89999999999999 ]
    train_on_user1to4_test_on5_softmax_with_fixed_support = [ 91.10000000000001,
      89.2,
      86.3,
      81.89999999999999,
      80.7,
      80.0,
      77.8,
      74.8,
      73.8,
      74.8,
      73.8,
      72.0,
      74.1,
      74.3,
      74.6,
      71.5,
      71.2,
      72.3,
      72.8,
      70.5,
      68.5,
      70.8,
      73.0,
      71.5 ]

    pltResults( [ train_on_user1to4_test_on5,test_on_user1_unseen_sign,train_on_user1to4_test_on5_softmax,train_on_user1to4_test_on5_softmax_with_fixed_support ] )
def split(x_all,y_all):
    start = np.where( y_all == 254 )[ 0 ]
    # end = start + 25
    test_labels_user1 = np.zeros( (250, 1), dtype=int )
    test_data_user1 = np.zeros((250,200,60,3))
    count = 0
    for i in start:
        test_labels_user1[ count:count + 25 ] = y_all[ i:i + 25 ]
        test_data_user1[count:count + 25] = x_all[i:i + 25]
        count += 25
    return [test_data_user1,test_labels_user1]
def OneShotPerformanceTest(source:str = '276'):
    testOneshot = PreTrainModel( )
    testSign = signDataLoder( dataDir=config.train_dir )
    if source == '276':
        _,trained_featureExtractor = defineModel( mode = 'Alexnet' )
        trained_featureExtractor.load_weights( 'D:\OneShotGestureRecognition\models\signFi_featureExtractor_weight_AlexNet_training_acc_0.95_on_250cls.h5' )
        testSign = signDataLoder( dataDir=config.train_dir )
        x_all, y_all = testSign.getFormatedData( source='home' )
        # test_data, test_labels = split( x_all, y_all )
        test_data, test_labels, _, _ = testSign.getTrainTestSplit( data=x_all, labels=y_all,
                                                                   N_train_classes=26,
                                                                   N_samples_per_class= 10)

        test_acc = testOneshot.signTest( test_data=test_data, test_labels=test_labels,
                                         N_test_sample=1000, embedding_model=trained_featureExtractor,
                                         isOneShotTask=True )
    elif source == '150':
        _, trained_featureExtractor = defineModel( mode='Alexnet' )
        trained_featureExtractor.load_weights('./models/signFi_wholeModel_weight_AlexNet_training_acc_0.90_on_125cls_user1to4.h5')
        x_all, y_all = testSign.getFormatedData( source='lab_other' )
        test_data = x_all[ 1250:1500 ]
        test_labels = y_all[ 1250:1500 ]
        test_acc = testOneshot.signTest( test_data=test_data, test_labels=test_labels,
                                         N_test_sample=1000, embedding_model=trained_featureExtractor,
                                         isOneShotTask=True,mode = 'fix' )
    return test_acc
def CnnModelTesting():
    model, _ = defineModel( mode='Alexnet' )
    model.load_weights('./models/signFi_wholeModel_weight_AlexNet_training_acc_0.96_on_276cls.h5')
    testSign = signDataLoder( dataDir=config.train_dir )
    x_all, y_all = testSign.getFormatedData( source='lab_other' )
    test_labels_1 = to_categorical(y_all[0:1250] - 1,num_classes=int(np.max(y_all)-2))
    test_labels_2 = to_categorical( y_all[ 1250:1500 ] - 3, num_classes=int( np.max( y_all ) - 2 ) )
    test_labels = np.concatenate((test_labels_1,test_labels_2),axis = 0)
    model.evaluate(x_all,test_labels)
def pltCrossDomain():
    labToHome = [0.894,0.918,0.934,0.935,0.961,0.98]
    HomeToLab = [0.611,0.734,0.814,0.888,0.904,0.955,0.968,0.977,0.988,0.997]
    user1to4On5= [0.878,0.942,0.947,0.981,0.981]
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    a1 = np.arange(1,6)
    a2 = np.arange( 1, 11 )
    a3 = np.arange(1,7)
    ax.plot(a3,labToHome,linestyle='dashed', marker='o',label = 'Source domain: Lab, Target domain: Home')
    ax.plot( a1, user1to4On5,linestyle='dashed', marker='o', label = 'Source domain: User 1 to 4, Target domain: User 5' )
    ax.plot( a2, HomeToLab,linestyle='dashed', marker='o', label = 'Source domain: Home, Target domain: Lab' )
    ax.legend()
    ax.set_xlabel('Number of shots')
    ax.set_title('Cross Domain performance')
    ax.set_ylabel('Accuracy')

if __name__ == '__main__':
    # plot_barchart()
    # test_acc = OneShotPerformanceTest('150')
    # record()
    # CnnModelTesting()
    pltCrossDomain()

