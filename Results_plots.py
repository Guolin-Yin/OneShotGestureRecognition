from modelPreTraining import *
from Preprocess.gestureDataLoader import signDataLoader

from Config import getConfig
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import random
from scipy.io import savemat,loadmat
# from methodTesting.plotResults import pltResults
config = getConfig()
def pltCrossDomain():
    labToHome = [0.875,0.918,0.934,0.935,0.961]
    # HomeToLab = [0.611,0.734,0.814,0.888,0.904,0.955,0.968,0.977,0.988,0.997]
    # user1to4On5= [0.878,0.942,0.947,0.981,0.981]
    ax = plt.figure(figsize = (12,10)).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # a1 = np.arange(1,6)
    # a2 = np.arange( 1, 11 )
    a3 = np.arange(1,6)
    ax.plot(a3,labToHome,linestyle='dashed', marker='o',label = 'Source domain: Lab, Target domain: Home')
    # ax.plot( a1, user1to4On5,linestyle='dashed', marker='o', label = 'Source domain: User 1 to 4, Target domain: User 5' )
    # ax.plot( a2, HomeToLab,linestyle='dashed', marker='o', label = 'Source domain: Home, Target domain: Lab' )
    ax.legend(fontsize=22)
    ax.set_xlabel('Number of shots',fontsize = 28)
    # ax.set_title('Cross Domain performance')
    ax.set_ylabel('Accuracy',fontsize = 28)
def pltWidar():
    widarNoTuning = [0.356,0.388,0.396,0.396,0.426]
    widar = [0.518,0.693,0.834,0.852,0.906]
    # labToHome = [0.894,0.918,0.934,0.935,0.961,0.98,0.983]
    # HomeToLab = [ 0.611, 0.734, 0.814, 0.888, 0.904, 0.955, 0.968]
    ax = plt.figure(figsize = (12, 10)).gca()
    plt.xticks(fontsize = 28)
    plt.yticks( fontsize = 28 )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # a1 = np.arange(1,8)
    # a2 = np.arange( 1, 11 )
    a3 = np.arange(1,6)
    # ax.plot(a1,labToHome,linestyle='dashed', marker='o',label = 'Cross Domain on SignFi dataset (source: Lab, Target: Home)')
    # ax.plot(
    #         a1, HomeToLab, linestyle = 'dashed', marker = 'o',
    #         label = 'Cross Domain on SignFi dataset (source: Home, Target: lab)'
    #         )
    ax.plot( a3, widar,linestyle='dashed', marker='o', label = 'With fine tuning' )
    ax.plot( a3, widarNoTuning,linestyle='dashed', marker='o', label = 'Without fine tuning' )
    ax.legend( fontsize = 22 )
    ax.set_xlabel( 'Number of shots' ,fontsize = 28 )
    # ax.set_title( 'Cross Domain performance' )
    ax.set_ylabel( 'Accuracy',fontsize = 28  )
def compareWidar():
    domain_611 = [0.496,0.677,0.815,0.862,0.891,0.873,0.873]
    domain_711 = [0.456,0.649,0.732,0.854,0.862,0.873,0.826]
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis = np.arange( 1, 8 )
    ax.plot(
            axis, domain_611, linestyle = 'dashed', marker = 'o',
            label = 'Test on Widar dataset location: 6, orientation: 1, Rx: 1'
            )
    ax.plot(
            axis, domain_711, linestyle = 'dashed', marker = 'o',
            label = 'Test on Widar dataset location: 7, orientation: 1, Rx: 1'
            )
    ax.legend( )
    ax.set_xlabel( 'Number of shots' )
    ax.set_title( 'Cross Domain performance' )
    ax.set_ylabel( 'Accuracy' )
def compareLinkWidar():
    N = 3
    id = np.arange(N)
    width = 0.35
    Rx_1_2_3 =[34.17,53.58,58.58]
    Rx_4_5_6 = [28.30,43.67,39.75 ]
    plt.bar( id, Rx_1_2_3, width, label = 'Receiver 1, 2, 3' )
    plt.bar( id+width, Rx_4_5_6, width, label = 'Receiver 4, 5, 6' )
    plt.ylabel( "Accuracy" )
    plt.title( "Testing on different Receiver, user location: 2" )
    plt.legend()
    plt.xticks( id + width / 2, ('Case 1', 'Case 2','Case 3') )
    for index, data in enumerate( Rx_1_2_3 ):
        plt.text( x = index-0.1, y = data + 1, s = f"{data}%", fontdict = dict( fontsize = 10 ) )
    for index, data in enumerate( Rx_4_5_6 ):
        plt.text( x = index+0.3, y = data + 1, s = f"{data}%", fontdict = dict( fontsize = 10 ) )
    plt.show()
def kFactorNreceiver():
    K = [17.70, 18.20,18.56,19.42,21.97,22.59]
    acc = [58.58, 53.58, 46.3, 43.67,28.30,34.17 ]
    ratio = [0.79,0.755,0.746,0.739,0.725,0.69]
    acc = [34.17,53.58,58.58,46.3,28.30,43.67]
    plt.plot(ratio,acc,linestyle='dashed', marker='o',)
    plt.xlabel( ' Power ratio' )
    plt.ylabel( 'Average accuracy')
    plt.ylim( 0,100 )
def pltResults(
        acc, resultsLabel, axis, linestl = None, markertype = None, linecolor = None, ncol = None, name = None,
        ifsetFigure = 'else', xtic:str = 'K' + ' (No. Novel classes)'
        ):

    ax = plt.figure( figsize = (8, 5) ).gca( )
    ax.xaxis.set_major_locator( MaxNLocator( integer = True ) )
    # markertype = [".","s","o","P","x","d",">"]
    for i in range( len( acc ) ):
        ax.plot(
                # np.arange( 2, len( acc[ i ] ) + 2 ),
                # [10,20,30,40,50,60,70,76],
                axis,
                # np.concatenate( (np.arange( 2, 10 ), np.arange( 10, 77, 10 ),np.asarray([76])), axis = 0 ),
                acc[ i ], linestyle = linestl[i],
                marker = markertype[i],
                # marker = "s",
                color = linecolor[i],
                markersize = 6,
                label = resultsLabel[ i ]
                )
    if ifsetFigure:
        pass
    if ifsetFigure == 'else':
        ax.set_ylim( 35, 101 )
    if ifsetFigure == 'widar':
        ax.set_ylim( 0, 100 )
    fsize = 14
    plt.xticks(fontsize = fsize)
    plt.yticks( fontsize = fsize )
    # ax.set_title( 'Feature extractor trained on lab environment with 125 classes' )
    ax.set_xlabel( xtic, fontsize = fsize )
    ax.set_ylabel( 'Accuracy(%)', fontsize = fsize )
    ax.legend( fontsize = 10,ncol = ncol,loc='upper center', bbox_to_anchor=(0.5, 1.17))
    out = f'C:/Users/29073/iCloudDrive/PhD Research Files/Publications/One-Shot ' \
           f'learning/Results/results_figs/Paperfigure/'+ name
    plt.savefig( out +'.pdf' )
def record():
    resultsLabel = []

    # resultsLabel.append(' Test on lab environment (no fine tuning)')
    # test_on_lab = [ 100.0, 100.0, 100.0, 100.0, 99.90, 100.0, 100.0, 100.0, 99.9, 100.0, 99.8, 100.0, 100.0, 99.9,
    #                 100.0, 99.7, 99.6, 99.8, 99.6, 100.0, 99.9, 99.9, 99.9, 100.0, 100.0 ]
    # resultsLabel.append(' Test on home environment (no fine tuning)')
    # test_on_home = [ 99.2, 98.8, 97.7, 96.39, 96.39, 95.8, 96.0, 95.0, 92.9, 93.7, 92.9, 92.4, 92.2, 90.9, 91.9, 91.8,
    #                  88.7, 89.0, 87.9, 87.0, 89.1, 87.9, 88.9, 87.6, 87.0 ]
    # resultsLabel.append( ' Test on home environment (with fine tuning)' )
    # test_on_home_with_tuning = [99.6, 99.1, 99. , 98.6, 97. , 97.2, 96.5, 96.2, 96.6, 94.8, 94.2,
    #    94.2, 93.9, 92.9, 93.5, 92.3, 91.7, 91.3, 92.7, 91.6, 91.5, 90.4,
    #    89.9, 89.6, 89. ]
    # resultsLabel.append( ' Test on lab_2 environment, user 1 (no fine tuning)' )
    # test_on_lab_2_user_1 = [65.5, 52. , 42.5, 35.7, 31.7, 31.3, 25.1, 21.7, 20.5, 17.3, 18.1,
    #    18.2, 16.2, 14.7, 16.4, 12.6, 13. , 10.8, 12.8, 11.9, 10.5,  9.5,
    #     9.5,  9.9]
    '''=======================================FE 200================================================================='''
    # resultsLabel.append( ' Test on lab_2 environment, user 1 (no fine tuning)' )
    # test_on_lab_2_user_5 = [ 89.5, 84.7, 83.6, 79.3, 79.4, 77.7, 75.9, 75.9, 75.2, 72.8, 75.1,
    #                          72.1, 70.9, 71.9, 71.7, 69.9, 71., 71.9, 70.7, 71.7, 69.9, 68.8,
    #                          69., 69. ]
    # resultsLabel.append( ' Test on lab_2 environment, user 2 (no fine tuning)' )
    # test_on_lab_2_user_2 =[93.4, 89.3, 84.8, 81.8, 78.9, 75.4, 75.5, 74.3, 73.8, 74.7, 73.3,
    #    72.4, 70.4, 68.6, 66.9, 68.9, 66.1, 68.5, 65.8, 66.8, 67.2, 69.2,
    #    66.2, 66.6]
    # resultsLabel.append( ' Test on lab_2 environment, user 3 (no fine tuning)' )
    # test_on_lab_2_user_3 = [96.3, 93.7, 91.7, 91.4, 88.8, 87.5, 87.8, 86.7, 82.9, 84.5, 83.1,
    #    82.4, 84.9, 83.4, 82.5, 81.4, 78.5, 80. , 79.6, 79.6, 76.7, 76.7,
    #    78.1, 76.1]
    # resultsLabel.append( ' Test on lab_2 environment, user 4 (no fine tuning)' )
    # test_on_lab_2_user_4 = [ 98.8, 98.9, 98.1, 97.6, 96.7, 96.6, 95.9, 95.6, 95.2, 94.6, 92.1,
    #   92.5, 92.2, 92.7, 91., 92.5, 92., 92.2, 91.8, 91.3, 91.5, 90.5,
    #   91.1, 91.6 ]

    # resultsLabel.append( ' In-domain performance (200 base classes)' )
    # train_with_lab_200cls_76_testcls_26 = [99.8, 99.5, 99.7, 99.1, 98.5, 98. , 98.6, 98.7, 98.3, 97.8, 98.1,
    #    97.2, 98.3, 96.7, 97.4, 95.9, 96.4, 97.1, 96.1, 95.9, 95.3, 95.7,
    #    95.7, 96.1, 96.5]
    '''COMPARE BASE CLASSES'''
    resultsLabel.append( ' In-domain performance (250 base classes)' )
    train_with_lab_250cls_26_testcls_26 = [ 100.0, 100.0, 100.0, 100.0, 99.90, 100.0, 100.0, 100.0, 99.9, 100.0, 99.8, 100.0, 100.0, 99.9,
                    100.0, 99.7, 99.6, 99.8, 99.6, 100.0, 99.9, 99.9, 99.9, 100.0, 100.0 ]

    resultsLabel.append( ' In-domain performance (200 base classes)' )
    train_with_lab_200cls_26_testcls_26 = [ 99.8, 99.3, 98.7, 98.8, 99.1, 98.7, 98.4, 98.6, 97.7, 98.2, 97.9,
      97.8, 97.6, 97.9, 97.3, 97.5, 97.5, 97.7, 97., 95.7, 98.1, 96.8,
      97.4, 97.5, 96.5 ]
    resultsLabel.append( ' In-domain performance (150 base classes)' )
    train_with_lab_150cls_26_testcls_26 = [99. , 98.4, 98. , 96.3, 95.5, 96.3, 95.1, 94.3, 94.3, 93.2, 93.5,
       92.9, 92.5, 91.7, 92.3, 91. , 92.6, 92.4, 89.8, 89.9, 89.5, 89.6,
       91.6, 90.9, 88.4]
    resultsLabel.append( ' In-domain performance (100 base classes)' )
    train_with_lab_100cls_26_testcls_26 = [98.4, 96.6, 95.8, 94.4, 94.3, 93.8, 93. , 92. , 92. , 88.1, 90. ,
       90.2, 89.7, 88.2, 87.9, 85.8, 87.4, 86.8, 87.8, 85.7, 86. , 85.6,
       87.5, 83.4, 86.1]
    resultsLabel.append( ' In-domain performance (50 base classes)' )
    train_with_lab_50cls_26_testcls_26 = [92.4, 85.9, 81.7, 78. , 74.4, 73.7, 73. , 69.1, 68.3, 66.2, 63. ,
       63.2, 61.6, 58.4, 62.7, 59.8, 57.7, 59.3, 53.6, 56.2, 54.2, 52.7,
       54.8, 52.3, 51.4]

    train_with_lab_50cls_26_testcls_26_test_home = [98. , 94.9, 92.6, 92.5, 88.6, 87.2, 86.8, 84.6, 84.7, 83.3, 83.5,
       80.1, 80.9, 79.6, 77.7, 75. , 76.6, 77. , 75.4, 73.2, 72. , 73.6,
       71.4, 73.6, 73. ]
    train_with_lab_100cls_26_testcls_26_test_home = [95.2, 91.8, 86.7, 85.5, 82. , 83.3, 83.9, 77.7, 77.2, 74.5, 74.3,
       75.4, 72.7, 70.7, 68.9, 69.9, 70.8, 68.3, 66. , 65.8, 68.5, 66.7,
       64.3, 64.4, 61.8]
    train_with_lab_150cls_26_testcls_26_test_home =[97.2, 95.3, 93.5, 91.6, 91.6, 88.8, 88.7, 85.3, 84.7, 82.8, 81.8,
       83.8, 79.5, 81.5, 77.8, 80.6, 78.2, 78.9, 76.8, 76.8, 74.6, 74.5,
       72.8, 74. , 72. ]
    train_with_lab_200cls_26_testcls_26_test_home = [98.7, 97.6, 95.4, 94.1, 95. , 91.5, 91.8, 89.3, 89.8, 89.7, 88.1,
       86.7, 84.7, 84.8, 83.4, 82.5, 82.1, 80. , 81.3, 78.7, 76.6, 77.8,
       77.8, 76.7, 76.9]
    train_with_lab_250cls_26_testcls_26_test_home = [ 99.1, 99.2, 97.6, 96.5, 97.1, 96.8, 95.9, 95.2, 94.7, 93.5, 93.2,
      92.9, 90.1, 91.8, 91.2, 90.4, 92., 88., 91.2, 86.9, 90., 85.3,
      83.7, 88.3, 87.3 ]
    '''=============================================================================================================='''
    in_domain_test = {
            '50_lab': [ 91.3, 86.6, 80.6, 78.9, 73.6, 72.2, 70.7, 71., 67.6, 67., 65.,
                        62.8, 61.8, 58.7, 57.6, 59., 57.9, 55., 56.1, 56.1, 53.7, 53.3,
                        56.1, 52.4, 52.3 ],
            '100_lab': [ 98.8, 97.1, 96.4, 95.3, 93.3, 94., 92.6, 90.3, 90.1, 91.1, 92.2,
                         89.4, 88., 89.1, 89., 85.8, 88.2, 88.1, 87.5, 84., 86., 84.7,
                         87.5, 85.8, 84.3 ],
            '150_lab': [ 98.5, 98.2, 97.3, 95.9, 95.9, 97.1, 95.1, 94.6, 94.2, 93.2, 94.8,
                         93., 93., 92.2, 92.2, 91.2, 91.5, 90.1, 89.1, 90.4, 91.1, 89.1,
                         91.1, 91.7, 91. ],
            '200_lab': [ 99.6, 99.3, 99., 99.4, 99.3, 98.3, 99.2, 98.3, 98., 98.6, 97.9,
                         98.1, 97.9, 98.2, 98.1, 97.5, 97.3, 96.2, 97.4, 96.8, 97.6, 97.4,
                         97.7, 97.3, 97.7 ],
            '250_lab': [ 100., 99.9, 99.9, 100., 99.9, 100., 100., 99.9, 100.,
                         99.9, 100., 99.9, 99.9, 100., 99.8, 100., 99.9, 100.,
                         99.9, 99.9, 99.9, 99.8, 100., 99.8, 99.9 ],

            }
    '''==============================================='''
    test_70_cls_FE_200 = {
            '200_home': [ 93.6, 91.1, 87.4, 82.3, 81.7, 81., 77.1, 76.5, 75.7, 68., 58.7,
                          56.9, 54.1, 52.7, 47.4, 46.0 ],
            '200_lab' : [ 99.9, 99.6, 99.3, 98.7, 98.9, 98.3, 98., 98., 98.2, 95.2, 96.7,
                          93.5, 93.8, 92.8, 92., 91.0]
            }
    test_76_cls_FE_200_256_1280_lab = [99.9,99.8,99.8,98.7,98.8,98.7,98.5,97.8,97.8,96.8,95.4,95.8,95.,93.3,92.3,91.7]
    test_76_cls_FE_200_256_1280_home = [95.2,93.4,88.7,87.8,86.1,84.7,82.5,81.8,80.2,72.5,63.7,64.5,61.3,59.1,
                                         60.9,57.9]
    '''==============================================='''
    test_on_user1_trained_sign = [ 84.6,80.3000,82.699,81.6,79.5,79.9,80.300,
                                   80.4,78.4,82.699,80.300,80.100,78.0,79.4,80.4,78.5,
                                   77.5,79.80,77.8,77.3,78.7,79.1,78.6,78.8,80.0 ]
    test_on_user1_unseen_sign = [ 64.9, 47, 37.9, 34, 29.5, 27.4, 25.1, 22.5, 19.1, 18.4, 17.1, 15.3, 14.3, 14.7, 13.8,
                                  14.5, 15.2, 11.5,
                                  12.4, 10.3, 8.5, 10, 11.6, 7.7 ]
    test_on_user5_trained_sign = [99.3, 99.5, 99.2, 98.2, 98.7, 97.9, 98. , 98.5, 98. , 97.9, 97.2,
       97.1, 97. , 97.1, 97.5, 97.2, 95.6, 96.4, 94.7, 96.8, 96.7, 97.2,
       95.3, 96. , 96.9]
    train_on_user1to4_test_on5 = [93.6, 88.4, 84.9, 83.3, 82.5, 80.6, 77.3, 74.5, 77.1, 73. , 76.9,
       75.2, 72.8, 73.5, 73.4, 71.5, 75.5, 71.5, 71.1, 70.3, 73.4, 72.6,
       70.2, 69.9]
    train_on_user1to4_test_on5_softmax = [93.9, 89. , 86.1, 83. , 81. , 80.2, 79. , 75.1, 77.6, 77.2, 73.6,
       71.6, 75.6, 76.4, 73.4, 72.6, 72.2, 73.4, 70.3, 70.3, 73. , 72.8,
       70.4, 69.9]
    train_on_user1to4_test_on5_softmax_with_fixed_support = [91.1, 89.2, 86.3, 81.9, 80.7, 80. , 77.8, 74.8, 73.8, 74.8, 73.8,
       72. , 74.1, 74.3, 74.6, 71.5, 71.2, 72.3, 72.8, 70.5, 68.5, 70.8,
       73. , 71.5]

    compare_base_classes = [train_with_lab_250cls_26_testcls_26,train_with_lab_200cls_26_testcls_26,
                 train_with_lab_150cls_26_testcls_26,train_with_lab_100cls_26_testcls_26,train_with_lab_50cls_26_testcls_26]
    # compare_environment = [test_on_lab,test_on_home,test_on_home_with_tuning]
    # compare_users = [ test_on_lab_2_user_5, test_on_lab_2_user_2, test_on_lab_2_user_3, test_on_lab_2_user_4,]
    # compare_70_cls = [test_70_cls_FE_200['200_lab'],test_70_cls_FE_200['200_home']]
    # resultsLabel.append('Testing on lab environment')
    # resultsLabel.append( 'Testing on home environment' )
    '''200 base classes compare with adversarial network'''
    adv = []
    adv.append('Adversarial learning')
    adv_training = [ 14.1,30,36,42,54.7,59.3,65.3,69.3]
    adv.append( 'One-shot learning' )
    one_shot_200 = [88,87,81,75,76,71,72,70]
    p = [adv_training,one_shot_200]
    pltResults( [test_76_cls_FE_200_256_1280_home]
           ,['Test in home environment (cross-domain)'] )
def plot_barchart(result):
    width = 0.17
    if result == 'widar':
        N = 5
        id = np.arange( N )
        widarNoTuning = np.asarray( [0.356, 0.388, 0.396, 0.396, 0.426] ) * 100
        widar = np.asarray([ 0.518, 0.693, 0.834, 0.852, 0.906 ]) * 100
        plt.figure( figsize = (12, 10) )
        bar1 = plt.bar(
                np.arange( len( widarNoTuning ) ) + 1.5 * width, widarNoTuning, width, align = 'center', alpha = 0.4,
                label = 'without fine '
                        'tuning'
                )
        bar2 = plt.bar( np.arange( len( widar ) ) + 2.5 * width, widar, width, alpha = 0.4, label = 'with fine tuning' )

        plt.ylabel( "Accuracy", fontsize = 28 )
        plt.legend( fontsize = 22 )
        plt.ylim( 0, 120 )
        plt.yticks( fontsize = 28 )
        plt.xticks(
                np.arange( len( widar ) ) + 2 * width, ('One-shot', 'Two-shot', 'Three-shot', 'Four-shot','Five-shot'),
                fontsize = 26
                )
    if result == 'C_adv':
        N = 1
        id = np.arange( N )
        adv = [4]
        # cnn = [70,0.1]
        oneshot = [46]
        oneshot_FT = [70.9,]
        plt.figure( figsize = (12, 10) )
        p = 0.05
        bar1 = plt.bar( id - p, adv, p, align = 'center', alpha = 0.4, label = 'Adversarial Learning' )
        bar2 = plt.bar( id , oneshot, p, alpha = 0.4, label = 'OSL ' )
        bar3 = plt.bar( id + p, oneshot_FT, p, alpha = 0.4, label = 'OSL with fine tuning' )
        plt.ylabel( "Accuracy", fontsize = 28 )
        plt.legend( fontsize = 22, loc = 'upper left' )
        plt.yticks( fontsize = 28 )
        plt.xticks( id + 2 * p, (' '),
                fontsize = 28 )
        plt.ylim( 0, 120 )
        # for rect in bar1 + bar2 + bar3:
        #     height = rect.get_height( )
        #     plt.text(
        #             rect.get_x( ) + rect.get_width( ) / 2.0, height, f'{height:.0f}%', ha = 'center', va = 'bottom',
        #             fontsize = 17
        #             )
    if result == 'crossDomain':
        N = 2
        id = np.arange(N)
        # oneshot_accuracy = [ 70.0, 88.6 ]
        # fiveshot_accuracy = [ 84.4, 98.0 ]
        oneshot_accuracy = [ 92, 46 ]
        twoshot_accuracy = [ 95, 54]
        threeshot_accuracy = [96,54]
        fourshot_accuracy = [99,61]
        fiveshot_accuracy= [100,62]
        plt.figure( figsize = (12, 10) )
        bar1 = plt.bar( id, oneshot_accuracy, width,align='center', alpha=0.4,label = '1-shots' )
        bar2 = plt.bar( id+width, twoshot_accuracy, width,  alpha=0.4,label = '2-shots' )
        bar3 = plt.bar( id+2*width, threeshot_accuracy, width, alpha=0.4,label = '3-shots' )
        bar4 = plt.bar( id+3*width, fourshot_accuracy, width, alpha=0.4,label = '4-shots' )
        bar5 = plt.bar( id+4*width, fiveshot_accuracy, width, alpha=0.4,label = '5-shots' )
        plt.ylabel( "Accuracy" ,fontsize = 28)
        plt.legend(fontsize = 22)
        plt.yticks(fontsize = 28)
        plt.xticks( id + 2*width , ('Lab', 'Home') ,fontsize = 28)
        plt.ylim( 0, 120 )
        # for rect in bar1 + bar2 + bar3 + bar4 + bar5:
        #     height = rect.get_height( )
        #     plt.text(
        #             rect.get_x( ) + rect.get_width( ) / 2.0, height, f'{height:.0f}%', ha = 'center', va = 'bottom',
        #             fontsize = 17
        #             )
    if result == 'crossDomainFT':
        compareFT = [ 46,  ]
        compareFT1 = [ 70,  ]
        N = 1
        id = np.arange( N )
        plt.figure( figsize = (12, 10) )
        p = 0.05
        bar1 = plt.bar( id - p/2, compareFT, p, alpha = 0.4, label = 'without fine tuning')
        bar2 = plt.bar( id + p/2, compareFT1, p, alpha = 0.4, label = 'with fine tuning' )
        plt.ylim( 0, 120 )
        # for rect in bar1+bar2:
        #     height = rect.get_height( )
        #     plt.text(
        #             rect.get_x( ) + rect.get_width( ) / 2.0, height, f'{height:.0f}%', ha = 'center',
        #             va = 'bottom',
        #             fontsize = 17
        #             )
        plt.ylabel( "Accuracy", fontsize = 28 )
        plt.xlim(-0.1,0.1)
        plt.legend( fontsize = 22 )
        plt.yticks( fontsize = 28 )
        plt.xticks( id  ,
                (' '),
                fontsize = 28 )
    if result == 'crossUser':
        oneshot_accuracy = [ 69, 66.6, 76.2, 91 ]
        twoshot_accuracy = [ 77.2, 69.9, 80.9, 91 ]
        threeshot_accuracy = [ 75.7, 72.6, 82.9, 93.2 ]
        fourshot_accuracy = [ 77.3, 72.2, 84.4, 92.2 ]
        fiveshot_accuracy = [ 77.6, 74.6, 85.6, 94 ]
        plt.figure( figsize = (12, 10) )
        bar1 = plt.bar( np.arange(len(oneshot_accuracy)), oneshot_accuracy, width,align='center', alpha=0.4,label = '1-shots' )
        bar2 = plt.bar( np.arange(len(oneshot_accuracy))+width, twoshot_accuracy, width,  alpha=0.4,label = '2-shots' )
        bar3 = plt.bar( np.arange(len(oneshot_accuracy))+2*width, threeshot_accuracy, width, alpha=0.4,label = '3-shots' )
        bar4 = plt.bar( np.arange(len(oneshot_accuracy))+3*width, fourshot_accuracy, width, alpha=0.4,label = '4-shots' )
        bar5 = plt.bar( np.arange(len(oneshot_accuracy))+4*width, fiveshot_accuracy, width, alpha=0.4,label = '5-shots' )
        plt.ylabel( "Accuracy" ,fontsize = 28)
        plt.legend(fontsize = 22)
        plt.ylim(0, 120)
        plt.yticks( fontsize = 28 )
        plt.xticks( np.arange(len(oneshot_accuracy)) + 2*width , ('User s1', 'User s2','User s3','User s4') ,fontsize = 28)

        # for rect in bar1 + bar2 + bar3 + bar4 + bar5:
        #     height = rect.get_height( )
        #     plt.text( rect.get_x( ) + rect.get_width( ) / 2.0, height, f'{height:.0f}%', ha = 'center', va = 'bottom' ,
        #             fontsize = 14)
    if result == 'crossUserFT':
        noFT = [ 69, 66.6, 76.2, 91 ]
        FT = [ 77,75,91,98 ]

        plt.figure( figsize = (12, 10) )
        bar1 = plt.bar( np.arange(len(noFT))+1.5*width, noFT, width,align='center', alpha=0.4,label = 'without fine '
                                                                                                    'tuning' )
        bar2 = plt.bar( np.arange(len(FT))+2.5*width, FT, width,  alpha=0.4,label = 'with fine tuning' )

        plt.ylabel( "Accuracy" ,fontsize = 28)
        plt.legend(fontsize = 22)
        plt.ylim(0, 120)
        plt.yticks( fontsize = 28 )
        plt.xticks( np.arange(len(FT)) + 2*width , ('User s1', 'User s2','User s3','User s4') ,
                fontsize = 28)
        plt.ylim( 0, 120 )
        # for rect in bar1 + bar2:
        #     height = rect.get_height( )
        #     plt.text( rect.get_x( ) + rect.get_width( ) / 2.0, height, f'{height:.0f}%', ha = 'center', va = 'bottom' ,
        #             fontsize = 14)
    plt.show()
def recordNew(result):
    if result == 'compare_Base':
        resultsLabel = []
        linestl = []
        markertype = []
        linecolor = []
        '''COMPARE BASE CLASSES'''
        resultsLabel.append( ' $N_b = 250$' )
        linecolor.append( 'darkslategrey' )
        linestl.append( 'dashdot' )
        markertype.append( 'h' )
        train_with_lab_250cls_26_testcls_26 = [ 100.0, 100.0, 100.0, 100.0, 99.90, 100.0, 100.0, 100.0, 99.9, 100.0,
                                                99.8, 100.0, 100.0, 99.9,
                                                100.0, 99.7, 99.6, 99.8, 99.6, 100.0, 99.9, 99.9, 99.9, 100.0, 100.0 ]

        resultsLabel.append( ' $N_b = 200$' )
        linecolor.append('cornflowerblue')
        linestl.append('solid')
        markertype.append('s')
        train_with_lab_200cls_26_testcls_26 = [ 99.8, 99.3, 98.7, 98.8, 99.1, 98.7, 98.4, 98.6, 97.7, 98.2, 97.9,
                                                97.8, 97.6, 97.9, 97.3, 97.5, 97.5, 97.7, 97., 95.7, 98.1, 96.8,
                                                97.4, 97.5, 96.5 ]
        resultsLabel.append( ' $N_b = 150$' )
        linecolor.append( 'darkolivegreen' )
        linestl.append( 'dashdot' )
        markertype.append( '4' )
        train_with_lab_150cls_26_testcls_26 = [ 99., 98.4, 98., 96.3, 95.5, 96.3, 95.1, 94.3, 94.3, 93.2, 93.5,
                                                92.9, 92.5, 91.7, 92.3, 91., 92.6, 92.4, 89.8, 89.9, 89.5, 89.6,
                                                91.6, 90.9, 88.4 ]
        resultsLabel.append( ' $N_b = 100$' )
        linecolor.append( 'sandybrown' )
        linestl.append( 'dashdot' )
        markertype.append( 11 )
        train_with_lab_100cls_26_testcls_26 = [ 98.4, 96.6, 95.8, 94.4, 94.3, 93.8, 93., 92., 92., 88.1, 90.,
                                                90.2, 89.7, 88.2, 87.9, 85.8, 87.4, 86.8, 87.8, 85.7, 86., 85.6,
                                                87.5, 83.4, 86.1 ]
        resultsLabel.append( ' $N_b = 50$' )
        linecolor.append( 'peru' )
        linestl.append( 'dashdot' )
        markertype.append( 10 )
        train_with_lab_50cls_26_testcls_26 = [ 92.4, 85.9, 81.7, 78., 74.4, 73.7, 73., 69.1, 68.3, 66.2, 63.,
                                               63.2, 61.6, 58.4, 62.7, 59.8, 57.7, 59.3, 53.6, 56.2, 54.2, 52.7,
                                               54.8, 52.3, 51.4 ]
        acc = [ train_with_lab_250cls_26_testcls_26,train_with_lab_200cls_26_testcls_26,
          train_with_lab_150cls_26_testcls_26,train_with_lab_100cls_26_testcls_26,train_with_lab_50cls_26_testcls_26 ]
        pltResults(
                acc
                , resultsLabel,
                np.arange( 2, 27 ),
                linestl = linestl,
                markertype = markertype,
                linecolor = linecolor,
                ncol = 1,
                name = 'compareBaseClasses'
                )
    if result == 'crossenvir_user1234':
        resultsLabel = []
        markertype = ["o","P","x","d"]
        linecolor = ['green','red','plum','tab:brown']
        resultsLabel.append( 'lab 2, user 1 (cross-domain)' )

        test_on_lab_2_user_1 = [ 95.6, 93.2, 89., 87.2, 83.6, 85.4, 83.6, 78.8, 80.2, 79.2, 79.2, 75.8, 72.8, 79.,
                                 76.4, 76.4, 72.2, 75.6, 74.8, 79.4, 74.2, 72., 73.4, 72. ]
        resultsLabel.append( 'lab 2, user 2 (cross-domain)' )
        test_on_lab_2_user_2 = [ 94., 91., 86.5, 86.6, 83.5, 82., 79.2, 78.5, 77.7, 73.8, 73.4,
                                 74.3, 74.2, 72.4, 73.1, 73.1, 71.2, 75., 71.6, 66.1, 68.5, 70.3,
                                 69.4, 70. ]
        resultsLabel.append( 'lab 2, user 3 (cross-domain)' )
        test_on_lab_2_user_3 = [ 97.1, 95.5, 92.6, 91.4, 90.1, 88., 88.3, 88.6, 86.3, 84.9, 86.2,
                                 85.9, 83.1, 85.5, 81.5, 84.5, 80.7, 82.1, 84.6, 79., 81.2, 81.3,
                                 77.1, 80.4 ]
        resultsLabel.append( 'lab 2, user 4 (cross-domain)' )
        test_on_lab_2_user_4 = [ 99.5, 99.1, 98.8, 98.1, 97.3, 96.2, 96.3, 96.6, 96., 95.2, 95.9,
                                 95.6, 95.4, 94.5, 93.7, 94.3, 93.9, 94.6, 92.4, 92.1, 91.7, 93.8,
                                 92.8, 92.4 ]
        pltResults(
                [ test_on_lab_2_user_1, test_on_lab_2_user_2, test_on_lab_2_user_3, test_on_lab_2_user_4 ],
                resultsLabel,
                np.arange( 2, 26 ),
                linestl = ['dotted','dotted','dotted','dotted'],
                markertype = markertype,
                linecolor = linecolor,
                ncol = 2,
                name = 'crossenvir_user1234'
                )
    if result == 'in_domain':
        test_76_cls_FE_200_256_1280_lab = [99.9,99.8,99.8,98.7,98.8,98.7,98.5,97.8,97.8,96.8,95.4,95.8,95.,93.3,92.3,91.7]

        pltResults(
                [ test_76_cls_FE_200_256_1280_lab ]
                , [ 'lab, user 5 (in-domain)' ],
                np.concatenate( (np.arange( 2, 10 ), np.arange( 10, 77, 10 ), np.asarray( [ 76 ] )), axis = 0 ),
                linestl = ['solid'],
                markertype = ['s'],
                linecolor = ['cornflowerblue'],
                name = 'novel_class_in_domain_new',
                ifsetFigure = True
                )
    if result == 'crossenvir_user5':
        test_76_cls_FE_200_256_1280_home = [95.2,93.4,88.7,87.8,86.1,84.7,82.5,81.8,80.2,72.5,63.7,64.5,61.3,59.1,
                                             60.9,57.9]
        pltResults(
                [ test_76_cls_FE_200_256_1280_home ]
                , [ 'home, user 5 (cross domain)' ],
                np.concatenate( (np.arange( 2, 10 ), np.arange( 10, 77, 10 ), np.asarray( [ 76 ] )), axis = 0 ),
                linestl = [ 'dashed' ],
                markertype = [ 's' ],
                linecolor = [ 'darkorange' ],
                name = 'crossenvir_user5'
                # linecolor = linecolor
                )
    if result == 'cross_domain_user_shots':
        indomain_user_5 = [91.7,95.4,96.9,98,98.3]
        crossdomain_user_5 = [57.9,62.5,68.7,69.9,71.5]
        crossdomain_user_1 = [72,72.9,76.2,75.4,76.6]
        crossdomain_user_2 = [70,74.3,73.2,74.2,77.2]
        crossdomain_user_3 = [80.4,80.6,84.2,84.4,85.3]
        crossdomain_user_4 = [92.4,93.0,94.6,94.5,95.2]
        markertype = ["s","s","o","P","x","d"]
        linecolor = ['cornflowerblue','darkorange','green','red','plum','tab:brown']
        linestl = [ 'solid', 'dashed', 'dotted', 'dotted', 'dotted', 'dotted' ]
        pltResults(
                [ indomain_user_5,crossdomain_user_5,crossdomain_user_1,crossdomain_user_2,crossdomain_user_3,
                  crossdomain_user_4 ]
                , [ 'lab, user 5 (in-domain)','home, user 5 (cross-domain)',
                    'lab 2, user 1 (cross-domain)','lab 2, user 2 (cross-domain)','lab 2, user 3 (cross-domain)',
                    'lab 2, user 4 (cross-domain)'],
                np.arange(1,6),
                linestl = linestl,
                linecolor = linecolor,
                markertype=markertype,
                ncol = 2,
                name = 'FSL',
                xtic = 'N' + ' (No. shots)'
                )
    if result == 'widar':
        FT_user1 = [51.8,69.3,83.4,85.2,90.6]
        FT_user2 = [85.5,87.5,90.5,90.5,99.0]
        FT_user3 = [72.3,90.2,95.2,95.83,100]
        nFT_user1 = [ 35.6, 38.8, 39.6, 39.6, 42.6 ]
        nFT_user2 = [ 69, 74.17, 79, 80.17, 80.17 ]
        nFT_user3 = [ 50, 58.16, 58.83, 64.5, 67.3 ]
        markertype = [ "*", 5, 6, "*", 5, 6 ]
        linecolor = [ 'burlywood','violet','lightseagreen', 'burlywood','violet','lightseagreen']
        linestl = [ 'solid', 'solid', 'solid', 'dotted', 'dotted', 'dotted' ]
        pltResults(
                [ FT_user1, FT_user2, FT_user3, nFT_user1, nFT_user2,
                  nFT_user3 ]
                , [ 'with FT, user W1',
                    'with FT, user W2',
                    'with FT, user W3',
                    'without FT, user W1',
                    'without FT, user W2',
                    'without FT, user W3' ],
                np.arange( 1, 6 ),
                linestl = linestl,
                linecolor = linecolor,
                markertype = markertype,
                ncol = 2,
                name = 'widarperform_3user',
                ifsetFigure = 'widar'
                )
    if result == 'l2_norm':
        without_lab = [99. , 98.5, 97.1, 97.6, 96. , 95.6, 94.7, 94.8, 94.8, 90.1, 88.3,
       85.1, 84.8, 84.2, 81.9, 80.8]
        without_home = [94.6, 88.7, 82. , 80.6, 79.2, 76.9, 79. , 74.7, 72.6, 66. , 62.1,
       58.7, 55.6, 53.3, 51.8, 51.1]
        with_lab = [99.6, 99.5, 99.4, 99. , 98.9, 99.5, 98.2, 98.9, 98.7, 97.4, 96. ,
       94.4, 92.3, 93.6, 93.4, 92.5]
        with_home = [96.4, 93.6, 91.7, 88.2, 88. , 83.4, 83. , 82.2, 81.5, 72.7, 68.2,
       65.9, 63.2, 62.7, 62.3, 61.3]
    plt.show( )
def barChartNew(result):
    width = 0.17
    fsize = 14
    figsize = (8, 5)
    # if result == 'crossDomain':
    #
    #     N = 2
    #     id = np.arange( N )
    #     oneshot_accuracy = [  91.7, 57.9 ]
    #     twoshot_accuracy = [ 95.4, 62.5 ]
    #     threeshot_accuracy = [ 96.9, 68.7 ]
    #     fourshot_accuracy = [ 98, 69.9 ]
    #     fiveshot_accuracy = [ 98.3, 71.5 ]
    #     plt.figure( figsize = (12, 10) )
    #     bar1 = plt.bar( id, oneshot_accuracy, width, align = 'center', alpha = 0.4, label = '1-shots' )
    #     bar2 = plt.bar( id + width, twoshot_accuracy, width, alpha = 0.4, label = '2-shots' )
    #     bar3 = plt.bar( id + 2 * width, threeshot_accuracy, width, alpha = 0.4, label = '3-shots' )
    #     bar4 = plt.bar( id + 3 * width, fourshot_accuracy, width, alpha = 0.4, label = '4-shots' )
    #     bar5 = plt.bar( id + 4 * width, fiveshot_accuracy, width, alpha = 0.4, label = '5-shots' )
    #     plt.ylabel( "Accuracy", fontsize = 28 )
    #     plt.legend( fontsize = 22,loc = 1,ncol = 2 )
    #     plt.yticks( fontsize = 28 )
    #     plt.xticks( id + 2 * width, ('Lab (in-domain)', 'Home (cross-domain)'), fontsize = 28 )
    #     plt.ylim( 0, 100 )
    if result == 'compareFT':
        u5 = [57.9,72.8]
        u1 = [72,81.3]
        u2 = [70,77]
        u3 = [80.4,91.6]
        u4 = [92.4,96.5]
        plt.figure( figsize = (8, 5) )

        bar1 = plt.bar(
                np.arange( len( u1 ) ) + 0 * width, u1, width, align = 'center', alpha = 0.4,
                label = 'lab 2, user 1 (cross-domain)'
                )
        bar1 = plt.bar(
                np.arange( len( u2 ) ) + 1 * width, u2, width, align = 'center', alpha = 0.4,
                label = 'lab 2, user 2 (cross-domain)'
                )
        bar1 = plt.bar(
                np.arange( len( u3 ) ) + 2 * width, u3, width, align = 'center', alpha = 0.4,
                label = 'lab 2, user 3 (cross-domain)'
                )
        bar1 = plt.bar(
                np.arange( len( u4 ) ) + 3 * width, u4, width, align = 'center', alpha = 0.4,
                label = 'lab 2, user 4 (cross-domain)'
                )
        bar1 = plt.bar(
                np.arange( len( u5 ) ) + 4 * width, u5, width, align = 'center', alpha = 0.4,
                label = 'home, user 5 (cross-domain)'
                )

        plt.ylabel( "Accuracy(%)", fontsize = fsize )
        plt.legend( fontsize = 11,loc = 3 )
        plt.ylim( 0, 100 )
        plt.yticks( fontsize = fsize )
        plt.xticks(
                np.arange( len( u4 ) ) + 2 * width, ('without fine-tuning','with fine-tuning'),
                fontsize = fsize
                )
        name = 'compareFinetuning'
        out = f'C:/Users/29073/iCloudDrive/PhD Research Files/Publications/One-Shot ' \
              f'learning/Results/results_figs/Paperfigure/' + name
        plt.savefig( out + '.pdf' )
    if result == 'adv':
        N = 1
        id = np.arange( N )
        adv = [ 4 ]
        # cnn = [70,0.1]
        oneshot = [ 57.9 ]
        oneshot_FT = [ 72.8, ]
        plt.figure( figsize = figsize )
        p = 0.05
        bar1 = plt.bar( id - p, adv, p, align = 'center', alpha = 0.4, label = 'Adversarial Learning' )
        bar2 = plt.bar( id, oneshot, p, alpha = 0.4, label = 'OSL ' )
        bar3 = plt.bar( id + p, oneshot_FT, p, alpha = 0.4, label = 'OSL with fine tuning' )
        plt.ylabel( "Accuracy", fontsize = fsize )
        plt.legend( fontsize = 11, loc = 'upper left' )
        plt.yticks( fontsize = fsize )
        plt.xticks(
                id + 2 * p, (' '),
                fontsize = fsize
                )
        plt.ylim( 0, 100 )
        name = 'advperform'
        out = f'C:/Users/29073/iCloudDrive/PhD Research Files/Publications/One-Shot ' \
              f'learning/Results/results_figs/Paperfigure/' + name
        plt.savefig( out + '.pdf' )
    # if result == 'crossUser':
    #     oneshot_accuracy = [  72, 70, 80.4, 92.4 ]
    #     twoshot_accuracy = [ 72.9, 74.3, 80.6, 93.0 ]
    #     threeshot_accuracy = [ 76.2, 73.2, 84.2,  94.6 ]
    #     fourshot_accuracy = [ 75.4, 74.4, 84.4, 94.6 ]
    #     fiveshot_accuracy = [ 76.6, 77.2, 85.3, 95.2 ]
    #     plt.figure( figsize = (12, 10) )
    #     bar1 = plt.bar(
    #             np.arange( len( oneshot_accuracy ) ), oneshot_accuracy, width, align = 'center', alpha = 0.4,
    #             label = '1-shots'
    #             )
    #     bar2 = plt.bar(
    #             np.arange( len( oneshot_accuracy ) ) + width, twoshot_accuracy, width, alpha = 0.4, label = '2-shots'
    #             )
    #     bar3 = plt.bar(
    #             np.arange( len( oneshot_accuracy ) ) + 2 * width, threeshot_accuracy, width, alpha = 0.4,
    #             label = '3-shots'
    #             )
    #     bar4 = plt.bar(
    #             np.arange( len( oneshot_accuracy ) ) + 3 * width, fourshot_accuracy, width, alpha = 0.4,
    #             label = '4-shots'
    #             )
    #     bar5 = plt.bar(
    #             np.arange( len( oneshot_accuracy ) ) + 4 * width, fiveshot_accuracy, width, alpha = 0.4,
    #             label = '5-shots'
    #             )
    #     plt.ylabel( "Accuracy", fontsize = 28 )
    #     plt.legend( fontsize = 22,loc = 2,ncol = 2 )
    #     plt.ylim( 0, 100 )
    #     plt.yticks( fontsize = 28 )
    #     plt.xticks(
    #             np.arange( len( oneshot_accuracy ) ) + 2 * width, ('User s1', 'User s2', 'User s3', 'User s4'),
    #             fontsize = 28
    #             )
    # if result == 'crossUserFT':
    #     noFT = [ 72, 70, 80.4, 92.4 ]
    #     # FT = [81.3,78.2,91.6,96.4]
    #     FT_test = [ 81.3, 77, 91.6, 96.5 ]
    #     plt.figure( figsize = (12, 10) )
    #     bar1 = plt.bar( np.arange(len(noFT))+1.5*width, noFT, width,align='center', alpha=0.4,label = 'without fine '
    #                                                                                                 'tuning' )
    #     bar2 = plt.bar( np.arange(len(FT_test))+2.5*width, FT_test, width,  alpha=0.4,label = 'with fine tuning' )
    #
    #     plt.ylabel( "Accuracy" ,fontsize = 28)
    #     plt.legend(fontsize = 22)
    #     plt.ylim(0, 100)
    #     plt.yticks( fontsize = 28 )
    #     plt.xticks( np.arange(len(FT_test)) + 2*width , ('User s1', 'User s2','User s3','User s4') ,
    #             fontsize = 28)
    # if result == 'crossDomainFT':
    #         compareFT = [ 57.9, ]
    #         compareFT1 = [ 72.8, ]
    #         N = 1
    #         id = np.arange( N )
    #         plt.figure( figsize = (12, 10) )
    #         p = 0.05
    #         bar1 = plt.bar( id - p / 2, compareFT, p, alpha = 0.4, label = 'without fine tuning' )
    #         bar2 = plt.bar( id + p / 2, compareFT1, p, alpha = 0.4, label = 'with fine tuning' )
    #         plt.ylim( 0, 100 )
    #         # for rect in bar1+bar2:
    #         #     height = rect.get_height( )
    #         #     plt.text(
    #         #             rect.get_x( ) + rect.get_width( ) / 2.0, height, f'{height:.0f}%', ha = 'center',
    #         #             va = 'bottom',
    #         #             fontsize = 17
    #         #             )
    #         plt.ylabel( "Accuracy", fontsize = 28 )
    #         plt.xlim( -0.1, 0.1 )
    #         plt.legend( fontsize = 22 )
    #         plt.yticks( fontsize = 28 )
    #         plt.xticks(
    #                 id,
    #                 (' '),
    #                 fontsize = 28
    #             )

    # if result == 'widar':
    #     N = 5
    #     id = np.arange( N )
    #     # widarNoTuning = np.asarray( [ 0.356, 0.388, 0.396, 0.396, 0.426 ] ) * 100
    #     # widar = np.asarray( [ 0.518, 0.693, 0.834, 0.852, 0.906 ] ) * 100
    #     oneshot_accuracy = [35.6, 51.8, ]
    #     twoshot_accuracy = [38.8, 69.3, ]
    #     threeshot_accuracy = [39.6,83.4]
    #     fourshot_accuracy = [39.6,85.2]
    #     fiveshot_accuracy = [42.6,90.6]
    #     plt.figure( figsize = (12, 10) )
    #     bar1 = plt.bar(
    #             np.arange( len( oneshot_accuracy ) ) + 0 * width, oneshot_accuracy, width, align = 'center', alpha = 0.4,
    #             label = '1-shot'
    #             )
    #     bar1 = plt.bar(
    #             np.arange( len( twoshot_accuracy ) ) + 1 * width, twoshot_accuracy, width, align = 'center',
    #             alpha = 0.4,
    #             label = '2-shot'
    #             )
    #     bar1 = plt.bar(
    #             np.arange( len( threeshot_accuracy ) ) + 2 * width, threeshot_accuracy, width, align = 'center',
    #             alpha = 0.4,
    #             label = '3-shot'
    #             )
    #     bar1 = plt.bar(
    #             np.arange( len( fourshot_accuracy ) ) + 3 * width, fourshot_accuracy, width, align = 'center',
    #             alpha = 0.4,
    #             label = '4-shot'
    #             )
    #     bar1 = plt.bar(
    #             np.arange( len( fiveshot_accuracy ) ) + 4 * width, fiveshot_accuracy, width, align = 'center',
    #             alpha = 0.4,
    #             label = '5-shot'
    #             )
    #
    #     plt.ylabel( "Accuracy(%)", fontsize = fsize )
    #     plt.legend( fontsize = 17,loc = 0 )
    #     plt.ylim( 0, 100 )
    #     plt.yticks( fontsize = fsize )
    #     plt.xticks(
    #             np.arange( len( fiveshot_accuracy ) ) + 2 * width, ('without fine-tuning','with fine-tuning'),
    #             fontsize = fsize
    #             )
    plt.show()
def multiRx():
    resultsLabel = []
    oneShot_NoFT = [25.83,27.83,29.83,31.33,38.33,41.83]
    resultsLabel.append('1 shot without FT')
    oneShot_FT_general = [38.16,47.33,53.5,55.67,56.17,64]
    resultsLabel.append( '1 shot with FT (GM)' )
    oneShot_FT_specific = [38.16,53.17,61,67.5,73.5,75.5]
    resultsLabel.append( '1 shot with FT (RSM)' )
    twoShot_FT_specific = [59.5,68.83,77.5,85.83,87.0,92]
    resultsLabel.append( '2 shot with FT (RSM)' )
    threeShot_FT_specific = [66.83,78,88,92,96.83,98.83]
    resultsLabel.append( '3 shot with FT (RSM)' )
    fourShot_FT_specific = [68.83,80.83,89.33,92.17,95.5,97.33]
    resultsLabel.append( '4 shot with FT (RSM)' )
    fiveShot_FT_specific = [76.83,87.33,95.5,98.5,99.5,100.00]
    resultsLabel.append( '5 shot with FT (RSM)' )
    acc = [ oneShot_NoFT, oneShot_FT_general, oneShot_FT_specific, twoShot_FT_specific, threeShot_FT_specific,
            fourShot_FT_specific, fiveShot_FT_specific ]
    markertype = [ "*","X",4,5,6,7,'D']
    linecolor = [ 'burlywood', 'violet', 'lightseagreen', 'lightseagreen', 'lightseagreen', 'lightseagreen','lightseagreen' ]
    linestl = [ 'dotted', 'dotted','solid', 'solid', 'solid','solid','solid' ]
    pltResults(
            acc,
            resultsLabel,
            np.arange( 1, 7 ),
            linestl = linestl,
            markertype = markertype,
            linecolor = linecolor,
            ncol = 3,
            name = 'multiRx_results',
            xtic = 'No. Receivers',
            ifsetFigure = True
            )
def wiar():
    resultsLabel = []
    oneShot_NoFT = [25.83,27.83,29.83,31.33,38.33,41.83]
    resultsLabel.append('1 shot without FT')
    oneShot_FT_general = [38.16,47.33,53.5,55.67,56.17,64]
    resultsLabel.append( '1 shot with FT (GM)' )
    oneShot_FT_specific = [38.16,53.17,61,67.5,73.5,75.5]
    resultsLabel.append( '1 shot with FT (RSM)' )
    twoShot_FT_specific = [59.5,68.83,77.5,85.83,87.0,92]
    resultsLabel.append( '2 shot with FT (RSM)' )
    threeShot_FT_specific = [66.83,78,88,92,96.83,98.83]
    resultsLabel.append( '3 shot with FT (RSM)' )
    fourShot_FT_specific = [68.83,80.83,89.33,92.17,95.5,97.33]
    resultsLabel.append( '4 shot with FT (RSM)' )
    fiveShot_FT_specific = [76.83,87.33,95.5,98.5,99.5,100.00]
    resultsLabel.append( '5 shot with FT (RSM)' )
    acc = [ oneShot_NoFT, oneShot_FT_general, oneShot_FT_specific, twoShot_FT_specific, threeShot_FT_specific,
            fourShot_FT_specific, fiveShot_FT_specific ]
    markertype = [ "*","X",4,5,6,7,'D']
    linecolor = [ 'burlywood', 'violet', 'lightseagreen', 'lightseagreen', 'lightseagreen', 'lightseagreen','lightseagreen' ]
    linestl = [ 'dotted', 'dotted','solid', 'solid', 'solid','solid','solid' ]
    pltResults(
            acc,
            resultsLabel,
            np.arange( 1, 7 ),
            linestl = linestl,
            markertype = markertype,
            linecolor = linecolor,
            ncol = 3,
            name = 'multiRx_results',
            xtic = 'No. Receivers',
            ifsetFigure = True
            )
if __name__ == '__main__':
    multiRx()
    # recordNew( result = 'compare_Base' )
    # recordNew( result = 'in_domain' )
    # recordNew( result = 'crossenvir_user5' )
    # recordNew(result = 'crossenvir_user1234')
    # recordNew( result = 'cross_domain_user_shots' )
    # recordNew( result = 'widar' )


