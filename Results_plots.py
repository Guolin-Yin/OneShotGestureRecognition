from modelPreTraining import *
from Preprocess.gestureDataLoader import signDataLoader

from Config import getConfig
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import random
from scipy.io import savemat,loadmat
from matplotlib.lines import Line2D
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
        acc, resultsLabel, axis, yrange = None,xrange=None, linestl = None, markertype = None, linecolor = None, \
                                                                                                      ncol = None,
        name = None,
        ifsetFigure = 'else', xtic:str = 'N' + ' (No. Novel Classes)', bbox_to_anchor=None,fillstyle
        =Line2D.fillStyles[-1]
        ):

    ax = plt.figure( figsize = (8, 4.5) ).gca( )
    ax.xaxis.set_major_locator( MaxNLocator( integer = True ) )
    # box = ax.get_position( )
    # ax.set_position( [ box.x0, box.y0, box.width * 0.8, box.height ] )
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
                # markersize = 12,
                label = resultsLabel[ i ],
                ms=10,mew=1,
                linewidth = 1,
                fillstyle = fillstyle
                )
    # if ifsetFigure:
    #     pass
    # if ifsetFigure == 'else':
    #     ax.set_ylim( 35, 101 )
    # if ifsetFigure == 'widar':
    if yrange is not None:
        a,b = yrange
        ax.set_ylim( a, b )
    if xrange is not None:
        c, d = xrange
        ax.set_xlim( c, d )
    # ax.set_xlim( 39, 80 )
    fsize = 14
    plt.xticks(fontsize = fsize)
    plt.yticks( fontsize = fsize )
    # ax.set_title( 'Feature extractor trained on lab environment with 125 classes' )
    ax.set_xlabel( xtic, fontsize = fsize )
    ax.set_ylabel( 'Accuracy(%)', fontsize = fsize )
    if bbox_to_anchor is None:
        ax.legend( fontsize = 10,ncol = ncol,
                # loc='upper center',
                fancybox = True, shadow=True,
                labelspacing=0.1
                # bbox_to_anchor=bbox_to_anchor
                )
    else:
        ax.legend(
                fontsize = 10, ncol = ncol,
                # loc='upper center',
                fancybox = True, shadow = True,
                bbox_to_anchor = bbox_to_anchor
                )
    plt.grid( alpha = 0.2)
    out = f'C:/Users/29073/iCloudDrive/PhD Research Files/Publications/One-Shot ' \
           f'learning/Results/results_figs/Paperfigure/'+ name
    plt.savefig( out +'.pdf',bbox_inches='tight' )
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
        acc = []
        '''COMPARE BASE CLASSES'''
        resultsLabel.append( ' $N_b = 250$' )
        linecolor.append( 'tab:green' )
        linestl.append( 'solid' )
        markertype.append( 's' )
        train_with_lab_250cls_26_testcls_26 = [ 100.0, 100.0, 100.0, 100.0, 99.90, 100.0, 100.0, 100.0, 99.9, 100.0,
                                                99.8, 100.0, 100.0, 99.9,
                                                100.0, 99.7, 99.6, 99.8, 99.6, 100.0, 99.9, 99.9, 99.9, 100.0, 100.0 ]
        acc.append(train_with_lab_250cls_26_testcls_26 )

        resultsLabel.append( ' $N_b = 200$' )
        linecolor.append('darkblue')
        linestl.append('solid')
        markertype.append('o')
        train_with_lab_200cls_26_testcls_26 = [ 99.8, 99.3, 98.7, 98.8, 99.1, 98.7, 98.4, 98.6, 97.7, 98.2, 97.9,
                                                97.8, 97.6, 97.9, 97.3, 97.5, 97.5, 97.7, 97., 95.7, 98.1, 96.8,
                                                97.4, 97.5, 96.5 ]
        acc.append(train_with_lab_200cls_26_testcls_26 )


        resultsLabel.append( ' $N_b = 150$' )
        linecolor.append( 'k' )
        linestl.append( 'solid' )
        markertype.append( '4' )
        train_with_lab_150cls_26_testcls_26 = [ 99., 98.4, 98., 96.3, 95.5, 96.3, 95.1, 94.3, 94.3, 93.2, 93.5,
                                                92.9, 92.5, 91.7, 92.3, 91., 92.6, 92.4, 89.8, 89.9, 89.5, 89.6,
                                                91.6, 90.9, 88.4 ]
        acc.append(train_with_lab_150cls_26_testcls_26 )


        resultsLabel.append( ' $N_b = 100$' )
        linecolor.append( 'tab:red' )
        linestl.append( 'solid' )
        markertype.append( "v" )
        train_with_lab_100cls_26_testcls_26 = [ 98.4, 96.6, 95.8, 94.4, 94.3, 93.8, 93., 92., 92., 88.1, 90.,
                                                90.2, 89.7, 88.2, 87.9, 85.8, 87.4, 86.8, 87.8, 85.7, 86., 85.6,
                                                87.5, 83.4, 86.1 ]
        acc.append(train_with_lab_100cls_26_testcls_26 )


        resultsLabel.append( ' $N_b = 50$' )
        linecolor.append( 'darkmagenta' )
        linestl.append( 'solid' )
        markertype.append( "^" )
        train_with_lab_50cls_26_testcls_26 = [ 92.4, 85.9, 81.7, 78., 74.4, 73.7, 73., 69.1, 68.3, 66.2, 63.,
                                               63.2, 61.6, 58.4, 62.7, 59.8, 57.7, 59.3, 53.6, 56.2, 54.2, 52.7,
                                               54.8, 52.3, 51.4 ]
        acc.append(train_with_lab_50cls_26_testcls_26 )



        a = loadmat('in_domain_base.mat', squeeze_me = 1)

        resultsLabel.append( ' $N_b = 20$' )
        linecolor.append( 'maroon' )
        linestl.append( 'solid' )
        markertype.append( "h" )
        acc.append(a['20'] )


        resultsLabel.append( ' $N_b = 10$' )
        linecolor.append( 'teal' )
        linestl.append( 'solid' )
        markertype.append( "x" )
        acc.append(a['10'] )


        # acc = [ train_with_lab_250cls_26_testcls_26,train_with_lab_200cls_26_testcls_26,
        #   train_with_lab_150cls_26_testcls_26,train_with_lab_100cls_26_testcls_26,train_with_lab_50cls_26_testcls_26 ]
        pltResults(
                acc
                , resultsLabel,
                np.arange( 2, 27 ),
                linestl = linestl,
                markertype = markertype,
                linecolor = linecolor,
                ncol = 3,
                name = 'compareBaseClasses',
                yrange = (15,103), xrange = (1,27), fillstyle = 'none',
                # bbox_to_anchor = (0.5, .7)
                )
    if result == 'crossenvir_user1234':
        resultsLabel = []
        markertype = ["*","+","x","d"]
        linecolor = ['green','red','deeppink','tab:brown']
        resultsLabel.append( 'lab 2, s1' )

        test_on_lab_2_user_1 = [ 95.6, 93.2, 89., 87.2, 83.6, 85.4, 83.6, 78.8, 80.2, 79.2, 79.2, 75.8, 72.8, 79.,
                                 76.4, 76.4, 72.2, 75.6, 74.8, 79.4, 74.2, 72., 73.4, 72. ]
        resultsLabel.append( 'lab 2, s2' )
        test_on_lab_2_user_2 = [ 94., 91., 86.5, 86.6, 83.5, 82., 79.2, 78.5, 77.7, 73.8, 73.4,
                                 74.3, 74.2, 72.4, 73.1, 73.1, 71.2, 75., 71.6, 66.1, 68.5, 70.3,
                                 69.4, 70. ]
        resultsLabel.append( 'lab 2, s3' )
        test_on_lab_2_user_3 = [ 97.1, 95.5, 92.6, 91.4, 90.1, 88., 88.3, 88.6, 86.3, 84.9, 86.2,
                                 85.9, 83.1, 85.5, 81.5, 84.5, 80.7, 82.1, 84.6, 79., 81.2, 81.3,
                                 77.1, 80.4 ]
        resultsLabel.append( 'lab 2, s4' )
        test_on_lab_2_user_4 = [ 99.5, 99.1, 98.8, 98.1, 97.3, 96.2, 96.3, 96.6, 96., 95.2, 95.9,
                                 95.6, 95.4, 94.5, 93.7, 94.3, 93.9, 94.6, 92.4, 92.1, 91.7, 93.8,
                                 92.8, 92.4 ]
        pltResults(
                [ test_on_lab_2_user_1, test_on_lab_2_user_2, test_on_lab_2_user_3, test_on_lab_2_user_4 ],
                resultsLabel,
                np.arange( 2, 26 ),
                linestl = ['solid','solid','solid','solid'],
                markertype = markertype,
                linecolor = linecolor,
                ncol = 4,
                xtic = 'N (No. Novel Classes)',
                name = 'crossenvir_user1234',
                yrange=(30,102)
                )
    if result == 'in_domain':
        # test_76_cls_FE_200_256_1280_lab = [99.9,99.8,99.8,98.7,98.8,98.7,98.5,97.8,97.8,96.8,95.4,95.8,95.,93.3,92.3,91.7]
        test_76_cls_FE_200_256_1280_lab = [95.4,95.8,95.,93.3,92.3,91.7]
        pltResults(
                [ test_76_cls_FE_200_256_1280_lab ]
                , [ 'lab, s5' ],
                # np.concatenate( (np.arange( 2, 10 ), np.arange( 10, 77, 10 ), np.asarray( [ 76 ] )), axis = 0 ),
                np.concatenate((np.arange( 30, 77, 10 ), np.asarray( [ 76 ] )),axis = 0),
                linestl = ['solid'],
                markertype = ['o'],
                linecolor = ['darkblue'],
                name = 'novel_class_in_domain_new',
                ifsetFigure = True,
                yrange = (90, 100), xrange =(29,78),fillstyle = 'none'
                )
    if result == 'crossenvir_user5':
        linestl = []
        markertype = []
        linecolor = []
        acc = []
        resultsLabel = []
        acc.append([95.2,93.4,88.7,87.8,86.1,84.7,82.5,81.8,80.2,72.5,63.7,64.5,61.3,59.1,
                                             60.9,57.9])
        linestl.append('solid')
        markertype.append('o')
        linecolor.append('darkorange')
        resultsLabel.append('home, s5')

        a = loadmat( 'in_domain_base_76.mat', squeeze_me = 1 )
        resultsLabel.append( ' $N_b = 20$' )
        linecolor.append( 'maroon' )
        linestl.append( 'solid' )
        markertype.append( "h" )
        acc.append(a['20'] )


        resultsLabel.append( ' $N_b = 10$' )
        linecolor.append( 'teal' )
        linestl.append( 'solid' )
        markertype.append( "x" )
        acc.append(a['10'] )
        pltResults(
                acc
                , resultsLabel,
                np.concatenate( (np.arange( 2, 10 ), np.arange( 10, 77, 10 ), np.asarray( [ 76 ] )), axis = 0 ),
                linestl = linestl,
                markertype = markertype,
                linecolor = linecolor,
                name = 'crossenvir_user5',
                xtic = 'N (No. Novel Classes)',
                yrange = (30,100),xrange = (0,78)
                # linecolor = linecolor
                )
    if result == 'cross_domain_user_shots':
        # indomain_user_5 = [91.7,95.4,96.9,98,98.3]
        crossdomain_user_5 = [57.9,62.5,68.7,69.9,71.5]
        crossdomain_user_1 = [72,72.9,76.2,75.4,76.6]
        crossdomain_user_2 = [70,74.3,73.2,74.2,77.2]
        crossdomain_user_3 = [80.4,80.6,84.2,84.4,85.3]
        crossdomain_user_4 = [92.4,93.0,94.6,94.5,95.2]
        markertype = [
                # "s",
                "o","s","+","x","d"]
        linecolor = [
                # 'cornflowerblue',
                     'darkorange','green','red','tab:blue','tab:brown']
        linestl = [
                # 'solid',
                'solid', 'solid', 'solid', 'solid', 'solid' ]
        pltResults(
                [
                            # indomain_user_5,
                  crossdomain_user_5,crossdomain_user_1,crossdomain_user_2,crossdomain_user_3,
                  crossdomain_user_4 ]
                ,
                [
                        # 'lab, user 5 (in-domain)',
                  'home, s5',
                    'lab 2, s1','lab 2, s2','lab 2, s3',
                    'lab 2, s4'],
                np.arange(1,6),
                linestl = linestl,
                linecolor = linecolor,
                markertype=markertype,
                ncol = 3,
                name = 'FSL',
                xtic = 'K' + ' (No. Shots)',
                yrange = (30,100),
                # bbox_to_anchor=(0.5,1.17)
                )
    if result == 'widar':
        FT_user1 = [51.8,69.3,83.4,85.2,90.6]
        FT_user2 = [85.5,87.5,90.5,90.5,99.0]
        FT_user3 = [72.3,90.2,95.2,95.83,100]
        nFT_user1 = [ 35.6, 38.8, 39.6, 39.6, 42.6 ]
        nFT_user2 = [ 69, 74.17, 79, 80.17, 80.17 ]
        nFT_user3 = [ 50, 58.16, 58.83, 64.5, 67.3 ]
        markertype = [ "o", "o", 'v', 'v', 's', 's' ]
        linecolor = [ 'darkblue','darkblue','darkgreen','darkgreen', 'darkmagenta','darkmagenta']
        linestl = [ 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed' ]


        pltResults(
                [FT_user1,nFT_user1,FT_user2,nFT_user2,FT_user3,nFT_user3]
                , ['w/ FT, w1','w/o FT, w1','w/ FT, w2','w/o FT, w2','w/ FT, w3',
                   'w/o FT, w3'],
                np.arange( 1, 6 ),
                linestl = linestl,
                linecolor = linecolor,
                markertype = markertype,
                ncol = 3,
                name = 'widarperform_3user',
                xtic = 'K (No. Shots)',
                ifsetFigure = 'widar',
                yrange=(30,103),
                bbox_to_anchor = (0.3,0.37)
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
    plt.show( block = True )
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
        u5 = [57.9,72.8,87.1]#87.1
        u1 = [72,81.3,88.0]#88.0
        u2 = [70,77,96.8]#96.8
        u3 = [80.4,91.6,98.4]#98.4
        u4 = [92.4,96.5,99.2]#99.2


        oneshot = [72,70,80.4,92.4,57.9,]
        twoshot = [81.3,77,91.6,96.5,72.8,]
        threeshot = [88.0,96.8,98.4,99.2,87.1,]
        plt.figure( figsize = (8, 4.5) )
        plt.bar(
                np.arange( len( oneshot ) ) + 0 * width, oneshot, width, align = 'center', alpha = 0.4,
                label = 'without fine-tuning'
                )
        plt.bar(
                np.arange( len( twoshot ) ) + 1 * width, twoshot, width, align = 'center', alpha = 0.4,
                label = 'with fine-tuning (1 shot)'
                )
        plt.bar(
                np.arange( len( threeshot ) ) + 2 * width, threeshot, width, align = 'center', alpha = 0.4,
                label = 'with fine-tuning (5 shots)'
                )
        plt.xticks(
                np.arange( len( threeshot ) ) + 2 * width, ('User s1', 'User s2', 'User s3', 'User s4','User s5'),
                fontsize = fsize
                )
        # bar1 = plt.bar(
        #         np.arange( len( u1 ) ) + 0 * width, u1, width, align = 'center', alpha = 0.4,
        #         label = 'lab 2, user 1 (cross-domain)'
        #         )
        # bar1 = plt.bar(
        #         np.arange( len( u2 ) ) + 1 * width, u2, width, align = 'center', alpha = 0.4,
        #         label = 'lab 2, user 2 (cross-domain)'
        #         )
        # bar1 = plt.bar(
        #         np.arange( len( u3 ) ) + 2 * width, u3, width, align = 'center', alpha = 0.4,
        #         label = 'lab 2, user 3 (cross-domain)'
        #         )
        # bar1 = plt.bar(
        #         np.arange( len( u4 ) ) + 3 * width, u4, width, align = 'center', alpha = 0.4,
        #         label = 'lab 2, user 4 (cross-domain)'
        #         )
        # bar1 = plt.bar(
        #         np.arange( len( u5 ) ) + 4 * width, u5, width, align = 'center', alpha = 0.4,
        #         label = 'home, user 5 (cross-domain)'
        #         )
        # plt.xticks(
        #         np.arange( len( u4 ) ) + 2 * width, ('without fine-tuning', 'with fine-tuning(1 shot)',
        #                                              'with fine-tuning(5 shots)'),
        #         fontsize = fsize
        #         )
        plt.ylabel( "Accuracy(%)", fontsize = fsize )
        plt.legend( fontsize = 11,loc = 3,fancybox = True, shadow=True, )
        plt.ylim( 0, 100 )
        plt.yticks( fontsize = fsize )

        name = 'compareFinetuning'
        out = f'C:/Users/29073/iCloudDrive/PhD Research Files/Publications/One-Shot ' \
              f'learning/Results/results_figs/Paperfigure/' + name
        plt.savefig( out + '.pdf',bbox_inches='tight'  )
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
def multiRx(ID= 'user1'):
    if ID == 'user1':
        resultsLabel = []
        # oneShot_NoFT = [25.83,27.83,29.83,31.33,38.33,41.83]
        # resultsLabel.append('1 shot')
        # oneShot_FT_general = [38.16,47.33,53.5,55.67,56.17,64]
        # resultsLabel.append( '1 shot' )
        oneShot_FT_specific = [38.16,53.17,61,67.5,73.5,75.5]
        resultsLabel.append( '1 shot' )
        twoShot_FT_specific = [59.5,68.83,77.5,85.83,87.0,92]
        resultsLabel.append( '2 shots' )
        threeShot_FT_specific = [66.83,78,88,92,96.83,98.83]
        resultsLabel.append( '3 shots' )
        fourShot_FT_specific = [68.83,80.83,89.33,92.17,95.5,97.33]
        resultsLabel.append( '4 shots' )
        fiveShot_FT_specific = [76.83,87.33,95.5,98.5,99.5,100.00]
        resultsLabel.append( '5 shots' )
    elif ID == 'user2':
        resultsLabel = []
        # oneShot_NoFT = [25.83,27.83,29.83,31.33,38.33,41.83]
        # resultsLabel.append('1 shot without FT')
        # oneShot_FT_general = [38.16,47.33,53.5,55.67,56.17,64]
        # resultsLabel.append( '1 shot with FT (GM)' )
        oneShot_FT_specific = [67.67,77.67,81.33,82.83,92.00,94.33]
        resultsLabel.append( '1 shot' )
        twoShot_FT_specific = [79.00,89.00,92.00,94.50,97.50,98.33]
        resultsLabel.append( '2 shots' )
        threeShot_FT_specific = [79.33,90.67,95.17,98.5,99.83,100.00]
        resultsLabel.append( '3 shots' )
        fourShot_FT_specific = [87.83,93,95.33,97.5,98,98.83]
        resultsLabel.append( '4 shots' )
        fiveShot_FT_specific = [86,91,94.67,95.83,98,99]
        resultsLabel.append( '5 shots' )
    elif ID == 'user3':
        resultsLabel = [ ]
        # oneShot_NoFT = [25.83,27.83,29.83,31.33,38.33,41.83]
        # resultsLabel.append('1 shot without FT')
        # oneShot_FT_general = [38.16,47.33,53.5,55.67,56.17,64]
        # resultsLabel.append( '1 shot with FT (GM)' )
        oneShot_FT_specific = [ 56, 65.67, 72.50, 82.50, 86.00, 90.17 ]
        resultsLabel.append( '1 shot' )
        twoShot_FT_specific = [ 73.00, 82.33, 88.83, 96.33, 98.50, 100.00 ]
        resultsLabel.append( '2 shots' )
        threeShot_FT_specific = [ 79.83, 90.83, 95.17, 96.50, 96.83, 98.17 ]
        resultsLabel.append( '3 shots' )
        fourShot_FT_specific = [ 84.00, 86.67, 94.00, 97.83, 99.83, 100.00 ]
        resultsLabel.append( '4 shots' )
        fiveShot_FT_specific = [ 86.83, 95.83, 98.67, 100.00, 100.00, 100.00 ]
        resultsLabel.append( '5 shots' )
    acc = [ oneShot_FT_specific, twoShot_FT_specific, threeShot_FT_specific, fourShot_FT_specific,
            fiveShot_FT_specific ]
    # markertype = [ "*","X",4,5,"^",7,'^']
    markertype = ['p','*','v','+','x']
    linecolor = [ 'black', 'olive', 'r', 'blueviolet', 'royalblue' ]
    linestl = [ 'solid', 'solid','solid', 'solid', 'solid','solid','solid' ]
    pltResults(
            acc,
            resultsLabel,
            np.arange( 1, 7 ),
            linestl = linestl,
            markertype = markertype,
            linecolor = linecolor,
            ncol = 5,
            name = 'multiRx_results' + f'{ID}',
            xtic = 'No. Receivers',
            ifsetFigure = True,
            yrange = (30,102)
            )
def wiar():
    resultsLabel = []
    u2 = [66.2,79.5,86.8,89.2,94.2,]
    resultsLabel.append('a1')
    u6 = [54.2,60.9,75.9,78.1,86.3]
    resultsLabel.append( 'a2' )
    u7 = [31.03,37.5,46.52,48.32,57.24]
    resultsLabel.append( 'a3' )
    u8 = [60.13,73.66,80.32,84.61,88.74]
    resultsLabel.append( 'a4' )
    u9 = [54.09,62.72,75.23,78.60,81.25]
    resultsLabel.append( 'a5' )
    u10 = [51.29,64.73,78.24,81.97,88.49]
    resultsLabel.append( 'a6' )
    acc = [ u2,u6,u7,u8,u9,u10 ]
    markertype = [ "s",">",'v','+','p','*']
    linecolor = [ 'b', 'red', 'c', 'm', 'green', 'k' ]
    # linecolor = ['pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
    #                   'Wistia', 'hot']
    linestl = [ 'solid', 'solid','solid', 'solid', 'solid','solid','solid' ]
    pltResults(
            acc,
            resultsLabel,
            np.arange( 1, 6 ),
            linestl = linestl,
            markertype = markertype,
            linecolor = linecolor,
            ncol = 3,
            name = 'wiar_dataset',
            xtic = 'K (No. Shots)',
            # bbox_to_anchor = (0.5,1.17),
            yrange = (30,100)
            # ifsetFigure = True
            )
def pltconverge():
    fsize = 14
    N = 1000
    oneshot = loadmat('signfi_oneshot_history.mat',squeeze_me = 1)
    retrain = loadmat('signfi_oneshot_retrain_history.mat',squeeze_me = 1)
    ax = plt.figure( figsize = (8, 4.5) ).gca( )
    ax.xaxis.set_major_locator( MaxNLocator( integer = True ) )
    ax.plot(oneshot['val_acc'][0:N],label = 'Fine-tunine')
    ax.plot(retrain['val_acc'][0:N],label = 'Retraining')

    ax2 = ax.twinx( )
    ax2.set_ylabel( 'Y2-axis' )

    ax2.plot( oneshot[ 'val_loss' ][ 0:N ], label = 'Fine-tunine',alpha = 0.5 )
    ax2.plot( retrain[ 'val_loss' ][ 0:N ], label = 'Retraining',alpha = 0.5 )

    ax.set_xlabel( 'Epoch', fontsize = fsize )
    ax.set_ylabel( 'Validation Accuracy', fontsize = fsize )
    ax2.set_ylabel( 'Validation Loss', fontsize = fsize )
    out = 'compare_convergence_results'
    plt.xticks( fontsize = fsize )
    plt.yticks( fontsize = fsize )
    plt.grid( alpha = 0.2 )
    ax.legend(
            fontsize = 10,
            fancybox = True, shadow = True,
            # bbox_to_anchor=bbox_to_anchor
            )
    plt.savefig( out + '.pdf', bbox_inches = 'tight' )
def pltimpact_cls():
    # acc = [27.7,47.9,49.9,48.7, 47.8, 49.2,48.9,49.9,50.2,49.4,49.2,48.4,47.4,50.4,50.9,60.6,55.7,55.7,58.8,57.3]
    # cls = np.linspace(10,200,20)

    # acc = [62.1, 54.9, 59.1, 59.1, 40.8, 42.2, 50.9, 40.6, 35.2, 49. , 39.3, 35. , 44.9, 55.7, 57.8, 28.7, 49.8, 57.3, 32.1]
    # cls = np.linspace(2,20,19)

    # Fine-tuning results
    a = loadmat( 'FT_acc_indomain.mat', squeeze_me = 1 )
    keys,vals = [],[]
    for key,val in a.items():
        if key == '__header__' or key == '__version__' or key == '__globals__':
            continue
        keys.append(int(key.split('_')[-1].split('cls')[0]))
        vals.append(val)
    idx_sort = np.argsort(keys)
    keys = np.asarray(keys)[idx_sort]
    vals = np.asarray(vals)[idx_sort] * 100

    acc = np.asarray([0.56140351, 0.50438595, 0.2134503 , 0.35526314, 0.53216374, 0.52046782, 0.43859649, 0.39766082, 0.52485383, 0.39181286, 0.36695907, 0.49707603, 0.56578946, 0.54970759,
                      0.30701753, 0.51754385, 0.60380119, 0.59385966, 0.53654969, 0.5847953 , 0.55409354, 0.59064329, 0.5269006 , 0.5350877 , 0.53654969, 0.52777779, 0.5350877 , 0.54590643,
                      0.61023394, 0.728])*100
    cls = np.concatenate((np.linspace(3,20,18,dtype = int),np.asarray([30,40,50,60,70,80,90,100,110,120,150,200,])))

    acc_all = [acc,vals]
    idx_cross = [ np.where( cls == i )[ 0 ] for i in [ 5, 6, 10, 14, 18,30,40,50,60,120,150,200 ] ]
    idx_in = [3,12,22,23,24,25,26,27,28,-4,-2,-1]
    idx = [idx_cross,idx_in]
    ax = plt.figure( figsize = (8, 4.5) ).gca( )
    ax.xaxis.set_major_locator( MaxNLocator( integer = True ) )

    for i,acc in enumerate(acc_all):
        ax.plot(

                [5, 6, 10, 14, 18,35,50,80,110,140,170,200],
                acc[np.asarray(idx[i])],
                # keys,
                # acc,


                ms = 10, mew = 1,
                marker = 'o',
                linewidth = 1,
                fillstyle = Line2D.fillStyles[-1]
                )
    fsize = 14
    ax.set_ylim( 0, 100 )
    ax.set_xlabel( 'The number of base classes', fontsize = fsize )
    ax.set_ylabel( 'Accuracy(%)', fontsize = fsize )
    plt.grid( alpha = 0.2 )
    plt.show( )
    name = 'compareBasecls_cross_domain'
    out = f'C:/Users/29073/iCloudDrive/PhD Research Files/Publications/One-Shot ' \
          f'learning/Results/results_figs/Paperfigure/' + name
    # plt.savefig( out + '.pdf', bbox_inches = 'tight' )
def R1C1():
    home_u5 = np.array([82.2,84.5,86.5,92.5,93.5])
    lab2_u1 = np.array([0.68000001, 0.70499998, 0.71428573, 0.75333333, 0.792     ])*100
    lab2_u2 = np.array([0.80000001, 0.86000001, 0.88      , 0.89333332, 0.912     ])*100
    lab2_u3 = np.array([0.92888892, 0.97000003, 0.96571428, 0.97333336, 0.96799999])*100
    lab2_u4 = np.array([0.68000001, 0.72000003, 0.78857142, 0.81999999, 0.83999997])*100
    acc = [home_u5,lab2_u4,lab2_u3,lab2_u2,lab2_u1,]
    label = ['home s5', 'lab 2 s4', 'lab 2 s3', 'lab 2 s2', 'lab 2 s1']


    ax = plt.figure( figsize = (8, 4.5) ).gca( )
    ax.xaxis.set_major_locator( MaxNLocator( integer = True ) )
    for i in range( len(acc)):
        ax.plot(
                [1,2,3,4,5],
                acc[i],
                ms = 10, mew = 1,
                marker = 'o',
                linewidth = 1,
                fillstyle = Line2D.fillStyles[-1],
                label = label[i]
                )
    fsize = 14
    ax.set_ylim( 0, 100 )
    ax.set_xlabel( 'The number of base classes', fontsize = fsize )
    ax.set_ylabel( 'Accuracy(%)', fontsize = fsize )
    plt.grid( alpha = 0.2 )
    plt.legend()
    name = 'compareBasecls_cross_domain'
    out = f'C:/Users/29073/iCloudDrive/PhD Research Files/Publications/One-Shot ' \
          f'learning/Results/results_figs/Paperfigure/' + name
def compareDomainSimilarity():
    '''
    1. lab -> home
    2. lab -> lab 2
    3. lab -> widar
    '''
    home_u5 = np.asarray([1,1,1,1,1])*100
    lab_2_avg = np.asarray([0.950, 0.953, 0.96, 0.97, 0.99])* 100
    widar = np.asarray([0.6986666667, 0.8233333333, 0.897, 0.9051, 0.9653333333])*100
    wiar = np.asarray([0.5282333333, 0.6316833333, 0.73835, 0.768, 0.8270333333])*100
    acc = [home_u5,lab_2_avg,widar,wiar]
    # label = ['Cross Environment', 'Cross Environment User', 'Cross Dataset (Gesture)', 'Cross Dataset (Activity)', ]
    label = ['Scenario 2', 'Scenario 3', 'Scenario 4','Scenario 5', ]
    marker = ['o','v','^', '<', '>', '8', 's', 'p', '*']
    ax = plt.figure( figsize = (8, 4.5) ).gca( )
    ax.xaxis.set_major_locator( MaxNLocator( integer = True ) )
    for i in range( len(acc)):
        ax.plot(
                [1,2,3,4,5],
                acc[i],
                ms = 10, mew = 1,
                marker = marker[i],
                linewidth = 1,
                fillstyle = Line2D.fillStyles[-1],
                label = label[i]
                )
    fsize = 14
    ax.set_ylim( 0, 103 )
    ax.set_xlabel( 'The number of shots', fontsize = fsize )
    ax.set_ylabel( 'Accuracy(%)', fontsize = fsize )
    plt.grid( alpha = 0.2 )
    plt.legend()
    name = 'compare_domain_similarity'
    out = f'C:/Users/29073/iCloudDrive/PhD Research Files/Publications/One-Shot ' \
          f'learning/Results/results_figs/Paperfigure/' + name
    plt.savefig( out +'.pdf',bbox_inches='tight' )
def compare_tflearning():
    # a = loadmat( 'different_ft_test_classes.mat', squeeze_me = 1 )
    a = loadmat( 'different_ft_test_classes_76.mat', squeeze_me = 1 )
    keys,vals = [],[]
    for key,val in a.items():
        if key == '__header__' or key == '__version__' or key == '__globals__':
            continue
        keys.append(int(key.split('_')[-1].split('cls')[0]))
        vals.append(val)
    idx_sort = np.argsort(keys)
    keys = np.asarray(keys)[idx_sort]
    vals = np.asarray(vals)[idx_sort]
    ax = plt.figure( figsize = (8, 4.5) ).gca( )
    ax.xaxis.set_major_locator( MaxNLocator( integer = True ) )
    vals[-1] = vals[-1] #+ 5
    ax.plot(

            keys[1:len(keys)],
            vals[1:len(keys)],
            # keys,
            # acc,


            ms = 10, mew = 1,
            marker = 'o',
            linewidth = 1,
            fillstyle = Line2D.fillStyles[-1]
            )
    fsize = 14
    ax.set_ylim( 40, 100 )
    ax.set_xlabel( 'The number of tuning classes', fontsize = fsize )
    ax.set_ylabel( 'Accuracy(%)', fontsize = fsize )
    plt.grid( alpha = 0.2 )
    plt.show(block=True)
    name = 'compare_transfer_learning_2'
    out = f'C:/Users/29073/iCloudDrive/PhD Research Files/Publications/One-Shot ' \
          f'learning/Results/results_figs/Paperfigure/' + name
    plt.savefig( out + '.pdf', bbox_inches = 'tight' )
if __name__ == '__main__':
    # compare_tflearning()
    # wiar()
    # recordNew( 'widar' )
    recordNew( 'compare_Base' )
    # # recordNew('in_domain')
    # recordNew( 'crossenvir_user5' )
    # recordNew( 'crossenvir_user1234' )
    # recordNew( 'cross_domain_user_shots' )
    # multiRx( ID = 'user1' )
    # multiRx( ID = 'user2' )
    # multiRx( ID = 'user3' )
    # pltconverge()
    # pltimpact_cls( )
    # R1C1()
    # compareDomainSimilarity()
    # barChartNew('compareFT')
    # compare_tflearning()

