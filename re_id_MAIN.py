import tensorflow as tf
from PIL import Image
import os
import numpy as np
import random as rd
import time
from re_id_test import compute_rank
from re_id_test import duke_compute_rank
import scipy.misc
import scipy.io as spio
import math
import matplotlib.pyplot as plt
'''작성한 py 삽입'''
from aug_files.re_id_aug_generator import draw_per_augmenter_images

'''******************************************************'''
'''                    Parameters                        '''
'''******************************************************'''
n_classes = 2  # same or Not
num_training_data = 316   #199, 3484, 1940 , 추출해서 뽑고자 하는 갯수
nAug = 2  # 증분되는 갯수(본인포함)
num_total_training_data = num_training_data * nAug * 2   # 원본 포함 16종류의 augmentation, same 과 diff 둘다

''' validation set '''
num_val_set = 10
img_height = 128
img_width = 48
img_size_1D = img_height * img_width * 3
num_total_test_data = 316  # 100, 73, 486
n_epochs = 300  # the number of epochs
batch_size_train = 16

''' test samples 을 담을 곳 '''
m_test_sample_cam_a = np.zeros((num_total_test_data, img_size_1D))
m_test_sample_cam_b = np.zeros((num_total_test_data, img_size_1D))

''' Dropout ratio '''
# keep_rate : drop out ratio, weight를 사용할때 100% 사용하겠다(값이 1이면 dropout을 안하겠다는 의미)
keep_rate = tf.placeholder(tf.float32)

start_load_same_box = time.time()  # time start
'''******************************************************'''
'''             Load ( Training data )  [SAME]           '''
'''******************************************************'''
# ''' cam a 에 대해서 load '''
# # path_cam_a = "E:\\re_id_python_exp\\MCT_extracted_samples\\train_mct_dataset2\\set a"
# # path_cam_a = "E:\\re_id_python_exp\\CUHK01\\Data_test_101_200\\Training_data\\cam_a"
# path_cam_a = "E:\\re_id_python_exp\\VIPeR\Training_data\\same_a"
# # path_cam_a = "E:\\DB\\Re-ID dataset\\prid_450s\\cam_a"
# Box_training_sample_cam_a = np.zeros((num_training_data, img_height, img_width, 3))
# cnt_load = 0
# for f in os.listdir(path_cam_a):
#     temp_load_img = Image.open(os.path.join(path_cam_a, f))
#     temp = temp_load_img.load()
#     temp_pixel = np.zeros((temp_load_img.height, temp_load_img.width, 3))
#     for ff in range(temp_load_img.height):
#         for gg in range(temp_load_img.width):
#             temp_pixel[ff][gg][:] = temp[gg, ff]  # temp[x, y]
#     temp_pixel = scipy.misc.imresize(temp_pixel, (img_height, img_width))
#     Box_training_sample_cam_a[cnt_load] = temp_pixel
#     cnt_load += 1
#     temp_load_img.close()
# np.save('E:\\re_id_python_exp\\Box_training_sample_cam_a.npy', Box_training_sample_cam_a)
#
# ''' cam b 에 대해서 load '''
# # path_cam_b = "E:\\re_id_python_exp\\MCT_extracted_samples\\train_mct_dataset2\\set b"
# # path_cam_b = "E:\\re_id_python_exp\\CUHK01\\Data_test_101_200\\Training_data\\cam_b"
# path_cam_b = "E:\\re_id_python_exp\\VIPeR\Training_data\\same_b"
# # path_cam_b = "E:\\DB\\Re-ID dataset\\prid_450s\\cam_b"
# Box_training_sample_cam_b = np.zeros((num_training_data, img_height, img_width, 3))
# cnt_load = 0
# for f in os.listdir(path_cam_b):
#     temp_load_img = Image.open(os.path.join(path_cam_b, f))
#     temp = temp_load_img.load()
#     temp_pixel = np.zeros((temp_load_img.height, temp_load_img.width, 3))
#     for ff in range(temp_load_img.height):
#         for gg in range(temp_load_img.width):
#             temp_pixel[ff][gg][:] = temp[gg, ff]  # temp[x, y]
#     temp_pixel = scipy.misc.imresize(temp_pixel, (img_height, img_width))
#     Box_training_sample_cam_b[cnt_load] = temp_pixel
#     cnt_load += 1
#     temp_load_img.close()
# print('Training File Load 완료')
# np.save('E:\\re_id_python_exp\\Box_training_sample_cam_b.npy', Box_training_sample_cam_b)
#
# # ''' sample 저장된 npy 불러옴 '''
# # Box_training_sample_cam_a = np.load('E:\\re_id_python_exp\\Box_training_sample_cam_a.npy')
# # Box_training_sample_cam_b = np.load('E:\\re_id_python_exp\\Box_training_sample_cam_b.npy')
# end_load_txt = time.time()  # time start
# print('Training File Load 완료, 시간: ', end_load_txt-start_load_same_box)


'''******************************************************'''
'''                      Data shuffle                    '''
'''******************************************************'''
def data_shuffle(Box_training_sample_cam_a, Box_training_sample_cam_b, nData):
    # start_load_all_training = time.time()  # time start
    '''
        overfitting 방지와 학습을 위한 training data 의 무작위 셔플
        16 종류(원본포함) 의 augmentation 을 고려한 shuffling
    '''
    ''' training samples 을 담을 곳 '''
    m_training_sample_cam_a = np.zeros((nData, img_size_1D))  # same 과 different 두개의 class 를 가지므로
    m_training_sample_cam_b = np.zeros((nData, img_size_1D))
    ''' Training label setting '''
    labels_training = np.zeros((nData, 2))  # 학습 할 label 의 값을 넣는다.

    check_duple = [None]  # 어차피 None 은 반복 matching 에 영향을 끼치지 않으므로 size 1을 만들기 위해서 삽입

    ''' identity 별로 영상을 묶음 '''
    train_img_cam_a = np.zeros((num_training_data, nAug, img_size_1D))
    train_img_cam_b = np.zeros((num_training_data, nAug, img_size_1D))
    cnt_nAug = 0
    for i in range(num_training_data):  # identity 갯수
        train_img_cam_a[i] = np.reshape(Box_training_sample_cam_a[cnt_nAug:cnt_nAug+nAug, :, :, :], [nAug, img_size_1D])
        train_img_cam_b[i] = np.reshape(Box_training_sample_cam_b[cnt_nAug:cnt_nAug + nAug, :, :, :], [nAug, img_size_1D])
        cnt_nAug += nAug  # 증분

    samesame = 0
    notnot = 0
    check_contain_complete_same = 0
    check_contain_complete_NOT = 0
    cnt = 0
    switch_cnt = 1
    while True:
        if switch_cnt % 2 == 0:  # Same 추출, even number
            samesame += 1
            if samesame <= num_training_data:
                while True:
                    random_idx = rd.randrange(0, num_training_data)  # 전체 학습 데이터 중 무작위로 하나 추출 (0,미만)
                    check_v = 0
                    for f in range(check_duple.__len__()):
                        if check_duple[f] == random_idx:
                            check_v = -1  # 새로 뽑은 random 정수가 기존에 뽑았던 정수들과 모두 불일치(New one)
                    if check_v != -1:  # 지금 뽑은 random index가 안 녛은 sample 이다.
                        aug_check = [None]
                        temp_cnt = 0
                        while True:
                            aug_idx = rd.randrange(0, nAug)
                            # 이미 계산한 타입인지 확인
                            exist_aug_idx = 0
                            for aug_check_i in range(aug_check.__len__()):
                                if aug_check[aug_check_i] == aug_idx:
                                    exist_aug_idx = -1

                            if exist_aug_idx != -1: #  이전에 값이 존재한 적이 없다면
                                aug_check.append(aug_idx)
                                aug_idx_2 = rd.randrange(0, nAug)  # b 는 아무거나 뽑아도 무관
                                m_training_sample_cam_a[cnt + temp_cnt][:] = train_img_cam_a[random_idx][aug_idx][:] / 255
                                m_training_sample_cam_b[cnt + temp_cnt][:] = train_img_cam_b[random_idx][aug_idx_2][:] / 255
                                labels_training[cnt + temp_cnt, 1] = 1  # [0, 1]   SAME
                                temp_cnt += 1

                            if temp_cnt >= nAug:
                                # cnt += temp_cnt
                                break
                        # 중복 matrix 에 값 추가
                        check_duple.append(random_idx)
                        break
                if samesame == num_training_data:
                    check_contain_complete_same = 1
            else:
                samesame = samesame -1
        else:  # 'Not' category 추출, odd number
            notnot += 1
            if notnot <= num_training_data:
                temp_cnt = 0
                while True:
                    random_idx = rd.randrange(0, num_training_data)  # 전체 학습 데이터 중 무작위로 하나 추출 (0,미만
                    # 아무거나 범위 내 sample 추출
                    while True:
                        temp_random_idx_diff = rd.randrange(0, num_training_data)
                        aug_idx = rd.randrange(0, nAug)
                        if random_idx != temp_random_idx_diff:  #  Not 추출하기 위해서 index 가 다를 경우 인정
                            m_training_sample_cam_a[cnt + temp_cnt][:] = train_img_cam_a[random_idx][aug_idx][:] / 255
                            labels_training[cnt + temp_cnt, 0] = 1  # [1, 0]   NOT
                            m_training_sample_cam_b[cnt + temp_cnt][:] = train_img_cam_b[temp_random_idx_diff][aug_idx][:] / 255
                            temp_cnt += 1
                            break
                    if temp_cnt >= nAug:
                        break
                if notnot == num_training_data:
                    check_contain_complete_NOT = 1
            else:
                notnot = notnot -1
        #  두 sample 다 원하는 갯수가 채워지면 break
        if check_contain_complete_same == 1 and check_contain_complete_NOT == 1:
            break
        else:
            cnt = (samesame + notnot) * nAug
        switch_cnt += 1
    return m_training_sample_cam_a, m_training_sample_cam_b, labels_training


start_load_test_time = time.time()
'''******************************************************'''
'''                 Load ( Test data )                   '''
'''******************************************************'''
''' cam a 에 대해서 load '''
path_cam_a = "./Data_test_101_200/Test_data/cam_a"
cnt_load = 0
for f in os.listdir(path_cam_a):
    Load_test_image_cam_a = Image.open(os.path.join(path_cam_a, f))
    temp = Load_test_image_cam_a.load()
    temp_pixel = np.zeros((Load_test_image_cam_a.height, Load_test_image_cam_a.width, 3))
    for ff in range(Load_test_image_cam_a.height):
        for gg in range(Load_test_image_cam_a.width):
            temp_pixel[ff][gg][:] = temp[gg, ff]  # temp[x, y]
    temp_pixel = scipy.misc.imresize(temp_pixel, (img_height, img_width))
    m_test_sample_cam_a[cnt_load][:] = np.reshape(temp_pixel, [img_size_1D]) / 255
    cnt_load += 1
    Load_test_image_cam_a.close()

''' cam b 에 대해서 load '''
path_cam_b = "./Data_test_101_200/Test_data/cam_b"
cnt_load = 0
for f in os.listdir(path_cam_b):
    Load_test_image_cam_b = Image.open(os.path.join(path_cam_b, f))
    temp = Load_test_image_cam_b.load()
    temp_pixel = np.zeros((Load_test_image_cam_b.height, Load_test_image_cam_b.width, 3))
    for ff in range(Load_test_image_cam_b.height):
        for gg in range(Load_test_image_cam_b.width):
            temp_pixel[ff][gg][:] = temp[gg, ff]  # temp[x, y]
    temp_pixel = scipy.misc.imresize(temp_pixel, (img_height, img_width))
    m_test_sample_cam_b[cnt_load][:] = np.reshape(temp_pixel, [img_size_1D]) / 255
    cnt_load += 1
    Load_test_image_cam_b.close()
''' npy 로 저장 '''
# np.save('E:\\re_id_python_exp\\test_data_a.npy', m_test_sample_cam_a)
# np.save('E:\\re_id_python_exp\\test_data_b.npy', m_test_sample_cam_b)

# ''' 시간 관계상 미리 npy 로 저장했다가 불러옴 '''
# m_test_sample_cam_a = np.load('E:\\re_id_python_exp\\test_data_a.npy')
# m_test_sample_cam_b = np.load('E:\\re_id_python_exp\\test_data_b.npy')
end_load_txt = time.time()
print(' test samples 불러옴, 시간: ', end_load_txt-start_load_test_time)


'''******************************************************'''
'''                    Tensor 선언                       '''
'''******************************************************'''
''' 2개 Input'''
x_a = tf.placeholder('float', shape=[None, img_size_1D])  # 학습에 필요한 input image 를 1D vector 형태로 matrix 에 담는다.
x_b = tf.placeholder('float', shape=[None, img_size_1D])  # 학습에 필요한 input image 를 1D vector 형태로 matrix 에 담는다.
''' 2개 label 의 Output'''
y = tf.placeholder('float')  # Label matrix 를 담는다.   Re-ID 이므로 Same or Not 의 두가지 카테고리

''' Batch Normalization '''
phase_train = tf.placeholder('float')

''' Test label setting '''
labels_test = np.zeros((num_total_test_data, 2))  # 학습 할 label 의 값을 넣는다.


'''******************************************************'''
'''               conv2d, maxpooling                     '''
'''******************************************************'''
def conv2d_1S(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv2d_5S(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 5, 5, 1], padding='VALID')

def conv2d_2S(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


'''******************************************************'''
'''                    CNN 구조                          '''
'''******************************************************'''
def convolutional_neural_network(x, y, phase_train, keep_rate):
    '''
    < Filter weight 와 bias 선언 >
    To apply the layer, we first reshape x to a 4d tensor, with the second and third dimensions corresponding to
    image width and height, and the final dimension corresponding to the number of color channels.
    The first two dimensions are the patch size, the next is the number of input channels, and the last is the number
    of output channel
    '''

    W_conv1 = tf.get_variable("W_conv1", shape=[5, 5, 3, 32], initializer=tf.contrib.layers.xavier_initializer())
    W_conv2 = tf.get_variable("W_conv2", shape=[5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
    W_conv3 = tf.get_variable("W_conv3", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    W_conv4 = tf.get_variable("W_conv4", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())

    b_conv1 = tf.get_variable("b_conv1", shape=[32], initializer=tf.contrib.layers.xavier_initializer())
    b_conv2 = tf.get_variable("b_conv2", shape=[64], initializer=tf.contrib.layers.xavier_initializer())
    b_conv3 = tf.get_variable("b_conv3", shape=[128], initializer=tf.contrib.layers.xavier_initializer())
    b_conv4 = tf.get_variable("b_conv4", shape=[128], initializer=tf.contrib.layers.xavier_initializer())

    ''' 영상을 다시 계산하기 위해서 Reshape '''
    x = tf.reshape(x, shape=[-1, img_height, img_width, 3])  # input image 의 matrix 형태 i x j x d
    y = tf.reshape(y, shape=[-1, img_height, img_width, 3])
    ''' CNN 적용 '''
    '''conv 1 : shared weight'''
    result_conv_1_x = conv2d_1S(x, W_conv1) + b_conv1
    result_conv_1_y = conv2d_1S(y, W_conv1) + b_conv1
    conv_1_x = tf.nn.relu(result_conv_1_x)
    conv_1_y = tf.nn.relu(result_conv_1_y)
    '''conv 2 : shared weight'''
    result_conv_2_x = conv2d_1S(conv_1_x, W_conv2) + b_conv2
    result_conv_2_y = conv2d_1S(conv_1_y, W_conv2) + b_conv2
    conv_2_x = tf.nn.relu(result_conv_2_x)
    conv_2_y = tf.nn.relu(result_conv_2_y)
    '''Maxpooling'''
    conv_2_x = maxpool2d(conv_2_x)
    conv_2_y = maxpool2d(conv_2_y)
    '''conv 3 : shared weight'''
    result_conv_3_x = conv2d_1S(conv_2_x, W_conv3) + b_conv3
    result_conv_3_y = conv2d_1S(conv_2_y, W_conv3) + b_conv3
    conv_3_x = tf.nn.relu(result_conv_3_x)
    conv_3_y = tf.nn.relu(result_conv_3_y)
    '''conv 4 : shared weight'''
    result_conv_4_x = conv2d_1S(conv_3_x, W_conv4) + b_conv4
    result_conv_4_y = conv2d_1S(conv_3_y, W_conv4) + b_conv4
    conv_4_x = tf.nn.relu(result_conv_4_x)
    conv_4_y = tf.nn.relu(result_conv_4_y)
    '''Maxpooling'''
    conv_4_x = maxpool2d(conv_4_x)
    conv_4_y = maxpool2d(conv_4_y)

    ''' 관계 matrix '''
    m_XY = relationship_M(conv_4_x, conv_4_y)
    m_YX = relationship_M(conv_4_y, conv_4_x)
    '''relationship matrix 각각 convolution'''
    W_mx_conv1 = tf.get_variable("W_mx_conv1", shape=[5, 5, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    W_mx_conv2 = tf.get_variable("W_mx_conv2", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    b_mx_conv1 = tf.get_variable("b_mx_conv1", shape=[128], initializer=tf.contrib.layers.xavier_initializer())
    b_mx_conv2 = tf.get_variable("b_mx_conv2", shape=[128], initializer=tf.contrib.layers.xavier_initializer())

    W_my_conv1 = tf.get_variable("W_my_conv1", shape=[5, 5, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    W_my_conv2 = tf.get_variable("W_my_conv2", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    b_my_conv1 = tf.get_variable("b_my_conv1", shape=[128], initializer=tf.contrib.layers.xavier_initializer())
    b_my_conv2 = tf.get_variable("b_my_conv2", shape=[128], initializer=tf.contrib.layers.xavier_initializer())

    ''' 각각 relationship matrix 에 대해서 convolution  5 '''
    result_conv_5_x = conv2d_5S(m_XY, W_mx_conv1) + b_mx_conv1
    conv_5_x = tf.nn.relu(result_conv_5_x)
    result_conv_5_y = conv2d_5S(m_YX, W_my_conv1) + b_my_conv1
    conv_5_y = tf.nn.relu(result_conv_5_y)
    ''' 각각 relationship matrix 에 대해서 convolution  6 '''
    result_conv_6_x = conv2d_1S(conv_5_x, W_mx_conv2) + b_mx_conv2
    conv_6_x = tf.nn.relu(result_conv_6_x)
    result_conv_6_y = conv2d_1S(conv_5_y, W_my_conv2) + b_my_conv2
    conv_6_y = tf.nn.relu(result_conv_6_y)
    '''Maxpooling'''
    conv_6_x = maxpool2d(conv_6_x)
    conv_6_y = maxpool2d(conv_6_y)

    ''' fully connected layers '''
    fc = tf.concat([conv_6_x, conv_6_y], 3)
    W_fc1 = tf.get_variable("W_fc1", shape=[11 * 1 * 256, 500], initializer=tf.contrib.layers.xavier_initializer())
    W_fc2 = tf.get_variable("W_fc2", shape=[500, n_classes], initializer=tf.contrib.layers.xavier_initializer())
    b_fc1 = tf.get_variable("b_fc1", shape=[500], initializer=tf.contrib.layers.xavier_initializer())
    b_fc2 = tf.get_variable("b_fc2", shape=[n_classes], initializer=tf.contrib.layers.xavier_initializer())

    fc = tf.reshape(fc, [-1, 11 * 1 * 256])  # (batch, feature)
    #  1 차 Fully connected layer
    fc = tf.nn.relu(tf.matmul(fc, W_fc1) + b_fc1)
    fc = tf.nn.dropout(fc, keep_rate)
    #  2 차 Fully connected layer
    fc = tf.matmul(fc, W_fc2) + b_fc2

    ''' Tensor board '''
    tf.summary.histogram("W_conv1", W_conv1)
    tf.summary.histogram("W_conv2", W_conv2)
    tf.summary.histogram("W_conv3", W_conv3)
    tf.summary.histogram("W_conv4", W_conv4)

    tf.summary.histogram("W_mx_conv1", W_mx_conv1)
    tf.summary.histogram("W_mx_conv2", W_mx_conv2)
    tf.summary.histogram("W_my_conv1", W_my_conv1)
    tf.summary.histogram("W_my_conv2", W_my_conv2)


    return fc, conv_4_x, conv_4_y

'''******************************************************'''
'''                Relationship matrix                   '''
'''******************************************************'''
def relationship_M(a, b):
    '''
    관계도를 만드는 function
    Input  : a, b 는 (batch, i, j, d)
    output : 결합된 하나의 matrix 이며 shape = (batch, 5*(i-4), 5*(j-4), d)
    '''

    p_a = a[:, 2, 2, :]
    p_a = tf.expand_dims(p_a, axis=1)
    p_a = tf.expand_dims(p_a, axis=1)
    m_a = p_a - b[:, 0:5, 0:5, :]

    cnt_w = 1
    for w in range(3, a.shape.dims[2].value-2):
        p_a = a[:, 2, w, :]
        p_a = tf.expand_dims(p_a, axis=1)
        p_a = tf.expand_dims(p_a, axis=1)
        temp_m_a = p_a - b[:, 0:5, cnt_w:cnt_w+5, :]
        m_a = tf.concat([m_a, temp_m_a], 1)  # height 에 이어 붙인다.
        cnt_w += 1

    cnt_h = 1
    for h in range(3, a.shape.dims[1].value-2):
        cnt_w = 0
        for w in range(2, a.shape.dims[2].value-2):
            p_a = a[:, h, w, :]
            p_a = tf.expand_dims(p_a, axis=1)
            p_a = tf.expand_dims(p_a, axis=1)
            temp_m_a = p_a - b[:, cnt_h: cnt_h+5, cnt_w:cnt_w+5, :]
            m_a = tf.concat([m_a, temp_m_a], 1)  # height 에 이어 붙인다.
            cnt_w += 1
        cnt_h += 1
    m_a = tf.reshape(m_a, shape=[-1, 5*(a.shape.dims[1].value-4), 5*(a.shape.dims[2].value-4), 128])  # 5x5 이므로 -4

    return m_a


'''******************************************************'''
'''                    NN model 학습                     '''
'''******************************************************'''
def train_neural_network(x_a, x_b, y, keep_rate, phase_train):
    # 전체 framework 에서 output
    score_matrix, r_a, r_b = convolutional_neural_network(x_a, x_b, phase_train, keep_rate)

    #  softmax_cross_entropy 가 내부에 softmax 가 있으므로 softmax 의 output 을 여기 logits 로 input 금지
    softmax_result = tf.nn.softmax_cross_entropy_with_logits(logits=score_matrix, labels=y)
    total_loss = tf.reduce_mean(softmax_result)

    #  The update rule for variable with gradient g uses an optimization described at the end of section2 of the paper
    L_rate = 0.0001  # 초기 rate
    check_L = 0
    optimizer = tf.train.AdamOptimizer(learning_rate=L_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(total_loss)

    # cycle of feed forward and back propagation
    start_training_time = time.time()  # time start

    # gpu memory full open
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    tf.summary.histogram("network_output", score_matrix)
    tf.summary.scalar("loss", total_loss)
    merged = tf.summary.merge_all()

    w_saver = tf.train.Saver()  # 모든 변수의 저장과 복구를 위한 오퍼레이션 추가
    with tf.Session(config=config) as sess:
        w_saver.restore(sess, "./model_deep/model_re_id_deep.ckpt")
        print("Model restored.")

        # 데이터 학습
        # sess.run(tf.initialize_all_variables())  # 초기화
        # writer = tf.summary.FileWriter("./board/v1", sess.graph)
        # box_loss = []
        # box_valid = []
        # box_acc = []
        # test_acc = []
        # save_r1 = []
        # acum = []  # accumulated_rank
        # for epoch in range(n_epochs):
        #     epoch_loss = 0  # epoch_x는 (batch_size x 784) ndarray , epoch_y는 127 x 10 으로 class 가 들어있는 matrix
        #     self_accuracy = 0
        #     cnt_batch = 0
        #
        #     aug_train_cam_a = draw_per_augmenter_images(Box_training_sample_cam_a)
        #     aug_train_cam_b = draw_per_augmenter_images(Box_training_sample_cam_b)
        #
        #     train_aug_shuffle_a, train_aug_shuffle_b, labels_training = data_shuffle(aug_train_cam_a, aug_train_cam_b, num_total_training_data)
        #
        #     ''' random 하게 섞기 '''
        #     random_idx = np.arange(0, np.size(labels_training, axis=0) - num_val_set)
        #     np.random.shuffle(random_idx)
        #     real_train_aug_shuffle_a = train_aug_shuffle_a[random_idx]
        #     real_train_aug_shuffle_b = train_aug_shuffle_b[random_idx]
        #     real_labels_training = labels_training[random_idx]
        #     num_real_train_data = np.size(real_labels_training, axis=0)
        #
        #     print('same 갯수 : %d' % np.sum(real_labels_training[:, 1], axis=0))
        #     print('differ 갯수 : %d' % np.sum(real_labels_training[:, 0], axis=0))
        #
        #     num_iter = int((num_real_train_data - num_val_set) / batch_size_train)  # 뽑힌 데이터에 따라서 그때마다 iteration 갯수가 다름
        #     for f_iter in range(num_iter):  # 전체 same sample 갯수 / epoch 횟수
        #         rt_softmax, rt_c, rt_tx = sess.run([score_matrix, r_a, r_b],
        #                                            feed_dict={x_a: real_train_aug_shuffle_a[cnt_batch:cnt_batch+batch_size_train],
        #                                                       x_b: real_train_aug_shuffle_b[cnt_batch:cnt_batch+batch_size_train],
        #                                                       phase_train: 1,
        #                                                       keep_rate: 0.5})
        #
        #         _, c = sess.run([optimizer, total_loss], feed_dict={x_a: real_train_aug_shuffle_a[cnt_batch:cnt_batch+batch_size_train],
        #                                                             x_b: real_train_aug_shuffle_b[cnt_batch:cnt_batch+batch_size_train],
        #                                                             y: real_labels_training[cnt_batch:cnt_batch+batch_size_train],
        #                                                             phase_train: 1,
        #                                                             keep_rate: 0.5})
        #         # 학습이 끝난 후 training data 에 대해서 self accuracy 측정
        #         for g in range(batch_size_train):
        #             if rt_softmax[g][0] <= rt_softmax[g][1]:  # SAME [0, 1]
        #                 check_label_0 = 0
        #                 check_label_1 = 1  # SAME
        #             else:
        #                 check_label_0 = 1
        #                 check_label_1 = 0  # NOT
        #             if check_label_0 == real_labels_training[cnt_batch + g][0] and check_label_1 == real_labels_training[cnt_batch + g][1]:
        #                 self_accuracy += 1  # accuracy 누적
        #
        #         # 증분
        #         epoch_loss = epoch_loss + c
        #         cnt_batch += batch_size_train
        #
        #     '''******************************************************'''
        #     '''        학습된 weights 저장 (계속 덮어씌우자)         '''
        #     '''******************************************************'''
        #     save_path = w_saver.save(sess, "E:\\re_id_python_exp\\model_deep\\model_re_id_deep.ckpt")
        #     print("model save in file : %s" % save_path)
        #
        #     ''' validation set accuracy '''
        #     val_s, val_a, val_b = sess.run([score_matrix, r_a, r_b],
        #                                    feed_dict={x_a: train_aug_shuffle_a[num_real_train_data-num_val_set:],
        #                                               x_b: train_aug_shuffle_b[num_real_train_data-num_val_set:],
        #                                               phase_train: 0,
        #                                               keep_rate: 1.0})
        #     val_correct = 0
        #     for f_val in range(num_val_set):
        #         if val_s[f_val][0] <= val_s[f_val][1]:  # SAME [0, 1], ViPeR test dataset 은 316개 same 을 넣었으므로
        #             cc = 1
        #         else:
        #             cc = 0
        #
        #         if cc == labels_training[num_real_train_data-num_val_set + f_val, 1]:
        #             val_correct += 1
        #
        #     box_loss.append(epoch_loss / num_iter)
        #     box_valid.append(val_correct / num_val_set)
        #     box_acc.append(self_accuracy/(num_real_train_data-num_val_set))
        #     print('valid_accuracy:', float(val_correct / num_val_set))
        #     print('training_accuracy:\t', float(self_accuracy/(num_real_train_data-num_val_set)))
        #     print('Epoch', epoch, 'completed out of', n_epochs, '각 sample 평균 loss:', epoch_loss / num_iter)
        #
        #     # '''Tensorboard value'''
        #     # result_merged = sess.run(merged, feed_dict={x_a: real_train_aug_shuffle_a[cnt_batch:cnt_batch+batch_size_train],
        #     #                                             x_b: real_train_aug_shuffle_b[cnt_batch:cnt_batch+batch_size_train],
        #     #                                             y: real_labels_training[cnt_batch:cnt_batch+batch_size_train],
        #     #                                             phase_train: 1,
        #     #                                             keep_rate: 0.5})
        #     # writer.add_summary(result_merged, epoch)
        #
        #     if (self_accuracy/(num_real_train_data-num_val_set)) >= 0.9 and check_L == 0: #  높은 accuracy 달성되면 rate 절반 감소
        #         L_rate = L_rate / 2
        #         check_L = 1
        #     if (self_accuracy/(num_real_train_data-num_val_set)) >= 0.95 and check_L == 1:  # 높은 accuracy 달성되면 rate 절반 감소
        #         L_rate = L_rate / 2
        #         check_L = 2
        #
        #     # if epoch % 10 == 0 or (epoch_loss / num_iter) < 0.01: # 10 의 배수 일때마다 한번씩 계산
        #     if epoch % 10 == 0:  # 10 의 배수 일때마다 한번씩 계산
        #         print('************ rank 테스트 시작 ************')
        #         start_rank_time = time.time()  # time start
        #         rank_score_box = np.zeros((num_total_test_data, num_total_test_data))
        #         ''' 테스트 GPU 병렬 처리를 위해서 '''
        #         temp_copy = np.zeros((num_total_test_data, img_size_1D))
        #         for f_rank in range(num_total_test_data):
        #             for g in range(num_total_test_data):
        #                 temp_copy[g][:] = m_test_sample_cam_a[f_rank][:]
        #             sr_pair, rt_c, rt_tx = sess.run([score_matrix, r_a, r_b],
        #                                             feed_dict={x_a: temp_copy, x_b: m_test_sample_cam_b,
        #                                                        phase_train: 0,
        #                                                        keep_rate: 1.0})
        #             # [0, 1] 클래스가 SAME 이므로
        #             for g in range(num_total_test_data):
        #             #     # 자릿수에 의한 동률 방지로 *2 증분
        #                 rank_score_box[f_rank][g] = sr_pair[g][1]  # 닮을 확률(0,1)클래스만 추출해서 담음.
        #         # 만들어진 similarity matrix rank 계산
        #         rr, acu = compute_rank(rank_score_box, epoch)
        #         save_r1.append(rr[0])  # rank 1 누적
        #         acum.append(acu[0])
        #         end_rank_time = time.time()  # time start
        #         print('RANK 테스트에 걸린 총 시간:\t', end_rank_time - start_rank_time)
        #         print('rank1 : ', save_r1)
        #         print('accumulated1 : ', acum)
        #
        #     print('************ accuracy 테스트 시작 ************')
        #     start_test_time = time.time()  # time start
        #     sr, rt_c, rt_tx = sess.run([score_matrix, r_a, r_b],
        #                                feed_dict={x_a: m_test_sample_cam_a, x_b: m_test_sample_cam_b,
        #                                           phase_train: 0,
        #                                           keep_rate: 1.0})
        #     sum_correct = 0
        #     for f_acc in range(num_total_test_data):
        #         if sr[f_acc][0] <= sr[f_acc][1]:  # SAME [0, 1], ViPeR test dataset 은 316개 same 을 넣었으므로
        #             number_correct = 1
        #         else:
        #             number_correct = 0
        #         sum_correct = sum_correct + number_correct
        #     print('Accuracy:', float(sum_correct / num_total_test_data))
        #     end_test_time = time.time()  # time start
        #     print('ACC 테스트에 걸린 총 시간:\t', end_test_time - start_test_time)
        #     print('')
        #     test_acc.append(float(sum_correct / num_total_test_data))
        #
        # end_training_time = time.time()  # time start
        # print('학습에 걸린 총 시간:\t', end_training_time - start_training_time)
        # '''******************************************************'''
        # '''                   학습된 weights 저장                '''
        # '''******************************************************'''
        # save_path = w_saver.save(sess, "E:\\re_id_python_exp\\model_deep\\model_re_id_deep.ckpt")
        # print("model save in file : %s" % save_path)
        #
        # plt.figure(1)
        # plt.plot(box_loss)
        # plt.figure(2)
        # plt.title('Red=validation, Blue=whole')
        # plt.plot(box_valid, 'r', box_acc, 'b')
        # plt.figure(3)
        # plt.plot(box_acc, 'b')
        # plt.figure(4)
        # plt.plot(test_acc)
        # print('stop')


        '''******************************************************'''
        '''               accuracy 테스트 시작                   '''
        '''******************************************************'''
        print('\n ************ accuracy 테스트 시작 ************')
        start_test_time = time.time()  # time start
        sr, rt_c, rt_tx = sess.run([score_matrix, r_a, r_b],
                                   feed_dict={x_a: m_test_sample_cam_a, x_b: m_test_sample_cam_b,
                                              phase_train: 0,
                                              keep_rate: 1.0})
        sum_correct = 0
        for f in range(num_total_test_data):
            if sr[f][0] <= sr[f][1]:  # SAME [0, 1], ViPeR test dataset 은 316개 same 을 넣었으므로
                number_correct = 1
            else:
                number_correct = 0
            sum_correct = sum_correct + number_correct
        print('Accuracy:', float(sum_correct / num_total_test_data))
        end_test_time = time.time()  # time start
        print('ACC 테스트에 걸린 총 시간:\t', end_test_time - start_test_time)

        '''******************************************************'''
        '''                   rank 테스트 시작                   '''
        '''******************************************************'''
        print('\n ************ rank 테스트 시작 ************')
        start_rank_time = time.time()  # time start
        rank_score_box = np.zeros((num_total_test_data, num_total_test_data))
        ''' 테스트 GPU 병렬 처리를 위해서 '''
        temp_copy = np.zeros((num_total_test_data, img_size_1D))
        for f in range(num_total_test_data):
            for g in range(num_total_test_data):
                temp_copy[g][:] = m_test_sample_cam_a[f][:]
            sr_pair, rt_c, rt_tx = sess.run([score_matrix, r_a, r_b],
                                            feed_dict={x_a: temp_copy, x_b: m_test_sample_cam_b,
                                                       phase_train: 0,
                                                       keep_rate: 1.0})

            # [0, 1] 클래스가 SAME 이므로
            for g in range(num_total_test_data):
            #     # 자릿수에 의한 동률 방지로 *2 증분
                rank_score_box[f][g] = sr_pair[g][1]  # 닮을 확률(0,1)클래스만 추출해서 담음.(안닮은 확률은 '1-닮음' 이므로 무시)
        end_rank_time = time.time()  # time start
        print('RANK 테스트에 걸린 총 시간:\t', end_rank_time - start_rank_time)
        # 만들어진 similarity matrix rank 계산
        rr, acu = compute_rank(rank_score_box, 9999)
        print('rank1 : ', rr)
        print('accumulated1 : ', acu)
        print('stop')


'''******************************************************'''
'''               Execution part (Main)                  '''
'''******************************************************'''
if __name__ == "__main__":  #is used to execute some code only if the file was run directly, and not imported.
    train_neural_network(x_a, x_b, y, keep_rate, phase_train)
    print("\n main_re_id.py 가 직접 실행 \n")
else:
    print(" \n main_re_id.py 가 import 되어 사용됨 \n") # main_re_id.py 가 import 되어 사용됨