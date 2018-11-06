import tensorflow as tf
from PIL import Image
import os
import numpy as np
from cumulative_figure import cumulative_rank
import random as rd
import time


'''******************************************************'''
'''                     Rank 계산 file                   '''
'''******************************************************'''
def compute_rank(rank_score_box, epoch):
    ''' rank 별 정확도와 CMC 를 위함 '''
    num_test_data = rank_score_box[0].size
    rank_index_box = np.zeros((num_test_data, num_test_data))  # (i,j) = (cam a, cam b) 에 해당되는 index
    check_rank = np.zeros(num_test_data)
    accumulated_rank = np.zeros(num_test_data)
    temp=[]
    ''' 각 row 별로 값이 큰 순서대로 index 를 정렬 '''
    for i in range(num_test_data):
        temp = rank_score_box[i][:]
        temp_index_sorted = sorted(range(len(temp)), key=lambda k: temp[k])  # 오름차순으로 index 정렬
        temp_index_sorted.reverse()  # 보기 편하게 내림차순으로 다시 정렬
        rank_index_box[i][:] = temp_index_sorted[:]

    ''' cam A 와 실제로 같은 probe rank 몇에 있는지 check '''
    check_rank = np.int32(check_rank)
    rank_index_box = np.int32(rank_index_box)
    for i in range(num_test_data):
        for j in range(num_test_data):
            if rank_index_box[i][j] == i:
                check_rank[j] = check_rank[j] +1 # 올바른 re-id 몇 번째 rank 에 속하는지 확인 후 증분

    ''' rank print '''
    cnt = 0
    name_w = 'epoch_%d.txt' % epoch
    file_w = open(name_w, 'w')
    for i in range(np.int32(num_test_data/10)):
        rank_w = '%d  %d  %d  %d  %d  %d  %d  %d  %d  %d' % (check_rank[cnt], check_rank[cnt+1], check_rank[cnt+2], check_rank[cnt+3],
                                                              check_rank[cnt + 4], check_rank[cnt+5], check_rank[cnt+6], check_rank[cnt+7],
                                                              check_rank[cnt + 8], check_rank[cnt+9])
        print(rank_w, file=file_w)
        cnt += 10
    file_w.close()

    ''' 추출된 rank 결과 값으로 accumulation vector 구성 '''
    for i in range(num_test_data):
        if i == 0:
            accumulated_rank[0] = check_rank[0]
        else:
            accumulated_rank[i] = accumulated_rank[i-1] + check_rank[i]

    cnt = 0
    name_p = 'P_epoch_%d.txt' % epoch
    file_p = open(name_p, 'w')
    accumulated_rank = accumulated_rank / num_test_data
    for i in range(np.int32(num_test_data / 10)):
        rank_p = '%.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f' % (
        accumulated_rank[cnt], accumulated_rank[cnt + 1], accumulated_rank[cnt + 2], accumulated_rank[cnt + 3],
        accumulated_rank[cnt + 4], accumulated_rank[cnt + 5], accumulated_rank[cnt + 6], accumulated_rank[cnt + 7],
        accumulated_rank[cnt + 8], accumulated_rank[cnt + 9])
        print(rank_p, file=file_p)
        cnt += 10
    file_p.close()

    return check_rank, accumulated_rank


'''******************************************************'''
'''  Rank 계산 ( dukeMCMT ) (파일 이름으로 매칭할 경우)  '''
'''******************************************************'''
def duke_compute_rank(rank_score_box, m_test_id_cam_prob, m_test_id_cam_gal, epoch):
    ''' rank 별 정확도와 CMC 를 위함 '''
    num_test_data_prob = rank_score_box.shape[0]
    num_test_data_gall = rank_score_box.shape[1]
    rank_index_box = np.zeros((num_test_data_prob, num_test_data_gall), dtype=int)  # (i,j) = (cam a, cam b) 에 해당되는 index
    rank_id_box = np.zeros((num_test_data_prob, num_test_data_gall), dtype=int)
    check_rank = np.zeros(num_test_data_gall, dtype=int)
    accumulated_rank = np.zeros(num_test_data_gall)
    temp = []
    ''' 각 row 별로 값이 큰 순서대로 index 를 정렬 '''
    for i in range(num_test_data_prob):
        temp = rank_score_box[i][:]
        temp_index_sorted = sorted(range(len(temp)), key=lambda k: temp[k])  # 오름차순으로 index 정렬
        temp_index_sorted.reverse()  # 보기 편하게 내림차순으로 다시 정렬
        rank_index_box[i][:] = temp_index_sorted[:]
        rank_id_box[i][:] = m_test_id_cam_gal[temp_index_sorted]

    ''' cam A 와 실제로 같은 probe rank 몇에 있는지 check '''
    check_rank = np.int32(check_rank)
    rank_index_box = np.int32(rank_index_box)
    for i in range(num_test_data_prob):
        for j in range(num_test_data_gall):
            if rank_id_box[i][j] == m_test_id_cam_prob[i]:
                check_rank[j] += 1  # 올바른 re-id 몇 번째 rank 에 속하는지 확인 후 증분
                break

    ''' rank print '''
    cnt = 0
    name_w = 'epoch_%d.txt' % epoch
    file_w = open(name_w, 'w')
    for i in range(np.int32(num_test_data_gall/10)):
        rank_w = '%d  %d  %d  %d  %d  %d  %d  %d  %d  %d' % (check_rank[cnt], check_rank[cnt+1], check_rank[cnt+2], check_rank[cnt+3],
                                                              check_rank[cnt + 4], check_rank[cnt+5], check_rank[cnt+6], check_rank[cnt+7],
                                                              check_rank[cnt + 8], check_rank[cnt+9])
        print(rank_w, file=file_w)
        cnt += 10
    file_w.close()

    ''' 추출된 rank 결과 값으로 accumulation vector 구성 '''
    for i in range(num_test_data_gall):
        if i == 0:
            accumulated_rank[0] = check_rank[0]
        else:
            accumulated_rank[i] = accumulated_rank[i-1] + check_rank[i]

    cnt = 0
    name_p = 'P_epoch_%d.txt' % epoch
    file_p = open(name_p, 'w')
    accumulated_rank = accumulated_rank / num_test_data_prob
    for i in range(np.int32(num_test_data_gall / 10)):
        rank_p = '%.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f' % (
        accumulated_rank[cnt], accumulated_rank[cnt + 1], accumulated_rank[cnt + 2], accumulated_rank[cnt + 3],
        accumulated_rank[cnt + 4], accumulated_rank[cnt + 5], accumulated_rank[cnt + 6], accumulated_rank[cnt + 7],
        accumulated_rank[cnt + 8], accumulated_rank[cnt + 9])
        print(rank_p, file=file_p)
        cnt += 10
    file_p.close()

    return check_rank, accumulated_rank