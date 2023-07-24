import gym
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
import numpy as np
from copy import deepcopy
import math
import csv
import os, shutil, random
import cv2
import time
import torch

class SACEnv(gym.Env):

    # 构造函数，搭建环境
    def __init__(self, database,
                n_agents = 4,
                output_Metric = False, train=True, full_random=False,
                max_episode=4, max_step=15, print_progress=False) -> None:
        # TODO: train and test
        start = time.time()
        self.n_agents = n_agents
        self.output_Metric = output_Metric      # 输出每个回合的指标到文件
        self.train = train                      # 当前是不是训练
        self.full_random = full_random          # 是不是不管当前的操作，每次操作都随机选（模拟ransac）
        self.database = database

        self.pt_num = 100                       # 总点数
        self.pt_channel = 8                    # 每个点的特征数，5+3=8
        self.action_size = 8                    # 需要选出几个点，求基础矩阵要7个点
        self.dis_thr = 0.1                      # 判断内点的阈值，±dis_thr
        self.step_count = 0                     # 统计这次回合中运行的步数
        self.episode_count = 0                  # 统计这对图像运行了多少回合
        self.data_count = 0                     # 统计本环境用了多少图片
        self.total_step_count = 0               # 统计本环境不论图像对，一共运行了多少步
        self.reward_min_threshold = 0.05        # 少于5%的点就退出
        self.reward_max_threshold = 0.5         # 多于50%的点就退出
        self.max_step = max_step                # 一个回合中，最多走多少步
        self.max_episode = max_episode          # 一对图片最多用多少回合
        # 如果是train，则一张图片只用一个回合
        if (train): self.max_episode = 1
        # self.used_pair = []                     # 记录用过的图像对

        self.is_async = False
        self.min_step = 99999999
        self.min_angle_error = 99999999
        self.angle_error_thr = 0.5
        self.print_progress = print_progress


        self.GT_R_Rel = None
        self.GT_t_Rel = None

        self.inliers = None
        self.outliers = None

        self.old_angle_error = 99999
        self.min_angle_error_episode = 99999
        self.angle_error_increase_count = 0
        self.angle_error_increase_thr = 5
        self.max_inliers_episode = 0
        self.fix_count = 0
        self.best_res = None

        self.imgpair_idx = 0    # 第00个轨迹中的图片

        # self.state = self.reset()
        # self.info['env_id'] = 99
        # self.render()
        self.old_inliers = np.array([])
        self.state = [np.zeros((self.pt_num,self.pt_channel), dtype=np.float32) for _ in range(self.n_agents)]
        self.state_global = np.zeros((self.pt_num,self.pt_channel), dtype=np.float32)
        # TODO: two dim state_size?
        self.state_size = self.pt_num * self.pt_channel
        self.next_state = [np.zeros((self.pt_num,self.pt_channel), dtype=np.float32) for _ in range(self.n_agents)]

        self.action_pts1 = [[] for _ in range(self.n_agents)]
        self.action_pts2 = [[] for _ in range(self.n_agents)]
        self.all_inliers = [[] for _ in range(self.n_agents)]
        self.inliers_ratio = [[] for _ in range(self.n_agents)]
        self.angle_error = [[] for _ in range(self.n_agents)]
        self.dt = [[] for _ in range(self.n_agents)]
        self.dR = [[] for _ in range(self.n_agents)]

        self.observation_space = Tuple(tuple([Box(
            low=np.zeros((self.pt_num,self.pt_channel), dtype=np.float32),
            high=np.ones((self.pt_num,self.pt_channel), dtype=np.float32),
            shape=(self.pt_num,self.pt_channel),
            dtype=np.float32
        ) for _ in range(self.n_agents)]))

        # 动作空间
        self.action_space = Tuple(tuple([MultiDiscrete([ self.pt_num,
                                            self.pt_num,
                                            self.pt_num,
                                            self.pt_num,
                                            self.pt_num,
                                            self.pt_num,
                                            self.pt_num,
                                            self.pt_num]
                                            ) for _ in range(self.n_agents)]))
        # self.data_count = 0
        # 删除结果输出文件夹
        if os.path.exists("result"):
            shutil.rmtree("result")
            os.mkdir("result")
        end = time.time()
        print("init cost: ", end - start)

    def seed(self, seed_num):
        # pass
        np.random.seed(seed_num)
        random.seed(seed_num)
        self.database.seed(seed_num)

    def reset_database(self, **kwargs):
        self.database.reset()

    def rand_select_row(self, array, num):  #array为需要采样的矩阵，dim_needed为想要抽取的行数
        row_total = array.shape[0]

        row_sequence= np.random.choice(row_total,num,replace=False, p=None)

        selected_points = array[row_sequence,:]

        unselected_points = []
        for i in range(row_total):
            if (i not in row_sequence):
                unselected_points.append(array[i,:])
        unselected_points = np.array(unselected_points)
        np.random.shuffle(selected_points)
        np.random.shuffle(unselected_points)
        return selected_points, unselected_points

    def padding_row(self, array, num):

        padding = np.array([-100]*array.shape[1]*(num-array.shape[0])).reshape((-1,array.shape[1]))
        if array.size != 0:
            padding_array = np.concatenate((array, padding))
        else:
            padding_array = padding
        return padding_array

    def addlable(self, array, label):
        labels = [label]*array.shape[0]
        labelarray = np.concatenate((array, np.array(labels).reshape((-1,1))), axis=1)

        # labelarray = np.concatenate((array[:,:position], np.array(labels).reshape((-1,1)), array[:,position:]), axis=1)
        # padding = np.array([-100]*array.shape[1]*(num-array.shape[0])).reshape((-1,array.shape[1]))
        # if array.size != 0:
        #     padding_array = np.concatenate((array, padding))
        # else:
        #     padding_array = padding
        return labelarray

    def addstatelable(self, array, state, statelabel):
        mask = np.isin(array[:,0], state[:,0])

        labels = np.zeros((mask.shape[0],1))
        labels[mask==True] = statelabel[0]
        labels[mask==False] = statelabel[1]
        labelarray = np.concatenate((array, labels), axis=1)
        return labelarray


    def calc_E_F(self, R, t, K1, K2):
        # construct the ground truth essential matrix from the ground truth relative pose
        gt_E = np.zeros((3,3))
        # gt_E[0, 1] = -float(t[2])
        # gt_E[0, 2] = float(t[1])
        # gt_E[1, 0] = float(t[2])
        # gt_E[1, 2] = -float(t[0])
        # gt_E[2, 0] = -float(t[1])
        # gt_E[2, 1] = float(t[0])

        gt_E[0, 1] = -float(t[2,0])
        gt_E[0, 2] = float(t[1,0])
        gt_E[1, 0] = float(t[2,0])
        gt_E[1, 2] = -float(t[0,0])
        gt_E[2, 0] = -float(t[1,0])
        gt_E[2, 1] = float(t[0,0])

        gt_E = np.matmul(gt_E, R)
        # fundamental matrix from essential matrix
        K1_inv = np.linalg.inv(K1)
        K2_inv = np.linalg.inv(K2)
        gt_F = np.matmul(np.matmul(K2_inv.T, gt_E), K1_inv)

        return gt_E, gt_F

    def reset(self, **kwargs):
        # 如果当前是新回合，则新获取一对图片
        if (self.episode_count == 0):
            pair_db = self.database.next_scene()

            self.sqs = pair_db['sqs']
            self.img1_id = pair_db['img1_id']
            self.img2_id = pair_db['img2_id']
            self.pair_id = self.img1_id + '-' + self.img2_id

            self.img1_fname = pair_db['img1_fname']
            self.img2_fname = pair_db['img2_fname']
            # self.img1 = cv2.imread(img1_fname)
            # self.img2 = cv2.imread(img2_fname)

            self.img1_K = pair_db['img1_K']
            self.img2_K = pair_db['img2_K']

            self.GT_E = pair_db['GT_E']
            self.GT_F = pair_db['GT_F']

            # img1_R表示从位置1到原点的旋转，
            # 要得到位置1到位置2的旋转，则如下计算
            self.GT_R_Rel = pair_db['GT_R_Rel']
            self.GT_t_Rel = pair_db['GT_t_Rel']

            # r3 = R.from_matrix(self.GT_R_Rel)
            # # qua = r3.as_quat()
            # euler_1 = r3.as_euler('zxy', degrees=True)

            # 读取特征点对
            self.all_points = pair_db['all_points']
            self.pts1_all = self.all_points[:,:2]
            self.pts2_all = self.all_points[:,2:4]

            # 储存一张图片所用过的点对
            # 不受回合影响,与图相关
            self.action_list = np.array([], dtype=np.int64).reshape((-1,self.action_size))


            # 把这场景中的指标存入文件
            if self.min_angle_error == 99999999:
                if os.path.exists("Metric.csv"):
                    os.remove("Metric.csv")

                    metrics = ['id','pair', 'inliers','min_angle_error','dR','dt','eposide','step','min_step']
                    with open("Metric.csv","a+") as csvfile: 
                        writer = csv.writer(csvfile)
                        writer.writerow(metrics)
                if os.path.exists("result"):
                    shutil.rmtree("result")
                    os.mkdir("result")
                # self.Metrics = {}
                # self.Metrics['valid'] = False
                # self.Metrics['F1'] = 0
                # self.Metrics['epi_inliers'] = 0
                # self.Metrics['epi_error'] = 99999
                # self.Metrics['dR'] = 99999
                # self.Metrics['dt'] = 99999
            if self.output_Metric and self.min_angle_error != 99999999 and self.min_angle_error != 99999 and self.step_count!=0:
                evaluation_result = [self.data_count-1,
                                    self.Metrics['pair'],
                                    self.Metrics['inliers'],
                                    self.Metrics['min_angle_error'],
                                    self.Metrics['dR'],
                                    self.Metrics['dt'],
                                    self.Metrics['eposide'],
                                    self.Metrics['step'],
                                    self.min_step]

                with open("Metric.csv","a+") as csvfile: 
                    writer = csv.writer(csvfile)
                    writer.writerow(evaluation_result)
            self.min_angle_error = 99999
            self.min_step = 99999
            self.Metrics = {}
            self.Metrics['pair'] = 99999
            self.Metrics['inliers'] = 0
            self.Metrics['min_angle_error'] = 99999
            self.Metrics['dR'] = 99999
            self.Metrics['dt'] = 99999
            self.Metrics['eposide'] = 99999
            self.Metrics['step'] = 99999
        
            # id+1，以下一次reset进入下一张
            self.data_count += 1
        
        

        self.old_inliers = np.array([])
        self.state = [np.zeros((self.pt_num,self.pt_channel), dtype=np.float32) for _ in range(self.n_agents)]
        for i in range(self.n_agents):
            # 生成一张图回合开始的时候的初始状态（xyxyd）
            self.state[i] = deepcopy(self.all_points)
            # 是否是当前选的操作点
            self.state[i] = self.addlable(self.state[i], -1)   # 倒数第四个位置插入
            # 是否已经选过
            self.state[i] = self.addlable(self.state[i], 0)    # 倒数第三个位置插入
            # 点到模型的误差
            self.state[i] = self.addlable(self.state[i], 100)   # 倒数第二个位置插入
            # # 点到模型的最优误差
            # self.state = self.addlable(self.state, 100)   # 倒数第一个位置插入

        self.prev_state = deepcopy(self.state)
        # 随机走第一步，实现随机采样
        self.state, _, _, _ = self.step(None)

        self.prev_state = deepcopy(self.state)
        self.next_state = np.zeros((self.n_agents, self.pt_num,self.pt_channel), dtype=np.float32)
        self.reward = 0
        self.done = False
        self.info = {}
        self.step_count = 0

        self.best_res = None
        self.old_angle_error = 99999
        self.min_angle_error_episode = 99999
        self.angle_error_increase_count = 0
        self.fix_count = 0
        self.max_inliers_episode = 0

        # 回合数+1
        self.episode_count += 1
        # 如果一对图片到达了最多的回合数，则要换一对图片了
        if (self.episode_count>=self.max_episode):
            self.episode_count = 0
            # self.imgpair_idx += 1
            # if (self.imgpair_idx ==len(self.vis_pairs)):
            #     self.imgpair_idx = 0
        # for point in self.pts0_all:
        #     cv2.circle(self.img0, (int(point[0]), int(point[1])), 2, 255, 4)
        # cv2.imwrite('test.png', self.img0)


        # 返回状态使用点
        return deepcopy(self.state)


    # 输入两个点在当前状态中的索引(int)，输出奖励、下一个状态（内点+外点）、是否结束、其他信息
    def single_step(self, action, agent_idx):
        
        if action is not None:
            action = action.cpu()
            self.step_count += 1    # 回合内步数+1
            self.total_step_count += 1

        if self.print_progress:
            print('id:' + str(self.data_count) + ' pair:' + self.pair_id + ' eposide:' + str(self.episode_count) + ' step:' + str(self.step_count))

        # assert (np.sort(self.state[:,:4],axis=0)==np.sort(self.prev_state[:,:4],axis=0)).all() == True


        if action is None:
            # 初始化，随机采样,也要判断是否重复,重复就重新随机

            idx = np.ones([1])
            n = 0
            while(idx.shape[0] != 0):
                action = np.sort(np.random.choice(self.pt_num, self.action_size, replace=False, p=None)).reshape((1,-1))
                idx = np.where((self.action_list==action).all(1))[0]
                n += 1
                if n > 1000:
                    break
            # 把选过的表记录下来
            self.action_list = np.concatenate((self.action_list, action))
            action = action.reshape((-1))
            unique_action = np.unique(action)

        else:
            action = action.reshape(-1)
            unique_action = np.unique(action)
            if (unique_action.size != self.action_size):
                self.inliers = np.array([])
                self.outliers = deepcopy(self.state[agent_idx])
            else:
                action = np.sort(action)
                self.action_list = np.concatenate((self.action_list, action.reshape((1,-1))), axis=0)


        if self.full_random:
            action = np.sort(np.random.choice(self.pt_num, self.action_size, replace=False, p=None))
        self.action_points = self.state[agent_idx][action,:]
        action_pts1 = self.action_points[:,:2]
        action_pts2 = self.action_points[:,2:4]
        self.action_pts1[agent_idx] = action_pts1
        self.action_pts2[agent_idx] = action_pts2

        pts1 = self.state[agent_idx][:,:2]
        pts2 = self.state[agent_idx][:,2:4]

        K1 = self.img1_K
        K2 = self.img2_K
        # K = np.eye(3, 3)

        # pts1_ = np.expand_dims(pts1, axis=0)
        # pts2_ = np.expand_dims(pts2, axis=0)
        # pts1_ = cv2.undistortPoints(pts1_, K1, None)
        # pts2_ = cv2.undistortPoints(pts2_, K2, None)
        p1n = normalize_keypoints(pts1, K1)
        p2n = normalize_keypoints(pts2, K2)


        # =============================================
        # 根据动作点，找基本矩阵
        if (unique_action.size == self.action_size):
            F, mask_inl = cv2.findFundamentalMat(action_pts1, action_pts2, 
                                                    cv2.USAC_MAGSAC, 
                                                    0.75,
                                                    confidence=0.9999)
            # para = {'max_epipolar_error':0.75, 'progressive_sampling':True, 'min_iterations':1, 'max_iterations':1}
            # F, info = poselib.estimate_fundamental(action_pts1, action_pts2, para, {})
            # F, mask = cv2.findFundamentalMat(action_pts1, action_pts2, cv2.FM_8POINT, 0.1, 0.99)
            # F_, mask_ = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 0.1, 0.99)
            res = None
            if F is None or F.shape == (1, 1) or F.shape[0]!=3:
                # no fundamental matrix found
                # print('no F')
                self.inliers = np.array([])
                self.outliers = deepcopy(self.state[agent_idx])
                angle_error = 3.14
                self.dR[agent_idx] = 3.14
                self.dt[agent_idx] = 3.14
                mask = np.array([False]*self.pt_num)
            else:
                # 评估基础矩阵
                E_from_F = np.matmul(np.matmul(K2.T, F), K1)
                if (E_from_F.shape[0]!=3):
                    print("error1")

                res, mask = f_error_mask(pts1.T, pts2.T, F, threshold=0.75)
                if (np.sum(mask)<5):
                    self.inliers = np.array([])
                    self.outliers = deepcopy(self.state[agent_idx])
                    err_q = 3.14
                    err_t = 3.14
                    angle_error = max(err_q, err_t)
                    mask = np.array([False]*mask.shape[0])
                else:
                    p1n = p1n[mask]
                    p2n = p2n[mask]
                    retval, R, t, _ = cv2.recoverPose(E_from_F, p1n, p2n)
                    err_q, err_t = evaluate_R_t(self.GT_R_Rel, self.GT_t_Rel, R, t)
                    angle_error = max(err_q, err_t)
                self.dR[agent_idx] = err_q
                self.dt[agent_idx] = err_t
                mask = mask.reshape((-1))


                if (self.episode_count == 0):
                    episode_count = self.max_episode
                else:
                    episode_count = self.episode_count

                # mask = mask.reshape((-1))
                self.inliers = self.state[agent_idx][mask==True]
                self.outliers = self.state[agent_idx][mask==False]

                # 保存一张图像中，所有回合中的最小的误差的值
                if angle_error < self.min_angle_error:
                    self.min_angle_error = angle_error
                    self.Metrics['pair'] = self.pair_id
                    self.Metrics['inliers'] = self.inliers.shape[0]
                    self.Metrics['min_angle_error'] = angle_error
                    self.Metrics['dR'] = err_q
                    self.Metrics['dt'] = err_t
                    self.Metrics['eposide'] = episode_count
                    self.Metrics['step'] = self.step_count


                # 到达预定误差阈值时用的最小步数
                if angle_error!=9999 and angle_error < self.angle_error_thr:
                    if (self.step_count < self.min_step):
                        self.min_step = self.step_count
                


                # if (self.inliers.shape[0] == self.pt_num):
                #     self.inliers = np.array([])
                #     self.outliers = deepcopy(self.state)

        if (self.inliers.shape[0] == 0): angle_error=3.14


        self.angle_error[agent_idx] = angle_error
        # 储存本回合中小的误差
        if (angle_error < self.min_angle_error_episode):
            self.min_angle_error_episode = angle_error
            # 当本回合误差递增时加一
            self.angle_error_increase_count = 0
        else:
            self.angle_error_increase_count += 1

        # 计算内点率
        inliers_num = self.inliers.shape[0]
        self.inliers_ratio[agent_idx] = inliers_num/self.pt_num
        self.old_inliers_ratio = self.old_inliers.shape[0]/self.pt_num
        if (self.inliers_ratio[agent_idx]>self.max_inliers_episode):
            self.max_inliers_episode=self.inliers_ratio[agent_idx]
            self.best_res = res

        # 奖励
        # 以当前的内点率为奖励值
        self.reward = self.inliers_ratio[agent_idx]

        # self.reward = self.inliers_ratio - self.old_inliers_ratio
        # self.reward = self.inliers_ratio
        self.inliers = self.inliers.reshape((-1, self.state[agent_idx].shape[1]))
        self.old_inliers = self.old_inliers.reshape((-1, self.state[agent_idx].shape[1]))
        # self.old_inliers == self.inliers
        if (self.old_inliers.shape[0] != 0 and self.inliers.shape[0] != 0 and 
            self.old_inliers.shape[0] == self.inliers.shape[0]):
            if (np.sort(self.old_inliers[:,:4],axis=0)==np.sort(self.inliers[:,:4],axis=0)).all():
                self.reward = -0.9999
                self.fix_count += 1
        else:
            self.fix_count = 0

        # # 使用当前的误差作为奖励
        # self.reward = -angle_error
        # if (self.reward<-1.0): self.reward = -0.99999999
        # if (self.reward>0.0): self.reward  = -0.00000001

        # # 奖励为固定的。若本次误差比上次误差大，则r=-0.6，若相等则r=-0.1，若小于则r=0.5
        # if((angle_error>999) and (self.old_angle_error>999)):   # 不出现了
        #     self.reward = -0.6
        #     self.fix_count = 0
        # elif((angle_error>999) and (self.old_angle_error<999)): # 不出现了
        #     # self.reward = -self.old_angle_error
        #     self.reward = -0.6
        #     self.fix_count = 0
        # elif((angle_error<999) and (self.old_angle_error>999)): # 第一次出现
        #     self.reward = -self.angle_error
        #     # self.reward = -0.6
        #     self.fix_count = 0
        # elif(angle_error > self.old_angle_error):
        #     self.reward = -(angle_error-self.old_angle_error)
        #     self.fix_count = 0
        # elif(angle_error == self.old_angle_error):
        #     self.reward = -0.1
        #     self.fix_count += 1
        # else:
        #     self.reward = -(self.old_angle_error-angle_error)
        #     self.fix_count = 0
        # if (self.reward<-1.0): self.reward = -0.99999999
        # if (self.reward>0.0): self.reward  = -0.00000001
        # # 把[-1,0]范围内的reward变换到[-1,1]
        # # self.reward = self.reward * 2 + 1.0


        # # 如果前后的内点一模一样
        # self.inliers = self.inliers.reshape((-1, self.state.shape[1]))
        # self.old_inliers = self.old_inliers.reshape((-1, self.state.shape[1]))
        # # self.old_inliers == self.inliers
        # if (self.old_inliers.shape[0] != 0 and self.inliers.shape[0] != 0 and 
        #     self.old_inliers.shape[0] == self.inliers.shape[0]):
        #     if (np.sort(self.old_inliers[:,:4],axis=0)==np.sort(self.inliers[:,:4],axis=0)).all():
        #         self.reward = -1
        #         self.fix_count += 1
        # else:
        #     self.fix_count = 0
        # # print(self.fix_count)



        # =============================================
        # 下一个状态
        self.prev_state[agent_idx] = deepcopy(self.state[agent_idx])


        # 当前操作点为1，其余-1
        # self.state[..., -3] = -1

        # 用过的点action加一
        if (unique_action.size == self.action_size):
            for p in action:
                self.state[agent_idx][p][-2] += 1
                self.state[agent_idx][p][-3] = 1
        
        # 内点为1，外点为-1
        # 把每个点的误差编码进去
        if res is not None:
            self.state[agent_idx][..., -1] = res
        else:
            self.state[agent_idx][..., -1][:] = 100

        # if self.best_res is not None:
        #     self.state[..., -1] = self.best_res
        # else:
        #     self.state[..., -1][:] = 100

        # if inliers_num != 0:
        #     self.state[..., 6][mask==True] = 1
        #     self.state[..., 6][mask==False] = -1
        # else:
        #     self.state[..., 6] = -1


        self.next_state[agent_idx] = deepcopy(self.state[agent_idx])

        self.old_angle_error = angle_error


        # 结束条件
        # 如果是训练
        if self.train:
            # 如果很长时间固定不动,或者长时间误差增加,或者本回合步数超过最大步数,则结束
            if (self.fix_count>=2) or (self.angle_error_increase_count >= self.angle_error_increase_thr) or (self.step_count >= self.max_step):
                self.done = True
            else:
                self.done = False
        # 如果是测试
        else:
            # 仅在达到最大步数时结束
            if (self.step_count >= self.max_step):
                self.done = True
            else:
                self.done = False

        
        self.info = {}
        # 一张图片中的最小误差
        self.info['min_angle_error'] = self.min_angle_error
        self.info['pair_id'] = self.pair_id
        self.info['sqs'] = self.sqs
        self.info['F'] = deepcopy(F)
        self.info['inliers'] = deepcopy(self.inliers[:, :4])
        self.info['mask'] = deepcopy(mask)
        
        self.old_inliers = deepcopy(self.inliers)
        self.all_inliers[agent_idx] = deepcopy(self.inliers)
        
        # arrSortedIndex = np.lexsort((self.action_list[:, 2], self.action_list[:, 1], self.action_list[:, 0]))
        # xxx=self.action_list[arrSortedIndex , :]
        # xxx = max(self.state[:,7])
        # yyy = self.state[:,7]
        # 返回
        return deepcopy(self.next_state[agent_idx]), deepcopy(self.reward), deepcopy(self.done), self.info

    def state_global_update(self, states, rewards):
        assert len(states) == self.n_agents
        self.state_global[...,:5] = states[0][...,:5]
        max_idx = np.argmax(np.array(rewards))
        # history
        self.state_global[...,-2] += np.array([state[...,-2] for state in states]).sum(axis=0)
        # action
        # self.state_global[...,-3] = states[max_idx][...,-3]
        self.state_global[...,-3] = np.array([state[...,-3] for state in states]).sum(axis=0)
        for idx in range(self.n_agents):
            # TODO: penalty
            penalty_coe = 0.01
            rewards[idx] -= (1 - ((states[idx][...,-3] != -1) * ((self.state_global[...,-3]-states[idx][...,-3]) != -(self.n_agents-1))).sum() / 8) * penalty_coe

        self.state_global[...,-3] += self.n_agents-1
        self.state_global[...,-3][self.state_global[...,-3] != -1] = 1

        # pointwise error
        self.state_global[...,-1] = states[max_idx][...,-1]

    def info_global_update(self, infos, rewards):
        max_idx = np.argmax(np.array(rewards))
        # Fundmental Matrix
        assert len(infos) > 0

        return infos[max_idx]

    def step(self, actions):
        # TODO: state is wrong, only uses the transition from the last agent, proper way should be accumulate all the transitions from all agents
        # this requires more delibrate design of the global state of RLSAC to MA setting, maybe use extra features 
        states = []
        rewards = []
        dones = []
        infos = []
        if actions == None:
            actions = [None] * self.n_agents
        for idx, action in enumerate(actions):
            self.state[idx][..., -3] = -1
            state, reward, done, info = self.single_step(action, idx)
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        # states = deepcopy(state)
        states = deepcopy(np.stack(states, axis=0))
        self.state_global_update(states, rewards)
        info_ret = self.info_global_update(infos, rewards)
        self.rewards = deepcopy(rewards)
        return states, rewards, dones, info_ret

    def close(self):
        cv2.destroyAllWindows()
        print("ransac line close")


    def drawline(self, img,pt1,pt2,color,thickness=1,style='dotted',gap=10): 
        dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5 
        pts= [] 
        for i in np.arange(0,dist,gap): 
            r=i/dist 
            x=int((pt1[0]*(1-r)+pt2[0]*r)+.5) 
            y=int((pt1[1]*(1-r)+pt2[1]*r)+.5) 
            p = (x,y) 
            pts.append(p) 
    
        if style=='dotted': 
            for p in pts: 
                cv2.circle(img,p,thickness,color,-1) 
        else: 
            s=pts[0] 
            e=pts[0] 
            i=0 
            for p in pts: 
                s=e 
                e=p 
                if i%2==1: 
                    cv2.line(img,s,e,color,thickness) 
                i+=1

    def render(self, dummy):
        show = False
        # 可视化点   
        scale = 150
        img0 = cv2.imread(self.img1_fname)
        img1 = cv2.imread(self.img2_fname)
        
        # 上下堆叠，统一宽度
        width = img0.shape[1] - img1.shape[1]
        if width > 0:
            # 上面图像宽，把下面图像也加宽
            image_padding = np.zeros((img1.shape[0], abs(width), 3), dtype=np.uint8)
            img1 = np.hstack([img1, image_padding])
        else:
            image_padding = np.zeros((img0.shape[0], abs(width), 3), dtype=np.uint8)
            img0 = np.hstack([img0, image_padding])
        image = np.vstack([img0, img1])
        offset = image.shape[0]
        image_text = np.zeros((300, img0.shape[1], 3), dtype=np.uint8)
        image = np.vstack([image, image_text])
        # 把内点, 操作点可视化出来

        # # 所有匹配点的连线 暗红色
        # for pt0, pt1 in zip(self.pts0_all, self.pts1_all):
        #     cv2.line(image, (int(pt0[0]), int(pt0[1])), (int(pt1[0]), int(pt1[1]+self.img0.shape[0])), (0, 0, 100), 1, 1)

        # # 当前状态点 黄色
        # for xyxy in self.prev_state[:,:4]:
        #     cv2.line(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]+self.img0.shape[0])), (0, 255, 255), 1, 1)

        # # 下一个状态点 青色
        # for xyxy in self.next_state[:,:4]:
        #     cv2.line(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]+self.img0.shape[0])), (255, 255, 0), 1, 1)

        # 选出来的八个点 蓝色
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for idx, (pts1, pts2) in enumerate(zip(self.action_pts1, self.action_pts2)):
            for pt0, pt1 in zip(pts1, pts2):
                cv2.line(image, (int(pt0[0]), int(pt0[1])), (int(pt1[0]), int(pt1[1]+img0.shape[0])), colors[idx], 2, 2)

        # 内点 红色
        
        best_agent = np.argmax(self.rewards)
        if self.all_inliers[best_agent].size != 0:
            for xyxy in self.all_inliers[best_agent][:,:4]:
                cv2.line(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]+img0.shape[0])), (0, 255, 255), 1, 1)



        if (self.episode_count == 0):
            episode_count = self.max_episode
        else:
            episode_count = self.episode_count

        # 打印信息
        cv2.putText(image, 'best inliers:'+str(self.inliers_ratio[best_agent]), (5, offset+25), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, 'maxinliers:'+str(self.max_inliers_episode), (300, offset+25), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2, cv2.LINE_AA)

        cv2.putText(image, 'step:'+str(self.step_count), (5, offset+55), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, 'reward:'+str(self.rewards[best_agent]), (140, offset+55), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2, cv2.LINE_AA)

        cv2.putText(image, 'episode:'+str(episode_count), (5, offset+85), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, 'total_step:'+str(self.total_step_count), (170, offset+85), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2, cv2.LINE_AA)

        cv2.putText(image, 'Done:'+str(self.done), (5, offset+115), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2, cv2.LINE_AA)

        cv2.putText(image, 'angle_error:'+str(self.angle_error[best_agent]), (5, offset+135), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(image, 'min_angle_error:'+str(self.min_angle_error), (5, offset+155), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(image, 'dR:'+str(self.dR[best_agent]), (5, offset+195), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1, cv2.LINE_AA)
        cv2.putText(image, 'dt:'+str(self.dt[best_agent]), (5, offset+215), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1, cv2.LINE_AA)



        # # 图例
        # cv2.putText(image, 'inliers', (5, 145), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 1)
        # cv2.putText(image, 'outliers', (140, 145), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1)
        # cv2.putText(image, 'state', (5, 175), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1)
        # cv2.putText(image, 'next_s', (140, 175), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 1)
        # cv2.putText(image, 'action', (5, 205), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1)
        # # cv2.putText(image, 'ransac', (140, 205), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 1)


        # A = 1/(self.p2[0]*scale-self.p1[0]*scale)
        # B = -1/(self.p2[1]*scale-self.p1[1]*scale)
        # C = self.p1[1]*scale/(self.p2[1]*scale-self.p1[1]*scale) - self.p1[0]*scale/(self.p2[0]*scale-self.p1[0]*scale)

        # # A = self.A*scale
        # # B = self.B*scale
        # # C = self.C

        # p1 = (0, int(-C/B))
        # p2 = (self.pic_size, int(-(C+self.pic_size*A)/B))

        self.info['env_id'] = 0
        if show:
            cv2.imshow('env_' + str(self.info['env_id']), image)

        image = cv2.resize(image, (int(image.shape[1] * 2), int(image.shape[0] * 2)))
        cv2.imwrite('results/env_' + str(self.info['env_id']) + '_num_' + str(self.total_step_count) + '_pic_' + str(self.data_count) +'_episode_'+str(episode_count)+'_step_' + str(self.step_count)+'.png', image)
        # cv2.imwrite('result/state_env_' + str(self.info['env_id']) + '_num_' + str(self.total_step_count) + '_pic_' + str(self.data_count) +'_episode_'+str(episode_count) +'_step_' + str(self.step_count)+'.png', image_state)

        cv2.waitKey(1)

    def clear_total_step_count(self):
        self.total_step_count = 0

    # def __len__(self):
    #     return 1

    def calc_NSE(self, inliers, GT_p1, GT_p2, p1, p2):
        # 求内点与估计值之间的误差
        dp = p1-p2
        dp *= 1./np.linalg.norm(dp)
        err = []
        # 求内点与真之间的误差
        GT_dp = GT_p1-GT_p2
        GT_dp *= 1./np.linalg.norm(GT_dp)
        GT_err = []
        for point in inliers:
            error = point - p1
            dis = error[1]*dp[0] - error[0]*dp[1]   # 负数表示在直线上面
            err.append(dis * dis)

            error = point - GT_p1
            dis = error[1]*GT_dp[0] - error[0]*GT_dp[1]
            GT_err.append(dis * dis)

        NSE = sum(err)/sum(GT_err)

        return NSE

    def calc_dis(self, p1, p2, p):
        dp = p1-p2
        dp *= 1./np.linalg.norm(dp)
        error = p - p1
        dis = error[1]*dp[0] - error[0]*dp[1]
        return dis


def normalize_keypoints(keypoints, K):
    '''Normalize keypoints using the calibration data.'''

    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])

    return keypoints


def quaternion_from_matrix(matrix, isprecise=False):
    '''Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True
    '''

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]

        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0

        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q


def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        print(R_gt, t_gt, R, t, q_gt)
        import IPython
        IPython.embed()

    return err_q, err_t


def f_error_mask(pts1, pts2, F, threshold=0.001):
    """Compute multiple evaluaton measures for a fundamental matrix.

    Return (False, 0, 0, 0) if the evaluation fails due to not finding inliers for the ground truth model, 
    else return() True, F1 score, % inliers, mean epipolar error of inliers).

    Follows the evaluation procedure in:
    "Deep Fundamental Matrix Estimation"
    Ranftl and Koltun
    ECCV 2018

    Keyword arguments:
    pts1 -- 3D numpy array containing the feature coordinates in image 1, dim 1: x and y coordinate, dim 2: correspondences, dim 3: dummy dimension
    pts2 -- 3D numpy array containing the feature coordinates in image 2, dim 1: x and y coordinate, dim 2: correspondences, dim 3: dummy dimension
    F -- 2D numpy array containing an estimated fundamental matrix
    gt_F -- 2D numpy array containing the corresponding ground truth fundamental matrix
    threshold -- inlier threshold for the epipolar error in pixels
    """
    EPS = 0.00000000001
    num_pts = pts1.shape[1]

    # 2D coordinates to 3D homogeneous coordinates
    hom_pts1 = np.concatenate((pts1, np.ones((1, num_pts))), axis=0)
    hom_pts2 = np.concatenate((pts2, np.ones((1, num_pts))), axis=0)

    def epipolar_error(hom_pts1, hom_pts2, F):
        """Compute the symmetric epipolar error."""
        res  = 1 / (np.linalg.norm(F.T.dot(hom_pts2)[0:2], axis=0) + EPS)
        res += 1 / (np.linalg.norm(F.dot(hom_pts1)[0:2], axis=0) + EPS)
        res *= abs(np.sum(hom_pts2 * np.matmul(F, hom_pts1), axis=0))
        return res

    # determine inliers based on the epipolar error
    est_res = epipolar_error(hom_pts1, hom_pts2, F)

    inlier_mask = (est_res < threshold)

    return est_res, inlier_mask


def f_error(pts1, pts2, F, gt_F, threshold=0.001):
    """Compute multiple evaluaton measures for a fundamental matrix.

    Return (False, 0, 0, 0) if the evaluation fails due to not finding inliers for the ground truth model, 
    else return() True, F1 score, % inliers, mean epipolar error of inliers).

    Follows the evaluation procedure in:
    "Deep Fundamental Matrix Estimation"
    Ranftl and Koltun
    ECCV 2018

    Keyword arguments:
    pts1 -- 3D numpy array containing the feature coordinates in image 1, dim 1: x and y coordinate, dim 2: correspondences, dim 3: dummy dimension
    pts2 -- 3D numpy array containing the feature coordinates in image 2, dim 1: x and y coordinate, dim 2: correspondences, dim 3: dummy dimension
    F -- 2D numpy array containing an estimated fundamental matrix
    gt_F -- 2D numpy array containing the corresponding ground truth fundamental matrix
    threshold -- inlier threshold for the epipolar error in pixels
    """
    EPS = 0.00000000001
    num_pts = pts1.shape[1]

    # 2D coordinates to 3D homogeneous coordinates
    hom_pts1 = np.concatenate((pts1, np.ones((1, num_pts))), axis=0)
    hom_pts2 = np.concatenate((pts2, np.ones((1, num_pts))), axis=0)

    def epipolar_error(hom_pts1, hom_pts2, F):
        """Compute the symmetric epipolar error."""
        res  = 1 / np.linalg.norm(F.T.dot(hom_pts2)[0:2], axis=0)
        res += 1 / np.linalg.norm(F.dot(hom_pts1)[0:2], axis=0)
        res *= abs(np.sum(hom_pts2 * np.matmul(F, hom_pts1), axis=0))
        return res

    # determine inliers based on the epipolar error
    est_res = epipolar_error(hom_pts1, hom_pts2, F)
    gt_res = epipolar_error(hom_pts1, hom_pts2, gt_F)
    
    inlier_mask = (est_res < threshold)
    est_inliers = (est_res < threshold)
    gt_inliers = (gt_res < threshold)
    true_positives = est_inliers & gt_inliers

    est_inliers = float(est_inliers.sum())
    gt_inliers = float(gt_inliers.sum())

    inliers = est_inliers / num_pts

    if gt_inliers > 0:
        true_positives = float(true_positives.sum())

        precision = true_positives / (est_inliers + EPS)
        recall = true_positives / (gt_inliers + EPS)

        F1 = 2 * precision * recall / (precision + recall + EPS)
        

        epi_mask = (gt_res < 1)
        if epi_mask.sum() > 0:
            epi_error = float(est_res[epi_mask].mean())
        else:
            # no ground truth inliers for the fixed 1px threshold used for epipolar errors
            return False, 0, inliers, 0, inlier_mask

        return True, F1, inliers, epi_error, inlier_mask
    else:
        # no ground truth inliers for the user provided threshold
        return False, 0, inliers, 0, inlier_mask

def pose_error(R, gt_R, t, gt_t):
    """Compute the angular error between two rotation matrices and two translation vectors.


    Keyword arguments:
    R -- 2D numpy array containing an estimated rotation
    gt_R -- 2D numpy array containing the corresponding ground truth rotation
    t -- 2D numpy array containing an estimated translation as column
    gt_t -- 2D numpy array containing the corresponding ground truth translation
    """

    # calculate angle between provided rotations
    dR = np.matmul(R, np.transpose(gt_R))
    dR = cv2.Rodrigues(dR)[0]
    dR = np.linalg.norm(dR) * 180 / math.pi
    
    # calculate angle between provided translations
    dT = float(np.dot(gt_t.T, t))
    dT /= float(np.linalg.norm(gt_t))

    if dT > 1 or dT < -1:
        print("Domain warning! dT:", dT)
        dT = max(-1, min(1, dT))
    dT = math.acos(dT) * 180 / math.pi

    return dR, dT

def denormalize_pts(pts, im_size):
    """Undo image coordinate normalization using the image size.

    In-place operation.

    Keyword arguments:
    pts -- N-dim array conainting x and y coordinates in the first dimension
    im_size -- image height and width
    """	
    pts *= max(im_size)
    pts[0] += im_size[1] / 2
    pts[1] += im_size[0] / 2