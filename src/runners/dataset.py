import numpy as np
import h5py, cv2
import random
import os
from copy import deepcopy
import torch

def create_batch(dataset_dir, vis_thresh, id):
    cal_db_list = {}
    vis_pairs = []

    data_dir = dataset_dir

    img_db = 'images.txt'
    vis_db = 'visibility.txt'
    cal_db = 'calibration.txt'

    img_db = open(data_dir + img_db, 'r')
    vis_db = open(data_dir + vis_db, 'r')
    cal_db = open(data_dir + cal_db, 'r')

    img_files = img_db.readlines()
    vis_files = vis_db.readlines()
    cal_files = cal_db.readlines()

    img_db.close()
    vis_db.close()
    cal_db.close()

    for i, cal_file in enumerate(cal_files):
        cal = h5py.File(data_dir + cal_file[:-1], 'r')

        K = np.array(cal['K'])
        R = np.array(cal['R'])
        T = np.array(cal['T'])
        imsize = np.array(cal['imsize'])
        #     print(imsize[0,0], imsize[0,1])

        #     K[0, 2] += imsize[0, 0] * 0.5
        #     K[1, 2] += imsize[0, 1] * 0.5
        K[0, 2] += 1024 * 0.5
        K[1, 2] += 1024 * 0.5

        cal_db_list[i] = (K, R, T)

    for i, vis_file in enumerate(vis_files):

        vis_file = open(data_dir + vis_file[:-1])
        vis_infos = vis_file.readlines()

        for j, vis_info in enumerate(vis_infos):
            vis_count = float(vis_info)
            if vis_count > vis_thresh:
                vis_pairs.append((i, j, id))

        vis_file.close()

    random.shuffle(vis_pairs)
    if id == 0:
        vis_mod = vis_pairs[0:10000]
    else:
        vis_mod = vis_pairs.copy()

    return data_dir, img_files, cal_db_list, vis_mod

def build_dataset(dataset_dir, vthresh):
    data_dir_arr = []
    img_files_arr = []
    vis_pairs_arr = []
    cal_db_arr = []

    data_dir1, img_files1, cal_db1, vis_pair1 = create_batch(dataset_dir, vthresh, 0)
    data_dir_arr.append(data_dir1)
    img_files_arr.append(img_files1)
    cal_db_arr.append(cal_db1)
    vis_pairs_arr = vis_pair1.copy()
    random.shuffle(vis_pairs_arr)

    return img_files1, vis_pairs_arr, cal_db1

def build_kitti_dataset(dataset_dir):
    # Odometry00~10为有真值的，故取00~08来训练，09~10测试
    img_dir = []
    vis_pairs = []
    cal_db = []
    pose_db = []
    seq_list = [2]
    for seq in seq_list:
        seq_image_path_list = []
        seq_vis_pair = []
        cal_list = []
        pose_list = []
        seq_s = "%02d" % seq
        # 内参
        cal_path = os.path.join(dataset_dir, seq_s, 'calib.txt')
        cal_file = open(cal_path, 'r')
        cal = cal_file.readlines()[2].split()[1:]
        cal_file.close()
        cal_float = [float(n) for n in cal]
        K = np.array(cal_float).reshape((3,4))[:,:3]

        # 图像地址
        images_path = os.path.join(dataset_dir, seq_s, 'image_2')
        image_path_list = os.listdir(images_path)
        image_path_list.sort()
        for path in image_path_list:
            seq_image_path_list.append(os.path.join(images_path, path))
        img_dir.append(seq_image_path_list)

        # pose真值
        pose_path = os.path.join(dataset_dir, seq_s, seq_s + '.txt')
        pose_file = open(pose_path, 'r')
        poses = pose_file.readlines()
        pose_file.close()
        for pose in poses:

            pose = [float(p) for p in pose.split()]
            pose += [0, 0, 0, 1]

            pose = np.array(pose).reshape((4,4))
            pose = np.linalg.inv(pose)

            R = pose[0:3,0:3]
            t = pose[0:3,3].reshape(1,3)
            cal_list.append([K, R, t])
        cal_db.append(cal_list)

        for i in range(len(seq_image_path_list)-1):
            seq_vis_pair.append((i, i+1, seq))



        #     GT_R1 = cal_list[i][1]
        #     GT_R2 = cal_list[i+1][1]
        #     GT_R_Rel = np.matmul(GT_R2, np.transpose(GT_R1))
            
        #     GT_t1 = cal_list[i][2]
        #     GT_t2 = cal_list[i+1][2]
        #     GT_t_Rel = GT_t2.T - np.matmul(GT_R_Rel, GT_t1.T)

        #     pose_list.append([GT_R_Rel, GT_t_Rel])
        # pose_db.append(pose_list)

    vis_pairs += seq_vis_pair
    random.shuffle(vis_pairs)
    
    return img_dir, vis_pairs, cal_db


def load_h5(filename):
    '''Loads dictionary from hdf5 file'''
    dict_to_load = {}
    import time
    start = time.time()
    try:
        checkpoint_filename = filename.replace(".h5", ".pkl")
        if os.path.exists(checkpoint_filename):
            dict_to_load = torch.load(checkpoint_filename)
            print(f"load {filename} from cache", checkpoint_filename)
        else:
            with h5py.File(filename, 'r') as f:
                keys = [key for key in f.keys()]
                for key in keys:
                    dict_to_load[key] = f[key][()]
            print(f"save {filename} to cache", checkpoint_filename)
            torch.save(dict_to_load, checkpoint_filename)
    except:
        print('Cannot find file {}'.format(filename))
    end = time.time()
    print(f"load {filename} cost: ", end-start)
    return dict_to_load


def build_cvpr_dataset(dataset_dir, train, ids):
    if train: DIR = dataset_dir + 'train'
    else: DIR = dataset_dir + 'val'

    seqs = os.listdir(DIR)
    seqs.sort()
    seqs_ = []
    for id in ids:
        seqs_.append(seqs[id])
    seqs = seqs_
    out_model = {}
    inls = {}

    img_dir = []
    vis_pairs = []
    cal_db = []
    pose_db = []
    seq_list = [2]
    
    seqs_db = {}
    min_pt = 99999
    # 遍历每个数据序列
    for seq in seqs:
        seqs_db[seq] = {}
        # 读取序列的数据
        import time
        start = time.time()
        matches = load_h5(f'{DIR}/{seq}/matches.h5')# 每对图片的特征点对坐标，xyxy
        matches_scores = load_h5(f'{DIR}/{seq}/match_conf.h5')  # 每对图片的特征点对分数
        F_gt = load_h5(f'{DIR}/{seq}/Fgt.h5')       # 每对图片的F真值
        E_gt = load_h5(f'{DIR}/{seq}/Egt.h5')       # 每对图片的E真值
        K1_K2 = load_h5(f'{DIR}/{seq}/K1_K2.h5')    # 每对图片的内参
        R = load_h5(f'{DIR}/{seq}/R.h5')            # 每张图片的绝对位姿
        T = load_h5(f'{DIR}/{seq}/T.h5')            # 每张图片的绝对位姿
        end = time.time()
        print("read h5 cost: ", end - start)

        seqs_db[seq]['matches'] = matches
        seqs_db[seq]['matches_scores'] = matches_scores
        seqs_db[seq]['F_gt'] = F_gt
        seqs_db[seq]['E_gt'] = E_gt
        seqs_db[seq]['K1_K2'] = K1_K2
        seqs_db[seq]['R'] = R
        seqs_db[seq]['t'] = T

        # # 遍历每个序列的每个图像对
        # for k, m in matches.items():
        #     ms = matches_scores[k].reshape(-1)
        #     if (ms.shape[0]<min_pt): min_pt = ms.shape[0]
        
        key = list(matches.keys())
        key_ = []
        for k in key:
            key_.append(seq + '-' + k)

    return seqs_db, key_


# 初始化输入数据集地址。每取一次数据，首先看当前数据是否用完了，如果用完了则换一个序列
class cvpr_dataloader:
    def __init__(self, dataset_dir, train, pt_num=100):
        if train: self.DIR = dataset_dir + 'train'
        else: self.DIR = dataset_dir + 'val'
        self.pt_num = pt_num
        # 读取数据集序列，排序好的
        self.seqs = os.listdir(self.DIR)
        self.seqs.sort()
        self.seq_total_num = len(self.seqs)
        self.seq_current_idx = 0
        self.pair_total_idx = 0
        self.pair_current_idx = 0

        self.epoch_end = False

        self.used_pair = []
        self.pair_list = []

    def next_scene(self):
        # 如果当前是新回合，或者当前序列的数据都用完了，则读取新的序列
        if (len(self.used_pair) >= len(self.pair_list)):
            # 初始化重置各个变量
            self.seqs_db = {}
            self.used_pair = []
            self.pair_list = []

            # 开始读取序列内的所有数据
            seq = self.seqs[self.seq_current_idx]
            self.seqs_db[seq] = {}
            # 读取序列的数据
            matches = load_h5(f'{self.DIR}/{seq}/matches.h5')# 每对图片的特征点对坐标，xyxy
            matches_scores = load_h5(f'{self.DIR}/{seq}/match_conf.h5')  # 每对图片的特征点对分数
            F_gt = load_h5(f'{self.DIR}/{seq}/Fgt.h5')       # 每对图片的F真值
            E_gt = load_h5(f'{self.DIR}/{seq}/Egt.h5')       # 每对图片的E真值
            K1_K2 = load_h5(f'{self.DIR}/{seq}/K1_K2.h5')    # 每对图片的内参
            R = load_h5(f'{self.DIR}/{seq}/R.h5')            # 每张图片的绝对位姿
            T = load_h5(f'{self.DIR}/{seq}/T.h5')            # 每张图片的绝对位姿

            self.seqs_db[seq]['matches'] = matches
            self.seqs_db[seq]['matches_scores'] = matches_scores
            self.seqs_db[seq]['F_gt'] = F_gt
            self.seqs_db[seq]['E_gt'] = E_gt
            self.seqs_db[seq]['K1_K2'] = K1_K2
            self.seqs_db[seq]['R'] = R
            self.seqs_db[seq]['t'] = T
            
            key = list(matches.keys())
            for k in key:
                self.pair_list.append(seq + '-' + k)
            # 打乱数据对
            random.shuffle(self.pair_list)
            self.pair_total_idx = len(self.pair_list)
            self.pair_current_idx = 0

            # 如果这个序列的数据都读完了，则读下一个序列的数据，如果所有序列都读完了，则回到第一个序列开始
            self.seq_current_idx += 1
            if(self.seq_current_idx>=self.seq_total_num):
                self.seq_current_idx = 0
                # 所有数据结束标志位，epoch结束
                self.epoch_end = True
            else:
                self.epoch_end = False


        # 读取下一个数据返回
        self.pair_db = {}
        pair_str = self.pair_list[self.pair_current_idx]
        sqs = pair_str.split('-')[0]
        img1_id = pair_str.split('-')[1]
        img2_id = pair_str.split('-')[2]

        # 读取特征点对的分数，如果特征点对少于100个，则跳过这对图片
        matchscore = self.seqs_db[sqs]['matches_scores'][img1_id+'-'+img2_id].reshape((-1,1))
        while(matchscore.shape[0]<self.pt_num):
            self.used_pair.append(self.pair_current_idx)
            self.pair_current_idx += 1
            if (self.pair_current_idx>=self.pair_total_idx):
                print("over load")
                self.pair_current_idx = 0
            pair_str = self.pair_list[self.pair_current_idx]
            sqs = pair_str.split('-')[0]
            img1_id = pair_str.split('-')[1]
            img2_id = pair_str.split('-')[2]
            matchscore = self.seqs_db[sqs]['matches_scores'][img1_id+'-'+img2_id].reshape((-1,1))

        img1_fname = f'{self.DIR}/{sqs}/images/' + img1_id + '.jpg'
        img2_fname = f'{self.DIR}/{sqs}/images/' + img2_id + '.jpg'

        img1_K = self.seqs_db[sqs]['K1_K2'][img1_id+'-'+img2_id][0][0]
        img2_K = self.seqs_db[sqs]['K1_K2'][img1_id+'-'+img2_id][0][1]

        img1_R = self.seqs_db[sqs]['R'][img1_id]
        img2_R = self.seqs_db[sqs]['R'][img2_id]

        img1_t = self.seqs_db[sqs]['t'][img1_id]
        img2_t = self.seqs_db[sqs]['t'][img2_id]

        GT_E = self.seqs_db[sqs]['E_gt'][img1_id+'-'+img2_id]
        GT_F = self.seqs_db[sqs]['F_gt'][img1_id+'-'+img2_id]

        # img1_R表示从位置1到原点的旋转，
        # 要得到位置1到位置2的旋转，则如下计算
        GT_R_Rel = np.dot(img2_R, img1_R.T)
        GT_t_Rel = img2_t - np.dot(GT_R_Rel, img1_t)


        # 读取特征点对
        real_points = self.seqs_db[sqs]['matches'][img1_id+'-'+img2_id]


        # matchscore = self.seqs_db[sqs]['matches_scores'][img1_id+'-'+img2_id].reshape((-1,1))
        all_points = np.concatenate((real_points, matchscore), axis=1)
        # 把特征点对按照分数排序，分数越小越好            这里膨胀一点点，为了给后面的描述子计算损失留出空间
        all_points = all_points[all_points[:,4].argsort()][:self.pt_num]


        # # 特征点描述子
        # pts1 = all_points[:,:2]
        # pts2 = all_points[:,2:4]
        # cv_kpts1 = [cv2.KeyPoint(pts1[i][0], pts1[i][1], 1)
        #             for i in range(pts1.shape[0])]
        # cv_kpts2 = [cv2.KeyPoint(pts2[i][0], pts2[i][1], 1)
        #             for i in range(pts2.shape[0])]
        # img1 = cv2.imread(img1_fname)
        # img2 = cv2.imread(img2_fname)
        # # brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=16, use_orientation=True)
        # sift = cv2.SIFT_create()


        # cv_kpts1, des1 = sift.compute(img1, cv_kpts1)
        # cv_kpts2, des2 = sift.compute(img2, cv_kpts2)
        # assert len(cv_kpts1) == pts1.shape[0]

        # # svd分解降维
        # U,S,V = np.linalg.svd(des1)
        # size_projected = 16
        # projector = V[:,:size_projected]
        # des1 = des1.dot(projector)
        # U,S,V = np.linalg.svd(des2)
        # projector = V[:,:size_projected]
        # des2 = des2.dot(projector)

        # all_points = np.concatenate((all_points,des1,des2), axis=1)




        # if(des1.shape[0]<self.pt_num):
        #     print(22222)
        # if(des2.shape[0]<self.pt_num):
        #     print(33333)

        # pts1=[]
        # pts2=[]
        # # 重新找分数
        # for i, kp in enumerate(cv_kpts1):
        #     odd = 0
        #     while((kp.pt[0]-all_points[i+odd][0])>0.001):
        #         odd+=1
        #         if(i+odd>self.pt_num*2): print(11111)
        #     pt = np.concatenate((all_points[i+odd,:2], des1[i], np.array([all_points[i+odd,4]])), axis=0)
        #     pts1.append(pt)
        #     if(len(pts1)>self.pt_num+50): break
        
        # for i, kp in enumerate(cv_kpts2):
        #     odd = 0
        #     while((kp.pt[0]-all_points[i+odd][2])>0.001):
        #         odd+=1
        #         if(i+odd>self.pt_num*2): print(11111)
        #     pt = np.concatenate((all_points[i+odd,2:4], des2[i], np.array([all_points[i+odd,4]])), axis=0)
        #     pts2.append(pt)
        #     if(len(pts2)>self.pt_num+50): break

        # # 找匹配的
        # pair_pts=[]
        # for i in range(len(pts1)):
        #     for j in range(len(pts2)):
        #         if((pts1[i][-1]-pts2[j][-1])<0.00001):
        #             pt = np.concatenate((pts1[i][:2], pts2[i][:2], np.array([pts1[i][-1]]), pts1[i][2:-1], pts2[i][2:-1]), axis=0)
        #             pair_pts.append(pt)
        #             break
        #     if (len(pair_pts)>=self.pt_num): break
        # if(len(pair_pts)<self.pt_num):
        #     print(44444)
        # all_points = np.array(pair_pts)[:self.pt_num,:]
        


        self.pair_db['sqs'] = sqs
        self.pair_db['img1_id'] = img1_id
        self.pair_db['img2_id'] = img2_id
        self.pair_db['img1_fname'] = img1_fname
        self.pair_db['img2_fname'] = img2_fname
        self.pair_db['img1_K'] = img1_K
        self.pair_db['img2_K'] = img2_K
        # self.pair_db['img1_R'] = img1_R
        # self.pair_db['img2_R'] = img2_R
        # self.pair_db['img1_t'] = img1_t
        # self.pair_db['img2_t'] = img2_t
        self.pair_db['GT_E'] = GT_E
        self.pair_db['GT_F'] = GT_F
        self.pair_db['GT_R_Rel'] = GT_R_Rel
        self.pair_db['GT_t_Rel'] = GT_t_Rel
        self.pair_db['all_points'] = all_points
        # self.pair_db['real_points'] = real_points
        self.pair_db['epoch_end'] = self.epoch_end
        self.pair_db['pair_str'] = pair_str


        # idx递增
        self.used_pair.append(self.pair_current_idx)
        self.pair_current_idx += 1

        # 序列结束标志位
        if (len(self.used_pair) >= len(self.pair_list)):
            seq_end = True
        else:
            seq_end = False
        self.pair_db['seq_end'] = seq_end

        return deepcopy(self.pair_db)
    
    # 重置当前进度，不重新读取数据
    def reset(self):
        self.used_pair = []
        self.pair_list = []
        self.seq_current_idx = 0

    def seed(self, seed_num):
        # pass
        np.random.seed(seed_num)
        random.seed(seed_num)
