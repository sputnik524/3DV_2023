import datetime
import os
import pprint
import time
import threading
import torch as th
import numpy as np
from tqdm import tqdm
from utils_rlsac import *
from metrics import *
from eval_F import evaluate_results, load_h5, f_error_mask
import poselib
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from runners import cvpr_dataloader

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]
    unique_token = f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}"

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)

def eval_F(cv2_results):
    DIR = '/cluster/project/infk/courses/252-0579-00L/group19_2023/RANSAC-Tutorial-Data/val'
    seqs = os.listdir(DIR)
    out_model = {}
    inls = {}
    # 遍历每个数据序列
    for seq in seqs:
        # 读取序列的数据
        matches = load_h5(f'{DIR}/{seq}/matches.h5')
        matches_scores = load_h5(f'{DIR}/{seq}/match_conf.h5')
        out_model[seq] = {}
        inls[seq] = {}

        # F矩阵
        init_F = cv2_results[0][seq]

        # 遍历每个序列的每个图像对
        total_cnt = 0
        valid_cnt = 0
        for k, m in tqdm(matches.items()):
            total_cnt += 1
            if k not in init_F:
                continue
            valid_cnt += 1

            ms = matches_scores[k].reshape(-1)
            mask = ms <= 0.8
            tentatives = m[mask]
            tentative_idxs = np.arange(len(mask))[mask]
            src_pts = tentatives[:,:2]
            dst_pts = tentatives[:,2:]
            # opencv计算基本矩阵
            if src_pts.shape[0]<7:
                F = None
                mask_inl = np.zeros((src_pts.shape[0],1),dtype=np.uint8)
            elif init_F[k] is None:
                F = None
                mask_inl = np.zeros((src_pts.shape[0],1),dtype=np.uint8)    
            else:
                # F, mask_inl = cv2.findFundamentalMat(src_pts, dst_pts, cv2.USAC_MAGSAC, 
                #                             inlier_th, confidence=0.9999)
                # F, mask_inl = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 
                #                             inlier_th, confidence=0.9999)
                # 默认参数
                # F, mask_inl = cv2.findFundamentalMat(src_pts, dst_pts, cv2.LMEDS , ransacReprojThreshold=3, confidence=0.99, maxIters=10000)
                # F, mask_inl = cv2.findFundamentalMat(src_pts, dst_pts, cv2.USAC_MAGSAC , ransacReprojThreshold=0.75, confidence=0.9999, maxIters=10000)
                # para = {'max_epipolar_error':0.75, 'progressive_sampling':False, 'min_iterations':10000, 'max_iterations':10000}
                # F, info = poselib.estimate_fundamental(src_pts, dst_pts, para, {})
                mask = f_error_mask(src_pts.T, dst_pts.T, init_F[k], threshold=2.5)

                src_pts_ = src_pts[mask]
                dst_pts_ = dst_pts[mask]


                F, info = poselib.refine_fundamental(src_pts_, dst_pts_, init_F[k], {})
                mask_inl = mask.astype(np.uint8)
            # inliers_num_cv = np.sum(mask_inl)
            # if F is not None and F.shape[0]==3:
            #     mask_ = f_error_mask(src_pts.T, dst_pts.T, F, threshold=inlier_th)
            #     mask_ = mask_.astype(np.uint8)
            #     inliers_num = np.sum(mask_)
            #     mask_inl = mask_.reshape((-1,1))

            # 把每个序列的每个图像对的基本矩阵与内点mask都存下来
            out_model[seq][k] = F
            inls[seq][k] = tentatives[(mask_inl==1).reshape((-1)),:4]
        print(f'{seq} total: {total_cnt}, valid: {valid_cnt}')

    new_cv2_result=(out_model, inls)

    MAEs, r_errors, t_errors = evaluate_results(new_cv2_result, DIR, all=False)
    # 计算mAA
    mAA = calc_mAA_FE(MAEs,ths = np.deg2rad([10]))
    mAA_r = calc_mAA_FE(r_errors,ths = np.deg2rad([10]))
    mAA_t = calc_mAA_FE(t_errors,ths = np.deg2rad([10]))
    final = 0
    for k,v in mAA.items():
        final+= v / float(len(mAA))
    final_r = 0
    for k,v in mAA_r.items():
        final_r+= v / float(len(mAA_r))
    final_t = 0
    for k,v in mAA_t.items():
        final_t+= v / float(len(mAA_t))
    print (f'Validation mAA = {final}, mAA_r = {final_r}, mAA_t = {final_t}')
    # 计算误差中位数
    r_error_list = []
    for e in r_errors:
        value = list(r_errors[e].values())
        r_error_list+=(value)
        np_value = np.array(value)
        mid = np.median(np_value)
        # print (f'{e} r_error_mid = {np.rad2deg(mid)}')

    np_value = np.array(r_error_list)
    mid = np.median(np_value)
    mean = np.mean(np_value)
    print (f'total r_error_mid = {np.rad2deg(mid)} r_error_mean = {np.rad2deg(mean)}')

    t_error_list = []
    for e in t_errors:
        value = list(t_errors[e].values())
        t_error_list+=(value)
        np_value = np.array(value)
        mid = np.median(np_value)
        # print (f'{e} t_error_mid = {np.rad2deg(mid)}')

    np_value = np.array(t_error_list)
    mid = np.median(np_value)
    mean = np.mean(np_value)
    print (f'total t_error_mid = {np.rad2deg(mid)} t_error_mean = {np.rad2deg(mean)}')


def evaluate_sequential(args, runner):

    seqs = ['st_peters_square', 'sacre_coeur']
    outmodel = {}
    inliers = {}
    for seq in seqs:
        outmodel[seq] = {}
        inliers[seq] = {}

    for _ in tqdm(range(args.test_nepisode)):
        _, policy_result = runner.run(test_mode=True)
        outmodel[policy_result['sqs']][policy_result['pair_id']] = policy_result['F']
        inliers[policy_result['sqs']][policy_result['pair_id']] = policy_result['inliers']
    
    eval_F((outmodel, inliers))

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):

    # Init runner so we can get env info
    database = cvpr_dataloader(args.data_dir, not args.evaluate, pt_num=100)
    args.env_args["database"] = database
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (8,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch, _ = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env)
            )
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)
            runner.env.render()

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
