import numpy as np




def compute_pck_pckh(dt_kpts,gt_kpts,thr):
    """
    pck指标计算
    :param dt_kpts:算法检测输出的估计结果,shape=[n,h,w]=[行人数，２，关键点个数]
    :param gt_kpts: groundtruth人工标记结果,shape=[n,h,w]
    :param refer_kpts: 尺度因子，用于预测点与groundtruth的欧式距离的scale。
    　　　　　　　　　　　pck指标：躯干直径，左肩点－右臀点的欧式距离；
    　　　　　　　　　　　pckh指标：头部长度，头部rect的对角线欧式距离；
    :return: 相关指标
    """
    dt=np.array(dt_kpts)
    gt=np.array(gt_kpts)
    assert(dt.shape[0]==gt.shape[0])
    kpts_num=gt.shape[2] #keypoints
    ped_num=gt.shape[0] #batch_size
    #compute dist
    scale=np.sqrt(np.sum(np.square(gt[:,:,5]-gt[:,:,12]),1)) #right shoulder--left hip
    dist=np.sqrt(np.sum(np.square(dt-gt),1))/np.tile(scale,(gt.shape[2],1)).T
    #compute pck
    pck = np.zeros(gt.shape[2]+1)
    for kpt_idx in range(kpts_num):
        pck[kpt_idx] = 100*np.mean(dist[:,kpt_idx] <= thr)
        # compute average pck
    pck[17] = 100*np.mean(dist <= thr)
    return pck


def compute_pck_pckh_18(dt_kpts,gt_kpts,thr):
    """
    pck指标计算
    :param dt_kpts:算法检测输出的估计结果,shape=[n,h,w]=[行人数，２，关键点个数]
    :param gt_kpts: groundtruth人工标记结果,shape=[n,h,w]
    :param refer_kpts: 尺度因子，用于预测点与groundtruth的欧式距离的scale。
    　　　　　　　　　　　pck指标：躯干直径，左肩点－右臀点的欧式距离；
    　　　　　　　　　　　pckh指标：头部长度，头部rect的对角线欧式距离；
    :return: 相关指标
    """
    dt=np.array(dt_kpts)
    gt=np.array(gt_kpts)
    assert(dt.shape[0]==gt.shape[0])
    kpts_num=gt.shape[2] #keypoints
    ped_num=gt.shape[0] #batch_size
    #compute dist
    scale=np.sqrt(np.sum(np.square(gt[:,:,6]-gt[:,:,13]),1)) #right shoulder--left hip
    dist=np.sqrt(np.sum(np.square(dt-gt),1))/np.tile(scale,(gt.shape[2],1)).T
    #compute pck
    pck = np.zeros(gt.shape[2]+1)
    for kpt_idx in range(kpts_num):
        pck[kpt_idx] = 100*np.mean(dist[:,kpt_idx] <= thr)
        # compute average pck
    pck[18] = 100*np.mean(dist <= thr)
    return pck


def compute_pck_pckh_15(dt_kpts, gt_kpts, thr):
    """15个关键点的PCK计算"""
    dt = np.array(dt_kpts)
    gt = np.array(gt_kpts)

    # 自动转换格式 [batch, 15, 2] -> [batch, 2, 15]
    if dt.shape[1] == 15 and dt.shape[2] == 2:
        dt = dt.transpose(0, 2, 1)
        gt = gt.transpose(0, 2, 1)

    kpts_num = gt.shape[2]  # 15个关键点

    # 使用neck到hip的距离作为scale
    scale = np.sqrt(np.sum(np.square(gt[:, :, 2] - gt[:, :, 12]), 1))
    scale = np.maximum(scale, 1e-6)

    dist = np.sqrt(np.sum(np.square(dt - gt), 1)) / np.tile(scale, (gt.shape[2], 1)).T

    pck = np.zeros(16)  # 15个关键点 + 1个总体 = 16个值
    for kpt_idx in range(15):
        pck[kpt_idx] = 100 * np.mean(dist[:, kpt_idx] <= thr)

    pck[15] = 100 * np.mean(dist <= thr)  # 总体PCK存在索引15
    return pck
