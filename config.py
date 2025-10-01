class Config:
    """全局配置类"""

    # 数据相关
    DATA_DIR = "preprocessed_csi_data"
    KEYPOINT_SCALE = 1000.0
    WINDOW_SIZE = 20
    NUM_KEYPOINTS = 15
    NUM_SUBCARRIERS = 540

    # 模型相关
    TCN_CHANNELS = [480, 360, 240]
    CONV_CHANNELS = [8, 16, 32, 64]
    DROPOUT = 0.5
    ATTENTION_GROUPS = 8

    # 训练相关
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 5e-5
    PATIENCE = 5

    # 损失权重
    POSITION_WEIGHT = 1.0
    BONE_WEIGHT = 0.2
    LOSS_TYPE = 'smooth_l1'

    # 骨架连接
    SKELETON_CONNECTIONS = [
        (0, 1), (1, 8), (1, 2), (2, 3), (3, 4),
        (1, 5), (5, 6), (6, 7), (8, 9), (8, 12),
        (9, 10), (10, 11), (12, 13), (13, 14)
    ]

    # 关键点名称
    KEYPOINT_NAMES = {
        0: "Neck", 1: "Chest", 2: "L_Shoulder", 3: "L_Elbow", 4: "L_Wrist",
        5: "R_Shoulder", 6: "R_Elbow", 7: "R_Wrist", 8: "Pelvis", 9: "L_Hip",
        10: "L_Knee", 11: "L_Ankle", 12: "R_Hip", 13: "R_Knee", 14: "R_Ankle"
    }

    # 设备配置
    GPU_IDS = [0]
    NUM_WORKERS = 0

    # 输出路径
    OUTPUT_DIR = "outputs"
    MODEL_SAVE_PATH = "outputs/best_model.pth"
    LOG_DIR = "outputs/logs"