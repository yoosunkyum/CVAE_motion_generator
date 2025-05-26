# # ========= H1 하이퍼파라미터 및 구조 설정 =========
# joint_pos_dim = 19
# joint_vel_dim = joint_pos_dim

# base_pos_dim = 3
# base_rot_dim = 4
# base_lin_vel_dim = 3
# base_ang_vel_dim = 3

# ========= G1 하이퍼파라미터 및 구조 설정 =========
joint_pos_dim = 29
joint_vel_dim = joint_pos_dim

# base_pos_dim = 3
base_pos_dim = 1 #absolute x,y position은 사용하지 않을 경우
# base_rot_dim = 4
base_rot_dim = 6 # rot6d 사용
base_lin_vel_dim = 3
base_ang_vel_dim = 3

# ============================================
#VAE structure related
input_dim = (joint_pos_dim + joint_vel_dim)\
            +base_pos_dim + base_rot_dim + base_lin_vel_dim + base_ang_vel_dim   
cond_dim = input_dim    # s_t_1 차원 (같은 구조라고 가정)
latent_dim = 64
hidden_dims = [256, 256]
num_epochs = 10000
# batch_size = 512
batch_size = 4096

#learning gains
learning_rate = 1e-3
lr_max = 1e-3
lr_min = 1e-7
beta = 10.0
