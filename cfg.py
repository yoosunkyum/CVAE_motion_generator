# ========= 하이퍼파라미터 및 구조 설정 =========
input_dim = (19 + 19) + 1*(3 + 4 + 3 + 3)   # dof_positions + dof_velocities + body_positions + body_rotations + body_linear_velocities + body_angular_velocities
cond_dim = input_dim    # s_t-1 차원 (같은 구조라고 가정)
latent_dim = 32
hidden_dims = [256, 128]
num_epochs = 6000
batch_size = 512
learning_rate = 1e-3
beta = 0.2