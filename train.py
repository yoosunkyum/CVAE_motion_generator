from cfg import *
from model.cVAE import *
import numpy as np
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils import *

import argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument('--folder', help=' : npz motion folder name in motion/')
parser.add_argument('--period', help=' : model numbers to save per epoch')
args = parser.parse_args()

def main(argv, args):
    global learning_rate
    # ========= 학습 루프 =========
    model = cVAE(input_dim, cond_dim, hidden_dims, latent_dim)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.RAdam(model.parameters(), lr=learning_rate)

    FOLDER = f"motion/{args.folder}"
    npz_files = [f for f in os.listdir(FOLDER) if f.endswith(".npz")]
    npz_files.sort(key=lambda x : int(os.path.splitext(x)[0]))
    
    s_t_list=[]
    s_t_1_data_list=[]
    s_t_1_pred_list=[]
    
    """Load & stack motion data"""
    for fname in npz_files:
        path = os.path.join(FOLDER,fname)
        # print("file list : ",path)
        motion = np.load(path,"rb")
        
        #base body rotation의 초기 yaw를 0으로 변환
        q_base_init = torch.Tensor(motion["body_rotations"][0,0,:])
        q_base_init_yaw0 = remove_yaw_torch(q_base_init)
        
        # print(f"q_base_init : {q_base_init}")
        # print(f"q_base_init_yaw0 : {q_base_init_yaw0}")
        q_base_trasform = compute_q4(q_base_init, q_base_init_yaw0, torch.Tensor(motion["body_rotations"][:,0,:]))

        s_t = torch.cat([torch.Tensor(motion["dof_positions"]),
                        torch.Tensor(motion["dof_velocities"]),
                        torch.Tensor(motion["body_positions"][:,0,2:3]),
                        torch.flatten(torch.Tensor(motion["body_linear_velocities"][:,0,:]), 1),
                        torch.flatten(torch.Tensor(motion["body_angular_velocities"][:,0,:]), 1),
                        #  torch.flatten(torch.Tensor(motion["body_rotations"][:,0,:]), 1),
                        # quat2rot6d(torch.Tensor(motion["body_rotations"][:,0,:])), # includes conversion from quat to rot6d for continuity
                        quat2rot6d(q_base_trasform) # includes conversion from quat to rot6d for continuity
                        ], dim=1)
        
        s_t_list.append(s_t)
        
        s_t_1_data= torch.zeros_like(s_t)
        s_t_1_data[0,:] = s_t[0,:]
        s_t_1_data[1:,:] = s_t[:-1,:]
        s_t_1_data_list.append(s_t_1_data)
        
        s_t_1_pred = torch.zeros_like(s_t)
        s_t_1_pred_list.append(s_t_1_pred) 

    print(f"number of loaded motion files : {len(s_t_list)}")
    for i in range(len(s_t_list)):
        print(f"file {i} size : {s_t_list[i].size()}")

    """Extract good indices for random motion transition frame sampling"""
    row_indices = []
    current_offset = 0
    for t in s_t_list:
        num_rows = t.shape[0]
        # 해당 텐서의 마지막 row 전까지 인덱스 추가
        row_indices.extend(range(current_offset, current_offset + num_rows - 1))
        current_offset += num_rows
        
    # for motion_number in range(len(s_t_list)):
    sampler = BatchSampler(
        SubsetRandomSampler(row_indices),
        batch_size,
        drop_last=False,
    )

    # 전체 쌓은 텐서 (필요하다면 실제로 사용 가능)
    s_t_list = torch.cat(s_t_list, dim=0)
    s_t_1_data_list = torch.cat(s_t_1_data_list, dim=0)
    s_t_1_pred_list = torch.cat(s_t_1_pred_list, dim=0)
    # print("s_t_list size : ",s_t_list.size())
    # print("s_t_data_list size : ",s_t_1_data_list.size())
    # print("s_t_pred_list size : ",s_t_1_pred_list.size())
    # print("row indices : ",row_indices)
    
    """Data normalization"""
    # s_t_list[:,:joint_pos_dim], joint_pos_mean, joint_pos_std = data_normalization(s_t_list[:,:joint_pos_dim])
    # s_t_1_data_list[:,:joint_pos_dim] = (s_t_1_data_list[:,:joint_pos_dim] - joint_pos_mean)/joint_pos_std
    
    # s_t_list[:,joint_pos_dim:joint_pos_dim+joint_vel_dim], joint_vel_mean, joint_vel_std = data_normalization(s_t_list[:,joint_pos_dim:joint_pos_dim+joint_vel_dim])
    # s_t_1_data_list[:,joint_pos_dim:joint_pos_dim+joint_vel_dim] = (s_t_1_data_list[:,joint_pos_dim:joint_pos_dim+joint_vel_dim] - joint_vel_mean)/joint_vel_std
    
    # s_t_list[:,joint_pos_dim+joint_vel_dim:joint_pos_dim+joint_vel_dim+base_pos_dim], base_pos_mean, base_pos_std = data_normalization(s_t_list[:,joint_pos_dim+joint_vel_dim:joint_pos_dim+joint_vel_dim+base_pos_dim])
    # s_t_1_data_list[:,joint_pos_dim+joint_vel_dim:joint_pos_dim+joint_vel_dim+base_pos_dim] = (s_t_1_data_list[:,joint_pos_dim+joint_vel_dim:joint_pos_dim+joint_vel_dim+base_pos_dim] - base_pos_mean)/base_pos_std
        
    # s_t_list[:,joint_pos_dim+joint_vel_dim+base_pos_dim+base_rot_dim:joint_pos_dim+joint_vel_dim+base_pos_dim+base_rot_dim+base_lin_vel_dim], base_lin_vel_mean, base_lin_vel_std = data_normalization(s_t_list[:,joint_pos_dim+joint_vel_dim+base_pos_dim+base_rot_dim:joint_pos_dim+joint_vel_dim+base_pos_dim+base_rot_dim+base_lin_vel_dim])
    # s_t_1_data_list[:,joint_pos_dim+joint_vel_dim+base_pos_dim+base_rot_dim:joint_pos_dim+joint_vel_dim+base_pos_dim+base_rot_dim+base_lin_vel_dim] = (s_t_1_data_list[:,joint_pos_dim+joint_vel_dim+base_pos_dim+base_rot_dim:joint_pos_dim+joint_vel_dim+base_pos_dim+base_rot_dim+base_lin_vel_dim] - base_lin_vel_mean)/base_lin_vel_std    

    # s_t_list[:,joint_pos_dim+joint_vel_dim+base_pos_dim+base_rot_dim+base_lin_vel_dim:joint_pos_dim+joint_vel_dim+base_pos_dim+base_rot_dim+base_lin_vel_dim+base_ang_vel_dim], base_ang_vel_mean, base_ang_vel_std = data_normalization(s_t_list[:,joint_pos_dim+joint_vel_dim+base_pos_dim+base_rot_dim+base_lin_vel_dim:joint_pos_dim+joint_vel_dim+base_pos_dim+base_rot_dim+base_lin_vel_dim+base_ang_vel_dim])
    # s_t_1_data_list[:,joint_pos_dim+joint_vel_dim+base_pos_dim+base_rot_dim+base_lin_vel_dim:joint_pos_dim+joint_vel_dim+base_pos_dim+base_rot_dim+base_lin_vel_dim+base_ang_vel_dim] = (s_t_1_data_list[:,joint_pos_dim+joint_vel_dim+base_pos_dim+base_rot_dim+base_lin_vel_dim:joint_pos_dim+joint_vel_dim+base_pos_dim+base_rot_dim+base_lin_vel_dim+base_ang_vel_dim] - base_ang_vel_mean)/base_ang_vel_std        
    
    # means = [joint_pos_mean, joint_vel_mean, base_pos_mean, base_lin_vel_mean, base_ang_vel_mean]
    # stds = [joint_pos_std, joint_vel_std, base_pos_std, base_lin_vel_std, base_ang_vel_std]
    s_t_list[:,:-base_rot_dim], means, stds = data_normalization(s_t_list[:,:-base_rot_dim])
    s_t_1_data_list[:,:-base_rot_dim] = (s_t_1_data_list[:,:-base_rot_dim]-means)/stds
    # means = [0,0,0,0,0]
    # stds = [1,1,1,1,1]
    #make output folder
    folder_path = get_unique_folder_name(f"output/{args.folder}")
    os.makedirs(folder_path)
    
    # print("means : ",torch.cat(means,dim=0))
    # print("stds : ",torch.cat(stds,dim=0))
    # print("s_0 : ",s_t_list[0,:])
    
    # VAE structure 저장
    np.savez(f"{folder_path}/cfg",
             input_dim = input_dim,
             cond_dim = cond_dim,
             latent_dim = latent_dim,
             hidden_dims = hidden_dims,
             num_epochs = num_epochs,
             batch_size = batch_size,
            #  means = torch.cat(means,dim=0),
            #  stds = torch.cat(stds,dim=0),
             means = means,
             stds = stds,
             s_0 = s_t_list[0,:])
    
    for epoch in range(num_epochs):
        total_loss = 0
        recon_loss = 0
        kl_div = 0
        
        trans_start = 0.2
        trans_end = 0.4
        p = 1 - (epoch - trans_start* num_epochs) / ((trans_end - trans_start) * num_epochs)# s_t_1_data를 고를 확률
        
        #gain scheduling
        learning_rate = lr_scheduler(lr_max, lr_min, epoch, num_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        scheduled_beta = beta_scheduler(beta, epoch, num_epochs/5)

        for mini_batch, indices in enumerate(sampler):
            # print(f"sampler incides : {indices}, mini batch : {mini_batch}")
            #sample [indices]-th motion in random order
            
            optimizer.zero_grad()
            
            rand = torch.rand(1)
            if rand < p:
                s_t_1 = s_t_1_data_list[indices]
            else:
                s_t_1 = s_t_1_pred_list[indices]
                
            s_t_pred, mu, logvar = model(s_t_list[indices], s_t_1)
            
            loss , recon_loss_1, kl_div_1 = vae_loss(s_t_pred, s_t_list[indices], mu, logvar, scheduled_beta)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            recon_loss += recon_loss_1.item()
            kl_div += kl_div_1.item()
            
            s_t_1_pred_list[np.array(indices)+1] = s_t_pred.detach().clone()
        # scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch : {epoch+1}/{num_epochs}, Total Loss: {total_loss:.4f}, Recon Loss : {recon_loss:.4f}, KL divergence :  {kl_div:.4f}, Learning rate : {learning_rate:.4f}, Beta : {scheduled_beta:.4f}")
        
        # 모델 저장
        if ((epoch+1) % int(args.period) == 0):
            model_path = f"{folder_path}/cvae_model_{epoch+1}.pt"
            torch.save({
                'encoder': model.encoder.state_dict(),
                'decoder': model.decoder.state_dict()
            }, model_path)

            print(f"{epoch+1} epoch 모델이 '{model_path}'에 저장되었습니다.")
    
if __name__ == "__main__":
    argv = sys.argv
    main(argv, args)
