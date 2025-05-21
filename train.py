from cfg import *
from model.cVAE import *
import numpy as np
import torch.optim as optim
from utils import *

if __name__ == "__main__":
    # ========= 학습 루프 =========
    model = cVAE(input_dim, cond_dim, hidden_dims, latent_dim)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.RAdam(model.parameters(), lr=learning_rate)

    # motion = np.load("motion/motion_example.npz","rb")
    motion = np.load("motion/g1_boxing.npz","rb")
    s_t = torch.cat([torch.Tensor(motion["dof_positions"]),
                     torch.Tensor(motion["dof_velocities"]),
                     torch.flatten(torch.Tensor(motion["body_positions"][:,0,:]), 1),
                     torch.flatten(torch.Tensor(motion["body_rotations"][:,0,:]), 1),
                     torch.flatten(torch.Tensor(motion["body_linear_velocities"][:,0,:]), 1),
                     torch.flatten(torch.Tensor(motion["body_angular_velocities"][:,0,:]), 1)], dim=1)
    
    s_t_1_data = torch.zeros_like(s_t)
    s_t_1_data[0,:] = s_t[0,:]
    s_t_1_data[1:,:] = s_t[:-1,:]
    s_t_1_pred = torch.zeros_like(s_t)

    dataset = torch.utils.data.TensorDataset(s_t, s_t_1_data, s_t_1_pred)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        total_loss = 0
        recon_loss = 0
        kl_div = 0
        s_t_1_temp = []
        
        trans_start = 0.2
        trans_end = 0.4
        p = 1 - (epoch - trans_start* num_epochs) / ((trans_end - trans_start) * num_epochs)# s_t_1_data를 고를 확률
        
        #gain scheduling
        learning_rate = lr_scheduler(lr_max, lr_min, epoch, num_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        scheduled_beta = beta_scheduler(beta, epoch, num_epochs/5)

        for x, y, z in loader:
            optimizer.zero_grad()          
            s_t_1 = torch.zeros_like(x)

            rand = torch.rand(1)
            if rand < p:
                # print("select dataset")
                s_t_1 = y
            else:
                # print("select prediction")
                s_t_1 = z

            s_t_pred, mu, logvar = model(x, s_t_1)
            # loss , recon_loss_1, kl_div_1= vae_loss(s_t_pred, x, mu, logvar, beta)            
            loss , recon_loss_1, kl_div_1 = vae_loss(s_t_pred, x, mu, logvar, scheduled_beta)
            # print("loss : ",loss.item())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            recon_loss += recon_loss_1.item()
            kl_div += kl_div_1.item()
            
            s_t_1_temp.append(s_t_pred)

        s_t_1_temp =torch.cat(s_t_1_temp, dim=0)

        s_t_1_save = torch.zeros_like(s_t_1_temp)
        s_t_1_save[0,:] = s_t[0,:]
        s_t_1_save[1:,:] = s_t_1_temp[:-1,:]
        s_t_1_save = s_t_1_save.detach().clone()

        dataset=torch.utils.data.TensorDataset(s_t, s_t_1_data, s_t_1_save)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch : {epoch+1}/{num_epochs}, Total Loss: {total_loss:.4f}, Recon Loss : {recon_loss:.4f},  KL divergence :  {kl_div:.4f}, learning rate : {learning_rate:.4f}, beta : {scheduled_beta:.4f}")

    # 학습 루프 끝에 모델 저장 추가
    folder_path = get_unique_folder_name("output/g1_boxing")
    os.makedirs(folder_path)
    model_path = f"{folder_path}/cvae_model.pth"

    # 모델 저장
    torch.save({
        'encoder': model.encoder.state_dict(),
        'decoder': model.decoder.state_dict()
    }, model_path)
    
    # VAE structure 저장
    np.savez(f"{folder_path}/cfg",
             input_dim = input_dim,
             cond_dim = cond_dim,
             latent_dim = latent_dim,
             hidden_dims = hidden_dims,
             num_epochs = num_epochs,
             batch_size = batch_size,
             s_0 = s_t[0,:])
    

    print(f"모델이 '{model_path}'에 저장되었습니다.")
