from cfg import *
from model.cVAE import *
import numpy as np
import torch.optim as optim


if __name__ == "__main__":
    # ========= 학습 루프 =========
    model = cVAE(input_dim, cond_dim, hidden_dims, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.9999)

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

    # print(s_t[0,0],s_t[1,0], s_t[2,0], s_t[-1,0])
    # print(s_t_1_data[0,0], s_t_1_data[1,0],s_t_1_data[2,0], s_t_1_data[-1,0])


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
        rand = torch.rand(1)
        # print("epoch : ",epoch, ", p : ", p, ", rand : ",rand)
        for x, y, z in loader:
            optimizer.zero_grad()
           
            s_t_1 = torch.zeros_like(x)
            if epoch < 0.2 * num_epochs:
                s_t_1 = y
            elif epoch < 0.6 * num_epochs:
                if rand < p:
                    # print("select dataset")
                    s_t_1 = y
                else:
                    # print("select prediction")
                    s_t_1 = z
            else:
                s_t_1 = z

            # s_t_1 = z

            s_t_pred, mu, logvar = model(x, s_t_1)
            loss , recon_loss_1, kl_div_1= vae_loss(s_t_pred, x, mu, logvar, beta)
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
            print(f"Epoch : {epoch+1}/{num_epochs}, Total Loss: {total_loss:.4f}, Recon Loss : {recon_loss:.4f},  KL divergence :  {kl_div:.4f}")

    # 학습 루프 끝에 모델 저장 추가
    model_path = "output/cvae_model.pth"

    torch.save({
        'encoder': model.encoder.state_dict(),
        'decoder': model.decoder.state_dict()
    }, model_path)

    print(f"모델이 '{model_path}'에 저장되었습니다.")
