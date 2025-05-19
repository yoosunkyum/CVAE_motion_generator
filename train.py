from cfg import *
from model.cVAE import *
import numpy as np
import torch.optim as optim


if __name__ == "__main__":
    # ========= 학습 루프 =========
    model = cVAE(input_dim, cond_dim, hidden_dims, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    motion = np.load("motion/motion_example.npz","rb")
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

    # print(s_t[0,:],s_t[1,:])
    # print(s_t_1[1,:],s_t_1[2,:])


    dataset = torch.utils.data.TensorDataset(s_t, s_t_1_data, s_t_1_pred)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        total_loss = 0
        s_t_1_temp = []

        for x, y, z in loader:
            optimizer.zero_grad()

            # s_t_1 = torch.zeros_like(x)
            # if epoch < 0.2 * num_epochs:
            #     s_t_1 = y
            # elif epoch < 0.6 * num_epochs:
            #     p = 1 - (epoch - 0.2* num_epochs) / ((0.6 - 0.2) * num_epochs)# s_t_1 고를 확률
            #     if torch.rand(1) < 1:
            #         s_t_1 = y
            #     else:
            #         s_t_1 = z
            # else:
            #     s_t_1 = z

            s_t_1 = y

            s_t_pred, mu, logvar = model(x, s_t_1)
            loss = vae_loss(s_t_pred, x, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            s_t_1_temp.append(s_t_pred)

        s_t_1_temp =torch.cat(s_t_1_temp, dim=0)
        # print("s_t_1_save.size() : ", s_t_1_temp.size())

        s_t_1_save = torch.zeros_like(s_t_1_temp)
        s_t_1_save[0,:] = s_t[0,:]
        s_t_1_save[1:,:] = s_t_1_temp[:-1,:]
        # print("s_t_1_temp.size() : ", s_t_1_temp.size())

        dataset=torch.utils.data.TensorDataset(s_t, s_t_1_data, s_t_1_save)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    # 학습 루프 끝에 모델 저장 추가
    model_path = "output/cvae_model.pth"

    torch.save({
        'encoder': model.encoder.state_dict(),
        'decoder': model.decoder.state_dict()
    }, model_path)

    print(f"모델이 '{model_path}'에 저장되었습니다.")