import mujoco
from mujoco import viewer
import numpy as np
import torch
from cfg import *
from model.cVAE import * 
import time

def load_trained_decoder(model_path, input_dim, cond_dim, hidden_dims, latent_dim):
    decoder = Decoder(output_dim=input_dim, cond_dim=cond_dim, hidden_dims=hidden_dims[::-1], latent_dim=latent_dim)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()
    return decoder

def visualize_in_mujoco_with_trained_decoder(model_path, xml_path, s_0, input_dim, cond_dim, hidden_dims, latent_dim):
    decoder = load_trained_decoder(model_path, input_dim, cond_dim, hidden_dims, latent_dim)

    # Mujoco 초기화
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    with viewer.launch_passive(m, d) as v:
        s_prev = s_0.unsqueeze(0)

        while True:
            z = torch.randn(1, latent_dim)
            with torch.no_grad():
                s_curr = decoder(z, s_prev)

            s_curr_np = s_curr[0].cpu().numpy()
            
            #H1
            body_pos = s_curr_np[38:(38+3)]
            body_quat = s_curr_np[(38 + 3):(38 + 3 + 4)]

            d.qpos[:3] = body_pos
            d.qpos[3:7] = body_quat

            for i in range(7):
                d.qpos[i] = 0
            d.qpos[7:] = s_curr_np[0:19]
            
            d.qvel[:3] = s_curr_np[(38 + (3+4)):(38 + (3+4) + 3)]
            d.qvel[3:6] = s_curr_np[(38 + (3+4) + 3):(38 + (3+4) + 6)]
            d.qvel[6:] = s_curr_np[19:38]
            
            # #G1
            # body_pos = s_curr_np[58:(58+3)]
            # body_quat = s_curr_np[(58 + 3):(58 + 3 + 4)]

            # d.qpos[:3] = body_pos
            # d.qpos[3:7] = body_quat
            # d.qpos[7:] = s_curr_np[0:29]
            
            # d.qvel[:3] = s_curr_np[(58 + (3+4)):(58 + (3+4) + 3)]
            # d.qvel[3:6] = s_curr_np[(58 + (3+4) + 3):(58 + (3+4) + 6)]
            # d.qvel[6:] = s_curr_np[29:58]

            mujoco.mj_step(m, d)
            v.sync()

            s_prev = s_curr
            time.sleep(1/30)
            

        print("시각화 완료. ESC로 종료.")

if __name__ == "__main__":
    model_path = "output/cvae_model.pth"
    xml_path = "assets/h1/h1.xml"
    # xml_path = "assets/g1/g1_29dof_rev_1_0.xml"

    motion = np.load("motion/motion_example.npz","rb")
    # motion = np.load("motion/g1_boxing.npz","rb")
    s_t = torch.cat([torch.Tensor(motion["dof_positions"]),
                     torch.Tensor(motion["dof_velocities"]),
                     torch.flatten(torch.Tensor(motion["body_positions"][:,0,:]), 1),
                     torch.flatten(torch.Tensor(motion["body_rotations"][:,0,:]), 1),
                     torch.flatten(torch.Tensor(motion["body_linear_velocities"][:,0,:]), 1),
                     torch.flatten(torch.Tensor(motion["body_angular_velocities"][:,0,:]), 1)], dim=1)
    s_0 = s_t[0,:]

    visualize_in_mujoco_with_trained_decoder(
        model_path=model_path,
        xml_path=xml_path,
        s_0=s_0,
        input_dim=input_dim,
        cond_dim=cond_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
    )
