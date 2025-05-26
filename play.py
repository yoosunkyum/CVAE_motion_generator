import mujoco
from mujoco import viewer
import numpy as np
import torch
from cfg import *
from model.cVAE import * 
import time
from utils import *

import argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument('--folder', help=' : .npz file names in motion/ (ignore typing .npz)')
parser.add_argument('--period', help=' : model number to load')
args = parser.parse_args()


def load_trained_decoder(model_path, input_dim, cond_dim, hidden_dims, latent_dim):
    # decoder = Decoder(output_dim=input_dim, cond_dim=cond_dim, hidden_dims=hidden_dims[::-1], latent_dim=latent_dim)
    decoder = MixedDecoder(input_dim, cond_dim, hidden_dims[0], latent_dim, 6)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()
    return decoder

def visualize_in_mujoco_with_trained_decoder(model_path, xml_path, s_0, input_dim, cond_dim, hidden_dims, latent_dim):
    decoder = load_trained_decoder(model_path, input_dim, cond_dim, hidden_dims, latent_dim)

    # Mujoco 초기화
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    
    target_body = "pelvis"    
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target_body)

    with viewer.launch_passive(m, d) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = body_id
        v.cam.distance = 5.0
        v.cam.elevation = -30
        v.cam.azimuth = 90
        
        s_prev = s_0.unsqueeze(0)
        d.qpos[0]=0.0
        d.qpos[1]=0.0
        # i = 0
        # z = torch.randn(1, latent_dim)
        while True:
            # i+=1
            # if (i % 30 == 0):
            z = torch.randn(1, latent_dim)
            # print(z.size())
            with torch.no_grad():
                s_curr = decoder(z, s_prev)
                
            s_curr_np = s_curr[0].cpu().numpy()
            
            # s_curr_lpf = lpf(s_curr, s_prev, 0.5)
            # s_curr_np = s_curr_lpf[0].cpu().numpy()            
            
            body_pos = s_curr_np[joint_pos_dim + joint_vel_dim 
                                :joint_pos_dim + joint_vel_dim + base_pos_dim]
            # body_quat = s_curr_np[joint_pos_dim + joint_vel_dim + base_pos_dim 
            #                      :joint_pos_dim + joint_vel_dim + base_pos_dim + base_rot_dim]
            body_quat = rot6d2quat(torch.Tensor([s_curr_np[joint_pos_dim + joint_vel_dim + base_pos_dim 
                                 :joint_pos_dim + joint_vel_dim + base_pos_dim + base_rot_dim]]))
            # d.qpos[:3] = body_pos
            d.qpos[2]=body_pos

            d.qpos[3:7] = body_quat
            
            d.qpos[7:] = s_curr_np[0:joint_pos_dim]
            
            d.qvel[:3] = s_curr_np[joint_pos_dim + joint_vel_dim + base_pos_dim + base_rot_dim 
                                  :joint_pos_dim + joint_vel_dim + base_pos_dim + base_rot_dim + base_lin_vel_dim]
            d.qvel[3:6] = s_curr_np[joint_pos_dim + joint_vel_dim + base_pos_dim + base_rot_dim + base_lin_vel_dim
                                   :joint_pos_dim + joint_vel_dim + base_pos_dim + base_rot_dim + base_lin_vel_dim + base_ang_vel_dim]
            d.qvel[6:] = s_curr_np[joint_pos_dim : joint_pos_dim + joint_vel_dim]
            
            d.qpos[:2] += d.qvel[:2] * 1/30
            
            # d.qpos[3:7] =integrate_quaternion(torch.Tensor(d.qpos[3:7]),torch.Tensor(d.qvel[3:6]),1/30)

            mujoco.mj_step(m, d)
            v.sync()

            s_prev = s_curr
            time.sleep(1/30)
            

        print("시각화 완료. ESC로 종료.")

def main(argv, args):
    FOLDER = args.folder
    folder_path = f"output/{FOLDER}"
    
    model_path = f"{folder_path}/cvae_model_{args.period}.pt"
    cfg = np.load(f"{folder_path}/cfg.npz","rb")

    # xml_path = "assets/h1/h1.xml"
    xml_path = "assets/g1/g1_29dof_rev_1_0.xml"
  
    s_0 = torch.Tensor(cfg["s_0"])

    visualize_in_mujoco_with_trained_decoder(
        model_path=model_path,
        xml_path=xml_path,
        s_0=s_0,
        input_dim=cfg["input_dim"],
        cond_dim=cfg["cond_dim"],
        hidden_dims=cfg["hidden_dims"],
        latent_dim=cfg["latent_dim"],
    )
    
if __name__ == "__main__":
    argv = sys.argv
    main(argv, args)
