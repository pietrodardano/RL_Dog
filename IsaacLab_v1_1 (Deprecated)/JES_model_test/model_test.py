import torch
import torch.nn as nn
import numpy as np

class Backup(nn.Module):
    def __init__(self):
        super(Backup, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(37, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 12),
        )
    
    def forward(self, x):
        return self.layers(x)

PATH = '/home/rl_sim/RL_Dog/models/FULL_STATE__NN_v3.pt'
dict = torch.load(PATH, map_location=torch.device('cuda'), weights_only=False)

# Change dict keys
new_keys = ["layers.0.weight", "layers.0.bias", "layers.2.weight", "layers.2.bias",
            "layers.4.weight", "layers.4.bias", "layers.6.weight", "layers.6.bias"]


old_keys = ["l1.weight", "l1.bias",
            "l3.weight", "l3.bias", "l5.weight", "l5.bias",
            "mean_layer.weight", "mean_layer.bias"] 

# old_keys = ["net.0.weight", "net.0.bias",
#             "net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias",
#             "mean_layer.weight", "mean_layer.bias"] 

from collections import OrderedDict
new_state_dict = OrderedDict()
new_dict = 0
for k, v in dict.items():
    if k in old_keys:
        name = new_keys[new_dict] # remove `module.`
        new_state_dict[name] = v
        print('v.shape',v.shape)
        #print('v',v)
        new_dict += 1

joint_def = np.array([  0.1000, -0.1000,  0.1000, -0.1000,
                        0.8000,  0.8000,  1.0000,  1.0000,
                       -1.5000, -1.5000, -1.5000, -1.5000])

model = Backup()
model.load_state_dict(new_state_dict)

vel = np.zeros(18)
th_hip = 0.
th_thigh = 0.75
th_calf = -1.5

# my input
data_order = np.array([0.,    0.,  0.39275, 1.0, 0.0, 0.0, 0.0, # trunk pos
                       0, 0, 0, 0, 0, 0,                    # trunk vel
                       th_hip,   th_hip,   th_hip,   th_hip, # hip pos             
                       th_thigh, th_thigh, th_thigh, th_thigh, # thigh pos
                       th_calf,  th_calf,  th_calf,  th_calf, # calf pos
                       0,0,0, 0,0,0, 0,0,0, 0,0,0]) # vels

print("MY INPUT TO NN:\n",data_order)
data_order[13:25] = data_order[13:25]# - joint_def
a = torch.from_numpy(data_order)
a = a.to(torch.float32)
print('joint_desired - joint_default',a)
res = model.forward(a)
print('network_output',res)
joint_def_tensor = torch.from_numpy(joint_def)
joint_def_tensor = joint_def_tensor.to(torch.float32)
pos_backup_2 = res + joint_def_tensor
print('network_output + joint_default',pos_backup_2)

print('DIFFERENCE WRT DEF_J_pos', pos_backup_2.to(torch.float32) -torch.from_numpy(joint_def))