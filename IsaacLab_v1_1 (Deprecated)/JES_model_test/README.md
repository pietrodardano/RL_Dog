# RESULTS FROM RUNNING THE PYTHON SCRIPT
```

v.shape torch.Size([256, 37])
v.shape torch.Size([256])
v.shape torch.Size([256, 256])
v.shape torch.Size([256])
v.shape torch.Size([128, 256])
v.shape torch.Size([128])
v.shape torch.Size([12, 128])
v.shape torch.Size([12])

MY INPUT TO NN:
 [ 0.       0.       0.39275  1.       0.       0.       0.       0.
  0.       0.       0.       0.       0.       0.      -0.       0.
 -0.       0.8      0.8      0.8      0.8     -1.5     -1.5     -1.5
 -1.5      0.       0.       0.       0.       0.       0.       0.
  0.       0.       0.       0.       0.     ]

joint_desired - joint_default tensor([ 0.0000,  0.0000,  0.3927,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.1000,  0.1000, -0.1000,
         0.1000,  0.0000,  0.0000, -0.2000, -0.2000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
         0.0000,  0.0000,  0.0000,  0.0000,  0.0000])

network_output tensor([ 0.1575,  0.1188,  0.0934,  0.1570,  0.2530,  0.1957,  0.0959,  0.1094,
         0.1503,  0.2678, -0.0733, -0.0957], grad_fn=<EluBackward0>)

network_output + joint_default tensor([ 0.2575,  0.0188,  0.1934,  0.0570,  1.0530,  0.9957,  1.0959,  1.1094,
        -1.3497, -1.2322, -1.5733, -1.5957], grad_fn=<AddBackward0>)
        
DIFFERENCE WRT DEF_J_pos tensor([ 0.1575,  0.1188,  0.0934,  0.1570,  0.2530,  0.1957,  0.0959,  0.1094,
         0.1503,  0.2678, -0.0733, -0.0957], dtype=torch.float64,
       grad_fn=<SubBackward0>)

```