# Sequence based model arguments
# Optimized for 1.5M params and 4.2G FLOPs (as reported in paper Table I)
encoder_arguments = {
   "hidden_size": 256,      # Reduced from 2048 to 256 for lightweight model
   "num_head": 4,           # Reduced from 8 to 4 heads
   "num_layer": 1,          # Keep 1 layer
   "num_class": 128,        # Keep projection dimension
   "gcn_base_channel": 32   # GCN base channel (reduced from 64)
}

# data_path = "./data"
data_path = "D:\Python_Project\data"

class  opts_ntu_60_cross_view():

  def __init__(self):

   self.encoder_args = encoder_arguments

   # feeder
   self.train_feeder_args = {
     "data_path": data_path + "/NTU-RGB-D-60-AGCN/xview/train_data_joint.npy",
     "num_frame_path": data_path + "/NTU-RGB-D-60-AGCN/xview/train_num_frame.npy",
     "l_ratio": [0.1, 1],
     "input_size": 64
   }

class  opts_ntu_60_cross_subject():

  def __init__(self):

   self.encoder_args = encoder_arguments
   
   # feeder
   self.train_feeder_args = {
     "data_path": data_path + "/NTU-RGB-D-60-AGCN/xsub/train_data_joint.npy",
     "num_frame_path": data_path + "/NTU-RGB-D-60-AGCN/xsub/train_num_frame.npy",
     "l_ratio": [0.1, 1],
     "input_size": 64
   }

class  opts_ntu_120_cross_subject():

  def __init__(self):

   self.encoder_args = encoder_arguments
   
   # feeder
   self.train_feeder_args = {
     "data_path": data_path + "/NTU-RGB-D-120-AGCN/xsub/train_data_joint.npy",
     "num_frame_path": data_path + "/NTU-RGB-D-120-AGCN/xsub/train_num_frame.npy",
     "l_ratio": [0.1, 1],
     "input_size": 64
   }

class  opts_ntu_120_cross_setup():

  def __init__(self):

   self.encoder_args = encoder_arguments
   
   # feeder
   self.train_feeder_args = {
     "data_path": data_path + "/NTU-RGB-D-120-AGCN/xsetup/train_data_joint.npy",
     "num_frame_path": data_path + "/NTU-RGB-D-120-AGCN/xsetup/train_num_frame.npy",
     "l_ratio": [0.1, 1],
     "input_size": 64
   }


class  opts_pku_part1_cross_subject():

  def __init__(self):

   self.encoder_args = encoder_arguments
   
   # feeder
   self.train_feeder_args = {
     "data_path": data_path + "/pku_v1/xsub/train_data_joint.npy",
     "num_frame_path": data_path + "/pku_v1/xsub/train_num_frame.npy",
     "l_ratio": [0.1, 1],
     "input_size": 64
   }

class  opts_pku_part2_cross_subject():

  def __init__(self):

   self.encoder_args = encoder_arguments
   
   # feeder
   self.train_feeder_args = {
     "data_path": data_path + "/pku_v2/xsub/train_data_joint.npy",
     "num_frame_path": data_path + "/pku_v2/xsub/train_num_frame.npy",
     "l_ratio": [0.1, 1],
     "input_size": 64
   }

