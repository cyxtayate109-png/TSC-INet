# data_path = "./data"
data_path = "D:\Python_Project\data"
class opts_ntu_60_cross_view():

    def __init__(self):
        # Sequence based model (Optimized: 1.5M params, 4.2G FLOPs)
        self.encoder_args = {
            "hidden_size": 256,          # Reduced from 2048
            "num_head": 4,               # Reduced from 8
            "num_layer": 1,
            "num_class": 60,
            "gcn_base_channel": 32       # GCN base channel
        }

        # feeder
        self.train_feeder_args = {
            "data_path": data_path + "/NTU-RGB-D-60-AGCN/xview/train_data_joint.npy",
            "label_path": data_path + "/NTU-RGB-D-60-AGCN/xview/train_label.pkl",
            'num_frame_path': data_path + "/NTU-RGB-D-60-AGCN/xview/train_num_frame.npy",
            'l_ratio': [0.95],
            'input_size': 64
        }

        self.test_feeder_args = {

            'data_path': data_path + "/NTU-RGB-D-60-AGCN/xview/val_data_joint.npy",
            'label_path': data_path + "/NTU-RGB-D-60-AGCN/xview/val_label.pkl",
            'num_frame_path': data_path + "/NTU-RGB-D-60-AGCN/xview/val_num_frame.npy",
            'l_ratio': [0.95],
            'input_size': 64
        }


class opts_ntu_60_cross_subject():

    def __init__(self):
        # Sequence based model (Optimized: 1.5M params, 4.2G FLOPs)
        self.encoder_args = {
            "hidden_size": 256,          # Reduced from 2048
            "num_head": 4,               # Reduced from 8
            "num_layer": 1,
            "num_class": 60,
            "gcn_base_channel": 32       # GCN base channel
        }

        # feeder
        self.train_feeder_args = {
            "data_path": data_path + "/NTU-RGB-D-60-AGCN/xsub/train_data_joint.npy",
            "label_path": data_path + "/NTU-RGB-D-60-AGCN/xsub/train_label.pkl",
            'num_frame_path': data_path + "/NTU-RGB-D-60-AGCN/xsub/train_num_frame.npy",
            'l_ratio': [0.95],
            'input_size': 64
        }

        self.test_feeder_args = {

            'data_path': data_path + "/NTU-RGB-D-60-AGCN/xsub/val_data_joint.npy",
            'label_path': data_path + "/NTU-RGB-D-60-AGCN/xsub/val_label.pkl",
            'num_frame_path': data_path + "/NTU-RGB-D-60-AGCN/xsub/val_num_frame.npy",
            'l_ratio': [0.95],
            'input_size': 64
        }


class opts_ntu_120_cross_subject():
    def __init__(self):
        # Sequence based model (Optimized: 1.5M params, 4.2G FLOPs)
        self.encoder_args = {
            "hidden_size": 256,          # Reduced from 2048
            "num_head": 4,               # Reduced from 8
            "num_layer": 1,
            "num_class": 120,
            "gcn_base_channel": 32       # GCN base channel
        }

        # feeder
        self.train_feeder_args = {
            "data_path": data_path + "/NTU-RGB-D-120-AGCN/xsub/train_data_joint.npy",
            "label_path": data_path + "/NTU-RGB-D-120-AGCN/xsub/train_label.pkl",
            'num_frame_path': data_path + "/NTU-RGB-D-120-AGCN/xsub/train_num_frame.npy",
            'l_ratio': [0.95],
            'input_size': 64
        }

        self.test_feeder_args = {

            'data_path': data_path + "/NTU-RGB-D-120-AGCN/xsub/val_data_joint.npy",
            'label_path': data_path + "/NTU-RGB-D-120-AGCN/xsub/val_label.pkl",
            'num_frame_path': data_path + "/NTU-RGB-D-120-AGCN/xsub/val_num_frame.npy",
            'l_ratio': [0.95],
            'input_size': 64
        }


class opts_ntu_120_cross_setup():

    def __init__(self):
        # Sequence based model (Optimized: 1.5M params, 4.2G FLOPs)
        self.encoder_args = {
            "hidden_size": 256,          # Reduced from 2048
            "num_head": 4,               # Reduced from 8
            "num_layer": 1,
            "num_class": 120,
            "gcn_base_channel": 32       # GCN base channel
        }

        # feeder
        self.train_feeder_args = {
            "data_path": data_path + "/NTU-RGB-D-120-AGCN/xsetup/train_data_joint.npy",
            "label_path": data_path + "/NTU-RGB-D-120-AGCN/xsetup/train_label.pkl",
            'num_frame_path': data_path + "/NTU-RGB-D-120-AGCN/xsetup/train_num_frame.npy",
            'l_ratio': [0.95],
            'input_size': 64
        }

        self.test_feeder_args = {

            'data_path': data_path + "/NTU-RGB-D-120-AGCN/xsetup/val_data_joint.npy",
            'label_path': data_path + "/NTU-RGB-D-120-AGCN/xsetup/val_label.pkl",
            'num_frame_path': data_path + "/NTU-RGB-D-120-AGCN/xsetup/val_num_frame.npy",
            'l_ratio': [0.95],
            'input_size': 64
        }


class opts_pku_part1_cross_subject():

    def __init__(self):
        # Sequence based model (Optimized: 1.5M params, 4.2G FLOPs)
        self.encoder_args = {
            "hidden_size": 256,          # Reduced from 2048
            "num_head": 4,               # Reduced from 8
            "num_layer": 1,
            "num_class": 51,
            "gcn_base_channel": 32       # GCN base channel
        }

        # feeder
        self.train_feeder_args = {
            "data_path": data_path + "/pku_v1/xsub/train_data_joint.npy",
            "label_path": data_path + "/pku_v1/xsub/train_label.pkl",
            "num_frame_path": data_path + "/pku_v1/xsub/train_num_frame.npy",
            'l_ratio': [0.95],
            'input_size': 64
        }

        self.test_feeder_args = {

            "data_path": data_path + "/pku_v1/xsub/val_data_joint.npy",
            "label_path": data_path + "/pku_v1/xsub/val_label.pkl",
            "num_frame_path": data_path + "/pku_v1/xsub/val_num_frame.npy",
            'l_ratio': [0.95],
            'input_size': 64
        }


class opts_pku_part2_cross_subject():
    def __init__(self):
        # Sequence based model (Optimized: 1.5M params, 4.2G FLOPs)
        self.encoder_args = {
            "hidden_size": 256,          # Reduced from 2048
            "num_head": 4,               # Reduced from 8
            "num_layer": 1,
            "num_class": 51,
            "gcn_base_channel": 32       # GCN base channel
        }

        # feeder
        self.train_feeder_args = {
            "data_path": data_path + "/pku_v2/xsub/train_data_joint.npy",
            "label_path": data_path + "/pku_v2/xsub/train_label.pkl",
            "num_frame_path": data_path + "/pku_v2/xsub/train_num_frame.npy",
            'l_ratio': [0.95],
            'input_size': 64
        }

        self.test_feeder_args = {

            "data_path": data_path + "/pku_v2/xsub/val_data_joint.npy",
            "label_path": data_path + "/pku_v2/xsub/val_label.pkl",
            "num_frame_path": data_path + "/pku_v2/xsub/val_num_frame.npy",
            'l_ratio': [0.95],
            'input_size': 64
        }
