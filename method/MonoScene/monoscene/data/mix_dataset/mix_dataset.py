import torch
import os
import glob
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from monoscene.data.utils.helpers import (
    vox2pix,
    compute_local_frustums,
    compute_CP_mega_matrix,
)

CLASS_NUM = 11
class MixDataset(Dataset):
    def __init__(
        self,
        split,
        kitti360_root,
        kitti360_preprocess_root,
        waymo_root,
        waymo_preprocess_root,
        project_scale=2,
        frustum_size=4,
        color_jitter=None,
        fliplr=0.0,
    ):
        super().__init__()
        
        # MixData path
        self.kitti360_root = kitti360_root
        self.waymo_root = waymo_root
        self.kitti360_label_root = os.path.join(kitti360_preprocess_root, "labels")
        self.waymo_label_root = os.path.join(waymo_preprocess_root, "labels")
        
        # data splits
        kitti360_splits = {
            # "train": ["2013_05_28_drive_0004_sync", "2013_05_28_drive_0000_sync", "2013_05_28_drive_0010_sync","2013_05_28_drive_0002_sync", "2013_05_28_drive_0003_sync", "2013_05_28_drive_0005_sync", "2013_05_28_drive_0007_sync"],
            "train": ["2013_05_28_drive_0004_sync"],
            "val": ["2013_05_28_drive_0006_sync"],
            "test": ["2013_05_28_drive_0009_sync"],
        }
        self.kitti360_split = split
        self.kitti360_sequences = kitti360_splits[split]
        
        a=['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '096', '097', '098', '099', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '280', '281', '282', '283', '284', '285', '286', '287', '288', '289', '290', '291', '292', '293', '294', '295', '296', '297', '298', '299', '300', '301', '302', '303', '304', '305', '306', '307', '308', '309', '310', '311', '312', '313', '314', '315', '316', '317', '318', '319', '320', '321', '322', '323', '324', '325', '326', '327', '328', '329', '330', '331', '332', '333', '334', '335', '336', '337', '338', '339', '340', '341', '342', '343', '344', '345', '346', '347', '348', '349', '350', '351', '352', '353', '354', '355', '356', '357', '358', '359', '360', '361', '362', '363', '364', '365', '366', '367', '368', '369', '370', '371', '372', '373', '374', '375', '376', '377', '378', '379', '380', '381', '382', '383', '384', '385', '386', '387', '388', '389', '390', '391', '392', '393', '394', '395', '396', '397', '398', '399', '400', '401', '402', '403', '404', '405', '406', '407', '408', '409', '410', '411', '412', '413', '414', '415', '416', '417', '418', '419', '420', '421', '422', '423', '424', '425', '426', '427', '428', '429', '430', '431', '432', '433', '434', '435', '436', '437', '438', '439', '440', '441', '442', '443', '444', '445', '446', '447', '448', '449', '450', '451', '452', '453', '454', '455', '456', '457', '458', '459', '460', '461', '462', '463', '464', '465', '466', '467', '468', '469', '470', '471', '472', '473', '474', '475', '476', '477', '478', '479', '480', '481', '482', '483', '484', '485', '486', '487', '488', '489', '490', '491', '492', '493', '494', '495', '496', '497', '498', '499']
        b=['500', '501', '502', '503', '504', '505', '506', '507', '508', '509', '510', '511', '512', '513', '514', '515', '516', '517', '518', '519', '520', '521', '522', '523', '524', '525', '526', '527', '528', '529', '530', '531', '532', '533', '534', '535', '536', '537', '538', '539', '540', '541', '542', '543', '544', '545', '546', '547', '548', '549', '550', '551', '552', '553', '554', '555', '556', '557', '558', '559', '560', '561', '562', '563', '564', '565', '566', '567', '568', '569', '570', '571', '572', '573', '574', '575', '576', '577', '578', '579', '580', '581', '582', '583', '584', '585', '586', '587', '588', '589', '590', '591', '592', '593', '594', '595', '596', '597', '598', '599', '600', '601', '602', '603', '604', '605', '606', '607', '608', '609', '610', '611', '612', '613', '614', '615', '616', '617', '618', '619', '620', '621', '622', '623', '624', '625', '626', '627', '628', '629', '630', '631', '632', '633', '634', '635', '636', '637', '638', '639', '640', '641', '642', '643', '644', '645', '646', '647', '648', '649', '650', '651', '652', '653', '654', '655', '656', '657', '658', '659', '660', '661', '662', '663', '664', '665', '666', '667', '668', '669', '670', '671', '672', '673', '674', '675', '676', '677', '678', '679', '680', '681', '682', '683', '684', '685', '686', '687', '688', '689', '690', '691', '692', '693', '694', '695', '696', '697', '698', '699', '700', '701', '702', '703', '704', '705', '706', '707', '708', '709', '710', '711', '712', '713', '714', '715', '716', '717', '718', '719', '720', '721', '722', '723', '724', '725', '726', '727', '728', '729', '730', '731', '732', '733', '734', '735', '736', '737', '738', '739', '740', '741', '742', '743', '744', '745', '746', '747', '748', '749', '750', '751', '752', '753', '754', '755', '756', '757', '758', '759', '760', '761', '762', '763', '764', '765', '766', '767', '768', '769', '770', '771', '772', '773', '774', '775', '776', '777', '778', '779', '780', '781', '782', '783', '784', '785', '786', '787', '788', '789', '790', '791', '792', '793', '794', '795', '796', '797']
        c=['798', '799', '800', '801', '802', '803', '804', '805', '806', '807', '808', '809', '810', '811', '812', '813', '814', '815', '816', '817', '818', '819', '820', '821', '822', '823', '824', '825', '826', '827', '828', '829', '830', '831', '832', '833', '834', '835', '836', '837', '838', '839', '840', '841', '842', '843', '844', '845', '846', '847', '848', '849', '850', '851', '852', '853', '854', '855', '856', '857', '858', '859', '860', '861', '862', '863', '864', '865', '866', '867', '868', '869', '870', '871', '872', '873', '874', '875', '876', '877', '878', '879', '880', '881', '882', '883', '884', '885', '886', '887', '888', '889', '890', '891', '892', '893', '894', '895', '896', '897', '898', '899', '900', '901', '902', '903', '904', '905', '906', '907', '908', '909', '910', '911', '912', '913', '914', '915', '916', '917', '918', '919', '920', '921', '922', '923', '924', '925', '926', '927', '928', '929', '930', '931', '932', '933', '934', '935', '936', '937', '938', '939', '940', '941', '942', '943', '944', '945', '946', '947', '948', '949', '950', '951', '952', '953', '954', '955', '956', '957', '958', '959', '960', '961', '962', '963', '964', '965', '966', '967', '968', '969', '970', '971', '972', '973', '974', '975', '976', '977', '978', '979', '980', '981', '982', '983', '984', '985', '986', '987', '988', '989', '990', '991', '992', '993', '994', '995', '996', '997', '998', '999']
        # Selecting a random 10% of 'a'
        random_10_percent_a = random.sample(a, len(a) // 100)
        
        waymo_splits = {"train": random_10_percent_a, # a, 
                        "val": b,
                        "test": c}
        self.waymo_split = split
        self.waymo_sequences = waymo_splits[split]
        
        # same dataset setting
        self.frustum_size = frustum_size
        self.project_scale = project_scale
        self.output_scale = int(self.project_scale / 2)
        self.voxel_size = 0.2
        self.vox_origin = np.array([0, -25.6, -2])
        self.scene_size = (51.2, 51.2, 6.4)
        
        # different image HW
        self.kitti360_img_H = 376 / 2
        self.kitti360_img_W = 1408 / 2
        self.waymo_img_H = 640
        self.waymo_img_W = 960

        self.fliplr = fliplr

        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.scans = []
        
        # kitti calib
        for sequence in self.kitti360_sequences:
            
            calib = self.read_kitti360_calib()
            P = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix = P @ T_velo_2_cam

            v_path = os.path.join(
                self.kitti360_root, "data_2d_raw", sequence, "voxels", "*.bin"
            )

            for voxel_path in glob.glob(v_path):
                self.scans.append(
                    {
                        "dataset": "kitti360",
                        "sequence": sequence,
                        "P": P,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix": proj_matrix,
                        "voxel_path": voxel_path,
                    }
                )
        
        # waymo calib
        for sequence in self.waymo_sequences:
            
            calib = self.read_waymo_calib(
                os.path.join(self.waymo_root, "kitti_format_cam", "calib_1", sequence, "calib.txt")
            )
            P = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix = P @ T_velo_2_cam

            glob_path = os.path.join(
                self.waymo_root, "voxels", sequence, "*.npz"
            )

            for voxel_path in glob.glob(glob_path):
                self.scans.append(
                    {
                        "dataset": "waymo",
                        "sequence": sequence,
                        "P": P,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix": proj_matrix,
                        "voxel_path": voxel_path,
                    }
                )

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        
    def __getitem__(self, index):
        scan = self.scans[index]
        dataset = scan["dataset"]
        voxel_path = scan["voxel_path"]
        sequence = scan["sequence"]
        P = scan["P"]
        T_velo_2_cam = scan["T_velo_2_cam"]
        proj_matrix = scan["proj_matrix"]

        filename = os.path.basename(voxel_path)
        frame_id = os.path.splitext(filename)[0]

        data = {
            "frame_id": frame_id,
            "sequence": sequence,
            "P": P,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix": proj_matrix,
        }
        
        target_height, target_width = 640, 960

        scale_3ds = [self.output_scale, self.project_scale]
        data["scale_3ds"] = scale_3ds
        cam_k = P[0:3, 0:3]

        # 设置图像尺寸和路径，根据数据集来源
        if dataset == "kitti360":
            img_H, img_W = int(376 / 2), int(1408 / 2)
            scale_y = target_height / img_H  # 高度比例
            scale_x = target_width / img_W    # 宽度比例
            
            cam_k[0, 0] *= scale_x  # 调整 fx
            cam_k[1, 1] *= scale_y  # 调整 fy
            cam_k[0, 2] *= scale_x  # 调整 cx
            cam_k[1, 2] *= scale_y  # 调整 cy
            
            rgb_path = os.path.join(
                self.kitti360_root, "data_2d_raw", sequence, "image_00/data_rect", frame_id + ".png"
            )
            label_root = self.kitti360_label_root
        elif dataset == "waymo":
            img_H, img_W = int(1280 / 2), int(1920 / 2)  # 假设 Waymo 图像尺寸为 1280x1920
            rgb_path = os.path.join(
                self.waymo_root, "kitti_format_cam", "image_1", sequence, frame_id + ".png"
            )
            label_root = self.waymo_label_root
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        data["cam_k"] = cam_k

        # 计算投影和视场掩码
        for scale_3d in scale_3ds:
            projected_pix, fov_mask, pix_z = vox2pix(
                T_velo_2_cam,
                cam_k,
                self.vox_origin,
                self.voxel_size * scale_3d,
                self.waymo_img_W,
                self.waymo_img_H,
                self.scene_size,
            )
            data[f"projected_pix_{scale_3d}"] = projected_pix
            data[f"fov_mask_{scale_3d}"] = fov_mask
            data[f"pix_z_{scale_3d}"] = pix_z

        # 加载目标标签
        if dataset == "kitti360":
            target_1_path = os.path.join(label_root, sequence, frame_id + "_1_1.npy")
            target_8_path = os.path.join(label_root, sequence, frame_id + "_1_8.npy")
        elif dataset == "waymo":
            target_1_path = os.path.join(label_root, sequence, frame_id + "_1_1.npy")
            target_8_path = os.path.join(label_root, sequence, frame_id + "_1_8.npy")
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        target = np.load(target_1_path)
        data["target"] = target
        target_1_8 = np.load(target_8_path)
        CP_mega_matrix = compute_CP_mega_matrix(target_1_8)
        data["CP_mega_matrix"] = CP_mega_matrix

        # 计算视锥体掩码和类别分布
        projected_pix_output = data[f"projected_pix_{self.output_scale}"]
        pix_z_output = data[f"pix_z_{self.output_scale}"]
        frustums_masks, frustums_class_dists = compute_local_frustums(
            projected_pix_output,
            pix_z_output,
            target,
            self.waymo_img_W,
            self.waymo_img_H,
            dataset="kitti",
            n_classes=11,  # 请根据实际类别数调整
            size=self.frustum_size,
        )
        data["frustums_masks"] = frustums_masks
        data["frustums_class_dists"] = frustums_class_dists

        # 加载并处理图像
        img = Image.open(rgb_path).convert("RGB")

        if self.color_jitter is not None:
            img = self.color_jitter(img)

        img = np.array(img, dtype=np.float32, copy=False) / 255.0

        # 根据数据集裁剪或缩放图像
        if dataset == "kitti360":
            img = img[:376, :1408, :]  # 裁剪图像
            img = zoom(img,(target_height / 376, target_width / 1408, 1))

        # 缩放图像到一半尺寸
        if dataset == "waymo":
            img = zoom(img, (0.5, 0.5, 1))
        
        # padding 调整到 同一尺寸

        # 随机水平翻转
        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            for scale in scale_3ds:
                key = f"projected_pix_{scale}"
                data[key][:, 0] = img.shape[1] - 1 - data[key][:, 0]

        data["img"] = self.normalize_rgb(img)

        return data
    
    def __len__(self):
        return len(self.scans)
    
    @staticmethod
    def read_kitti360_calib():
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        P = np.array(
                [
                    552.554261 / 2,
                    0.000000,
                    682.049453 / 2,
                    0.000000,
                    0.000000,
                    552.554261 / 2,
                    238.769549 / 2,
                    0.000000,
                    0.000000,
                    0.000000,
                    1.000000,
                    0.000000,
                ]
            ).reshape(3, 4)

        cam2velo = np.array(
                [   
            0.04307104361,
            -0.08829286498,
            0.995162929,
            0.8043914418,
            -0.999004371,
            0.007784614041,
            0.04392796942,
            0.2993489574,
            -0.01162548558,
            -0.9960641394,
            -0.08786966659,
            -0.1770225824,
                ]
        ).reshape(3, 4)
        C2V = np.concatenate(
            [cam2velo, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0
        )
        # print("C2V: ", C2V)
        V2C = np.linalg.inv(C2V)
        # print("V2C: ", V2C)
        V2C = V2C[:3, :]
        # print("V2C: ", V2C)
  
        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        
        calib_out["P2"] = P
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = V2C
        return calib_out
    
    @staticmethod
    def read_waymo_calib(calib_path):
        """
        Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera

        calib_all["P2"][0] = calib_all["P2"][0] / 2
        calib_all["P2"][2] = calib_all["P2"][2] / 2
        calib_all["P2"][5] = calib_all["P2"][5] / 2
        calib_all["P2"][6] = calib_all["P2"][6] / 2

        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out