import torch
import torchvision.transforms.functional as F
import torch.nn.functional as F_pad

def resize_and_pad(image, target_size):
    """
    对图像进行保持宽高比的缩放，并在较短边上进行填充到目标大小。
    
    Args:
    - image: 需要处理的图像张量，形状为 [C, H, W]
    - target_size: 目标尺寸 (target_h, target_w)

    Returns:
    - 经过缩放和填充后的图像，形状为 [C, target_h, target_w]
    """
    _, h, w = image.shape
    target_h, target_w = target_size

    # 计算缩放比例，保持宽高比，先根据较短边进行缩放
    scale_factor = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    # 缩放图像
    resized_image = F.resize(image, (new_h, new_w))

    # 计算需要的填充量
    pad_h = target_h - new_h
    pad_w = target_w - new_w

    # 计算填充的上下和左右的大小，确保图像居中
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # 填充图像（使用常数值0进行填充，即黑色）
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    padded_image = torch.nn.functional.pad(resized_image, padding, mode='constant', value=0)
    
    return padded_image

def collate_fn(batch):
    data = {}
    imgs = []
    frame_ids = []
    img_paths = []
    sequences = []
    CP_mega_matrices = []
    targets = []

    cam_ks = []
    T_velo_2_cams = []

    frustums_masks = []
    frustums_class_dists = []

    scale_3ds = batch[0]["scale_3ds"]
    for scale_3d in scale_3ds:
        data["projected_pix_{}".format(scale_3d)] = []
        data["fov_mask_{}".format(scale_3d)] = []

    # 记录图像尺寸，找出最大图像尺寸作为参考
    max_height = 0
    max_width = 0

    for _, input_dict in enumerate(batch):
        if "img_path" in input_dict:
            img_paths.append(input_dict["img_path"])

        for key in data:
            data[key].append(torch.from_numpy(input_dict[key]))

        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).float())
        T_velo_2_cams.append(torch.from_numpy(input_dict["T_velo_2_cam"]).float())

        if "frustums_masks" in input_dict:
            frustums_masks.append(torch.from_numpy(input_dict["frustums_masks"]))
            frustums_class_dists.append(
                torch.from_numpy(input_dict["frustums_class_dists"]).float()
            )

        sequences.append(input_dict["sequence"])

        img = input_dict["img"]
        _, h, w = img.shape
        max_height = max(max_height, h)
        max_width = max(max_width, w)

        imgs.append(img)
        target = torch.from_numpy(input_dict["target"])
        targets.append(target)
        CP_mega_matrices.append(torch.from_numpy(input_dict["CP_mega_matrix"]))         
        frame_ids.append(input_dict["frame_id"])

    # 对所有图像进行检查和调整尺寸
    target_size = (max_height, max_width)
    adjusted_imgs = [resize_and_pad(img, target_size) if img.shape[1:] != target_size else img for img in imgs]

    ret_data = {
        "sequence": sequences,
        "frame_id": frame_ids,
        "frustums_class_dists": frustums_class_dists,
        "frustums_masks": frustums_masks,
        "cam_k": cam_ks,
        "T_velo_2_cam": T_velo_2_cams,
        "img": torch.stack(adjusted_imgs),  # 使用调整后的图像
        "img_path": img_paths,
        "CP_mega_matrices": CP_mega_matrices,
        "target": torch.stack(targets)
    }
    for key in data:
        ret_data[key] = data[key]

    return ret_data