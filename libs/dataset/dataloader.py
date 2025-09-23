import os
import sys
from PIL import Image
from typing import NamedTuple, Optional
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from copy import deepcopy
import random
import json, torch
from typing import Union, List
import math
import torch.nn.functional as F
from libs.utils.general_utils import PILtoTorch
from libs.utils.camera_utils import Camera,fov2focal,focal2fov

class CameraDataset(torch.utils.data.Dataset):
    def __init__(self, cameras: List[Camera]):
        self.cameras = cameras

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        sample = deepcopy(self.cameras[idx])

        if sample.image is None:
            image = Image.open(sample.image_path)
        else:
            image = sample.image
                
        im_data = np.array(image.convert("RGBA"))
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + sample.bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        resized_image_rgb = PILtoTorch(image, (sample.image_width, sample.image_height))
        image = resized_image_rgb[:3, ...]
        sample.original_image = image.clamp(0.0, 1.0)

        mask = Image.open(sample.mask_path)
        mask_data= PILtoTorch(mask, (sample.image_width, sample.image_height))
        sample.original_mask = mask_data.clamp(0.0,1.0)
        
        if sample.normal_path is not None:
            normal = Image.open(sample.normal_path)
            normal_data= PILtoTorch(normal, (sample.image_width, sample.image_height))
            sample.original_normal = normal_data.clamp(0.0,1.0)
        else:
            sample.original_normal = None

        if sample.albedo_path is not None:
            albedo = Image.open(sample.albedo_path)
            albedo_data= PILtoTorch(albedo, (sample.image_width, sample.image_height))[:3,...]
            sample.original_albedo = albedo_data.clamp(0.0,1.0)
        else:
            sample.original_albedo = None

        if sample.roughness_path is not None:
            roughness = Image.open(sample.roughness_path)
            roughness_data= PILtoTorch(roughness, (sample.image_width, sample.image_height))
            sample.original_roughness = roughness_data.clamp(0.0,1.0)
        else:
            sample.original_roughness = None

        if sample.metallic_path is not None:
            metallic = Image.open(sample.metallic_path)
            metallic_data= PILtoTorch(metallic, (sample.image_width, sample.image_height))
            sample.original_metallic = metallic_data.clamp(0.0,1.0)
        else:
            sample.original_metallic = None

        flame_param = np.load(sample.flame_path, allow_pickle=True)
        
        sample.flame_param = {
            'shape': torch.from_numpy(flame_param["shape"]).float(),
            'expr': torch.from_numpy(flame_param["expr"]).float(),
            'rotation': torch.from_numpy(flame_param["rotation"]).float(),
            'neck_pose': torch.from_numpy(flame_param["neck_pose"]).float(),
            'jaw_pose': torch.from_numpy(flame_param["jaw_pose"]).float(),
            'eyes_pose': torch.from_numpy(flame_param["eyes_pose"]).float(),
            'translation': torch.from_numpy(flame_param["translation"]).float(),
            'static_offset': torch.from_numpy(flame_param["static_offset"]).float(),
        }
        
        return sample
    

        
class MDIDataMultiview:
    # only path is loaded 
    def __init__(self, args):
        self.base_path = args.base_path
        self.white_background = args.white_background
        self.lgt_type= args.lgt_type
        self.load_DB()
    
    def load_DB(self):
        if self.lgt_type=='cube':
            self.train_cameras = self.make_cameras_list(mode="train")
            self.test_cameras = self.make_cameras_list(mode="test")
            self.light_list = list(set(self.make_light_list(mode="train") + self.make_light_list(mode="train")))
        elif self.lgt_type=='hdr': # 360  light map
            self.train_cameras = self.make_cameras_list(mode="train_hdr")
            self.test_cameras = self.make_cameras_list(mode="test")
            self.light_list = list(set(self.make_light_list(mode="train_hdr") + self.make_light_list(mode="train_hdr")))
        else:
            self.train_cameras = self.make_cameras_list(mode="train")
            self.test_cameras = self.make_cameras_list(mode="test")
            self.light_list = list(set(self.make_light_list(mode="train") + self.make_light_list(mode="train")))
    
    def get_id_params(self, mean: bool = True):
        """
        Return subject-level identity parameters (shape, static_offset) aggregated
        across all sequences for this subject ID, or from the first available sequence.
        """
        base_name = os.path.basename(self.base_path)
        if base_name.startswith("UNION_"):
            subj_id = base_name.split("_")[-1]
        else:
            digits = [ch for ch in base_name if ch.isdigit()]
            subj_id = "".join(digits) if digits else base_name

        root_dir = os.path.dirname(self.base_path)
        candidates = []
        try:
            for d in sorted(os.listdir(root_dir)):
                if d.startswith(f"{subj_id}_"):
                    seq_dir = os.path.join(root_dir, d)
                    if os.path.isdir(seq_dir):
                        flame_file = os.path.join(seq_dir, "flame_param", "00000.npz")
                        if os.path.exists(flame_file):
                            candidates.append(flame_file)
        except FileNotFoundError:
            candidates = []

        if not candidates:
            fallback_file = os.path.join(self.base_path, "flame_param", "00000.npz")
            fallback_file_full = os.path.join(self.base_path, "flame_param_full", "00000.npz")
            if os.path.exists(fallback_file):
                candidates = [fallback_file]
            elif os.path.exists(fallback_file_full):
                candidates = [fallback_file_full]
            else:
                # Instead of raising an error, return zeros
                id_name = subj_id
                # shape 크기는 실제 데이터 shape 크기에 맞게 수정 필요
                shape = np.zeros((300,), dtype=np.float32)
                static_offset = None
                return shape, static_offset, id_name

        if not mean:
            flame_params = np.load(candidates[0], allow_pickle=True)
            shape = flame_params["shape"]
            static_offset = flame_params["static_offset"]
        else:
            shapes, offsets = [], []
            for fp in candidates:
                p = np.load(fp, allow_pickle=True)
                shapes.append(p["shape"])
                offsets.append(p["static_offset"])
            shape = np.mean(np.stack(shapes, axis=0), axis=0)
            static_offset = np.mean(np.stack(offsets, axis=0), axis=0)

        id_name = subj_id
        return shape, static_offset, id_name


    def make_cameras_list(self, mode):
        cam_infos=[]
        with open(os.path.join(self.base_path, f"transforms_{mode}.json")) as json_file:
            contents = json.load(json_file)
            frames = contents["frames"]
            uniq_light = set()
            light_maps = []
            for idx, frame in tqdm(enumerate(frames), total=len(frames)):
                export_path = Path(self.base_path).parents[0]
                base_name = Path(self.base_path).stem
                
                if base_name.startswith("UNION_"):
                    subj_id = base_name.split("_")[-1]
                    seq_name = Path(frame["file_path"]).parts[1]
                    file_path = frame["file_path"].lstrip('./')
                    mask_file_path = frame["fg_mask_path"].lstrip('./')

                    c2w = np.array(frame["transform_matrix"])
                    
                    c2w[:3, 1:3] *= -1 # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                    
                    cx = np.array(frame["cx"])
                    cy = np.array(frame["cy"])

                    w2c = np.linalg.inv(c2w)
                    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                    T = w2c[:3, 3]

                    bg = np.array([1,1,1]) if self.white_background else np.array([0, 0, 0])

                    image_path = os.path.join(export_path, file_path)
                    image_name = Path(image_path).stem
                    frame_i = int(image_name.split("_")[0])
                    cam_i = int(image_name.split("_")[1])
                    mask_path = os.path.join(export_path, mask_file_path)
                    flame_path = os.path.join(export_path, seq_name, "flame_param", f"{frame_i:05d}.npz")
                    # normal_path = os.path.join(self.base_path, "normal", "sapiens_2b", f"{image_name}.jpg")
                    normal_path = os.path.join(export_path, seq_name, "photo_normal", f"{image_name}.jpg")
                    albedo_path = os.path.join(export_path, seq_name, "albedo", f"albedo_{image_name}.png")
                    roughness_path = os.path.join(self.base_path, "roughness", f"roughness_{image_name}.png")
                    metallic_path = os.path.join(self.base_path, "metallic", f"metallic_{image_name}.png")
                    lgt_path, light_maps = None, None

                    if os.path.exists(normal_path) is not True:
                        normal_path = None
                    if os.path.exists(albedo_path) is not True:
                        albedo_path = None
                    if os.path.exists(roughness_path) is not True:
                        roughness_path = None
                    if os.path.exists(metallic_path) is not True:
                        metallic_path = None
                else:
                    file_path = frame["file_path"]
                    mask_file_path = frame["fg_mask_path"]
                    flame_path = frame['flame_param_path']
                    c2w = np.array(frame["transform_matrix"])
                    
                    c2w[:3, 1:3] *= -1 # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                    
                    cx = np.array(frame["cx"])
                    cy = np.array(frame["cy"])

                    w2c = np.linalg.inv(c2w)
                    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
                    T = w2c[:3, 3]

                    bg = np.array([1,1,1]) if self.white_background else np.array([0, 0, 0])

                    export_path = Path(self.base_path).parents[0]
                    image_path = os.path.join(self.base_path, file_path)
                    image_name = Path(image_path).stem
                    frame_i = int(image_name.split("_")[0])
                    cam_i = int(image_name.split("_")[1])
                    mask_path = os.path.join(self.base_path, mask_file_path)
                    flame_path = os.path.join(self.base_path, flame_path)
                    # normal_path = os.path.join(self.base_path, "normal", "sapiens_2b", f"{image_name}.jpg")
                    normal_path = os.path.join(self.base_path, "photo_normal", f"{image_name}.jpg")
                    albedo_path = os.path.join(self.base_path, "albedo", f"albedo_{image_name}.png")
                    roughness_path = os.path.join(self.base_path, "roughness", f"roughness_{image_name}.png")
                    metallic_path = os.path.join(self.base_path, "metallic", f"metallic_{image_name}.png")
                    lgt_path = frame["lgt_path"] if "lgt_path" in frame else None 

                    
                    if lgt_path not in uniq_light:
                        light_maps.append(lgt_path)

                    if os.path.exists(normal_path) is not True:
                        normal_path = None
                    if os.path.exists(albedo_path) is not True:
                        albedo_path = None
                    if os.path.exists(roughness_path) is not True:
                        roughness_path = None
                    if os.path.exists(metallic_path) is not True:
                        metallic_path = None
            
                image = None
                width = frame['w']
                height = frame['h']

                fovx = frame["camera_angle_x"]
                fovy = focal2fov(fov2focal(fovx, width), height)
                
                cam_infos.append(Camera(
                    frame_i=frame_i, cam_i=cam_i, R=R, T=T, FoVx=fovx, FoVy=fovy, cx=cx, cy=cy, bg=bg,
                    image=image, image_width=width, image_height=height, 
                    base_path=self.base_path, image_path=image_path, mask_path=mask_path, normal_path = normal_path, 
                    albedo_path=albedo_path, roughness_path=roughness_path, metallic_path=metallic_path, lgt_path=lgt_path, flame_path=flame_path,
                    image_name=image_name))
        
        return cam_infos
    

    def make_light_list(self, mode):
        cam_infos=[]
        with open(os.path.join(self.base_path, f"transforms_{mode}.json")) as json_file:
            contents = json.load(json_file)
            frames = contents["frames"]
            uniq_light = set()
            light_maps = []
            for idx, frame in tqdm(enumerate(frames), total=len(frames)):
                export_path = Path(self.base_path).parents[0]
                base_name = Path(self.base_path).stem
                
                if base_name.startswith("UNION_"):
                    lgt_path, light_maps = None, None

                else:
                    lgt_path = frame["lgt_path"] if "lgt_path" in frame else None 

                    if lgt_path not in uniq_light:
                        uniq_light.add(lgt_path)
                        light_maps.append(lgt_path)
        
        return light_maps
    
    def getTrainCameras(self):
        return CameraDataset(self.train_cameras)
    
    def getValCameras(self):
        return CameraDataset(self.val_cameras)

    def getTestCameras(self):
        return CameraDataset(self.test_cameras)
    
    def getLightList(self):
        return self.light_list
    

    
    def get_canonical_rays(self, scale: float = 1.0) -> torch.Tensor:
        # NOTE: some datasets do not share the same intrinsic (e.g. DTU)
        # get reference camera
        ref_camera: Camera = self.train_cameras[0]
        # TODO: inject intrinsic
        H, W = ref_camera.image_height, ref_camera.image_width
        cen_x = W / 2
        cen_y = H / 2
        tan_fovx = math.tan(ref_camera.FoVx * 0.5)
        tan_fovy = math.tan(ref_camera.FoVy * 0.5)
        focal_x = W / (2.0 * tan_fovx)
        focal_y = H / (2.0 * tan_fovy)

        x, y = torch.meshgrid(
            torch.arange(W),
            torch.arange(H),
            indexing="xy",
        )
        x = x.flatten()  # [H * W]
        y = y.flatten()  # [H * W]
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - cen_x + 0.5) / focal_x,
                    (y - cen_y + 0.5) / focal_y,
                ],
                dim=-1,
            ),
            (0, 1),
            value=1.0,
        )  # [H * W, 3]
        # NOTE: it is not normalized
        return camera_dirs.cuda()