## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn.functional as F
import os
from skimage import img_as_ubyte
import cv2
import argparse
import imageio
from skimage.transform import resize
from scipy.spatial import ConvexHull
from tqdm import tqdm
import numpy as np
import modules.generator as G
import modules.keypoint_detector as KPD
import yaml
from collections import OrderedDict
import depth
parser = argparse.ArgumentParser(description='Test DaGAN on your own images')
parser.add_argument('--source_image', default='./temp/source.jpg', type=str, help='Directory of input source image')
parser.add_argument('--driving_video', default='./temp/driving.mp4', type=str, help='Directory for driving video')
parser.add_argument('--output', default='./temp/result.mp4', type=str, help='Directory for driving video')


args = parser.parse_args()
def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])
    return kp_new
def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    sources = []
    drivings = []
    with torch.no_grad():
        predictions = []
        depth_gray = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        if not cpu:
            source = source.cuda()
            driving = driving.cuda()
        outputs = depth_decoder(depth_encoder(source))
        depth_source = outputs[("disp", 0)]

        outputs = depth_decoder(depth_encoder(driving[:, :, 0]))
        depth_driving = outputs[("disp", 0)]
        source_kp = torch.cat((source,depth_source),1)
        driving_kp = torch.cat((driving[:, :, 0],depth_driving),1)
       
        kp_source = kp_detector(source_kp)
        kp_driving_initial = kp_detector(driving_kp) 

        # kp_source = kp_detector(source)
        # kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]

            if not cpu:
                driving_frame = driving_frame.cuda()
            outputs = depth_decoder(depth_encoder(driving_frame))
            depth_map = outputs[("disp", 0)]

            gray_driving = np.transpose(depth_map.data.cpu().numpy(), [0, 2, 3, 1])[0]
            gray_driving = 1-gray_driving/np.max(gray_driving)

            frame = torch.cat((driving_frame,depth_map),1)
            kp_driving = kp_detector(frame)

            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm,source_depth = depth_source, driving_depth = depth_map)

            drivings.append(np.transpose(driving_frame.data.cpu().numpy(), [0, 2, 3, 1])[0])
            sources.append(np.transpose(source.data.cpu().numpy(), [0, 2, 3, 1])[0])
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
            depth_gray.append(gray_driving)
    return sources, drivings, predictions,depth_gray
with open("config/vox-adv-256.yaml") as f:
    config = yaml.load(f)
generator = G.SPADEDepthAwareGenerator(**config['model_params']['generator_params'],**config['model_params']['common_params'])
config['model_params']['common_params']['num_channels'] = 4
kp_detector = KPD.KPDetector(**config['model_params']['kp_detector_params'],**config['model_params']['common_params'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


g_checkpoint = torch.load("generator.pt", map_location=device)
kp_checkpoint = torch.load("kp_detector.pt", map_location=device)

ckp_generator = OrderedDict((k.replace('module.',''),v) for k,v in g_checkpoint.items())
generator.load_state_dict(ckp_generator)
ckp_kp_detector = OrderedDict((k.replace('module.',''),v) for k,v in kp_checkpoint.items())
kp_detector.load_state_dict(ckp_kp_detector)

depth_encoder = depth.ResnetEncoder(18, False)
depth_decoder = depth.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))
loaded_dict_enc = torch.load('encoder.pth')
loaded_dict_dec = torch.load('depth.pth')
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
depth_encoder.load_state_dict(filtered_dict_enc)
ckp_depth_decoder= {k: v for k, v in loaded_dict_dec.items() if k in depth_decoder.state_dict()}
depth_decoder.load_state_dict(ckp_depth_decoder)
depth_encoder.eval()
depth_decoder.eval()
    
# device = torch.device('cpu')
# stx()

generator = generator.to(device)
kp_detector = kp_detector.to(device)
depth_encoder = depth_encoder.to(device)
depth_decoder = depth_decoder.to(device)

generator.eval()
kp_detector.eval()
depth_encoder.eval()
depth_decoder.eval()

img_multiple_of = 8

with torch.inference_mode():
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
    source_image = imageio.imread(args.source_image)
    reader = imageio.get_reader(args.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]



    i = find_best_frame(source_image, driving_video)
    print ("Best frame: " + str(i))
    driving_forward = driving_video[i:]
    driving_backward = driving_video[:(i+1)][::-1]
    sources_forward, drivings_forward, predictions_forward,depth_forward = make_animation(source_image, driving_forward, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False)
    sources_backward, drivings_backward, predictions_backward,depth_backward = make_animation(source_image, driving_backward, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False)
    predictions = predictions_backward[::-1] + predictions_forward[1:]
    sources = sources_backward[::-1] + sources_forward[1:]
    drivings = drivings_backward[::-1] + drivings_forward[1:]
    depth_gray = depth_backward[::-1] + depth_forward[1:]

    imageio.mimsave(args.output, [np.concatenate((img_as_ubyte(p)),1) for (s,d,p) in zip(sources, drivings, predictions)], fps=fps)
    imageio.mimsave("gray.mp4", depth_gray, fps=fps)
    # merge the gray video
    animation = np.array(imageio.mimread(args.output,memtest=False))
    gray = np.array(imageio.mimread("gray.mp4",memtest=False))

    src_dst = animation[:,:,:512,:]
    animate = animation[:,:,512:,:]
    merge = np.concatenate((src_dst,gray,animate),2)
    imageio.mimsave(args.output, merge, fps=fps)

    # print(f"\nRestored images are saved at {out_dir}")