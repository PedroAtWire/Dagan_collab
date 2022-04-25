import os
import shutil
import gradio as gr
from PIL import Image
import subprocess
#os.chdir('Restormer')
from demo_dagan import *
# Download sample images
import torch
import torch.nn.functional as F
import os
from skimage import img_as_ubyte
import imageio
from skimage.transform import resize
import numpy as np
import modules.generator as G
import modules.keypoint_detector as KPD
import yaml
from collections import OrderedDict
import depth

examples = [['project/cartoon2.jpg','project/video1.mp4'],
						['project/cartoon3.jpg','project/video2.mp4'],
						['project/celeb1.jpg','project/video1.mp4'],
						['project/celeb2.jpg','project/video2.mp4'],
						]


inference_on = ['Full Resolution Image', 'Downsampled Image']

title = "DaGAN"
description = """
Gradio demo for <b>Depth-Aware Generative Adversarial Network for Talking Head Video Generation</b>, CVPR 2022L. <a href='https://arxiv.org/abs/2203.06605'>[Paper]</a><a href='https://github.com/harlanhong/CVPR2022-DaGAN'>[Github Code]</a>\n 
"""
##With Restormer, you can perform: (1) Image Denoising, (2) Defocus Deblurring, (3)  Motion Deblurring, and (4) Image Deraining. 
##To use it, simply upload your own image, or click one of the examples provided below.

article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2203.06605'>Depth-Aware Generative Adversarial Network for Talking Head Video Generation</a> | <a href='https://github.com/harlanhong/CVPR2022-DaGAN'>Github Repo</a></p>"


def inference(source_image, video):
    if not os.path.exists('temp'):
        os.system('mkdir temp')
    cmd = f"ffmpeg -y -ss 00:00:00 -i {video} -to 00:00:08 -c copy video_input.mp4"
    subprocess.run(cmd.split())
    driving_video = "video_input.mp4"
    output = "rst.mp4"
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
        source_image = imageio.imread(source_image)
        reader = imageio.get_reader(driving_video)
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

        imageio.mimsave(output, [np.concatenate((img_as_ubyte(s),img_as_ubyte(d),img_as_ubyte(p)),1) for (s,d,p) in zip(sources, drivings, predictions)], fps=fps)
        imageio.mimsave("gray.mp4", depth_gray, fps=fps)
        # merge the gray video
        animation = np.array(imageio.mimread(output,memtest=False))
        gray = np.array(imageio.mimread("gray.mp4",memtest=False))

        src_dst = animation[:,:,:512,:]
        animate = animation[:,:,512:,:]
        merge = np.concatenate((src_dst,gray,animate),2)
        imageio.mimsave(output, merge, fps=fps)

    return output
		
gr.Interface(
		inference,
		[
				gr.inputs.Image(type="filepath", label="Source Image"),
				gr.inputs.Video(type='mp4',label="Driving Video"),
		],
		gr.outputs.Video(type="mp4", label="Output Video"),
		title=title,
		description=description,
		article=article,
		theme ="huggingface",
		examples=examples,
		allow_flagging=False,
		).launch(debug=False,enable_queue=True)
