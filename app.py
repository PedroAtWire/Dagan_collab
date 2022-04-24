import os
import shutil
import gradio as gr
from PIL import Image

#os.chdir('Restormer')

# Download sample images


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


def inference(img, video):
    if not os.path.exists('temp'):
      os.system('mkdir temp')
      
    ####  Resize the longer edge of the input image
      max_res = 256
      width, height = img.size
      if max(width,height) > max_res:
        scale = max_res /max(width,height)
        width = int(scale*width)
        height = int(scale*height)
        img = img.resize((width,height), Image.ANTIALIAS)
      
    img.save("temp/image.jpg", "JPEG")
    video.save('temp/video.mp4')
    os.system("python demo_dagan.py --source_image 'temp/image.jpg' --driving_video {} --output 'temp/rst.mp4'".format(video))

    return f'temp/rst.mp4'
    
gr.Interface(
    inference,
    [
        gr.inputs.Image(type="pil", label="Source Image"),
        gr.inputs.Video(type='mp4',label="Driving Video"),
    ],
    gr.outputs.Video(type="mp4", label="Output"),
    title=title,
    description=description,
    article=article,
    theme ="huggingface",
    examples=examples,
    allow_flagging=False,
    ).launch(debug=False,enable_queue=True)
