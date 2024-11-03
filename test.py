
import torch 
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from diffusers.utils import export_to_video, load_image 
from cogvideox_interpolation.pipeline import CogVideoXInterpolationPipeline 

torch.set_default_device("mps")


model_path = './CogvideoX-Interpolation-model' 
pipe = CogVideoXInterpolationPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
).to("mps")

#pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

prompt = 'a happy guy'

first_image = load_image('./cases/6.jpg')
last_image = load_image('./cases/66.jpg')
video = pipe(
	prompt=prompt,
	first_image=first_image,
	last_image=last_image,
	num_videos_per_prompt=50,
	num_inference_steps=50,
	num_frames=49,
	guidance_scale=6,
	generator=torch.Generator(device="mps").manual_seed(42),
)[0]
export_to_video(video_save_path, fps=8)