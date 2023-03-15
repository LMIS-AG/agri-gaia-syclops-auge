
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

img_in = 'test.png'

# load the pipeline
device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained('model', torch_dtype=torch.float16).to(
    device
)

init_image = Image.open(img_in)
#init_image.thumbnail((768, 768))

prompt = "rust disease"

images = pipe(prompt=prompt, image=init_image, strength=0.3, guidance_scale=7.5).images

images[0].save("test_img2img.png")

# # create a diffusion pipeline for image-to-image translation
# diffusion_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained('model',
#                                                                     image_size=(512, 512),
#                                                                     num_timesteps=1000,
#                                                                     noise_schedule='linear',
#                                                                     num_samples=1,
#                                                                     device='cuda'
# )
#
# input_image = Image.open('input.png').convert('RGB')
# preprocess = Compose([
#     Resize((512, 512)),
#     ToTensor(),
#     Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
# input_tensor = preprocess(input_image).unsqueeze(0)
#
# output_tensor = diffusion_pipeline(input_tensor)
#
# pipe = StableDiffusionPipeline.from_pretrained("model", torch_dtype=torch.float16)
# pipe = pipe.to("cuda")
#
# prompt = "a full maize plant with rust disease"
# image = pipe(prompt).images[0]
# image.save('test.png')