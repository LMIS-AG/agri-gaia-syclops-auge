import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionInpaintPipelineLegacy
from torchvision.transforms import PILToTensor
import cv2
import PIL
import numpy as np

img = np.load('img.npy')
components_map = np.load('components.npy')
bg_class = 1
k_size = 7
downscale_factor = 1.5

pil_to_tensor = PILToTensor()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
shape = [int(img_rgb.shape[1] // downscale_factor), int(img_rgb.shape[0] // downscale_factor)]
#height and width have to be divisible by 8
shape[0] -= shape[0] % 8
shape[1] -= shape[1] % 8


# init_image_full = PIL.Image.fromarray(img_rgb)
init_image = PIL.Image.fromarray(img_rgb).resize(shape)
# init_image.thumbnail((768, 768)) # was in huggingface example

fg_mask = np.array(components_map != bg_class, dtype=np.uint8) * 255
# fg_mask = components_map != bg_class
kernel = np.ones((k_size, k_size), np.uint8)
fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
# fg_mask_full = PIL.Image.fromarray(fg_mask)
fg_mask = PIL.Image.fromarray(fg_mask).resize(shape)

# load the pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    # "runwayml/stable-diffusion-v1-5",
    # "CompVis/stable-diffusion-v1-4",
    #"stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()  # to save some gpu memory in exchange for a small speed decrease

# tensor_image = to_tensor(init_image).float()
# latents_shape = (1, pipe.unet.in_channels, 512, 512)  # example shape, adjust to match your needs
# tensor_image = tensor_image.view(latents_shape)
# tensor_image = torch.reshape(tensor_image, latents_shape)

# convert the image to a tensor and add a batch dimension
# image_tensor = pil_to_tensor(init_image).to(torch.float16).unsqueeze(0)
# image_tensor = image_tensor / np.max(init_image) * 2. - 1.
# # encode the image into its latent space using the VAE
# with torch.no_grad():
#     latents = pipe.vae.encode(image_tensor.cuda())
# tensor_image = latents.latent_dist.sample() * pipe.scheduler.init_noise_sigma
# tensor_image = 1 / 0.18215 * tensor_image
# tensor_image = 0.18215 * tensor_image

image = pipe(prompt="",
             negative_prompt="",
             # strength=1.,
             image=init_image,
             mask_image=fg_mask,
            guidance_scale = 0.,
             height=shape[1],
             width=shape[0]
             # add_predicted_noise=True,
             #latents=tensor_image
             ).images[0]

# image.save("test_img2img.png")
image = image.resize((img_rgb.shape[1], img_rgb.shape[0]))
image_response_rgb = np.array(image)
image_response = cv2.cvtColor(image_response_rgb, cv2.COLOR_RGB2BGRA)

# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
# image = PIL.Image.fromarray(img_rgb)
# image_enc = str(encode_pil_to_base64(image), 'utf8')
# fg_mask = np.array(components_map != bg_class, dtype=np.uint8) * 255
# kernel = np.ones((k_size, k_size), np.uint8)
# fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
# fg_mask = PIL.Image.fromarray(fg_mask)
# fg_mask_enc = str(encode_pil_to_base64(fg_mask), 'utf8')
# payload = {'init_images': [image_enc],
#            "mask": fg_mask_enc,
#            "mask_blur": 0,
#            'inpainting_fill': 1,
#            "inpaint_full_res": True,
#            "inpaint_full_res_padding": 8,
#            "inpainting_mask_invert": 0,
#            'denoising_strength': 1,
#            "steps": 20,
#            "cfg_scale": 0,}
#
# response = requests.post(url='http://127.0.0.1:7860/sdapi/v1/img2img', json=payload)
# image_response = decode_base64_to_image(response.json()['images'][0])
# image_response_rgb = np.array(image_response)
# image_response = cv2.cvtColor(image_response_rgb, cv2.COLOR_RGB2BGRA)



cv2.imshow('img', img)
# cv2.imshow('fg_mask', fg_mask)
cv2.imshow('img_response', image_response)
cv2.waitKey()

