from PIL import Image, PngImagePlugin
import cv2
import numpy as np
import requests
import io
import base64

def encode_pil_to_base64(image, format='png'):
    with io.BytesIO() as output_bytes:

        if format == 'png':
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None))

        elif format in ("jpg", "jpeg", "webp"):
            parameters = image.info.get('parameters', None)
            exif_bytes = piexif.dump({
                "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") }
            })
            if format in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG", exif = exif_bytes, quality=opts.jpeg_quality)
            else:
                image.save(output_bytes, format="WEBP", exif = exif_bytes, quality=opts.jpeg_quality)

        else:
            raise HTTPException(status_code=500, detail="Invalid image format")

        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)

def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(io.BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as err:
        raise HTTPException(status_code=500, detail="Invalid encoded image")

def contour_prune(mask):

    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find contour with largest area
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # Create new binary mask with only the largest contour
    result = np.zeros_like(mask)
    cv2.drawContours(result, [max_contour], 0, 255, -1)
    return result


def test(thresh_v=10):
    img = cv2.imread(r'C:\Users\HEW\Projekte\syclops-dev\assets\syclops-assets-examples\models\plants\textures\young_maize.png')
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fg_mask = np.array((img_hsv[:,:,2] < 255 - thresh_v) * 255, dtype=np.uint8)
    fg_mask = contour_prune(fg_mask)
    fg_mask_erode = cv2.erode(fg_mask, np.ones((5, 5), np.uint8))
    fg_mask = np.mean((fg_mask, fg_mask_erode), axis=0)
    fg_mask = np.array(fg_mask, dtype=np.uint8)
    img_rgb[fg_mask==0] *= 0

    fg_pil = Image.fromarray(fg_mask)
    img_pil = Image.fromarray(img_rgb)
    image_enc = str(encode_pil_to_base64(img_pil), 'utf8')
    fg_mask_enc = str(encode_pil_to_base64(fg_pil), 'utf8')
    payload = {'init_images': [image_enc],
                "steps": 150,
               'prompt': '<lora:rust_disease_new:1>',
               'denoising_strength': 1,
               'cfg_scale': 7,
               'batch_size': 1,
                "mask": fg_mask_enc,
                "mask_blur": 0,
                'inpainting_fill': 1,
                "inpaint_full_res": True,
                "inpaint_full_res_padding": 0,
                "inpainting_mask_invert": 0,
               }
    response = requests.post(url='http://127.0.0.1:7860/sdapi/v1/img2img', json=payload)
    image_response = decode_base64_to_image(response.json()['images'][0])
    image_response_rgb = np.array(image_response)
    image_response_hsv = cv2.cvtColor(image_response_rgb, cv2.COLOR_RGB2HSV)
    image_response = cv2.cvtColor(image_response_rgb, cv2.COLOR_RGB2BGRA)
    image_response[:,:,3][image_response_hsv[:,:,2] < thresh_v] = 0
    cv2.imwrite(r'C:\Users\HEW\Projekte\syclops-dev\assets\syclops-assets-examples\models\plants\textures\augmented.png',
                image_response)

if __name__ == '__main__':
    test()


