from PIL import Image
import base64
import boto3
import io
import json

bedrock_client = boto3.client('bedrock-runtime')
model_id = 'stability.stable-diffusion-xl'

# 
def bytes_to_image(image_bytes: bytes) -> Image.Image:
  return Image.open(io.BytesIO(image_bytes))


def image_to_png_bytes(image: Image.Image) -> bytes:
  buffer = io.BytesIO()
  image.save(buffer, format='png')
  return buffer.getvalue()


def invoke_bytes(image: bytes, prompt: str) -> bytes:
  init_image = bytes_to_image(image)
  response_image = invoke(image=init_image, prompt=prompt)
  return image_to_png_bytes(response_image)


def invoke(image:Image.Image, prompt: str) -> Image.Image:

  size = image.size

  init_image = image_to_png_bytes(image=image.resize((512, 512)))
  init_image = base64.b64encode(init_image).decode('utf-8')

  body = json.dumps({
      'text_prompts': [{'text': prompt}],
      'init_image': init_image,
      })

  response = bedrock_client.invoke_model(
    body=body, 
    modelId=model_id
    )
  
  response_body = json.loads(response.get('body').read())
  base64_str = response_body['artifacts'][0].get('base64')

  image_byte = base64.decodebytes(bytes(base64_str, 'utf-8'))
  response_image = bytes_to_image(image_byte)
  response_image = response_image.resize(size)

  return response_image
