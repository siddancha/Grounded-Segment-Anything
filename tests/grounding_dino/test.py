# taken from https://github.com/IDEA-Research/GroundingDINO#README.md

from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

grounding_dino_home = 'GroundingDINO/groundingdino'

config_file_path = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
model = load_model(config_file_path, "weights/groundingdino_swint_ogc.pth")
IMAGE_PATH = "tests/grounding_dino/data/cat_dog.jpeg"
TEXT_PROMPT = "chair . person . cat . dog ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imshow('', annotated_frame)
cv2.waitKey(3000)
