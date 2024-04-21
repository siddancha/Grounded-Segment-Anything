echo "Running Grounding-DINO demo on horses ..."
python grounding_dino_demo.py

echo "Running Grounding-SAM demo on beavers with segmentation ..."
python grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint ./weights/groundingdino_swint_ogc.pth \
  --sam_checkpoint ./weights/sam_vit_h_4b8939.pth \
  --input_image assets/demo1.jpg \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "bear" \
  --device "cuda"
