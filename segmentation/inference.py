from argparse import ArgumentParser
import mmcv
import mmcv_custom
import mmseg_custom
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
#from mmseg.core.evaluation import get_palette
from mmseg_custom.core.evaluation import get_classes, get_palette
from mmcv.runner import load_checkpoint
#from mmseg.core import get_classes
import glob
import numpy as np
import cv2
import os
import os.path as osp
import tempfile


def split_image(img_path):
    """Split the image into 4 parts."""
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    h_half, w_half = h // 2, w // 2

    return [img[:h_half, :w_half], img[:h_half, w_half:], 
            img[h_half:, :w_half], img[h_half:, w_half:]]


def patch_images(image_parts, output_path):
    """Patch the 4 parts of the image together and save."""
    top = np.concatenate(image_parts[:2], axis=1)
    bottom = np.concatenate(image_parts[2:], axis=1)
    full_img = np.concatenate([top, bottom], axis=0)
    cv2.imwrite(output_path, full_img)


def process_image_part(model, img_part, args):
    """Process a single part of the image and return the results."""
    result = inference_segmentor(model, img_part)
    seg_map = np.where(result[0] == 1, 255, 0).astype(np.uint8)

    img_overlay = model.show_result(img_part, result, palette=get_palette(args.palette), show=False, opacity=args.opacity)

    return seg_map, img_overlay


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('img_dir', help='Directory containing images')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--palette', default='cityscapes', help='Color palette used for segmentation map')
    parser.add_argument('--opacity', type=float, default=0.3, help='Opacity of painted segmentation map')
    args = parser.parse_args()

    # Build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', get_classes(args.palette))

    # Image extensions
    image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif']
    patterns = [osp.join(args.img_dir, f'*.{ext}') for ext in image_extensions]
    img_files = [file for pattern in patterns for file in glob.glob(pattern)]

    for img_file in img_files:
        # Split image into 4 parts
        img_parts = split_image(img_file)

        # Process each part
        class_segments = []
        overlay_segments = []
        for img_part in img_parts:
            class_seg, overlay_seg = process_image_part(model, img_part, args)
            class_segments.append(class_seg)
            overlay_segments.append(overlay_seg)

        # Patching together the outputs
        base, extension = osp.splitext(osp.basename(img_file))
        out_path_class = osp.join(args.out, f"{base}_class{extension}")
        out_path_overlay = osp.join(args.out, f"{base}_overlay{extension}")

        patch_images(class_segments, out_path_class)
        patch_images(overlay_segments, out_path_overlay)

        print(f"Result for {img_file} is saved at {out_path_class} and {out_path_overlay}")

if __name__ == '__main__':
    main()

