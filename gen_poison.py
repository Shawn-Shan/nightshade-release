import os

import sys
from PIL import Image
import glob
import argparse
import pickle
from torchvision import transforms
from opt import PoisonGeneration


def crop_to_square(img):
    size = 512
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
        ]
    )
    return image_transforms(img)


def main():
    poison_generator = PoisonGeneration(target_concept=args.target_name, device="cuda", eps=args.eps)
    all_data_paths = glob.glob(os.path.join(args.directory, "*.p"))
    all_imgs = [pickle.load(open(f, "rb"))['img'] for f in all_data_paths]
    all_texts = [pickle.load(open(f, "rb"))['text'] for f in all_data_paths]
    all_imgs = [Image.fromarray(img) for img in all_imgs]

    all_result_imgs = poison_generator.generate_all(all_imgs, args.target_name)
    os.makedirs(args.outdir, exist_ok=True)

    for idx, cur_img in enumerate(all_result_imgs):
        cur_data = {"text": all_texts[idx], "img": cur_img}
        pickle.dump(cur_data, open(os.path.join(args.outdir, "{}.p".format(idx)), "wb"))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str,
                        help="", default='')
    parser.add_argument('-od', '--outdir', type=str,
                        help="", default='')
    parser.add_argument('-e', '--eps', type=float, default=0.04)
    parser.add_argument('-t', '--target_name', type=str, default="cat")
    return parser.parse_args(argv)


if __name__ == '__main__':
    import time

    args = parse_arguments(sys.argv[1:])
    main()
