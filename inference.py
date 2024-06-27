import os
import argparse
from PIL import Image
import glob
import torch
from tqdm import tqdm
from torchvision import transforms
from diffusion_ffpe.model import Diffusion_FFPE, initialize_text_encoder
from diffusion_ffpe.my_utils import build_transform


def parse_args_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default="TEST_PATH")
    parser.add_argument('--model_path', type=str, default="stabilityai/sd-turbo")
    parser.add_argument('--pretrained_path', type=str, default="./checkpoints/model.pkl")
    parser.add_argument('--output_path', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--prompt', type=str, default="paraffin section")
    parser.add_argument('--image_prep', type=str, default='no_resize', help='the image preparation method')
    parser.add_argument('--direction', type=str, default='a2b')
    args = parser.parse_args()
    return args


def main(args):
    # make output folder
    os.makedirs(args.output_path, exist_ok=True)
    print(args.output_path)

    # initialize the model
    model = Diffusion_FFPE(pretrained_path=args.pretrained_path, model_path=args.model_path,
                           enable_xformers_memory_efficient_attention=True)
    model.cuda()
    model.eval()

    tokenizer, text_encoder = initialize_text_encoder(args.model_path)
    a2b_tokens = tokenizer(args.prompt, max_length=tokenizer.model_max_length, padding="max_length",
                           truncation=True, return_tensors="pt").input_ids[0]
    text_emb = text_encoder(a2b_tokens.cuda().unsqueeze(0))[0].detach().cuda()

    T_val = build_transform(args.image_prep)
    paths = glob.glob(os.path.join(args.img_path, '*'))
    for path in tqdm(paths):
        input_image = Image.open(path).convert('RGB')
        # translate the image
        with torch.no_grad():
            input_img = T_val(input_image)
            x_t = transforms.ToTensor()(input_img)
            x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()

            output = model(x_t, direction=args.direction, text_emb=text_emb)

        output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_image.width, input_image.height))

        # save the output image
        img_name = os.path.basename(path)

        output_pil.save(os.path.join(args.output_path, img_name))


if __name__ == "__main__":
    inference_args = parse_args_inference()
    main(inference_args)

