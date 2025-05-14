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
    parser.add_argument('--img_path', type=str, default="TEST_FF_PATH")
    parser.add_argument('--model_path', type=str, default="stabilityai/sd-turbo")
    parser.add_argument('--pretrained_path', type=str, default="./checkpoints/model.pkl")
    parser.add_argument('--output_path', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--prompt', type=str, default="paraffin section")
    parser.add_argument('--image_prep', type=str, default='no_resize', help='the image preparation method')
    parser.add_argument('--batch_size', type=int, default=8)
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

    # initialize text embeddings
    tokenizer, text_encoder = initialize_text_encoder(args.model_path)
    a2b_tokens = tokenizer(args.prompt, max_length=tokenizer.model_max_length, padding="max_length",
                           truncation=True, return_tensors="pt").input_ids[0]
    text_emb = text_encoder(a2b_tokens.cuda().unsqueeze(0))[0].detach().cuda()
    text_emb_batch = text_emb.expand(args.batch_size, -1, -1)

    # data preprocess
    T_val = transforms.Compose([
        build_transform(args.image_prep),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    paths = glob.glob(os.path.join(args.img_path, '*'))
    
    for i in tqdm(range(0, len(paths), args.batch_size)):
        batch_paths = paths[i:i + args.batch_size]

        if len(batch_paths) != args.batch_size:
            text_emb_batch = text_emb.expand(len(batch_paths), -1, -1)

        save_paths = []
        for path in batch_paths:
            img_name = os.path.basename(path)
            save_path = os.path.join(args.output_path, img_name)
            save_paths.append(save_path)

        if os.path.exists(save_paths[-1]):
            continue
        
        batch_images = []
        original_sizes = []
        for path in batch_paths:
            img = Image.open(path).convert('RGB')
            original_sizes.append((img.width, img.height))
            batch_images.append(T_val(img))
        
        x_t = torch.stack(batch_images).cuda()

        # translate image
        with torch.no_grad():
            outputs = model(x_t, direction=args.direction, text_emb=text_emb_batch)

        # save image
        for j, output in enumerate(outputs):
            output_cpu = (output * 0.5 + 0.5).cpu().numpy().clip(0, 1)
            output_pil = Image.fromarray((output_cpu * 255).astype('uint8').transpose(1, 2, 0))
            os.makedirs(os.path.dirname(save_paths[j]), exist_ok=True)
            output_pil.resize(original_sizes[j]).save(save_paths[j])


if __name__ == "__main__":
    inference_args = parse_args_inference()
    main(inference_args)

