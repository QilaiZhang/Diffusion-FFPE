import os
import random
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from cleanfid.fid import get_files_features, frechet_distance, kernel_distance


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(path, seed=0, shuffle=False, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(path) or os.path.islink(path), '%s is not a valid directory' % path

    for root, _, fnames in sorted(os.walk(path, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    image_num = min(max_dataset_size, len(images))

    random.seed(seed)
    if shuffle:
        images = random.sample(images, int(image_num))
    else:
        images = images[:image_num]

    return images


def build_transform(image_prep="no_resize"):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    else:
        raise NotImplementedError("transform is not Implemented.")

    return T


def get_mu_sigma(path, feat_model, transform=None):
    files = make_dataset(path, shuffle=True)
    features = get_files_features(files, model=feat_model, num_workers=0, batch_size=2048, device='cuda', mode="clean",
                                  custom_fn_resize=None, description="", fdir=None, verbose=True,
                                  custom_image_tranform=transform)
    mu, sigma = np.mean(features, axis=0), np.cov(features, rowvar=False)
    return features, mu, sigma


def get_features(path, feat_model, transform=None):
    files = make_dataset(path, shuffle=True)
    features = get_files_features(files, model=feat_model, num_workers=0, batch_size=512, device='cuda', mode="clean",
                                  custom_fn_resize=None, description="", fdir=None, verbose=True,
                                  custom_image_tranform=transform)
    return features


def calculate_fid(ref_path, test_path, feat_model):
    if ref_path.endswith(".npz"):
        loaded_arrays = np.load(ref_path)
        ref_mu, ref_sigma = loaded_arrays['mu'], loaded_arrays['sigma']
    else:
        _, ref_mu, ref_sigma = get_mu_sigma(ref_path, feat_model)

    _, ed_mu, ed_sigma = get_mu_sigma(test_path, feat_model)
    fid_score = frechet_distance(ref_mu, ref_sigma, ed_mu, ed_sigma)
    return fid_score


def calculate_kid(ref_path, test_path, feat_model):
    if ref_path.endswith(".npz"):
        loaded_arrays = np.load(ref_path)
        ref_features = loaded_arrays['features']
    else:
        ref_features = get_features(ref_path, feat_model)

    ed_features = get_features(test_path, feat_model)
    kid_score = kernel_distance(ref_features, ed_features)
    return kid_score


def calculate_dino(data_path, test_path, net_dino):
    img_names = os.listdir(data_path)
    l_dino_scores = []
    for name in tqdm(img_names):
        fake = Image.open(os.path.join(data_path, name)).convert("RGB")
        real = Image.open(os.path.join(test_path, name)).convert("RGB")
        a = net_dino.preprocess(fake).unsqueeze(0).cuda()
        b = net_dino.preprocess(real).unsqueeze(0).cuda()
        dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
        l_dino_scores.append(dino_ssim)

    return np.mean(l_dino_scores)


def evaluate(model, net_dino, img_path, fixed_emb, direction, fid_output_dir, img_prep, num_images):
    l_dino_scores = []
    T_val = build_transform(img_prep)

    for idx, input_img_path in enumerate(tqdm(img_path)):
        if idx > num_images > 0:
            break
        file_name = os.path.join(fid_output_dir, os.path.basename(input_img_path).replace(".png", ".jpg"))
        with torch.no_grad():
            input_img = T_val(Image.open(input_img_path).convert("RGB"))
            img_a = transforms.ToTensor()(input_img)
            img_a = transforms.Normalize([0.5], [0.5])(img_a).unsqueeze(0).cuda()
            eval_fake_b = model(img_a, direction, fixed_emb[0:1])
            eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b[0] * 0.5 + 0.5)
            eval_fake_b_pil.save(file_name)
            a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
            b = net_dino.preprocess(eval_fake_b_pil).unsqueeze(0).cuda()
            dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
            l_dino_scores.append(dino_ssim)

    return l_dino_scores
