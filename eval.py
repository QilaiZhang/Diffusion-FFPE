import os
import argparse
import numpy as np
from diffusion_ffpe.my_utils import get_mu_sigma, calculate_fid, get_features, calculate_kid
from cleanfid.fid import build_feature_extractor


def parse_args():
    parser = argparse.ArgumentParser(description="Script for calculating statistics.")
    parser.add_argument("--data_path", type=str, default="TEST_FFPE_PATH")
    parser.add_argument("--ref_path", type=str, default="testB_statistic.npz")
    parser.add_argument("--save-stats", action='store_true', default=False)
    parser.add_argument("--fid", action='store_true', default=True)
    parser.add_argument("--kid", action='store_true', default=False)
    args = parser.parse_args()
    return args


def main(args):
    print(os.path.abspath(args.data_path))

    if args.fid:
        feat_model = build_feature_extractor("clean", 'cuda', use_dataparallel=False)
        if args.save_stats:
            features, mu, sigma = get_mu_sigma(args.data_path, feat_model)
            np.savez(args.ref_path, mu=mu, sigma=sigma, features=features)
            print("The statistics has been saved to {}".format(os.path.abspath(args.ref_path)))
        else:
            fid_score = calculate_fid(args.ref_path, args.data_path, feat_model)
            print("FID: {}".format(fid_score))

    if args.kid:
        feat_model = build_feature_extractor("clean", 'cuda', use_dataparallel=False)
        if args.save_stats:
            features = get_features(args.data_path, feat_model)
            np.savez(args.ref_path, features=features)
            print("The statistics has been saved to {}".format(os.path.abspath(args.ref_path)))
        else:
            kid_score = calculate_kid(args.ref_path, args.data_path, feat_model)
            print("KID: {}".format(kid_score))


if __name__ == '__main__':
    input_args = parse_args()
    main(input_args)
