from collections import OrderedDict
import torch
import argparse


def obtain_skysense_hr(ckpt):
    # For high-resolution optical data

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('backbone_gep.'):
            new_k = k.replace('backbone_gep.', 'backbone.')
            print(f'{k} -> {new_k}', flush=True)
        elif k.startswith('backbone_s1.'):
            continue
        elif k.startswith('backbone_s2.'):
            continue
        elif k.startswith('fusion.'):
            continue
        elif k.startswith('head_gep'):
            continue
        elif k.startswith('head'):
            continue
        else:
            new_k = k

        new_ckpt[new_k] = v

    return new_ckpt


def obtain_skysense_s2(ckpt):
    # For sentinel-2 data

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('backbone_gep.'):
            continue
        elif k.startswith('backbone_s1.'):
            continue
        elif k.startswith('backbone_s2.'):
            new_k = k.replace('backbone_s2.', '')
            print(f'{k} -> {new_k}', flush=True)
        elif k.startswith('fusion.'):
            continue
        elif k.startswith('head_gep'):
            continue
        elif k.startswith('head'):
            continue
        else:
            new_k = k

        new_ckpt[new_k] = v

    return new_ckpt


def obtain_skysense_s1(ckpt):
    # For sentinel-1 data

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('backbone_gep.'):
            continue
        elif k.startswith('backbone_s2.'):
            continue
        elif k.startswith('backbone_s1.'):
            new_k = k.replace('backbone_s1.', '')
            print(f'{k} -> {new_k}', flush=True)
        elif k.startswith('fusion.'):
            continue
        elif k.startswith('head_gep'):
            continue
        elif k.startswith('head'):
            continue
        else:
            new_k = k

        new_ckpt[new_k] = v

    return new_ckpt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the pre-trained weights of SkySense")
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--data-type',
                        type=str,
                        choices=['rgb', 's2', 's1', 'rgbnir', 'mm'])
    parser.add_argument('--output-path', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    checkpoint = torch.load(args.input_path, map_location='cpu')['model']
    if args.data_type == 'rgb':
        # Swin Transformer v2-huge
        weight = obtain_skysense_hr(checkpoint)
    elif args.data_type == 's2':
        # Vision Transformer-large
        weight = obtain_skysense_s2(checkpoint)
    elif args.data_type == 's1':
        # Vision Transformer-large
        weight = obtain_skysense_s1(checkpoint)
    elif args.data_type == 'rgbnir':
        # Swin Transformer v2-huge
        weight = obtain_skysense_hr(checkpoint)
        # In order to support optical RGBNIR image inputs, we recommend copying the weights of the red band to the near-infrared band.
        weight['backbone.patch_embed.projection.weight'] = torch.cat(
            (weight['backbone.patch_embed.projection.weight'],
             weight['backbone.patch_embed.projection.weight']
             [:, 0, :, :].unsqueeze(1)),
            dim=1)
    elif args.data_type == 'mm':
        weight = checkpoint
    else:
        print('Error. Please verify that the data input type is correct.',
              flush=True)

    torch.save(weight, args.output_path)
    print('=========>saved<==========', flush=True)
