import torch
from torchvision.utils import save_image

import os
import utils
import WGAN_split
from tqdm import tqdm
from argparse import ArgumentParser

def get_label(test_file):

    with open(test_file, 'r') as file:
        file.readline()
        file.readline()
        label = []
        for line in file:
            line = line.split()
            label.append([int(l) for l in line])
    return label

def generate_images(model, label, output_dir, device):
    #torch.manual_seed(42)
    all_tag = torch.FloatTensor(label).to(device)
    #z = torch.randn(1, 100).to(device)
#    tag = all_tag[:64,:]
#    output_img = model(z, tag)
#    save_image(utils.denorm(output_img), './test.png', nrow=16)
    for i in tqdm(range(len(label))):
        z = torch.randn(1, 100).to(device)
        tag = all_tag[i].unsqueeze(0)
        output = model(z, tag)
        save_image(utils.denorm(output),
                   os.path.join(output_dir, '{}.png'.format(i)))

def main():
    parser = ArgumentParser()
    parser.add_argument('--test_file', type=str,
                        help='testing label file')
    parser.add_argument('--model', type=str,
                        help='pretrained generator model')
    parser.add_argument('--output_dir', type=str,
                        help='output images dir')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device you want to run this code')
    args = parser.parse_args()
    label = get_label(args.test_file)
    
    G = WGAN_split.Generator(latent_dim=100, class_dim=15)
    checkpoint = torch.load(args.model)
    G.load_state_dict(checkpoint['model'])
    G.to(args.device)
    G = G.eval()
    generate_images(G, label, args.output_dir, args.device)

if __name__ == '__main__':
    main()



