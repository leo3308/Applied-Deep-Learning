import torch
import torch.nn
import torch.optim as optim
import torchvision.transforms as Transform
from torchvision.utils import save_image

import numpy as np
import os
from argparse import ArgumentParser

from datasets import Anime, Shuffler
from WGAN_split import Generator, Discriminator
from utils import save_model, denorm, plot_loss, plot_classifier_loss, show_process
from utils import generation_by_attributes, get_random_label

parser = ArgumentParser()
parser.add_argument('-d', '--device', help = 'Device to train the model on', 
                    default = 'cpu', type = str)
parser.add_argument('-i', '--iterations', help = 'Number of iterations to train ACGAN', 
                    default = 50000, type = int)
parser.add_argument('-b', '--batch_size', help = 'Training batch size',
                    default = 128, type = int)
parser.add_argument('-t', '--train_dir', help = 'Training data directory', 
                    default = '../data', type = str)
parser.add_argument('-s', '--sample_dir', help = 'Directory to store generated images', 
                    default = './samples', type = str)
parser.add_argument('-c', '--checkpoint_dir', help = 'Directory to save model checkpoints', 
                    default = './checkpoints', type = str)
parser.add_argument('--sample', help = 'Sample every _ steps', 
                    default = 500, type = int)
parser.add_argument('--check', help = 'Save model every _ steps', 
                    default = 2000, type = int)
parser.add_argument('--lr', help = 'Learning rate of ACGAN. Default: 0.0002', 
                    default = 0.0001, type = float)
parser.add_argument('--beta', help = 'Momentum term in Adam optimizer. Default: 0.5', 
                    default = 0.5, type = float)
parser.add_argument('--classification_weight', help = 'Classification loss weight. Default: 1',
                    default = 1, type = float)
args = parser.parse_args()

if args.device == 'cuda' and not torch.cuda.is_available():
    print("Your device currenly doesn't support CUDA.")
    exit()
print('Using device: {}'.format(args.device))

def calculate_gradient_penalty(D, real_image, fake_image, batch_size, device):
    ''' calculate penalty gradient '''
    '''
    input args
    D : Discriminator
    real_image : (batch_size, channel, height, width) -> (*, 3, 128, 128)
    fake_image : (batch_size, channel, height, width) -> (*, 3, 128, 128)
    '''
    eta = torch.FloatTensor(batch_size, 1,1,1).uniform_(0, 1)
    eta = eta.expand(batch_size,
                     real_image.size(1),
                     real_image.size(2),
                     real_image.size(3)).to(device)
                     
    LAMBDA = 10

    interpolated = eta * real_image + ((1-eta) * fake_image)
    interpolated.requires_grad = True

    prob_interpolated, _, _, _, _ = D(interpolated)

    gradient = torch.autograd.grad(outputs=prob_interpolated,
                             inputs=interpolated,
                             grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                             create_graph=True,
                             retain_graph=True)[0]


    gradient = gradient.view(gradient.size(0), -1)
    gradient_penalty = ((gradient.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    
    return gradient_penalty

def main():
    batch_size = args.batch_size
    iterations =  args.iterations
    device = args.device
    
#    hair_classes, eye_classes = 12, 10
#    num_classes = hair_classes + eye_classes
    hair_class, eye_class, face_class, glass_class = 6, 4, 3, 2
    num_classes = hair_class + eye_class + face_class + glass_class
    latent_dim = 100
    smooth = 0.9
    
    config = 'WGANGP_batch{}_steps{}'.format(batch_size, iterations)
    print('Configuration: {}'.format(config))
    
    
    root_dir = './{}/images'.format(args.train_dir)
    tags_file = './{}/cartoon_attr.txt'.format(args.train_dir)
#    hair_prior = np.load('../{}/hair_prob.npy'.format(args.train_dir))
#    eye_prior = np.load('../{}/eye_prob.npy'.format(args.train_dir))

    random_sample_dir = '{}/{}/random_generation'.format(args.sample_dir, config)
    fixed_attribute_dir = '{}/{}/fixed_attributes'.format(args.sample_dir, config)
    checkpoint_dir = '{}/{}'.format(args.checkpoint_dir, config)
    
    if not os.path.exists(random_sample_dir):
    	os.makedirs(random_sample_dir)
    if not os.path.exists(fixed_attribute_dir):
    	os.makedirs(fixed_attribute_dir)
    if not os.path.exists(checkpoint_dir):
    	os.makedirs(checkpoint_dir)
        
    ########## Start Training ##########

    transform = Transform.Compose([Transform.ToTensor(),
                                   Transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = Anime(root_dir = root_dir, tags_file = tags_file, transform = transform)
    shuffler = Shuffler(dataset = dataset, batch_size = args.batch_size)
    
    G = Generator(latent_dim = latent_dim, class_dim = num_classes).to(device)
    D = Discriminator(hair_classes=hair_class, eye_classes=eye_class, face_classes=face_class, glass_classes=glass_class).to(device)

    G_optim = optim.Adam(G.parameters(), betas = [args.beta, 0.999], lr = args.lr)
    D_optim = optim.Adam(D.parameters(), betas = [args.beta, 0.999], lr = args.lr)
    
    d_log, g_log, classifier_log = [], [], []
    criterion = torch.nn.BCELoss()
#    criterion = torch.nn.NLLLoss()

    for step_i in range(1, iterations + 1):

        real_label = torch.ones(batch_size).to(device)
        fake_label = torch.zeros(batch_size).to(device)
        soft_label = torch.Tensor(batch_size).uniform_(smooth, 1).to(device)
        
        
        # we need gradient when training descriminator
        for p in D.parameters():
            p.requires_grad = True

        for d_iter in range(5):

            D_optim.zero_grad()
#            D.zero_grad()

            # Train discriminator
            real_img, hair_tags, eye_tags, face_tags, glass_tags = shuffler.get_batch()
            real_img, hair_tags, eye_tags, face_tags, glass_tags = real_img.to(device), hair_tags.to(device), eye_tags.to(device), face_tags.to(device), glass_tags.to(device)
            # real_tag = torch.cat((hair_tags, eye_tags), dim = 1)

            with torch.no_grad(): # totally freeze G , training D
                z = torch.randn(batch_size, latent_dim).to(device)
            
            fake_tag = get_random_label(batch_size = batch_size,
                                        hair_classes = hair_class,
                                        eye_classes = eye_class,
                                        face_classes = face_class,
                                        glass_classes = glass_class).to(device)

            real_img.requires_grad = True
            real_score, real_hair_predict, real_eye_predict, real_face_predict, real_glass_predict = D(real_img)

            real_discrim_loss = real_score.mean()
            
#            real_discrim_loss = criterion(real_score, soft_label)
#            fake_discrim_loss = criterion(fake_score, fake_label)

            real_hair_aux_loss = criterion(real_hair_predict, hair_tags)
            real_eye_aux_loss = criterion(real_eye_predict, eye_tags)
            real_face_aux_loss = criterion(real_face_predict, face_tags)
            real_glass_aux_loss = criterion(real_glass_predict, glass_tags)
            real_classifier_loss = real_hair_aux_loss + real_eye_aux_loss + real_face_aux_loss + real_glass_aux_loss

            fake_img = G(z, fake_tag).to(device)
            fake_score, _ , _ , _ , _ = D(fake_img)

            fake_discrim_loss = fake_score.mean()

            gradient_penalty = calculate_gradient_penalty(D, real_img.detach(), fake_img.detach(), batch_size, device)
        
#            discrim_loss = real_discrim_loss + fake_discrim_loss
            discrim_loss = fake_discrim_loss - real_discrim_loss + gradient_penalty
            classifier_loss = real_classifier_loss * args.classification_weight
        
            classifier_log.append(classifier_loss.item())

            D_loss = discrim_loss + classifier_loss
#            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()

        # Train generator
        for p in D.parameters():
            p.requires_grad = False

        G_optim.zero_grad()
#        G.zero_grad()

        z = torch.randn(batch_size, latent_dim).to(device)
        z.requires_grad = True
        fake_tag = get_random_label(batch_size = batch_size, 
                                    hair_classes = hair_class,
                                    eye_classes = eye_class,
                                    face_classes = face_class,
                                    glass_classes = glass_class).to(device)
    
        hair_tag = fake_tag[:, :hair_class]
        eye_tag = fake_tag[:, 6:10]
        face_tag = fake_tag[:, 10:13]
        glass_tag = fake_tag[:, 13:15]
        
        fake_img = G(z, fake_tag).to(device)
        
        fake_score, hair_predict, eye_predict, face_predict, glass_predict = D(fake_img)

        discrim_loss = fake_score.mean()
        G_discrim_loss = -discrim_loss
        
#        discrim_loss = criterion(fake_score, real_label)
        hair_aux_loss = criterion(hair_predict, hair_tag)
        eye_aux_loss = criterion(eye_predict, eye_tag)
        face_aux_loss = criterion(face_predict, face_tag)
        glass_aux_loss = criterion(glass_predict, glass_tag)
        classifier_loss = hair_aux_loss + eye_aux_loss + face_aux_loss + glass_aux_loss
        
        G_loss = classifier_loss * args.classification_weight + G_discrim_loss
#        G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()
            
        ########## Updating logs ##########
        d_log.append(D_loss.item())
        g_log.append(G_loss.item())
        show_process(total_steps = iterations, step_i = step_i,
        			 g_log = g_log, d_log = d_log, classifier_log = classifier_log)

        ########## Checkpointing ##########

        if step_i == 1:
            save_image(denorm(real_img[:16,:,:,:]), os.path.join(random_sample_dir, 'real.png'), nrow=4)
        if step_i % args.sample == 0:
            save_image(denorm(fake_img[:16,:,:,:]), os.path.join(random_sample_dir, 'fake_step_{}.png'.format(step_i)), nrow=4)
            
        if step_i % args.check == 0:
            save_model(model = G, optimizer = G_optim, step = step_i, log = tuple(g_log), 
                       file_path = os.path.join(checkpoint_dir, 'G_{}.ckpt'.format(step_i)))
            save_model(model = D, optimizer = D_optim, step = step_i, log = tuple(d_log), 
                       file_path = os.path.join(checkpoint_dir, 'D_{}.ckpt'.format(step_i)))

            plot_loss(g_log = g_log, d_log = d_log, file_path = os.path.join(checkpoint_dir, 'loss.png'))
            plot_classifier_loss(log = classifier_log, file_path = os.path.join(checkpoint_dir, 'classifier loss.png'))

            generation_by_attributes(model = G, device = args.device, step = step_i,
                                     latent_dim = latent_dim, hair_classes = hair_class,
                                     eye_classes = eye_class, face_classes = face_class,
                                     glass_classes = glass_class,
                                     sample_dir = fixed_attribute_dir)
    
if __name__ == '__main__':
    main()
            

        

    
    
    
