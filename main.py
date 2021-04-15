import argparse
import os

from trainer import Trainer

def main(cfg):
    trainer = Trainer(cfg)
    if cfg.mode == 'train':
        trainer.train()
    else:
        trainer.test()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='GAN model practice')
    
    # Dataset
    parser.add_argument('--img_dir', type=str, default='/datasets/CELEBA/img_align_celeba', help='path of image data')
    parser.add_argument('--sample_dir', type=str, default='/home/hyunin/GAN_practice/assets/sample', help='path of sample image data')
    parser.add_argument('--test_dir', type=str, default='/home/hyunin/GAN_practice/assets/test', help='path of saving generated image data')
    parser.add_argument('--g_ckpt_dir', type=str, default='/home/hyunin/GAN_practice/assets/g_ckpt', help='checkpoint path')
    parser.add_argument('--d_ckpt_dir', type=str, default='/home/hyunin/GAN_practice/assets/d_ckpt', help='checkpoint path')
    parser.add_argument('--img_size', type=int, default=64, help='size of image')

    # Training 
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    parser.add_argument('--scheduler_name', type=str, default='No_scheduler', choices=['No_scheduler', 'ReduceLROnPlateau', 'CosineAnnealingLR'], help='get scheduler')
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--max_g_dim', type=int, default=1024)
    parser.add_argument('--max_d_dim', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=128, help='size of batch size')
    parser.add_argument('--max_iter', type=int, default=15_820, help='size of batch size')

    parser.add_argument('--loss', type=str, default='vanilla', help='loss function')

    parser.add_argument('--g_optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'RMSProp'])
    parser.add_argument('--d_optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'RMSProp'])
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD optimizer')

    parser.add_argument('--factor', type=float, default=0.2)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--T_max', type=int, default=6)
    parser.add_argument('--min_lr', type=float, default=1e-6)

    # result
    parser.add_argument('--ckpt_every', type=int, default=1_000, help='number of iterations between saving ckpts')
    parser.add_argument('--sample_every', type=int, default=1_000, help='number of iterations between saving fake samples')
    parser.add_argument('--eval_every', type=int, default=2_000, help='number of iterations between evaluation')
    parser.add_argument('--print_every', type=int, default=500, help='number of iterations between stdouts')

    # test
    parser.add_argument('--test_batch_size', type=int, default=64, help='batch size in test')
    
    args = parser.parse_args()
    print(args)

    main(args)
    