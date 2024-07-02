import argparse
import os
import random
import numpy as np
import pandas as pd
import wandb
# PyTorch
import torch
import torchvision
# Importing our custom module(s)
import models
import utils

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main.py')    
    parser.add_argument('--epochs', default=90, help='Number of epochs (default: 90)', type=int)
    parser.add_argument('--experiments_directory', default='', help='Directory to save experiments (default: \'\')', type=str)
    parser.add_argument('--cycles', default=1, help='Number of cycles (default: 1)', type=int)
    parser.add_argument('--lr_0', default=0.1, help='Initial learning rate (default: 0.1)', type=float)
    parser.add_argument('--model_name', default='test', help='Model name (default: \'test\')', type=str)
    parser.add_argument('--num_workers', default=16, help='Number of workers (default: 16)', type=int)
    parser.add_argument('--random_state', default=42, help='Random state (default: 42)', type=int)
    parser.add_argument('--wandb', action='store_true', default=False, help='Whether or not to log to wandb')
    parser.add_argument('--wandb_project', default='test', help='Wandb project name (default: \'test\')', type=str)
    args = parser.parse_args()
    
    if args.wandb:
        wandb.login()
        os.environ['WANDB_API_KEY'] = '4bfaad8bea054341b5e8729c940150221bdfbb6c'
        wandb.init(
            project = args.wandb_project,
            name = args.model_name,
            config={
                'epochs': args.epochs,
                'experiments_directory': args.experiments_directory,
                'cycles': args.cycles,
                'lr_0': args.lr_0,
                'model_name': args.model_name,
                'num_workers': args.num_workers,
                'random_state': args.random_state,
                'wandb': args.wandb,
                'wandb_project': args.wandb_project,
            }
        )
        
    torch.manual_seed(args.random_state)
    if args.experiments_directory: utils.makedir_if_not_exist(args.experiments_directory)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    #model = torchvision.models.resnet50(weights=weights).to(device)
    #model = torchvision.models.resnet50().to(device)
    model = torchvision.models.resnet50()
    #model.fc = torch.nn.Linear(in_features=2048, out_features=1000, bias=True)
    utils.apply_bounded_spectral_norm(model, spec_norm_bound=6.0)
    model.fc = models.RandomFeatureGaussianProcess(in_features=2048, out_features=1000)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    other_params = [param for name, param in model.named_parameters() if 'fc' not in name]
    fc_params = [param for name, param in model.named_parameters() if 'fc' in name]
    optimizer = torch.optim.SGD([
        {'params': other_params, 'weight_decay': 1e-4},
        {'params': fc_params, 'weight_decay': 0.0},
    ], lr=args.lr_0, momentum=0.9)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, momentum=0.9, weight_decay=1e-4)
    
    augmented_train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # 1. Random resize and crop trianing images to 224.
        torchvision.transforms.RandomResizedCrop(size=(224, 224)),
        # 2. Random horizontal flip
        torchvision.transforms.RandomHorizontalFlip(),
        # 3. Normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # 1. Resize training images to 256.
        torchvision.transforms.Resize(size=(256, 256)),
        # 2. Center crop training images to 224.
        torchvision.transforms.CenterCrop(size=(224, 224)),
        # 3. Normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # 1. Resize validation images to 256.
        torchvision.transforms.Resize(size=(256, 256)),
        # 2. Center crop validation images to 224.
        torchvision.transforms.CenterCrop(size=(224, 224)),
        # 3. Normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    augmented_train_dataset = torchvision.datasets.ImageFolder(root='/cluster/tufts/hugheslab/datasets/ImageNet/train/', transform=augmented_train_transform)
    train_dataset = torchvision.datasets.ImageFolder(root='/cluster/tufts/hugheslab/datasets/ImageNet/train/', transform=train_transform)
    val_dataset = torchvision.datasets.ImageFolder(root='/cluster/tufts/hugheslab/datasets/ImageNet/val/', transform=val_transform)
    
    augmented_train_loader = torch.utils.data.DataLoader(augmented_train_dataset, batch_size=32, shuffle=True, num_workers=args.num_workers)        
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=args.num_workers)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    last_epoch = -1
    if os.path.exists(f'{args.experiments_directory}/{args.model_name}.pt'):
        checkpoint = torch.load(f'{args.experiments_directory}/{args.model_name}.pt')
        model_history_df = pd.read_csv(f'{args.experiments_directory}/{args.model_name}.csv', index_col=0)
        last_epoch = checkpoint['epoch']
        torch.set_rng_state(checkpoint['random_state'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    else:
        columns = ['epoch', 'train_acc1', 'train_acc5', 'train_loss', 'val_acc1', 'val_acc5', 'val_loss']
        model_history_df = pd.DataFrame(columns=columns)
        
    for epoch in range(last_epoch+1, args.epochs):
        
        train_metrics = utils.train_one_epoch(model, criterion, optimizer, augmented_train_loader)
        lr_scheduler.step()
        val_metrics = utils.evaluate(model, criterion, val_loader)

        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'lr': lr_scheduler.get_last_lr()[0],
                'train_acc1': train_metrics['acc1'],
                'train_acc5': train_metrics['acc5'],
                'train_loss': train_metrics['loss'],
                'val_acc1': val_metrics['acc1'],
                'val_acc5': val_metrics['acc5'],
                'val_loss': val_metrics['loss'],
            })
            
        row = [epoch, train_metrics['acc1'], train_metrics['acc5'], train_metrics['loss'], val_metrics['acc1'], val_metrics['acc5'], val_metrics['loss']]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        model_history_df.to_csv(f'{args.experiments_directory}/{args.model_name}.csv')
        
        torch.save({
            'epoch': epoch,
            'random_state': torch.get_rng_state(),
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, f'{args.experiments_directory}/{args.model_name}.pt')