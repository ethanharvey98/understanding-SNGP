import argparse
import os
import random
import numpy as np
import pandas as pd
import wandb
# PyTorch
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
# Importing our custom module(s)
import losses
import models
import utils

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main.py') 
    parser.add_argument('--batch_size', default=128, help='Batch size (default: 128)', type=int)
    parser.add_argument('--bb_weight_decay', default=1e-2, help='Backbone weight decay (default: 1e-2)', type=float)
    parser.add_argument('--clf_weight_decay', default=1e-2, help='Classifier weight decay (default: 1e-2)', type=float)
    parser.add_argument('--dataset_path', default='', help='Path to dataset (default: \'\')', type=str)
    parser.add_argument('--experiments_path', default='', help='Path to save experiments (default: \'\')', type=str)
    parser.add_argument('--k', default=5, help='Rank of low-rank covariance matrix (default: 5)', type=float)
    parser.add_argument('--lr_0', default=0.5, help='Initial learning rate (default: 0.5)', type=float)
    parser.add_argument('--m', default=6, help='Number of ensembles in prior (default: 6)', type=int)
    parser.add_argument('--model_name', default='test', help='Model name (default: \'test\')', type=str)
    parser.add_argument('--n', default=1000, help='Number of training samples (default: 1000)', type=int)
    parser.add_argument('--num_workers', default=1, help='Number of workers (default: 1)', type=int)
    parser.add_argument('--prior_eps', default=1e-1, help='Added to prior variance (default: 1e-1)', type=float) # Default from "Pre-Train Your Loss"
    parser.add_argument('--prior_path', help='Path to saved priors (default: \'\')', type=str)
    parser.add_argument('--prior_type', default='StdPrior', help='Determines criterion', type=str)
    parser.add_argument('--save', action='store_true', default=False, help='Whether or not to save the model (default: False)')
    parser.add_argument('--tune', action='store_true', default=False, help='Whether validation or test set is used (default: False)')
    parser.add_argument('--random_state', default=42, help='Random state (default: 42)', type=int)
    parser.add_argument('--wandb', action='store_true', default=False, help='Whether or not to log to wandb')
    parser.add_argument('--wandb_project', default='test', help='Wandb project name (default: \'test\')', type=str)    
    args = parser.parse_args()
    
    if args.wandb:
        # TODO: Add prior_type to wandb config
        wandb.login()
        os.environ['WANDB_API_KEY'] = '4bfaad8bea054341b5e8729c940150221bdfbb6c'
        wandb.init(
            project = args.wandb_project,
            name = args.model_name,
            config={
                'batch_size': args.batch_size,
                'bb_weight_decay': args.bb_weight_decay,
                'clf_weight_decay': args.clf_weight_decay,
                'experiments_path': args.experiments_path,
                'clf_weight_decay': args.clf_weight_decay,
                'k': args.k,
                'lr_0': args.lr_0,
                'model_name': args.model_name,
                'n': args.n,
                'num_workers': args.num_workers,
                'prior_eps': args.prior_eps,
                'prior_path': args.prior_path,
                'save': args.save,
                'tune': args.tune,
                'random_state': args.random_state,
                'wandb': args.wandb,
                'wandb_project': args.wandb_project,                
            }
        )

    torch.manual_seed(args.random_state)
    utils.makedir_if_not_exist(args.experiments_path)           
           
    augmented_train_dataset, train_dataset, val_or_test_dataset = utils.get_cifar10_datasets(root=args.dataset_path, n=args.n, tune=args.tune, random_state=args.random_state)
    # PyTorch DataLoaders
    augmented_train_loader = torch.utils.data.DataLoader(augmented_train_dataset, batch_size=min(args.batch_size, len(augmented_train_dataset)), shuffle=True, num_workers=args.num_workers, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)), num_workers=args.num_workers)
    val_or_test_loader = torch.utils.data.DataLoader(val_or_test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        
    num_heads = 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    checkpoint = torch.load(f'{args.prior_path}/resnet50_torchvision_model.pt', map_location=torch.device('cpu'))
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Identity()
    model.load_state_dict(checkpoint)
    utils.apply_bounded_spectral_norm(model, spec_norm_bound=6.0)
    model.fc = models.RandomFeatureGaussianProcess(in_features=2048, out_features=num_heads, rank=1024)
    model.to(device)
    
    if args.prior_type == 'StdPrior':
        d = len(torch.load(f'{args.prior_path}/resnet50_ssl_prior_mean.pt', map_location=torch.device('cpu')))
        bb_prior = {
            'cov_diag': torch.ones(d),
            'cov_factor': torch.zeros(1, d),
            'k': 1,
            'loc': torch.zeros(d),
        }
        clf_prior = {
            'cov_diag': torch.ones((2048*num_heads)+num_heads),
            'cov_factor': torch.zeros(1, (2048*num_heads)+num_heads),
            'loc': torch.zeros((2048*num_heads)+num_heads),
        }
        bb_weight_decay = 0 if args.bb_weight_decay == 0 else 1/(len(augmented_train_dataset) * args.bb_weight_decay)
        clf_weight_decay = 0 if args.clf_weight_decay == 0 else 1/(len(augmented_train_dataset) * args.clf_weight_decay)
        criterion = losses.MAPTransferLearning(bb_prior, bb_weight_decay, clf_prior, clf_weight_decay, device, len(augmented_train_dataset))
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, momentum=0.9, weight_decay=0, nesterov=True)
    elif args.prior_type == 'nonlearned':
        # Note: StdPrior experiments were run using the following code.
        criterion = losses.CustomCrossEntropyLoss()
        
        other_params = [param for name, param in model.named_parameters() if 'fc' not in name]
        fc_params = [param for name, param in model.named_parameters() if 'fc' in name]
        optimizer = torch.optim.SGD([
            {'params': other_params, 'weight_decay': args.clf_weight_decay},
            {'params': fc_params, 'weight_decay': 1e-6},
        ], lr=args.lr_0, momentum=0.9, nesterov=True)
        
        #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, momentum=0.9, weight_decay=args.clf_weight_decay, nesterov=True)
    elif args.prior_type == 'LearnedPriorIso':
        d = len(torch.load(f'{args.prior_path}/resnet50_ssl_prior_mean.pt', map_location=torch.device('cpu')))
        bb_prior = {
            'cov_diag': torch.ones(d),
            'cov_factor': torch.zeros(1, d),
            'k': 1,
            'loc': torch.load(f'{args.prior_path}/resnet50_ssl_prior_mean.pt', map_location=torch.device('cpu')),
        }
        clf_prior = {
            'cov_diag': torch.ones((2048*num_heads)+num_heads),
            'cov_factor': torch.zeros(1, (2048*num_heads)+num_heads),
            'loc': torch.zeros((2048*num_heads)+num_heads),
        }
        bb_weight_decay = 0 if args.bb_weight_decay == 0 else 1/(len(augmented_train_dataset) * args.bb_weight_decay)
        clf_weight_decay = 0 if args.clf_weight_decay == 0 else 1/(len(augmented_train_dataset) * args.clf_weight_decay)
        criterion = losses.MAPTransferLearning(bb_prior, bb_weight_decay, clf_prior, clf_weight_decay, device, len(augmented_train_dataset))
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, momentum=0.9, weight_decay=0, nesterov=True)
    elif args.prior_type == 'adapted':
        # Note: LearnedPriorIso experiments were run using the following code.
        loc = torch.load(f'{args.prior_path}/resnet50_torchvision_mean.pt', map_location=torch.device('cpu'))
        loc = torch.cat((loc, torch.zeros((2048*num_heads)+num_heads)))
        criterion = losses.MAPAdaptationLoss(loc, args.clf_weight_decay, m=args.m)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, momentum=0.9, weight_decay=0, nesterov=True)
    elif args.prior_type == 'LearnedPriorLR':
        bb_prior = {
            'cov_diag': torch.load(f'{args.prior_path}/resnet50_torchvision_variance.pt', map_location=torch.device('cpu')),
            'cov_factor': torch.load(f'{args.prior_path}/resnet50_torchvision_covmat.pt', map_location=torch.device('cpu')),
            'prior_eps': args.prior_eps,
            'loc': torch.load(f'{args.prior_path}/resnet50_torchvision_mean.pt', map_location=torch.device('cpu')),
        }
        bb_prior['cov_factor'] = bb_prior['cov_factor'][:args.k]
        # $\Sigma = \frac{1}{2} ( \Sigma_{\text{diag}} + \Sigma_{\text{LR}} )$
        bb_prior['cov_diag'] = (1/2)*bb_prior['cov_diag']
        bb_prior['cov_factor'] = np.sqrt(1/2)*bb_prior['cov_factor']
        # $\Sigma_{\text{LR}} = \frac{1}{k-1} Q Q^T$
        bb_prior['cov_factor'] = np.sqrt(1/(args.k-1))*bb_prior['cov_factor']
        clf_prior = {
            'cov_diag': torch.ones((2048*num_heads)+num_heads),
            'cov_factor': torch.zeros(1, (2048*num_heads)+num_heads),
            'loc': torch.zeros((2048*num_heads)+num_heads),
        }
        clf_weight_decay = 0 if args.clf_weight_decay == 0 else 1/(len(augmented_train_dataset) * args.clf_weight_decay)
        criterion = losses.MAPTransferLearning(bb_prior, args.bb_weight_decay, clf_prior, clf_weight_decay, device, len(augmented_train_dataset))
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, momentum=0.9, weight_decay=0, nesterov=True)
    else:
        raise NotImplementedError(f'The specified prior type \'{args.prior_type}\' is not implemented.')

    steps = 6000
    num_batches = len(augmented_train_loader)
    epochs = int(steps/num_batches)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*num_batches)

    columns = ['epoch', 'train_acc1', 'train_bb_log_prior', 'train_clf_log_prior', 'train_loss', 'train_nll', 'val_or_test_acc1', 'val_or_test_bb_log_prior', 'val_or_test_clf_log_prior', 'val_or_test_loss', 'val_or_test_nll']
    model_history_df = pd.DataFrame(columns=columns)
        
    for epoch in range(epochs):
        train_metrics = utils.train_one_epoch(model, criterion, optimizer, augmented_train_loader, scheduler)
        #train_metrics = utils.evaluate(model, criterion, train_loader)
        
        if args.tune or epoch == epochs-1:
            val_or_test_metrics = utils.evaluate(model, criterion, val_or_test_loader)
        else:
            val_or_test_metrics = {'acc1': 0.0, 'bb_log_prior': 0.0, 'clf_log_prior': 0.0, 'loss': 0.0, 'nll': 0.0}
                            
        # Append evaluation metrics to DataFrame
        row = [epoch, train_metrics['acc1'], train_metrics['bb_log_prior'], train_metrics['clf_log_prior'], train_metrics['loss'], train_metrics['nll'], val_or_test_metrics['acc1'], val_or_test_metrics['bb_log_prior'], val_or_test_metrics['clf_log_prior'], val_or_test_metrics['loss'], val_or_test_metrics['nll']]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        if args.wandb:
            wandb.log({
                'epoch': epoch, 
                'train_acc1': train_metrics['acc1'], 
                'train_bb_log_prior': train_metrics['bb_log_prior'], 
                'train_clf_log_prior': train_metrics['clf_log_prior'], 
                'train_loss': train_metrics['loss'], 
                'train_nll': train_metrics['nll'], 
                'val_or_test_acc1': val_or_test_metrics['acc1'], 
                'val_or_test_bb_log_prior': val_or_test_metrics['bb_log_prior'], 
                'val_or_test_clf_log_prior': val_or_test_metrics['clf_log_prior'], 
                'val_or_test_loss': val_or_test_metrics['loss'], 
                'val_or_test_nll': val_or_test_metrics['nll'], 
            })
        
        model_history_df.to_csv(f'{args.experiments_path}/{args.model_name}.csv')
    
    if args.save:
        torch.save(model.state_dict(), f'{args.experiments_path}/{args.model_name}.pt')
        