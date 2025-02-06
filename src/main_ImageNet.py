import argparse
import os
import time
import types
import pandas as pd
import wandb
# PyTorch
import torch
import torchvision
import torchmetrics
# Importing our custom module(s)
import layers
import losses
import utils

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res
    
def evaluate(model, criterion, dataloader):
    
    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.eval()   
    
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'acc1': 0.0, 'acc5': 0.0, 'loss': 0.0, 'nll': 0.0}
            
    with torch.no_grad():
        for images, labels in dataloader:
            
            batch_size = len(images)
                        
            if device.type == 'cuda':
                images, labels = images.to(device), labels.to(device)
                
            params = torch.nn.utils.parameters_to_vector(model.fc.linear.parameters())
            logits = model(images)
            losses = criterion(labels, logits, params, dataset_size)
            
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
            metrics['acc1'] += (batch_size/dataset_size)*acc1.item()
            metrics['acc5'] += (batch_size/dataset_size)*acc5.item()
            metrics['loss'] += (batch_size/dataset_size)*losses['loss'].item()
            metrics['nll'] += (batch_size/dataset_size)*losses['nll'].item()
    
    return metrics

def train_one_epoch(model, criterion, optimizer, dataloader, lr_scheduler=None):

    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.train()
        
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'acc1': 0.0, 'acc5': 0.0, 'loss': 0.0, 'nll': 0.0, 'tau': 0.0}

    for images, labels in dataloader:
        
        batch_size = len(images)
        
        if device.type == 'cuda':
            images, labels = images.to(device), labels.to(device)

        model.zero_grad()
        params = torch.nn.utils.parameters_to_vector(model.fc.linear.parameters())
        logits = model(images)
        losses = criterion(labels, logits, params, dataset_size)
        losses['loss'].backward()
        optimizer.step()

        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        metrics['acc1'] += (batch_size/dataset_size)*acc1.item()
        metrics['acc5'] += (batch_size/dataset_size)*acc5.item()
        metrics['loss'] += (batch_size/dataset_size)*losses['loss'].item()
        metrics['nll'] += (batch_size/dataset_size)*losses['nll'].item()
        metrics['tau'] += (batch_size/dataset_size)*losses['tau_star'].item()
        
        if lr_scheduler:
            lr_scheduler.step()
            
    return metrics

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--batch_size', default=32, help='Batch size (default: 32)', type=int)
    parser.add_argument('--epochs', default=90, help='Number of epochs (default: 90)', type=int)
    parser.add_argument('--experiments_directory', default='', help='Directory to save experiments (default: \'\')', type=str)
    parser.add_argument('--kappa', default=1.0, help='TODO', type=float)
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
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'experiments_directory': args.experiments_directory,
                'kappa': args.kappa,
                'lr_0': args.lr_0,
                'model_name': args.model_name,
                'num_workers': args.num_workers,
                'random_state': args.random_state,
                'wandb': args.wandb,
                'wandb_project': args.wandb_project,
            }
        )
        
    torch.manual_seed(args.random_state)
    utils.makedir_if_not_exist(args.experiments_directory)
    
    augmented_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # 1. Random resize and crop trianing images to 224.
        torchvision.transforms.RandomResizedCrop(size=(224, 224)),
        # 2. Random horizontal flip
        torchvision.transforms.RandomHorizontalFlip(),
        # 3. Normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # 1. Resize training images to 256.
        torchvision.transforms.Resize(size=(256, 256)),
        # 2. Center crop training images to 224.
        torchvision.transforms.CenterCrop(size=(224, 224)),
        # 3. Normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    augmented_train_dataset = torchvision.datasets.ImageFolder(root='/cluster/tufts/hugheslab/datasets/ImageNet/train/', transform=augmented_transform)
    train_dataset = torchvision.datasets.ImageFolder(root='/cluster/tufts/hugheslab/datasets/ImageNet/train/', transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root='/cluster/tufts/hugheslab/datasets/ImageNet/val/', transform=transform)

    augmented_train_loader = torch.utils.data.DataLoader(augmented_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet50(weights=weights)
    model.fc = layers.RandomFeatureGaussianProcess(in_features=2048, out_features=1000)
    model.fc.sigma_param = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(1e-4))))
    utils.add_variational_layers(model.fc, model.fc.sigma_param)
    model.use_posterior = types.MethodType(utils.use_posterior, model)
    model.fc.lengthscale = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(20))))
    torch.nn.utils.parametrize.register_parametrization(model.fc, 'lengthscale', torch.nn.Softplus())
    model.fc.outputscale = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(1.0))))
    torch.nn.utils.parametrize.register_parametrization(model.fc, 'outputscale', torch.nn.Softplus())
    #utils.apply_bounded_spectral_norm(model, name='weight', spec_norm_iteration=1, spec_norm_bound=6.0)
    model.to(device)
    
    criterion = losses.ERMLoss(criterion=torch.nn.CrossEntropyLoss())
    criterion = losses.DataEmphasizedELBo(args.kappa, model.fc.sigma_param, criterion=torch.nn.CrossEntropyLoss())
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, weight_decay=1e-4, momentum=0.9)
    #optimizer = torch.optim.SGD(model.fc.parameters(), lr=args.lr_0, weight_decay=1e-6, momentum=0.9)
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=args.lr_0, weight_decay=0.0, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    last_epoch = -1
    if os.path.exists(f'{args.experiments_directory}/{args.model_name}.pt'):
        checkpoint = torch.load(f'{args.experiments_directory}/{args.model_name}.pt', weights_only=False)
        model_history_df = pd.read_csv(f'{args.experiments_directory}/{args.model_name}.csv', index_col=0)
        last_epoch = checkpoint['epoch']
        torch.set_rng_state(checkpoint['random_state'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    else:
        #columns = ['epoch', 'train_acc1', 'train_acc5', 'train_loss', 'train_nll', 'val_acc1', 'val_acc5', 'val_loss', 'val_nll']
        columns = ['epoch', 'lengthscale', 'outputscale', 'sigma', 'tau', 'train_acc1', 'train_acc5', 'train_loss', 'train_nll', 'val_acc1', 'val_acc5', 'val_loss', 'val_nll']
        model_history_df = pd.DataFrame(columns=columns)
        
    for epoch in range(last_epoch+1, args.epochs):
        train_metrics = train_one_epoch(model, criterion, optimizer, augmented_train_loader)
        lr_scheduler.step()
        #train_metrics = evaluate(model, criterion, train_loader)
        val_metrics = evaluate(model, criterion, val_loader)
            
        #row = [epoch, train_metrics['acc1'], train_metrics['acc5'], train_metrics['loss'], train_metrics['nll'], val_metrics['acc1'], val_metrics['acc5'], val_metrics['loss'], val_metrics['nll']]
        row = [epoch, model.fc.lengthscale.item(), model.fc.outputscale.item(), torch.nn.functional.softplus(model.fc.sigma_param).item(), train_metrics['tau'], train_metrics['acc1'], train_metrics['acc5'], train_metrics['loss'], train_metrics['nll'], val_metrics['acc1'], val_metrics['acc5'], val_metrics['loss'], val_metrics['nll']]
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
        
        if args.wandb:
            wandb.log({
                'epoch': epoch,
                'lengthscale': model.fc.lengthscale,
                'lr': lr_scheduler.get_last_lr()[0],
                'outputscale': model.fc.outputscale,
                'sigma': torch.nn.functional.softplus(model.fc.sigma_param).item(),
                'tau': train_metrics['tau'],
                'train_acc1': train_metrics['acc1'],
                'train_acc5': train_metrics['acc5'],
                'train_loss': train_metrics['loss'],
                'train_nll': train_metrics['nll'],
                'val_acc1': val_metrics['acc1'],
                'val_acc5': val_metrics['acc5'],
                'val_loss': val_metrics['loss'],
                'val_nll': val_metrics['nll'],
            })
