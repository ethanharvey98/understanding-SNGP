import os
import ast
import copy
import numpy as np
import pandas as pd
# PyTorch
import torch
import torchvision
import torchmetrics
# Importing our custom module(s)
import folds

def mixup(inputs, labels, alpha=0.2):
    
    labels = torch.nn.functional.one_hot(labels, num_classes=10)
    batch_size = len(inputs)
    
    dist = torch.distributions.beta.Beta(alpha, alpha)
    
    lambdas = dist.sample((batch_size,))
    inputs_shape = (-1,) + (1,) * (len(inputs.shape) - 1)
    labels_shape = (-1,) + (1,) * (len(labels.shape) - 1)

    shuffled_indices = torch.randperm(batch_size)
    
    inputs = lambdas.view(inputs_shape) * inputs + (1-lambdas).view(inputs_shape) * inputs[shuffled_indices]
    labels = lambdas.view(labels_shape) * labels + (1-lambdas).view(labels_shape) * labels[shuffled_indices]
    
    return inputs, labels

class _BoundedSpectralNorm(torch.nn.utils.parametrizations._SpectralNorm):
    """
    _BoundedSpectralNorm extends the _SpectralNorm class from PyTorch and adds a bound.
    
    Reference:
        For the original _SpectralNorm class implementation, see:
        https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/parametrizations.py
    """
    def __init__(self, weight, spec_norm_iteration=1, spec_norm_bound=1.0, dim=0, eps=1e-12):
        super(_BoundedSpectralNorm, self).__init__(weight, spec_norm_iteration, dim=0, eps=1e-12)
        self.spec_norm_bound = spec_norm_bound

    def forward(self, weight):
        if weight.ndim == 1:
            sigma = torch.linalg.vector_norm(weight)
            #return F.normalize(weight, dim=0, eps=self.eps)
            return self.spec_norm_bound * F.normalize(weight, dim=0, eps=self.eps) if self.spec_norm_bound < sigma else weight
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            sigma = torch.vdot(u, torch.mv(weight_mat, v))
            #return weight / sigma
            return self.spec_norm_bound * weight / sigma if self.spec_norm_bound < sigma else weight
        
def apply_bounded_spectral_norm(model, name='weight', spec_norm_iteration=1, spec_norm_bound=1.0):
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.utils.parametrize.register_parametrization(
                module, name, _BoundedSpectralNorm(getattr(module, name, None), spec_norm_iteration=spec_norm_iteration, spec_norm_bound=spec_norm_bound)
            )

def makedir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (self.X[index], self.y[index]) if self.transform == None else (self.transform(self.X[index]), self.y[index])

def get_cifar10_datasets(root, n, tune=True, random_state=42):
    assert n in [10, 50, 100, 150, 200, 250, 500, 1000, 5000, 10000, 50000], f'Invalid number of samples n={n}.'

    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(int(random_state))
    if not hasattr(random_state, 'rand'):
        raise ValueError('Not a valid random number generator')
    
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    cifar10_train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
    cifar10_test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, transform=transform, download=True)

    class_indices = {cifar10_label: [idx for idx, (image, label) in enumerate(cifar10_train_dataset) if label == cifar10_label] for cifar10_label in range(10)}
    shuffled_sampled_class_indices = {cifar10_label: random_state.choice(class_indices[cifar10_label], int(n/10), replace=False) for cifar10_label in class_indices.keys()}
    
    if tune:
        if n == 10:
            mask = random_state.choice([True, True, True, True, False]*2, 10, replace=False)
            train_indices = np.array(list(shuffled_sampled_class_indices.values())).flatten()[mask]
            val_or_test_indices = np.array(list(shuffled_sampled_class_indices.values())).flatten()[~mask]
        else:
            train_indices = {cifar10_label: shuffled_sampled_class_indices[cifar10_label][:int(4/5*int(n/10))] for cifar10_label in shuffled_sampled_class_indices.keys()}
            val_or_test_indices = {cifar10_label: shuffled_sampled_class_indices[cifar10_label][int(4/5*int(n/10)):] for cifar10_label in shuffled_sampled_class_indices.keys()}
            train_indices = np.array(list(train_indices.values())).flatten()
            val_or_test_indices = np.array(list(val_or_test_indices.values())).flatten()  
    else:
        train_indices = np.array(list(shuffled_sampled_class_indices.values())).flatten()

    sampled_train_images = torch.stack([cifar10_train_dataset[index][0] for index in train_indices])
    sampled_train_labels = torch.tensor([cifar10_train_dataset[index][1] for index in train_indices])
    mean = torch.mean(sampled_train_images, axis=(0, 2, 3))
    std = torch.std(sampled_train_images, axis=(0, 2, 3))
    augmented_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.Resize(size=(256, 256)),
        torchvision.transforms.RandomCrop(size=(224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
    ])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.Resize(size=(256, 256)),
        torchvision.transforms.CenterCrop(size=(224, 224)),
    ])
    
    if tune:
        val_or_test_dataset = [cifar10_train_dataset[index] for index in val_or_test_indices]
    else:
        val_or_test_dataset = cifar10_test_dataset
    
    sampled_val_or_test_images = torch.stack([image for image, label in val_or_test_dataset])
    sampled_val_or_test_labels = torch.tensor([label for image, label in val_or_test_dataset])

    augmented_train_dataset = Dataset(sampled_train_images, sampled_train_labels, augmented_transform)
    train_dataset = Dataset(sampled_train_images, sampled_train_labels, transform)
    val_or_test_dataset = Dataset(sampled_val_or_test_images, sampled_val_or_test_labels, transform)
                
    return augmented_train_dataset, train_dataset, val_or_test_dataset

def get_oxford_flowers_datasets(root, n, tune=True, random_state=42):

    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(int(random_state))
    if not hasattr(random_state, 'rand'):
        raise ValueError('Not a valid random number generator')

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    oxflowers_train_dataset = torchvision.datasets.Flowers102(root=root, split='train', transform=transform, download=True)
    oxflowers_test_dataset = torchvision.datasets.Flowers102(root=root, split='test', transform=transform, download=True)

    class_indices = {oxfl_label: [np.where(np.array(oxflowers_train_dataset._labels)==oxfl_label)[0]] for oxfl_label in range(102)}
    shuffled_sampled_class_indices = {oxfl_label: [random_state.choice(class_indices[oxfl_label][0], int(n/102), replace=False)] for oxfl_label in range(102)}

    if tune:
        if n < 510:
            mask = random_state.choice(([True]*82+[False]*20)*int(n/102), n, replace=False)
            train_indices = np.array(list(shuffled_sampled_class_indices.values())).flatten()[mask]
            val_or_test_indices = np.array(list(shuffled_sampled_class_indices.values())).flatten()[~mask]
        else:
            train_indices = {oxfl_label: shuffled_sampled_class_indices[oxfl_label][0][:int(4/5*int(n/102))] for oxfl_label in shuffled_sampled_class_indices.keys()}
            val_or_test_indices = {oxfl_label: shuffled_sampled_class_indices[oxfl_label][0][int(4/5*int(n/102)):] for oxfl_label in shuffled_sampled_class_indices.keys()}
            train_indices = np.array(list(train_indices.values())).flatten()
            val_or_test_indices = np.array(list(val_or_test_indices.values())).flatten()  
    else:
        train_indices = np.array(list(shuffled_sampled_class_indices.values())).flatten()

    # For Oxford Flowers we resize images before normalizing since images are different sizes
    resize = torchvision.transforms.Resize(size=(256, 256))
    sampled_train_images = torch.stack([resize(torchvision.io.read_image(str(oxflowers_train_dataset._image_files[ind])).float())/255 for ind in train_indices])
    sampled_train_labels = torch.tensor([oxflowers_train_dataset._labels[ind] for ind in train_indices]).squeeze()

    mean = torch.mean(sampled_train_images, axis=(0, 2, 3))
    std = torch.std(sampled_train_images, axis=(0, 2, 3))
    augmented_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.RandomCrop(size=(224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
    ])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.CenterCrop(size=(224, 224)),
    ])
    
    if tune:
        sampled_val_or_test_images = torch.stack([resize(torchvision.io.read_image(str(oxflowers_train_dataset._image_files[ind])).float())/255 for ind in val_or_test_indices])
        sampled_val_or_test_labels = torch.tensor([oxflowers_train_dataset._labels[ind] for ind in val_or_test_indices]).squeeze()
    else:
        sampled_val_or_test_images = torch.stack([resize(torchvision.io.read_image(str(path)).float())/255 for path in oxflowers_test_dataset._image_files])
        sampled_val_or_test_labels = torch.tensor([oxflowers_test_dataset._labels]).squeeze()  

    augmented_train_dataset = Dataset(sampled_train_images, sampled_train_labels, augmented_transform)
    train_dataset = Dataset(sampled_train_images, sampled_train_labels, transform)
    val_or_test_dataset = Dataset(sampled_val_or_test_images, sampled_val_or_test_labels, transform)
                
    return augmented_train_dataset, train_dataset, val_or_test_dataset

def get_oxford_pets_datasets(root, n, tune=True, random_state=42):
    assert n in [37, 185, 370, 555, 740, 925, 1850, 3441], f'Invalid number of samples n={n}.'

    if not os.path.exists(f'{root}/annotations'):
        URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
        torchvision.datasets.utils.download_and_extract_archive(URL, root)
    if not os.path.exists(f'{root}/images'):
        URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
        torchvision.datasets.utils.download_and_extract_archive(URL, root)
        
    # Read annotations into dataframes 
    trainval_df = pd.read_csv(f'{root}/annotations/trainval.txt', sep=' ', names=['image', 'id', 'species', 'breed_id'])
    test_df = pd.read_csv(f'{root}/annotations/test.txt', sep=' ', names=['image', 'id', 'species', 'breed_id'])
    # Add file paths to dataframe 
    trainval_df['path'] = trainval_df['image'].apply(lambda image: f'{root}/images/{image}.jpg')
    test_df['path'] = test_df['image'].apply(lambda image: f'{root}/images/{image}.jpg')
    
    if tune:
        if n < 370:
            sampled_df = trainval_df.groupby('id').apply(lambda group: group.sample(n=int(n/37), random_state=random_state)).reset_index(drop=True)
            # Sample 4/5 of trainval_df for training
            sampled_train_df = sampled_df.sample(n=int(4/5*n), random_state=random_state).reset_index(drop=True)
            # Use remaining 1/5 of trainval_df for validation
            sampled_val_or_test_df = sampled_df[~sampled_df.index.isin(sampled_train_df.index)]
            # Reset indices
            sampled_train_df = sampled_train_df.reset_index(drop=True)
            sampled_val_or_test_df = sampled_val_or_test_df.reset_index(drop=True)
        else:
            sampled_train_df = trainval_df.groupby('id').apply(lambda group: group.sample(n=int(n/37), random_state=random_state)[:int(4/5*int(n/37))]).reset_index(drop=True)
            sampled_val_or_test_df = trainval_df.groupby('id').apply(lambda group: group.sample(n=int(n/37), random_state=random_state)[int(4/5*int(n/37)):]).reset_index(drop=True)
    else:
        sampled_train_df = trainval_df.groupby('id').apply(lambda group: group.sample(n=int(n/37), random_state=random_state)).reset_index(drop=True)
        sampled_val_or_test_df = test_df
    
    # For Oxford-IIIT Pets we resize images before normalizing since images are different sizes
    resize = torchvision.transforms.Resize(size=(256, 256))
    # Some of the images have 4 channels (RGBA). We ignore the last channel.
    sampled_train_images = torch.stack([resize(torchvision.io.read_image(path).float()[:3,:,:])/255 for path in sampled_train_df.path])
    sampled_train_labels = torch.tensor([label-1 for label in sampled_train_df.id]).squeeze()
    mean = torch.mean(sampled_train_images, axis=(0, 2, 3))
    std = torch.std(sampled_train_images, axis=(0, 2, 3))
    augmented_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.RandomCrop(size=(224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
    ])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.CenterCrop(size=(224, 224)),
    ])
    
    sampled_val_or_test_images = torch.stack([resize(torchvision.io.read_image(path).float()[:3,:,:])/255 for path in sampled_val_or_test_df.path])
    sampled_val_or_test_labels = torch.tensor([label-1 for label in sampled_val_or_test_df.id]).squeeze()

    augmented_train_dataset = Dataset(sampled_train_images, sampled_train_labels, augmented_transform)
    train_dataset = Dataset(sampled_train_images, sampled_train_labels, transform)
    val_or_test_dataset = Dataset(sampled_val_or_test_images, sampled_val_or_test_labels, transform)
    return augmented_train_dataset, train_dataset, val_or_test_dataset

def get_aircrafts_datasets(root, n, tune=True, random_state=42):

    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(int(random_state))
    if not hasattr(random_state, 'rand'):
        raise ValueError('Not a valid random number generator')

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    aircraft_train_dataset = torchvision.datasets.FGVCAircraft(root=root, split='trainval', transform=transform, download=True)
    aircraft_test_dataset = torchvision.datasets.FGVCAircraft(root=root, split='test', transform=transform, download=True)

    class_indices = {airc_label: [np.where(np.array(aircraft_train_dataset._labels)==airc_label)[0]] for airc_label in range(100)}
    shuffled_sampled_class_indices = {airc_label: [random_state.choice(class_indices[airc_label][0],int(n/100),replace=False)] for airc_label in class_indices.keys()}

    if tune:
        if n == 100:
            mask = random_state.choice([True]*80 + [False]*20, 100, replace=False)
            train_indices = np.array(list(shuffled_sampled_class_indices.values())).flatten()[mask]
            val_or_test_indices = np.array(list(shuffled_sampled_class_indices.values())).flatten()[~mask]
        else:
            train_indices = {airc_label: shuffled_sampled_class_indices[airc_label][0][:int(4/5*int(n/100))] for airc_label in shuffled_sampled_class_indices.keys()}
            val_or_test_indices = {airc_label: shuffled_sampled_class_indices[airc_label][0][int(4/5*int(n/100)):] for airc_label in shuffled_sampled_class_indices.keys()}
            train_indices = np.array(list(train_indices.values())).flatten()
            val_or_test_indices = np.array(list(val_or_test_indices.values())).flatten()
    else:
        train_indices = np.array(list(shuffled_sampled_class_indices.values())).flatten()

    # For Aircrafts we resize images before normalizing since images are different sizes
    resize = torchvision.transforms.Resize(size=(256, 256))
    sampled_train_images = torch.stack([resize(torchvision.io.read_image(str(aircraft_train_dataset._image_files[ind])).float())/255 for ind in train_indices])
    sampled_train_labels = torch.tensor([aircraft_train_dataset._labels[ind] for ind in train_indices]).squeeze()

    mean = torch.mean(sampled_train_images, axis=(0, 2, 3))
    std = torch.std(sampled_train_images, axis=(0, 2, 3))
    augmented_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.RandomCrop(size=(224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
    ])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.CenterCrop(size=(224, 224)),
    ])

    if tune:
        sampled_val_or_test_images = torch.stack([resize(torchvision.io.read_image(str(aircraft_train_dataset._image_files[ind])).float())/255 for ind in val_or_test_indices])
        sampled_val_or_test_labels = torch.tensor([aircraft_train_dataset._labels[ind] for ind in val_or_test_indices]).squeeze()
    else:
        sampled_val_or_test_images = torch.stack([resize(torchvision.io.read_image(str(path)).float())/255 for path in aircraft_test_dataset._image_files])
        sampled_val_or_test_labels = torch.tensor([aircraft_test_dataset._labels]).squeeze()

    augmented_train_dataset = Dataset(sampled_train_images, sampled_train_labels, augmented_transform)
    train_dataset = Dataset(sampled_train_images, sampled_train_labels, transform)
    val_or_test_dataset = Dataset(sampled_val_or_test_images, sampled_val_or_test_labels, transform)

    return augmented_train_dataset, train_dataset, val_or_test_dataset

def get_ham10000_datasets(root, n, tune=True, random_state=42):
    # Load HAM10000 datasets (see HAM10000.ipynb to create labels.csv)
    ham10000_train_df = pd.read_csv(f'{root}/train/labels.csv', index_col='lesion_id')
    ham10000_test_df = pd.read_csv(f'{root}/test/labels.csv', index_col='lesion_id')
    ham10000_train_df.label = ham10000_train_df.label.apply(lambda item: ast.literal_eval(item))
    ham10000_test_df.label = ham10000_test_df.label.apply(lambda item: ast.literal_eval(item))
    # Randomly sample n datapoints from HAM10000 training DataFrame
    sampled_ham10000_train_df = ham10000_train_df.sample(n=n, random_state=random_state)
    if tune:
        # Create folds
        sampled_ham10000_train_df['Fold'] = folds.create_folds(sampled_ham10000_train_df, index_name='lesion_id', random_state=random_state)
        # Split folds
        train_df, val_or_test_df = folds.split_folds(sampled_ham10000_train_df)
    else:
        train_df = sampled_ham10000_train_df
        val_or_test_df = ham10000_test_df
    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda item: item/255),
    ])
    sampled_train_images = torch.stack([to_tensor(torchvision.io.read_image(path).float()) for path in train_df.path])
    train_mean = torch.mean(sampled_train_images, axis=(0, 2, 3))
    train_std = torch.std(sampled_train_images, axis=(0, 2, 3))
    augmented_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=train_mean, std=train_std),
        torchvision.transforms.Resize(size=(256, 256)),
        torchvision.transforms.RandomCrop(size=(224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
    ])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=train_mean, std=train_std),
        torchvision.transforms.Resize(size=(256, 256)),
        torchvision.transforms.CenterCrop(size=(224, 224)),
    ])
    sampled_train_images = torch.stack([to_tensor(torchvision.io.read_image(path).float()) for path in train_df.path])
    sampled_train_labels = torch.tensor([label for label in train_df.label]).squeeze()
    sampled_val_or_test_images = torch.stack([to_tensor(torchvision.io.read_image(path).float()) for path in val_or_test_df.path])
    sampled_val_or_test_labels = torch.tensor([label for label in val_or_test_df.label]).squeeze()
    # Create HAM10000 datasets
    augmented_train_dataset = Dataset(sampled_train_images, sampled_train_labels, augmented_transform)
    train_dataset = Dataset(sampled_train_images, sampled_train_labels, transform)
    val_or_test_dataset = Dataset(sampled_val_or_test_images, sampled_val_or_test_labels, transform)
    return augmented_train_dataset, train_dataset, val_or_test_dataset
        
def flatten_params(model):
    return torch.cat([param.view(-1) for param in model.parameters()])

def train_one_epoch(model, criterion, optimizer, dataloader, lr_scheduler=None, metric='accuracy', num_classes=10):

    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.train()

    if metric == 'accuracy':
        acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro')
    elif metric == 'auroc':
        acc = torchmetrics.AUROC(task='multiclass', num_classes=num_classes, average='macro')
    else:
        raise NotImplementedError(f'The specified metric \'{metric}\' is not implemented.')
    
    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'acc1': 0.0, 'bb_log_prior': 0.0, 'clf_log_prior': 0.0, 'labels': [], 'loss': 0.0, 'nll': 0.0, 'probabilities': []}

    for images, labels in dataloader:
                        
        if device.type == 'cuda':
            images, labels = images.to(device), labels.to(device)

        model.zero_grad()
        logits = model(images)
        params = flatten_params(model)
        losses = criterion(logits, labels, params[:criterion.d], params[criterion.d:])
        losses['loss'].backward()
        optimizer.step()

        batch_size = len(images)
        probabilities = torch.softmax(logits, dim=1)
        metrics['bb_log_prior'] += batch_size/dataset_size*losses['bb_log_prior'].item()
        metrics['clf_log_prior'] += batch_size/dataset_size*losses['clf_log_prior'].item()
        metrics['loss'] += batch_size/dataset_size*losses['loss'].item()
        metrics['nll'] += batch_size/dataset_size*losses['nll'].item()
        
        if lr_scheduler:
            lr_scheduler.step()

        if device.type == 'cuda':
            labels, probabilities = labels.cpu(), probabilities.cpu()
        
        for label, probability in zip(labels, probabilities):
            metrics['labels'].append(label)
            metrics['probabilities'].append(probability)

    labels = torch.stack(metrics['labels'])
    probabilities = torch.stack(metrics['probabilities'])
    metrics['acc1'] = acc(probabilities, labels).item()
            
    return metrics

def evaluate(model, criterion, dataloader, metric='accuracy', num_classes=10):
    
    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.eval()   
    
    if metric == 'accuracy':
        acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro')
    elif metric == 'auroc':
        acc = torchmetrics.AUROC(task='multiclass', num_classes=num_classes, average='macro')
    else:
        raise NotImplementedError(f'The specified metric \'{metric}\' is not implemented.')

    dataset_size = len(dataloader) * dataloader.batch_size if dataloader.drop_last else len(dataloader.dataset)
    metrics = {'acc1': 0.0, 'bb_log_prior': 0.0, 'clf_log_prior': 0.0, 'labels': [], 'loss': 0.0, 'nll': 0.0, 'probabilities': []}
            
    with torch.no_grad():
        for images, labels in dataloader:
                        
            if device.type == 'cuda':
                images, labels = images.to(device), labels.to(device)
                
            logits = model(images)
            params = flatten_params(model)
            losses = criterion(logits, labels, params[:criterion.d], params[criterion.d:])
            
            batch_size = len(images)
            probabilities = torch.softmax(logits, dim=1)
            metrics['bb_log_prior'] += batch_size/dataset_size*losses['bb_log_prior'].item()
            metrics['clf_log_prior'] += batch_size/dataset_size*losses['clf_log_prior'].item()
            metrics['loss'] += batch_size/dataset_size*losses['loss'].item()
            metrics['nll'] += batch_size/dataset_size*losses['nll'].item()

            if device.type == 'cuda':
                labels, probabilities = labels.cpu(), probabilities.cpu()
            
            for label, probability in zip(labels, probabilities):
                metrics['labels'].append(label)
                metrics['probabilities'].append(probability)

        labels = torch.stack(metrics['labels'])
        probabilities = torch.stack(metrics['probabilities'])
        metrics['acc1'] = acc(probabilities, labels).item()

    return metrics