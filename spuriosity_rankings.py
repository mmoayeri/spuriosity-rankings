import torch
import numpy as np
import cv2
import torchvision
from collections import OrderedDict
import os
from abc import ABC
import pickle
from tqdm import tqdm

### General caching and loading

def cache_data(cache_path: str, data_to_cache):
    os.makedirs('/'.join(cache_path.split('/')[:-1]), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(data_to_cache, f)

def load_cached_data(cache_path: str):
    with open(cache_path, 'rb') as f:
        dat = pickle.load(f)
    return dat

### Robust feature encoder loading

_ROBUST_MODELS_ROOT = '/cmlscratch/mmoayeri/models/pretrained-robust/'

def load_robust_encoder(mkey='resnet50_l2_eps3'):
    full_model_dict = torch.load(_ROBUST_MODELS_ROOT + mkey + '.ckpt')['model']
    arch = mkey.split('_')[0]
    model = torchvision.models.get_model(arch)

    # Reformat model_dict to be compatible with torchvision
    model_keys = [k for k in full_model_dict if 'attacker' not in k and 'normalizer' not in k]
    model_dict = dict({k.split('module.model.')[-1]:full_model_dict[k] for k in model_keys})
    model.load_state_dict(model_dict)
    
    normalizer_keys = [k for k in full_model_dict if 'attacker' not in k and 'normalizer' in k]
    normalizer_dict = dict({k.split('_')[-1]:full_model_dict[k] for k in normalizer_keys})
    normalizer = torchvision.transforms.Normalize(mean=normalizer_dict['mean'], std=normalizer_dict['std'])
    
    encoder = torch.nn.Sequential(
        OrderedDict([
            ('normalizer',normalizer), 
            *list(model.named_children())[:-1], 
            ('flatten', torch.nn.Flatten())
        ])
    )
    return encoder.eval().cuda()

### Computing and caching features

def compute_features(encoder, loader, cache_fname):
    """
    Expects model to have a fn 'forward_features' that maps inputs to features.
    Models from the timm library already have this built-in.

    Expects loader to return (inputs, labels) from a classification dataset.
    """
    if not os.path.exists(cache_fname):
        all_ftrs, labels = [], []
        encoder = encoder.eval().cuda()
        for dat in tqdm(loader):
            x, y = dat[0], dat[1]
            with torch.no_grad():
                ftrs = encoder(x.cuda()).flatten(1)
                all_ftrs.extend(ftrs.detach().cpu().numpy())
                labels.extend(y)
        ftrs, labels = [np.array(x) for x in [all_ftrs, labels]]
        # encoder = encoder.cpu()

        dat = dict({'ftrs': ftrs, 'labels': labels})
        cache_data(cache_fname, dat)
    else:
        dat = load_cached_data(cache_fname)
        ftrs, labels = [dat[x] for x in ['ftrs', 'labels']]
    return ftrs, labels

### Training or tuning linear heads -- used for both feature discovery and bias mitigation

def fit_head(ftrs_dict, labels_dict, close_spurious_gap_info=None, 
             init_head=None, lr=0.01, epochs=20, batch_size=128, verbose=False):
    train_ftrs, train_labels = [d['train'] for d in [ftrs_dict, labels_dict]]
    val_ftrs, val_labels = [d['val'] for d in [ftrs_dict, labels_dict]]

    accuracy_fn = Accuracy()
    
    head = torch.nn.Linear(train_ftrs.shape[1], max(train_labels)+1)
    if init_head:
        head.load_state_dict(init_head.state_dict())

    if close_spurious_gap_info:
        spurious_gap_eval_fn, stop_thresh = close_spurious_gap_info

    optimizer = torch.optim.SGD(list(head.parameters()), momentum=0.9, lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    rand_idx = np.arange(train_ftrs.shape[0])
    for epoch in range(epochs):
        head = head.train().cuda()
        cc, ctr, running_loss = 0, 0, 0
        np.random.shuffle(rand_idx)
        for i in range(0, train_ftrs.shape[0], batch_size):
            batch_idx = rand_idx[i:i+batch_size]
            batch_ftrs = torch.tensor(train_ftrs[batch_idx]).cuda()
            batch_ys = torch.LongTensor(train_labels[batch_idx]).cuda()
            logits = head(batch_ftrs)

            loss = criterion(logits, batch_ys)
            ctr += batch_ys.shape[0]
            cc += (logits.argmax(1) == batch_ys).sum().item()
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if verbose:
            train_acc = cc / ctr * 100
            val_acc = accuracy_fn.metric(head, val_ftrs, val_labels)
            msg = f'Epoch: {epoch+1:>3}/{epochs:>3}, Training Accuracy: {train_acc:.2f}%, Validation Accuracy: {val_acc:.2f}%'

            if close_spurious_gap_info:
                spurious_gap = spurious_gap_eval_fn.metric(head, val_ftrs, val_labels)
                msg += f', Spurious gap: {spurious_gap:.2f}%'
            print(msg)

            if spurious_gap < stop_thresh:
                break
    
    acc = accuracy_fn.metric(head, val_ftrs, val_labels)
    return head, acc

### Some evaluation functions used in linear layer training/tuning

class EvalFn(ABC):
    def metric(self, head, ftrs, labels):
        raise NotImplementedError

def basic_accuracy(head, ftrs, labels, batch_size=32):
    head = head.eval()
    with torch.no_grad():
        cc, ctr = 0, 0
        for i in range(0, ftrs.shape[0], batch_size):
            preds = head(torch.tensor(ftrs[i:i+batch_size]).cuda()).argmax(1)
            cc += (preds == torch.LongTensor(labels[i:i+batch_size]).cuda()).sum().item()
            ctr += preds.shape[0]
    return cc / ctr * 100

class Accuracy(EvalFn):
    def metric(self, head, ftrs, labels):
        return basic_accuracy(head, ftrs, labels)

class SpuriousGap(EvalFn):
    def __init__(self, low_spur_idx, high_spur_idx):
        self.low_spur_idx = low_spur_idx
        self.high_spur_idx = high_spur_idx

    def metric(self, head, ftrs, labels):
        accs = dict()
        for key, idx in zip(['low spuriosity', 'high spuriosity'], [self.low_spur_idx, self.high_spur_idx]):
            accs[key] = basic_accuracy(head, ftrs[idx], labels[idx])
        
        spurious_gap = accs['high spuriosity'] - accs['low spuriosity']
        return spurious_gap

### Utils for Generating and Visualizing NAMs; see salient imagenet repo: https://github.com/singlasahil14/salient_imagenet

def compute_feature_maps(images, model, layer_name='layer4'):
    x = images.cuda()
    for name, module in model._modules.items():
        x = module(x)
        if name == layer_name:
            break
    return x

def compute_nams(model, images, feature_index, layer_name='layer4'):
    b_size = images.shape[0]
    feature_maps = compute_feature_maps(images, model, layer_name=layer_name)
    nams = (feature_maps[:, feature_index, :, :]).detach()
    nams_flat = nams.view(b_size, -1) 
    nams_max, _ = torch.max(nams_flat, dim=1, keepdim=True)
    nams_flat = nams_flat/nams_max
    nams = nams_flat.view_as(nams)
    nams_resized = []
    for nam in nams:
        nam = nam.cpu().numpy()
        nam = cv2.resize(nam, images.shape[2:])
        nams_resized.append(nam)
    nams = np.stack(nams_resized, axis=0)
    nams = torch.from_numpy(1-nams)
    return nams

def compute_heatmaps(imgs, masks):
    imgs = imgs.swapaxes(1,2).swapaxes(2,3)
    heatmaps = []
    for (img, mask) in zip(imgs, masks):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap + np.float32(img)
        heatmap = heatmap / np.max(heatmap)
        heatmaps.append(heatmap)
    heatmaps = np.stack(heatmaps, axis=0)
    heatmaps = torch.from_numpy(heatmaps).permute(0, 3, 1, 2)
    return heatmaps

def grad_step(adv_inputs, grad, step_size):
    l = len(adv_inputs.shape) - 1
    grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l))
    scaled_grad = grad / (grad_norm + 1e-10)
    return adv_inputs + scaled_grad * step_size

def feature_attack(encoder, seed_images, feature_indices, eps=500, 
                   step_size=10, iterations=100):
    seed_images = seed_images.cuda()
    batch_size = seed_images.shape[0]
    for i in range(iterations+1):
        seed_images.requires_grad_()
        features = encoder(seed_images).flatten(1)
        features_select = features[torch.arange(batch_size), feature_indices]
        if i==iterations:
            seed_images = seed_images.detach().cpu()
            features_select = features_select.detach().cpu().numpy()
            break   
        adv_loss = features_select.sum()
        grads = torch.autograd.grad(adv_loss, [seed_images])[0]
        seed_images = grad_step(seed_images.detach(), grads, step_size)
        seed_images = torch.clamp(seed_images, min=0., max=1.)
    return seed_images, features_select

def find_important_ftrs(ftrs, labels, head):
    head = head.eval().cuda()
    preds = head(torch.tensor(ftrs).cuda()).argmax(1).detach().cpu().numpy()
    W = head.weight.detach().cpu().numpy()

    cc = np.where(labels == preds)[0]
    most_important_ftrs = dict()
    for i in range(max(labels) + 1):
        cls_cc_idx = cc[labels[cc] == i]
        avg_ftrs = ftrs[cls_cc_idx].mean(0)
        importances = avg_ftrs * W[i]
        most_important_ftrs[i] = np.argsort(-1*importances)

    return most_important_ftrs, preds

to_pil = torchvision.transforms.ToPILImage()
def visualize_ftr(cls_ind, ftr_ind, ftrs, labels, dset, encoder, save_path, preds=None, nrow=8, k=25, min_gap=40):
    gap, in_cls_idx, idx = measure_bias(cls_ind, ftr_ind, ftrs, labels, preds, k)
    if np.abs(gap) < min_gap:
        return

    top_imgs = torch.stack([dset[in_cls_idx[i]][0] for i in idx[:nrow]])
    bot_imgs = torch.stack([dset[in_cls_idx[i]][0] for i in idx[-1*nrow:]])
    nams = compute_nams(encoder, top_imgs, ftr_ind)
    hmaps = compute_heatmaps(top_imgs, nams)
    ftr_atks, _ = feature_attack(encoder, top_imgs, [ftr_ind])

    all_viz = torch.vstack([top_imgs, hmaps, ftr_atks, bot_imgs])

    grid = torchvision.utils.make_grid(all_viz, nrow=nrow, padding=4, pad_value=1)
    os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
    curr_save_path = save_path[:-4] + '.jpg'
    to_pil(grid).save(curr_save_path)

    return grid

### Measuring biases - used to only save visualizations for neural features that incur a bias

def measure_bias(cls_ind, ftr_ind, ftrs, labels, preds, k=100):
    in_cls_idx = np.where(labels == cls_ind)[0]
    idx = np.argsort(-1*ftrs[in_cls_idx][:, ftr_ind])
    gap = ((preds[in_cls_idx[idx[:k]]] == cls_ind).mean() - (preds[in_cls_idx[idx[-1*k:]]] == cls_ind).mean())*100
    return gap, in_cls_idx, idx

### Computing Rankings

def sort_data(ftrs_dict, labels_dict, spurious_ftrs_by_cls):
    # We obtain a ranking per class, tho we only really care about the top and bot idx
    bot_idx, top_idx = [dict({s:dict() for s in ['train', 'val']}) for _ in range(2)]
    for split in ['train', 'val']:
        ftrs, labels = [d[split] for d in [ftrs_dict, labels_dict]]
        for cls_ind, spur_ftrs in spurious_ftrs_by_cls.items():
            zscores = []
            cls_idx = np.where(labels == cls_ind)[0]
            for ftr_ind in spur_ftrs:
                vals = ftrs[cls_idx, ftr_ind]
                mean, std = vals.mean(), vals.std()
                zscores.append((vals - mean) / std)
            
            avg_zscores = np.array(zscores).mean(0)
            sorted_idx = np.argsort(avg_zscores)
            bot_idx[split][cls_ind] = cls_idx[sorted_idx[:1000]]
            top_idx[split][cls_ind] = cls_idx[sorted_idx[-1000:]]

    return bot_idx, top_idx


def consolidate_rankings(bot_idx, top_idx, spurious_ftrs_by_cls, k=100):
    ### Makes a single set of rankings per class
    low_spur_idx, high_spur_idx = [dict({split:[] for split in ['train', 'val']}) for _ in range(2)]
    for split in ['train', 'val']:
        for cls_ind in spurious_ftrs_by_cls:
            low_spur_idx[split].extend(bot_idx[split][cls_ind][:k])
            high_spur_idx[split].extend(top_idx[split][cls_ind][:k])

    return low_spur_idx, high_spur_idx

### Putting it all together

def ftr_discovery(train_dset, val_dset, dsetname, num_ftrs_per_class=15, min_gap=20):
    _RESULTS_ROOT = './'
    dset_results_root = os.path.join(_RESULTS_ROOT, dsetname)
    os.makedirs(dset_results_root, exist_ok=True)

    print("\n\nLoading interpretable feature encoder.")
    encoder = load_robust_encoder()
    
    print("\n\nEncoding images.")
    ftrs_dict, labels_dict = dict(), dict()
    for split, dset in zip(['train', 'val'], [train_dset, val_dset]):
        loader = torch.utils.data.DataLoader(dset, shuffle=False, num_workers=4, batch_size=32)
        ftrs, labels = compute_features(encoder, loader, os.path.join(dset_results_root, f"{split}_ftrs.pkl"))
        ftrs_dict[split] = ftrs
        labels_dict[split] = labels

    most_important_ftrs_path = os.path.join(dset_results_root, 'most_important_ftrs.pkl')
    predictions_path = os.path.join(dset_results_root, 'predictions.pkl')
    lin_head_path = os.path.join(dset_results_root, 'lin_head_data.pkl')
    if not os.path.exists(most_important_ftrs_path):
        print("\n\nTraining linear layer on encoded features.")
        eval_fn = Accuracy()
        head, acc = fit_head(ftrs_dict, labels_dict, eval_fn)
        print(f'\nLinear layer trained. Accuracy: {acc*100:.2f}%.')
        print("\n\nExtracting salient features.")

        most_important_ftrs, preds = find_important_ftrs(ftrs_dict['train'], labels_dict['train'], head)
        cache_data(most_important_ftrs_path, most_important_ftrs)
        cache_data(predictions_path, preds)

        head_data = dict({'state': head.state_dict(), 'val_acc': acc, 'shape': head.weight.shape})
        cache_data(head_data, head_save_path)
    else: 
        print("\n\nUsing cached important features and predictions.")
        most_important_ftrs = load_cached_data(most_important_ftrs_path)
        preds = load_cached_data(predictions_path)

    for cls_ind in most_important_ftrs:
        cls_save_root = os.path.join(dset_results_root, 'ftr_viz', f'class_{cls_ind}_{train_dset.classes[cls_ind].title().replace(" ","")}')
        os.makedirs(cls_save_root, exist_ok=True)
        in_cls_idx = np.where(labels_dict['train'] == cls_ind)[0]
        class_acc = (preds[in_cls_idx] == cls_ind).mean().item()*100
        print(f"\n\nProcessing {train_dset.classes[cls_ind]} class. Overall accuracy for class: {class_acc:.2f}%")
        for ftr_rank in tqdm(range(num_ftrs_per_class)):
            ftr_ind = most_important_ftrs[cls_ind][ftr_rank]
            save_path = os.path.join(cls_save_root, f"rank_{ftr_rank}_ftr_{ftr_ind}.jpg")
            visualize_ftr(cls_ind, ftr_ind, ftrs_dict['train'], labels_dict['train'], 
                          train_dset, encoder, save_path, preds=preds, min_gap=min_gap)

def mitigate_biases(head, ftrs_dict, labels_dict, spurious_ftrs_by_cls, desired_spur_gap):
    # Note that the provided head should match ftrs in ftrs_dict, 
    # so that head(ftrs) is the predictions of the model you wish to de-bias.
    bot_idx, top_idx = sort_data(ftrs_dict, labels_dict, spurious_ftrs_by_cls)
    low_spur_idx, high_spur_idx = consolidate_rankings(bot_idx, top_idx, spurious_ftrs_by_cls)
    spurious_gap_eval_fn = SpuriousGap(low_spur_idx['val'], high_spur_idx['val'])

    new_ftrs_dict, new_labels_dict = [
        dict({
            'train': d['train'][low_spur_idx['train']],
            'val': d['val']
        }) for d in [ftrs_dict, labels_dict]
    ]

    og_spur_gap = spurious_gap_eval_fn.metric(head.cuda(), new_ftrs_dict['val'], new_labels_dict['val'])
    accuracy = Accuracy()
    og_accuracy = accuracy.metric(head.cuda(), ftrs_dict['val'], labels_dict['val'])

    print(f'Before bias mitigation, the provided head achieves an accuracy of {og_accuracy:.2f}% and a spurious gap of {og_spur_gap:.2f}%')
    close_spurious_gap_info = (spurious_gap_eval_fn, desired_spur_gap)
    new_head, new_acc = fit_head(new_ftrs_dict, new_labels_dict, close_spurious_gap_info, init_head=head, verbose=True, lr=0.001)
    new_spur_gap = spurious_gap_eval_fn.metric(new_head, new_ftrs_dict['val'], new_labels_dict['val'])
    print(f'After bias mitigation, the tuned head achieves an accuracy of {new_acc:.2f}% and a spurious gap of {new_spur_gap:.2f}%')
    return new_head 