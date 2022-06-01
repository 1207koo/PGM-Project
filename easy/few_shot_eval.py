import torch
import numpy as np
from args import *
from utils import *

n_runs = args.n_runs
batch_few_shot_runs = args.batch_fs
assert(n_runs % batch_few_shot_runs == 0)

def define_runs(n_ways, n_shots, n_queries, num_classes, elements_per_class):
    shuffle_classes = torch.LongTensor(np.arange(num_classes))
    run_classes = torch.LongTensor(n_runs, n_ways).to(args.device)
    run_indices = torch.LongTensor(n_runs, n_ways, n_shots + n_queries).to(args.device)
    for i in range(n_runs):
        run_classes[i] = torch.randperm(num_classes)[:n_ways]
        for j in range(n_ways):
            run_indices[i,j] = torch.randperm(elements_per_class[run_classes[i, j]])[:n_shots + n_queries]
    return run_classes, run_indices

def generate_runs(data, run_classes, run_indices, batch_idx):
    if len(data.shape) == 4:
        n_feat = data.shape[0]
        res_list = []
        for i in range(n_feat):
            res_list.append(generate_runs(data[i], run_classes, run_indices, batch_idx))
        return torch.stack(res_list)
    n_runs, n_ways, n_samples = run_classes.shape[0], run_classes.shape[1], run_indices.shape[2]
    run_classes = run_classes[batch_idx * batch_few_shot_runs : (batch_idx + 1) * batch_few_shot_runs]
    run_indices = run_indices[batch_idx * batch_few_shot_runs : (batch_idx + 1) * batch_few_shot_runs]
    run_classes = run_classes.unsqueeze(2).unsqueeze(3).repeat(1,1,data.shape[1], data.shape[2])
    run_indices = run_indices.unsqueeze(3).repeat(1, 1, 1, data.shape[2])
    datas = data.unsqueeze(0).repeat(batch_few_shot_runs, 1, 1, 1)
    cclasses = torch.gather(datas, 1, run_classes)
    res = torch.gather(cclasses, 2, run_indices)
    return res

def ncm(train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        if len(features.shape) == 3:
            train_features = train_features.unsqueeze(0)
            features = features.unsqueeze(0)
        dim = features.shape[-1]
        n_feat = features.shape[0]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:,:n_shots], dim = 3)
            distances = []
            for i in range(n_feat):
                for j in range(n_feat):
                    distances.append(torch.norm(runs[i,:,:,n_shots:].reshape(batch_few_shot_runs, args.n_ways, 1, -1, dim) - means[j].reshape(batch_few_shot_runs, 1, args.n_ways, 1, dim), dim = 4, p = 2))
            distances = torch.stack(distances).mean(0)
            winners = torch.min(distances, dim = 2)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def transductive_ncm(train_features, features, run_classes, run_indices, n_shots, n_iter_trans = args.transductive_n_iter, n_iter_trans_sinkhorn = args.transductive_n_iter_sinkhorn, temp_trans = args.transductive_temperature, alpha_trans = args.transductive_alpha, cosine = args.transductive_cosine, elements_train=None):
    with torch.no_grad():
        if len(features.shape) == 3:
            train_features = train_features.unsqueeze(0)
            features = features.unsqueeze(0)
        dim = features.shape[-1]
        n_feat = features.shape[0]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        if cosine:
            features = features / torch.norm(features, dim = 2, keepdim = True)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:,:n_shots], dim = 3)
            if cosine:
                means = means / torch.norm(means, dim = 3, keepdim = True)
            sims = []
            for i in range(n_feat):
                for j in range(n_feat):
                    for _ in range(n_iter_trans):
                        if cosine:
                            similarities = torch.einsum("bswd,bswd->bsw", runs[i,:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim), means[j].reshape(batch_few_shot_runs, 1, args.n_ways, dim))
                            soft_sims = torch.softmax(temp_trans * similarities, dim = 2)
                        else:
                            similarities = torch.norm(runs[i,:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim) - means[j].reshape(batch_few_shot_runs, 1, args.n_ways, dim), dim = 3, p = 2)
                            soft_sims = torch.exp( -1 * temp_trans * similarities)
                        for _ in range(n_iter_trans_sinkhorn):
                            soft_sims = soft_sims / soft_sims.sum(dim = 2, keepdim = True) * args.n_ways
                            soft_sims = soft_sims / soft_sims.sum(dim = 1, keepdim = True) * args.n_queries
                        new_means = ((runs[i,:,:,:n_shots].mean(dim = 2) * n_shots + torch.einsum("rsw,rsd->rwd", soft_sims, runs[i,:,:,n_shots:].reshape(runs[i].shape[0], -1, runs[i].shape[3])))) / runs[i].shape[2]
                        if cosine:
                            new_means = new_means / torch.norm(new_means, dim = 2, keepdim = True)
                        means[j] = means[j] * alpha_trans + (1 - alpha_trans) * new_means
                        if cosine:
                            means[j] = means[j] / torch.norm(means, dim = 2, keepdim = True)
                    sims.append(similarities)
            similarities = torch.stack(sims).mean(0)
            if cosine:
                winners = torch.max(similarities.reshape(runs.shape[1], runs.shape[2], runs.shape[3] - n_shots, -1), dim = 3)[1]
            else:
                winners = torch.min(similarities.reshape(runs.shape[1], runs.shape[2], runs.shape[3] - n_shots, -1), dim = 3)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def kmeans(train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        if len(features.shape) == 3:
            train_features = train_features.unsqueeze(0)
            features = features.unsqueeze(0)
        dim = features.shape[-1]
        n_feat = features.shape[0]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:,:n_shots], dim = 3)
            sims = []
            for i in range(n_feat):
                for j in range(n_feat):
                    for _ in range(500):
                        similarities = torch.norm(runs[i,:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim) - means[j].reshape(batch_few_shot_runs, 1, args.n_ways, dim), dim = 3, p = 2)
                        new_allocation = (similarities == torch.min(similarities, dim = 2, keepdim = True)[0]).float()
                        new_allocation = new_allocation / new_allocation.sum(dim = 1, keepdim = True)
                        allocation = new_allocation
                        means[j] = (runs[i,:,:,:n_shots].mean(dim = 2) * n_shots + torch.einsum("rsw,rsd->rwd", allocation, runs[i,:,:,n_shots:].reshape(runs[i].shape[0], -1, runs[i].shape[3])) * args.n_queries) / runs[i].shape[2]
                    sims.append(similarities)
            similarities = torch.stack(sims).mean(0)
            winners = torch.min(similarities.reshape(runs.shape[1], runs.shape[2], runs.shape[3] - n_shots, -1), dim = 3)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def softkmeans(train_features, features, run_classes, run_indices, n_shots, transductive_temperature_softkmeans=args.transductive_temperature_softkmeans, elements_train=None):
    with torch.no_grad():
        if len(features.shape) == 3:
            train_features = train_features.unsqueeze(0)
            features = features.unsqueeze(0)
        dim = features.shape[-1]
        n_feat = features.shape[0]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            runs = postprocess(runs)
            means = torch.mean(runs[:,:,:,:n_shots], dim = 3)
            sims = []
            for i in range(n_feat):
                for j in range(n_feat):
                    for _ in range(30):
                        similarities = torch.norm(runs[i,:,:,n_shots:].reshape(batch_few_shot_runs, -1, 1, dim) - means[j].reshape(batch_few_shot_runs, 1, args.n_ways, dim), dim = 3, p = 2)
                        soft_allocations = F.softmax(-similarities.pow(2)*args.transductive_temperature_softkmeans, dim=2)
                        means[j] = torch.sum(runs[i,:,:,:n_shots], dim = 2) + torch.einsum("rsw,rsd->rwd", soft_allocations, runs[i,:,:,n_shots:].reshape(runs[i].shape[0], -1, runs[i].shape[3]))
                        means[j] = means[j]/(n_shots+soft_allocations.sum(dim = 1).reshape(batch_few_shot_runs, -1, 1))
                    sims.append(similarities)
            winners = torch.min(similarities, dim = 2)[1]
            winners = winners.reshape(batch_few_shot_runs, args.n_ways, -1)
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def ncm_cosine(train_features, features, run_classes, run_indices, n_shots, elements_train=None):
    with torch.no_grad():
        if len(features.shape) == 3:
            train_features = train_features.unsqueeze(0)
            features = features.unsqueeze(0)
        dim = features.shape[-1]
        n_feat = features.shape[0]
        targets = torch.arange(args.n_ways).unsqueeze(1).unsqueeze(0).to(args.device)
        features = preprocess(train_features, features, elements_train=elements_train)
        features = sphering(features)
        scores = []
        for batch_idx in range(n_runs // batch_few_shot_runs):
            runs = generate_runs(features, run_classes, run_indices, batch_idx)
            means = torch.mean(runs[:,:,:,:n_shots], dim = 3)
            means = sphering(means)
            distances = []
            for i in range(n_feat):
                for j in range(n_feat):
                    distances.append(torch.einsum("bwysd,bwysd->bwys",runs[i,:,:,n_shots:].reshape(batch_few_shot_runs, args.n_ways, 1, -1, dim), means[j].reshape(batch_few_shot_runs, 1, args.n_ways, 1, dim)))
            distances = torch.stack(distances).mean(0)
            winners = torch.max(distances, dim = 2)[1]
            scores += list((winners == targets).float().mean(dim = 1).mean(dim = 1).to("cpu").numpy())
        return stats(scores, "")

def get_features(model, loader, n_aug = args.sample_aug):
    for k in model.keys():
        model[k].eval()
    if 'sampler' in model.keys():
        for augs in range(n_aug):
            all_features, offset, max_offset = [], 1000000, 0
            for i in range(args.false_sample + 1):
                all_features.append([])
            for batch_idx, (data, target) in enumerate(loader):        
                with torch.no_grad():
                    data, target = data.to(args.device), target.to(args.device)
                    _, features = model['model'](data)
                    all_features[0].append(features)
                    for i in range(args.false_sample):
                        features = model['sampler'](features)
                        all_features[i + 1].append(features)
                    offset = min(min(target), offset)
                    max_offset = max(max(target), max_offset)
            num_classes = max_offset - offset + 1
            print(".", end='')
            if augs == 0:
                features_total = [torch.cat(af, dim = 0).reshape(num_classes, -1, af[0].shape[1]) for af in all_features]
            else:
                features_total = [features_total[i] + torch.cat(all_features[i], dim = 0).reshape(num_classes, -1, all_features[i][0].shape[1]) for i in range(args.false_sample + 1)]
        return [features / n_aug for features in features_total]
    else:
        for augs in range(n_aug):
            all_features, offset, max_offset = [], 1000000, 0
            for batch_idx, (data, target) in enumerate(loader):        
                with torch.no_grad():
                    data, target = data.to(args.device), target.to(args.device)
                    _, features = model['model'](data)
                    all_features.append(features)
                    offset = min(min(target), offset)
                    max_offset = max(max(target), max_offset)
            num_classes = max_offset - offset + 1
            print(".", end='')
            if augs == 0:
                features_total = torch.cat(all_features, dim = 0).reshape(num_classes, -1, all_features[0].shape[1])
            else:
                features_total += torch.cat(all_features, dim = 0).reshape(num_classes, -1, all_features[0].shape[1])
        return features_total / n_aug

def eval_few_shot(train_features, val_features, novel_features, val_run_classes, val_run_indices, novel_run_classes, novel_run_indices, n_shots, transductive = False,elements_train=None):
    if transductive:
        if args.transductive_softkmeans:
            return softkmeans(train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), softkmeans(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)
        else:
            return kmeans(train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), kmeans(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)
    else:
        return ncm(train_features, val_features, val_run_classes, val_run_indices, n_shots, elements_train=elements_train), ncm(train_features, novel_features, novel_run_classes, novel_run_indices, n_shots, elements_train=elements_train)

def update_few_shot_meta_data(model, train_clean, novel_loader, val_loader, few_shot_meta_data):

    if "M" in args.preprocessing or args.save_features != '':
        train_features = get_features(model, train_clean)
    else:
        if 'sampler' in model.keys():
            train_features = torch.Tensor(0,0,0,0)
        else:
            train_features = torch.Tensor(0,0,0)
    val_features = get_features(model, val_loader)
    novel_features = get_features(model, novel_loader)

    res = []
    for i in range(len(args.n_shots)):
        res.append(evaluate_shot(i, train_features, val_features, novel_features, few_shot_meta_data, model = model))

    return res

def evaluate_shot(index, train_features, val_features, novel_features, few_shot_meta_data, model = None, transductive = False):
    (val_acc, val_conf), (novel_acc, novel_conf) = eval_few_shot(train_features, val_features, novel_features, few_shot_meta_data["val_run_classes"][index], few_shot_meta_data["val_run_indices"][index], few_shot_meta_data["novel_run_classes"][index], few_shot_meta_data["novel_run_indices"][index], args.n_shots[index], transductive = transductive, elements_train=few_shot_meta_data["elements_train"])
    if val_acc > few_shot_meta_data["best_val_acc"][index]:
        if val_acc > few_shot_meta_data["best_val_acc_ever"][index]:
            few_shot_meta_data["best_val_acc_ever"][index] = val_acc
            if args.save_model != "":
                if len(args.devices) == 1:
                    for k in model.keys():
                        if k == 'model':
                            torch.save(model[k].state_dict(), args.save_model + str(args.n_shots[index]))
                        else:
                            l = len(args.save_model.split('.')[-1]) + 1
                            torch.save(model[k].state_dict(), args.save_model[:-l] + '_' + k + args.save_model[-l:] + str(args.n_shots[index]))
                else:
                    for k in model.keys():
                        if k == 'model':
                            torch.save(model[k].module.state_dict(), args.save_model + str(args.n_shots[index]))
                        else:
                            l = len(args.save_model.split('.')[-1])
                            torch.save(model[k].module.state_dict(), args.save_model[:-l] + '_' + k + args.save_model[-l:] + str(args.n_shots[index]))
            if args.save_features != "":
                if 'sampler' in model.keys():
                    torch.save(torch.cat([train_features, torch.cat(val_features), torch.cat(novel_features)], dim = 0), args.save_features + str(args.n_shots[index]))
                else:
                    torch.save(torch.cat([train_features, val_features, novel_features], dim = 0), args.save_features + str(args.n_shots[index]))
        few_shot_meta_data["best_val_acc"][index] = val_acc
        few_shot_meta_data["best_novel_acc"][index] = novel_acc
    return val_acc, val_conf, novel_acc, novel_conf

print("eval_few_shot, ", end='')
