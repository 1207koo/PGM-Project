### global imports
import torch
from torchvision import transforms
import numpy as np
import sys
import glob
from PIL import Image
import os

### local imports
from args import args
sys.path.append('../easy')
import resnet
import wideresnet
import resnet12
import s2m2
import mlp
import sample

assert args.n_shots[0] == 1

### function to create model
def create_model(model_type):
    input_shape = [3, 84, 84]
    num_classes = 64 # (64, 16, 20, 600)
    few_shot = True

    model_type = model_type.lower()
    nondeterministic = (model_type[-3:] == '_nd')
    if nondeterministic:
        model_type = model_type[:-3]
    if model_type == "resnet18":
        return resnet.ResNet18(args.feature_maps, input_shape, num_classes, few_shot, args.rotations, nondeterministic=nondeterministic)
    if model_type == "resnet20":
        return resnet.ResNet20(args.feature_maps, input_shape, num_classes, few_shot, args.rotations, nondeterministic=nondeterministic)
    if model_type == "wideresnet":
        return wideresnet.WideResNet(args.feature_maps, input_shape, few_shot, args.rotations, num_classes = num_classes, nondeterministic=nondeterministic)
    if model_type == "resnet12":
        return resnet12.ResNet12(args.feature_maps, input_shape, num_classes, few_shot, args.rotations, nondeterministic=nondeterministic)
    if model_type[:3] == "mlp":
        return mlp.MLP(args.feature_maps, int(model_type[3:]), input_shape, num_classes, args.rotations, few_shot, nondeterministic=nondeterministic)
    if model_type == "s2m2r":
        return s2m2.S2M2R(args.feature_maps, input_shape, args.rotations, num_classes = num_classes, nondeterministic=nondeterministic)
    if model_type == 'sampler2':
        return sample.Sampler2(args.feature_maps, nondeterministic=nondeterministic)
    if model_type == 'sampler3':
        return sample.Sampler3(args.feature_maps, nondeterministic=nondeterministic)
    if model_type == 'discriminatorsingle2':
        return sample.DiscriminatorSingle2(args.feature_maps)
    if model_type == 'discriminatorsingle3':
        return sample.DiscriminatorSingle3(args.feature_maps)
    if model_type == 'discriminatordouble3':
        return sample.DiscriminatorDouble3(args.feature_maps)
    if model_type == 'discriminatordouble4':
        return sample.DiscriminatorDouble4(args.feature_maps)

with torch.no_grad():
    print('Model Loading...')
    model = {}
    model['model'] = create_model(args.model)
    model['model'].load_state_dict(torch.load(args.load_model, map_location='cpu'))
    if args.sampler_model != "":
        model['sampler'] = create_model(args.sampler_model)
        model['sampler'].load_state_dict(torch.load(args.load_sampler_model, map_location='cpu'))
    print('Model Loaded!')

    for k in ['model', 'sampler', 'discriminator']:
        if k not in model.keys():
            continue
        print("Number of trainable parameters in %s is: "%k + str(np.sum([p.numel() for p in model[k].parameters()])))
    print()

    images = []
    for class_path in sorted(glob.glob(os.path.join(args.dataset_path, args.dataset, 'test', '*')))[:args.n_ways]:
        for img_path in sorted(glob.glob(os.path.join(class_path, '*.*')))[:args.n_shots[0]+args.n_queries]:
            images.append(transforms.ToTensor()(np.array(Image.open(img_path).convert('RGB'))))
    print('Images loaded: %d images'%len(images))
    print('%d-way %d-shot, %d queries each'%(args.n_ways, args.n_shots[0], args.n_queries))
    images = torch.stack(images)

    descriptors = []
    descriptors.append(model['model'](images)[1])
    for _ in range(args.false_sample):
        descriptors.append(model['sampler'](descriptors[-1]))
    descriptors = torch.stack(descriptors)
    desc_num, _, desc_dim = descriptors.shape
    descriptors -= descriptors.mean(dim=1, keepdim=True)
    descriptors = descriptors / torch.norm(descriptors, dim=-1, p=2).unsqueeze(-1)
    print('Got descriptors from model')

    descriptors = descriptors.view(desc_num, args.n_ways, args.n_shots[0]+args.n_queries, desc_dim)
    support = descriptors[:, :, :args.n_shots[0]]
    query = descriptors[:, :, args.n_shots[0]:]
    distances = torch.norm(query.view(desc_num, 1, args.n_ways, 1, args.n_queries, 1, desc_dim) - support.view(1, desc_num, 1, args.n_ways, 1, args.n_shots[0], desc_dim), dim=-1, p=2)
    if args.support_aggregate == 'min':
        distances = distances.min(1)[0]
    else:
        distances = distances.mean(1)
    if args.query_aggregate == 'min':
        distances = distances.min(0)[0]
    else:
        distances = distances.mean(0)

    gt = []
    pd = []
    space = 6
    space_text = '%%%ds'%space
    print('\n Distances:')
    text = space_text%''
    for i in range(args.n_ways * args.n_shots[0]):
        text += space_text%str(i)
    text += space_text%'support'
    print(text)
    for i in range(args.n_ways * args.n_queries):
        text = space_text%str(i)
        iw = i // args.n_queries
        iq = i % args.n_queries
        for j in range(args.n_ways * args.n_shots[0]):
            jw = j // args.n_shots[0]
            jq = j % args.n_shots[0]
            text += space_text%('%.2f'%distances[iw, jw, iq, jq].item())
        gt.append(iw)
        pd.append(distances[iw, :, iq, :].min(1)[0].argmin().item())
        print(text)
    print(space_text%'query')

    print()
    for i in range(len(gt)):
        print('query%d: class%d (gt class%d)'%(i, pd[i], gt[i]))
    print('accuracy: %.2f %%'%(100.0 * (np.array(gt) == np.array(pd)).mean()))