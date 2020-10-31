# Improving Auto-Augment via Augmentation-Wise Weight Sharing
Unofficial [AWS AutoAugment](https://arxiv.org/abs/2009.14737 ) implementation in PyTorch.

- AWS AutoAugment learns augmentation policies using augmentation-wise shared model weights


<p align="center">
<img src="etc/method.png" height=350>
</p>

## To-do
### Essentials
- [x] baseline structure
- [x] augmentation list
- [ ] shared policy
- [ ] augmentaition-wise shared model weights
- [ ] PPO+baseline trick
- [ ] CIFAR100 WRN 
- [ ] CIFAR100 Shake-Shake
- [ ] CIFAR100 PyramidNet+ShakeDrop

### Possible Modification
- [ ] other policy optimization methods
- [ ] weight exponential moving average for shared model weights
- [ ] Stocastic Depth 
- [ ] CIFAR10 
- [ ] ImageNet

### Future Works
- [ ] FastAugment + AWS
- [ ] ProxylessNAS + AWS
- [ ] Gradient-basedNAS + AWS

## Results

### CIFAR-10 / 100

Search : **3.5 GPU Hours (1428x faster than AutoAugment)**, WResNet-40x2 on Reduced CIFAR-10

| Model(CIFAR-10)         | Baseline   | Cutout     | AutoAugment | Fast AutoAugment<br/>(transfer/direct) |   |
|-------------------------|------------|------------|-------------|------------------|----|
| Wide-ResNet-40-2        | 5.3        | 4.1        | 3.7         | 3.6 / 3.7        | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar10_wresnet40x2_top1_3.52.pth) |
| Wide-ResNet-28-10       | 3.9        | 3.1        | 2.6         | 2.7 / 2.7        | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar10_wresnet28x10_top1.pth) |
| Shake-Shake(26 2x32d)   | 3.6        | 3.0        | 2.5         | 2.7 / 2.5        | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar10_shake26_2x32d_top1_2.68.pth) |
| Shake-Shake(26 2x96d)   | 2.9        | 2.6        | 2.0         | 2.0 / 2.0        | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar10_shake26_2x96d_top1_1.97.pth) |
| Shake-Shake(26 2x112d)  | 2.8        | 2.6        | 1.9         | 2.0 / 1.9        | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar10_shake26_2x112d_top1_2.04.pth) |
| PyramidNet+ShakeDrop    | 2.7        | 2.3        | 1.5         | 1.8 / 1.7        | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar10_pyramid272_top1_1.44.pth) |

| Model(CIFAR-100)      | Baseline   | Cutout     | AutoAugment | Fast AutoAugment<br/>(transfer/direct) |    |
|-----------------------|------------|------------|-------------|------------------|----|
| Wide-ResNet-40-2      | 26.0       | 25.2       | 20.7        | 20.7 / 20.6      | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar100_wresnet40x2_top1_20.43.pth) |
| Wide-ResNet-28-10     | 18.8       | 18.4       | 17.1        | 17.3 / 17.3      | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar100_wresnet28x10_top1_17.17.pth) |
| Shake-Shake(26 2x96d) | 17.1       | 16.0       | 14.3        | 14.9 / 14.6      | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar100_shake26_2x96d_top1_15.15.pth) |
| PyramidNet+ShakeDrop  | 14.0       | 12.2       | 10.7        | 11.9 / 11.7      | [Download](https://arena.kakaocdn.net/brainrepo/fast-autoaugment/cifar100_pyramid272_top1_11.74.pth) |



## Run

We conducted experiments under

- python 3.7.0
- pytorch 1.6.0, torchvision 0.5.0, cuda10

### Search a augmentation policy

Please read ray's document to construct a proper ray cluster : https://github.com/ray-project/ray, and run search.py with the master's redis address.

```
$ python search.py -c confs/wresnet40x2_cifar10_b512.yaml --dataroot ... --redis ...
```

### Train a model with found policies

You can train network architectures on CIFAR-10 / 100 and ImageNet with our searched policies.

- fa_reduced_cifar10 : reduced CIFAR-10(4k images), WResNet-40x2
- fa_reduced_imagenet : reduced ImageNet(50k images, 120 classes), ResNet-50

```
$ export PYTHONPATH=$PYTHONPATH:$PWD
$ python FastAutoAugment/train.py -c confs/wresnet40x2_cifar10_b512.yaml --aug fa_reduced_cifar10 --dataset cifar10
$ python FastAutoAugment/train.py -c confs/wresnet40x2_cifar10_b512.yaml --aug fa_reduced_cifar10 --dataset cifar100
$ python FastAutoAugment/train.py -c confs/wresnet28x10_cifar10_b512.yaml --aug fa_reduced_cifar10 --dataset cifar10
$ python FastAutoAugment/train.py -c confs/wresnet28x10_cifar10_b512.yaml --aug fa_reduced_cifar10 --dataset cifar100
...
$ python FastAutoAugment/train.py -c confs/resnet50_b512.yaml --aug fa_reduced_imagenet
$ python FastAutoAugment/train.py -c confs/resnet200_b512.yaml --aug fa_reduced_imagenet
```

By adding --only-eval and --save arguments, you can test trained models without training.

If you want to train with multi-gpu/node, use `torch.distributed.launch` such as

```bash
$ python -m torch.distributed.launch --nproc_per_node={num_gpu_per_node} --nnodes={num_node} --master_addr={master} --master_port={master_port} --node_rank={0,1,2,...,num_node} FastAutoAugment/train.py -c confs/efficientnet_b4.yaml --aug fa_reduced_imagenet
```

## References & Opensources

We increase the batch size and adapt the learning rate accordingly to boost the training. Otherwise, we set other hyperparameters equal to AutoAugment if possible. For the unknown hyperparameters, we follow values from the original references or we tune them to match baseline performances.

- **ResNet** : [paper1](https://arxiv.org/abs/1512.03385), [paper2](https://arxiv.org/abs/1603.05027), [code](https://github.com/osmr/imgclsmob/tree/master/pytorch/pytorchcv/models)
- **PyramidNet** : [paper](https://arxiv.org/abs/1610.02915), [code](https://github.com/dyhan0920/PyramidNet-PyTorch)
- **Wide-ResNet** : [code](https://github.com/meliketoy/wide-resnet.pytorch)
- **Shake-Shake** : [code](https://github.com/owruby/shake-shake_pytorch)
- **ShakeDrop Regularization** : [paper](https://arxiv.org/abs/1802.02375), [code](https://github.com/owruby/shake-drop_pytorch)
- **AutoAugment** : [code](https://github.com/tensorflow/models/tree/master/research/autoaugment)
- **Ray** : [code](https://github.com/ray-project/ray)
- **HyperOpt** : [code](https://github.com/hyperopt/hyperopt)
- **Fast AutoAugment** : [code](https://github.com/kakaobrain/fast-autoaugment)
