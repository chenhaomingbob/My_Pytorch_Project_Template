# My Pytorch Project Template

# Getting started

## Single Machine

```shell
python main.py
```

## Multiple Machines

Node 0:

```shell
python main.py --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --world-size 2 --node 0 
```

Node 1:

```shell
python main.py --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --world-size 2 --node 1
```

# TODO items
- [ ] DDP
- [ ] Model ema
- [ ] Gradient update frequency
- [ ] Check lr/wd scheduler
# Acknowledgements

Great thanks for these open-source codes:

- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) (DDP)
- [pytorch-examples-imagenet](https://github.com/pytorch/examples/tree/main/imagenet) (DDP)
- [mmpose](https://github.com/open-mmlab/mmpose)
- [timm](https://github.com/rwightman/pytorch-image-models)
