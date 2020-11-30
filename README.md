##### Disclaimer
Right now, the repo is being rewritten from `firelab` to `pytorch-lightning`, so you may like to wait for the updates before using it.

### INR-based GAN

Generating images in their implicit form.


To run the reconstruction model, you should first install `firelab` library:
```
pip install firelab
```

To run the experiment on a single GPU, use the following command:
```
CUDA_VISIBLE_DEVICES=0 python src/run.py -c configs/inr-gan.yml --config.dataset lsun_256
```

To run a multi-gpu training, we use [horovod](https://github.com/horovod/horovod) which is launched via:
```
horovodrun -np NUM_GPUS --mpi-args=--oversubscribe python src/run.py -c configs/inr-gan.yml --config.distributed_training.enabled true --config.dataset lsun_256 --config.hp.batch_size BATCH_SIZE
```
