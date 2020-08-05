# cityscapes-unet
Pytorch implementation of a U-Net-inspired neural network, for performing semantic segmentation on the Cityscapes dataset.

I'm actively training this model/playing around with hyperparameters, and will update this with a better description/summary once I'm satisfied with the results.

Despite the underwhelming README doc, this repository is fully functional as is! Just make sure you've downloaded the Cityscapes dataset to the directory /data/Cityscapes/. After only 2 epochs is able to achieve > 70% pixel accuracy across 34 categories. I'm training the model on an AWS EC2 server (P series). It takes roughly 40 minutes per epoch when using 1 GPU, and 6 minutes per epoch on 8 GPUs in parallel.
