python main.py --wandb "1207koo" --device "cuda:0123" --dataset miniimagenet --model ResNet12_nd --sampler-model Sampler2_nd --discriminator-model DiscriminatorDouble3 --rotations --cosine --save-model "save_nd/mini7.pt" --lr 0.01 --gamma 0.5 --epochs 480 --lambda-g 0.1 --lambda-d 0.1