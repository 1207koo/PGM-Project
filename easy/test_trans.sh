python main.py --device "cuda:0" --dataset miniimagenet --model ResNet12_nd --sampler-model Sampler3_nd --discriminator-model DiscriminatorDouble4 --rotations --cosine --save-features test/$1.pt --epochs 0 --load-model $1.pt1 --load-sampler-model $1_sampler.pt1 --load-discriminator-model $1_discriminator.pt1 --transductive
python main.py --device "cuda:0" --dataset miniimagenet --test-features test/$1.pt1 --transductive