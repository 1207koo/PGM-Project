python main.py --device "cuda:0" --dataset miniimagenet --model ResNet12_nd --sampler-model Sampler3_nd --discriminator-model DiscriminatorDouble4 --rotations --cosine --save-features test/$1.pt --epochs 0 --load-model save/$1.pt1 --load-sampler-model save/$1_sampler.pt1 --load-discriminator-model save/$1_discriminator.pt1
python main.py --device "cuda:0" --dataset miniimagenet --test-features test/$1.pt1