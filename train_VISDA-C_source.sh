PORT=10000
MEMO="source"

for SEED in 2020 2021 2022
do
    python main_adacontrast.py train_source=true learn=source \
    seed=${SEED} port=${PORT} memo=${MEMO} project="VISDA-C" \
    data.data_root="${PWD}/datasets" data.workers=8 \
    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
    learn.epochs=10 \
    model_src.arch="resnet101" \
    optim.lr=2e-4 
done

# Vanilla
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="source_debug" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=3.0 data.pasta_k=2.0 data.pasta_b=0.25 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="resnet101" \
optim.lr=2e-4

# Vanilla
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="vanilla_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="resnet101" \
optim.lr=2e-4

# PASTA (Weak)
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a3k2b025_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=3.0 data.pasta_k=2.0 data.pasta_b=0.25 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="resnet101" \
optim.lr=2e-4

# PASTA (Strong)
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="resnet101" \
optim.lr=2e-4

# PASTA (Strong) + Focal
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_focal_g2_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 learn.loss_fn="focal" learn.gamma=2 \
model_src.arch="resnet101" \
optim.lr=2e-4

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_focal_g0.5_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 learn.loss_fn="focal" learn.gamma=0.5 \
model_src.arch="resnet101" \
optim.lr=2e-4

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_focal_g1_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 learn.loss_fn="focal" learn.gamma=1 \
model_src.arch="resnet101" \
optim.lr=2e-4

# PASTA (Strong) + DropNL
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_dropnl_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=20 \
model_src.arch="resnet101" \
optim.lr=2e-4 model_src.drop_nl=1

# PASTA (Strong) + High-LR
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_source_highLR_2e-3" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=20 \
model_src.arch="resnet101" \
optim.lr=2e-3


# PASTA (Strong) + 10*(Vanilla)MMCE
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_v_mmce_w10_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 learn.loss_fn="smoothed_ce_include_mmce" learn.mmce_wt=10.0 \
model_src.arch="resnet101" \
optim.lr=2e-4

# PASTA (Strong) + 10*(Weighted)MMCE
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_w_mmce_w10_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 learn.loss_fn="smoothed_ce_include_mmce" learn.mmce_wt=10.0 learn.mmce_mode="weighted" \
model_src.arch="resnet101" \
optim.lr=2e-4

# PASTA (Strong) + 10*(Weighted)MMCE + DropNL
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_w_mmce_w10_dropnl_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 learn.loss_fn="smoothed_ce_include_mmce" learn.mmce_wt=10.0 learn.mmce_mode="weighted" \
model_src.arch="resnet101" \
optim.lr=2e-4 model_src.drop_nl=1

# PASTA (Strong) + 10*(Vanilla)MMCE + PASTA Prob=0.5
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 data.pasta_prob=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 learn.loss_fn="smoothed_ce_include_mmce" learn.mmce_wt=10.0 \
model_src.arch="resnet101" \
optim.lr=2e-4

# PASTA (Strong) + 10*(Vanilla)MMCE + PASTA Prob=0.5 + DropNL
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_dropnl_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 data.pasta_prob=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 learn.loss_fn="smoothed_ce_include_mmce" learn.mmce_wt=10.0 \
model_src.arch="resnet101" \
optim.lr=2e-4 model_src.drop_nl=1

# PASTA (Strong) + PASTA Prob=0.5
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 data.pasta_prob=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="resnet101" \
optim.lr=2e-4

# PASTA (Strong) + 10*(Vanilla)MMCE + PASTA Prob=0.5 + High-WD
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_wd_1e-2_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 data.pasta_prob=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 learn.loss_fn="smoothed_ce_include_mmce" learn.mmce_wt=10.0 \
model_src.arch="resnet101" \
optim.lr=2e-4 optim.weight_decay=1e-2