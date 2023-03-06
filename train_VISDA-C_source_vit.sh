PORT=10000
MEMO="source"

# Vanilla
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="vit_source_debug" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="vit_b_16" \
optim.lr=2e-4

# Vanilla ViT-B/16 Supervised
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="vanilla_vitb16_sup_source_v2" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="vit_b_16" \
optim.lr=2e-4

# PASTA (Weak) ViT-B/16 Supervised
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a3k2b025_vitb16_sup_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=3.0 data.pasta_k=2.0 data.pasta_b=0.25 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="vit_b_16" \
optim.lr=2e-4

# PASTA (Strong) ViT-B/16 Supervised
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_vitb16_sup_source_v2" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="vit_b_16" \
optim.lr=2e-4

# PASTA (Strong; p=0.5) + MMCE ViT-B/16 Supervised
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_vitb16_sup_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 data.pasta_prob=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 learn.loss_fn="smoothed_ce_include_mmce" learn.mmce_wt=10.0 \
model_src.arch="vit_b_16" \
optim.lr=2e-4

# Vanilla ViT-B/16 DINO
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="vanilla_vitb16_dino_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="dino_vitb16" \
optim.lr=2e-4

# PASTA (Weak) ViT-B/16 DINO
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a3k2b025_vitb16_dino_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=3.0 data.pasta_k=2.0 data.pasta_b=0.25 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="dino_vitb16" \
optim.lr=2e-4

# PASTA (Strong) ViT-B/16 DINO
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_vitb16_dino_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="dino_vitb16" \
optim.lr=2e-4

# PASTA (Strong; p=0.5) + MMCE ViT-B/16 DINO
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_vitb16_dino_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 data.pasta_prob=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 learn.loss_fn="smoothed_ce_include_mmce" learn.mmce_wt=10.0 \
model_src.arch="dino_vitb16" \
optim.lr=2e-4


# PASTA (Strong; p=0.5) + MMCE ViT-B/16 DINO + AdamW
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_vitb16_dino_adamw_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 data.pasta_prob=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 learn.loss_fn="smoothed_ce_include_mmce" learn.mmce_wt=10.0 \
model_src.arch="dino_vitb16" \
optim.lr=2e-4 optim.name="adamw"

# PASTA (Strong; p=0.5) + Ent-Reg DINO + 20 Epochs
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_entc_0.1_vitb16_dino_source_20ep" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 data.pasta_prob=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=20 \
model_src.arch="dino_vitb16" \
optim.lr=2e-4 learn.include_ent_reg=true learn.ent_coeff=0.1

# PASTA (Strong; p=0.5) + Ent-Reg DINO + High-LR
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_vitb16_dino_lr2e-3_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 data.pasta_prob=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 learn.loss_fn="smoothed_ce_include_mmce" learn.mmce_wt=10.0 \
model_src.arch="dino_vitb16" \
optim.lr=2e-4

# PASTA (Strong; p=0.5) + Weak-Strong Consistency ViT-B/16 Sup
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_no_v_mmce_vitb16_sup_0.1_sw_consis_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="vit_b_16" \
optim.lr=2e-4 learn.weak_strong_consistency=true learn.consis_coeff=0.1

# PASTA (Strong) + Weak-Strong Consistency  ViT-B/16 Sup
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_no_v_mmce_vitb16_sup_0.1_sw_consis_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="vit_b_16" \
optim.lr=2e-4 learn.weak_strong_consistency=true learn.consis_coeff=0.1

# PASTA + MoCo (Strong; p=0.5) + Weak-Strong Consistency  ViT-B/16 Sup HighLR
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_mocov2_v2_no_v_mmce_vitb16_sup_0.1_ws_consis_lr2e-3_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta_mocov2_v2" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 data.pasta_prob=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="vit_b_16" \
optim.lr=2e-3 learn.weak_strong_consistency=true learn.consis_coeff=0.1

# PASTA + MoCo (Strong; p=0.5) + Weak-Strong Consistency  ViT-B/16 Sup
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_mocov2_v2_no_v_mmce_vitb16_sup_0.1_ws_consis_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta_mocov2_v2" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 data.pasta_prob=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="vit_b_16" \
optim.lr=2e-4 learn.weak_strong_consistency=true learn.consis_coeff=0.1

# Masking (0.7) + Weak-Strong Consistency  ViT-B/16 Sup
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="mask_v1_p64r0.7_no_v_mmce_vitb16_sup_0.1_ws_consis_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="mask_v1" \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="vit_b_16" \
optim.lr=2e-3 learn.weak_strong_consistency=true learn.consis_coeff=0.1


# PASTA (Strong; p=0.5) + Weak-Strong Consistency ViT-B/16 DINO
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_no_v_mmce_vitb16_dino_0.1_ws_consis_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="dino_vitb16" \
optim.lr=2e-4 learn.weak_strong_consistency=true learn.consis_coeff=0.1

# PASTA (Strong) + Weak-Strong Consistency  ViT-B/16 DINO
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_adacontrast.py train_source=true learn=source \
seed=2020 port=10000 memo="pasta_a10k1b05_no_v_mmce_vitb16_dino_0.1_ws_consis_source" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
learn.epochs=10 \
model_src.arch="dino_vitb16" \
optim.lr=2e-4 learn.weak_strong_consistency=true learn.consis_coeff=0.1
