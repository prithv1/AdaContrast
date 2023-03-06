PORT=10001
MEMO="target"

# Low LR
# Vanilla ViT Source + Vanilla Adapt (Debug)
python main_adacontrast.py \
seed=2020 port=10000 memo="vanilla_vit_source_adapt_debug" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="vit_b_16" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/vanilla_vitb16_sup_source" \
optim.lr=2e-4

# Vanilla ViT-B/16 Supervised Source + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="vanilla_vitb16_sup_source_adapt" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="vit_b_16" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/vanilla_vitb16_sup_source" \
optim.lr=2e-4

# PASTA (Strong) ViT-B/16 Supervised Source + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_vitb16_sup_source_adapt" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="vit_b_16" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_vitb16_sup_source" \
optim.lr=2e-4

# PASTA (Strong) ViT-B/16 Supervised V-MMCE (w=10) Source + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_vitb16_sup_source_adapt" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="vit_b_16" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_vitb16_sup_source" \
optim.lr=2e-4

# PASTA (Strong) ViT-B/16 Supervised V-MMCE (w=10) Source + Self-training & IM-Loss Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_vitb16_sup_source_adapt_1.0_entmin_gent" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="vit_b_16" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_vitb16_sup_source" \
optim.lr=2e-4 learn.ent_min=true learn.ent_coeff=1.0 learn.eta=1.0 learn.beta=0.0

# Vanilla ViT-B/16 DINO Source + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="vanilla_vitb16_dino_source_adapt" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="dino_vitb16" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/vanilla_vitb16_dino_source" \
optim.lr=2e-4

# PASTA (Strong) ViT-B/16 DINO Source + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_vitb16_dino_source_adapt" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="dino_vitb16" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_vitb16_dino_source" \
optim.lr=2e-4

# PASTA (Strong) ViT-B/16 DINO V-MMCE (w=10) Source + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_vitb16_dino_source_adapt" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="dino_vitb16" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_vitb16_dino_source" \
optim.lr=2e-4

# Self-distillation for ViTs on train data
# PASTA (Strong) ViT-B/16 Supervised V-MMCE (w=10) Source + Vanilla Self-distillation
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_vitb16_sup_source_self_distill_vanilla" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="vit_b_16" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_vitb16_sup_source" \
optim.lr=2e-4 learn.use_source_distill=true learn.epochs=5


# Adapt Weak-Strong Consistency Checkpoints

# PASTA (Strong) ViT-B/16 Supervised Source + 0.1 WS + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_no_v_mmce_vitb16_sup_0.1_ws_consis_source_adapt" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="vit_b_16" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_no_v_mmce_vitb16_sup_0.1_ws_consis_source" \
optim.lr=2e-4

# PASTA (Strong; p=0.5) ViT-B/16 Supervised Source + 0.1 WS + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_no_v_mmce_vitb16_sup_0.1_ws_consis_source_adapt" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="vit_b_16" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_no_v_mmce_vitb16_sup_0.1_ws_consis_source" \
optim.lr=2e-4

# PASTA (Strong) ViT-B/16 Supervised Source + 0.1 SW + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_no_v_mmce_vitb16_sup_0.1_sw_consis_source_adapt" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="vit_b_16" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_no_v_mmce_vitb16_sup_0.1_sw_consis_source" \
optim.lr=2e-4

python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_no_v_mmce_vitb16_sup_0.1_sw_consis_source_adapt_src_ep1" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="vit_b_16" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_no_v_mmce_vitb16_sup_0.1_sw_consis_source" \
optim.lr=2e-4 learn.src_ep=1

# PASTA (Strong) ViT-B/16 Supervised Source + 0.1 WS + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10001 memo="pasta_a10k1b05_no_v_mmce_vitb16_sup_0.1_ws_consis_source_adapt_src_ep1" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="vit_b_16" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_no_v_mmce_vitb16_sup_0.1_ws_consis_source" \
optim.lr=2e-4 learn.src_ep=1

python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_no_v_mmce_vitb16_sup_0.1_ws_consis_source_adapt_src_ep1_focal_g1" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="vit_b_16" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_no_v_mmce_vitb16_sup_0.1_ws_consis_source" \
optim.lr=2e-4 learn.src_ep=1 learn.use_focal=true learn.focal_gamma=1.0
