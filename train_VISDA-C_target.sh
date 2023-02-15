SRC_MODEL_DIR=$1

PORT=10001
MEMO="target"

for SEED in 2020 2021 2022
do
    python main_adacontrast.py \
    seed=${SEED} port=${PORT} memo=${MEMO} project="VISDA-C" \
    data.data_root="${PWD}/datasets" data.workers=8 \
    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
    model_src.arch="resnet101" \
    model_tta.src_log_dir=${SRC_MODEL_DIR} \
    optim.lr=2e-4
done

# Vanilla Source + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="vanilla_source_adapt_v2" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/vanilla_source" \
optim.lr=2e-4

# PASTA (Weak) Source + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a3k2b025_source_adapt" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a3k2b025_source" \
optim.lr=2e-4

# PASTA (Strong) Source + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_source_adapt" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_source" \
optim.lr=2e-4

# PASTA (Weak) Source + PASTA (Weak) Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a3k2b025_source_adapt_pasta_a3k2b025" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=3.0 data.pasta_k=2.0 data.pasta_b=0.25 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a3k2b025_source" \
optim.lr=2e-4

# PASTA (Weak) Source + PASTA (Strong) Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a3k2b025_source_adapt_pasta_a10k1b05" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a3k2b025_source" \
optim.lr=2e-4

# PASTA (Weak) Source + PASTA (Strong) Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_source_adapt_pasta_a10k1b05_mocov2" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta_mocov2" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_source" \
optim.lr=2e-4

python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a3k2b025_source_adapt_pasta_a3k2b025_mocov2" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta_mocov2" data.pasta_a=3.0 data.pasta_k=2.0 data.pasta_b=0.25 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a3k2b025_source" \
optim.lr=2e-4

# PASTA (Strong) [Focal-g2] Source + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_focal_g2_source_adapt" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_focal_g2_source" \
optim.lr=2e-4

# PASTA (Strong) [Focal-g1] Source + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_focal_g1_source_adapt" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_focal_g1_source" \
optim.lr=2e-4


# PASTA (Strong) [MMCE-Vanilla] Source + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_v_mmce_w10_source_adapt" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_v_mmce_w10_source" \
optim.lr=2e-4

# PASTA (Strong) [MMCE-Vanilla] Source + Vanilla Adapt (LowLR)
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_v_mmce_w10_source_adapt_lowLR" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_v_mmce_w10_source" \
optim.lr=2e-5

# PASTA (Strong) [MMCE-Vanilla] Source + Vanilla Adapt (HighLR)
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_v_mmce_w10_source_adapt_highLR" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_v_mmce_w10_source" \
optim.lr=2e-3

# PASTA (Strong) [MMCE-Vanilla] Source + Vanilla Adapt (Self-training)
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_v_mmce_w10_source_adapt_only_st" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_v_mmce_w10_source" \
optim.lr=2e-4 learn.beta=0.0 learn.eta=0.0

# PASTA (Strong) [MMCE-Vanilla] Source + Vanilla Adapt (Contrastive)
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_v_mmce_w10_source_adapt_only_ins" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_v_mmce_w10_source" \
optim.lr=2e-4 learn.alpha=0.0 learn.eta=0.0

# PASTA (Strong) [MMCE-Vanilla] Source + Vanilla Adapt (Self-training + Contrastive)
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_v_mmce_w10_source_adapt_only_st_ins" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_v_mmce_w10_source" \
optim.lr=2e-4 learn.eta=0.0

# PASTA (Strong) [MMCE-Vanilla] Source + Vanilla Adapt + PASTA_MOCOv2_v2
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_v_mmce_w10_source_adapt_s_pasta_mocov2_v2" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta_mocov2_v2" \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_v_mmce_w10_source" \
optim.lr=2e-4

# PASTA (Strong) [MMCE-Vanilla] Source + Vanilla Adapt + PASTA_MOCOv2_v2  (High LR)
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_v_mmce_w10_source_adapt_s_pasta_mocov2_v2_highLR" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta_mocov2_v2" \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_v_mmce_w10_source" \
optim.lr=2e-3

# PASTA (Strong p=0.5) [MMCE-Vanilla] Source + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source_adapt_v2" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_source" \
optim.lr=2e-4

# PASTA (Strong p=0.5) [MMCE-Vanilla] Source + Vanilla Adapt + No-refine
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source_adapt_v2_no_refine" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_source" \
optim.lr=2e-4 learn.refine_method=null

# PASTA (Strong p=0.5) [MMCE-Vanilla] Source + Self-training
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source_adapt_v2_only_st" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_source" \
optim.lr=2e-4 learn.beta=0.0 learn.eta=0.0

# PASTA (Strong p=0.5) [MMCE-Vanilla] Source + Self-training + No-refine
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source_adapt_v2_only_st_no_refine" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_source" \
optim.lr=2e-4 learn.beta=0.0 learn.eta=0.0 learn.refine_method=null

# PASTA (Strong p=0.5) [MMCE-Vanilla] Source + Self-training + Diversity
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source_adapt_v2_only_st_div" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_source" \
optim.lr=2e-4 learn.beta=0.0

# PASTA (Strong p=0.5) Source + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_source_adapt_v2" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_source" \
optim.lr=2e-4

# PASTA (Strong p=0.5) [MMCE-Vanilla] Source + DropNL + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_dropnl_source_adapt_v2" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_dropnl_source" \
optim.lr=2e-4 model_src.drop_nl=1

# PASTA (Strong p=0.5) Source + Vanilla Adapt
# Debug Pseudo-Label Selection
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_source_adapt_pl_thresh0.7" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_source" \
optim.lr=2e-4 learn.use_pl_thresh=true learn.pl_thresh=0.7

# PASTA (Strong p=0.5) Source + Vanilla Adapt
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source_adapt_pl_thresh0.5_only_st_highLR" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_source" \
optim.lr=2e-3 learn.use_pl_thresh=true learn.pl_thresh=0.5 learn.beta=0.0 learn.eta=0.0

python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source_adapt_pl_thresh0.7_ent_thresh1.24" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_source" \
optim.lr=2e-4 learn.use_pl_thresh=true learn.pl_thresh=0.9

# High-confidence Self-training + Entropy-Minimization + Diversity Loss
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source_adapt_pl_thresh0.5_0.1_entmin_gent" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_source" \
optim.lr=2e-4 learn.use_pl_thresh=true learn.pl_thresh=0.5 learn.ent_min=true learn.ent_coeff=0.1 learn.eta=0.1 learn.beta=0.0

python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source_adapt_pl_thresh0.5_1.0_entmin_gent" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_source" \
optim.lr=2e-4 learn.use_pl_thresh=true learn.pl_thresh=0.5 learn.ent_min=true learn.ent_coeff=1.0 learn.eta=1.0 learn.beta=0.0

python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source_adapt_pl_thresh0.8_1.0_entmin_gent" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_source" \
optim.lr=2e-4 learn.use_pl_thresh=true learn.pl_thresh=0.8 learn.ent_min=true learn.ent_coeff=1.0 learn.eta=1.0 learn.beta=0.0

python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source_adapt_pl_thresh0.5_2.0_entmin_gent" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_source" \
optim.lr=2e-4 learn.use_pl_thresh=true learn.pl_thresh=0.5 learn.ent_min=true learn.ent_coeff=2.0 learn.eta=2.0 learn.beta=0.0

python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_adapt_pl_thresh0.5_0.1_entmin_gent" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_source" \
optim.lr=2e-4 learn.use_pl_thresh=true learn.pl_thresh=0.5 learn.ent_min=true learn.ent_coeff=0.1 learn.eta=0.1 learn.beta=0.0

python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_adapt_pl_thresh0.5_1.0_entmin_gent" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_source" \
optim.lr=2e-4 learn.use_pl_thresh=true learn.pl_thresh=0.5 learn.ent_min=true learn.ent_coeff=1.0 learn.eta=1.0 learn.beta=0.0