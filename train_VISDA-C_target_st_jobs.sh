SRC_MODEL_DIR=$1

PORT=10001
MEMO="target"

# High-confidence Self-training + Entropy-Minimization + Diversity Loss
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source_adapt_pl_thresh0.5_0.1_entmin_gent" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_source" \
optim.lr=2e-4 learn.use_pl_thresh=true learn.pl_thresh=0.5 learn.ent_min=true learn.ent_coeff=0.1 learn.eta=0.1 learn.beta=0.0

# High-confidence Self-training (0.5) + Ent-Min + Div-Max + WW-Consistency PL
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source_adapt_WW_thresh0.5_pl_2.0IM" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_source" \
optim.lr=2e-4 \
learn.use_pl_thresh=true learn.pl_thresh=0.5 learn.ent_min=true learn.ent_coeff=2.0 learn.eta=2.0 learn.beta=0.0 learn.ce_sup_type="weak_weak"

# High-confidence Self-training (0.5) + Ent-Min + Div-Max + WS-PASTA-Consistency PL
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source_adapt_WS_pasta_a10k1b05_thresh0.5_pl_2.0IM" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 data.aug_type="pasta_mocov2" data.pasta_a=10.0 data.pasta_k=1.0 data.pasta_b=0.5 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_source" \
optim.lr=2e-4 \
learn.use_pl_thresh=true learn.pl_thresh=0.5 learn.ent_min=true learn.ent_coeff=2.0 learn.eta=2.0 learn.beta=0.0

# High-confidence Self-training + Entropy-Minimization + Diversity Loss + BN-Only
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source_adapt_WS_thresh0.5_pl_1.0IM_BN_only" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_source" \
optim.lr=2e-4 learn.use_pl_thresh=true learn.pl_thresh=0.5 learn.ent_min=true learn.ent_coeff=1.0 learn.eta=1.0 learn.beta=0.0 learn.bn_only=true

# High-confidence Self-training + Entropy-Minimization + Diversity Loss + BN-Only (High LR)
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source_adapt_WS_thresh0.5_pl_1.0IM_BN_only_highLR" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_source" \
optim.lr=2e-3 learn.use_pl_thresh=true learn.pl_thresh=0.5 learn.ent_min=true learn.ent_coeff=1.0 learn.eta=1.0 learn.beta=0.0 learn.bn_only=true

# High-confidence Self-training + Entropy-Minimization + Diversity Loss + BN-Only (High LR)
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source_adapt_WS_thresh0.5_pl_2.0IM_BN_only_highLR" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_source" \
optim.lr=2e-3 learn.use_pl_thresh=true learn.pl_thresh=0.5 learn.ent_min=true learn.ent_coeff=2.0 learn.eta=2.0 learn.beta=0.0 learn.bn_only=true

# High-confidence Self-training + Entropy-Minimization + BN-Only (High LR)
python main_adacontrast.py \
seed=2020 port=10000 memo="pasta_a10k1b05_p0.5_v_mmce_w10_source_adapt_WS_thresh0.5_pl_2.0Ent_BN_only_highLR" project="VISDA-C" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
model_src.arch="resnet101" \
model_tta.src_log_dir="/coc/scratch/prithvi/dg_for_da/recognition_sfda/adacontrast/VISDA-C/pasta_a10k1b05_p0.5_v_mmce_w10_source" \
optim.lr=2e-3 learn.use_pl_thresh=true learn.pl_thresh=0.5 learn.ent_min=true learn.ent_coeff=2.0 learn.eta=0.0 learn.beta=0.0 learn.bn_only=true