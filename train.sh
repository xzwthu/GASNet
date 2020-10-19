sleep 1s
CUDA_VISIBLE_DEVICES=3 \
python main_lung.py train \
--config-path configs/Gan.yaml \
--is-validation
