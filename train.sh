sleep 1s
CUDA_VISIBLE_DEVICES=0 \
python main.py train \
--config-path configs/Gan.yaml \
--is-validation
