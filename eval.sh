

# python tools/eval.py --cfg configs/evaluate/hyper/005-6-adamw.yaml \
#                           TEST.MODEL_FILE pretrained_models/mymodels/hyper/005-6/best.pt
# wait                          

# python tools/eval.py --cfg configs/evaluate/hyper/005-12-adamw.yaml \
#                           TEST.MODEL_FILE pretrained_models/mymodels/hyper/005-12/best.pt
# wait

# python tools/eval.py --cfg configs/evaluate/hyper/005-24-adamw.yaml \
#                           TEST.MODEL_FILE pretrained_models/mymodels/hyper/005-24/best.pt
# wait

# python tools/eval.py --cfg configs/evaluate/hyper/005-30-adamw.yaml \
#                           TEST.MODEL_FILE pretrained_models/mymodels/hyper/005-30/best.pt
# wait

# python tools/eval.py --cfg configs/evaluate/hyper/001-24-adamw.yaml \
#                           TEST.MODEL_FILE pretrained_models/mymodels/hyper/001-24/best.pt
# wait

# python tools/eval.py --cfg configs/evaluate/hyper/0005-24-adamw.yaml \
#                           TEST.MODEL_FILE pretrained_models/mymodels/hyper/0005-24/best.pt
# wait

# python tools/eval.py --cfg configs/evaluate/hyper/00025-24-adamw.yaml \
#                           TEST.MODEL_FILE pretrained_models/mymodels/hyper/00025-24/best.pt

# python tools/eval.py --cfg configs/evaluate/hyper/0005-24-adamw.yaml \
#                           TEST.MODEL_FILE output/facial/0005-24-adamw/best.pt
# wait
python tools/eval.py --cfg configs/evaluate/syn_vs_real/0005-24-adamw-real.yaml \
                          TEST.MODEL_FILE output/facial/0005-24-adamw-real/best.pt
wait
python tools/eval.py --cfg configs/evaluate/syn_vs_real/0005-24-adamw-syn.yaml \
                          TEST.MODEL_FILE output/facial/0005-24-adamw-syn/best.pt
wait
                          