# rm data/facial
# ln -s /mnt/ramdisk2/target_noleftright data/facial
# python tools/train.py --cfg configs/facial/class_ablation/0005-24-adamw-noleftright.yaml
# wait
# rm data/facial
# ln -s /mnt/ramdisk2/target_noneck data/facial
# python tools/train.py --cfg configs/facial/class_ablation/0005-24-adamw-noneck.yaml
# wait
# rm data/facial
# wait
# ln -s /mnt/ramdisk/target_nonostrils data/facial
# wait
# python tools/train.py --cfg configs/facial/hyper/005-6-adamw.yaml
# wait
# python tools/train.py --cfg configs/facial/hyper/005-30-adamw.yaml

rm data/facial
ln -s /mnt/ramdisk2/target_nonostrils data/facial
wait

python tools/train.py --cfg configs/facial/hyper/0005-24-adamw.yaml
wait
python tools/train.py --cfg configs/facial/syn_vs_real/0005-24-adamw-syn.yaml
wait
python tools/train.py --cfg configs/facial/syn_vs_real/0005-24-adamw-real.yaml