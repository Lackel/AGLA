# export HF_ENDPOINT=https://hf-mirror.com
dataset_name='coco' # 'aokvqa' 'gqa'
image_folder='/workspace/data/val2014' # '/workspace/data/gqa'
for seed in 1 2 3
do
for type in 'random' 'popular' 'adversarial'
do
python run_instructblip.py \
--question-file ../data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ../output/blip_${dataset_name}_pope_${type}_answers_alga_seed${seed}.jsonl \
--use_agla \
--alpha 2 \
--beta 0.5 \
--seed $seed
done
done