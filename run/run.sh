cuda=1
out_dir='out'
dataset='mvsa-s'            # Options: 't2015', 't2017', 'masad', 'mvsa-s', 'tumemo'
train_file='few-shot1.tsv'  # Options: 'few-shot1.tsv', 'few-shot2.tsv'
template=1                  # Options: 1, 2, 3

case $dataset in
    't2015')
        img_dir='IJCAI2019_data/twitter2015_images'
        prompt_shape_pt='333-0'
        prompt_shape_pvlm='333-3'
        early_stop=40
        ;;
    't2017')
        img_dir='IJCAI2019_data/twitter2017_images'
        prompt_shape_pt='333-0'
        prompt_shape_pvlm='333-3'
        early_stop=40
        ;;
    'masad')
        img_dir='MASAD_imgs'
        prompt_shape_pt='333-0'
        prompt_shape_pvlm='333-3'
        early_stop=25
        ;;
    'mvsa-s')
        img_dir='MVSA-S_data'
        prompt_shape_pt='33-0'
        prompt_shape_pvlm='33-3'
        early_stop=40
        ;;
    'tumemo')
        img_dir='TumEmo_data'
        prompt_shape_pt='33-0'
        prompt_shape_pvlm='33-3'
        early_stop=15
        ;;
esac

# PT
for lr in 1e-5 2e-5 3e-5 4e-5 5e-5
do
    for seed in 5 13 17
    do
        python main.py \
            --cuda $cuda \
            --out_dir $out_dir \
            --no_img \
            --dataset $dataset \
            --template $template \
            --prompt_shape $prompt_shape_pt \
            --few_shot_file $train_file \
            --batch_size 32 \
            --lr_lm_model $lr \
            --early_stop $early_stop \
            --seed $seed
    done
done

# PVLM
for img_token_len in 1 2 3 4 5
do
    for lr in 1e-5 2e-5 3e-5 4e-5 5e-5
    do
        for seed in 5 13 17
        do
            python main.py \
                --cuda $cuda \
                --out_dir $out_dir \
                --dataset $dataset \
                --img_dir $img_dir \
                --template $template \
                --prompt_shape $prompt_shape_pvlm \
                --few_shot_file $train_file \
                --img_token_len $img_token_len \
                --batch_size 32 \
                --lr_lm_model $lr \
                --lr_visual_encoder 0 \
                --early_stop $early_stop \
                --seed $seed
        done
    done
done