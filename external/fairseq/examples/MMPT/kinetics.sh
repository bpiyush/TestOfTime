#!/bin/bash

classes=(
    "bouncing_on_trampoline" 
    "breakdancing"
    "busking"
    "cartwheeling"
    "cleaning_shoes"
    "country_line_dancing"
    "drop_kicking"
    "gymnastics_tumbling"
    "hammer_throw"
    "high_kick"
    "jumpstyle_dancing"
    "kitesurfing"
    "parasailing"
    "playing_cards"
    "playing_cymbals"
    "playing_drums"
    "playing_ice_hockey"
    "robot_dancing"
    "shining_shoes"
    "shuffling_cards"
    "side_kick"
    "ski_jumping"
    "skiing_not_slalom_or_crosscountry"
    "skiing_crosscountry"
    "skiing_slalom"
    "snowboarding"
    "somersaulting"
    "tap_dancing"
    "throwing_ball"
    "throwing_discus"
    "vault"
    "wrestling"
)
data_root=/var/scratch/pbagad/datasets/kinetics/VideoData
feat_root=/var/scratch/pbagad/datasets/kinetics
for class in "${classes[@]}"; do
    echo "Extracting features for class $class"
    echo "Number of videos for class $class: $(ls -1 $data_root/$class | wc -l)"
    echo "----------------------------------------"
    python scripts/video_feature_extractor/extract.py \
        --vdir $data_root/${class} \
        --fdir $feat_root/feat/feat_how2_s3d/${class}/ \
        --type=s3d --num_decoding_thread=4 \
        --batch_size 32 --half_precision 1
done
