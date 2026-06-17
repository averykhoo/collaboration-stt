#!/bin/bash

in_asr_model=/mount/data/alx/models/s0001
databases=("fleurs" "mesolitica" "rerecorded")
stage=0

# Copy asr model
rm -fr asr_model
mkdir asr_model
cp ${in_asr_model}/* asr_model

run_inference() {
    local db=$1
    local path_dir=input/paths_${db}.txt
    local gt=input/gt_${db}.txt

    # Stage 1: Preparation (Sequential per DB)
    if [ $stage -le 1 ]; then
        echo "Stage 1 for ${db}"
        cp ../Stage_3_Test_Data_and_Baseline_Model/inter/${db}_norm.txt input/gt_${db}.txt
        cp ../Stage_3_Test_Data_and_Baseline_Model/inter/${db}.wav input/${db}.wav
        python python/create_paths.py input/${db}.wav input/paths_${db}.txt
    fi

    # Stage 2: Predict
    if [ $stage -le 2 ]; then
        echo "Stage 2 for ${db}"
        rm -rf inter/${db}/predict
        mkdir -p inter/${db}/predict
        python predict.py --wav_pathes ${path_dir} \
                        --out_predict inter/${db}/predict/predict_${db}.txt
    fi

    # Stage 3: Get WER
    if [ $stage -le 3 ]; then
        echo "Stage 3 for ${db}"
        rm -rf inter/${db}/stat
        mkdir -p inter/${db}/stat    
        python wer.py input/gt_${db}.txt inter/${db}/predict/predict_${db}.txt \
                                        allowed_replacements.txt \
                                        allowed_insertions.txt \
                                        allowed_deletions.txt \
                                        inter/${db}/stat/stat_${db}.txt
    fi
}

# --- Main Execution ---

# Clean and setup input directory once
if [ $stage -le 0 ]; then
    rm -rf input
    mkdir -p input
fi

# Loop through databases and run in parallel
for db in "${databases[@]}"; do
    run_inference "$db"
done

wait
echo "All processes completed."