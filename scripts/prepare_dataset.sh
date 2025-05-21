datasets=( "arc_challenge" "bbh" "bbq" "boolq" "crows_pairs" "commonsense_qa" "ethics" "glue_sst2" "hellaswag" "math_qa" "mmlu_pro" "openbookqa" "superglue_rte" "superglue_wic" "glue_mnli" "glue_qnli" "glue_cola" "deepmind" "mmlu" )

for task in "${datasets[@]}"; do
    echo "Process ${task}"
    python process_data.py --task $task
done