

python bart_finetuning.py \
--model_fpath model_records/kobart-model-poem.pth \
--train_data_path ../datasets/integrated_pre_poem_datasets/train_data.tsv \
--valid_data_path ../datasets/integrated_pre_poem_datasets/valid_data.tsv \
--test_data_path ../datasets/integrated_pre_poem_datasets/test_data.tsv \
--batch_size_per_device 4 \
--gradient_accumulation_steps 32 \
--n_epochs 5