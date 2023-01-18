

python bart_finetuning.py \
--model_fpath model_records/kobart-model2.pth \
--batch_size_per_device 4 \
--gradient_accumulation_steps 32 \
--n_epochs 5