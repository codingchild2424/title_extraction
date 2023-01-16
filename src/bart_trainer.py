
import transformers
import datetime
from pathlib import Path
import torch

class Trainer():

    def __init__(
        self,
        config
    ):
        self.config = config

    def _training_args(
        self,
        config
    ): 
        nowtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path(self.config.ckpt, nowtime)
        logging_dir = Path(self.config.logs, nowtime, "run")

        training_args = transformers.Seq2SeqTrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            per_device_train_batch_size=config.batch_size_per_device,
            per_device_eval_batch_size=config.batch_size_per_device,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.lr,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            num_train_epochs=config.n_epochs,
            logging_dir=logging_dir,
            logging_strategy="steps",
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            dataloader_num_workers=0, # 원래 4
            disable_tqdm=False,
            load_best_model_at_end=True,
        )

        return training_args


    def _train(
        self,
        model,
        data_collator,
        train_dataset,
        valid_dataset,
        config,
        tokenizer
    ):

        training_args = self._training_args(self.config)

        ## Define trainer.
        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
        )

        print("KoBART Fine Tuning Start")

        trainer.train()

        print("KoBART Fine Tuning Done!!!")

        # Save the best model
        torch.save({
            "model": trainer.model.state_dict(),
            "config": config,
            "tokenizer": tokenizer
        }, Path(config.model_fpath))

        

