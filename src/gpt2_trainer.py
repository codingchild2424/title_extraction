
import transformers
import datetime
from pathlib import Path
import torch
from tqdm import tqdm

class Trainer():

    def __init__(
        self,
        model,
        optimizer,
        n_epochs,
        device,
        crit,
        max_seq_len,
        config
        ):
        self.model=model
        self.optimizer=optimizer
        self.n_epochs=n_epochs
        self.device=device
        self.crit=crit
        self.max_seq_len=max_seq_len
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


    def old_train(
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

        print("KoGPT2 Fine Tuning Start")

        trainer.train()

        print("KoGPT2 Fine Tuning Done!!!")

        # Save the best model
        torch.save({
            "model": trainer.model.state_dict(),
            "config": config,
            "tokenizer": tokenizer
        }, Path(config.model_fpath))

    '''
    Custom Trainer
    '''

    def _train(self, train_loader):

        loss_list = []

        for idx, data in enumerate(tqdm(train_loader)):
            self.model.train()

            input_ids = data['input_ids']
            attention_mask = data['attention_mask']
            labels = data['labels']

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            # |input_ids| = (bs, sq)
            # |labels| = (bs, sq)

            # print("input_ids", input_ids.size())
            # print("labels", labels.size())

            outputs = self.model(
                input_ids,
                labels=labels
            )#.to(self.device)

            loss, logits = outputs[:2]

            loss.backward()
            if (idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            loss_list.append(loss)

        return loss_list
        

    def _valid(self, valid_loader):

        loss_list = []

        with torch.no_grad():
            for idx, data in enumerate(tqdm(valid_loader)):
                self.model.eval()

                input_ids = data['input_ids']
                attention_mask = data['attention_mask']
                labels = data['labels']

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                # |input_ids| = (bs, sq)
                # |labels| = (bs, sq)

                # print("input_ids", input_ids.size())
                # print("labels", labels.size())

                outputs = self.model(
                    input_ids,
                    labels=labels,
                    attention_mask=attention_mask
                )#.to(self.device)

                loss, logits = outputs[:2]

                loss_list.append(loss)

        return loss_list



    def train(self, train_loader, valid_loader, config):

        train_loss_list = []
        valid_loss_list = []
        best_valid_loss = float('inf')

        for epoch_index in range(self.n_epochs):

            print("Epoch(%d/%d) start" % (
                epoch_index + 1,
                self.n_epochs
            ))

            # Training Session
            train_loss = self._train(train_loader)
            valid_loss = self._validate(valid_loader)

            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)

            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss

            print("Epoch(%d/%d) result: train_score=%.4f  valid_score=%.4f best_valid_score=%.4f" % (
                epoch_index + 1,
                self.n_epochs,
                train_loss,
                valid_loss,
                best_valid_loss,
            ))

        print("\n")
        print("The Best Valid Score in Testing Session is %.4f" % (
                best_valid_loss,
            ))
        print("\n")


