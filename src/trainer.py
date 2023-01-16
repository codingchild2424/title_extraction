
# import torch
# from tqdm import tqdm

# class KoBARTTrainer():
    
#     def __init__(
#         self,
#         model,
#         optimizer,
#         n_epochs,
#         device,
#         crit,
#         max_seq_len,
#         warmup_ratio,
#         grad_acc=None,
#         grad_acc_iter=None
#         ):

#         self.model = model
#         self.optimizer = optimizer
#         self.n_epochs = n_epochs
#         self.device = device
#         self.crit = crit
#         self.max_seq_len = max_seq_len
#         self.warmup_ratio = warmup_ratio

#     def _train(
#         self,
#         train_loader
#         ):

#         loss_list = []

#         for idx, data in enumerate(tqdm(train_loader)):
#             self.model.train()

#             #print("data", data)

#             '''
#             data = {
#                 'input_ids':
#                 'attention_mask':
#                 'text': 
#             }
#             '''

#             input_ids = data['input_ids'].to(self.device)
#             attention_mask = data['attention_mask'].to(self.device)
#             decoder_ids = data['decoder_ids'].to(self.device)
#             decoder_attention_mask = data['decoder_attention_mask'].to(self.device)
#             label_ids = data['label_ids'].to(self.device)

#             print("input_ids", input_ids.size())
#             print("attention_mask", attention_mask.size())
#             print("decoder_ids", decoder_ids.size())
#             print("decoder_attention_mask", decoder_attention_mask.size())
#             print("label_ids", label_ids.size())

#             print("label", label_ids)

#             y_hat = self.model(
#                 input_ids = input_ids,
#                 attention_mask = attention_mask,
#                 decoder_input_ids = decoder_ids,
#                 decoder_attention_mask = decoder_attention_mask,
#                 labels = label_ids
#             ) #.to(self.device)

#             #print("y_hat", y_hat)

#             '''
#             y_hat = {
#                 loss:
#                 logits:
#                 hidden_states:
#                 attentions:
#                 cross_attentions:
#                 past_key_values:
#             }
#             '''

#             # nll_loss
#             loss = y_hat['loss']
#             # |loss| = (1)

#             loss.backward()
#             self.optimizer.step()
#             self.optimizer.zero_grad()

#             loss_list.append(loss)

#         loss_result = torch.mean(
#             torch.Tensor(loss_list)
#         ).detach().cpu().numpy()

#         return loss_result

#     def _validate(
#         self,
#         valid_loader
#         ):

#         loss_list = []

#         with torch.no_grad():
#             for idx, data in enumerate(tqdm(valid_loader)):
#                 self.model.eval()

#                 '''
#                 data = {
#                     'input_ids':
#                     'attention_mask':
#                     'text': 
#                 }
#                 '''

#                 input_ids = data['input_ids'].to(self.device)
#                 attention_mask = data['attention_mask'].to(self.device)
#                 #decoder_ids = data['decoder_ids'].to(self.device)
#                 #decoder_attention_mask = data['decoder_attention_mask'].to(self.device)
#                 #label_ids = data['label_ids'].to(self.device)

#                 y_hat = self.model(
#                     input_ids = input_ids,
#                     attention_mask = attention_mask,
#                     #decoder_input_ids = decoder_ids,
#                     #decoder_attention_mask = decoder_attention_mask,
#                     #labels = label_ids
#                 ) #.to(self.device)

#                 print("input_ids", input_ids.size())
#                 print("attention_mask", attention_mask.size())
#                 #print("decoder_ids", decoder_ids.size())
#                 #print("decoder_attention_mask", decoder_attention_mask.size())
#                 #print("label_ids", label_ids.size())

#                 #print("label", label_ids)

#                 '''
#                 y_hat = {
#                     loss:
#                     logits:
#                     hidden_states:
#                     attentions:
#                     cross_attentions:
#                     past_key_values:
#                 }
#                 '''

#                 # nll_loss
#                 loss = y_hat['loss']
#                 # |loss| = (1)

#                 loss_list.append(loss)

#             loss_result = torch.mean(
#                 torch.Tensor(loss_list)
#             ).detach().cpu().numpy()

#             return loss_result

#     def _test(
#         self,
#         test_loader
#         ):

#         loss_list = []

#         with torch.no_grad():
#             for idx, data in enumerate(tqdm(test_loader)):
#                 self.model.eval()

#                 '''
#                 data = {
#                     'input_ids':
#                     'attention_mask':
#                     'text': 
#                 }
#                 '''

#                 input_ids = data['input_ids'].to(self.device)
#                 attention_mask = data['attention_mask'].to(self.device)
#                 #decoder_ids = data['decoder_ids'].to(self.device)
#                 #decoder_attention_mask = data['decoder_attention_mask'].to(self.device)
#                 #label_ids = data['label_ids'].to(self.device)

#                 print("input_ids", input_ids.size())
#                 print("attention_mask", attention_mask.size())
#                 #print("decoder_ids", decoder_ids.size())
#                 #print("decoder_attention_mask", decoder_attention_mask.size())
#                 #print("label_ids", label_ids.size())

#                 #print("label", label_ids)

#                 y_hat = self.model(
#                     input_ids = input_ids,
#                     attention_mask = attention_mask,
#                     #decoder_input_ids = decoder_ids,
#                     #decoder_attention_mask = decoder_attention_mask,
#                     #labels = label_ids
#                 ) #.to(self.device)
#                 '''
#                 y_hat = {
#                     loss:
#                     logits:
#                     hidden_states:
#                     attentions:
#                     cross_attentions:
#                     past_key_values:
#                 }
#                 '''

#                 # nll_loss
#                 loss = y_hat['loss']
#                 # |loss| = (1)

#                 loss_list.append(loss)

#             loss_result = torch.mean(
#                 torch.Tensor(loss_list)
#             ).detach().cpu().numpy()

#             return loss_result


#     def train(
#         self,
#         train_loader,
#         valid_loader,
#         test_loader,
#         config
#         ):

#         best_valid_score = float('inf')
#         best_test_score = float('inf')


#         train_scores = []
#         valid_scores = []
#         test_scores = []

#         for epoch_index in range(self.n_epochs):
#             print("Epoch(%d/%d) start" % (
#                 epoch_index + 1,
#                 self.n_epochs
#             ))

#             # Training Session
#             train_score = self._train(train_loader)
#             valid_score = self._validate(valid_loader)
#             test_score = self._test(test_loader)

#             # train, test record 저장
#             train_scores.append(train_score)
#             valid_scores.append(valid_score)
#             test_scores.append(test_score)

#             if test_score <= best_test_score:
#                 best_test_score = test_score

#             print(
#                     "Epoch(%d/%d) result: train_score=%.4f  valid_score=%.4f test_score=%.4f best_test_score=%.4f" % (
#                     epoch_index + 1,
#                     self.n_epochs,
#                     train_score,
#                     valid_score,
#                     test_score,
#                     best_test_score,
#                 )
#             )

#         print("\n")
#         print("The Best Test Score in Testing Session is %.4f" % (
#                 best_test_score,
#             ))
#         print("\n")

        

#         #self.model.load_state_dict(torch.load("./checkpoint_root/checkpoint.pt"))

#         return train_scores, valid_scores, best_valid_score, best_test_score
