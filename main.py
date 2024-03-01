import torch
from numpy import mean
from tqdm import tqdm
from transformers import logging, AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from config import get_config
from data import load_data
from loss import CELoss
from model import MP_TFWA


class Instructor:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.subject = args.subject
        self.max_lengths = args.max_lengths
        self.query_lengths = args.query_lengths
        self.prompt_lengths = args.prompt_lengths

        self.logger.info('> creating model {}'.format(args.model_name))
        if args.model_name == 'bert-base':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.mrc_model = AutoModel.from_pretrained('bert-base-uncased')
            self.co_model = AutoModel.from_pretrained('bert-base-uncased')
            self.pl_model = AutoModel.from_pretrained('bert-base-uncased')
        elif args.model_name == 'albert-base-v2':
            self.tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
            self.mrc_model = AutoModel.from_pretrained('albert-base-v2')
            self.co_model = AutoModel.from_pretrained('albert-base-v2')
            self.pl_model = AutoModel.from_pretrained('albert-base-v2')
        elif args.model_name == 'roberta-base':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base',add_prefix_space=True)
            self.mrc_model = AutoModel.from_pretrained('roberta-base')
            self.co_model = AutoModel.from_pretrained('roberta-base')
            self.pl_model = AutoModel.from_pretrained('roberta-base')
        elif args.model_name == 'electra-base-discriminator':
            self.tokenizer = AutoTokenizer.from_pretrained('electra-base-discriminator',add_prefix_space=True)
            self.mrc_model = AutoModel.from_pretrained('electra-base-discriminator')
            self.co_model = AutoModel.from_pretrained('electra-base-discriminator')
            self.pl_model = AutoModel.from_pretrained('electra-base-discriminator')
        else:
            raise ValueError('unknown model')
        if args.method_name == 'MP-TFWA':
            self.model = MP_TFWA(self.mrc_model, self.co_model,
                                 self.pl_model, args.num_classes, args.max_lengths, self.query_lengths,
                                 self.prompt_lengths)
        else:
            raise ValueError('unknown method')
        self.model.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train(self, dataloader, criterion, optimizer, scheduler):
        train_loss, n_correct, n_train = 0, 0, 0

        self.model.train()
        for mrc_inputs, targets, text_inputs, mask_inputs, mask_index in tqdm(dataloader, disable=self.args.backend,
                                                                              ascii=' >='):
            mrc_inputs = {k: v.to(self.args.device) for k, v in mrc_inputs.items()}
            text_inputs = {k: v.to(self.args.device) for k, v in text_inputs.items()}
            mask_inputs = {k: v.to(self.args.device) for k, v in mask_inputs.items()}
            targets = targets.to(self.args.device)

            predicts = self.model(mrc_inputs, text_inputs, mask_inputs, mask_index)
            loss = criterion(predicts, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
            n_train += targets.size(0)

        return train_loss / n_train, n_correct / n_train

    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        self.model.eval()

        with torch.no_grad():
            for mrc_inputs, targets, text_inputs, mask_inputs, mask_index in tqdm(dataloader, disable=self.args.backend,
                                                                                  ascii=' >='):
                mrc_inputs = {k: v.to(self.args.device) for k, v in mrc_inputs.items()}
                text_inputs = {k: v.to(self.args.device) for k, v in text_inputs.items()}
                mask_inputs = {k: v.to(self.args.device) for k, v in mask_inputs.items()}
                targets = targets.to(self.args.device)

                predicts = self.model(mrc_inputs, text_inputs, mask_inputs, mask_index)
                loss = criterion(predicts, targets)
                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(predicts, dim=1) == targets).sum().item()
                n_test += targets.size(0)

        return test_loss / n_test, n_correct / n_test

    def run(self, index_fold):
        train_dataloader, test_dataloader = load_data(dataset=self.args.dataset,
                                                      data_dir=self.args.data_dir,
                                                      tokenizer=self.tokenizer,
                                                      train_batch_size=self.args.train_batch_size,
                                                      test_batch_size=self.args.test_batch_size,
                                                      workers=0,
                                                      index_fold=index_fold,
                                                      subject=self.subject)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        # Criterion
        criterion = CELoss()
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.decay, eps=self.args.eps)
        # Warm up with linear strategy
        total_steps = len(train_dataloader) * self.args.num_epoch
        warmup_steps = 0.1 * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)
        best_loss, best_acc = 0, 0
        for epoch in range(self.args.num_epoch):
            train_loss, train_acc = self._train(train_dataloader, criterion, optimizer, scheduler)
            test_loss, test_acc = self._test(test_dataloader, criterion)
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss = test_acc, test_loss
            self.logger.info(
                '{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch, 100 * (epoch + 1) / self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}'.format(train_loss, train_acc * 100))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}'.format(test_loss, test_acc * 100))
        self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc * 100))
        self.logger.info('log saved: {}'.format(self.args.log_name))
        return best_acc


if __name__ == '__main__':
    accs = []
    for i in range(50):
        logging.set_verbosity_error()
        args, logger = get_config()
        # 10-fold validation
        if (args.dataset in ['mr', 'cr', 'subj', 'mpqa']):
            k_fold_accs = []
            for index_fold in range(10):
                ins = Instructor(args, logger)
                acc = ins.run(index_fold)
                k_fold_accs.append(acc)
            print('k_fold_accs:', k_fold_accs)
            accs.append(mean(k_fold_accs))
            print('total_accs', accs)
        else:
            ins = Instructor(args, logger)
            acc = ins.run(0)
            accs.append(acc)
            print(accs)
    print('The final results')
    print(accs)
