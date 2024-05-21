# -*- coding:utf-8 -*-
import os
import random
import warnings
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as opt
from modules.utils import *
from torch.utils.data import DataLoader
from modules.main import AudioSleepNet
from datasets.data_loader import AudioDataset
from datasets.utils import subject_group_cross_validation
from sklearn.metrics import f1_score


warnings.filterwarnings(action='ignore')

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--base_path', default=os.path.join('..', 'data'))
    parser.add_argument('--k_splits', default=5)
    parser.add_argument('--n_fold', default=0, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--temporal_context_length', default=40, type=int)
    parser.add_argument('--temporal_context_window_size', default=4, type=int)

    # Train Hyperparameter
    parser.add_argument('--train_epochs', default=10, type=int)
    parser.add_argument('--train_base_learning_rate', default=1e-4, type=float)
    parser.add_argument('--train_batch_size', default=4, type=int)
    parser.add_argument('--train_batch_accumulation', default=4, type=int)

    # Model Hyperparameter
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--rnn_hidden', default=128, type=int)
    parser.add_argument('--output_classes', default={'sleep_staging': 4,
                                                     'apnea': 2, 'hypopnea': 2, 'arousal': 2,
                                                     'snore': 2}, type=int)
    parser.add_argument('--print_point', default=20, type=int)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.model = AudioSleepNet(seq_len=args.temporal_context_length, rnn_hidden=args.rnn_hidden,
                                   output_classes=args.output_classes).to(device)
        self.train_paths, self.test_paths = self.get_paths()

        self.eff_batch_size = self.args.train_batch_size * self.args.train_batch_accumulation
        self.lr = self.args.train_base_learning_rate * self.eff_batch_size / 256
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.train_epochs)
        self.criterion = nn.CrossEntropyLoss()

        print('[AudioSleepNet Parameter]')
        print('   >> Model Size : {0:.2f}MB'.format(model_size(self.model)))
        print('   >> Leaning Rate : {0}'.format(self.lr))

    def train(self):
        train_dataset = AudioDataset(paths=self.train_paths[:1],
                                     temporal_context_length=self.args.temporal_context_length,
                                     temporal_context_window_size=self.args.temporal_context_window_size)
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size)
        test_dataset = AudioDataset(paths=self.test_paths[:1],
                                    temporal_context_length=self.args.temporal_context_length,
                                    temporal_context_window_size=self.args.temporal_context_window_size)
        test_dataloader = DataLoader(test_dataset, batch_size=self.args.train_batch_size)

        total_step = 0
        for epoch in range(self.args.epochs):
            step = 0
            self.model.train()
            self.optimizer.zero_grad()

            # [Train Step]
            for batch in train_dataloader:
                self.optimizer.zero_grad()
                x, y = batch
                x, y = x.to(device), y.to(device)
                outs = self.model(x)
                loss1, loss2, loss3, loss4, loss5, total_loss = self.get_total_loss(outs, y)
                total_loss.backward()

                if (step + 1) % self.args.train_batch_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (total_step + 1) % self.args.print_point == 0:
                    print('[Epoch] : {0:03d}  [Step] : {1:08d}  '
                          '[Loss 1] : {2:02.4f}  '
                          '[Loss 2] : {3:02.4f}  '
                          '[Loss 3] : {4:02.4f}  '
                          '[Loss 4] : {5:02.4f}  '
                          '[Loss 5] : {6:02.4f}  '
                          '[Total Loss] : {7:02.4f}  '.format(epoch, total_step + 1,
                                                              loss1, loss2, loss3, loss4, loss5,
                                                              total_loss))
                step += 1
                total_step += 1

            # [Evaluation Step]
            self.model.eval()
            eval_mf1, eval_mf2, eval_mf3, eval_mf4, eval_mf5 = [], [], [], [], []
            eval_total_loss = []

            for batch in test_dataloader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                outs = self.model(x)
                (loss1, mf1), (loss2, mf2), (loss3, mf3), (loss4, mf4), (loss5, mf5), total_loss \
                    = self.get_total_loss(outs, y, get_performance=True)
                eval_mf1.append(mf1)
                eval_mf2.append(mf2)
                eval_mf3.append(mf3)
                eval_mf4.append(mf4)
                eval_mf5.append(mf5)
                eval_total_loss.append(total_loss.detach().numpy())

            print('[Evaluation] - {0:03d}'.format(epoch))
            print('[Sleep Stage] : {0:02.4f}  '
                  '[Apnea] : {1:02.4f}  '
                  '[Hypopnea] : {2:02.4f}  '
                  '[Arousal] : {3:02.4f}  '
                  '[Snore] : {4:02.4f}  '
                  '[Total Loss] : {5:02.4f}  '.format(np.mean(eval_mf1), np.mean(eval_mf2),
                                                      np.mean(eval_mf3), np.mean(eval_mf4),
                                                      np.mean(eval_mf5),
                                                      np.mean(eval_total_loss)),
                  end='\n\n')

    def get_paths(self):
        paths = subject_group_cross_validation(self.args.base_path,
                                               n_splits=self.args.k_splits)[self.args.n_fold]
        train_paths, test_paths = paths['train_paths'], paths['test_paths']
        return train_paths, test_paths

    def get_total_loss(self, preds, reals, get_performance=False):
        preds = list(preds.values())
        o1, o2, o3, o4, o5 = preds[0], preds[1], preds[2], preds[3], preds[4]
        y1, y2, y3, y4, y5 = reals[..., 0], reals[..., 1], reals[..., 2], reals[..., 3], reals[..., 4]

        if not get_performance:
            loss1 = self.get_loss(o1, y1)[0]
            loss2 = self.get_loss(o2, y2)[0]
            loss3 = self.get_loss(o3, y3)[0]
            loss4 = self.get_loss(o4, y4)[0]
            loss5 = self.get_loss(o5, y5)[0]
            total_loss = loss1 + loss2 + loss3 + loss4 + loss5
            return loss1, loss2, loss3, loss4, loss5, total_loss
        else:
            loss1, pred1, real1 = self.get_loss(o1, y1)
            loss2, pred2, real2 = self.get_loss(o2, y2)
            loss3, pred3, real3 = self.get_loss(o3, y3)
            loss4, pred4, real4 = self.get_loss(o4, y4)
            loss5, pred5, real5 = self.get_loss(o5, y5)
            total_loss = loss1 + loss2 + loss3 + loss4 + loss5
            mf1 = self.get_performance(pred1, real1)
            mf2 = self.get_performance(pred2, real2)
            mf3 = self.get_performance(pred3, real3)
            mf4 = self.get_performance(pred4, real4)
            mf5 = self.get_performance(pred5, real5)
            return (loss1, mf1), (loss2, mf2), (loss3, mf3), (loss4, mf4), (loss5, mf5), total_loss

    def save_ckpt(self, model_state):
        ckpt_path = os.path.join(self.args.ckpt_path, self.args.ckpt_name, 'model')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        torch.save({
            'model_name': 'AudioSleepNet-Exp1',
            'model_state': model_state,
            'model_parameter': {
            },
            'hyperparameter': self.args.__dict__,
            'paths': {'train_paths': self.train_paths, 'test_paths': self.test_paths}
        }, os.path.join(ckpt_path, 'best_model.pth'))

    def get_loss(self, pred, real):
        if pred.dim() == 3:
            pred = pred.view(-1, pred.size(2))
            real = real.view(-1)
        loss = self.criterion(pred, real)
        return loss, pred, real

    @staticmethod
    def get_performance(pred, real):
        pred = torch.argmax(pred, dim=-1).detach().cpu().numpy()
        real = real.detach().cpu().numpy()
        score = f1_score(real, pred, average='macro')
        return score


if __name__ == '__main__':
    Trainer(get_args()).train()
