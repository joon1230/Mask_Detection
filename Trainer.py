import os
import pandas as pd
import time
from functools import partial
import glob
import numpy as np

import torch
import torch.optim

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import utils
import Datasets


class Trainer():
    def __init__(self, model, model_name, transform, dataset, optimizer, learning_rate, weight_decay, batch_size,
                 train_ratio, loss_function, epoch, input_size, nick_name, load_size=None, scheduler=None):
        self.model = model()
        self.dataset = partial(dataset, transform = transform, size = load_size)
        self.val_dataset = partial(Datasets.MaskDataset, transform=Datasets.transform_test(input_size=input_size))
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.train_test_split()
        self.epoch = epoch
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = scheduler(optimizer=self.optimizer)
        self.criterion = loss_function
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



        self.model.to(self.device)

        self.log = f"""| model_name : {model_name} | load_size : {load_size} | input_size : {input_size} | batch : {batch_size} |\n| train_people : {len(self.train_paths)} , val_people : {len(self.val_paths)} \n| loss function : {loss_function}\n| optimizer : {optimizer} | weight_decay : {weight_decay} | learning_rate : {learning_rate} \n| transform : {transform} """
        self.save_path = f'{model_name}/{optimizer}/{model_name + nick_name}'
        print(self.log)

    def train(self):
        for i in range(1, self.epoch + 1):
            self.train_one_epoch(i)

            if 1 - self.train_ratio:
                self.test_one_epoch(i)
                self.save_model(i, name='epoch', save_path=self.save_path)
        if not (1 - self.train_ratio):
            self.save_model(i, name='epoch', save_path=self.save_path)

    def train_one_epoch(self, epoch):
        self.model.train()

        train_loss = 0
        correct = 0
        total = 0
        load_time = 0
        inference_time = 0
        f1_score = 0

        self.log += f"\n\n====train model epoch : {epoch}====\n"
        print(f"\n\n====train model epoch : {epoch}====")
        start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            s = time.time()
            self.optimizer.zero_grad()  # 각각의 모델 파라미터 안의 gradient가 update 된다.
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            f1_score += self.f1_score(predicted, labels)

            inference_time += time.time() - s
            load_time = time.time() - start_time - inference_time

            if batch_idx % 10 == 0:
                self.log += "batch : %.3d | Loss: %.3f | Acc : %.3f%% | f1_score : %.3f%% | inference_time : %.3f / load_time : %.3f \n" \
                            % (
                            batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, f1_score / (batch_idx + 1),
                            inference_time / (batch_idx + 1), load_time / (batch_idx + 1e-04))
                print(
                    "batch : %.3d | Loss: %.3f | Acc : %.3f%% | f1_score : %.3f%% | inference_time : %.3f / load_time : %.3f \r" \
                    % (batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, f1_score / (batch_idx + 1),
                       inference_time / (batch_idx + 1), load_time / (batch_idx + 1e-04)), end="")

    def test_one_epoch(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        f1_score = 0

        print()
        print(f"====Evaluation model epoch : {epoch}====")
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.val_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                f1_score += self.f1_score(predicted, labels)

                print(f"progress:{batch_idx * self.batch_size} | {len(self.val_paths)} \r", end="")
        self.log += f'Accuracy of the network on the {total} test image : %d %% | f1_score : %.3f %%' % (
        100 * correct / total, f1_score / (batch_idx))
        print(f'Accuracy of the network on the {total} test image : %d %% | f1_score : %.3f %%' % (
        100 * correct / total, f1_score / (batch_idx)))

    def save_model(self, epoch, save_path, name='student'):
        print(f'saved model {save_path}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = f"{name}_{epoch}.pth"
        file_path = os.path.join(save_path, file_name)
        torch.save(self.model.state_dict(), file_path)
        with open(f'{save_path}/log.txt', 'w') as f:
            f.write(self.log)
        return None


    def train_test_split(self):
        meta_data = pd.read_csv('../input/data/train/train.csv')
        if 1 - self.train_ratio:
            meta_data['cls'] = meta_data.path.map(self.get_person_class)
            train_paths, val_paths = train_test_split(meta_data, train_size=self.train_ratio, stratify=meta_data['cls'])
            self.train_paths, self.val_paths = train_paths.path.values, val_paths.path.values

            self.train_loader = DataLoader(self.dataset(dir_paths=self.train_paths), batch_size=self.batch_size,
                                           shuffle=True, num_workers=4)
            self.val_loader = DataLoader(self.val_dataset(dir_paths=self.val_paths), batch_size=self.batch_size,
                                         shuffle=True, num_workers=4)

        else:
            self.train_paths, self.val_paths = list(meta_data.path.values), []
            self.train_loader = DataLoader(self.dataset(dir_paths=self.train_paths), batch_size=self.batch_size,
                                           shuffle=True, num_workers=4)

    def f1_score(self, y_pred, target):
        y_pred, target = np.asarray(y_pred.to('cpu')), np.asarray(target.to('cpu'))
        return f1_score(y_pred, target, average='macro')


    def get_person_class(self, data):
        gender = utils.get_gender_logit(data)
        age_cls = utils.get_age_class(data.split('_')[-1])
        return gender*3 + age_cls





