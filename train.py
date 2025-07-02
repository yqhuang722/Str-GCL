import torch
import torch.nn as nn
import tqdm
import random
import torch.nn.functional as F
from utils import label_classification
from model.model_entry import select_model
from loss import loss, loss_cross, loss_rule
from options import prepare_args, add_data_args
from data_process import data_loader, update

class Trainer:
    def __init__(self):
        self.args = prepare_args()
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        self.data = data_loader(self.args)
        self.args = add_data_args(self.args, self.data)
        self.model = select_model(self.args, self.data.alpha, self.data.beta)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                      lr = self.args.learning_rate,
                      weight_decay=self.args.weight_decay)
        
    def train(self):
        torch.cuda.set_device(self.args.gpu_id)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.data.to(device)
        
        print("=== Train ===")
        with tqdm.tqdm(total = self.args.num_epochs, desc = '(T)') as pbar:  
            for epoch in range(1, self.args.num_epochs + 1):
                self.model.train()
                self.optimizer.zero_grad()
                self.data = update(self.data, self.args)
                ori_z_1, wei_z_1 = self.model(self.data.aug_1)
                ori_z_2, wei_z_2 = self.model(self.data.aug_2)
                
                digits_1 = loss(self.model.projector(ori_z_1), self.model.projector(ori_z_2), self.args)
                wei_digits_1 = loss_rule(wei_z_1, self.args.tau_rule)
                wei_digits_2 = loss_rule(wei_z_2, self.args.tau_rule)
                cross_digits_1 = loss_cross(self.model.projector(wei_z_1), self.model.projector(ori_z_1))
                cross_digits_2 = loss_cross(self.model.projector(wei_z_2), self.model.projector(ori_z_2))
                digits = self.args.loss_base * digits_1 + self.args.loss_rule * 0.5 * (wei_digits_1 + wei_digits_2) + self.args.loss_cross * 0.5 * (cross_digits_1 + cross_digits_2)
                digits.backward()
                self.optimizer.step()
                pbar.set_postfix({'loss': digits.item()})
                pbar.update()

    def test(self):
        print("=== Test ===")
        self.model.eval()
        ori_z, _ = self.model(self.data.raw)
        result = label_classification(ori_z, self.data.y, ratio=0.1)
        with open(self.args.log_dir, 'a') as f:
            f.write(str(result) + '\n')

def main():
    trainer = Trainer()
    with open(trainer.args.log_dir, 'a') as f:
        f.write('****'*20+'\n')
        f.write('\n\n'+'##'*20+'\n')
        f.write(str(trainer.args) + '\n')
    trainer.train()
    trainer.test()
    
if __name__ == '__main__':
    main()
