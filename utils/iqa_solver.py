import torch
from scipy import stats
import numpy as np

from utils.dataset import data_loader
import os
from tqdm import tqdm

def cal_srocc_plcc(pred_score, gt_score):
    try:
        srocc, _ = stats.spearmanr(pred_score, gt_score)
        plcc, _ = stats.pearsonr(pred_score, gt_score)
        KRCC = stats.kendalltau(pred_score, gt_score)[0]
    except:
        return 0, 0, 0

    return KRCC, srocc, plcc

class Solver:
    def __init__(self, config, path, train_index, test_index):
        train_loader = data_loader.Data_Loader(config, path, train_index, istrain=True)
        test_loader = data_loader.Data_Loader(config, path, test_index, istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()
        print('Traning data number: ', len(train_index))
        print('Testing data number: ', len(test_index))
        
        if config.loss == 'MAE':
            self.loss = torch.nn.L1Loss().cuda()
        elif config.loss == 'MSE':
            self.loss = torch.nn.MSELoss().cuda()
        else:
            raise 'Only Support MAE and MSE loss.'

        if config.teacher_pkl is not None:
            pretrain = config.teacher_pkl
            print('Loading teacher_pkl...', pretrain)
            from models import MobileVit_IQA as Teacher
            self.Teacher = Teacher.Model(is_teacher=True).cuda()
            self.Teacher.load_state_dict(torch.load(pretrain))
            self.Teacher.train(False)
            
            print('Loading netAttIQAMoNet...')
            from models import MobileNet_IQA as Student
            self.Student = Student.model().cuda()
            self.Student.train(True)
        else:
            if config.model == 'MobileVit_IQA':
                print('Loading MobileVit_IQA...')
                from models import MobileVit_IQA as model
                self.model = model.Model().cuda()
                self.model.train(True)

        self.epochs = config.epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.T_max, eta_min=config.eta_min)

        self.model_save_path = os.path.join(config.save_path, 'best_model.pkl')
        self.config = config

    def train(self):
        if self.config.teacher_pkl is not None:
            return self.train_distill()
        else:
            return self.train_teacher()
    
    def train_distill(self):
        def test():
            """Testing"""
            self.Teacher.train(False)
            self.Student.train(False)
            pred_scores, gt_scores = [], []

            with torch.no_grad():
                for img, label in tqdm(self.test_data,ncols=85):
                    full_img = img.cuda()
                    label = label.view(-1).cuda()

                    _, S_DOF, stu_score, tea_score = self.Student(full_img, self.Teacher)
                    
                    pred_scores = pred_scores + stu_score.cpu().tolist()
                    gt_scores = gt_scores + label.cpu().tolist()

            test_krcc, test_srocc, test_plcc = cal_srocc_plcc(pred_scores, gt_scores)
            self.Student.train(True)
            return test_krcc, test_srocc, test_plcc
    
        """Training"""
        best_srocc = 0.0
        best_plcc = 0.0
        best_kcc = 0
        print('----------------------------------')
        print('Epoch\tTrain_Loss\tTrain_KRCC\tTrain_SROCC\tTrain_PLCC\tTest_KRCC\tTest_SROCC\tTest_PLCC')
        
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for idx, (img, label) in enumerate(tqdm(self.train_data,ncols=85)):
                full_img = img.cuda()
                label = label.view(-1).cuda()
                
                with torch.no_grad():
                    T_x, T_DOF, T_score = self.Teacher(full_img)
                S_x, S_DOF, stu_score, tea_score = self.Student(full_img, self.Teacher)
                
                DOF_loss = self.loss(S_DOF, T_DOF.detach())
                x_loss = self.loss(S_x, T_x.detach())
                score_loss = self.loss(stu_score, label.float().detach())

                loss = DOF_loss + score_loss + x_loss
                
                pred_scores = pred_scores + stu_score.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()

                epoch_loss.append(score_loss.item())
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.scheduler.step()

            train_kcc, train_srocc, train_plcc = cal_srocc_plcc(pred_scores, gt_scores)
            test_kcc, test_srocc, test_plcc = test()
            if test_kcc + test_srocc + test_plcc > best_kcc + best_srocc + best_plcc:
                best_srocc = test_srocc
                best_plcc = test_plcc
                best_kcc = test_kcc
                torch.save(self.Student.state_dict(), self.model_save_path)
                print('Model saved in: ', self.model_save_path)

            print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(t + 1, round(np.mean(epoch_loss), 4), round(train_kcc, 4), round(train_srocc, 4),
                                                  round(train_plcc, 4), round(test_kcc, 4), round(test_srocc, 4), round(test_plcc, 4)))

        print('Best test SROCC {}, PLCC {}'.format(round(best_srocc, 6), round(best_plcc, 6)))

        return best_kcc, best_srocc, best_plcc

    def train_teacher(self):
        def test():
            self.model.train(False)
            pred_scores, gt_scores = [], []

            with torch.no_grad():
                for img, label in tqdm(self.test_data,ncols=85):
                    full_img = img.cuda()
                    label = label.view(-1).cuda()
                    pred = self.model(full_img)
                    pred_scores = pred_scores + pred.cpu().tolist()
                    gt_scores = gt_scores + label.cpu().tolist()

            test_krcc, test_srocc, test_plcc = cal_srocc_plcc(pred_scores, gt_scores)
            self.model.train(True)
            return test_krcc, test_srocc, test_plcc

        best_srocc = 0.0
        best_plcc = 0.0
        best_kcc = 0
        print('----------------------------------')
        print('Epoch\tTrain_Loss\tTrain_KRCC\tTrain_SROCC\tTrain_PLCC\tTest_KRCC\tTest_SROCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for idx, (img, label) in enumerate(tqdm(self.train_data,ncols=85)):
                full_img = img.cuda()
                label = label.view(-1).cuda()
                
                pred  = self.model(full_img)
                loss = self.loss(pred.view(-1), label.float().detach()).unsqueeze(0)
                
                pred_scores = pred_scores + pred.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()
                
                loss = torch.mean(loss)
                epoch_loss.append(loss.item())
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.scheduler.step()

            train_kcc, train_srocc, train_plcc = cal_srocc_plcc(pred_scores, gt_scores)
            test_kcc, test_srocc, test_plcc = test()
            if test_kcc + test_srocc + test_plcc > best_kcc + best_srocc + best_plcc:
                best_srocc = test_srocc
                best_plcc = test_plcc
                best_kcc = test_kcc
                torch.save(self.model.state_dict(), self.model_save_path)
                print('Model saved in: ', self.model_save_path)

            print('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(t + 1, round(np.mean(epoch_loss), 4), round(train_kcc, 4), round(train_srocc, 4),
                                                  round(train_plcc, 4), round(test_kcc, 4), round(test_srocc, 4), round(test_plcc, 4)))

        print('Best test SROCC {}, PLCC {}'.format(round(best_srocc, 6), round(best_plcc, 6)))

        return best_kcc, best_srocc, best_plcc

    