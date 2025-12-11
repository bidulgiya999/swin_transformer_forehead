import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision import models #resnet50_custom 사용하기때문에 필요없음
from utils import FocalLoss, AverageMeter, CB_loss, LabelSmoothingCrossEntropy
import os
from sklearn.metrics import balanced_accuracy_score, f1_score
# ### 수정된 부분 1: 새로운 import 문 추가 ###
from swin_custom import swin_tiny_custom

class Model(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.2, samples_per_cls=None):
        super(Model, self).__init__()
        
        # ### 수정된 부분 2: 모델 로드 부분을 swin_tiny_custom으로 교체 ###
        # 기존 코드: self.model = resnet50_custom(pretrained=True)
        # 새로운 코드:
        self.model = swin_tiny_custom(pretrained=True)
        
        # Swin Transformer는 fc 대신 head를 사용합니다.
        # 마지막 classification layer 수정
        in_features = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(in_features, num_classes)
        )
        
        # 클래스별 샘플 수 저장
        self.samples_per_cls = samples_per_cls
        
    def forward(self, x):
        return self.model(x)
    
    # train_epoch, validate, test 등 나머지 모든 메소드는 그대로 유지됩니다.
    # ... (이하 모든 코드는 기존과 동일) ...
    
    def train_epoch(self, train_loader, optimizer, device, loss_type, gamma=None, alpha=None, beta=None, smoothing=None, epoch=None):
        self.model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        
        # 매 에폭마다 랜덤 시드 설정 (실험 재현성 보장)
        if epoch is not None:
            import random
            import numpy as np
            import torch
            
            # 에폭별 고유한 시드 설정
            epoch_seed = 42 + epoch * 1000  # 에폭마다 다른 시드
            
            # Python 랜덤 시드 설정
            random.seed(epoch_seed)
            # NumPy 랜덤 시드 설정
            np.random.seed(epoch_seed)
            # PyTorch 랜덤 시드 설정
            torch.manual_seed(epoch_seed)
            
            # GPU 랜덤 시드 설정
            if torch.cuda.is_available():
                torch.cuda.manual_seed(epoch_seed)
                torch.cuda.manual_seed_all(epoch_seed)
            
            # MPS 랜덤 시드 설정
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(epoch_seed)
        
        # 손실 함수 선택
        if loss_type == 'focal':
            criterion = FocalLoss(gamma=gamma, alpha=alpha)
        elif loss_type == 'cb':
            # 저장된 클래스별 샘플 수 사용
            if self.samples_per_cls is None:
                # fallback: 데이터셋에서 직접 가져오기
                try:
                    # Subset으로 감싸진 경우 원본 데이터셋에 접근
                    if hasattr(train_loader.dataset, 'dataset'):
                        samples_per_cls = train_loader.dataset.dataset.samples_per_cls
                    else:
                        samples_per_cls = train_loader.dataset.samples_per_cls
                except AttributeError:
                    # 더 깊은 중첩된 경우
                    current_dataset = train_loader.dataset
                    while hasattr(current_dataset, 'dataset'):
                        current_dataset = current_dataset.dataset
                    samples_per_cls = current_dataset.samples_per_cls
                
                self.samples_per_cls = samples_per_cls
            
            criterion = CB_loss(samples_per_cls=self.samples_per_cls, 
                              no_of_classes=len(self.samples_per_cls),
                              gamma=gamma,
                              beta=beta)
        elif loss_type == 'label_smoothing':
            criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
        
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = self(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 통계 업데이트
            train_loss.update(loss.item(), images.size(0))
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            train_acc.update(correct / images.size(0), images.size(0))
            
        return train_loss.avg, train_acc.avg
    
    def validate(self, val_loader, device, log_dir=None, epoch=None, experiment_id=None):
        if log_dir is None:
            log_dir = "."
        self.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        all_preds = []
        all_labels = []

        # 로그 파일 준비 - 실험별로 고유한 파일명 사용
        log_path = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            
            # 실험 ID가 있으면 고유한 파일명 사용, 없으면 기본 파일명 사용
            if experiment_id is not None:
                log_path = os.path.join(log_dir, f"val_probs_log_{experiment_id}.csv")
            else:
                log_path = os.path.join(log_dir, "val_probs_log.csv")
            
            # 파일이 없으면 헤더 작성
            if not os.path.exists(log_path):
                with open(log_path, "w") as f:
                    f.write("epoch,filename,probs,gt,pred,correct\n")

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels, fnames in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                _, predicted = outputs.max(1)

                loss = criterion(outputs, labels)
                val_loss.update(loss.item(), images.size(0))
                correct = predicted.eq(labels).sum().item()
                val_acc.update(correct / images.size(0), images.size(0))

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 로그 파일에 기록
                if log_path is not None:
                    with open(log_path, "a") as log_f:
                        for i in range(len(fnames)):
                            gt = labels[i].item()
                            pred = predicted[i].item()
                            is_correct = 1 if gt == pred else 0
                            probs_str = '[' + ','.join([f"{p:.6f}" for p in probs[i]]) + ']'
                            log_f.write(f"{epoch},{fnames[i]},{probs_str},{gt},{pred},{is_correct}\n")

        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        return val_loss.avg, val_acc.avg, balanced_acc, macro_f1
    
    def test(self, test_loader, device, log_dir=None, experiment_id=None):
        if log_dir is None:
            log_dir = "."
        self.eval()
        test_loss = AverageMeter()
        test_acc = AverageMeter()
        all_preds = []
        all_labels = []

        # 로그 파일 준비 - 실험별로 고유한 파일명 사용
        log_path = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            
            # 실험 ID가 있으면 고유한 파일명 사용, 없으면 기본 파일명 사용
            if experiment_id is not None:
                log_path = os.path.join(log_dir, f"test_probs_log_{experiment_id}.csv")
            else:
                log_path = os.path.join(log_dir, "test_probs_log.csv")
            
            # 테스트 파일은 매번 새로 작성
            with open(log_path, "w") as f:
                f.write("filename,probs,gt,pred,correct\n")

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels, fnames in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                _, predicted = outputs.max(1)

                loss = criterion(outputs, labels)
                test_loss.update(loss.item(), images.size(0))
                correct = predicted.eq(labels).sum().item()
                test_acc.update(correct / images.size(0), images.size(0))

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 로그 파일에 기록
                if log_path is not None:
                    with open(log_path, "a") as log_f:
                        for i in range(len(fnames)):
                            gt = labels[i].item()
                            pred = predicted[i].item()
                            is_correct = 1 if gt == pred else 0
                            probs_str = '[' + ','.join([f"{p:.6f}" for p in probs[i]]) + ']'
                            log_f.write(f"{fnames[i]},{probs_str},{gt},{pred},{is_correct}\n")

        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')

        return test_loss.avg, test_acc.avg, balanced_acc, macro_f1
    
    def save_checkpoint(self, optimizer, epoch, best_acc, path, scheduler=None):
        """체크포인트 저장 (옵션: scheduler)"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'samples_per_cls': self.samples_per_cls
        }
        if scheduler is not None:
            state['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(state, path)
        
    def load_checkpoint(self, path, optimizer, scheduler=None, device=None):
        """체크포인트 로드 (옵션: scheduler, device)"""
        # 1. CPU로 안전하게 로드
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 2. samples_per_cls 복원
        if 'samples_per_cls' in checkpoint:
            self.samples_per_cls = checkpoint['samples_per_cls']
            
        # 3. scheduler 복원
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 4. 지정된 디바이스로 모델 이동
        if device is not None:
            self.to(device)
        
        start_epoch = checkpoint['epoch'] + 1  # 다음 에포크부터 시작
        best_acc = checkpoint['best_acc']
        return self, optimizer, start_epoch, best_acc 