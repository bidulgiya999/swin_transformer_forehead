import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil
import numpy as np
from PIL import Image

class AverageMeter(object):
    """평균값을 계산하고 저장하는 클래스"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class FocalLoss(nn.Module):
    """Focal Loss 구현"""
    def __init__(self, gamma, alpha, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CB_loss(nn.Module):
    """
    Class-Balanced Loss with Focal modulation, suitable for multi-class classification
    Based on: "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., CVPR 2019)
    """
    def __init__(self, samples_per_cls, no_of_classes, gamma=2.0, beta=0.999):
        super(CB_loss, self).__init__()
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.gamma = gamma
        self.beta = beta

        # 0으로 나누기 방지
        samples_per_cls = np.array(samples_per_cls)
        samples_per_cls = np.maximum(samples_per_cls, 1)  # 최소값 1로 설정
        
        effective_num = 1.0 - np.power(self.beta, samples_per_cls)
        weights = (1.0 - self.beta) / effective_num
        weights = weights / np.sum(weights) * self.no_of_classes
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, logits, labels):
        """
        logits: [batch_size, num_classes]
        labels: [batch_size] (class indices)
        """
        device = logits.device
        labels = labels.to(torch.long)
        
        # 디바이스 호환성 확인
        try:
            class_weights = self.class_weights.to(device)
        except Exception as e:
            # MPS에서 호환성 문제가 있을 경우 CPU로 계산
            print(f"디바이스 호환성 문제로 CPU에서 계산: {e}")
            device = torch.device('cpu')
            logits = logits.to(device)
            labels = labels.to(device)
            class_weights = self.class_weights.to(device)

        # Class-balanced weights per sample
        sample_weights = class_weights[labels]  # shape: [batch_size]

        # Standard cross-entropy loss
        log_probs = F.log_softmax(logits, dim=1)
        ce_loss = F.nll_loss(log_probs, labels, reduction='none')  # shape: [batch_size]

        # Focal modulation
        pt = torch.exp(-ce_loss)  # pt = probability of the true class
        focal_modulation = (1 - pt) ** self.gamma

        # Final loss
        cb_focal_loss = sample_weights * focal_modulation * ce_loss
        return cb_focal_loss.mean()

# class CB_loss(nn.Module):
#     """Class Balanced Loss 구현"""
#     def __init__(self, samples_per_cls, no_of_classes, gamma, beta):
#         super(CB_loss, self).__init__()
#         self.samples_per_cls = samples_per_cls
#         self.no_of_classes = no_of_classes
#         self.beta = beta
#         self.gamma = gamma

#     def forward(self, logits, labels):
#         effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
#         weights = (1.0 - self.beta) / np.array(effective_num)
#         weights = weights / np.sum(weights) * self.no_of_classes

#         labels_one_hot = F.one_hot(labels.to(torch.int64), self.no_of_classes).float()
#         weights = torch.tensor(weights).float().to(logits.device)
#         weights = weights.unsqueeze(0)
#         weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
#         weights = weights.sum(1)
#         weights = weights.unsqueeze(1)
#         weights = weights.repeat(1, self.no_of_classes)

#         cb_loss = self.focal_loss(logits, labels_one_hot, weights)
#         return cb_loss

#     def focal_loss(self, logits, labels, alpha):
#         BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")
#         if self.gamma == 0.0:
#             modulator = 1.0
#         else:
#             modulator = torch.exp(-self.gamma * labels * logits - self.gamma * torch.log(1 + torch.exp(-1.0 * logits)))
#         loss = modulator * BCLoss
#         weighted_loss = alpha * loss
#         focal_loss = torch.sum(weighted_loss)
#         focal_loss /= torch.sum(labels)
#         return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing이 적용된 Cross Entropy Loss"""
    def __init__(self, smoothing):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def mkdir(path):
    """디렉토리 생성"""
    if not os.path.exists(path):
        os.makedirs(path)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """체크포인트 저장"""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def analyze_dataset_structure(root_dirs):
    """
    데이터셋 구조를 분석하여 클래스별 샘플 수를 계산하는 함수
    
    Args:
        root_dirs (list): 데이터셋 루트 디렉토리 리스트
        
    Returns:
        dict: 분석 결과
            - classes: 클래스 목록
            - class_to_idx: 클래스명을 인덱스로 매핑
            - samples_per_class: 클래스별 샘플 수 (딕셔너리)
            - samples_per_cls: 클래스별 샘플 수 (리스트, 클래스 순서대로)
            - total_images: 전체 이미지 수
            - dataset_images: 각 데이터셋별 이미지 수
    """
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]
    
    # 모든 루트 디렉토리에서 클래스 폴더 수집
    all_classes = set()
    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"디렉토리가 존재하지 않습니다: {root_dir}")
        
        classes = [cls for cls in os.listdir(root_dir) 
                  if cls != '.DS_Store' and os.path.isdir(os.path.join(root_dir, cls))]
        all_classes.update(classes)
    
    if not all_classes:
        raise ValueError("유효한 클래스 폴더를 찾을 수 없습니다.")
    
    # 클래스 정렬 및 인덱스 매핑
    classes = sorted(list(all_classes))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    # 클래스별 샘플 수를 저장할 딕셔너리
    samples_per_class = {cls_name: 0 for cls_name in classes}
    
    # 각 데이터셋별 이미지 수를 저장할 딕셔너리
    dataset_images = {}
    
    for root_dir in root_dirs:
        dataset_images[root_dir] = 0
        for class_name in classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')) and img_name != '.DS_Store':
                    img_path = os.path.join(class_dir, img_name)
                    # 이미지 파일 유효성 검사
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        samples_per_class[class_name] += 1
                        dataset_images[root_dir] += 1
                    except Exception as e:
                        print(f"손상된 이미지 파일 무시: {img_path} - {e}")
    
    # 클래스별 샘플 수를 리스트로 변환 (클래스 순서대로)
    samples_per_cls = [samples_per_class[cls_name] for cls_name in classes]
    
    # 전체 이미지 수 계산
    total_images = sum(samples_per_class.values())
    
    if total_images == 0:
        raise ValueError("유효한 이미지 파일을 찾을 수 없습니다.")
    
    return {
        'classes': classes,
        'class_to_idx': class_to_idx,
        'samples_per_class': samples_per_class,
        'samples_per_cls': samples_per_cls,
        'total_images': total_images,
        'dataset_images': dataset_images
    }

def print_dataset_analysis(analysis_result):
    """
    데이터셋 분석 결과를 출력하는 함수
    
    Args:
        analysis_result (dict): analyze_dataset_structure 함수의 결과
    """
    print(f"총 {analysis_result['total_images']}개의 이미지가 로드되었습니다.")
    print(f"클래스 수: {len(analysis_result['classes'])}")
    print(f"클래스 목록: {', '.join(analysis_result['classes'])}")
    print("클래스별 샘플 수:")
    for cls_name, count in analysis_result['samples_per_class'].items():
        print(f"  {cls_name}: {count}개")
    
    # 각 데이터셋별 이미지 수 출력
    for root_dir, count in analysis_result['dataset_images'].items():
        print(f"데이터셋 ({root_dir}): {count}개 이미지")

def calculate_augmentation_needs(analysis_result, target_samples_per_class=None):
    """
    데이터 증강이 필요한 클래스와 증강 횟수를 계산하는 함수
    
    Args:
        analysis_result (dict): analyze_dataset_structure 함수의 결과
        target_samples_per_class (int, optional): 목표 샘플 수. None이면 가장 많은 클래스의 샘플 수로 설정
        
    Returns:
        dict: 증강 계획
            - target_samples: 목표 샘플 수
            - augmentation_needs: 클래스별 필요한 증강 횟수
            - total_augmentations: 전체 필요한 증강 횟수
    """
    samples_per_class = analysis_result['samples_per_class']
    
    # 목표 샘플 수 설정
    if target_samples_per_class is None:
        target_samples = max(samples_per_class.values())
    else:
        target_samples = target_samples_per_class
    
    # 클래스별 필요한 증강 횟수 계산
    augmentation_needs = {}
    total_augmentations = 0
    
    for class_name, current_samples in samples_per_class.items():
        if current_samples < target_samples:
            needed = target_samples - current_samples
            augmentation_needs[class_name] = needed
            total_augmentations += needed
        else:
            augmentation_needs[class_name] = 0
    
    return {
        'target_samples': target_samples,
        'augmentation_needs': augmentation_needs,
        'total_augmentations': total_augmentations
    }

def print_augmentation_plan(augmentation_plan, analysis_result):
    """
    증강 계획을 출력하는 함수
    
    Args:
        augmentation_plan (dict): calculate_augmentation_needs 함수의 결과
        analysis_result (dict): analyze_dataset_structure 함수의 결과
    """
    print(f"\n=== 데이터 증강 계획 ===")
    print(f"목표 샘플 수: {augmentation_plan['target_samples']}개")
    print(f"전체 필요한 증강 횟수: {augmentation_plan['total_augmentations']}개")
    print("\n클래스별 증강 필요량:")
    
    for class_name in analysis_result['classes']:
        current = analysis_result['samples_per_class'][class_name]
        needed = augmentation_plan['augmentation_needs'][class_name]
        if needed > 0:
            print(f"  {class_name}: {current}개 → {current + needed}개 (+{needed}개 증강)")
        else:
            print(f"  {class_name}: {current}개 (증강 불필요)") 