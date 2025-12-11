import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from utils import analyze_dataset_structure, print_dataset_analysis, calculate_augmentation_needs, print_augmentation_plan
import random
import numpy as np
from tqdm import tqdm


class DoubleSkinDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        """
        Args:
            root_dirs (list): 데이터셋 루트 디렉토리 리스트
            transform (callable, optional): 이미지 변환을 위한 transform
        """
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
            
        self.root_dirs = root_dirs
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
        # --- 여기서부터 데이터 증강 기법 추가 (증강하여 학습하지 않으려면 아래 세 줄은 주석처리 하기)---
        transforms.RandomHorizontalFlip(p=0.5), # 50% 확률로 이미지를 좌우로 뒤집습니다.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 밝기, 대비, 채도를 랜덤하게 변경합니다.
        transforms.RandomRotation(10), # 이미지를 -10도에서 10도 사이로 랜덤하게 회전시킵니다.
        # --- 여기까지 데이터 증강 ---
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 기본 transform이 없는 경우를 위한 안전장치
        self.fallback_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # 데이터셋 구조 분석
        analysis_result = analyze_dataset_structure(root_dirs)
        
        # 분석 결과를 인스턴스 변수로 저장
        self.classes = analysis_result['classes']
        self.class_to_idx = analysis_result['class_to_idx']
        self.samples_per_class = analysis_result['samples_per_class']
        self.samples_per_cls = analysis_result['samples_per_cls']
        
        # 이미지 파일 경로와 레이블 수집
        self.images = []
        self.labels = []
        
        print("📁 데이터셋 로딩 중...")
        total_files = 0
        for root_dir in root_dirs:
            for class_name in self.classes:
                class_dir = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                total_files += len([f for f in os.listdir(class_dir) 
                                  if f.endswith(('.jpg', '.jpeg', '.png')) and f != '.DS_Store'])
        
        with tqdm(total=total_files, desc="이미지 파일 검증 및 로딩", unit="파일") as pbar:
            for root_dir in root_dirs:
                for class_name in self.classes:
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
                                self.images.append(img_path)
                                self.labels.append(self.class_to_idx[class_name])
                            except Exception as e:
                                print(f"손상된 이미지 파일 무시: {img_path} - {e}")
                            pbar.update(1)
        
        # 분석 결과 출력
        print_dataset_analysis(analysis_result)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 이미지 로드 및 변환
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, os.path.basename(img_path)
        except Exception as e:
            print(f"이미지 로드 실패: {img_path} - {e}")
            # 기본 이미지 반환 (fallback transform 사용)
            fallback_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                return self.fallback_transform(fallback_image), label, os.path.basename(img_path)
            else:
                # transform이 없는 경우에도 torch.Tensor 형태로 반환
                return self.fallback_transform(fallback_image), label, os.path.basename(img_path)

class AugmentedSkinDataset(Dataset):
    """
    데이터 증강이 적용된 피부 질환 데이터셋
    클래스 불균형을 해결하기 위해 필요한 만큼 이미지를 증강합니다.
    """
    
    def __init__(self, root_dirs, transform=None, target_samples_per_class=None, 
                 augmentation_methods=['horizontal_flip', 'rotation', 'color_jitter'],
                 train_indices=None):
        """
        Args:
            root_dirs (list): 데이터셋 루트 디렉토리 리스트
            transform (callable, optional): 기본 이미지 변환
            target_samples_per_class (int, optional): 목표 샘플 수. None이면 가장 많은 클래스의 샘플 수로 설정
            augmentation_methods (list): 사용할 증강 방법들
            train_indices (list, optional): train에 사용할 인덱스 리스트. None이면 전체 데이터셋 사용
        """
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
            
        self.root_dirs = root_dirs
        self.augmentation_methods = augmentation_methods
        self.train_indices = train_indices
        
        # 기본 transform 설정
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 증강용 transform 설정
        self.augmentation_transforms = self._create_augmentation_transforms()
        
        # 데이터셋 구조 분석
        analysis_result = analyze_dataset_structure(root_dirs)
        
        # 분석 결과를 인스턴스 변수로 저장
        self.classes = analysis_result['classes']
        self.class_to_idx = analysis_result['class_to_idx']
        self.samples_per_class = analysis_result['samples_per_class']
        self.samples_per_cls = analysis_result['samples_per_cls']
        
        # 원본 이미지 파일 경로와 레이블 수집
        self.original_images = []
        self.original_labels = []
        self.class_image_indices = {cls_name: [] for cls_name in self.classes}
        
        print("📁 원본 데이터셋 로딩 중...")
        total_files = 0
        for root_dir in root_dirs:
            for class_name in self.classes:
                class_dir = os.path.join(root_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                total_files += len([f for f in os.listdir(class_dir) 
                                  if f.endswith(('.jpg', '.jpeg', '.png')) and f != '.DS_Store'])
        
        with tqdm(total=total_files, desc="원본 이미지 파일 검증 및 로딩", unit="파일") as pbar:
            for root_dir in root_dirs:
                for class_name in self.classes:
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
                                self.original_images.append(img_path)
                                self.original_labels.append(self.class_to_idx[class_name])
                                self.class_image_indices[class_name].append(len(self.original_images) - 1)
                            except Exception as e:
                                print(f"손상된 이미지 파일 무시: {img_path} - {e}")
                            pbar.update(1)
        
        # Train 인덱스가 지정된 경우, 해당 부분만 사용하여 증강 계획 계산
        if self.train_indices is not None:
            # Train 부분의 클래스별 샘플 수 계산
            train_samples_per_class = {cls_name: 0 for cls_name in self.classes}
            for idx in self.train_indices:
                if idx < len(self.original_labels):
                    class_name = self.classes[self.original_labels[idx]]
                    train_samples_per_class[class_name] += 1
            
            # Train 부분에 대한 증강 계획 계산
            train_analysis_result = {
                'classes': self.classes,
                'class_to_idx': self.class_to_idx,
                'samples_per_class': train_samples_per_class,
                'samples_per_cls': [train_samples_per_class[cls_name] for cls_name in self.classes],
                'total_images': sum(train_samples_per_class.values()),
                'dataset_images': analysis_result['dataset_images']
            }
            
            augmentation_plan = calculate_augmentation_needs(train_analysis_result, target_samples_per_class)
            self.augmentation_needs = augmentation_plan['augmentation_needs']
            self.target_samples = augmentation_plan['target_samples']
            
            # 증강 계획 출력
            print_augmentation_plan(augmentation_plan, train_analysis_result)
        else:
            # 전체 데이터셋에 대한 증강 계획 계산
            augmentation_plan = calculate_augmentation_needs(analysis_result, target_samples_per_class)
            self.augmentation_needs = augmentation_plan['augmentation_needs']
            self.target_samples = augmentation_plan['target_samples']
            
            # 증강 계획 출력
            print_augmentation_plan(augmentation_plan, analysis_result)
        
        # 증강된 데이터셋 구성
        self._build_augmented_dataset()
        
        print(f"증강 후 총 {len(self.images)}개의 이미지가 로드되었습니다.")
    
    def _create_augmentation_transforms(self):
        """증강 방법별 transform 생성"""
        transforms_dict = {}
        
        if 'horizontal_flip' in self.augmentation_methods:
            transforms_dict['horizontal_flip'] = transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0),  # 항상 반전
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        if 'rotation' in self.augmentation_methods:
            transforms_dict['rotation'] = transforms.Compose([
                transforms.RandomRotation(degrees=15),  # ±15도 회전
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        if 'color_jitter' in self.augmentation_methods:
            transforms_dict['color_jitter'] = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        return transforms_dict
    
    def _build_augmented_dataset(self):
        """증강된 데이터셋 구성"""
        self.images = []
        self.labels = []
        self.is_augmented = []  # 원본인지 증강된 것인지 표시
        self.original_indices = []  # 원본 이미지 인덱스
        self.augmentation_types = []  # 증강 방법
        
        # 사용할 이미지 인덱스 결정
        if self.train_indices is not None:
            # Train 인덱스만 사용
            use_indices = self.train_indices
        else:
            # 전체 데이터셋 사용
            use_indices = list(range(len(self.original_images)))
        
        # 원본 이미지 추가 (사용할 인덱스만)
        for idx in use_indices:
            if idx < len(self.original_images):
                img_path = self.original_images[idx]
                label = self.original_labels[idx]
                self.images.append(img_path)
                self.labels.append(label)
                self.is_augmented.append(False)
                self.original_indices.append(idx)
                self.augmentation_types.append('original')
        
        # 증강된 이미지 추가
        total_augmentations = sum(self.augmentation_needs.values())
        if total_augmentations > 0:
            print("🔄 데이터 증강 중...")
            with tqdm(total=total_augmentations, desc="증강된 이미지 생성", unit="이미지") as pbar:
                for class_name, needed_count in self.augmentation_needs.items():
                    if needed_count > 0:
                        class_idx = self.class_to_idx[class_name]
                        
                        # 해당 클래스의 train 이미지 인덱스 찾기
                        class_train_indices = []
                        for idx in use_indices:
                            if idx < len(self.original_labels) and self.original_labels[idx] == class_idx:
                                class_train_indices.append(idx)
                        
                        if len(class_train_indices) == 0:
                            print(f"경고: {class_name} 클래스에 train 이미지가 없습니다.")
                            continue
                        
                        # 필요한 만큼 증강
                        for i in range(needed_count):
                            # 원본 이미지 랜덤 선택 (train 인덱스 중에서)
                            original_idx = random.choice(class_train_indices)
                            original_img_path = self.original_images[original_idx]
                            
                            # 증강 방법 랜덤 선택
                            aug_method = random.choice(list(self.augmentation_transforms.keys()))
                            
                            # 증강된 이미지 정보 저장 (실제 이미지는 __getitem__에서 생성)
                            self.images.append(original_img_path)  # 원본 경로 저장
                            self.labels.append(class_idx)
                            self.is_augmented.append(True)
                            self.original_indices.append(original_idx)
                            self.augmentation_types.append(aug_method)
                            pbar.update(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        is_augmented = self.is_augmented[idx]
        aug_type = self.augmentation_types[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if is_augmented:
                # 증강된 이미지인 경우
                if aug_type in self.augmentation_transforms:
                    image = self.augmentation_transforms[aug_type](image)
                else:
                    # 기본 transform 적용
                    image = self.transform(image)
            else:
                # 원본 이미지인 경우
                image = self.transform(image)
            
            # 파일명에 증강 정보 추가
            filename = os.path.basename(img_path)
            if is_augmented:
                filename = f"{filename}_aug_{aug_type}"
            
            return image, label, filename
            
        except Exception as e:
            print(f"이미지 로드 실패: {img_path} - {e}")
            # 기본 이미지 반환
            fallback_image = Image.new('RGB', (224, 224), color='black')
            return self.transform(fallback_image), label, os.path.basename(img_path) 