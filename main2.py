import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models
import wandb
from dataset_double import DoubleSkinDataset #, AugmentedSkinDataset
from model import Model
import argparse
from datetime import datetime
import numpy as np
import random
from sklearn.model_selection import StratifiedShuffleSplit
import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirs', nargs='+', required=True, help='데이터셋 디렉토리 경로들 (여러 개 가능)')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='체크포인트 저장 경로')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=100, help='학습 에포크 수')
    parser.add_argument('--lr', type=float, default=0.0005, help='초기 학습률')
    parser.add_argument('--num_workers', type=int, default=4, help='데이터 로더 워커 수')
    parser.add_argument('--resume', type=str, default=None, help='체크포인트 경로')
    
    # wandb 관련 파라미터
    parser.add_argument('--wandb_project', type=str, default='skin-classifier', help='wandb 프로젝트 이름')
    parser.add_argument('--wandb_name', type=str, default=None, help='wandb 실험 이름')
    parser.add_argument('--wandb_dir', type=str, default=None, help='wandb 로그 저장 디렉토리 (기본값: 현재 디렉토리/wandb_logs)')
    parser.add_argument('--wandb_id', type=str, default=None, help='wandb run id (resume 시 사용)')
    parser.add_argument('--wandb_resume', type=str, default='never', choices=['never', 'allow', 'must'], help='wandb resume 옵션 (never/allow/must)')
    
    
    # 규제 관련 파라미터
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='가중치 감소 (L2 규제) 강도') # 1e-7, 1e-4 ~ 1e-3
    parser.add_argument('--dropout_prob', type=float, default=0.3, help='드롭아웃 확률') # 0.2, 0.3 ~ 0.4
    
    # 손실 함수 관련 파라미터
    parser.add_argument('--train_loss', type=str, default='focal', 
                       choices=['focal', 'cb', 'label_smoothing', 'cross_entropy'],
                       help='학습에 사용할 손실 함수')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal Loss와 CB Loss의 gamma 값')
    parser.add_argument('--beta', type=float, default=0.999, help='CB Loss의 beta 값')
    parser.add_argument('--alpha', type=float, default=1.0, help='Focal Loss의 alpha 값')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label Smoothing의 smoothing 값')
    
    # 데이터 증강 관련 파라미터
    parser.add_argument('--use_augmentation', action='store_true', help='데이터 증강 사용 여부')
    parser.add_argument('--target_samples_per_class', type=int, default=None, 
                       help='클래스별 목표 샘플 수 (None이면 가장 많은 클래스의 샘플 수로 설정)')
    parser.add_argument('--augmentation_methods', nargs='+', 
                       default=['horizontal_flip', 'rotation', 'color_jitter'],
                       choices=['horizontal_flip', 'rotation', 'color_jitter'],
                       help='사용할 증강 방법들')
    
    # 실험 재현성 관련 파라미터
    parser.add_argument('--reproducible', action='store_true', default=True,
                       help='실험 재현성 보장 (기본값: True)')
    parser.add_argument('--no_reproducible', action='store_true', default=False,
                       help='실험 재현성 비활성화 (완전 랜덤)')
    
    return parser.parse_args()

def set_seed(seed):
    """랜덤 시드 설정 - 실험 재현성을 보장하기 위한 모든 랜덤 요소 제어"""
    # Python 표준 라이브러리 랜덤 시드 설정
    random.seed(seed)
    # NumPy 랜덤 시드 설정 (데이터셋 분할, 샘플링 등에 사용)
    np.random.seed(seed)
    # PyTorch CPU 연산 랜덤 시드 설정
    torch.manual_seed(seed)
    
    # cuDNN 재현성 설정
    # deterministic=True: cuDNN 연산을 결정적으로 만들어 같은 입력에 대해 항상 같은 결과 보장
    # benchmark=False: cuDNN이 자동으로 최적 알고리즘을 찾는 기능을 비활성화하여 고정된 알고리즘 사용
    # 이 두 설정은 성능보다 재현성을 우선시하는 설정 (연구/실험에서 필수)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # CUDA GPU가 사용 가능한 경우 GPU 랜덤 시드 설정
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)      # 현재 GPU 랜덤 시드
        torch.cuda.manual_seed_all(seed)  # 모든 GPU 랜덤 시드 (멀티 GPU 환경)
    
    # Apple Silicon MPS가 사용 가능한 경우 MPS 랜덤 시드 설정
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def stratified_split_dataset(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
    """클래스 균형을 고려한 데이터셋 분할"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "비율의 합이 1이어야 합니다."
    
    labels = [dataset.labels[i] for i in range(len(dataset))]
    labels = np.array(labels)
    
    # 먼저 train과 나머지로 분할
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=random_state)
    train_idx, temp_idx = next(sss1.split(range(len(dataset)), labels))
    
    # 나머지를 val과 test로 분할
    temp_labels = labels[temp_idx]
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - val_test_ratio), random_state=random_state)
    val_idx, test_idx = next(sss2.split(range(len(temp_idx)), temp_labels))
    
    # 인덱스 매핑
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    return train_dataset, val_dataset, test_dataset

def get_device():
    """안전한 디바이스 설정"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        # MPS에서 일부 연산이 지원되지 않을 수 있으므로 CPU fallback 고려
        try:
            # 간단한 테스트로 MPS 호환성 확인
            test_tensor = torch.randn(2, 2, device='mps')
            _ = torch.softmax(test_tensor, dim=1)
            return torch.device('mps')
        except:
            print("MPS 디바이스에서 일부 연산이 지원되지 않아 CPU를 사용합니다.")
            return torch.device('cpu')
    else:
        return torch.device('cpu')

def main():
    args = parse_args()
    
    # 디바이스 설정
    device = get_device()
    print(f"Using device: {device}")
    
    # 랜덤 시드 설정
    if args.no_reproducible:
        print("실험 재현성 비활성화: 완전 랜덤 모드")
        # 랜덤 시드 설정하지 않음
    else:
        print("실험 재현성 보장: 고정된 랜덤 시드 사용")
        set_seed(42)
    
    # 현재 시간으로 디렉토리 생성
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"{args.output_dir}_{current_time}"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 실험 ID 생성 (체크포인트 재시작 시에도 동일한 ID 사용)
    experiment_id = current_time
    if args.resume:
        # 체크포인트에서 재시작하는 경우, 체크포인트 파일명에서 실험 ID 추출
        try:
            checkpoint_dir = os.path.dirname(args.resume)
            if os.path.basename(checkpoint_dir).startswith('checkpoints_'):
                experiment_id = os.path.basename(checkpoint_dir).replace('checkpoints_', '')
        except:
            pass  # 실패하면 현재 시간 사용
    
    # wandb 디렉토리 경로 설정
    if args.wandb_dir:
        wandb_dir = args.wandb_dir
    else:
        wandb_dir = os.path.join(os.getcwd(), 'wandb')
    
    os.environ['WANDB_DIR'] = wandb_dir
    os.makedirs(wandb_dir, exist_ok=True)
    print(f"wandb 로그 디렉토리: {wandb_dir}")
    
    # wandb 초기화
    try:
        wandb_config = {
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "architecture": "ResNet50",
            "weight_decay": args.weight_decay,
            "dropout_prob": args.dropout_prob,
            "gamma": args.gamma,
            "use_augmentation": args.use_augmentation
        }
        
        if args.use_augmentation:
            wandb_config.update({
                "augmentation_methods": args.augmentation_methods,
                "target_samples_per_class": args.target_samples_per_class
            })
        
        wandb_config.update({
            "reproducible": not args.no_reproducible
        })
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=wandb_config,
            id=args.wandb_id,
            resume=args.wandb_resume,
            dir=wandb_dir
        )
        use_wandb = True
        print("wandb 초기화 성공")
    except Exception as e:
        print(f"wandb 초기화 실패: {e}")
        print("wandb 없이 실행합니다.")
        use_wandb = False
    
    # 데이터셋 로드
    dataset = DoubleSkinDataset(args.data_dirs)
    
    # 클래스 균형을 고려한 데이터셋 분할
    train_dataset, val_dataset, test_dataset = stratified_split_dataset(
        dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42
    )
    
    # train 데이터셋의 클래스 분포 계산 함수
    def get_train_class_counts(dataset, num_classes):
        counts = [0] * num_classes
        # Subset인 경우
        if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            for idx in dataset.indices:
                label = dataset.dataset.labels[idx]
                counts[label] += 1
        # AugmentedSkinDataset인 경우
        elif hasattr(dataset, 'labels'):
            for label in dataset.labels:
                counts[label] += 1
        else:
            for i in range(len(dataset)):
                _, label, _ = dataset[i]
                counts[label] += 1
        return counts

    num_classes = len(dataset.classes)
    train_samples_per_cls = get_train_class_counts(train_dataset, num_classes)

    # Train 데이터셋에만 증강 적용 (옵션)
    if args.use_augmentation:
        print("\n=== 데이터 증강 적용 ===")
        print(f"증강 방법: {', '.join(args.augmentation_methods)}")
        if args.target_samples_per_class:
            print(f"목표 샘플 수: {args.target_samples_per_class}개")
        else:
            print("목표 샘플 수: 자동 설정 (가장 많은 클래스 기준)")
        
        # 증강된 train 데이터셋 생성 (train 인덱스만 사용)
        augmented_train_dataset = AugmentedSkinDataset(
            root_dirs=args.data_dirs,
            transform=dataset.transform,
            target_samples_per_class=args.target_samples_per_class,
            augmentation_methods=args.augmentation_methods,
            train_indices=train_dataset.indices  # train 인덱스만 전달
        )
        train_dataset = augmented_train_dataset
        # 증강된 데이터셋의 분포로 다시 계산
        train_samples_per_cls = get_train_class_counts(train_dataset, num_classes)
        print("증강된 train 데이터셋으로 교체 완료")
    else:
        print("\n=== 데이터 증강 미사용 ===")
        print("원본 데이터셋 그대로 사용")
    
    print(f"데이터셋 분할 완료:")
    print(f"  학습: {len(train_dataset)}개")
    print(f"  검증: {len(val_dataset)}개")
    print(f"  테스트: {len(test_dataset)}개")
    
    # 클래스별 분포 확인
    def get_class_distribution(dataset):
        class_counts = {}
        
        # AugmentedSkinDataset인 경우
        if hasattr(dataset, 'labels') and hasattr(dataset, 'classes'):
            # 직접 labels와 classes 속성에 접근
            for idx in range(len(dataset)):
                label = dataset.labels[idx]
                class_name = dataset.classes[label]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        # Subset인 경우 (원본 데이터셋이 감싸져 있음)
        elif hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            # 원본 데이터셋의 labels와 classes에 접근
            for idx in range(len(dataset)):
                label = dataset.dataset.labels[dataset.indices[idx]]
                class_name = dataset.dataset.classes[label]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        else:
            # 기타 경우: 직접 데이터를 순회하여 레이블 수집
            for idx in range(len(dataset)):
                try:
                    _, label, _ = dataset[idx]
                    # label이 숫자인 경우 클래스명으로 변환
                    if isinstance(label, int):
                        # 원본 데이터셋의 클래스 정보 사용
                        class_name = dataset.classes[label] if hasattr(dataset, 'classes') else f"class_{label}"
                    else:
                        class_name = str(label)
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                except Exception as e:
                    print(f"레이블 추출 실패 (인덱스 {idx}): {e}")
                    continue
        
        return class_counts
    
    print("\n클래스별 분포:")
    print("  학습:", get_class_distribution(train_dataset))
    print("  검증:", get_class_distribution(val_dataset))
    print("  테스트:", get_class_distribution(test_dataset))
    
    # 데이터 로더 생성
    # MPS 디바이스에서는 num_workers를 0으로 설정 (호환성 문제)
    safe_num_workers = 0 if device.type == 'mps' else args.num_workers
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=safe_num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                          shuffle=False, num_workers=safe_num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=safe_num_workers)
    
    # 모델 초기화
    model = Model(num_classes, dropout_prob=args.dropout_prob, 
                 samples_per_cls=train_samples_per_cls).to(device)
    
    # 옵티마이저 설정
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 학습률 스케줄러 추가
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # wandb에 모델 구조 기록
    if use_wandb:
        wandb.watch(model)
    
    # 체크포인트 로드
    start_epoch = 0
    best_acc = 0
    best_epoch = -1  # 최고 성능 모델이 저장된 에폭 추적
    
    if args.resume:
        try:
            model, optimizer, start_epoch, best_acc = model.load_checkpoint(args.resume, optimizer, scheduler, device)
            print(f"체크포인트 로드 완료: 에포크 {start_epoch}부터 시작")
        except Exception as e:
            print(f"체크포인트 로드 실패: {e}")
            print("처음부터 학습을 시작합니다.")
    
    # 학습 루프
    for epoch in range(start_epoch, args.epochs):
        # 학습
        train_loss, train_acc = model.train_epoch(
            train_loader, optimizer, device,
            loss_type=args.train_loss,
            gamma=args.gamma if args.train_loss in ['focal', 'cb'] else None,
            beta=args.beta if args.train_loss == 'cb' else None,
            alpha=args.alpha if args.train_loss == 'focal' else None,
            smoothing=args.smoothing if args.train_loss == 'label_smoothing' else None,
            epoch=epoch if not args.no_reproducible else None
        )
        
        # 검증
        val_loss, val_acc, val_bal_acc, val_macro_f1 = model.validate(
            val_loader, device, log_dir=args.output_dir, epoch=epoch, experiment_id=experiment_id
        )
        
        # wandb에 기록
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_balanced_acc": val_bal_acc,
                "val_macro_f1": val_macro_f1,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
        
        print(f'Epoch: {epoch} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.4f} | '
              f'Val Balanced Acc: {val_bal_acc:.4f} | '
              f'Val Macro F1: {val_macro_f1:.4f}')
        
        # 최고 성능 모델 저장 (val_balanced_acc 기준)
        if val_bal_acc > best_acc:
            best_acc = val_bal_acc
            best_epoch = epoch  # 최고 성능 모델이 저장된 에폭 기록
            best_model_path = os.path.join(args.output_dir, f'best_model_epoch_{epoch}.pth')
            print(f"🎯 새로운 최고 성능 달성! 에포크 {epoch}에서 {os.path.basename(best_model_path)} 저장 (검증 Balanced Accuracy: {val_bal_acc:.4f})")
            model.save_checkpoint(optimizer, epoch, best_acc, best_model_path, scheduler)

            # 최근 3개만 남기고 이전 모델 삭제
            model_files = sorted(
                glob.glob(os.path.join(args.output_dir, "best_model_epoch_*.pth")),
                key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            if len(model_files) > 3:
                for old_file in model_files[:-3]:
                    try:
                        os.remove(old_file)
                        print(f"🗑️ 이전 모델 삭제: {os.path.basename(old_file)}")
                    except Exception as e:
                        print(f"⚠️ 모델 삭제 실패: {old_file} ({e})")
            
        # 매 에포크 종료 시 스케줄러 업데이트
        scheduler.step()
    
    # 최종 테스트
    print("\n최종 테스트 시작...")
    test_loss, test_acc, test_bal_acc, test_macro_f1 = model.test(test_loader, device, args.output_dir, experiment_id)
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test Balanced Acc: {test_bal_acc:.4f} | Test Macro F1: {test_macro_f1:.4f}')
    
    # 학습 완료 요약
    print(f"\n📊 학습 완료 요약:")
    if best_epoch >= 0:
        best_model_path = os.path.join(args.output_dir, f'best_model_epoch_{best_epoch}.pth')
        print(f"   - 최고 성능 모델 저장 에포크: {best_epoch}")
        print(f"   - 최고 검증 정확도: {best_acc:.4f}")
        print(f"   - 최종 테스트 정확도: {test_acc:.4f}")
        print(f"   - 모델 저장 경로: {best_model_path}")
    else:
        print(f"   - 최고 성능 모델 저장 에포크: 저장된 모델 없음")
        print(f"   - 최고 검증 정확도: {best_acc:.4f}")
        print(f"   - 최종 테스트 정확도: {test_acc:.4f}")
        print(f"   - 모델 저장 경로: 없음")
    
    # wandb에 테스트 결과 기록
    if use_wandb:
        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_balanced_acc": test_bal_acc,
            "test_macro_f1": test_macro_f1,
        })
    
    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main() 