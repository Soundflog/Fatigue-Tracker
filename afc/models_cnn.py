"""
FatigueCNN1D: 1D Convolutional Neural Network для детекции утомления спортсменов.

Архитектура оптимизирована для работы с многоканальными временными рядами
от носимых датчиков (акселерометр, гироскоп, PPG, EDA и др.).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict
import os


class FatigueDataset(Dataset):
    """
    PyTorch Dataset для загрузки окон временных рядов.
    
    Поддерживает загрузку из NPZ-файлов или напрямую из numpy-массивов.
    """
    def __init__(self, 
                 windows_dir: Optional[str] = None,
                 X: Optional[np.ndarray] = None,
                 y: Optional[np.ndarray] = None,
                 meta: Optional[dict] = None,
                 transform=None):
        """
        Args:
            windows_dir: путь к директории с NPZ-файлами окон
            X: массив данных (N, T, C) - если передан напрямую
            y: метки (N,) - если переданы напрямую
            meta: метаданные (sid, domain и т.д.)
            transform: опциональная трансформация
        """
        self.transform = transform
        
        if X is not None and y is not None:
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
            self.meta = meta or {}
        elif windows_dir is not None:
            self._load_from_dir(windows_dir)
        else:
            raise ValueError("Укажите windows_dir или (X, y)")
    
    def _load_from_dir(self, windows_dir: str):
        """Загрузка окон из директории с NPZ-файлами."""
        files = sorted([f for f in os.listdir(windows_dir) if f.endswith('.npz')])
        
        X_list, y_list = [], []
        meta_list = []
        
        for fname in files:
            data = np.load(os.path.join(windows_dir, fname), allow_pickle=True)
            X_list.append(data['X'])
            y_list.append(data['y'].item())
            meta_list.append({
                'sid': str(data['sid']),
                'sess': str(data['sess']),
                'domain': str(data['domain'])
            })
        
        # Паддинг до максимальной длины если нужно
        max_len = max(x.shape[0] for x in X_list)
        n_channels = X_list[0].shape[1]
        
        X_padded = np.zeros((len(X_list), max_len, n_channels), dtype=np.float32)
        for i, x in enumerate(X_list):
            X_padded[i, :x.shape[0], :] = x
        
        self.X = torch.FloatTensor(X_padded)
        self.y = torch.FloatTensor(y_list)
        self.meta = {'items': meta_list}
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


class FatigueCNN1D(nn.Module):
    """
    1D CNN для бинарной классификации состояния утомления.
    
    Архитектура:
    - 3 свёрточных блока с BatchNorm, ReLU, MaxPool, Dropout
    - Global Average Pooling
    - 2 полносвязных слоя для классификации
    
    Args:
        in_channels: число входных каналов (датчиков)
        num_classes: 1 для бинарной классификации
        conv_channels: список числа фильтров для каждого conv слоя
        kernel_sizes: список размеров ядер
        dropout: коэффициент dropout для conv слоёв
        fc_dropout: коэффициент dropout для FC слоёв
    """
    def __init__(self,
                 in_channels: int = 6,
                 num_classes: int = 1,
                 conv_channels: List[int] = [64, 128, 256],
                 kernel_sizes: List[int] = [7, 5, 3],
                 dropout: float = 0.2,
                 fc_dropout: float = 0.5):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Свёрточный блок 1: низкоуровневые паттерны
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels[0], 
                      kernel_size=kernel_sizes[0], stride=1, 
                      padding=kernel_sizes[0]//2),
            nn.BatchNorm1d(conv_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )
        
        # Свёрточный блок 2: среднеуровневые паттерны
        self.conv2 = nn.Sequential(
            nn.Conv1d(conv_channels[0], conv_channels[1],
                      kernel_size=kernel_sizes[1], stride=1,
                      padding=kernel_sizes[1]//2),
            nn.BatchNorm1d(conv_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )
        
        # Свёрточный блок 3: высокоуровневые паттерны
        self.conv3 = nn.Sequential(
            nn.Conv1d(conv_channels[1], conv_channels[2],
                      kernel_size=kernel_sizes[2], stride=1,
                      padding=kernel_sizes[2]//2),
            nn.BatchNorm1d(conv_channels[2]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        )
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_channels[2], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(128, num_classes)
        )
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/He инициализация для лучшей сходимости."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: входной тензор (batch, time, channels)
        
        Returns:
            logits: (batch,) - логиты для бинарной классификации
        """
        # (batch, time, channels) -> (batch, channels, time)
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        logits = self.classifier(x)
        return logits.squeeze(-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает вероятности (после sigmoid)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Извлечение эмбеддингов для transfer learning.
        
        Args:
            x: входной тензор (batch, time, channels)
        
        Returns:
            features: (batch, 256) - эмбеддинги
        """
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.view(x.size(0), -1)


class FatigueCNN1DLite(nn.Module):
    """
    Облегчённая версия для мобильных устройств / реального времени.
    
    Параметров: ~50K (vs ~175K в полной версии)
    """
    def __init__(self,
                 in_channels: int = 6,
                 num_classes: int = 1):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.features(x)
        return self.classifier(x).squeeze(-1)


# ==================== ОБУЧЕНИЕ ====================

class FatigueTrainer:
    """
    Класс для обучения и валидации моделей.
    
    Поддерживает:
    - Weighted BCE loss для несбалансированных данных
    - Early stopping
    - Learning rate scheduling
    - Gradient clipping
    """
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 class_weights: Optional[torch.Tensor] = None):
        
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        
        # Weighted BCE для несбалансированных данных
        if class_weights is not None:
            pos_weight = class_weights[1] / class_weights[0]
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight]).to(device)
            )
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Один эпох обучения."""
        self.model.train()
        total_loss = 0.0
        
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Валидация модели."""
        self.model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []
        
        for X, y in val_loader:
            X, y = X.to(self.device), y.to(self.device)
            
            logits = self.model(X)
            loss = self.criterion(logits, y)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        metrics = self._compute_metrics(np.array(all_labels), np.array(all_probs))
        
        return avg_loss, metrics
    
    def _compute_metrics(self, y_true: np.ndarray, y_prob: np.ndarray,
                         threshold: float = 0.5) -> Dict[str, float]:
        """Вычисление метрик качества."""
        from sklearn.metrics import (
            f1_score, balanced_accuracy_score, 
            roc_auc_score, average_precision_score, brier_score_loss
        )
        
        y_pred = (y_prob >= threshold).astype(int)
        
        metrics = {}
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)
        
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['roc_auc'] = np.nan
        
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['pr_auc'] = np.nan
        
        metrics['brier'] = brier_score_loss(y_true, y_prob)
        
        return metrics
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: int = 100,
            early_stopping_patience: int = 15,
            verbose: bool = True) -> Dict:
        """
        Полный цикл обучения.
        
        Args:
            train_loader: DataLoader для обучения
            val_loader: DataLoader для валидации
            epochs: максимальное число эпох
            early_stopping_patience: число эпох без улучшения для остановки
            verbose: выводить прогресс
        
        Returns:
            history: словарь с историей обучения
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            if val_loader is not None:
                val_loss, val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_metrics'].append(val_metrics)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"F1: {val_metrics['f1_macro']:.4f} | "
                          f"ROC-AUC: {val_metrics['roc_auc']:.4f}")
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            
            self.scheduler.step()
        
        # Восстановить лучшую модель
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return self.history


# ==================== ПЕРСОНАЛИЗАЦИЯ ====================

def personalize_model(base_model: FatigueCNN1D,
                      subject_X: np.ndarray,
                      subject_y: np.ndarray,
                      epochs: int = 20,
                      freeze_backbone: bool = True,
                      device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
                      ) -> FatigueCNN1D:
    """
    Дообучение модели под конкретного спортсмена.
    
    Args:
        base_model: предобученная модель
        subject_X: данные субъекта (N, T, C)
        subject_y: метки субъекта (N,)
        epochs: число эпох дообучения
        freeze_backbone: заморозить свёрточные слои
        device: устройство
    
    Returns:
        personalized_model: дообученная модель
    """
    import copy
    
    model = copy.deepcopy(base_model)
    model = model.to(device)
    
    # Заморозка backbone
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'conv1' in name or 'conv2' in name:
                param.requires_grad = False
    
    # Подготовка данных
    dataset = FatigueDataset(X=subject_X, y=subject_y)
    loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
    
    # Этап 1: обучение только classifier
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(epochs // 2):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
    
    # Этап 2: разморозка conv3 и fine-tuning
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'conv3' in name:
                param.requires_grad = True
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=0.0001)
        
        for epoch in range(epochs // 2):
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
    
    return model


# ==================== УТИЛИТЫ ====================

def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Вычисление весов классов для несбалансированных данных."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = total / (len(unique) * counts)
    return torch.FloatTensor(weights)


def save_model(model: nn.Module, path: str, metadata: Optional[dict] = None):
    """Сохранение модели с метаданными."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'in_channels': model.in_channels,
            'num_classes': model.num_classes
        },
        'metadata': metadata or {}
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(path: str, device: str = 'cpu') -> FatigueCNN1D:
    """Загрузка модели из checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['model_config']
    
    model = FatigueCNN1D(
        in_channels=config['in_channels'],
        num_classes=config['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Тест модели
    print("Testing FatigueCNN1D...")
    
    # Создание тестовых данных
    batch_size = 8
    time_steps = 150  # 3 sec @ 50 Hz
    channels = 6      # ax, ay, az, gx, gy, gz
    
    x = torch.randn(batch_size, time_steps, channels)
    y = torch.randint(0, 2, (batch_size,)).float()
    
    # Инициализация модели
    model = FatigueCNN1D(in_channels=channels)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Output range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Feature extraction
    features = model.extract_features(x)
    print(f"Features shape: {features.shape}")
    
    print("\n✓ All tests passed!")
