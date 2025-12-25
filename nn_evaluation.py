import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, 
    accuracy_score, precision_score, recall_score, f1_score
)
import torch
from torch.utils.data import DataLoader
from typing import Dict, Callable
from sklearn.model_selection import StratifiedKFold

from models_evaluate import plot_confusion_matrix_and_roc



def cross_validate(
    model,
    train_indices,
    labels: np.ndarray,
    cv: StratifiedKFold,
    dataset_class,
    collate_fn: Callable,
    device,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Выполняет k-fold кросс-валидацию с твоим TweetDataset и collate_fn.
    
    Args:
        model: Инициализированная модель (BiGRUClassifier)
        train_indices: Список индексов текстов
        labels: Массив меток
        cv: StratifiedKFold объект
        dataset_class: TweetDataset класс
        collate_fn: Функция для обработки батча
        device: Устройство (cpu/cuda)
        epochs: Количество эпох
        batch_size: Размер батча
        lr: Коэффициент обучения
        weight_decay: L2 регуляризация
        verbose: Выводить ли логи
        
    Returns:
        Dict[str, Dict[str, float]]: Результаты для каждого фолда и агреггированные
    """
    import torch.nn as nn
    import torch.optim as optim
    import copy
    
    fold_results = []
    all_fold_preds = []
    all_fold_trues = []
    all_fold_probs = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(train_indices, labels), 1):
        if verbose:
            print(f"\n{'='*50}")
            print(f"FOLD {fold}/{cv.get_n_splits()}")
            print(f"{'='*50}")
        
        # Разбиваем индексы и метки
        X_train = [train_indices[i] for i in train_idx]
        X_val = [train_indices[i] for i in val_idx]
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        
        # Создаём датасеты и лоадеры
        train_dataset = dataset_class(X_train, y_train)
        val_dataset = dataset_class(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                  shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
        
        # Переинициализируем модель для каждого фолда
        fold_model = copy.deepcopy(model).to(device)
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(fold_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=1, factor=0.5
        )
        
        # Обучение
        best_f1 = 0
        best_preds = None
        best_trues = None
        best_probs = None
        
        for epoch in range(1, epochs + 1):
            # Train epoch
            model.train()
            total_loss = 0
            for texts, epoch_labels in train_loader:
                texts, epoch_labels = texts.to(device), epoch_labels.to(device).float()
                optimizer.zero_grad()
                outputs = model(texts).squeeze()
                loss = criterion(outputs, epoch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Evaluate
            model.eval()
            preds = []
            trues = []
            probs = []
            with torch.no_grad():
                for texts, epoch_labels in val_loader:
                    texts = texts.to(device)
                    outputs = model(texts).squeeze()
                    probabilities = torch.sigmoid(outputs)
                    pred = probabilities > 0.5
                    preds.extend(pred.cpu().numpy())
                    probs.extend(probabilities.cpu().numpy())
                    trues.extend(epoch_labels.numpy())
            
            preds = np.array(preds)
            trues = np.array(trues)
            probs = np.array(probs)
            
            fold_f1 = f1_score(trues, preds)
            scheduler.step(fold_f1)
            
            if verbose:
                print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Val F1: {fold_f1:.5f}")
            
            if fold_f1 > best_f1:
                best_f1 = fold_f1
                best_preds = preds
                best_trues = trues
                best_probs = probs
        
        all_fold_preds.extend(best_preds)
        all_fold_trues.extend(best_trues)
        all_fold_probs.extend(best_probs)
        
        fold_results.append({
            'fold': fold,
            'best_f1': best_f1,
            'preds': best_preds,
            'trues': best_trues,
            'probs': best_probs
        })
    
    all_fold_preds = np.array(all_fold_preds)
    all_fold_trues = np.array(all_fold_trues)
    all_fold_probs = np.array(all_fold_probs)
    
    if verbose:
        print(f"\n{'='*60}")
        print("AGGREGATED RESULTS (All Folds)")
        print(f"{'='*60}\n")
    
    plot_confusion_matrix_and_roc(all_fold_preds, all_fold_trues, all_fold_probs)
    
    results: Dict[str, Dict[str, float]] = {}
    
    for fold_result in fold_results:
        fold_num = fold_result['fold']
        fold_preds = fold_result['preds']
        fold_trues = fold_result['trues']
        fold_probs = fold_result['probs']
        
        accuracy = accuracy_score(fold_trues, fold_preds)
        precision = precision_score(fold_trues, fold_preds)
        recall = recall_score(fold_trues, fold_preds)
        f1 = f1_score(fold_trues, fold_preds)
        fpr, tpr, _ = roc_curve(fold_trues, fold_probs)
        roc_auc = auc(fpr, tpr)
        
        results[f'fold_{fold_num}'] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc)
        }
    
    accuracy = accuracy_score(all_fold_trues, all_fold_preds)
    precision = precision_score(all_fold_trues, all_fold_preds)
    recall = recall_score(all_fold_trues, all_fold_preds)
    f1 = f1_score(all_fold_trues, all_fold_preds)
    fpr, tpr, _ = roc_curve(all_fold_trues, all_fold_probs)
    roc_auc = auc(fpr, tpr)
    
    results['aggregated'] = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc)
    }
    
    # Выводим результаты
    if verbose:
        print("\nDetailed Results:\n")
        for fold_name, metrics in results.items():
            print(f"{fold_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name:12s}: {value:.5f}")
            print()
    
    return results
