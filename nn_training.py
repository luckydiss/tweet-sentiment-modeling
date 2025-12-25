import logging
from tqdm import tqdm
from typing import Dict, Callable
import copy
import time

import numpy as np

from sklearn.metrics import (
    roc_curve, auc, 
    accuracy_score, precision_score, recall_score, f1_score

)
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from eval_utils import plot_confusion_matrix_and_roc

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def cross_validate(
    model,
    train_indices,
    labels: np.ndarray,
    cv: StratifiedKFold,
    dataset_class,
    collate_fn: Callable,
    device,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    patience: int = 5,
    min_delta: float = 1e-4,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:

    """K-fold кросс-валидация"""
    fold_results = []
    all_fold_preds = []
    all_fold_trues = []
    all_fold_probs = []

    total_start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(cv.split(train_indices, labels), 1):

        logger.info(f"===== FOLD {fold}/{cv.get_n_splits()} =====")

        train_dataset = dataset_class([train_indices[i] for i in train_idx], labels[train_idx])
        val_dataset   = dataset_class([train_indices[i] for i in val_idx],   labels[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        fold_model = copy.deepcopy(model).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(fold_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.5)

        best_f1 = 0.0
        best_epoch = 0
        no_improve_epochs = 0
        best_state_dict = None

        best_preds = None
        best_trues = None
        best_probs = None

        epoch_iter = tqdm(range(1, epochs + 1), desc=f"Fold {fold}", leave=False)

        for epoch in epoch_iter:

            # TRAIN
            fold_model.train()
            total_loss = 0

            for texts, targets in train_loader:
                texts = texts.to(device)
                targets = targets.to(device).float()

                optimizer.zero_grad()
                outputs = fold_model(texts).squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fold_model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # VALIDATION
            fold_model.eval()
            val_preds, val_trues, val_probs = [], [], []

            with torch.no_grad():
                for texts, targets in val_loader:
                    texts = texts.to(device)

                    outputs = fold_model(texts).squeeze()
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).cpu().numpy()

                    val_preds.extend(preds)
                    val_probs.extend(probs.cpu().numpy())
                    val_trues.extend(targets.numpy())

            val_preds = np.array(val_preds)
            val_trues = np.array(val_trues)
            val_probs = np.array(val_probs)
            val_f1 = f1_score(val_trues, val_preds)

            scheduler.step(val_f1)

            # tqdm вывод метрики
            epoch_iter.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "val_f1": f"{val_f1:.4f}"
            })

            # EARLY STOPPING
            if val_f1 > best_f1 + min_delta:
                best_f1 = val_f1
                best_epoch = epoch
                no_improve_epochs = 0

                best_state_dict = copy.deepcopy(fold_model.state_dict())
                best_preds = val_preds.copy()
                best_trues = val_trues.copy()
                best_probs = val_probs.copy()

            else:
                no_improve_epochs += 1

                if no_improve_epochs >= patience:
                    logger.info(
                        f"Early stopping on epoch {epoch} — "
                        f"best F1: {best_f1:.4f} (epoch {best_epoch})"
                    )
                    fold_model.load_state_dict(best_state_dict)
                    break

        all_fold_preds.extend(best_preds)
        all_fold_trues.extend(best_trues)
        all_fold_probs.extend(best_probs)

        fold_results.append({
            'fold': fold,
            'best_f1': float(best_f1),
            'best_epoch': best_epoch,
            'preds': best_preds,
            'trues': best_trues,
            'probs': best_probs
        })

    all_fold_preds = np.array(all_fold_preds)
    all_fold_trues = np.array(all_fold_trues)
    all_fold_probs = np.array(all_fold_probs)

    logger.info("Cross-validation finished.")

    plot_confusion_matrix_and_roc(all_fold_preds, all_fold_trues, all_fold_probs)

    results = {}
    for r in fold_results:
        preds = r['preds']
        trues = r['trues']
        probs = r['probs']
        fpr, tpr, _ = roc_curve(trues, probs)

        results[f'fold_{r["fold"]}'] = {
            'accuracy':  float(accuracy_score(trues, preds)),
            'precision': float(precision_score(trues, preds)),
            'recall':    float(recall_score(trues, preds)),
            'f1':        float(f1_score(trues, preds)),
            'roc_auc':   float(auc(fpr, tpr)),
            'best_epoch': r['best_epoch']
        }

    fpr, tpr, _ = roc_curve(all_fold_trues, all_fold_probs)
    results['aggregated'] = {
        'accuracy':  float(accuracy_score(all_fold_trues, all_fold_preds)),
        'precision': float(precision_score(all_fold_trues, all_fold_preds)),
        'recall':    float(recall_score(all_fold_trues, all_fold_preds)),
        'f1':        float(f1_score(all_fold_trues, all_fold_preds)),
        'roc_auc':   float(auc(fpr, tpr)),
    }

    total_time = time.time() - total_start_time
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    return results

import logging
import copy
import numpy as np
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)

logger = logging.getLogger(__name__)

def train_fold(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    patience: int = 3,
    min_delta: float = 1e-4,
    warmup_ratio: float = 0.1,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Обучение одного fold'а с early stopping.
    """
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    total_steps = len(train_loader) * epochs
    num_warmup_steps = max(1, int(total_steps * warmup_ratio))
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=num_warmup_steps)
    decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps - num_warmup_steps)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[num_warmup_steps]
    )
    
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda') if use_amp else None
    
    best_metrics = None
    best_score = -float('inf')
    best_state_dict = None
    no_improve = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).float()
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(input_ids, attention_mask).squeeze()
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, attention_mask).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader)
        
        model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device).float()
                
                if use_amp:
                    with autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(input_ids, attention_mask).squeeze()
                        probs = torch.sigmoid(outputs)
                else:
                    outputs = model(input_ids, attention_mask).squeeze()
                    probs = torch.sigmoid(outputs)
                
                all_probs.append(probs)
                all_labels.append(labels)
        
        all_probs = torch.cat(all_probs).to(torch.float32).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        all_preds = (all_probs > 0.5).astype(int)
        
        metrics = {
            'accuracy': float(accuracy_score(all_labels, all_preds)),
            'precision': float(precision_score(all_labels, all_preds, zero_division=0)),
            'recall': float(recall_score(all_labels, all_preds, zero_division=0)),
            'f1': float(f1_score(all_labels, all_preds, zero_division=0)),
        }
        
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        metrics['roc_auc'] = float(auc(fpr, tpr))
        
        current_score = metrics['f1']
        
        if current_score > best_score + min_delta:
            best_score = current_score
            best_metrics = metrics.copy()
            best_state_dict = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    return best_metrics if best_metrics is not None else metrics


def cross_validate_bertweet(
    model_class,
    model_kwargs: Dict,
    full_dataset,
    device: torch.device,
    n_splits: int = 5,
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    patience: int = 3,
    random_state: int = 42,
    use_amp: bool = True,
) -> tuple[Dict[str, List[float]], nn.Module]:
    """
    K-Fold кросс-валидация для BERTweet модели.
    """
    # Получаем labels для stratification
    labels = np.array([full_dataset[i]['label'].item() for i in range(len(full_dataset))])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cv_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }
    
    best_f1 = -float('inf')
    best_model_state = None
    
    logger.info(f"Starting {n_splits}-fold cross-validation")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        logger.info(f"FOLD {fold}/{n_splits}")
        
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        model = model_class(**model_kwargs)
        
        fold_metrics = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            use_amp=use_amp,
        )
        
        for metric_name, metric_value in fold_metrics.items():
            cv_metrics[metric_name].append(metric_value)
        
        logger.info(
            f"Fold {fold} Results: "
            f"Acc={fold_metrics['accuracy']:.4f}, "
            f"Prec={fold_metrics['precision']:.4f}, "
            f"Rec={fold_metrics['recall']:.4f}, "
            f"F1={fold_metrics['f1']:.4f}, "
            f"ROC-AUC={fold_metrics['roc_auc']:.4f}"
        )
        
        if fold_metrics['f1'] > best_f1:
            best_f1 = fold_metrics['f1']
            best_model_state = copy.deepcopy(model.state_dict())
 
    for metric_name, values in cv_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        logger.info(f"{metric_name.upper():12s}: {mean_val:.4f} ± {std_val:.4f}")
    
    final_model = model_class(**model_kwargs)
    if best_model_state is not None:
        final_model.load_state_dict(best_model_state)
        logger.info(f"\nBest model loaded (F1={best_f1:.4f})")
    
    return cv_metrics, final_model