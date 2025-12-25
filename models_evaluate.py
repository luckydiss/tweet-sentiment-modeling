from typing import Dict, List, Tuple, Optional, Any
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score

import re

def train_evaluate_models_cv(
    models: List[Tuple[str, BaseEstimator]], 
    X: Any, 
    y: Any, 
    preprocessor: Pipeline,
    cv: Any,
    seed: Optional[int] = None
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Pipeline]]:
    """Возвращает метрики И обученные pipelines."""
    results_dict: Dict[str, Dict[str, float]] = {}
    trained_pipelines: Dict[str, Pipeline] = {}
    """производитобучение и кросс-валидацию переданных моделей, 
       возвращает словарь с метриками и обученные пайплайны"""

    scoring = {
        'accuracy': 'accuracy',
        'f1': make_scorer(f1_score, average='macro', zero_division=0),
        'precision': make_scorer(precision_score, average='macro', zero_division=0),
        'recall': make_scorer(recall_score, average='macro', zero_division=0),
        'roc_auc': 'roc_auc'
    }
    
    for model_name, model in models:
        start_time = time.time()
        
        if seed is not None:
            if hasattr(model, 'random_state'):
                model.set_params(random_state=seed)
            elif hasattr(model, 'seed'):
                model.set_params(seed=seed)
        
        if isinstance(preprocessor, Pipeline):
            steps = preprocessor.steps.copy()
            steps.append(('classifier', model))
            pipeline = Pipeline(steps)
        else:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
            error_score='raise'
        )
        
        pipeline.fit(X, y)
        trained_pipelines[model_name] = pipeline
        
        total_time = time.time() - start_time
        
        results_dict[model_name] = {
            'roc_auc': float(np.mean(cv_results['test_roc_auc'])),
            'f1_score': float(np.mean(cv_results['test_f1'])),
            'precision': float(np.mean(cv_results['test_precision'])),
            'recall': float(np.mean(cv_results['test_recall'])),
            'accuracy': float(np.mean(cv_results['test_accuracy'])),
            'training_time': float(total_time)
        }

    
    return results_dict, trained_pipelines

def preprocess_tweet_deep(text):
    """глубокаю очистка текста твита"""

    text = str(text).lower()
    
    # URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # (@username)
    text = re.sub(r'@\w+', '', text)
    
    # (#hashtag)
    text = re.sub(r'#\w+', '', text)
    
    # У(&amp; &lt; и т.д.)
    text = re.sub(r'&\w+;', '', text)
    
    # цифры
    text = re.sub(r'\d+', '', text)
    
    # спецсимволы и пунктуация
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # множественные пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns

def compare_models_metrics(
    results_dict: Dict[str, Dict[str, float]],
    metrics_to_compare: List[str] = ['roc_auc', 'f1_score', 'precision', 'recall', 'accuracy', 'training_time'],
    figsize: Tuple[int, int] = (12, 6)
) -> pd.DataFrame:
    """
    Сравнивает метрики нескольких моделей в виде таблицы и графика.
    """
    metrics_df = pd.DataFrame.from_dict(results_dict, orient='index')
    metrics_df.index.name = 'model'
    metrics_df = metrics_df.reset_index()
    
    print(metrics_df)
    
    quality_metrics = [m for m in metrics_to_compare if m != 'training_time']
    time_metrics = [m for m in metrics_to_compare if m == 'training_time']
    
    if time_metrics:
        fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [len(quality_metrics), 1]})
        
        heatmap_data_quality = metrics_df.set_index('model')[quality_metrics]
        sns.heatmap(
            heatmap_data_quality, 
            annot=True, 
            fmt='.3f', 
            cmap='RdYlGn', 
            vmin=0,  
            vmax=1,  
            cbar_kws={'label': 'Score (0-1)'},
            ax=axes[0]
        )
        axes[0].set_title('Метрики качества')
        
        heatmap_data_time = metrics_df.set_index('model')[time_metrics]
        sns.heatmap(
            heatmap_data_time, 
            annot=True, 
            fmt='.2f', 
            cmap='YlOrRd',
            cbar_kws={'label': 'Time (s)'},
            ax=axes[1]
        )
        axes[1].set_title('Время обучения')
        
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        heatmap_data_quality = metrics_df.set_index('model')[quality_metrics]
        sns.heatmap(
            heatmap_data_quality, 
            annot=True, 
            fmt='.3f', 
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Score (0-1)'},
            ax=ax
        )
        ax.set_title('Метрики качества')
    
    plt.tight_layout()
    plt.show()
    
    return metrics_df

def plot_feature_importance_for_models(
    trained_pipelines: Dict[str, Pipeline],
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """Выводит feature importance для обученных моделей."""
    for model_name, pipeline in trained_pipelines.items():
        trained_model = pipeline.named_steps['classifier']
        
        has_importance = hasattr(trained_model, 'feature_importances_')
        has_coef = hasattr(trained_model, 'coef_')
        
        if not has_importance and not has_coef:
            print(f"[{model_name}] does not support feature importance\n")
            continue
        
        if has_importance:
            importances = trained_model.feature_importances_
        else:
            if len(trained_model.coef_.shape) > 1:
                importances = np.abs(trained_model.coef_).mean(axis=0)
            else:
                importances = np.abs(trained_model.coef_[0])
        
        vectorizer = pipeline.named_steps['vectorizer']
        
        if hasattr(vectorizer, 'vocabulary_'):
            vocab = vectorizer.vocabulary_
            feature_names = [word for word, idx in sorted(vocab.items(), key=lambda x: x[1])]
        else:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=figsize)

        colors = sns.color_palette("viridis", len(importance_df))

        bars = plt.barh(
            importance_df['feature'],
            importance_df['importance'],
            color=colors
        )

        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + max(importance_df['importance']) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                va="center",
                fontsize=9
            )

        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importances: {model_name}")

        plt.gca().invert_yaxis() 
        plt.grid(axis="x", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.show()

def compare_models_improvement(
    new_results: Dict[str, Dict[str, float]],  
    base_results: Dict[str, Dict[str, float]], 
    title: str = '',
    figsize: Tuple[int, int] = (10, 6),
    metrics_to_compare: List[str] = ['roc_auc', 'f1_score', 'precision', 'recall', 'accuracy', 'training_time'],
    normalize: bool = True
) -> pd.DataFrame:
    """Рассчитывает разницу метрик между новыми рез-ми и бейзлайном,
        и отображает её в виде heatmap"""
    new_df = pd.DataFrame.from_dict(new_results, orient='index')
    new_df.index.name = 'model'
    new_df = new_df.reset_index()
    
    base_df = pd.DataFrame.from_dict(base_results, orient='index')
    base_df.index.name = 'model'
    base_df = base_df.reset_index()
    
    new_df = new_df[['model'] + [m for m in metrics_to_compare if m in new_df.columns]]
    base_df = base_df[['model'] + [m for m in metrics_to_compare if m in base_df.columns]]
    
    df_diff = new_df.set_index('model') - base_df.set_index('model')
    
    if 'training_time' in df_diff.columns:
        df_diff['training_time'] = -df_diff['training_time']
    
    if normalize:
        df_diff_norm = (df_diff - df_diff.mean()) / df_diff.std()
    else:
        df_diff_norm = df_diff
    
    colors = ["Red", "White", "Green"]
    cmap = LinearSegmentedColormap.from_list("rwg", colors)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        df_diff_norm, 
        annot=df_diff,
        fmt='.4f',
        cmap=cmap,
        cbar=True,
        center=0,
        cbar_kws={'label': 'Нормализованное изменение' if normalize else 'Изменение'}
    )
    plt.title(title)
    plt.ylabel('Модель')
    plt.xlabel('Метрика')
    plt.tight_layout()
    plt.show()

    return df_diff

def plot_feature_importance_anova_1k(cv_pipelines_dict, X, y, cv, top_n=20):
    """Анализирует и визуализирует усредненную по фолдам 
       важность признаков для заданного набора классификаторов, 
       использующих ANOVA-отбор"""
    sample_pipe = cv_pipelines_dict['LogisticRegression']
    preprocessor = Pipeline(sample_pipe.steps[:-1])  # vectorizer + anova_top
    
    preprocessor.fit(X, y)
    
    vectorizer = preprocessor.named_steps['vectorizer']
    full_feature_names = vectorizer.get_feature_names_out()
    
    anova_selector = preprocessor.named_steps['anova_top']
    selected_indices_global = anova_selector.selected_idx
    selected_feature_names = full_feature_names[selected_indices_global] 
    
    print(f"Отобрано {len(selected_feature_names)} признаков.")
    print("\nТоп-20 n-грамм по ANOVA F-value:")
    for i, name in enumerate(selected_feature_names[:20], 1):
        print(f"{i:2}. {name}")

    fold_importances = {} 
    
    fold = 0
    for train_idx, test_idx in cv.split(X, y):
        X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        
        importances_per_model = {}
        for model_name, base_pipeline in cv_pipelines_dict.items():
            if model_name == 'DummyClassifier':
                continue
            pipe = base_pipeline.fit(X_train_fold, y_train_fold)
            clf = pipe.named_steps['classifier']
            
            if model_name == 'LogisticRegression':
                imp = np.abs(clf.coef_[0])
            else: 
                imp = clf.feature_importances_
            
            importances_per_model[model_name] = imp
        
        for name, imp in importances_per_model.items():
            if name not in fold_importances:
                fold_importances[name] = []
            fold_importances[name].append(imp)
        
        fold += 1
    
    mean_importances = {}
    for name, imps_list in fold_importances.items():
        mean_importances[name] = np.mean(imps_list, axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    model_idx = 0
    
    for model_name in ['LogisticRegression', 'DecisionTreeClassifier', 
                       'RandomForestClassifier', 'LGBMClassifier']:
        importances = mean_importances[model_name]
        
        indices = np.argsort(importances)[::-1]
        top_names = selected_feature_names[indices[:top_n]]
        top_values = importances[indices[:top_n]]
        
        ax = axes[model_idx]
        sns.barplot(x=top_values, y=top_names, ax=ax, palette="viridis")
        ax.set_title(f"{model_name} топ-{top_n})")
        ax.set_xlabel("Mean Importance / |coef|")
        
        model_idx += 1
    
    plt.tight_layout()
    plt.show()