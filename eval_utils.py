from nltk.corpus import stopwords
import nltk
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import f_classif

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc)

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+|@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


def clean_text_for_twitter_glove(text):
    """
    Препроцессинг текста для GloVe Twitter эмбеддингов [web:14].
    GloVe Twitter заменяет URL, упоминания и хэштеги на спецтокены.
    """
    text = text.lower()
    
    text = re.sub(r'http\S+|www\S+|https\S+', '<url>', text)
    
    text = re.sub(r'@\w+', '<user>', text)
    
    text = re.sub(r'#(\w+)', r'<hashtag> \1', text)
    
    text = re.sub(r'\d+', '<number>', text)
    
    text = re.sub(r'[^\w\s<>]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_top_ngrams_anova(corpus, target, n=None, top=20):
    """
    Извлечение топ n-грамм на основе ANOVA F-value.
    corpus: весь столбец текстов
    target: столбец меток классов
    """
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english', min_df=3)
    X = vec.fit_transform(corpus)
    feature_names = vec.get_feature_names_out()

    y = target.values
    
    f_scores, p_values = f_classif(X, y)
    
    X_disaster = X[y == 1]
    X_non_disaster = X[y == 0]
    
    mean_disaster = np.array(X_disaster.mean(axis=0)).flatten()
    mean_non_disaster = np.array(X_non_disaster.mean(axis=0)).flatten()
    
    disaster_feats = []
    non_disaster_feats = []
    
    for i, word in enumerate(feature_names):
        score = f_scores[i]
        
        if np.isnan(score) or np.isinf(score):
            continue
            
        if mean_disaster[i] > mean_non_disaster[i]:
            disaster_feats.append((word, score))
        else:
            non_disaster_feats.append((word, score))
            
    disaster_top = sorted(disaster_feats, key=lambda x: x[1], reverse=True)[:top]
    non_disaster_top = sorted(non_disaster_feats, key=lambda x: x[1], reverse=True)[:top]
    
    return disaster_top, non_disaster_top

def get_top_ngrams(corpus, n=None, top=20):
    """Извлечение топ n-грамм из корпуса текстов"""
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english', max_features=2000).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top]

def get_top_words(texts, n=20, remove_stopwords=True):
    """Получение топ-N слов из текстов"""
    words = []
    for text in texts:
        text_lower = str(text).lower()
        text_cleaned = re.sub(r'http\S+|www\.\S+|@\w+|#|[^a-zA-Z\s]', '', text_lower)
        words.extend(text_cleaned.split())
    
    if remove_stopwords:
        words = [w for w in words if w not in stop_words and len(w) > 2]
    
    return Counter(words).most_common(n)

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def prepare_text_for_wordcloud(texts, remove_stopwords=True):
    """Подготовка текста для wordcloud"""
    all_text = []
    for text in texts:
        text_lower = str(text).lower()
        # Удаление URLs, mentions, спец. символов
        text_cleaned = re.sub(r'http\S+|www\.\S+|@\w+|#|[^a-zA-Z\s]', '', text_lower)
        all_text.append(text_cleaned)
    
    combined_text = ' '.join(all_text)
    
    if remove_stopwords:
        words = combined_text.split()
        words = [w for w in words if w not in stop_words and len(w) > 2]
        combined_text = ' '.join(words)
    
    return combined_text

def plot_ngram(ax, data, title, color):
    if not data: return
    words, scores = zip(*data)
    ax.barh(range(len(words)), scores, color=color)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=9)
    ax.set_xlabel('ANOVA F-Score (Важность)', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)


def plot_confusion_matrix_and_roc(all_preds: np.ndarray, all_trues: np.ndarray, 
                                   all_probs: np.ndarray) -> None:
    """
    Выводит confusion matrix и ROC кривую по агреггированным метрикам.
    
    Args:
        all_preds (np.array): Агреггированные предсказания со всех фолдов
        all_trues (np.array): Агреггированные истинные значения со всех фолдов
        all_probs (np.array): Агреггированные вероятности со всех фолдов
    """
    cm = confusion_matrix(all_trues, all_preds)
    
    fpr, tpr, _ = roc_curve(all_trues, all_probs)
    roc_auc = auc(fpr, tpr)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax1)
    
    classes = ['Negative', 'Positive']
    tick_marks = np.arange(len(classes))
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(classes)
    ax1.set_yticklabels(classes)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, str(cm[i, j]), 
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('True label', fontsize=12)
    ax1.set_xlabel('Predicted label', fontsize=12)
    ax1.set_title('Confusion Matrix (Aggregated)', fontsize=14, fontweight='bold')
    
    ax2 = axes[1]
    ax2.plot(fpr, tpr, color='darkorange', lw=2.5, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curve (Aggregated)', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower right", fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_models_heatmap(metrics_list: list[dict], model_names: list[str],
                        title: str = "Сравнение метрик моделей"):
    """
    Рисует heatmap сравнения метрик нескольких моделей.
    
    Параметры:
    - metrics_list: список словарей с метриками (каждый словарь — одна модель)
    - model_names: список названий моделей (должен быть той же длины)
    - title: заголовок графика
    """


    metrics_order = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    data_rows = []
    for metrics in metrics_list:
        row = [metrics.get(m, None) for m in metrics_order]
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows, index=model_names, columns=metrics_order)
    
    annot = df.round(4).astype(str)
    
    plt.figure(figsize=(10, 1 + len(model_names) * 0.8))
    
    sns.heatmap(df, annot=annot, fmt='', cmap='YlGnBu', linewidths=.5,
                cbar_kws={'label': 'Значение метрики'}, annot_kws={"size": 12})
    
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel('Модели', fontsize=12)
    plt.xlabel('Метрики', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()