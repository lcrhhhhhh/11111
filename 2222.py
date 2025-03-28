import os
import random
import jieba
import numpy as np
import matplotlib.pyplot as plt
from gensim import corpora, models
from sklearn.svm import SVC

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def process_texts(input_dir, use_stopwords=True, stop_file='cn_stopwords.txt'):
    text_collection = []
    category_labels = []
    
    special_chars = " `1234567890-=/*-~!@#$%^&*()_+qwertyuiop[]\\QWERTYUIOP{}|asdfghjkl;" \
                    "'ASDFGHJKL:\"zxcvbnm,./ZXCVBNM<>?~！@#￥%……&*（）——+【】：；“‘’”《》？，。" \
                    "、★「」『』～＂□ａｎｔｉ－ｃｌｉｍａｘ＋．／０１２３４５６７８９＜＝＞＠Ａ" \
                    "ＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＶＷＸＹＺ［＼］ｂｄｅｆｇｈｊｋｏｐｒｓ" \
                    "ｕｖｗｙｚ￣\u3000\x1a"

    stop_words = set()
    if use_stopwords:
        with open(stop_file, 'r', encoding='utf-8') as f:
            stop_words = set(f.read().splitlines())

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='GB18030') as f:
                content = f.read().replace("\n", " ")
                filtered = [c for c in content if c not in special_chars]
                if use_stopwords:
                    filtered = [c for c in filtered if c not in stop_words]
                text_collection.append("".join(filtered))
                category_labels.append(filename.split(".")[0])
    
    return text_collection, category_labels

def create_datasets(documents, labels, window, sample_count, use_words=True):
    samples = []
    per_class = sample_count // len(set(labels))

    for idx, (doc, lbl) in enumerate(zip(documents, labels)):
        tokens = list(jieba.cut(doc)) if use_words else list(doc)
        for _ in range(per_class):
            start_pos = random.randint(0, len(tokens) - window)
            sample = tokens[start_pos:start_pos+window]
            samples.append((idx, sample))

    random.shuffle(samples)
    train_samples = samples[:900]
    test_samples = samples[900:]

    return (
        [s[1] for s in train_samples],
        [s[0] for s in train_samples],
        [s[1] for s in test_samples],
        [s[0] for s in test_samples]
    )

def model_training(topic_num, window_size, X_train, y_train, X_test, y_test):
    vocab = corpora.Dictionary(X_train)
    train_corpus = [vocab.doc2bow(doc) for doc in X_train]
    
    lda = models.LdaModel(
        corpus=train_corpus,
        id2word=vocab,
        num_topics=topic_num
    )

    def get_features(corpus, num_topics):
        dists = lda.get_document_topics(corpus)
        features = np.zeros((len(corpus), num_topics))
        for i, dist in enumerate(dists):
            for topic, prob in dist:
                features[i][topic] = prob
        return features

    train_features = get_features(train_corpus, topic_num)
    svm = SVC(kernel='linear', probability=True)
    svm.fit(train_features, y_train)
    
    train_acc = svm.score(train_features, y_train)
    
    test_corpus = [vocab.doc2bow(doc) for doc in X_test]
    test_features = get_features(test_corpus, topic_num)
    test_acc = svm.score(test_features, y_test)
    
    return train_acc, test_acc

def execute_experiments():
    topic_numbers = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    window_sizes = [20, 100, 500, 1000, 3000]
    total_samples = 1000
    data_dir = 'data'
    
    raw_texts, text_labels = process_texts(input_dir=data_dir)
    
    # 实验一：不同主题数的影响
    window = 1000
    X_tr, y_tr, X_te, y_te = create_datasets(
        raw_texts, text_labels, window, total_samples
    )
    
    train_results, test_results = [], []
    for t in topic_numbers:
        tr_acc, te_acc = model_training(t, window, X_tr, y_tr, X_te, y_te)
        print(f"主题数={t}, 窗口={window}, 训练准确率={tr_acc:.4f}, 测试准确率={te_acc:.4f}")
        train_results.append(tr_acc)
        test_results.append(te_acc)
    
    plt.figure(figsize=(8,6))
    plt.plot(topic_numbers, train_results, 'g--', label='训练集')
    plt.plot(topic_numbers, test_results, 'b-', label='测试集')
    plt.xlabel('主题数量', fontsize=12)
    plt.ylabel('分类精度', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig('主题数影响.png', dpi=300)
    plt.close()
    
    # 实验二：分词效果比较
    optimal_topics = 200
    X_w, y_w, X_t, y_t = create_datasets(raw_texts, text_labels, window, total_samples)
    acc_train, acc_test = model_training(optimal_topics, window, X_w, y_w, X_t, y_t)
    print(f"[分词] 训练精度={acc_train:.4f}, 测试精度={acc_test:.4f}")
    
    X_w2, y_w2, X_t2, y_t2 = create_datasets(raw_texts, text_labels, window, total_samples, False)
    acc_train2, acc_test2 = model_training(optimal_topics, window, X_w2, y_w2, X_t2, y_t2)
    print(f"[未分词] 训练精度={acc_train2:.4f}, 测试精度={acc_test2:.4f}")
    
    # 实验三：不同窗口大小影响
    performance_train = []
    performance_test = []
    for ws in window_sizes:
        X_t1, y_t1, X_t2, y_t2 = create_datasets(raw_texts, text_labels, ws, total_samples)
        tr_a, te_a = model_training(optimal_topics, ws, X_t1, y_t1, X_t2, y_t2)
        print(f"窗口={ws}, 训练精度={tr_a:.4f}, 测试精度={te_a:.4f}")
        performance_train.append(tr_a)
        performance_test.append(te_a)
    
    plt.figure(figsize=(8,6))
    plt.semilogx(window_sizes, performance_train, 'g^--', label='训练集')
    plt.semilogx(window_sizes, performance_test, 'bs-', label='测试集')
    plt.xlabel('窗口尺寸', fontsize=12)
    plt.ylabel('分类精度', fontsize=12)
    plt.xticks(window_sizes, labels=[str(ws) for ws in window_sizes])
    plt.legend()
    plt.grid(True)
    plt.savefig('窗口尺寸影响.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    execute_experiments()