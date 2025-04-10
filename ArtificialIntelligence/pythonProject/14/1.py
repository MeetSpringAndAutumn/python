import os
import jieba


def load_emails(folder_path, end, start=0):
    emails = []
    for filename in [str(i) + '.txt' for i in range(start, end + 1)]:
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                words = jieba.lcut(content)
                emails.append(words)
    return emails


folder_path = 'email-text'
emails = load_emails(folder_path, end=150)
# print(len(emails))

from sklearn.feature_extraction.text import CountVectorizer


def get_top_words(emails, top_n=600):
    vectorizer = CountVectorizer(max_features=top_n, token_pattern=r'(?u)\b[^0-9\W_]+\b')
    all_words = [' '.join(email) for email in emails]
    vectorizer.fit(all_words)
    top_words = vectorizer.get_feature_names_out()
    return top_words, vectorizer


top_words, vectorizer = get_top_words(emails)
# print(top_words)

def get_feature_vectors(emails, vectorizer):
    all_words = [' '.join(email) for email in emails]
    # print(vectorizer.transform(all_words).toarray())
    feature_vectors = vectorizer.transform(all_words).toarray()
    return feature_vectors


feature_vectors = get_feature_vectors(emails, vectorizer)
from numpy import array

labels = array([1] * 127 + [0] * 24)  #
from sklearn.naive_bayes import MultinomialNB
import joblib


# 创建并训练模型
model = MultinomialNB()
model.fit(feature_vectors, labels)

# 保存模型和前600个单词
joblib.dump(model, 'email_classifier_model.pkl')
joblib.dump(top_words, 'top_600_words.pkl')
# 加载模型和前600个单词
model = joblib.load('email_classifier_model.pkl')
top_words = joblib.load('top_600_words.pkl')
# vectorizer = CountVectorizer(vocabulary=top_words, token_pattern=r'(?u)\b\w+\b')
vectorizer = CountVectorizer(vocabulary=top_words, token_pattern=r'(?u)\b[^0-9\W_]+\b')

test_folder_path = 'email-text'
test_emails = load_emails(test_folder_path, start=152, end=155)
test_feature_vectors = get_feature_vectors(test_emails, vectorizer)

# 进行预测
predictions = model.predict(test_feature_vectors)
test_emails_name=[f'{k}.txt' for k in range(152,156)]
for email, prediction in zip(test_emails_name, predictions):
    print(f"邮件: {' '.join(email)}")
    print(f"预测结果: {prediction}")
    print()