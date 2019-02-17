
import jieba
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.ensemble import VotingClassifier
from sklearn.externals import joblib
import numpy as np
stop_words = []  # 停用词表
train_tags = []  # 训练集标签
train_text = []  # 与训练集标签相对应的句子
test_tags = []  # 测试集标签
test_text = []  # 测试集...
vali_tags = []  # 验证集
vali_text = []
seg_train_text = []  # 分词后的训练集
seg_test_text = []  # 分词后的测试集
seg_vali_text = []  # 分词后的验证集


def devide():  # 将标签与句子分离
    with open('./data/train_set.txt', encoding='utf-8') as f:
        train_set = f.readlines()
    with open('./data/test_set.txt', encoding='utf-8') as f:
        test_set = f.readlines()
    with open('./data/ver_set.txt', encoding='utf-8') as f:
        vali_set = f.readlines()
    for line in train_set:
        t = line.split('\t', 1)  # 分割标签与句子
        train_tags.append(t[0])
        train_text.append(t[1].rstrip())
    for line in test_set:
        t = line.split('\t', 1)
        test_tags.append(t[0])
        test_text.append(t[1].rstrip())
    for line in vali_set:
        t = line.split('\t', 1)
        vali_tags.append(t[0])
        vali_text.append(t[1].rstrip())


def segment():

    for i in train_text:
        words = ' '.join(jieba.cut(i))
        seg_train_text.append(words)
    for i in test_text:
        words = ' '.join(jieba.cut(i))
        seg_test_text.append(words)
    for i in vali_text:
        words = ' '.join(jieba.cut(i))
        seg_vali_text.append(words)


def get_stop_words():  # 获取停用词列表

    with open('./keywords/stopwords.txt', encoding='utf-8') as f:
        stop_word = f.read()
    string = stop_word.rstrip().split('\n')
    for s in string:
        stop_words.append(s)



def validate(i):
    clf = joblib.load('train_model' + str(i) + '.m')
    predicted = clf.predict(seg_vali_text)
    accuracy = metrics.accuracy_score(vali_tags, predicted)  # 准确率
    precision = metrics.precision_score(
        vali_tags, predicted, average='weighted')
    recall = metrics.recall_score(vali_tags, predicted, average='weighted')
    f1 = metrics.f1_score(vali_tags, predicted, average='weighted')

    print("%.3f" % (accuracy), end='\t')
    print("%.3f" % (precision), end='\t')
    print("%.3f" % (recall), end='\t')
    print("%.3f" % (f1))



def train():

    steps = [('vect', CountVectorizer(stop_words=stop_words)),
             ('tfidf', TfidfTransformer())]
    steps.append(('select', SelectKBest(chi2, k=8000)))
    #+++++++++++++++++++++++++++++++++++++++++++++++准确率    精确度    召回率
    mnb = ('clf', MultinomialNB())  # ++++++++++++++0.6026    0.46     0.60
    svc = ('clf', SVC(kernel='linear', C=2))  # +++++++0.7016    0.68     0.70
    sgd = ('clf', SGDClassifier())  # +++++++0.7103    0.69     0.71
    lsvc = ('clf', LinearSVC())  # +++++++++++++++++0.6972    0.67     0.70
    logis = ('clf', LogisticRegression())  # +++++++0.6317    0.58     0.63
    rfc = ('clf', RandomForestClassifier())  # ++++ +0.6870    0.65     0.69
    knn = ('clf', KNeighborsClassifier())  # +++++++0.5953    0.35     0.6
    dtc = ('clf', tree.DecisionTreeClassifier())  # 0.6667    0.65     0.67
    gdbt = ('clf', GradientBoostingClassifier())  # ++0.6972    0.66    0.70
    # nc = ('clf', NearestCentroid())


    steps.append(sgd)
    text_clf1 = Pipeline(steps)
    steps.pop()
    steps.append(svc)
    text_clf2 = Pipeline(steps)
    steps.pop()
    steps.append(lsvc)
    text_clf3 = Pipeline(steps)
    # 模型聚合
    eclf = VotingClassifier(estimators=[(
        'lr', text_clf1), ('rf', text_clf2), ('gnb', text_clf3)], voting='hard')
    eclf.fit(seg_train_text, train_tags)
    predicted = eclf.predict(seg_test_text)
    pre = np.mean(predicted == test_tags)  # 准确率
    print('准确率:', pre)
    print('classification report:')
    print(metrics.classification_report(test_tags, predicted))
    print('混淆矩阵:')
    print(metrics.confusion_matrix(test_tags, predicted))


def rate():
    num = 0
    for tag in train_tags:
        if tag == '0':
            num += 1
    return num / len(train_tags)


def main():
    devide()
    segment()
    get_stop_words()
    train()


main()
