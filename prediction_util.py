import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split  # 함수
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn.linear_model import LogisticRegression  # Logistic Regression 알고리즘


class PredictionUtil: #gilburt
    df = 0
    def read(self, aaa):
        self.df = pd.read_csv(aaa) #df = data frame
        print(self.df.head(10))

    # 컬럼별 유닉한 값의 수
    def show_unique_column(self):
        for column in self.df:
            print(column,':', self.df[column].nunique())

    # 지정한 컬럼들 간의 관계를 그래프로 그림. 이때 h로 지정된 컬럼의 값에 따라 색을 달리 표시함.
    def pairplots(self, cols, h):
        plt.figure(figsize=(10,6))
        sns.plotting_context('notebook',font_scale=1.2)
        g = sns.pairplot(self.df[cols], hue=h,height=2)
        g.set(xticklabels=[])
        plt.show()

    def lmplot(self, a, b, c):
        # sqft_living과 price간의 관계를 표시하되 등급(grade)을 다른 색으로 출력함.
        sns.lmplot(x=a, y=b, hue=c, data=self.df, fit_reg=False)
        plt.show()

    def heatmap2(self, columns):
        plt.figure(figsize=(15,10))
        sns.heatmap(self.df[columns].corr(),annot=True)
        plt.show()
        #CHECK THE PPT SLIDE

    def heatmap1(self):
        plt.figure(figsize=(14, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='cubehelix_r')
        plt.show()


    def boxplot(self, a, b):
        f, sub = plt.subplots(1, 1,figsize=(12.18,5))
        sns.boxplot(x=self.df[a],y=self.df[b], ax=sub)
        sub.set(xlabel=a, ylabel=b);
        plt.show()

    def plot_3d(self, a, b, c):
        from mpl_toolkits.mplot3d import Axes3D

        fig=plt.figure(figsize=(12,8))

        ax=fig.add_subplot(1,1,1, projection="3d")
        ax.scatter(self.df[a],self.df[b],self.df[c],c="darkred",alpha=.5)
        ax.set(xlabel=a,ylabel=b,zlabel=c)
        plt.show()

    def drop(self, col):
        df = self.df.drop(['id', 'date'], axis=1)

    def split(self):
        i, j = train_test_split(self.df, train_size = 0.8, random_state=5)  # 3=seed
        return i, j

    def histogram(self):
        self.df.hist(edgecolor='black', linewidth=1.2)
        fig = plt.gcf()
        fig.set_size_inches(12, 10)
        plt.show()

    def myviolinplot(self, a, b):
        plt.figure(figsize=(5,4))
        plt.subplot(1,1,1)
        sns.violinplot(x=a,y=b,data=self.df)
        plt.show()

    def run_linear_regress(self, input_cols, target):
        from sklearn import linear_model

        a, b = self.split()

        gildong = LinearRegression()

        gildong.fit(a[input_cols], a[target])

        predicted = gildong.predict(b[input_cols])
        print(predicted, '\n', predicted.shape)

        score = gildong.score(b[input_cols], b[target])
        print('LR - '+format(score, '.3f'))

    def run_kneighbor_regress(self, input_cols, target):
        from sklearn.neighbors import KNeighborsRegressor
        a, b = self.split(self.df)
        babo = KNeighborsRegressor(n_neighbors=10)
        babo.fit(a[input_cols], a[target])
        score = babo.score(b[input_cols], b[target])
        print('K-NR - ' + format(score, '.3f'))

    def run_decision_tree(self, input_cols, target):
        youngja = DecisionTreeRegressor(random_state = 0)

        a, b = self.split(self.df)
        youngja.fit(a[input_cols], a[target])

        predicted = youngja.predict(b[input_cols])
        print(predicted, '\n', predicted.shape)

        score = youngja.score(b[input_cols], b[target])
        print('DT - ' + format(score,'.3f'))

    def run_random_forest(self, input_cols, target):
        cheolsu = RandomForestRegressor(n_estimators=28, random_state=0)

        a, b = self.split(self.df)
        cheolsu.fit(a[input_cols], a[target])
        score = cheolsu.score(b[input_cols], b[target])
        print('Random F. - ' + format(score, '.3f'))

    def run_all(self, input_cols, target):
        self.run_linear_regress(input_cols, target)
        self.run_kneighbor_regress(input_cols, target)
        self.run_decision_tree(input_cols, target)
        self.run_random_forest(input_cols, target)

    def run_svm(self, list, target):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[list]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[list]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = svm.SVC()  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        rate = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', rate)
        return rate


    def run_logistic_regression(self, list, target):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[list]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[list]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = LogisticRegression()  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        rate = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', rate)
        return rate


    def run_neighbor_classifier (self, list, target, num):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[list]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[list]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = KNeighborsClassifier(n_neighbors=num)  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        rate = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', rate)
        return rate


    def run_decision_tree_classifier (self, list, target):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[list]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[list]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = DecisionTreeClassifier()  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        rate = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', rate)
        return rate

    def run_svm2(self, list, target):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[list]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[list]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = svm.SVC()  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        rate = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', rate)
        return rate


    def run_logistic_regression2(self, list, target):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[list]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[list]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = LogisticRegression()  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        rate = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', rate)
        return rate


    def run_neighbor_classifier2 (self, list, target, num):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[list]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[list]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = KNeighborsClassifier(n_neighbors=num)  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        rate = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', rate)
        return rate


    def run_decision_tree_classifier2 (self, list, target):
        train, test = train_test_split(self.df, test_size=0.3)
        train_X = train[list]  # 키와 발크기만 선택
        train_y = train[target]  # 정답 선택
        test_X = test[list]  # taking test data features
        test_y = test[target]  # output value of test data

        baby1 = DecisionTreeClassifier()  # 애기
        baby1.fit(train_X, train_y)  # 가르친 후
        prediction = baby1.predict(test_X)  # 테스트

        rate = metrics.accuracy_score(prediction, test_y) * 100
        print('인식률:', rate)
        return rate

