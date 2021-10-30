#!/usr/bin/env python
# _*_ coding:utf-8 _*_

from jieba import analyse


def textrank_extract(text, keyword_num=10):
    textrank = analyse.textrank
    keywords = textrank(text, keyword_num)
    # 输出抽取出的关键词
    for keyword in keywords:
        print(keyword + "/ ", end='')
    print()


def tfidf_extract(text, keyword_num=10):
    tfidf = analyse.extract_tags
    keywords = tfidf(text, keyword_num)
    # 输出抽取出的关键词
    for keyword in keywords:
        print(keyword + "/ ", end='')
    print()


if __name__ == '__main__':
    text = '比比才知道月初卖场超值平板电视点评'

    print('TF-IDF模型结果：')
    tfidf_extract(text)
    print('TextRank模型结果：')
    textrank_extract(text)
