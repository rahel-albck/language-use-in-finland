import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
import re
import contractions
import spacy

from statistics import mean
from cleantext import clean
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from emoji import UNICODE_EMOJI
from trankit import Pipeline

tweet_tokenizer = TweetTokenizer()

nltk.download('stopwords')


def remove_tags(s):
    """
    method for removing the tags resulting from the clean() method
    """
    s = s.replace('<url>', '')
    s = s.replace('<number>', '')
    s = s.replace('<cur>', '')

    return s


def clean_tweet(txt, lang='en', remove_hashtags=False, remove_contractions=False):
    """
    method for cleaning a single tweet

    returns the cleaned tweet, the list of hashtags, and the list of emojis
    """
    # remove mentions
    cleaned = re.sub('(@[A-Za-zäöåéÄÖÅÉ0-9-_’\']+)', '', txt)
    # identify and extract hashtags
    hashtags = re.findall('(#[A-Za-zäöåéÄÖÅÉ0-9-_]+)', cleaned)
    # identify and extract emojis
    emojis = []
    for ch in cleaned:
        if ch in UNICODE_EMOJI['en']:
            emojis.append(ch)

    if remove_hashtags:
        cleaned = re.sub('(#[A-Za-zäöÄÖåéÅÉ0-9-_’\']+)', '', cleaned)

    cleaned = clean(cleaned, to_ascii=False, no_urls=True, no_line_breaks=True)
    # remove tags resulting from the clean() method
    cleaned = remove_tags(cleaned)
    if lang == 'en' and remove_contractions:
        cleaned = contractions.fix(cleaned)

    # most special characters are excluded, except for the ones listed here, due to their appearance in the
    # Swedish language / frequency of appearance in the dataset
    cleaned = re.sub('([^a-zA-ZäöåéÄÖÅÉ\s])', '', cleaned)

    # remove whitespaces at the beginning and end
    cleaned = cleaned.strip()
    # remove double (or more) whitespaces
    cleaned = re.sub(' +', ' ', cleaned)

    return cleaned, hashtags, emojis


def count_types(text):
    """
    method for counting the number of word types
    """
    types = dict()
    for token in text:
        if token in types.keys():
            types[token] += 1
        else:
            types[token] = 1

    return types, len(types.keys())


def preprocess(text_data, data, lang='en', remove_hashtags=False, remove_contractions=False):
    """
    preprocessing method
    cleans each tweet and tokenises it
    stores the cleaned and tokenised version of the tweet in the dataframe along with additional information
    calculates the number of tokens, the number of types, and the type-token-ratio, and stores them in the dataframe as well

    returns the preprocessed data, the list of tokens, and the list of types
    """
    tokens = []
    types = dict()
    for t in range(0, len(text_data)):
        cln, hashtags, emojis = clean_tweet(text_data[t][1], lang=lang, remove_hashtags=remove_hashtags,
                                            remove_contractions=remove_contractions)
        data.loc[t, 'account_id'] = text_data[t][0]
        data.loc[t, 'original'] = text_data[t][1].replace('\n', ' ')
        data.loc[t, 'cleaned'] = cln
        data.loc[t, 'len'] = len(cln)
        tkns = tweet_tokenizer.tokenize(cln)
        tokens.append(tkns)
        data.loc[t, 'num_tokens'] = len(tkns)
        tps, num_types = count_types(tkns)
        for tp in tps.keys():
            if tp in types.keys():
                types[tp] += tps[tp]
            else:
                types[tp] = tps[tp]
        data.loc[t, 'num_types'] = num_types
        data.loc[t, 'hashtags'] = hashtags
        data.loc[t, 'emojis'] = emojis
        if not (data.loc[t, 'num_tokens'] == 0):
            data.loc[t, 'type_token_ratio'] = data.loc[t, 'num_types'] / data.loc[t, 'num_tokens']
        else:
            data.loc[t, 'type_token_ratio'] = 0

    return data, tokens, types


def remove_stop_words(data, stop_words):
    """
    method for removing stop words in the dataframe that contains the cleaned tweets
    """
    data_no_stopwords = pd.DataFrame(
        columns=['account_id', 'original', 'cleaned', 'len', 'num_tokens', 'num_types', 'hashtags', 'emojis',
                 'type_token_ratio'])

    for i in range(0, len(data)):
        data_no_stopwords.loc[i, 'account_id'] = data.loc[i, 'account_id']
        data_no_stopwords.loc[i, 'original'] = data.loc[i, 'original']
        if not isinstance(data.loc[i, 'cleaned'], str):
            data_no_stopwords.loc[i, 'cleaned'] = data.loc[i, 'cleaned']
            data_no_stopwords.loc[i, 'len'] = data.loc[i, 'len']
            data_no_stopwords.loc[i, 'num_tokens'] = data.loc[i, 'num_tokens']
            data_no_stopwords.loc[i, 'num_types'] = data.loc[i, 'num_types']
            data_no_stopwords.loc[i, 'type_token_ratio'] = data.loc[i, 'type_token_ratio']
        else:
            twt = data.loc[i, 'cleaned'].split()
            twt_no_stopwords = [w for w in twt if w not in stop_words]
            types, num_types = count_types(twt_no_stopwords)
            new_twt = ' '.join(twt_no_stopwords)
            data_no_stopwords.loc[i, 'cleaned'] = new_twt
            data_no_stopwords.loc[i, 'len'] = len(new_twt)
            data_no_stopwords.loc[i, 'num_tokens'] = len(twt_no_stopwords)
            data_no_stopwords.loc[i, 'num_types'] = num_types
            if not len(twt_no_stopwords) == 0:
                data_no_stopwords.loc[i, 'type_token_ratio'] = num_types / len(twt_no_stopwords)
            else:
                data_no_stopwords.loc[i, 'type_token_ratio'] = 0
        data_no_stopwords.loc[i, 'hashtags'] = data.loc[i, 'hashtags']
        data_no_stopwords.loc[i, 'emojis'] = data.loc[i, 'emojis']

    return data_no_stopwords


def lemmatize_and_pos_tag_row(cell, processor, lang='en'):
    """
    method for lemmatizing and POS tagging a single tweet

    returns a list of POS tagged lemmas
    """
    pos = []
    if lang == 'en':
        if isinstance(cell, str):
            sen = processor(cell)
            for word in sen:
                lm = word.lemma_
                pos.append((word.text, word.pos_, word.tag_, lm))
    elif lang == 'fi':
        if isinstance(cell, list):
            sen = processor(cell, is_sent=True)
            for word in sen['tokens']:
                lm = word['lemma']
                pos.append((word['text'], word['upos'], word['xpos'], lm))

    return pos


def add_lemma(cell):
    """
    method for extracting only the lemmas from a list containing the POS tags and the lemma for each token
    """
    lemmas = []

    for item in cell:
        lemmas.append(item[3])

    return lemmas


def add_lemma_unique(cell):
    """
    method for getting the unique lemma types in a list of lemmas
    """
    lemmas_unique = []

    for item in cell:
        if item[3] not in lemmas_unique:
            lemmas_unique.append(item[3])

    return lemmas_unique


def type_token_ratio(data, tokens_lower, tokens_upper):
    """
    method for calculating the average type-token-ratio in a group of tweets
    """
    data = data[(data['num_tokens'] >= tokens_lower) & (data['num_tokens'] < tokens_upper)]

    average_ratio = mean(data['type_token_ratio'])

    return average_ratio


def calculate_stats(twt_stats, data, type_dict, lang):
    """
    method for calculating statistics regarding the length of tweets
    """
    # exclude empty tweets
    data_no_empty_tweets = data.loc[data['cleaned'] != '']
    data_no_empty_tweets.index = range(0, len(data_no_empty_tweets))
    idx = len(twt_stats)
    twt_stats.loc[idx, 'lang'] = lang
    # calculate the average length of tweets
    twt_stats.loc[idx, 'average_len'] = mean(data_no_empty_tweets['len'])
    # calculate the total, average, minimum, and maximum number of tokens
    twt_stats.loc[idx, 'total_num_tokens'] = sum(data_no_empty_tweets['num_tokens'])
    twt_stats.loc[idx, 'average_num_tokens'] = mean(data_no_empty_tweets['num_tokens'])
    twt_stats.loc[idx, 'min_num_tokens'] = min(data_no_empty_tweets['num_tokens'])
    twt_stats.loc[idx, 'max_num_tokens'] = max(data_no_empty_tweets['num_tokens'])
    # calculate the total and average number of types
    twt_stats.loc[idx, 'total_num_types'] = len(type_dict)
    twt_stats.loc[idx, 'average_num_types'] = mean(data_no_empty_tweets['num_types'])
    # calculate the type-token-ratio for different size groups
    twt_stats.loc[idx, 'average_type_token_ratio_<10'] = type_token_ratio(data_no_empty_tweets, 0, 10)
    twt_stats.loc[idx, 'average_type_token_ratio_>10<20'] = type_token_ratio(data_no_empty_tweets, 10, 20)
    twt_stats.loc[idx, 'average_type_token_ratio_>20<30'] = type_token_ratio(data_no_empty_tweets, 20, 30)
    twt_stats.loc[idx, 'average_type_token_ratio_>30'] = type_token_ratio(data_no_empty_tweets, 30, 60)
    twt_stats.loc[idx, 'average_type_token_ratio_<15'] = type_token_ratio(data_no_empty_tweets, 0, 15)
    twt_stats.loc[idx, 'average_type_token_ratio_>=15'] = type_token_ratio(data_no_empty_tweets, 15, 60)

    return twt_stats


def plot_length_and_token_dist(df):
    """
    method for plotting tweet lengths in terms of tokens and characters
    """
    fig, axes = plt.subplots(1, 2)

    axes[0].hist(df['len'], bins=100, color='#00A59D')
    axes[0].set_title('Histogram by length of tweets')
    axes[0].set_xlabel('Length in characters')
    axes[0].set_ylabel('Frequency')
    axes[1].hist(df['num_tokens'], bins=100, color='#00A59D')
    axes[1].set_title('Histogram by the number of tokens of all tweets')
    axes[1].set_xlabel('Number of tokens')
    axes[1].set_ylabel('Frequency')

    plt.show()



if __name__ == '__main__':

    ###### Preprocessing ######

    # load the metadata
    info = pd.read_csv('info/info-municipalities-only.csv')

    # load the tweet objects
    os.chdir('..')
    os.chdir('tweets')

    data = []
    userids = [str(id) for id in info['userid'].values]
    tweets_per_account = dict()

    # extract the tweets and store them in a list
    for file in os.listdir(os.getcwd()):
        if file[:-5] in userids:
            if os.path.isfile(file):
                twts = []
                for line in open(file, 'r'):
                    twts.append(json.loads(line))
                    data.append(json.loads(line))
                tweets_per_account[file[:-5]] = twts

    # create separate lists for the total set of tweets, the set of Finnish tweets, the set of Swedish tweets, the
    # set of English tweets, and the set of tweets in other languages
    tweets = []
    tweetsfi = []
    tweetssv = []
    tweetsen = []
    tweetsoth = []

    for obj in data:
        for tweet in obj['data']:
            tweets.append(tweet)
            if tweet['lang'] == 'fi':
                tweetsfi.append(tweet)
            elif tweet['lang'] == 'sv':
                tweetssv.append(tweet)
            elif tweet['lang'] == 'en':
                tweetsen.append(tweet)
            else:
                tweetsoth.append(tweet)

    # create a collection of only the textual content of the tweets
    text_dat_all = [(twt['author_id'], twt['text']) for twt in tweets]
    text_dat_fi = [(twt['author_id'], twt['text']) for twt in tweetsfi]
    text_dat_sv = [(twt['author_id'], twt['text']) for twt in tweetssv]
    text_dat_en = [(twt['author_id'], twt['text']) for twt in tweetsen]

    ### Cleaning the Data ###

    ## English
    cleaned_data_en = pd.DataFrame(
        columns=['account_id', 'original', 'cleaned', 'len', 'num_tokens', 'num_types', 'hashtags', 'emojis',
                 'type_token_ratio'])

    cleaned_data_en, tokens_en, types_en = preprocess(text_dat_en, cleaned_data_en, remove_hashtags=True,
                                                      remove_contractions=True)

    # stop words
    stop_words_en = set(stopwords.words('english'))
    types_en_no_stopwords = {k: types_en[k] for k in types_en.keys() if k not in stop_words_en}

    # identify high frequency tokens that should be in the stop word list
    types_en_no_stopwords_freq = pd.DataFrame(columns=['type', 'frequency'])

    for type in types_en_no_stopwords.keys():
        idx = np.shape(types_en_no_stopwords_freq)[0]
        types_en_no_stopwords_freq.loc[idx, 'type'] = type
        types_en_no_stopwords_freq.loc[idx, 'frequency'] = types_en_no_stopwords[type]

    types_en_no_stopwords_freq_sorted = types_en_no_stopwords_freq.sort_values('frequency', ascending=False)
    types_en_no_stopwords_freq_sorted.index = range(0, len(types_en_no_stopwords_freq_sorted))
    types_en_no_stopwords_freq_sorted_subset = types_en_no_stopwords_freq_sorted[0:200]

    # update the stop word list
    stop_words_en.update(['ping', 'th', 'pm'])
    stop_words_en.update(['cannot', 'cant', 'id', 'theres', 'wed', 'dont', 'doesnt', 'whats'])

    # update the list of types excluding stop words
    types_en_no_stopwords = {k: types_en[k] for k in types_en.keys() if k not in stop_words_en}

    # remove stop words
    cleaned_data_en_no_stopwords = remove_stop_words(cleaned_data_en, stop_words_en)

    ## Finnish
    cleaned_data_fi = pd.DataFrame(
        columns=['account_id', 'original', 'cleaned', 'len', 'num_tokens', 'num_types', 'hashtags', 'emojis',
                 'type_token_ratio'])

    cleaned_data_fi, tokens_fi, types_fi = preprocess(text_dat_fi, cleaned_data_fi, lang='fi', remove_hashtags=True)

    # stop words
    stop_words_fi = set(stopwords.words('finnish'))

    types_fi_no_stopwords = {k: types_fi[k] for k in types_fi.keys() if k not in stop_words_fi}

    # identify high frequency tokens that should be in the stop word list
    types_fi_no_stopwords_freq = pd.DataFrame(columns=['type', 'frequency'])

    for type in types_fi_no_stopwords.keys():
        idx = np.shape(types_fi_no_stopwords_freq)[0]
        types_fi_no_stopwords_freq.loc[idx, 'type'] = type
        types_fi_no_stopwords_freq.loc[idx, 'frequency'] = types_fi_no_stopwords[type]

    types_fi_no_stopwords_freq_sorted = types_fi_no_stopwords_freq.sort_values('frequency', ascending=False)
    types_fi_no_stopwords_freq_sorted.index = range(0, len(types_fi_no_stopwords_freq_sorted))
    types_fi_no_stopwords_freq_sorted_subset = types_fi_no_stopwords_freq_sorted[0:200]

    # fix two typos in the stop word list
    stop_words_fi.remove('tallä')
    stop_words_fi.remove('tuotä')

    # update the stop word list
    stop_words_fi.update(['klo', 'mm', 'n', 'ma', 'la', 'to', 'ke', 'i', 'v', 'lla', 'ti', 'pe', 'su', 'tällä', 'tuota'])

    # update the list of types excluding stop words
    types_fi_no_stopwords = {k: types_fi[k] for k in types_fi.keys() if k not in stop_words_fi}

    # remove stop words
    cleaned_data_fi_no_stopwords = remove_stop_words(cleaned_data_fi, stop_words_fi)


    ### Tokenization ###

    ## English
    cleaned_data_en_no_stopwords['tokenized'] = np.nan
    for i in range(0, len(cleaned_data_en_no_stopwords)):
        if isinstance(cleaned_data_en_no_stopwords.loc[i, 'cleaned'], str):
            cleaned_data_en_no_stopwords.loc[:, 'tokenized'].loc[i] = tweet_tokenizer.tokenize(
                cleaned_data_en_no_stopwords.loc[i, 'cleaned'])

    ## Finnish
    cleaned_data_fi_no_stopwords['tokenized'] = np.nan
    for i in range(0, len(cleaned_data_fi_no_stopwords)):
        if isinstance(cleaned_data_fi_no_stopwords.loc[i, 'cleaned'], str):
            cleaned_data_fi_no_stopwords.loc[:, 'tokenized'].loc[i] = tweet_tokenizer.tokenize(
                cleaned_data_fi_no_stopwords.loc[i, 'cleaned'])

    ### Lemmatization ###

    ## English: spaCy
    sp = spacy.load('en_core_web_sm')

    cleaned_data_en_lemmas = cleaned_data_en_no_stopwords
    cleaned_data_en_lemmas['pos_and_lemma'] = cleaned_data_en_lemmas['cleaned'].apply(lemmatize_and_pos_tag_row, args=(sp, 'en'))
    cleaned_data_en_lemmas['lemmatized'] = cleaned_data_en_lemmas['pos_and_lemma'].apply(add_lemma)
    cleaned_data_en_lemmas['unique_lemmas'] = cleaned_data_en_lemmas['pos_and_lemma'].apply(add_lemma_unique)
    cleaned_data_en_lemmas['num_lemmas'] = cleaned_data_en_lemmas['unique_lemmas'].apply(len)

    ## Finnish: trankit
    pftb = Pipeline('finnish-ftb')

    cleaned_data_fi_lemmas = cleaned_data_fi_no_stopwords
    cleaned_data_fi_lemmas['pos_and_lemma'] = cleaned_data_fi_lemmas['tokenized'].apply(lemmatize_and_pos_tag_row, args=(pftb, 'fi'))
    cleaned_data_fi_lemmas['lemmatized'] = cleaned_data_fi_lemmas['pos_and_lemma'].apply(add_lemma)
    cleaned_data_fi_lemmas['unique_lemmas'] = cleaned_data_fi_lemmas['pos_and_lemma'].apply(add_lemma_unique)
    cleaned_data_fi_lemmas['num_lemmas'] = cleaned_data_fi_lemmas['unique_lemmas'].apply(len)


    ###### Length and Type-Token-Analysis ######

    ### English ###

    twt_stats = pd.DataFrame(columns=['lang', 'average_len', 'total_num_tokens', 'average_num_tokens',
                                      'min_num_tokens', 'max_num_tokens', 'total_num_types', 'average_num_types',
                                      'total_num_lemmas', 'average_num_lemmas',
                                      'average_type_token_ratio_<10', 'average_type_token_ratio_>10<20',
                                      'average_type_token_ratio_>20<30', 'average_type_token_ratio_>30',
                                      'average_type_token_ratio_<15', 'average_type_token_ratio_>=15'])

    # calculate length statistics
    twt_stats = calculate_stats(twt_stats, cleaned_data_en_no_stopwords, types_en_no_stopwords, 'en')

    # plot the distribution of tweet length in terms of characters and tokens
    plot_length_and_token_dist(cleaned_data_en_no_stopwords)


    ### Finnish ###

    # calculate length statistics
    twt_stats = calculate_stats(twt_stats, cleaned_data_fi_no_stopwords, types_fi_no_stopwords, lang='fi')

    # plot length and token distribution
    plot_length_and_token_dist(cleaned_data_fi_no_stopwords)




