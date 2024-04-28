import pandas as pd
import numpy as np
import tqdm
import gensim
import gensim.corpora as corpora
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import math
import random

from wordcloud import WordCloud
from gensim.models import CoherenceModel


def string_to_list(row, splt=' '):
    """
    method for transforming the lists of token, lemmas, etc. back into a list when they are in string format (after
    reading the dataframe from a file)

    takes a string as argument and transforms it into the respective list
    """
    items = []
    if isinstance(row, str):  # check for nan
        lst = row.split(splt)
        for l in lst:
            if l != '[]':
                item = l.strip('([\',])')
                if splt == '), (':
                    item_list = item.split(', ')
                    item = (item_list[0].strip('\''), item_list[1].strip('\''), item_list[2].strip('\''),
                            item_list[3].strip('\''))
                items.append(item)

    return items


def remove_stop_words_from_list(lst, stop_words, tuple=False):
    """
    method for removing stop words from a list

    returns the list excluding the stop words
    """
    if not tuple:
        new_list = [item for item in lst if item not in stop_words]
    else:
        new_list = [item for item in lst if item[3] not in stop_words]

    return new_list


def remove_stop_words_and_city_names_after_lemmatization(data, stop_words, city_names):
    """
    method for removing the additional stop words and city names from the data file after lemmatization has taken place

    returns the new dataframe with excluded stop words and city names
    """
    data_no_stopwords = pd.DataFrame(columns=['account_id', 'original', 'cleaned', 'len', 'num_tokens', 'num_types',
                                              'hashtags', 'emojis', 'type_token_ratio', 'tokenized', 'pos_and_lemma',
                                              'lemmatized', 'unique_lemmas'])
    exclude = stop_words + city_names

    data_no_stopwords['account_id'] = data['account_id']
    data_no_stopwords['original'] = data['original']
    data_no_stopwords['cleaned'] = data['cleaned']
    data_no_stopwords['len'] = data['len']
    data_no_stopwords['num_tokens'] = data['num_tokens']
    data_no_stopwords['num_types'] = data['num_types']
    data_no_stopwords['hashtags'] = data['hashtags']
    data_no_stopwords['emojis'] = data['emojis']
    data_no_stopwords['type_token_ratio'] = data['type_token_ratio']
    data_no_stopwords['tokenized'] = data['tokenized']
    data_no_stopwords['original'] = data['original']
    data_no_stopwords['pos_and_lemma'] = data['pos_and_lemma'].apply(remove_stop_words_from_list, args=(exclude, True))
    data_no_stopwords['lemmatized'] = data['lemmatized'].apply(remove_stop_words_from_list, args=(exclude, False))
    data_no_stopwords['unique_lemmas'] = data['unique_lemmas'].apply(remove_stop_words_from_list, args=(exclude, False))

    return data_no_stopwords


def compute_coherence(corpus, id2word, texts, num_topics, alpha, beta):
    """
    method for running a topic model with different values for the hyperparameters and computing the coherence score

    returns a dictionary containing the different hyperparameter combinations and the coherence score for each
    combination, as well as a list containing the model objects
    """
    results = {'modelID': [],
               'number of topics': [],
               'alpha': [],
               'beta': [],
               'coherence': []}
    models = []

    model_id = 0

    iter = len(num_topics) * len(alpha) * len(beta)

    if 1 == 1:
        pbar = tqdm.tqdm(total=iter)
        for num in num_topics:
            for a in alpha:
                for b in beta:
                    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                                id2word=id2word,
                                                                num_topics=num,
                                                                random_state=100,
                                                                chunksize=200,
                                                                passes=10,
                                                                alpha=a,
                                                                eta=b,
                                                                per_word_topics=True
                                                                )
                    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word,
                                                         coherence='c_v')
                    coherence_lda = coherence_model_lda.get_coherence()

                    results['number of topics'].append(num)
                    results['alpha'].append(a)
                    results['beta'].append(b)
                    results['coherence'].append(coherence_lda)
                    results['modelID'] = model_id
                    models.append(lda_model)
                    model_id += 1
                    pbar.update(1)
        pbar.close()

    return results, models



def find_optimal_alpha_beta_combination(data, num_topics):
    """
    method for finding the combination of alpha and beta values that leads to the best coherence score for a specific
    number of topics

    returns the alpha value, the beta value, the coherence score, and the ID of the respective model
    """
    topic_subset = data[data['number of topics'] == num_topics]
    optimal_row = topic_subset[topic_subset['coherence'] == max(topic_subset['coherence'])]

    return optimal_row.iloc[0, 1], optimal_row.iloc[0, 2], optimal_row.iloc[0, 3], optimal_row.iloc[0, 4]


def plot_topics(lda_model, corpus, id2word, filename):
    """
    method for plotting the topic models with pyLDAvis
    """
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, sort_topics=False)
    pyLDAvis.save_html(vis, filename)


def get_dominant_topic_per_tweet(model, corpus, texts):
    """
    method for retrieving the most dominant topic for each tweet and storing it in a dataframe, along with the topic
    proportion and the topic keywords

    returns the dominant topic dataframe and the list of topic percentages
    """
    topics_df = pd.DataFrame()
    topic_percentages = []

    # get main topic in each document
    for i, row_list in enumerate(model[corpus]):  # enumerate the tweets in the model
        row = row_list[0] if model.per_word_topics else row_list  # extract the topic numbers and their percentages
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        topic_percentages.append(row)
        dominant_topic = row[0][0]
        dominant_topic_prop = row[0][1]

        keyword_list = model.show_topic(dominant_topic)  # get the keywords (and their percentages) for the topic
        topic_keywords = ", ".join([word for word, prop in keyword_list])
        topics_df = pd.concat(
            [topics_df, pd.DataFrame([[int(dominant_topic), round(dominant_topic_prop, 4), topic_keywords]])],
            ignore_index=True)

    # add original text to the output
    contents = pd.Series(texts)
    topics_df = pd.concat([topics_df, contents], axis=1)
    topics_df.columns = ['Dominant_Topic', 'Topic_Perc_Contrib', 'Topic_Keywords', 'Text']

    return topics_df, topic_percentages


def get_dominant_topic_per_account(cleaned_data, topic_percentages, info):
    """
    method for retrieving the dominant topic per account

    returns a dataframe containing the account ID, the dominant topic, its weighting, as well as the regional
    information for the account
    """
    topic_weighting_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
    account_ids = pd.unique(cleaned_data['account_id'])
    dominant_topic_per_account = pd.DataFrame(columns=['ID', 'dominant_topic'])
    dominant_topic_per_account['ID'] = account_ids
    for i in range(0, len(dominant_topic_per_account)):
        id = dominant_topic_per_account.loc[i, 'ID']
        account_subset = cleaned_data[cleaned_data['account_id'] == id]
        tweet_ids = account_subset.index
        tweet_subset = topic_weighting_by_doc.loc[tweet_ids]
        topic_weightings = tweet_subset.sum().to_frame(name='count').reset_index()
        topic_weightings = topic_weightings.sort_values(by='count', ascending=False).reset_index()
        dominant_topic_per_account.loc[i, 'dominant_topic'] = topic_weightings.loc[0, 'index']
        dominant_topic_per_account.loc[i, 'topic_weighting'] = topic_weightings.loc[0, 'count']
        dominant_topic_per_account.loc[i, 'location'] = info.loc[info['userid'] == id, 'location'].iloc[0]
        dominant_topic_per_account.loc[i, 'region'] = info.loc[info['userid'] == id, 'region'].iloc[0]
        dominant_topic_per_account.loc[i, 'smaller_region'] = info.loc[info['userid'] == id, 'smaller_region'].iloc[0]
        dominant_topic_per_account.loc[i, 'size_group'] = info.loc[info['userid'] == id, 'size_group'].iloc[0]

    return dominant_topic_per_account


def get_dominant_topic_per_group(cleaned_data, dominant_topic_per_account, topic_percentages, group_type='region'):
    """
    method for retrieving the dominant topic for a group of tweets (within a region, sub-region, or size group)

    returns a dataframe containing each group, the respective dominant topic, and the topic proportion of that topic
    """
    topic_weighting_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
    dominant_topic_per_group = pd.DataFrame(columns=[group_type, 'topic', 'weighting'])
    groups = pd.unique(dominant_topic_per_account[group_type])
    for i in range(0, len(groups)):
        group = groups[i]
        group_subset = cleaned_data[cleaned_data[group_type] == group]
        tweet_ids = group_subset.index
        tweet_subset = topic_weighting_by_doc.loc[tweet_ids]
        topic_weightings = tweet_subset.sum().to_frame(name='count').reset_index()
        for j in range(0, len(topic_weightings)):
            row = {group_type: group, "topic": topic_weightings.loc[j, 'index'],
                   "weighting": topic_weightings.loc[j, 'count']}
            dominant_topic_per_group.loc[len(dominant_topic_per_group)] = row

    # modify the names of the size groups to make them suitable for visualisation
    if group_type == 'size_group':
        dominant_topic_per_group.loc[dominant_topic_per_group['size_group'] == 'bigger_cities', 'size_group'] = 'bigger cities'
        dominant_topic_per_group.loc[dominant_topic_per_group['size_group'] == 'smaller_cities', 'size_group'] = 'smaller cities'

    return dominant_topic_per_group



def dominant_topic_analysis(cleaned_data, info, topic_percentages):
    """
    method for retrieving the dominant topic for each account, region, sub-region, and size group

    returns four dataframes containing the respective information
    """
    dominant_topic_per_account = get_dominant_topic_per_account(cleaned_data, topic_percentages, info)

    dominant_topic_per_region = get_dominant_topic_per_group(cleaned_data, dominant_topic_per_account,
                                                             topic_percentages, group_type='region')
    dominant_topic_per_smaller_region = get_dominant_topic_per_group(cleaned_data, dominant_topic_per_account,
                                                                     topic_percentages, group_type='smaller_region')
    dominant_topic_per_size_group = get_dominant_topic_per_group(cleaned_data, dominant_topic_per_account,
                                                                 topic_percentages, group_type='size_group')

    return dominant_topic_per_account, dominant_topic_per_region, dominant_topic_per_smaller_region, dominant_topic_per_size_group


def plot_wordcloud_per_topic(model):
    """
    method for plotting the most dominant words in each topic in the form of a wordcloud
    """
    cols = ['#002F6C', '#365ABD', '#4293FF', '#22A055', '#007070', '#00A59D']

    cloud = WordCloud(background_color='white',
                      width=500,
                      height=300,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = model.show_topics(formatted=False)

    num_topics = len(topics)
    if num_topics <= 2:
        num_rows = 1
        if num_topics == 1:
            num_cols = 1
        else:
            num_cols = 2
    else:
        num_rows = 2
        num_cols = math.ceil(num_topics / 2)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    for i, ax in enumerate(axes.flatten()):
        if i >= len(topics):
            break
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=50)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i + 1), fontdict=dict(size=11))
        plt.gca().axis('off')

    if (num_topics % 2) != 0 and num_topics != 1:
        fig.delaxes(axes[num_rows - 1, num_cols - 1])
    fig.suptitle('Word Cloud of Most Dominant Words for each Topic', fontsize=10)

    plt.margins(x=0, y=0)
    plt.show()


def plot_topic_dominance(dominant_topic_per_group, num_topics, group_type, lang='English'):
    """
    method for plotting the dominance of each topic in each of the different regions, sub-regions, or size groups
    """
    cols = ['#002F6C', '#365ABD', '#4293FF', '#22A055', '#007070', '#00A59D']
    xlabels = pd.unique(dominant_topic_per_group[group_type[0]])
    if not group_type[0] == 'size_group':
        xlabels.sort()
    data = {}
    topics = pd.unique(dominant_topic_per_group['topic'])
    topics.sort()
    for topic in topics:
        topic_subset = dominant_topic_per_group.loc[
            dominant_topic_per_group['topic'] == topic]
        if not group_type[0] == 'size_group':
            topic_subset = topic_subset.sort_values(by=group_type[0])
            topic_subset.index = range(0, len(topic_subset))
        data[('Topic ' + str(topic + 1))] = list(topic_subset.weighting)

    x = np.arange(len(xlabels))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots()
    for tpc, weightings in data.items():
        offset = width * multiplier
        topic_id = int(tpc[-1]) - 1
        rects = ax.bar(x + offset, weightings, width, label=tpc, color=cols[topic_id])
        multiplier += 1

    ax.set_ylabel('Topic Weighting', fontsize=12)
    ax.set_title('Topic Weighting per ' + group_type[1] + ' - ' + lang + ' data', fontsize=16)
    ax.set_xticks(x + math.floor(num_topics / 2) * width, xlabels, fontsize=12, rotation=15)
    ax.legend(loc='best', ncols=math.ceil(num_topics / 2))

    plt.show()

def get_tweet_topic_matrix(model, corpus, num_topics):
    """
    method for calculating a matrix that maps each tweet to the proportion of each topic in the tweet

    returns the tweet topic matrix
    """
    twt_topic_matrix = np.zeros((len(model[corpus]), num_topics))

    for i, row_list in enumerate(model[corpus]):  # enumerate the tweets in the model
        row = row_list[0] if model.per_word_topics else row_list  # extract the topic numbers and their proportions
        for item in row:
            topic_id = item[0]
            twt_topic_matrix[i, topic_id] = item[1]

    return twt_topic_matrix


def get_most_dominant_tweets_per_topic(tweet_topic_matrix, data, num_topics, num_tweets=10, threshold=0.9):
    """
    method for retrieving a sample of 10 tweets from the set of most dominant tweets in a topic

    returns the sample of tweets in the form of a dictionary
    """
    tweet_topic_df = pd.DataFrame(tweet_topic_matrix)
    tweet_topic_df = tweet_topic_df.reset_index()
    tweet_topic_df = tweet_topic_df.rename({'index': 'Tweet_No'}, axis='columns')
    tweet_topic_df['Tweet'] = data['original']

    top_tweets = {}

    for topic in range(0, num_topics):
        tweet_perc = tweet_topic_df[['Tweet_No', topic, 'Tweet']]
        tweet_perc = tweet_perc.sort_values(by=topic, ascending=False)
        tweet_perc.index = range(0, len(tweet_perc))
        top_tweets_subset = tweet_perc[tweet_perc[topic] >= threshold]
        if len(top_tweets_subset) > num_tweets:
            top_tweets_sample = random.sample(list(top_tweets_subset['Tweet_No']), num_tweets)
            top_tweets_sample_subset = top_tweets_subset[top_tweets_subset['Tweet_No'].isin(top_tweets_sample)]
        else:
            top_tweets_sample_subset = top_tweets_subset
        top_tweets[topic] = top_tweets_sample_subset

    return top_tweets


if __name__ == '__main__':

    ###### Topic Modelling Analysis ######

    # load the metadata
    info = pd.read_csv('info/info-municipalities-only.csv')

    # load the English data
    cleaned_data_en = pd.read_csv('preprocessed-data-en.tsv', sep='\t', header=0)
    cleaned_data_en['hashtags'] = cleaned_data_en['hashtags'].apply(string_to_list)
    cleaned_data_en['emojis'] = cleaned_data_en['emojis'].apply(string_to_list)
    cleaned_data_en['tokenized'] = cleaned_data_en['tokenized'].apply(string_to_list)
    cleaned_data_en['lemmatized'] = cleaned_data_en['lemmatized'].apply(string_to_list)
    cleaned_data_en['unique_lemmas'] = cleaned_data_en['unique_lemmas'].apply(string_to_list)
    cleaned_data_en['pos_and_lemma'] = cleaned_data_en['pos_and_lemma'].apply(string_to_list,
                                                                                            args=('), (',))
    # replace empty tweets with nan and remove nan tweets for the topic modelling analysis
    cleaned_data_en['cleaned'].replace('', np.nan, inplace=True)
    cleaned_data_en_no_nan = cleaned_data_en.dropna()
    cleaned_data_en_no_nan.index = range(0, len(cleaned_data_en_no_nan))

    # load the Finnish data
    cleaned_data_fi = pd.read_csv('preprocessed-data-fi.tsv', sep='\t', header=0)
    cleaned_data_fi['hashtags'] = cleaned_data_fi['hashtags'].apply(string_to_list)
    cleaned_data_fi['emojis'] = cleaned_data_fi['emojis'].apply(string_to_list)
    cleaned_data_fi['tokenized'] = cleaned_data_fi['tokenized'].apply(string_to_list)
    cleaned_data_fi['lemmatized'] = cleaned_data_fi['lemmatized'].apply(string_to_list)
    cleaned_data_fi['unique_lemmas'] = cleaned_data_fi['unique_lemmas'].apply(string_to_list)
    cleaned_data_fi['pos_and_lemma'] = cleaned_data_fi['pos_and_lemma'].apply(string_to_list,
                                                                                            args=('), (',))
    # replace empty tweets with nan and remove nan tweets for the topic modelling analysis
    cleaned_data_fi['cleaned'].replace('', np.nan, inplace=True)
    cleaned_data_fi_no_nan = cleaned_data_fi.dropna()
    cleaned_data_fi_no_nan.index = range(0, len(cleaned_data_fi_no_nan))


    ### Building and Running the Models - English ###

    # create a list of lemmatized tweets and exclude city names
    lemmatized_en_no_cities = []
    city_names_en = ['helsinki', 'tampere', 'espoo', 'oulu', 'vaasa', 'lappeenranta', 'kuopio']

    for sent in cleaned_data_en_no_nan['lemmatized']:
        sent_no_cities = [w for w in sent if w not in city_names_en]
        if sent_no_cities:
            lemmatized_en_no_cities.append(sent_no_cities)

    # create dictionary
    id2word_en = corpora.Dictionary(lemmatized_en_no_cities)
    # term document frequency
    corpus_en = [id2word_en.doc2bow(text) for text in lemmatized_en_no_cities]

    # set the hyperparameter values
    num_topics_range = range(1, 11)
    alphas = list(np.arange(0.01, 1, 0.3))
    alphas.append('symmetric')
    alphas.append('asymmetric')
    betas = list(np.arange(0.01, 1, 0.3))
    betas.append('symmetric')

    # run the models for the different hyperparameter values and compute the coherence score
    results_en, models_en = compute_coherence(corpus_en, id2word_en, lemmatized_en_no_cities, num_topics_range,
                                              alphas, betas)

    # store the results in a dataframe
    df_results_en = pd.DataFrame(results_en, columns=['number of topics', 'alpha', 'beta', 'coherence'])

    # create a dataframe containing the best combination of alpha and beta for each number of topics
    df_best_coherence_scores_en = pd.DataFrame(columns=['number of topics', 'alpha', 'beta', 'coherence', 'modelID'])

    for i in range(0, len(num_topics_range)):
        alpha, beta, coherence, model_id = find_optimal_alpha_beta_combination(df_results_en, num_topics=num_topics_range[i])
        df_best_coherence_scores_en.loc[i, 'number of topics'] = num_topics_range[i]
        df_best_coherence_scores_en.loc[i, 'alpha'] = alpha
        df_best_coherence_scores_en.loc[i, 'beta'] = beta
        df_best_coherence_scores_en.loc[i, 'coherence'] = coherence
        df_best_coherence_scores_en.loc[i, 'modelID'] = model_id

    # extract the model with the best coherence score for each number of topics
    model_en_1 = models_en[df_best_coherence_scores_en.loc[0, 'modelID']]
    model_en_2 = models_en[df_best_coherence_scores_en.loc[1, 'modelID']]
    model_en_3 = models_en[df_best_coherence_scores_en.loc[2, 'modelID']]
    model_en_4 = models_en[df_best_coherence_scores_en.loc[3, 'modelID']]
    model_en_5 = models_en[df_best_coherence_scores_en.loc[4, 'modelID']]
    model_en_6 = models_en[df_best_coherence_scores_en.loc[5, 'modelID']]
    model_en_7 = models_en[df_best_coherence_scores_en.loc[6, 'modelID']]
    model_en_8 = models_en[df_best_coherence_scores_en.loc[7, 'modelID']]
    model_en_9 = models_en[df_best_coherence_scores_en.loc[8, 'modelID']]
    model_en_10 = models_en[df_best_coherence_scores_en.loc[9, 'modelID']]

    # plot the models with pyLDAvis
    plot_topics(model_en_1, corpus_en, id2word_en, 'lda-result-en-1-topic.html')
    plot_topics(model_en_2, corpus_en, id2word_en, 'lda-result-en-2-topics.html')
    plot_topics(model_en_3, corpus_en, id2word_en, 'lda-result-en-3-topics.html')
    plot_topics(model_en_4, corpus_en, id2word_en, 'lda-result-en-4-topics.html')
    plot_topics(model_en_5, corpus_en, id2word_en, 'lda-result-en-5-topics.html')
    plot_topics(model_en_6, corpus_en, id2word_en, 'lda-result-en-6-topics.html')
    plot_topics(model_en_7, corpus_en, id2word_en, 'lda-result-en-7-topics.html')
    plot_topics(model_en_8, corpus_en, id2word_en, 'lda-result-en-8-topics.html')
    plot_topics(model_en_9, corpus_en, id2word_en, 'lda-result-en-9-topics.html')
    plot_topics(model_en_10, corpus_en, id2word_en, 'lda-result-en-10-topics.html')

    # plot the word clouds for each model
    plot_wordcloud_per_topic(model_en_1)
    plot_wordcloud_per_topic(model_en_2)
    plot_wordcloud_per_topic(model_en_3)
    plot_wordcloud_per_topic(model_en_4)
    plot_wordcloud_per_topic(model_en_5)
    plot_wordcloud_per_topic(model_en_6)
    plot_wordcloud_per_topic(model_en_7)
    plot_wordcloud_per_topic(model_en_8)
    plot_wordcloud_per_topic(model_en_9)
    plot_wordcloud_per_topic(model_en_10)

    # based on the coherence scores and the visualisations of the results, the best model is selected
    # for the English data, the model with 5 topics is selected


    ### Building and Running the Models - Finnish ###

    # read the updated stop word list (this list was chosen after noticing that the previously used stop word list was
    # not sufficient)
    f = open('stop_words_finnish.txt', 'r', encoding='utf-8')
    lines = f.readlines()
    stopwords_fi = []
    for word in lines:
        l = word.strip('\n')
        stopwords_fi.append(l)

    # create a list of lemmatized tweets, excluding additional stop words and city names
    lemmatized_fi_updated_no_cities = []
    city_names_fi = ['oulu', 'jyväskylä', 'tampere', 'turku', 'helsinki', 'jyvaskyla', 'lahti', 'kuopio', 'rauma',
                     'kouvola', 'pori', 'rovaniemi', 'espoo', 'laitila']

    for sent in cleaned_data_fi_no_nan['lemmatized']:
        sent_no_stop_words = [w for w in sent if w not in stopwords_fi]
        sent_no_stop_words_no_cities = [w for w in sent_no_stop_words if w not in city_names_fi]
        if sent_no_stop_words_no_cities:
            lemmatized_fi_updated_no_cities.append(sent_no_stop_words_no_cities)

    # create dictionary
    id2word_fi = corpora.Dictionary(lemmatized_fi_updated_no_cities)
    # term document frequency
    corpus_fi = [id2word_fi.doc2bow(text) for text in lemmatized_fi_updated_no_cities]

    # run the models for the different hyperparameter values and compute the coherence score
    results_fi, models_fi = compute_coherence(corpus_fi, id2word_fi, lemmatized_fi_updated_no_cities, num_topics_range,
                                              alphas, betas)

    # store the results in a dataframe
    df_results_fi = pd.DataFrame(results_fi, columns=['Number of topics', 'alpha', 'beta', 'coherence'])

    # create a dataframe containing the best combination of alpha and beta for each number of topics
    df_best_coherence_scores_fi = pd.DataFrame(columns=['Number of topics', 'alpha', 'beta', 'coherence', 'Model ID'])

    for i in range(0, len(num_topics_range)):
        alpha, beta, coherence, model_id = find_optimal_alpha_beta_combination(df_results_fi, num_topics=num_topics_range[i])
        df_best_coherence_scores_fi.loc[i, 'Number of topics'] = num_topics_range[i]
        df_best_coherence_scores_fi.loc[i, 'alpha'] = alpha
        df_best_coherence_scores_fi.loc[i, 'beta'] = beta
        df_best_coherence_scores_fi.loc[i, 'coherence'] = coherence
        df_best_coherence_scores_fi.loc[i, 'Model ID'] = model_id

    # extract the model with the best coherence score for each number of topics
    model_fi_1 = models_fi[df_best_coherence_scores_fi.loc[0, 'Model ID']]
    model_fi_2 = models_fi[df_best_coherence_scores_fi.loc[1, 'Model ID']]
    model_fi_3 = models_fi[df_best_coherence_scores_fi.loc[2, 'Model ID']]
    model_fi_4 = models_fi[df_best_coherence_scores_fi.loc[3, 'Model ID']]
    model_fi_5 = models_fi[df_best_coherence_scores_fi.loc[4, 'Model ID']]
    model_fi_6 = models_fi[df_best_coherence_scores_fi.loc[5, 'Model ID']]
    model_fi_7 = models_fi[df_best_coherence_scores_fi.loc[6, 'Model ID']]
    model_fi_8 = models_fi[df_best_coherence_scores_fi.loc[7, 'Model ID']]
    model_fi_9 = models_fi[df_best_coherence_scores_fi.loc[8, 'Model ID']]
    model_fi_10 = models_fi[df_best_coherence_scores_fi.loc[9, 'Model ID']]

    # plot the models with pyLDAvis
    plot_topics(model_fi_1, corpus_fi, id2word_fi, 'lda-result-fi-1-topic.html')
    plot_topics(model_fi_2, corpus_fi, id2word_fi, 'lda-result-fi-2-topics.html')
    plot_topics(model_fi_3, corpus_fi, id2word_fi, 'lda-result-fi-3-topics.html')
    plot_topics(model_fi_4, corpus_fi, id2word_fi, 'lda-result-fi-4-topics.html')
    plot_topics(model_fi_5, corpus_fi, id2word_fi, 'lda-result-fi-5-topics.html')
    plot_topics(model_fi_6, corpus_fi, id2word_fi, 'lda-result-fi-6-topics.html')
    plot_topics(model_fi_7, corpus_fi, id2word_fi, 'lda-result-fi-7-topics.html')
    plot_topics(model_fi_8, corpus_fi, id2word_fi, 'lda-result-fi-8-topics.html')
    plot_topics(model_fi_9, corpus_fi, id2word_fi, 'lda-result-fi-9-topics.html')
    plot_topics(model_fi_10, corpus_fi, id2word_fi, 'lda-result-fi-10-topics.html')

    # plot the word clouds for each model
    plot_wordcloud_per_topic(model_fi_1)
    plot_wordcloud_per_topic(model_fi_2)
    plot_wordcloud_per_topic(model_fi_3)
    plot_wordcloud_per_topic(model_fi_4)
    plot_wordcloud_per_topic(model_fi_5)
    plot_wordcloud_per_topic(model_fi_6)
    plot_wordcloud_per_topic(model_fi_7)
    plot_wordcloud_per_topic(model_fi_8)
    plot_wordcloud_per_topic(model_fi_9)
    plot_wordcloud_per_topic(model_fi_10)

    # based on the coherence scores and the visualisations of the results, the best model is selected
    # for the Finnish data, the model with 6 topics is selected


    ### Analysis - English ###

    ## Topic dominance across locations

    dominant_topic_df_en, topic_percentages_en = get_dominant_topic_per_tweet(model_en_5, corpus_en, lemmatized_en_no_cities)

    dominant_topic_df_en = dominant_topic_df_en.reset_index()
    dominant_topic_df_en.columns = ['Tweet_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Topic_Keywords', 'Text']

    # remove the city names also in the data file
    cleaned_data_en_lemmas_no_nan = remove_stop_words_and_city_names_after_lemmatization(cleaned_data_en_no_nan,
                                                                                        [], city_names_en)

    # remove empty tweets resulting from the city name removal
    cleaned_data_en_lemmas_no_nan['lemmatized'].replace([], np.nan, inplace=True)
    empty_list_idx = []
    for i in range(0, len(cleaned_data_en_lemmas_no_nan)):
        if cleaned_data_en_lemmas_no_nan.loc[i, 'lemmatized'] == []:
            empty_list_idx.append(i)
    cleaned_data_en_lemmas_no_nan = cleaned_data_en_lemmas_no_nan.drop(empty_list_idx)
    cleaned_data_en_lemmas_no_nan.index = range(0, len(cleaned_data_en_lemmas_no_nan))

    # add the regional and municipality size information to the data
    for i in range(0, len(cleaned_data_en_lemmas_no_nan)):
        id = cleaned_data_en_lemmas_no_nan.loc[i, 'account_id']
        cleaned_data_en_lemmas_no_nan.loc[i, 'region'] = info.loc[info['userid'] == id, 'region'].iloc[0]
        cleaned_data_en_lemmas_no_nan.loc[i, 'smaller_region'] = info.loc[info['userid'] == id, 'smaller_region'].iloc[0]
        cleaned_data_en_lemmas_no_nan.loc[i, 'size_group'] = info.loc[info['userid'] == id, 'size_group'].iloc[0]

    # get the dominant topic per account according to topic weighting, as well as the topic weightings for each region,
    # sub-region, and size group
    dmnt_topic_account_en, dmnt_topic_region_en, dmnt_topic_smaller_region_en, dmnt_topic_size_group_en = dominant_topic_analysis(
        cleaned_data_en_lemmas_no_nan, info, topic_percentages_en)

    # plot the topic weightings per region
    plot_topic_dominance(dmnt_topic_region_en, num_topics=5, group_type=('region', 'Region'), lang='English')

    # plot the topic weightings per sub-region
    plot_topic_dominance(dmnt_topic_smaller_region_en, num_topics=5, group_type=('smaller_region', 'Smaller Region'),
                         lang='English')

    # plot the topic weightings per municipality size group
    plot_topic_dominance(dmnt_topic_size_group_en, num_topics=5, group_type=('size_group', 'Size Group'),
                         lang='English')


    ## Individual Tweet Analysis

    tweet_topic_matrix_en = get_tweet_topic_matrix(model_en_5, corpus_en, num_topics=5)

    most_dominant_tweets_per_topic = get_most_dominant_tweets_per_topic(tweet_topic_matrix_en,
                                                                        cleaned_data_en_lemmas_no_nan,
                                                                        num_topics=5, num_tweets=10)

    # extract the tweet sample dataframes from the dictionary
    top_tweets_sample_0 = most_dominant_tweets_per_topic[0]
    top_tweets_sample_1 = most_dominant_tweets_per_topic[1]
    top_tweets_sample_2 = most_dominant_tweets_per_topic[2]
    top_tweets_sample_3 = most_dominant_tweets_per_topic[3]
    top_tweets_sample_4 = most_dominant_tweets_per_topic[4]

    # these samples can be examined manually to investigate some of the most dominant tweets for the topic


    ### Analysis - Finnish ###

    ## Topic dominance across locations
    dominant_topic_df_fi, topic_percentages_fi = get_dominant_topic_per_tweet(model_fi_6,
                                                                                  corpus_fi,
                                                                                  lemmatized_fi_updated_no_cities)
    dominant_topic_df_fi = dominant_topic_df_fi.reset_index()
    dominant_topic_df_fi.columns = ['Tweet_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Topic_Keywords', 'Text']


    # remove the stop words also in the data file
    cleaned_data_fi_lemmas_no_nan_no_stop_words = remove_stop_words_and_city_names_after_lemmatization(
        cleaned_data_fi_no_nan, stopwords_fi, city_names_fi)

    # remove empty tweets resulting from the stop word removal
    cleaned_data_fi_lemmas_no_nan_no_stop_words['lemmatized'].replace([], np.nan, inplace=True)
    empty_list_idx = []
    for i in range(0, len(cleaned_data_fi_lemmas_no_nan_no_stop_words)):
        if cleaned_data_fi_lemmas_no_nan_no_stop_words.loc[i, 'lemmatized'] == []:
            empty_list_idx.append(i)
    cleaned_data_fi_lemmas_no_nan_no_stop_words = cleaned_data_fi_lemmas_no_nan_no_stop_words.drop(empty_list_idx)
    cleaned_data_fi_lemmas_no_nan_no_stop_words.index = range(0, len(cleaned_data_fi_lemmas_no_nan_no_stop_words))

    # add the regional and municipality size information to the data
    for i in range(0, len(cleaned_data_fi_lemmas_no_nan_no_stop_words)):
        id = cleaned_data_fi_lemmas_no_nan_no_stop_words.loc[i, 'account_id']
        cleaned_data_fi_lemmas_no_nan_no_stop_words.loc[i, 'region'] = info.loc[info['userid'] == id, 'region'].iloc[0]
        cleaned_data_fi_lemmas_no_nan_no_stop_words.loc[i, 'smaller_region'] = info.loc[info['userid'] == id, 'smaller_region'].iloc[0]
        cleaned_data_fi_lemmas_no_nan_no_stop_words.loc[i, 'size_group'] = info.loc[info['userid'] == id, 'size_group'].iloc[0]

    # get the dominant topic per account according to topic weighting, as well as the topic weightings for each region,
    # sub-region, and size groups
    dmnt_topic_account_fi, dmnt_topic_region_fi, dmnt_topic_smaller_region_fi, dmnt_topic_size_group_fi = dominant_topic_analysis(
        cleaned_data_fi_lemmas_no_nan_no_stop_words, info, topic_percentages_fi)

    # plot the topic weightings per region
    plot_topic_dominance(dmnt_topic_region_fi, num_topics=6, group_type=('region', 'Region'), lang='Finnish')

    # plot the topic weightings per sub-region
    plot_topic_dominance(dmnt_topic_smaller_region_fi, num_topics=6, group_type=('smaller_region', 'Smaller Region'),
                         lang='Finnish')

    # plot the topic weightings per municipality size group
    plot_topic_dominance(dmnt_topic_size_group_fi, num_topics=6, group_type=('size_group', 'Size Group'),
                         lang='Finnish')


    ## Individual Tweet Analysis

    tweet_topic_matrix_fi = get_tweet_topic_matrix(model_fi_6, corpus_fi, num_topics=6)

    most_dominant_tweets_per_topic_fi = get_most_dominant_tweets_per_topic(tweet_topic_matrix_fi,
                                                                           cleaned_data_fi_lemmas_no_nan_no_stop_words,
                                                                           num_topics=6, num_tweets=10, threshold=0.8)

    # extract the tweet sample dataframes from the dictionary
    top_tweets_sample_fi_0 = most_dominant_tweets_per_topic_fi[0]
    top_tweets_sample_fi_1 = most_dominant_tweets_per_topic_fi[1]
    top_tweets_sample_fi_2 = most_dominant_tweets_per_topic_fi[2]
    top_tweets_sample_fi_3 = most_dominant_tweets_per_topic_fi[3]
    top_tweets_sample_fi_4 = most_dominant_tweets_per_topic_fi[4]
    top_tweets_sample_fi_5 = most_dominant_tweets_per_topic_fi[5]

    # these samples can be examined manually to investigate some of the most dominant tweets for the topic

