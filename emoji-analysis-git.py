import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from statistics import mean
from random import sample


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
                    item = (item_list[0].strip('\''), item_list[1].strip('\''), item_list[2].strip('\''), item_list[3].strip('\''))
                items.append(item)

    return items

def add_group_info_to_data(data, info):
    """
    method for adding the regional information to the data

    returns the modified dataframe containing the regional information
    """
    for i in range(0, len(data)):
        id = data.loc[i, 'account_id']
        data.loc[i, 'location'] = info.loc[info['userid'] == id, 'location'].iloc[0]
        data.loc[i, 'region'] = info.loc[info['userid'] == id, 'region'].iloc[0]
        data.loc[i, 'smaller_region'] = info.loc[info['userid'] == id, 'smaller_region'].iloc[0]
        data.loc[i, 'size_group'] = info.loc[info['userid'] == id, 'size_group'].iloc[0]

    return data

def sample_tweets(ids, num_samples=1000, num_sets=10):
    """
    method for sampling random tweet IDs

    returns the sample of IDs
    """
    samples = pd.DataFrame()

    for i in range(0, num_sets):
        samples[('samples_' + str(i))] = sample(ids, num_samples)

    return samples

def emojis_per_tweets_sample(data, num_samples=1000, num_sets=10):
    """
    method for calculating the average number of emoji tokens and types per samples of 1000 tweets

    returns a dataframe that contains the average for each sample
    """
    ids = list(data.index)

    samples = sample_tweets(ids, num_samples=num_samples, num_sets=num_sets)

    emojis_per_tweet_samples = pd.DataFrame()

    for i in range(0, num_sets):
        tweet_subset = data.loc[samples['samples_' + str(i)]]
        emoji_list = tweet_subset['emojis'].apply(len)
        unique_emojis = []
        for twt in tweet_subset['emojis']:
            for em in twt:
                if em not in unique_emojis:
                    unique_emojis.append(em)
        emojis_per_tweet_samples.loc[i, 'sample'] = i
        emojis_per_tweet_samples.loc[i, 'average_emojis_per_tweet'] = mean(emoji_list)
        emojis_per_tweet_samples.loc[i, 'number_emojis_total'] = sum(emoji_list)
        emojis_per_tweet_samples.loc[i, 'number_distinct_emojis'] = len(unique_emojis)

    return emojis_per_tweet_samples

def get_unique_emojis(data):
    """
    method for retrieving the unique emojis from a dataset

    returns a list of unique emojis
    """
    unique_emojis = []

    for i in range(0, len(data)):
        for emoji in data.loc[i, 'emojis']:
            if emoji not in unique_emojis:
                unique_emojis.append(emoji)

    return unique_emojis


def get_emoji_popularity(data):
    """
    method for calculating the popularity of emojis in the data
    for each unique emoji, the number of tweets that contain at least one instance of it is divided by the total number
    of tweets

    returns a dataframe containing each emoji along with its relative popularity
    """
    emoji_popularity_dict = dict()

    for i in range(0, len(data)):
        unique_emojis_in_tweet = pd.unique(data.loc[i, 'emojis'])
        for emoji in unique_emojis_in_tweet:
            if emoji in emoji_popularity_dict.keys():
                emoji_popularity_dict[emoji] += 1
            else:
                emoji_popularity_dict[emoji] = 1

    # transform the dictionary into a dataframe and sort it
    emoji_popularity_df = pd.DataFrame(emoji_popularity_dict.items(), columns=['emoji', 'frequency'])
    emoji_popularity_df['relative_popularity'] = emoji_popularity_df['frequency'] / len(data)
    emoji_popularity_df_sorted = emoji_popularity_df.sort_values(by='relative_popularity', ascending=False)
    emoji_popularity_df_sorted.index = range(0, len(emoji_popularity_df_sorted))

    return emoji_popularity_df_sorted


def get_emoji_statistics_per_group(data, group_type='region'):
    """
    method for calculating the emoji statistics for regions, sub-regions, or size groups
    statistics: total number of emoji tokens
                average number of emoji tokens per tweet
                average number of emoji types per tweet
                number of tweets containing at least one emoji
                proportion of tweets containing at least one emoji
                average number of emoji tokens per tweet across tweets that contain emojis
                average number of emoji types per tweet across tweets that contain emojis

    returns a dataframe containing the statistics for each region, sub-region, or size group
    """
    emoji_data = pd.DataFrame(columns=[group_type, 'num_emojis', 'average_num_emojis_per_tweet',
                                       'average_num_unique_emojis_per_tweet', 'num_tweets_with_emoji',
                                       'perc_tweets_with_emoji'])


    for i, group in enumerate(list(pd.unique(data[group_type]))):
        group_subset = data[data[group_type] == group]
        # total number of emojis per tweet
        group_emoji_list_lens = group_subset['emojis'].apply(len)
        # number of unique emojis per tweet
        group_unique_emojis_list = group_subset['emojis'].apply(pd.unique)
        group_unique_emojis_list_lens = group_unique_emojis_list.apply(len)
        emoji_data.loc[i, group_type] = group
        emoji_data.loc[i, 'num_emojis'] = sum(group_emoji_list_lens)
        emoji_data.loc[i, 'num_tweets_with_emoji'] = len(group_emoji_list_lens.loc[group_emoji_list_lens > 0])
        emoji_data.loc[i, 'perc_tweets_with_emoji'] = len(group_emoji_list_lens.loc[group_emoji_list_lens > 0]) / len(group_emoji_list_lens)
        if sum(group_emoji_list_lens) > 0:
            emoji_data.loc[i, 'average_num_emojis_per_tweet'] = mean(group_emoji_list_lens)
            emoji_data.loc[i, 'average_num_unique_emojis_per_tweet'] = mean(group_unique_emojis_list_lens)
            # average number of emojis per tweet based on the subset of tweets that contain at least one emoji
            tweets_with_emoji = group_emoji_list_lens.loc[group_emoji_list_lens > 0]
            emoji_data.loc[i, 'average_num_emojis_per_tweet_across_tweets_with_emoji'] = mean(tweets_with_emoji)
            tweets_with_emoji_unique = group_unique_emojis_list_lens.loc[group_unique_emojis_list_lens > 0]
            emoji_data.loc[i, 'average_num_unique_emojis_per_tweet_across_tweets_with_emoji'] = mean(tweets_with_emoji_unique)
        else:
            emoji_data.loc[i, 'average_num_emojis_per_tweet'] = 0
            emoji_data.loc[i, 'average_num_unique_emojis_per_tweet'] = 0
            emoji_data.loc[i, 'average_num_emojis_per_tweet_across_tweets_with_emoji'] = 0
            emoji_data.loc[i, 'average_num_unique_emojis_per_tweet_across_tweets_with_emoji'] = 0

    return emoji_data


def plot_emoji_data_across_languages(emoji_data_list, langs, to_plot, fontsize, x_rotation, title, ylabel, group_type=['region', 'Region']):
    """
    method for visualising the emoji statistics (proportion of emoji-containing tweets or number of unique emoji types
    per tweet) across regions, sub-regions, or size groups
    """
    cols = ['#4293FF', '#22A055']

    xlabels = emoji_data_list[0][group_type[0]]
    data = {}
    for i in range(0, len(emoji_data_list)):
        data[langs[i]] = list(emoji_data_list[i][to_plot])

    x = np.arange(len(xlabels))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots()
    for lang, values in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, values, width, label=(lang + ' data'), color=cols[multiplier])
        multiplier += 1

    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x + math.floor(len(langs)/2)*width*0.5, xlabels, fontsize=fontsize, rotation=x_rotation)
    ax.legend(labels=langs, loc='best', ncols=math.ceil(len(langs)/2))

    plt.show()



if __name__ == '__main__':

    ###### Emoji Analysis ######

    # load the English data
    cleaned_data_en = pd.read_csv('preprocessed-data-en.tsv', sep='\t', header=0)
    cleaned_data_en['hashtags'] = cleaned_data_en['hashtags'].apply(string_to_list)
    cleaned_data_en['emojis'] = cleaned_data_en['emojis'].apply(string_to_list)
    cleaned_data_en['tokenized'] = cleaned_data_en['tokenized'].apply(string_to_list)
    cleaned_data_en['lemmatized'] = cleaned_data_en['lemmatized'].apply(string_to_list)
    cleaned_data_en['unique_lemmas'] = cleaned_data_en['unique_lemmas'].apply(string_to_list)
    cleaned_data_en['pos_and_lemma'] = cleaned_data_en['pos_and_lemma'].apply(string_to_list, args=('), (', ))

    # load the Finnish data
    cleaned_data_fi = pd.read_csv('preprocessed-data-fi.tsv', sep='\t', header=0)
    cleaned_data_fi['hashtags'] = cleaned_data_fi['hashtags'].apply(string_to_list)
    cleaned_data_fi['emojis'] = cleaned_data_fi['emojis'].apply(string_to_list)
    cleaned_data_fi['tokenized'] = cleaned_data_fi['tokenized'].apply(string_to_list)
    cleaned_data_fi['lemmatized'] = cleaned_data_fi['lemmatized'].apply(string_to_list)
    cleaned_data_fi['unique_lemmas'] = cleaned_data_fi['unique_lemmas'].apply(string_to_list)
    cleaned_data_fi['pos_and_lemma'] = cleaned_data_fi['pos_and_lemma'].apply(string_to_list, args=('), (', ))

    # note: empty tweets are not excluded, as they might still contain emojis even though they do not contain any text

    ### Emoji Statistics ###

    # extract the number of emojis in each tweet
    emoji_list_lens_en = cleaned_data_en['emojis'].apply(len)
    emoji_list_lens_fi = cleaned_data_fi['emojis'].apply(len)

    # extract the unique emoji types in each tweet and the number of unique emoji types in each tweet
    unique_emojis_list_en = cleaned_data_en['emojis'].apply(pd.unique)
    unique_emojis_list_fi = cleaned_data_fi['emojis'].apply(pd.unique)
    unique_emojis_list_lens_en = unique_emojis_list_en.apply(len)
    unique_emojis_list_lens_fi = unique_emojis_list_fi.apply(len)

    emoji_statistics = pd.DataFrame(columns=['lang', 'emoji_tokens_total', 'average_num_emoji_tokens', 'average_num_emoji_types',
                                             'prop_tweets_with_emojis', 'average_num_emoji_tokens_across_emoji_containing_tweets',
                                             'average_num_emoji_types_across_emoji_containing_tweets'])

    emoji_statistics.loc[0, 'lang'] = 'en'
    emoji_statistics.loc[1, 'lang'] = 'fi'
    # total number of emojis across all tweets
    emoji_statistics.loc[0, 'emoji_tokens_total'] = sum(emoji_list_lens_en)
    emoji_statistics.loc[1, 'emoji_tokens_total'] = sum(emoji_list_lens_fi)
    # average number of emoji tokens per tweet
    emoji_statistics.loc[0, 'average_num_emoji_tokens'] = mean(emoji_list_lens_en)
    emoji_statistics.loc[1, 'average_num_emoji_tokens'] = mean(emoji_list_lens_fi)
    # average number of unique emoji types per tweet
    emoji_statistics.loc[0, 'average_num_emoji_types'] = unique_emojis_list_en.apply(len)
    emoji_statistics.loc[1, 'average_num_emoji_types'] = unique_emojis_list_fi.apply(len)

    # proportion of tweets that contain emojis
    tweets_with_emoji_en = emoji_list_lens_en.loc[emoji_list_lens_en > 0]
    tweets_with_emoji_fi = emoji_list_lens_fi.loc[emoji_list_lens_fi > 0]

    emoji_statistics.loc[0, 'prop_tweets_with_emojis'] = len(tweets_with_emoji_en) / len(emoji_list_lens_en)
    emoji_statistics.loc[1, 'prop_tweets_with_emojis'] = len(tweets_with_emoji_fi) / len(emoji_list_lens_fi)

    # average number of emoji tokens per tweet based on the subset of tweets that contain at least one emoji
    emoji_statistics.loc[0, 'average_num_emoji_tokens_across_emoji_containing_tweets'] = mean(tweets_with_emoji_en)
    emoji_statistics.loc[1, 'average_num_emoji_tokens_across_emoji_containing_tweets'] = mean(tweets_with_emoji_fi)

    # average number of unique emojis per tweet based on the subset of tweets that contain at least one emoji
    tweets_with_unique_emoji_en = unique_emojis_list_lens_en.loc[unique_emojis_list_lens_en > 0]
    tweets_with_unique_emoji_fi = unique_emojis_list_lens_fi.loc[unique_emojis_list_lens_fi > 0]

    emoji_statistics.loc[0, 'average_num_emoji_types_across_emoji_containing_tweets'] = mean(tweets_with_unique_emoji_en)
    emoji_statistics.loc[1, 'average_num_emoji_types_across_emoji_containing_tweets'] = mean(tweets_with_unique_emoji_fi)


    # get the number of emoji tokens and types per 1000 tweets
    emojis_per_tweet_samples_en = emojis_per_tweets_sample(cleaned_data_en, num_samples=1000, num_sets=10)
    emojis_per_tweet_samples_fi = emojis_per_tweets_sample(cleaned_data_fi, num_samples=1000, num_sets=10)

    # get the total average of emoji tokens and types across all samples in each language
    emojis_per_tweet_samples_en.loc['total_average', 'average_emojis_per_tweet'] = mean(emojis_per_tweet_samples_en.loc[0:9, 'average_emojis_per_tweet'])
    emojis_per_tweet_samples_en.loc['total_average', 'number_emojis_total'] = mean(emojis_per_tweet_samples_en.loc[0:9, 'number_emojis_total'])
    emojis_per_tweet_samples_en.loc['total_average', 'number_distinct_emojis'] = mean(emojis_per_tweet_samples_en.loc[0:9, 'number_distinct_emojis'])

    emojis_per_tweet_samples_fi.loc['total_average', 'average_emojis_per_tweet'] = mean(emojis_per_tweet_samples_fi.loc[0:9, 'average_emojis_per_tweet'])
    emojis_per_tweet_samples_fi.loc['total_average', 'number_emojis_total'] = mean(emojis_per_tweet_samples_fi.loc[0:9, 'number_emojis_total'])
    emojis_per_tweet_samples_fi.loc['total_average', 'number_distinct_emojis'] = mean(emojis_per_tweet_samples_fi.loc[0:9, 'number_distinct_emojis'])


    ### Emoji Popularity ###

    # extract the subset of tweets that contain emojis from the data
    cleaned_data_en_emoji_subset = cleaned_data_en[cleaned_data_en['emojis'].astype(bool)]
    cleaned_data_fi_emoji_subset = cleaned_data_fi[cleaned_data_fi['emojis'].astype(bool)]

    cleaned_data_en_emoji_subset.index = range(0, len(cleaned_data_en_emoji_subset))
    cleaned_data_fi_emoji_subset.index = range(0, len(cleaned_data_fi_emoji_subset))

    # get the lists of unique emojis across the whole datasets
    unique_emojis_en = get_unique_emojis(cleaned_data_en_emoji_subset)
    unique_emojis_fi = get_unique_emojis(cleaned_data_fi_emoji_subset)

    # get the relative popularity of each emoji
    emoji_popularity_df_en = get_emoji_popularity(cleaned_data_en_emoji_subset)
    emoji_popularity_df_fi = get_emoji_popularity(cleaned_data_fi_emoji_subset)

    # retrieve the 20 most popular emojis in each language for comparison
    most_popular_emojis_en = list(emoji_popularity_df_en.loc[0:20, 'emoji'])
    most_popular_emojis_fi = list(emoji_popularity_df_fi.loc[0:20, 'emoji'])


    #### Emoji Use Across Locations ####

    # load the metadata
    info = pd.read_csv('info/info-municipalities-only.csv')

    # add regional info to data
    cleaned_data_en = add_group_info_to_data(cleaned_data_en, info)
    cleaned_data_fi = add_group_info_to_data(cleaned_data_fi, info)

    # get emoji statistics for the major regions
    emoji_data_regions_en = get_emoji_statistics_per_group(cleaned_data_en, group_type='region')
    emoji_data_regions_fi = get_emoji_statistics_per_group(cleaned_data_fi, group_type='region')

    emoji_data_regions_en = emoji_data_regions_en.sort_values(by='region')
    emoji_data_regions_en.index = range(0, len(emoji_data_regions_en))
    # exclude Åland from the English dataframe, as there are no tweets with emojis from Åland
    emoji_data_regions_en = emoji_data_regions_en.drop(4, axis='index')
    emoji_data_regions_fi = emoji_data_regions_fi.sort_values(by='region')
    emoji_data_regions_fi.index = range(0, len(emoji_data_regions_fi))
    # note: the Finnish dataset already excludes Åland, as there are no Finnish tweets from Åland

    # get emoji statistics for the major sub-regions
    emoji_data_smaller_regions_en = get_emoji_statistics_per_group(cleaned_data_en, group_type='smaller_region')
    emoji_data_smaller_regions_fi = get_emoji_statistics_per_group(cleaned_data_fi, group_type='smaller_region')

    emoji_data_smaller_regions_en = emoji_data_smaller_regions_en.sort_values(by='smaller_region')
    emoji_data_smaller_regions_en.index = range(0, len(emoji_data_smaller_regions_en))
    # exclude Åland from the English dataframe, as there are no tweets with emojis from Åland
    emoji_data_smaller_regions_en = emoji_data_smaller_regions_en.drop(18, axis='index')
    emoji_data_smaller_regions_fi = emoji_data_smaller_regions_fi.sort_values(by='smaller_region')
    emoji_data_smaller_regions_fi.index = range(0, len(emoji_data_smaller_regions_fi))

    # get emoji statistics for the size groups
    emoji_data_size_group_en = get_emoji_statistics_per_group(cleaned_data_en, group_type='size_group')
    emoji_data_size_group_fi = get_emoji_statistics_per_group(cleaned_data_fi, group_type='size_group')

    # change the names to make them suitable for visualisation
    emoji_data_size_group_en.loc[emoji_data_size_group_en['size_group'] == 'smaller_cities', 'size_group'] = 'smaller cities'
    emoji_data_size_group_en.loc[emoji_data_size_group_en['size_group'] == 'bigger_cities', 'size_group'] = 'bigger cities'
    emoji_data_size_group_fi.loc[emoji_data_size_group_fi['size_group'] == 'smaller_cities', 'size_group'] = 'smaller cities'
    emoji_data_size_group_fi.loc[emoji_data_size_group_fi['size_group'] == 'bigger_cities', 'size_group'] = 'bigger cities'

    # plot the proportion of tweets with emojis across major regions
    plot_emoji_data_across_languages([emoji_data_regions_fi, emoji_data_regions_en], langs=['Finnish', 'English'],
                                     to_plot='perc_tweets_with_emoji', fontsize=12, x_rotation=10,
                                     title='Proportion of Tweets with Emojis for Finnish and English Tweets by Region',
                                     ylabel='Proportion of Tweets', group_type=['region', 'Region'])

    # plot the proportion of tweets with emojis across sub-regions
    plot_emoji_data_across_languages([emoji_data_smaller_regions_fi, emoji_data_smaller_regions_en], langs=['Finnish', 'English'],
                                     to_plot='perc_tweets_with_emoji', fontsize=8, x_rotation=35,
                                     title='Proportion of Tweets with Emojis for Finnish and English Tweets by Smaller Region',
                                     ylabel='Proportion of Tweets', group_type=['smaller_region', 'Smaller Region'])

    # plot the proportion of tweets with emojis across size groups
    plot_emoji_data_across_languages([emoji_data_size_group_fi, emoji_data_size_group_en], langs=['Finnish', 'English'],
                                     to_plot='perc_tweets_with_emoji', fontsize=12, x_rotation=10,
                                     title='Proportion of Tweets with Emojis for Finnish and English Tweets by Size Group',
                                     ylabel='Proportion of Tweets', group_type=['size_group', 'Size Group'])

    # plot the average number of unique emojis per tweet across tweets with emojis across major regions
    plot_emoji_data_across_languages([emoji_data_regions_fi, emoji_data_regions_en], langs=['Finnish', 'English'],
                                     to_plot='average_num_unique_emojis_per_tweet_across_tweets_with_emoji', fontsize=12, x_rotation=10,
                                     title='Average Number of Unique Emojis across Tweets with Emojis for Finnish and English Tweets by Region',
                                     ylabel='Average number of unique emojis', group_type=['region', 'Region'])

    # plot the average number of unique emojis per tweet across tweets with emojis across sub-regions
    plot_emoji_data_across_languages([emoji_data_smaller_regions_fi, emoji_data_smaller_regions_en], langs=['Finnish', 'English'],
                                     to_plot='average_num_unique_emojis_per_tweet_across_tweets_with_emoji', fontsize=8, x_rotation=35,
                                     title='Average Number of Unique Emojis across Tweets with Emojis for Finnish and English Tweets by Smaller Region',
                                     ylabel='Average number of unique emojis', group_type=['smaller_region', 'Smaller Region'])

    # plot the average number of unique emojis per tweet across tweets with emojis across size groups
    plot_emoji_data_across_languages([emoji_data_size_group_fi, emoji_data_size_group_en], langs=['Finnish', 'English'],
                                     to_plot='average_num_unique_emojis_per_tweet_across_tweets_with_emoji', fontsize=12, x_rotation=10,
                                     title='Average Number of Unique Emojis across Tweets with Emojis for Finnish and English Tweets by Size Group',
                                     ylabel='Average number of unique emojis', group_type=['size_group', 'Size Group'])



