import json
import os
import matplotlib.pyplot as plt
import pandas as pd

from statistics import mean
from statistics import median


def interaction_distribution(tweets_lang, count_type='retweet_count'):
    """
    method for calculating interactional statistics (retweet count, reply count, like count, and quote count)

    returns the mean, median, minimum, and maximum of the selected count
    """
    stats = {'retweet_count': [], 'reply_count': [], 'like_count': [], 'quote_count': []}

    for tweet in tweets_lang:
        stats['retweet_count'].append(tweet['public_metrics']['retweet_count'])
        stats['reply_count'].append(tweet['public_metrics']['reply_count'])
        stats['like_count'].append(tweet['public_metrics']['like_count'])
        stats['quote_count'].append(tweet['public_metrics']['quote_count'])

    mean_count = mean(stats[count_type])
    median_count = median(stats[count_type])
    min_count = min(stats[count_type])
    max_count = max(stats[count_type])

    return mean_count, median_count, min_count, max_count


if __name__ == '__main__':
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

    ###### Distribution of Languages ######

    ### Language Distribution across Accounts ###

    # Distribution of languages across the full dataset
    lang_dist = pd.DataFrame()
    lang_dist['lang'] = ['Finnish', 'Swedish', 'English', 'Other']
    lang_dist.loc[0, 'Number of tweets'] = len(tweetsfi)
    lang_dist.loc[1, 'Number of tweets'] = len(tweetssv)
    lang_dist.loc[2, 'Number of tweets'] = len(tweetsen)
    lang_dist.loc[3, 'Number of tweets'] = len(tweetsoth)

    labels = lang_dist['lang']
    sizes = lang_dist['Number of tweets']
    cols = ['cornflowerblue', 'gold', 'mediumseagreen', 'crimson']

    fig, ax = plt.subplots()
    ax.pie(sizes, colors=cols)
    ax.legend(labels, bbox_to_anchor=(1.05, 0.5), loc='upper right', fontsize=12)
    ax.set_title('Distribution of Languages across Municipality Accounts', fontdict={'fontsize': 16})

    # Distribution of Finnish tweets across accounts
    account_dist_fi = pd.DataFrame()
    account_dist_fi['perc_fi'] = ['Percentage of Finnish Tweets < 20%', '20% <= Percentage of Finnish Tweets < 50%',
                                  '50% <= Percentage of Finnish Tweets < 80%', 'Percentage of Finnish Tweets >= 80%']
    account_dist_fi.loc[0, 'num_accounts'] = len(info[info['perc_fi'] < 0.2])
    account_dist_fi.loc[1, 'num_accounts'] = len(info[(info['perc_fi'] >= 0.2) & (info['perc_fi'] < 0.5)])
    account_dist_fi.loc[2, 'num_accounts'] = len(info[(info['perc_fi'] >= 0.5) & (info['perc_fi'] < 0.8)])
    account_dist_fi.loc[3, 'num_accounts'] = len(info[info['perc_fi'] >= 0.8])

    labels_fi = account_dist_fi['perc_fi']
    sizes_fi = account_dist_fi['num_accounts']
    cols_fi = ['#fb8072', '#bebada', '#ffffb3', '#80b1d3']

    fig, ax = plt.subplots()
    ax.pie(sizes_fi, colors=cols_fi)
    ax.legend(labels_fi, bbox_to_anchor=(1.55, 0.5), loc='center right', fontsize=12)
    ax.set_title('Distribution of Finnish Tweets across Accounts', fontdict={'fontsize': 16})

    # Distribution of English tweets across accounts
    account_dist_en = pd.DataFrame()
    account_dist_en['perc_en'] = ['Percentage of English Tweets < 2%', '2% <= Percentage of English Tweets < 5%',
                                  '5% <= Percentage of English Tweets < 8%', 'Percentage of English Tweets >= 8%']
    account_dist_en.loc[0, 'num_accounts'] = len(info[info['perc_en'] < 0.02])
    account_dist_en.loc[1, 'num_accounts'] = len(info[(info['perc_en'] >= 0.02) & (info['perc_en'] < 0.05)])
    account_dist_en.loc[2, 'num_accounts'] = len(info[(info['perc_en'] >= 0.05) & (info['perc_en'] < 0.08)])
    account_dist_en.loc[3, 'num_accounts'] = len(info[info['perc_en'] >= 0.08])

    labels_en = account_dist_en['perc_en']
    sizes_en = account_dist_en['num_accounts']
    cols_en = ['#7fc97f', '#beaed4', '#fdc086', '#ffff99']

    fig, ax = plt.subplots()
    ax.pie(sizes_en, colors=cols_en)
    ax.legend(labels_en, bbox_to_anchor=(1.55, 0.5), loc='center right', fontsize=12)
    ax.set_title('Distribution of English Tweets across Accounts', fontdict={'fontsize': 16})

    ### Language Distribution across Locations ###

    # adjust some of the city names so they are compatible with the data obtained from Statistics Finland
    info['updated_location'] = info['location']

    info.loc[info['location'] == 'Raseborg', 'updated_location'] = 'Raasepori'
    info.loc[info['location'] == 'Toijala', 'updated_location'] = 'Akaa'
    info.loc[info['location'] == 'Turenki', 'updated_location'] = 'Janakkala'
    info.loc[info['location'] == 'Mänttä', 'updated_location'] = 'Mänttä-Vilppula'
    info.loc[info['location'] == 'Kausala', 'updated_location'] = 'Iitti'
    info.loc[info['location'] == 'Pulkkila', 'updated_location'] = 'Siikalatva'
    info.loc[info['location'] == 'Lappträsk', 'updated_location'] = 'Lapinjärvi'
    info.loc[info['location'] == 'Vääksy', 'updated_location'] = 'Asikkala'
    info.loc[info['location'] == 'Kimito', 'updated_location'] = 'Kimitoön'
    info.loc[info['location'] == 'Simpele', 'updated_location'] = 'Rautjärvi'
    info.loc[info['location'] == 'Oitti', 'updated_location'] = 'Hausjärvi'

    # load the information on regional divisions
    regional_info = pd.read_csv('info/regional_info.csv', sep=';', encoding='latin-1')

    for i in range(0, len(info)):
        location = info.loc[i, 'updated_location']
        if location in regional_info['sourcename'].values:
            info.loc[i, 'region'] = regional_info.loc[regional_info['sourcename'] == location, 'targetname'].item()

    # load more detailed regional information from Statistics Finland
    stats_fi = pd.read_csv(
        'info/en23_municipalities_and_regional_divisions_based_on_municipalities_finnish_swedish_english.csv',
        sep=';')

    for i in range(0, len(info)):
        location = info.loc[i, 'updated_location']
        if location in stats_fi['Name of municipality in English'].values:
            info.loc[i, 'sub_region'] = stats_fi.loc[
                stats_fi['Name of municipality in English'] == location, 'Name of region in English'].item()

    ## Major Regions
    regions = pd.DataFrame(columns=['region', 'tweets_fi', 'tweets_en', 'tweets_total', 'perc_fi', 'perc_en'])

    regionslist = ['Helsinki-Uusimaa', 'Northern and Eastern Finland', 'Southern Finland', 'Western Finland',
                   'Åland']

    for i in range(0, len(regionslist)):
        region = regionslist[i]
        regions.loc[i, 'region'] = region
        regionsubset = info[info['region'] == region]
        regions.loc[i, 'tweets_fi'] = sum(regionsubset['tweets_fi'])
        regions.loc[i, 'tweets_en'] = sum(regionsubset['tweets_en'])
        regions.loc[i, 'tweets_total'] = sum(regionsubset['tweets_collected'])
        regions.loc[i, 'perc_fi'] = regions.loc[i, 'tweets_fi'] / regions.loc[i, 'tweets_total']
        regions.loc[i, 'perc_en'] = regions.loc[i, 'tweets_en'] / regions.loc[i, 'tweets_total']

    # visualisation
    regions = regions.sort_values(by='region')
    regions.plot(x='region', y=['perc_fi', 'perc_en'], kind='bar', color=['cornflowerblue', 'mediumseagreen'], width=0.4)
    plt.xticks(fontsize=10, rotation=15)
    plt.xlabel('region', fontdict={'fontsize': 12})
    plt.ylabel('proportion', fontdict={'fontsize': 12})
    plt.legend(['Finnish tweets', 'English tweets'])
    plt.title('Proportion of Finnish and English Tweets per Region', fontdict={'fontsize': 16, 'fontweight': 'bold'})


    ## Sub-regions
    subregions = pd.DataFrame(columns=['region', 'tweets_fi', 'tweets_en', 'tweets_total', 'perc_fi', 'perc_en'])

    subregionslist = ['Åland', 'Southwest Finland', 'Satakunta', 'Kanta-Häme', 'Pirkanmaa', 'Päijät-Häme', 'Central Finland',
                  'South Ostrobothnia', 'Ostrobothnia', 'Central Ostrobothnia', 'Uusimaa', 'Kymenlaakso', 'South Karelia',
                  'South Savo', 'North Savo', 'North Karelia', 'North Ostrobothnia', 'Kainuu', 'Lapland']

    for i in range(0, len(subregionslist)):
        region = subregionslist[i]
        subregions.loc[i, 'region'] = region
        regionsubset = info[info['smaller_region'] == region]
        subregions.loc[i, 'tweets_fi'] = sum(regionsubset['tweets_fi'])
        subregions.loc[i, 'tweets_en'] = sum(regionsubset['tweets_en'])
        subregions.loc[i, 'tweets_total'] = sum(regionsubset['tweets_collected'])
        subregions.loc[i, 'perc_fi'] = subregions.loc[i, 'tweets_fi'] / subregions.loc[i, 'tweets_total']
        subregions.loc[i, 'perc_en'] = subregions.loc[i, 'tweets_en'] / subregions.loc[i, 'tweets_total']

    # visualisation
    subregions = subregions.sort_values(by='region')
    subregions.plot(x='region', y=['perc_fi', 'perc_en'], kind='bar', color=['cornflowerblue', 'mediumseagreen'])
    plt.xticks(fontsize=9, rotation=25)
    plt.xlabel('sub-region', fontdict={'fontsize': 12})
    plt.ylabel('proportion', fontdict={'fontsize': 12})
    plt.legend(['Finnish tweets', 'English tweets'])
    plt.title('Proportion of Finnish and English Tweets per Sub-Region', fontdict={'fontsize': 16, 'fontweight': 'bold'})

    ## Municipality size groups

    # load residential information
    size_info = pd.read_csv('info/kunta_vaki2022.csv')

    # update the metadata with municipality size and size groups
    for i in range(0, len(info)):
        location = info.loc[i, 'updated_location']
        if location in size_info['name'].values:
            info.loc[i, 'size_of_municipality'] = size_info.loc[size_info['name'] == location, 'vaesto'].item()

    for i in range(0, len(info)):
        metropolitan_list = ['Helsinki', 'Espoo', 'Vantaa', 'Kauniainen']
        if info.loc[i, 'location'] in metropolitan_list:
            info.loc[i, 'size_group'] = 'metropolitan'
        elif info.loc[i, 'size_of_municipality'] > 100000:
            info.loc[i, 'size_group'] = 'bigger_cities'
        elif info.loc[i, 'size_of_municipality'] <= 100000 and info.loc[i, 'size_of_municipality'] > 10000:
            info.loc[i, 'size_group'] = 'smaller_cities'
        elif info.loc[i, 'size_of_municipality'] <= 10000:
            info.loc[i, 'size_group'] = 'rural'

    # create the dataframe for the size groups
    municipality_groups = pd.DataFrame(columns=['group', 'accounts'])
    municipality_groups['group'] = ['metropolitan', 'bigger_cities', 'smaller_cities', 'rural']

    for i in range(0, len(municipality_groups)):
        groupsubset = info[info['size_group'] == municipality_groups.loc[i, 'group']]
        municipality_groups.loc[i, 'accounts'] = len(groupsubset)
        municipality_groups.loc[i, 'tweets_fi'] = sum(groupsubset['tweets_fi'])
        municipality_groups.loc[i, 'tweets_en'] = sum(groupsubset['tweets_en'])
        municipality_groups.loc[i, 'tweets_total'] = sum(groupsubset['tweets_collected'])
        municipality_groups.loc[i, 'perc_fi'] = municipality_groups.loc[i, 'tweets_fi'] / municipality_groups.loc[i, 'tweets_total']
        municipality_groups.loc[i, 'perc_en'] = municipality_groups.loc[i, 'tweets_en'] / municipality_groups.loc[i, 'tweets_total']

    # visualisation
    municipality_groups = municipality_groups.iloc[::-1] # reverting the dataframe so that it is sorted according to increasing municipality size
    municipality_groups.loc[2, 'group'] = 'smaller cities'
    municipality_groups.loc[1, 'group'] = 'bigger cities'
    municipality_groups.plot(x='group', y=['perc_fi', 'perc_en'], kind='bar', color=['cornflowerblue', 'mediumseagreen'], width=0.3)
    plt.xticks(fontsize=10, rotation=15)
    plt.xlabel('size group', fontdict={'fontsize': 12})
    plt.ylabel('proportion', fontdict={'fontsize': 12})
    plt.legend(['Finnish tweets', 'English tweets'])
    plt.title('Proportion of Finnish and English Tweets per Size Group', fontdict={'fontsize': 16, 'fontweight': 'bold'})


    ###### Interaction Analysis ######

    # set up a dataframe for the interaction data
    interaction = pd.DataFrame(columns=['type', 'language', 'mean', 'median', 'min', 'max'])
    interaction_types = ['retweet', 'reply', 'like', 'quote']
    tweet_groups = {'fi': tweetsfi, 'en': tweetsen}

    for i in range(0, len(interaction_types)):
        tp = interaction_types[i]
        for lng in tweet_groups.keys():
            mean_count, median_count, min_count, max_count = interaction_distribution(tweet_groups[lng], tp + '_count')
            row = [tp, lng, mean_count, median_count, min_count, max_count]
            interaction.loc[len(interaction)] = row

