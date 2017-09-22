# import modules
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xgb

get_ipython().magic('matplotlib inline')

# load data
train_variants = pd.read_csv('training_variants', sep = ',')
test_variants = pd.read_csv('test_variants', sep = ',')
train_text = pd.read_csv('training_text', sep = '\|\|', skiprows=1, engine='python', names=["ID","text"])
test_text = pd.read_csv('test_text', sep = '\|\|', skiprows=1, engine='python' ,names=["ID","text"])


# combine variant and text datasets
train = train_variants.merge(train_text, on='ID', how='left')
test = test_variants.merge(test_text, on='ID', how='left')

ID = test.ID

print('train shape: ', train.shape)
print('test shape: ', test.shape)


combined = pd.concat([train, test], axis=0)

print('Number of Unique Genes: ', combined.Gene.nunique())
print('Number of Unique Variations: ', combined.Variation.nunique())

combined.Gene.unique()

# Remove useless symbols from description
def clean(s):    
    # Remove any tags:
    cleaned = re.sub(r"(?s)<.?>", " ", s)
    # Keep only regular chars:
    cleaned = re.sub(r"[^A-Za-z0-9(),*!?\'\`]", " ", cleaned)
    # Remove unicode chars
    cleaned = re.sub("\\\\u(.){4}", " ", cleaned)
    
    return cleaned.strip()

# clean text
train['clean_text'] = train.text.apply(lambda x: clean(x))
test['clean_text'] = test.text.apply(lambda x: clean(x))

# convert Gene, Variation, and text to lowercase
train['clean_text'] = train.clean_text.str.lower()
test['clean_text'] = test.clean_text.str.lower()

train['Gene'] = train.Gene.str.lower()
test['Gene'] = test.Gene.str.lower()

train['Variation'] = train.Variation.str.lower()
test['Variation'] = test.Variation.str.lower()

# create list of all words in text
train['text_list'] = train.clean_text.str.split(' ')
test['text_list'] = test.clean_text.str.split(' ')

# # this code is correct, but it takes awhile to run. I've therefore saved
# # it's results to a .csv file (parsed_data.csv) where I will upload from.
#######

# # only find text surrounding the mutation's Variation (i.e. within +/- 250 words)
# def textParser(df):
#     n = 250
#     df['text_list_parsed'] = ''
    
#     end = len(df['text_list'])
#     text_list_parsed = []
#     final_text_list_parsed = []

#     print(df['Variation'])
#     for spot in range(end):
# #         if ' ' in df['Variation']:
# #             df['Variation'] = df['Variation'].split(' ')[0]
# #             print(df['Variation'])
            
# #         if df['Variation'] in df['text_list'][spot]:
#         if df['Variation'].split(' ')[0] in df['text_list'][spot]:
#             if spot - n > 0:
#                 start = spot - n
#             else:
#                 start = 0
#             if spot + n < end:
#                 stop = spot + n
#             else:
#                 stop = end
#             text_list_parsed.append([start, stop, spot])

#     for i in range(len(text_list_parsed)):
#         if (i < len(text_list_parsed) - 1) and (text_list_parsed[i][1] > text_list_parsed[i+1][0]):
#             text_list_parsed[i+1][0] = text_list_parsed[i][0]
#         else:
#             final_text_list_parsed.append(df['text_list'][text_list_parsed[i][0] : text_list_parsed[i][1]])
#             print(text_list_parsed[i][0], text_list_parsed[i][1])

#     final_text_list_parsed = [val for sublist in final_text_list_parsed for val in sublist]
#     df['text_list_parsed'] = final_text_list_parsed
#     df['text_parsed'] = ' '.join(df['text_list_parsed'])
    
#     print(df['text_parsed'][:50])
#     print()
    
#     return df

# train = train.apply(textParser, axis=1)
# test = test.apply(textParser, axis=1)

# saved = pd.concat([train, test], axis=0)
# saved = saved[['text_list_parsed', 'text_parsed']]
# saved.to_csv('parsed_data.csv', index=False)


# load parsed text data from saved file
parsed = pd.read_csv('parsed_data.csv')
train_parsed = parsed[:len(train)].reset_index(drop=True)
test_parsed = parsed[len(train):].reset_index(drop=True)

train = pd.concat([train, train_parsed], axis=1)
test = pd.concat([test, test_parsed], axis=1)

train['text_parsed'] = train['text_parsed'].fillna('')
test['text_parsed'] = test['text_parsed'].fillna('')


# combine like-words in Variation
def remove_Var_Redundancy(df):
    for i in ['deletions', 'deletion']:
        df['Variation'] = df['Variation'].replace(to_replace=i, value='del')

    for i in ['delins', 'intsertions/deletions', 'insertions/deletions']:
        df['Variation'] = df['Variation'].replace(to_replace=i, value='deletion/insertion')

    for i in ['insertions', 'insertion']:
        df['Variation'] = df['Variation'].replace(to_replace=i, value='ins')

    for i in ['duplications', 'duplication']:
        df['Variation'] = df['Variation'].replace(to_replace=i, value='dup')
            
    return df

train = remove_Var_Redundancy(train)
test = remove_Var_Redundancy(test)

print(train.shape)
print(test.shape)


combined = pd.concat([train, test], axis=0)

# find most common genes
gene_pop = combined.groupby(['Gene'])['clean_text'].count().sort_values(ascending=False)
gene_popularity = dict(zip(np.array(gene_pop.index), gene_pop.values))


# create feature that indicates gene 'popularity'
train['gene_pop'] = [gene_popularity.get(train['Gene'].loc[i]) for i in range(len(train))]
test['gene_pop'] = [gene_popularity.get(test['Gene'].loc[i]) for i in range(len(test))]


combined = pd.concat([train, test], axis=0)

# find most common variations
var_pop = combined.groupby(['Variation'])['clean_text'].count().sort_values(ascending=False)
var_popularity = dict(zip(np.array(var_pop.index), var_pop.values))


# create feature that indicates variation 'popularity'
train['var_pop'] = [var_popularity.get(train['Variation'].loc[i]) for i in range(len(train))]
test['var_pop'] = [var_popularity.get(test['Variation'].loc[i]) for i in range(len(test))]

# create features from text meta data
def wordFeatures(df):
    # count number of characters in text
    df['char_count_total'] = df.clean_text.str.len()
    df['char_count_parsed'] = df.text_parsed.str.len()

    # count number of characters removed
    df['char_count_dirty'] = df.text.str.len() - df.clean_text.str.len()
    df['char_count_unparsed'] = df.clean_text.str.len() - df.text_parsed.str.len()

    # count number of words in text
    df['word_count_total'] = df.text_list.str.len()
    df['word_count_parsed'] = df.text_list_parsed.str.len()

    # find average word length in text
    df['avg_word_len_total'] = df.char_count_dirty / df.word_count_total
    df['avg_word_len_parsed'] = df.char_count_unparsed / df.word_count_parsed

    df['avg_word_len_total'] = df['avg_word_len_total'].replace(np.inf, 100)
    df['avg_word_len_parsed'] = df['avg_word_len_parsed'].replace(np.inf, 100)
    
    # count number of words in Variation
    df['var_word_count'] = df.Variation.str.split().str.len()

    # find Gene word length
    df['Gene_length'] = df.Gene.str.len()

    # find Variation word length
    df['Variation_length'] = df.Variation.str.len()

    return df

train = wordFeatures(train)
test = wordFeatures(test)

# print sorted (ABC) list of all Genes in train set
print(train.Gene.sort_values().unique())

# many Genes are related or similar, and follow a similar naming convention:
# letters, number
# letters, number, letters
# letters, number, -, letters
# letters, number, -, numbers


# Segment Gene data into its rightful components. For example:
# RAD51B --> RAD, 51, B
# YAP1 --> YAP, 1
# WHSC1L1 --> WHSC, 1, L, 1


# determine if string character is an integer
def isInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

# segment gene into four parts (no genes have more than that)
def gene_Segment(df):
    first_letters = []
    first_nums = []
    second_letters = []
    second_nums = []
    
    for char in df.Gene:
        if isInt(char) == False and len(first_nums) == 0:
            first_letters.append(char)
        if isInt(char) == True and len(second_letters) == 0:
            first_nums.append(char)
        if isInt(char) == False and len(first_nums) != 0:
            second_letters.append(char)
        if isInt(char) == True and len(second_letters) != 0:
            second_nums.append(char)
        continue

    df['gene_1st_letters'] = ''.join(first_letters)
    df['gene_1st_nums'] = ''.join(first_nums)
    df['gene_2nd_letters'] = ''.join(second_letters)
    df['gene_2nd_nums'] = ''.join(second_nums)
    
    return df

train = train.apply(gene_Segment, axis=1)
test = test.apply(gene_Segment, axis=1)



# Segment Variation data into its rightful components.
def var_Segment(df):
    first_letters = []
    first_nums = []
    second_letters = []
    second_nums = []
    third_letters = []
    third_nums = []
# using six parts because some Variations have that many
    
    for char in df.Variation:
        if isInt(char) == False and len(first_nums) == 0:
            first_letters.append(char)
        if isInt(char) == True and len(second_letters) == 0:
            first_nums.append(char)
        if isInt(char) == False and len(first_nums) != 0:
            second_letters.append(char)
        if isInt(char) == True and len(second_letters) != 0:
            second_nums.append(char)
        if isInt(char) == False and len(second_nums) != 0:
            third_letters.append(char)
        if isInt(char) == True and len(third_letters) != 0:
            third_nums.append(char)
        continue

    df['var_1st_letters'] = ''.join(first_letters)
    df['var_1st_nums'] = ''.join(first_nums)
    df['var_2nd_letters'] = ''.join(second_letters)
    df['var_2nd_nums'] = ''.join(second_nums)
    df['var_3rd_letters'] = ''.join(third_letters)
    df['var_3rd_nums'] = ''.join(third_nums)
    
    return df

train = train.apply(var_Segment, axis=1)
test = test.apply(var_Segment, axis=1)


print('Number of Unique Train Gene 1st Letters: ', train.gene_1st_letters.nunique())
print('Number of Unique Train Genes: ', train.Gene.nunique())
print()
print('Number of Unique Test Gene 1st Letters: ', test.gene_1st_letters.nunique())
print('Number of Unique Test Genes: ', test.Gene.nunique())


print('Number of Unique Train Variation 1st Letters: ', train.var_1st_letters.nunique())
print('Number of Unique Train Variation: ', train.Variation.nunique())
print()
print('Number of Unique Test Variation 1st Letters: ', test.var_1st_letters.nunique())
print('Number of Unique Test Variation: ', test.Variation.nunique())


combined = pd.concat([train, test], axis=0)

print('max gene length: ')
print(combined.Gene.str.len().sort_values(ascending=False)[:5])
print()
print('max variation length: ')
print(combined.Variation.str.len().sort_values(ascending=False)[:20])


print('train: ', train.shape)
print('test: ', test.shape)


# Ravels Genes and Variations by their segments
gene_columns = ['gene_1st_letters', 'gene_1st_nums', 'gene_2nd_letters', 'gene_2nd_nums']
var_columns = ['var_1st_letters', 'var_1st_nums', 'var_2nd_letters', 'var_2nd_nums',
              'var_3rd_letters', 'var_3rd_nums']

def segment_Breaker(df, columns):
    for column in columns:
#         for char in range(min(train[column].str.len().max(), 12)):
        for char in range(min(train[column].str.len().max(), 6)):
            column_value = []
            for element in df[column]:
                if len(element) > char:
                    column_value.append(element[char])
                else:
                    column_value.append('')
            df[column + '_' + str(char)] = column_value
    return df

combined = pd.concat([train, test], axis=0)

train = segment_Breaker(train, gene_columns)
train = segment_Breaker(train, var_columns)
test = segment_Breaker(test, gene_columns)
test = segment_Breaker(test, var_columns)


train.columns.values

# Might want to remove all columns with more than .95 NaNs
# -- they probably won't be useful

combined = pd.concat([train, test], axis=0)

print('max gene length: ')
print(combined.Gene.str.len().sort_values(ascending=False)[:5])
print()
print('max variation length: ')
print(combined.Variation.str.len().sort_values(ascending=False)[:10])


different_variations = ['truncating mutations', 'deletion', 'promoter mutations',
                        'amplification', 'promoter hypermethylation', 'overexpression',
                       'copy number loss']

# # *** might want to use this mixed_variations instead ***
# mixed_variations = ['Del', 'Deletion/Insertion', 'Ins', 'Dup', 'Fusion',
#                     'missense', 'splice', 'domain', 'binding', 'Polymorphism']

mixed_variations = ['del', 'deletion/Insertion', 'ins', 'dup', 'fusion',
                    'splice', 'polymorphism']


# split gene and variation into individual 1-sized components (to understand what the
# words and letters mean). Do not do this for variations with 'special' designations.
def gene_breaker(df):
    for i in range(6):
        df['Gene_' + str(i)] = df['Gene'].map(lambda x: str(x[i]) if len(x) > i else '')
    return df

def variation_breaker(df):
    for i in different_variations:
        if i in df['Variation']:
            df[str(i) + '_var_breaker'] = 1
        else:
            df[str(i) + '_var_breaker'] = 0
            for j in range(10):
                df['Variation_' + str(j)] = df['Variation'].map(lambda x: str(x[j]) if len(x) > j else '')

    for i in mixed_variations:
        if i in df['Variation']:
            df[str(i) + '_var_breaker'] = 1
        else:
            df[str(i) + '_var_breaker'] = 0

    return df
    
train = gene_breaker(train)
test = gene_breaker(test)
train = variation_breaker(train)
test = variation_breaker(test)

print('train: ', train.shape)
print('test: ', test.shape)


# # A different version of gene/var breaker that does not split the gene and variation into individual components
######

# different_variations = ['Truncating Mutations', 'Deletion', 'Promoter Mutations',
#                         'Amplification', 'Promoter Hypermethylation', 'Overexpression',
#                        'Copy Number Loss']

# # # *** might want to use this mixed_variations instead ***
# # mixed_variations = ['Del', 'Deletion/Insertion', 'Ins', 'Dup', 'Fusion',
# #                     'missense', 'splice', 'domain', 'binding', 'Polymorphism']

# mixed_variations = ['del', 'deletion/Insertion', 'ins', 'dup', 'fusion',
#                     'splice', 'polymorphism']

# def variation_breaker(df):
#     for i in mixed_variations:
#         if i in df['Variation']:
#             df[str(i)] = 1
#         else:
#             df[str(i)] = 0

#     return df

# train = variation_breaker(train)
# test = variation_breaker(test)

# print('train: ', train.shape)
# print('test: ', test.shape)


# count how often the Gene is mentioned in text
def find_Gene(df):
    df['Gene_Share'] = df.apply(lambda x: x['clean_text'].count(x['Gene']), axis=1)
    return df

# count how often the Variation is mentioned in text
def find_Variation(df):
    df['Variation_Share'] = df.apply(lambda x: x['clean_text'].count(x['Variation']), axis=1)
    return df

train = find_Gene(train)
test = find_Gene(test)
train = find_Variation(train)
test = find_Variation(test)

print('train: ', train.shape)
print('test: ', test.shape)


####################################
# # #   Run TF-IDF for 50 features   # # # 

print('Starting TF-IDF...')

combined = pd.concat([train, test], axis=0)
full_text = combined['clean_text']

num_features = 50
tfidf = TfidfVectorizer(max_features = num_features, strip_accents='unicode',
                        lowercase=True, stop_words='english')

tfidf.fit(full_text)

print()
print('Starting Transform...')

train_text_tfidf = tfidf.transform(train['clean_text'])
test_text_tfidf = tfidf.transform(test['clean_text'])

print()
print('Label and Incorporate TF-IDF')

train_array = pd.DataFrame(train_text_tfidf.toarray())
test_array = pd.DataFrame(test_text_tfidf.toarray())

feature_names = tfidf.get_feature_names()

for i in range(num_features):
    feature_names[i] = 'tf-idf_' + feature_names[i] + '_ct'

train_array.columns = feature_names
test_array.columns = feature_names

train = pd.concat([train, train_array], axis=1)
test = pd.concat([test, test_array], axis=1)

train.head()

TF_IDF = pd.concat([train_array, test_array], axis=0)
TF_IDF.to_csv('tf-idf.csv', index=False)


####################################
# # #   Read from top 50 TF-IDF .csv file   # # # 

# tf_idf = pd.read_csv('tf-idf.csv')

# tf_idf_train = pd.DataFrame(tf_idf[:train.shape[0]]).reset_index(drop=True)
# tf_idf_test = pd.DataFrame(tf_idf[train.shape[0]:]).reset_index(drop=True)

# train = pd.concat([train, tf_idf_train], axis=1)
# test = pd.concat([test, tf_idf_test], axis=1)

# print('train: ', train.shape)
# print('test: ', test.shape)



####################################
# # #   Run TF-IDF for 250 features, pass through truncated SVD to get 50   # # # 

print('Starting TF-IDF...')

combined = pd.concat([train, test], axis=0).reset_index(drop=True)
full_text = combined['text_parsed']

num_features = 500
tfidf = TfidfVectorizer(max_features = num_features, strip_accents='unicode',
                        lowercase=True, stop_words='english')

tfidf.fit(full_text)

print()
print('Starting Transform...')

train_text_tfidf = tfidf.transform(train['text_parsed'])
test_text_tfidf = tfidf.transform(test['text_parsed'])

print()
print('Label and Incorporate TF-IDF')

train_array = pd.DataFrame(train_text_tfidf.toarray())
test_array = pd.DataFrame(test_text_tfidf.toarray())

feature_names = tfidf.get_feature_names()

for i in range(num_features):
    feature_names[i] = 'tf-idf_' + feature_names[i] + '_svd'

train_array.columns = feature_names
test_array.columns = feature_names

from sklearn.decomposition import TruncatedSVD

tf_idf_full = pd.concat([train_array, test_array], axis=0)

print()
print('Starting Truncated SVD...')
svd = TruncatedSVD(100)
svd_full = svd.fit_transform(tf_idf_full)

svd_train = pd.DataFrame(svd_full[:train.shape[0]]).reset_index(drop=True)
svd_test = pd.DataFrame(svd_full[train.shape[0]:]).reset_index(drop=True)

svd_columns = []
for i in range(len(svd_train.columns)):
    svd_columns.append('svd_' + str(i))

svd_train.columns = svd_columns
svd_test.columns = svd_columns

train = pd.concat([train, svd_train], axis=1)
test = pd.concat([test, svd_test], axis=1)

print('train: ', train.shape)
print('test: ', test.shape)



####################################
# # #   Read saved file with 500 features count_vect features   # # # 
# count_vect = pd.read_csv('count_vec.csv')
# train_vect = count_vect[:train.shape[0]].reset_index(drop=True)
# test_vect = count_vect[train.shape[0]:].reset_index(drop=True)

# train = pd.concat([train, train_vect], axis=1)
# test = pd.concat([test, test_vect], axis=1)

# print('train: ', train.shape)
# print('test: ', test.shape)

# train.head()


####################################
# # #   Run count_vect for 500 features   # # # 

# create list of all words in train and test
print('Combining Text Lists...')
print()

combined = pd.concat([train, test], axis=0)
full_text = combined['text_list']

train_text_list = [item for sublist in train['text_list'].ravel() for item in sublist]
test_text_list = [item for sublist in test['text_list'].ravel() for item in sublist]

print('train list: ', len(train_text_list))
print('test list: ', len(test_text_list))

text_list = train_text_list + test_text_list

print('combined list: ', len(text_list))

print()
print('Starting Count Vectorizer...')

# find the 500 most common words
feature_transform = CountVectorizer(stop_words='english', max_features=500)
feature_transform.fit(text_list)

print()
print('Starting Data Transform...')

#  create new occurence features from most common words
def transform_data(X):
    feat_sparse = feature_transform.transform(X['clean_text'])
    vocabulary = feature_transform.vocabulary_
    
    X1 = pd.DataFrame([pd.Series(feat_sparse[i].toarray().ravel()) for i in np.arange(feat_sparse.shape[0])])
    X1.columns = list(sorted(vocabulary.keys()))
    X1 = X1.reset_index(drop=True)
    return X1

train_count_vect = transform_data(train)
test_count_vect = transform_data(test)

count_vect = pd.concat([train_count_vect, test_count_vect], axis=0)
count_vect.to_csv('count_vec.csv', index=False)

train = pd.concat([train, train_count_vect], axis=1)
test = pd.concat([test, test_count_vect], axis=1)


print('train: ', train.shape)
print('test: ', test.shape)


# check for repeat columns
import collections
a = train.columns.tolist()
print('list of repeated columns: ', [item for item, count in collections.Counter(a).items() if count > 1])


# drop non-numerical data
combined = pd.concat([train, test], axis=0)
remove = ['Variation', 'text', 'clean_text', 'text_list', 'text_list_parsed', 'text_parsed']

train = train.drop(remove, axis=1)
test = test.drop(remove, axis=1)

print(train.shape)
print(test.shape)

# save transformed dataframe into .csv file
combined = pd.concat([train, test], axis=0)
combined.to_csv('V21_combined_unencoded.csv')


# encode object columns into numericals
def label_encoder(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            le = preprocessing.LabelEncoder().fit(combined[column].values)
            df[column] = le.transform(df[column].values)
    return df

train = label_encoder(train)
test = label_encoder(test)


# double-check for object columns (remove if found)
for i in train.columns:
    if train[i].dtype == object:
        print(i)
# no objects columns found


print('train: ', train.shape)
print('train: ', test.shape)

print('nulls: ', train[train['Class'].isnull()].shape)
print(train.Class.unique())

# necessary because XGB expects classes to be labeled from 0, not 1
train.Class = train.Class - 1

# separate train data into a CV train and test
x_train = train.drop(['Class'], axis=1)
x_test = train['Class']

print('Start Training')

xgb_params = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class' : 9,

    'eta': 0.05,
    'max_depth': 4,
    'min_child_weight': 3,
    'n_folds': 5,
    'silent': 1,    
}

dtrain = xgb.DMatrix(x_train, x_test)

early = 20
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=2000, early_stopping_rounds = early, verbose_eval=5)

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round = int(num_boost_rounds / 0.80))

fig, ax = plt.subplots(1, 1, figsize=(6, 10))
xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

# *** V18 ***
# [225]	train-mlogloss:0.315438+0.00413038	test-mlogloss:0.905464+0.0078782
# 0.922039206

# make predictions on test data
test_data = xgb.DMatrix(test)

y_predict = pd.concat([ID, pd.DataFrame(model.predict(test_data))], axis=1)
output = pd.DataFrame(y_predict)

output.columns = ['ID', 'class1', 'class2', 'class3', 'class4',
                  'class5', 'class6', 'class7', 'class8', 'class9']

output.to_csv('submission.csv', index=False)
print(output.head())