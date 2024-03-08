import konlpy
import re

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
def downWord(fileName,stopword,imgFileName):
    moon = open("../Data/"+fileName,encoding="UTF-8").read()
    moon = re.sub("[^가-힣]",' ',moon)

    for word in stopword:
        moon=re.sub(word,' ',moon)

    hannanum = konlpy.tag.Hannanum()
    nouns = hannanum.nouns(moon)
    df_word=pd.DataFrame({
        'word':nouns
    })

    df_word['count'] = df_word['word'].str.len()
    df_word = df_word.query('count>=2')
    df_word=df_word.groupby('word',as_index=False).agg(n=('word','count'))
    dic_word = df_word.set_index('word').to_dict()['n']
    font = "../Data/DoHyeon-Regular.ttf"
    wc =WordCloud(
        font_path=font,
        width=400,
        height=400,
        background_color='white'
    )

    img_wordcloud = wc.generate_from_frequencies(dic_word)
    plt.figure(figsize=(10,10))
    plt.axis('off')
    plt.imshow(img_wordcloud)
    plt.savefig("c:/result/"+imgFileName)
    print('wordCloud를 만들었습니다')


# moon = open("../Data/speech_moon.txt", encoding="UTF-8").read()
# moon = re.sub("[^가-힣]", ' ', moon)
# stop_word = ['국민', '우리', '나라', '우리나라', '대통령', '대한민국']
# for word in stop_word:
#     moon = re.sub(word, ' ', moon)
#
# hannanum = konlpy.tag.Hannanum()
# nouns = hannanum.nouns(moon)
# df_word = pd.DataFrame({
#     'word': nouns
# })
#
# df_word['count'] = df_word['word'].str.len()
# df_word = df_word.query('count>=2')
# df_word = df_word.groupby('word', as_index=False).agg(n=('word', 'count'))
# dic_word = df_word.set_index('word').to_dict()['n']
# font = "../Data/DoHyeon-Regular.ttf"
# wc = WordCloud(
#     font_path=font,
#     width=400,
#     height=400,
#     background_color='white'
# )
#
# img_wordcloud = wc.generate_from_frequencies(dic_word)
# plt.figure(figsize=(10, 10))
# plt.axis('off')
# plt.imshow(img_wordcloud)
# plt.savefig("c:/result/out.png")
# print('wordCloud를 만들었습니다')

fileName="speech_moon.txt"
stopword=['국민', '우리', '나라', '우리나라', '대통령', '대한민국']
imgFileName="byMethod.png"
downWord(fileName,stopword,imgFileName)