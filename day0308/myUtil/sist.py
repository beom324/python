import konlpy
import re

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

def makeWordCloud(fileName,stopWord,imgFileName,font):
    moon = open("../Data/" + fileName, encoding="UTF-8").read()
    moon = re.sub("[^가-힣]", ' ', moon)

    for word in stopWord:
        moon = re.sub(word, ' ', moon)

    hannanum = konlpy.tag.Hannanum()
    nouns = hannanum.nouns(moon)
    df_word = pd.DataFrame({
        'word': nouns
    })

    df_word['count'] = df_word['word'].str.len()
    df_word = df_word.query('count>=2')
    df_word = df_word.groupby('word', as_index=False).agg(n=('word', 'count'))
    dic_word = df_word.set_index('word').to_dict()['n']
    wc = WordCloud(
        font_path=font,
        width=400,
        height=400,
        background_color='white'
    )

    img_wordcloud = wc.generate_from_frequencies(dic_word)
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(img_wordcloud)
    plt.savefig("./static/" + imgFileName)
    print('wordCloud를 만들었습니다')