import psycopg2
import pandas as pd
import pymorphy2
import re

db = {
    'host': 'localhost',
    'username': 'postgres',
    'password': 'postgres',
    'database': 'monitor',
    'table': 'public.appeal'
}
# выгрузка данных в Pandas DataFrame
conn = psycopg2.connect(host=db['host'], dbname=db['database'], user=db['username'], password=db['password'])
# null роняет скрипт
stmt = "SELECT id,text,categoryid FROM {database}.{table} WHERE text is not null and categoryid is not null" .format(
    database=db['database'], table=db['table'])
df = pd.read_sql(stmt, conn)

ma = pymorphy2.MorphAnalyzer()

emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)

def clean_text(text):
    text = text.lower()
    text = text.replace("\\", " ").replace('здравствуйте', ' ').replace('пожалуйста', ' ')
    # стоит ли убирать @link или они могут указывать на категорию?
    text = re.sub('<[^>]*>', ' ', text)  # tags
    text = emoji_pattern.sub(' ', text)  # emoji etc
    text = re.sub(r'http\S+', ' ', text)  # urls

    text = re.sub(r'\-\s\r\n\s{1,}|\-\s\r\n|\r\n', '', text)  # deleting newlines and line-breaks
    text = re.sub(r'[\'.,…:«»;_%©?*,|!@#$%^&()\d]|[+=]|[[]|[]]|[/]|"|\s{2,}|-', ' ', text)  # deleting symbols
    # в python3 нет unicode https://habr.com/ru/post/208192/
    # text = " ".join(ma.parse(unicode(word))[0].normal_form for word in text.split())
    text = " ".join(ma.parse(word)[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word) > 2)
    # text = text.encode("utf-8")
    return text


df['Description'] = df.apply(lambda x: clean_text(x[u'text']), axis=1)
# мешаем записи
for line in df['Description']:
    print(line)
df.to_pickle('dataframe_ver_1.pkl')
