import requests
import dotenv
import os
import json
from rule34Py import rule34Py
import time
client = rule34Py()


def get_file_extension_os(url):
    _,ext = os.path.splitext(url)
    return ext


dotenv.load_dotenv()

user_id = os.getenv('USER_ID')
api_key = os.getenv('API_KEY')

client.user_id = user_id
client.api_key = api_key


ai_scored = ['sort:score', '-video', '-gif', '-comic', 'ai_generated']
ai_recent = ['-video', '-gif', '-comic', 'ai_generated']
hu_scored = ['sort:score', '-video', '-gif', '-comic', '-ai_generated', '-ai_assisted', 'self_upload']
hu_recent = ['-video', '-gif', '-comic', '-ai_generated', '-ai_assisted', 'self_upload']


search = {ai_scored:[30000,'ai_scored'],ai_recent:[10000,'ai_recent'],hu_scored:[30000,'hu_scored'],hu_recent:[10000,'hu_recent']}

for tags,details in search:
    count = details[0]
    name = details[1]


    i = 0
    posts_scaned = 0


    while True:
        
        posts = client.search(tags, page_id=int(posts_scaned/100), limit=100)
        
        for p in posts[(posts_scaned%100):]:
            posts_scaned += 1
            print(f'downloading: {p.image}')
            img_data = requests.get(p.image).content
            ext = get_file_extension_os(p.image)
            f = open(f'{name}/img{i}{ext}','wb')
            f.write(img_data)
            f.close()
            i += 1
            if i >= count:
                break
            time.sleep(1)
    print(f'added {count} imgs with tags {tags} ')


#img_data = requests.get(img_url).content


#f = open('img.png','wb')

#f.write(img_data)
#f.close()



