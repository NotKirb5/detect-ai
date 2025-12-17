import requests
import dotenv
import os
import json
from rule34Py import rule34Py
import time
client = rule34Py()


dotenv.load_dotenv()

user_id = os.getenv('USER_ID')
api_key = os.getenv('API_KEY')

client.user_id = user_id
client.api_key = api_key





i = 0
posts_scaned = 0


while i < 150:
    
    posts = client.search(["-ai_generated"], page_id=int(posts_scaned/100), limit=100)
    
    for p in posts[:99]:
        posts_scaned += 1
        if p.content_type != 'image':
            print(f'post is a {p.content_type}')
            continue
        print(f'downloading: {p.image}')
        img_data = requests.get(p.image).content
        f = open(f'huimgs/img{i}.png','wb')
        f.write(img_data)
        f.close()
        i += 1
        time.sleep(1)


#img_data = requests.get(img_url).content


#f = open('img.png','wb')

#f.write(img_data)
#f.close()



