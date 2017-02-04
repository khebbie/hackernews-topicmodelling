from apiclient import APIClient
import json
from goose import Goose
import codecs


class AcmePublicAPI(APIClient):
    BASE_URL = 'https://hacker-news.firebaseio.com/v0/'

acme_api = AcmePublicAPI()
res = acme_api.call('topstories.json')
jason = json.loads(str(res))

print(jason)
for id in jason:
    print('.')
    story = acme_api.call('item/' + str(id) + '.json')
    url = str(story['url'])
    g = Goose()
    article = g.extract(url=url)

    text_file = codecs.open('content/' +str(id), "w", "utf-8")
    text_file.write(article.cleaned_text)
    text_file.close()
