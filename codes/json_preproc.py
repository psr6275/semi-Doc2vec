import json
import os, sys

cwd = os.getcwd()
data_cwd = cwd + '/data'
f = open(data_cwd+'yelp_academic_dataset_review.json','rb')
js = json.loads(f.read())
f.close()
