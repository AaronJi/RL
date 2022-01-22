#encoding=utf8
import sys


from gen_emb import EmbeddingGeneration

root = "/home/hengyang.ymy"

gen_emb = EmbeddingGeneration(root+'/RL_rec/python/LIRD/feature_conf.json')
path = root+"/RL_rec/data/MovieLens/emb_data"
gen_emb.build_model([
    path,
    "u.user",
    "u.item",
    "u.data",
    "u1.test"
])

gen_emb.train(path)
gen_emb.predict(path)





