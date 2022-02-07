from python.DIN.model import DIN
import os
import numpy  as  np

main_path = os.path.abspath(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(main_path)))

data_dir = os.path.join(project_dir, 'data')
data_dir = os.path.join(data_dir, 'MovieLens')

model = DIN(num_epochs=1, batch_size=128, use_din=True)

model.gen_data(data_dir)

model.build_model()

model.train(os.path.join(data_dir, 'train.txt'))

model.eval(os.path.join(data_dir, 'test.txt'))
# with open(os.path.join(path,"train.txt")) as f:
#     for line in f.readlines():
#         if not len(line.split(",")) == 902:
#             print(line)
#             break

predict_result_test = model.predict(os.path.join(data_dir, 'train.txt'))

result_dict = {}

results = []
for predict in predict_result_test:
    userid = predict['userid']
    itemid = predict['itemid']
    score = predict['probabilities'][0]
    rating = predict['rating']
    results.append([userid, itemid, score, rating])
    key = str(userid) + '_' + str(itemid)
    result_dict[key] = {'userid': str(userid), 'itemid': str(itemid), 'score': score, 'rating': int(rating)}
results = np.array(results)
#print(results)
print(results.shape)

predict_result_test = model.predict(os.path.join(data_dir, 'test.txt'))

results = []
for predict in predict_result_test:
    userid = predict['userid']
    itemid = predict['itemid']
    score = predict['probabilities'][0]
    rating = predict['rating']
    results.append([userid, itemid, score, rating])
    key = str(userid) + '_' + str(itemid)
    result_dict[key] = {'userid': str(userid), 'itemid': str(itemid), 'score': score, 'rating': int(rating)}
results = np.array(results)
#print(results)
print(results.shape)

print(len(result_dict))

'''
n = 1
while True:
    if next(predict_result) is None:
        break
    else:
        n += 1
        print(next(predict_result))
print(n)
'''
