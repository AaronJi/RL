from python.DIN.model import DIN
import os

main_path = os.path.abspath(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(main_path)))

data_dir = os.path.join(project_dir, 'data')
data_dir = os.path.join(data_dir, 'MovieLens')

model = DIN(num_epochs=1, batch_size=128, use_din=True)

model.gen_data(data_dir)

model.build_model()

model.train(data_dir)

model.eval(data_dir)
# with open(os.path.join(path,"train.txt")) as f:
#     for line in f.readlines():
#         if not len(line.split(",")) == 902:
#             print(line)
#             break

predict_result = model.predict(data_dir)

print(predict_result)

for predict in predict_result:
    print(type(predict))
    print(predict)
    score = predict['probabilities']
    print(score)
    print(type(score))
    exit(1)

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
