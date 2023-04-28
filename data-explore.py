##VQA-RAD train-val loading 
import json 
import pickle

with open('data/cache/train_target.pkl', 'rb') as f:
    data_train = pickle.load(f)
print('Number of instances of train (data_train)', len(data_train))
## Output form of data_train_qs
# {'qid': 1, 'image_nname': 'synpic54610.jpg', 'labels': [0], 'scores': [1.0]}

with open('data/cache/test_target.pkl', 'rb') as f:
    data_test = pickle.load(f)
print('Number of instances of test (data_test)', len(data_test))
## Output form of data_test_qs
# {'qid': 11, 'image_name': 'synpic42202.jpg', 'labels': [0], 'scores': [1.0]}

f_train = open('data/trainset.json')
data_train_qs = json.load(f_train)
print('Number of instances of train (data_train_qs)', len(data_train_qs))
## Output form of data_train_qs
# {'qid': 1, 'image_name': 'synpic54610.jpg', 'image_organ': 'HEAD', 'answer': 'Yes', 'answer_type': 'CLOSED', 'question_type': 'PRES', 'question': 'Are regions of the brain infarcted?', 'phrase_type': 'freeform'}

f_test = open('data/testset.json')
data_test_qs = json.load(f_test)
print('Number of instances of test (data_test_qs)', len(data_test_qs))
## Output form of data_test_qs
# {'qid': 11, 'image_name': 'synpic42202.jpg', 'image_organ': 'CHEST', 'answer': 'yes', 'answer_type': 'CLOSED', 'question_type': 'PRES', 'question': 'Is there evidence of an aortic aneurysm?', 'phrase_type': 'freeform'}


# Dropping Test data points with no classes
data_test_with_dropped_datapoints = []
count = 0
for i in range(451):
  if i%20==0:
      print(i)
      print(data_test[i])
      print(data_test_qs[i])
  if(len(data_test[i]['labels'])!=1):
    count += 1
    print("NO LABEL", i)
    print(data_test[i])
    print(data_test_qs[i])

  else:
    data_test_with_dropped_datapoints.append(data_test[i])

print(f"Number of test datapoints with no class label:{count}")
data_test = data_test_with_dropped_datapoints # Updating the testing dataset
print(len(data_test))

## Understanding the dataset!!! Important

# We have the dataset in two forms - (data_train & data_test) and (data_train_qs & data_test_qs)
# I verified that both the forms have the same questions and imgaes
# However the (data_train_qs & data_test_qs) has classes in text form (for e.g - "Yes" and not 0). And it has 514 classes, not 
# 458 because there are classes like "Yes" and "yes". It is becoming cumbersome to cleam them.
# So, we go ahead with (data_train & data_test). However, in data_test, 43 datapoints do not have classes, so I drop these.

with open('data/cache/trainval_ans2label.pkl', 'rb') as f:
    label_train = pickle.load(f)

# This basically maps the answer type to an index for e.g 'yes':0
print('Total classes', len(label_train.keys()))