import csv

# with open("model_train_log", 'a') as file:
#   writer = csv.writer(file)

#   writer.writerow(['epoch_no', 'loss', 'mIou'])

for idx in range(10):
  with open("model_train_log", 'a') as file:
    writer = csv.writer(file)

    writer.writerow([idx, "hello", "123"])