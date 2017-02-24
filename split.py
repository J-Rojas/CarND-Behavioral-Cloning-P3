import os
import shutil
import sys
import model
import csv
import numpy as np
import sklearn.model_selection

if len(sys.argv) < 6:
    sys.exit("Error in arguments: csv_file input_dir output_dir [sample_ratio|sample_count] [validation_ratio|validation_count]")

csv_file = sys.argv[1].strip()
input_dir = sys.argv[2].strip()
output_dir = sys.argv[3].strip()
sample_count = float(sys.argv[4])
validation_count = float(sys.argv[5])

csv_reader = csv.DictReader(open(csv_file))
input_data = []
for row in csv_reader:
    input_data.append(row)

if sample_count < 1:
    sample_count = len(input_data) * sample_count

np.random.shuffle(input_data)
buckets = [sample_count * 0.4, sample_count * 0.2, sample_count * 0.4]
labels = []

total = 0
idx = 0
data = []

while idx < len(input_data) and len(data) < sample_count:

    row = input_data[idx]

    if float(row['steering']) > 0.01 and buckets[0] > 0:
        data.append(row)
        labels.append(0)
        buckets[0] -= 1
    elif float(row['steering']) <= 0.01 and float(row['steering']) >= -0.01 and buckets[1] > 0:
        data.append(row)
        labels.append(1)
        buckets[1] -= 1
    elif float(row['steering']) < -0.01 and buckets[2] > 0:
        data.append(row)
        labels.append(2)
        buckets[2] -= 1

    idx+=1

var = 'y'
if os.path.isdir(output_dir):
    var = input("Are you sure you would like to delete {}? [y/n]".format(output_dir))

if var != 'y':
    exit()

if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
os.makedirs(output_dir + '/test')
os.makedirs(output_dir + '/train')

print("Total number of samples: {}".format(len(labels)))
print("Number of samples in each class: {}".format(np.histogram(labels, bins=[0, 1, 2, 3])))

if validation_count < 1:
    validation_count = len(data) * validation_count

if validation_count > len(data):
    exit('Error: validation_count > samples')


if validation_count > 0:
    model_selection = sklearn.model_selection.StratifiedShuffleSplit(1, int(validation_count))
    fold = model_selection.split(data, labels)
else:
    fold = (range(len(data)), None)

for train, test in fold:
    with open(output_dir + '/train.csv', mode='w') as csvfile:
        writer = csv.DictWriter(csvfile, csv_reader.fieldnames)
        writer.writeheader()

        for idx in train:

            d = data[idx]

            files = shutil.copyfile(input_dir + d['left'].strip(), output_dir + '/train/' + os.path.basename(d['left'].strip()))
            files = shutil.copyfile(input_dir + d['center'].strip(), output_dir + '/train/' + os.path.basename(d['center'].strip()))
            files = shutil.copyfile(input_dir + d['right'].strip(), output_dir + '/train/' + os.path.basename(d['right'].strip()))

            d['left'] = 'train/' + os.path.basename(d['left'].strip())
            d['center'] = 'train/' + os.path.basename(d['center'].strip())
            d['right'] = 'train/' + os.path.basename(d['right'].strip())

            writer.writerow(d)

        csvfile.close()

    with open(output_dir + '/test.csv', mode='w') as csvfile:
        writer = csv.DictWriter(csvfile, csv_reader.fieldnames)
        writer.writeheader()

        for idx in test:

            d = data[idx]

            files = shutil.copyfile(input_dir + d['left'].strip(), output_dir + '/test/' + os.path.basename(d['left'].strip()))
            files = shutil.copyfile(input_dir + d['center'].strip(), output_dir + '/test/' + os.path.basename(d['center'].strip()))
            files = shutil.copyfile(input_dir + d['right'].strip(), output_dir + '/test/' + os.path.basename(d['right'].strip()))

            d['left'] = 'test/' + os.path.basename(d['left'].strip())
            d['center'] = 'test/' + os.path.basename(d['center'].strip())
            d['right'] = 'test/' + os.path.basename(d['right'].strip())

            writer.writerow(d)

        csvfile.close()

exit('Complete!')
