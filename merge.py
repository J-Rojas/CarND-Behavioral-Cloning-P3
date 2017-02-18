import os
import shutil
import sys
import csv
import sklearn.model_selection

def writeRecords(csv_reader, csv_writer, input_dir, output_dir, prefix_path):
    for row in csv_reader:
        d = row

        d['left'] = os.path.basename(d['left'].strip())
        d['center'] = os.path.basename(d['center'].strip())
        d['right'] = os.path.basename(d['right'].strip())

        files = shutil.copyfile(input_dir + d['left'], output_dir + d['left'])
        files = shutil.copyfile(input_dir + d['center'], output_dir + d['center'].strip())
        files = shutil.copyfile(input_dir + d['right'], output_dir + d['right'].strip())

        odir = output_dir.replace(prefix_path, '')

        d['left'] = odir + d['left']
        d['center'] = odir + d['center']
        d['right'] = odir + d['right']

        csv_writer.writerow(d)

if len(sys.argv) < 6:
    sys.exit("Error in arguments: csv_file1 dir1 csv_file2 dir2 csv_file_output output_dir")

csv_file1 = sys.argv[1].strip()
dir1 = sys.argv[2].strip()
csv_file2 = sys.argv[3].strip()
dir2 = sys.argv[4].strip()
csv_file_output = sys.argv[5].strip()
output_dir = sys.argv[6].strip()

if not(os.path.isdir(output_dir)):
    os.makedirs(output_dir)

csv_reader = csv.DictReader(open(csv_file1))
input_data = []

csv_output = open(csv_file_output, 'w')
csv_writer = csv.DictWriter(csv_output, csv_reader.fieldnames)
csv_writer.writeheader()

writeRecords(csv_reader, csv_writer, dir1, output_dir, os.path.dirname(csv_file_output))

csv_reader = csv.DictReader(open(csv_file2))

writeRecords(csv_reader, csv_writer, dir2, output_dir, os.path.dirname(csv_file_output))

csv_output.close()
