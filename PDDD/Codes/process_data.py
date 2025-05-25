
import os
import pandas as pd
import argparse

# HACK: preprocessing the raw dataset.

parser = argparse.ArgumentParser()
parser.add_argument('--base_file', type=str, help='Path to the base folder of the dataset')
args = parser.parse_args()

base_file = args.base_file

label_id = os.listdir(os.path.join(base_file, 'images'))

label_id = dict(zip(label_id, range(len(label_id))))

index = 0
print("1")
files = open(base_file + '/all_data.csv', 'w')
files.write('id\ttext\tfile_path\tlabel\n')
for name, id in label_id.items():
    image_list = os.listdir(os.path.join(base_file, 'images', name))
    for i in image_list:
        if 'jpg' not in i:
            continue

        image_path = os.path.join('images', name, i)
        text_path = os.path.join(base_file, 'captions', name, i.replace('.jpg', '.txt'))
        temp_text = open(text_path, 'r').readline().strip()
        files.write(str(index) + '\t' + temp_text + '\t' + image_path + '\t' + str(id) + '\n')
        index += 1
        # print(image_path, temp_text)


files.close()
data = pd.read_csv(base_file + '/all_data.csv', sep='\t')

test = data.sample(frac=0.2, random_state=0, axis=0)

train = data[~data.index.isin(test.index)]
test.to_csv(base_file + '/test.csv', sep='\t', index=False)
train.to_csv(base_file + '/train.csv', sep='\t', index=False)


