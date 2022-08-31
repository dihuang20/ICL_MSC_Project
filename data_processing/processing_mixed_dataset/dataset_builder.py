import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import random
import os
import json
import patch_processor
from patch_processor import comment_remover

EMPTY_COMMIT_PATCH = '<EMPTY_COMMIT_PATCH>'
ONLY_COMMENT_COMMIT_PATCH = '<ONLY_COMMENT_COMMIT_PATCH>'
CUSTOM_DATA_MASK = '<CUSTOM_DATA_MASK>'
INPUT_DATA_DIRECTORY = 'SPI_init_csv_dataset'
INVALID_PATCH = '(no file ends with .c , .cpp or .h)'
PATCH_SEPERATE_SYMBOL = '\n-----<485_PATCH_SEP_217>-----\n'
PART_SEPERATE_SYMBOL = '\n=====<485_PART_SEP_217>=====\n'

def mkdir_if_not_exist(directory):
    if not directory: return
    if not os.path.exists(directory):
        os.mkdir(directory)

def baseline_preprocessing_single_sample(config, commit_patch):
    maximum_lines = config['maximum_lines']

    commit_patch = comment_remover(commit_patch)

    commit_patch = ' '.join(commit_patch.split('\n')[:maximum_lines])
    commit_patch = re.sub(r'\t', '', commit_patch)
    commit_patch = re.sub('[ ]+', ' ', commit_patch)
    commit_patch = commit_patch.strip()

    return commit_patch

def build_json_data_helper(config, data_masking, output_subdir, pos_list, neg_list, process_mode):
    RANDOM_SEED = config['random_seed']
    train_split_idx_rate = config['train_val_test_split_rates'][0]
    val_split_idx_rate = config['train_val_test_split_rates'][0] + config['train_val_test_split_rates'][1]

    pos_count = len(pos_list)
    neg_count = len(neg_list)
    print(f'Building {output_subdir} mode={process_mode}:', f'{pos_count}(pos),', f'{neg_count}(neg),', f'{pos_count + neg_count}(total)')

    random.seed(RANDOM_SEED)
    pos_samples = random.sample(pos_list, pos_count)
    random.seed(RANDOM_SEED)
    neg_samples = random.sample(neg_list, neg_count)

    # pos_train_data, pos_val_data, pos_test_data = np.split(pos_samples, [int(train_split_idx_rate * pos_count), int(val_split_idx_rate * pos_count)])
    # neg_train_data, neg_val_data, neg_test_data = np.split(neg_samples, [int(train_split_idx_rate * neg_count), int(val_split_idx_rate * neg_count)])
    pos_train_data, pos_val_data = np.split(pos_samples, [int(.8 * pos_count)])
    neg_train_data, neg_val_data = np.split(neg_samples, [int(.8 * neg_count)])
    print(output_subdir, 'pos', len(pos_train_data), len(pos_val_data))
    print(output_subdir, 'neg', len(neg_train_data), len(neg_val_data))

    pos_train_data = [{'id': sample[2], 'label':1, 'commit_patch':sample[1], 'commit_message':sample[0]} for sample in pos_train_data]
    neg_train_data = [{'id': sample[2], 'label':0, 'commit_patch':sample[1], 'commit_message':sample[0]} for sample in neg_train_data]
    pos_val_data = [{'id': sample[2], 'label':1, 'commit_patch':sample[1], 'commit_message':sample[0]} for sample in pos_val_data]
    neg_val_data = [{'id': sample[2], 'label':0, 'commit_patch':sample[1], 'commit_message':sample[0]} for sample in neg_val_data]
    # pos_test_data = [{'id': sample[2], 'label':1, 'commit_patch':sample[1], 'commit_message':sample[0]} for sample in pos_test_data]
    # neg_test_data = [{'id': sample[2], 'label':0, 'commit_patch':sample[1], 'commit_message':sample[0]} for sample in neg_test_data]

    # data masking
    if data_masking:
        data_masking_rate = config['data_masking_rate']
        for i in range(int(data_masking_rate * len(pos_train_data))): pos_train_data[i]['commit_message'] = CUSTOM_DATA_MASK
        for i in range(int(data_masking_rate * len(neg_train_data))): neg_train_data[i]['commit_message'] = CUSTOM_DATA_MASK
        for i in range(int(data_masking_rate * len(pos_test_data))): pos_test_data[i]['commit_message'] = CUSTOM_DATA_MASK
        for i in range(int(data_masking_rate * len(neg_test_data))): neg_test_data[i]['commit_message'] = CUSTOM_DATA_MASK

    train_data = pos_train_data + neg_train_data
    val_data = pos_val_data + neg_val_data
    # test_data = pos_test_data + neg_test_data
    test_data = val_data

    output_dataset_directory = config['output_dataset_directory'] + '_' + str(process_mode)
    mkdir_if_not_exist(output_dataset_directory)
    output_dataset_directory = f'{output_dataset_directory}/{output_subdir}'
    mkdir_if_not_exist(output_dataset_directory)

    random.seed(RANDOM_SEED)
    random.shuffle(train_data)
    with open(f'{output_dataset_directory}/train.json', 'w') as f:
        json.dump(train_data, f, indent=4)

    random.seed(RANDOM_SEED)
    random.shuffle(test_data)
    with open(f'{output_dataset_directory}/test.json', 'w') as f:
        json.dump(test_data, f, indent=4)

    random.seed(RANDOM_SEED)
    random.shuffle(val_data)
    with open(f'{output_dataset_directory}/val.json', 'w') as f:
        json.dump(val_data, f, indent=4)

def preprocessing(config, target_list):
    include_empty_commit_patch = config['include_empty_commit_patch']
    positive_extraction_rate = config['positive_extraction_rate']
    negative_extraction_rate = config['negative_extraction_rate']
    RANDOM_SEED = config['random_seed']
    commit_message_set = set()

    for target in target_list:
        directory = f'{INPUT_DATA_DIRECTORY}/{target}.csv'
        df = pd.read_csv(directory)

        pos_list_0 = []
        neg_list_0 = []
        pos_list_1 = []
        neg_list_1 = []

        for idx, row in tqdm(df.iterrows()):
            patch = row['patch']
            commit_msg = row['commit_msg']
            label = row['vulnerability']

            if not commit_msg or type(commit_msg) != str:
                print('commit_msg is empty')
                assert(False)

            commit_msg_1 = commit_msg
            if not commit_msg_1:
                print('preprocessed commit_msg is empty')
                assert(False)

            commit_message_set.add(commit_msg_1)

            if not patch or type(patch) != str:
                # print('patch is empty')
                if include_empty_commit_patch:
                    neg_list_1.append([commit_msg_1, EMPTY_COMMIT_PATCH, f'{target}_{idx}'])
                continue

            commit_patch_0 = baseline_preprocessing_single_sample(config, row['patch'])
            commit_patch_1 = patch_processor.preprocess_single_commit_patch(config, patch)
            if commit_patch_1 == EMPTY_COMMIT_PATCH or commit_patch_1 == ONLY_COMMENT_COMMIT_PATCH:
                # print('commit_patch_1 is EMPTY_COMMIT_PATCH or ONLY_COMMENT_COMMIT_PATCH:\n', patch)
                if include_empty_commit_patch:
                    neg_list_1.append([commit_msg_1, EMPTY_COMMIT_PATCH, f'{target}_{idx}'])
                continue
            if not commit_patch_0:
                if include_empty_commit_patch:
                    neg_list_0.append([commit_msg_1, EMPTY_COMMIT_PATCH, f'{target}_{idx}'])
                continue

            if int(label) == 0:
                neg_list_1.append([commit_msg_1, commit_patch_1, f'{target}_{idx}'])
                neg_list_0.append([commit_msg_1, commit_patch_0, f'{target}_{idx}'])

            if int(label) == 1:
                pos_list_1.append([commit_msg_1, commit_patch_1, f'{target}_{idx}'])
                pos_list_0.append([commit_msg_1, commit_patch_0, f'{target}_{idx}'])

        random.seed(RANDOM_SEED)
        pos_list_1 = random.sample(pos_list_1, int(len(pos_list_1) * positive_extraction_rate))
        random.seed(RANDOM_SEED)
        neg_list_1 = random.sample(neg_list_1, int(len(neg_list_1) * negative_extraction_rate))
        random.seed(RANDOM_SEED)
        pos_list_0 = random.sample(pos_list_0, int(len(pos_list_0) * positive_extraction_rate))
        random.seed(RANDOM_SEED)
        neg_list_0 = random.sample(neg_list_0, int(len(neg_list_0) * negative_extraction_rate))

        build_json_data_helper(config, False, target, pos_list_1, neg_list_1, 1)
        # build_json_data_helper(config, False, target, pos_list_0, neg_list_0, 0)

    return commit_message_set

def preprocessing_bigvul(config, commit_message_set):
    RANDOM_SEED = config['random_seed']

    init_file_path = 'all_c_cpp_release2.0.csv'
    shaID2msg = {}
    df = pd.read_csv(init_file_path)
    for idx, row in tqdm(df.iterrows()):
        sha_id = row['commit_id']
        if row['commit_message'] and type(row['commit_message']) == str: commit_message = row['commit_message']
        else: commit_message = ''
        shaID2msg[sha_id] = commit_message

    directory = 'BigVul_dataset/pos'
    pos_list_0 = []
    pos_list_1 = []
    neg_list_0 = []
    neg_list_1 = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if not filepath.endswith('.txt'): continue
        with open(filepath) as f:
            textlist = f.read().split(PART_SEPERATE_SYMBOL)
            sha_id = textlist[0]
            commit_msg = shaID2msg[sha_id]
            patch = textlist[-1]

            if commit_msg in commit_message_set: continue
            if not commit_msg or type(commit_msg) != str: commit_msg = CUSTOM_DATA_MASK
            if not patch or type(patch) != str or patch == INVALID_PATCH: continue

            commit_patch_0 = baseline_preprocessing_single_sample(config, patch)
            commit_patch_1 = patch_processor.preprocess_single_commit_patch(config, patch)
            if commit_patch_0 and type(commit_patch_0) == str:
                pos_list_0.append([commit_msg, commit_patch_0, f'bigvul_{sha_id}'])
            if commit_patch_1 != EMPTY_COMMIT_PATCH and commit_patch_1 != ONLY_COMMENT_COMMIT_PATCH:
                pos_list_1.append([commit_msg, commit_patch_1, f'bigvul_{sha_id}'])

    random.seed(RANDOM_SEED)
    pos_list_0 = random.sample(pos_list_0, len(pos_list_0))
    random.seed(RANDOM_SEED)
    pos_list_1 = random.sample(pos_list_1, len(pos_list_1))

    build_json_data_helper(config, False, 'bigvul', pos_list_1, neg_list_1, 1)
    # build_json_data_helper(config, False, 'bigvul', pos_list_0, neg_list_0, 0)

def combine_datasets():
    def combine_datasets_helper(directory, target_list, data_file):
        combined_data = []
        mkdir_if_not_exist(f'{directory}/combined')

        count = 0
        for target in tqdm(target_list):
            path = f'{directory}/{target}/{data_file}'
            with open(path) as f:
                json_data = json.load(f)
                for data in json_data:
                    combined_data.append(dict(data))

            random.seed(0)
            random.shuffle(combined_data)
            count += len(combined_data)
            with open(f'{directory}/combined/{data_file}', 'w') as f:
                json.dump(combined_data, f, indent=4)
        print(data_file, count)

    print('Combining datasets...')
    # combine_datasets_helper('output_dataset_0', ['bigvul', 'ffmpeg', 'qemu'], 'train.json')
    # combine_datasets_helper('output_dataset_0', ['bigvul', 'ffmpeg', 'qemu'], 'val.json')
    # combine_datasets_helper('output_dataset_0', ['bigvul', 'ffmpeg', 'qemu'], 'test.json')

    combine_datasets_helper('output_dataset_1', ['bigvul', 'ffmpeg', 'qemu'], 'train.json')
    combine_datasets_helper('output_dataset_1', ['bigvul', 'ffmpeg', 'qemu'], 'val.json')
    combine_datasets_helper('output_dataset_1', ['bigvul', 'ffmpeg', 'qemu'], 'test.json')

def dataset_statistics():
    def get_data_amount_from_csv(path):
        pos_count = 0
        neg_count = 0
        df = pd.read_csv(path)
        for idx, row in df.iterrows():
            pos_count += 1
        return neg_count + pos_count, pos_count, neg_count

    def get_data_amount_from_json(path):
        pos_count = 0
        neg_count = 0
        with open(path) as f:
            json_data = json.load(f)
            for data in json_data:
                label = int(data['label'])
                if label == 0: neg_count += 1
                if label == 1: pos_count += 1
        return neg_count + pos_count, pos_count, neg_count

    def get_data_amount_from_subdataset(path, subdataset):
        pos_count = 0
        neg_count = 0
        with open(path) as f:
            json_data = json.load(f)
            for data in json_data:
                id_ = data['id']
                label = int(data['label'])
                if not id_.startswith(subdataset): continue
                if label == 0: neg_count += 1
                if label == 1: pos_count += 1
        return neg_count + pos_count, pos_count, neg_count

    print('init bigvul:', get_data_amount_from_csv('all_c_cpp_release2.0.csv'))

    print('------------------')

    a, b, c = get_data_amount_from_json('output_dataset_1/combined/train.json')
    d, e, f = get_data_amount_from_json('output_dataset_1/combined/val.json')
    print('json combined train:', (a, b, c))
    print('json combined val:', (d, e, f))
    print('json combined total:', (a+d, b+e, c+f))

    print('------------------')

    a, b, c = get_data_amount_from_subdataset('output_dataset_1/combined/train.json', 'bigvul_')
    d, e, f = get_data_amount_from_subdataset('output_dataset_1/combined/val.json', 'bigvul_')
    print('bigvul subdataset train:', (a, b, c))
    print('bigvul subdataset val:', (d, e, f))
    print('bigvul subdataset total:', (a+d, b+e, c+f))

    print('------------------')

    a, b, c = get_data_amount_from_subdataset('output_dataset_1/combined/train.json', 'ffmpeg_')
    d, e, f = get_data_amount_from_subdataset('output_dataset_1/combined/val.json', 'ffmpeg_')
    print('ffmpeg subdataset train:', (a, b, c))
    print('ffmpeg subdataset val:', (d, e, f))
    print('ffmpeg subdataset total:', (a+d, b+e, c+f))

    print('------------------')

    a, b, c = get_data_amount_from_subdataset('output_dataset_1/combined/train.json', 'qemu_')
    d, e, f = get_data_amount_from_subdataset('output_dataset_1/combined/val.json', 'qemu_')
    print('qemu subdataset train:', (a, b, c))
    print('qemu subdataset val:', (d, e, f))
    print('qemu subdataset total:', (a+d, b+e, c+f))

if __name__ == '__main__':
    config = {
               'output_dataset_directory': 'output_dataset',
               'random_seed': 0,
               'positive_extraction_rate': 0.8, # 0.6
               'negative_extraction_rate': 0.8, # 0.8
               'train_val_test_split_rates': [.75, .2, .05],
               # ----- for commit patch:
               'only_using_first_diff': False,
               'removing_comments': True,
               'removing_redundant_whitespace': True,
               'keeping_code_block_headers': False,
               'merging_revisions': False,
               'commit_patch_processing_mode': 1,
               'central_diffusion_range': 3,
               'maximum_lines': 300,
               'include_empty_commit_patch': False,
               # ----- for commit message:
               'data_masking_rate': 0.4,
             }

    commit_message_set = preprocessing(config, ['ffmpeg', 'qemu'])
    preprocessing_bigvul(config, commit_message_set)
    combine_datasets()

    dataset_statistics()



