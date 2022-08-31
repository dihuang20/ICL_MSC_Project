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

def preprocessing(config, target_list):
    def build_json_data_helper(config, data_masking, output_subdir, pos_list, neg_list, process_mode):
        RANDOM_SEED = config['random_seed']

        pos_count = len(pos_list)
        neg_count = len(neg_list)
        print(f'Building {output_subdir}:', f'{pos_count}(pos),', f'{neg_count}(neg),', f'{pos_count + neg_count}(total)')

        random.seed(RANDOM_SEED)
        pos_samples = random.sample(pos_list, pos_count)
        neg_samples = random.sample(neg_list, neg_count)

        pos_train_data, pos_test_data = np.split(pos_samples, [int(.75 * pos_count)])
        neg_train_data, neg_test_data = np.split(neg_samples, [int(.75 * neg_count)])

        pos_train_data = [{'label':1, 'commit_patch':sample[1], 'commit_message':sample[0]} for sample in pos_train_data]
        neg_train_data = [{'label':0, 'commit_patch':sample[1], 'commit_message':sample[0]} for sample in neg_train_data]
        pos_test_data = [{'label':1, 'commit_patch':sample[1], 'commit_message':sample[0]} for sample in pos_test_data]
        neg_test_data = [{'label':0, 'commit_patch':sample[1], 'commit_message':sample[0]} for sample in neg_test_data]

        # data masking
        if data_masking:
            data_masking_rate = config['data_masking_rate']
            for i in range(int(data_masking_rate * len(pos_train_data))): pos_train_data[i]['commit_message'] = CUSTOM_DATA_MASK
            for i in range(int(data_masking_rate * len(neg_train_data))): neg_train_data[i]['commit_message'] = CUSTOM_DATA_MASK
            for i in range(int(data_masking_rate * len(pos_test_data))): pos_test_data[i]['commit_message'] = CUSTOM_DATA_MASK
            for i in range(int(data_masking_rate * len(neg_test_data))): neg_test_data[i]['commit_message'] = CUSTOM_DATA_MASK

        train_data = pos_train_data + neg_train_data
        test_data = pos_test_data + neg_test_data
        val_data = test_data

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

    include_empty_commit_patch = config['include_empty_commit_patch']
    pos_list_combined_1 = []
    neg_list_combined_1 = []
    pos_list_combined_0 = []
    neg_list_combined_0 = []

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

            if not patch or type(patch) != str:
                print(f'##### not patch or type(patch) != str:({patch})')
                if include_empty_commit_patch:
                    neg_list_1.append([commit_msg_1, EMPTY_COMMIT_PATCH])
                continue

            commit_patch_0 = baseline_preprocessing_single_sample(config, row['patch'])
            commit_patch_1 = patch_processor.preprocess_single_commit_patch(config, patch)
            if commit_patch_1 == EMPTY_COMMIT_PATCH or commit_patch_1 == ONLY_COMMENT_COMMIT_PATCH:
                # print('commit_patch_1 == EMPTY_COMMIT_PATCH or commit_patch_1 == ONLY_COMMENT_COMMIT_PATCH:\n', patch)
                # print('commit_patch_1 == EMPTY_COMMIT_PATCH or commit_patch_1 == ONLY_COMMENT_COMMIT_PATCH:\n')
                if include_empty_commit_patch:
                    neg_list_1.append([commit_msg_1, EMPTY_COMMIT_PATCH])
                continue
            if not commit_patch_0:
                if include_empty_commit_patch:
                    neg_list_0.append([commit_msg_1, EMPTY_COMMIT_PATCH])
                continue

            if int(label) == 0:
                neg_list_1.append([commit_msg_1, commit_patch_1])
                neg_list_0.append([commit_msg_1, commit_patch_0])

            if int(label) == 1:
                pos_list_1.append([commit_msg_1, commit_patch_1])
                pos_list_0.append([commit_msg_1, commit_patch_0])

        build_json_data_helper(config, False, target, pos_list_1, neg_list_1, 1)
        build_json_data_helper(config, True, f'masked_{target}', pos_list_1, neg_list_1, 1)

        build_json_data_helper(config, False, target, pos_list_0, neg_list_0, 0)
        build_json_data_helper(config, True, f'masked_{target}', pos_list_0, neg_list_0, 0)

        pos_list_combined_1.extend(pos_list_1)
        neg_list_combined_1.extend(neg_list_1)
        pos_list_combined_0.extend(pos_list_0)
        neg_list_combined_0.extend(neg_list_0)

    build_json_data_helper(config, False, 'combined', pos_list_combined_1, neg_list_combined_1, 1)
    build_json_data_helper(config, True, 'masked_combined', pos_list_combined_1, neg_list_combined_1, 1)

    build_json_data_helper(config, False, 'combined', pos_list_combined_0, neg_list_combined_0, 0)
    build_json_data_helper(config, True, 'masked_combined', pos_list_combined_0, neg_list_combined_0, 0)

def dataset_statistics():
    def get_data_amount_from_csv(path):
        pos_count = 0
        neg_count = 0
        df = pd.read_csv(path)
        for idx, row in df.iterrows():
            label = int(row['vulnerability'])
            if label == 0: neg_count += 1
            if label == 1: pos_count += 1
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

    def get_masked_data_amount_from_json(path):
        pos_count = 0
        neg_count = 0
        with open(path) as f:
            json_data = json.load(f)
            for data in json_data:
                msg = data['commit_message']
                if msg != CUSTOM_DATA_MASK: continue
                label = int(data['label'])
                if label == 0: neg_count += 1
                if label == 1: pos_count += 1
        return neg_count + pos_count, pos_count, neg_count

    print('init ffmpeg:', get_data_amount_from_csv('SPI_init_csv_dataset/ffmpeg.csv'))

    a, b, c = get_data_amount_from_json('output_dataset_1/ffmpeg/train.json')
    d, e, f = get_data_amount_from_json('output_dataset_1/ffmpeg/val.json')
    print('json ffmpeg train:', (a, b, c))
    print('json ffmpeg val:', (d, e, f))
    print('json ffmpeg total:', (a+d, b+e, c+f))

    a, b, c = get_data_amount_from_json('output_dataset_1/masked_ffmpeg/train.json')
    d, e, f = get_data_amount_from_json('output_dataset_1/masked_ffmpeg/val.json')
    print('json masked_ffmpeg train:', (a, b, c))
    print('json masked_ffmpeg val:', (d, e, f))
    print('json masked_ffmpeg total:', (a+d, b+e, c+f))

    print('json masked_ffmpeg masked data in train:', get_masked_data_amount_from_json('output_dataset_1/masked_ffmpeg/train.json'))
    print('json masked_ffmpeg masked data in val:', get_masked_data_amount_from_json('output_dataset_1/masked_ffmpeg/val.json'))

    print('--------------------------------------')

    print('init qemu:', get_data_amount_from_csv('SPI_init_csv_dataset/qemu.csv'))

    a, b, c = get_data_amount_from_json('output_dataset_1/qemu/train.json')
    d, e, f = get_data_amount_from_json('output_dataset_1/qemu/val.json')
    print('json qemu train:', (a, b, c))
    print('json qemu val:', (d, e, f))
    print('json qemu total:', (a+d, b+e, c+f))

    a, b, c = get_data_amount_from_json('output_dataset_1/masked_qemu/train.json')
    d, e, f = get_data_amount_from_json('output_dataset_1/masked_qemu/val.json')
    print('json masked_qemu train:', (a, b, c))
    print('json masked_qemu val:', (d, e, f))
    print('json masked_qemu total:', (a+d, b+e, c+f))


if __name__ == '__main__':
    config = {
               'output_dataset_directory': 'output_dataset',
               'random_seed': 0,
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

    preprocessing(config, ['ffmpeg', 'qemu'])

    dataset_statistics()

