import re
import os
import pandas as pd
import requests
import random
import numpy as np
from random import shuffle
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from ast import literal_eval
import pickle
import json
import time
import patch_processor

MY_GITHUB_TOKEN = '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'
PATCH_SEPERATE_SYMBOL = '\n-----<485_PATCH_SEP_217>-----\n'
PART_SEPERATE_SYMBOL = '\n=====<485_PART_SEP_217>=====\n'
INVALID_PATCH = '(no file ends with .c , .cpp or .h)'
EMPTY_COMMIT_MESSAGE = '<EMPTY_COMMIT_MESSAGE>'
EMPTY_COMMIT_PATCH = '<EMPTY_COMMIT_PATCH>'

def my_request(query):
    headers = {'Authorization': 'token ' + MY_GITHUB_TOKEN}
    return requests.get(query, headers=headers)

def get_commit_message_and_patch_from_github(query):
    commit_message, patch = '', INVALID_PATCH
    try:
        res = my_request(query)
        data = res.json()
        commit_message = data['commit']['message']
        files = data['files']
        files_list = []
        for file in files:
            filename = file['filename']
            if filename.endswith('.c') or filename.endswith('.cpp') or filename.endswith('.h'):
                if 'patch' in file:
                    files_list.append(file['patch'])
        if files_list:
            patch = PATCH_SEPERATE_SYMBOL.join(files_list)
    except Exception as e:
        print('get_commit_message_and_patch_from_github Error (this one will not be written):', query, e)
        return None, None
    return commit_message, patch

def write_to_file(text, path, mode='a'): # 'a': append; 'w': overwrite
    with open(path, mode) as f:
        f.write(text)

def preprocess_one_single_commit_message_for_BigVul_dataset(commit_message):
    commit_message_lines = commit_message.split('\n')
    new_commit_message_lines = []
    for line in commit_message_lines:
        line = line.strip()
        if line[:3] == 'CC:' or line[:3] == 'Cc:':
            new_commit_message_lines.append('')
            continue
        by_idx = line.find('-by:')
        if by_idx >= 0:
            new_commit_message_lines.append('')
            continue
        new_commit_message_lines.append(line)

    new_commit_message = '\n'.join(new_commit_message_lines)
    # new_commit_message = re.sub('[ ]+', ' ', new_commit_message)
    new_commit_message = re.sub(r'\n+', '\n', new_commit_message)
    new_commit_message = re.sub(r'\t+', '\t', new_commit_message).strip()
    return new_commit_message

def check_ffmpeg_qemu_unique_and_get_commit_hash_set():
    def helper(directory, commit_hash_set, commit_message_dict):
        commit_message_repeated_count = 0
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if not filepath.endswith('.txt'): continue
            sha_id = filename.split('_')[2].split('.')[0]
            if sha_id in commit_hash_set:
                print(f'{sha_id} already in commit_hash_set')
                assert(False)
            commit_hash_set.add(sha_id)

            with open(filepath) as f:
                text = f.read()
                textlist = text.split(PART_SEPERATE_SYMBOL)
                commit_message = textlist[3]
                if commit_message in commit_message_dict:
                    commit_message_dict[commit_message].append(sha_id)
                    # print(f'In {directory}, the commit_message of {sha_id} already in commit_message_dict: {commit_message_dict[commit_message]}')
                    commit_message_repeated_count += 1
                else:
                    commit_message_dict[commit_message].append(sha_id)
        return commit_message_repeated_count

    commit_hash_set = set()
    commit_message_dict = defaultdict(list)
    commit_message_repeated_count = helper('ffmpeg' + '_dataset' + '/pos', commit_hash_set, commit_message_dict)
    commit_message_repeated_count += helper('ffmpeg' + '_dataset' + '/neg', commit_hash_set, commit_message_dict)
    commit_message_repeated_count += helper('qemu' + '_dataset' + '/pos', commit_hash_set, commit_message_dict)
    commit_message_repeated_count += helper('qemu' + '_dataset' + '/neg', commit_hash_set, commit_message_dict)
    print('All ffmpeg and qemu commit hash is unique; commit_message_repeated_count:', commit_message_repeated_count)
    return commit_hash_set

def is_valid_BigVul_data(row, idx, commit_hash_set, print_enable=False):
    cve_id = row['cve_id']
    cwe_id = row['cwe_id']
    link = row['ref_link']
    lang = row['lang']
    commit_id = row['commit_id']
    version_after_fix = row['version_after_fix']

    cond1 = (type(cve_id) == str and cve_id and len(cve_id) > 4 and cve_id[:4].upper() == 'CVE-') or (type(cwe_id) == str and cwe_id and len(cwe_id) > 4 and cwe_id[:4].upper() == 'CWE-')
    cond2 = '//github.com' in link and ('/commit/' in link or '/commits/' in link) # 注意处理/commits/的情况
    cond3 = lang.upper() == 'C' or lang.upper() == 'C++'
    cond4 = commit_id == version_after_fix
    cond5 = commit_id not in commit_hash_set

    if not cond1:
        if print_enable: print(f'row {idx} with commit_id {commit_id}: cond1 not satisfied')
        return False
    if not cond2:
        if print_enable: print(f'row {idx} with commit_id {commit_id}: cond2 not satisfied')
        return False
    if not cond3:
        if print_enable: print(f'row {idx} with commit_id {commit_id}: cond3 not satisfied')
        return False
    if not cond4:
        if print_enable: print(f'row {idx} with commit_id {commit_id}: cond4 not satisfied')
        return False
    if not cond5:
        if print_enable: print(f'row {idx} with commit_id {commit_id}: cond5 not satisfied')
        return False
    return True

def build_github_api_query(link, sha_id):
    if 'github.com/' not in link:
        print('github.com/ not in link')
        assert(False)

    start_idx = link.find('github.com/') + len('github.com/')
    link_list = link[start_idx:].split('/')
    a, b = link_list[0], link_list[1]
    query = f'https://api.github.com/repos/{a}/{b}/commits/{sha_id}'
    return query

def generate_BigVul_dataset_from_original_release_csv(csv_file):
    if os.path.exists('BigVul_dataset/'):
        print('BigVul_dataset/ already exists')
        assert(False)
    os.mkdir('BigVul_dataset/')
    os.mkdir('BigVul_dataset/pos/')
    BigVul_dataset_path = 'BigVul_dataset/pos/'

    commit_hash_set = check_ffmpeg_qemu_unique_and_get_commit_hash_set()
    df = pd.read_csv(csv_file)
    valid_count = 0
    total_count = 0
    file_number = 0
    for idx, row in tqdm(df.iterrows()):
        try:
            is_valid = is_valid_BigVul_data(row, idx, commit_hash_set)
        except Exception as e:
            print(f'row {idx} throws error:', e)
            assert(False)

        total_count += 1
        if is_valid:
            valid_count += 1
            link = row['ref_link']
            sha_id = row['commit_id']
            query = build_github_api_query(link, sha_id)
            commit_message, patch = get_commit_message_and_patch_from_github(query)
            if commit_message == None or patch == None:
                continue
            processed_commit_message = preprocess_one_single_commit_message_for_BigVul_dataset(commit_message)

            textlist = [sha_id, f'BigVul ({query})', '1', commit_message, processed_commit_message, patch]
            text = PART_SEPERATE_SYMBOL.join(textlist)
            file_to_write = f'{BigVul_dataset_path}/BigVul_{file_number}_{sha_id}.txt'
            write_to_file(text, file_to_write)
            # print(file_to_write)
            write_to_file(file_to_write + '\n', 'log.txt')
        file_number += 1

    print('BigVul_dataset valid_count:', valid_count)
    print('BigVul_dataset total_count:', total_count)

# ---------------------------------- Main
if __name__ == '__main__':
    generate_BigVul_dataset_from_original_release_csv('all_c_cpp_release2.0.csv')


