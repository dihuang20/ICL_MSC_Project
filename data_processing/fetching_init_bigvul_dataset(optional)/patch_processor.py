import re
import pandas as pd

INVALID_PATCH = '(no file ends with .c , .cpp or .h)'
EMPTY_COMMIT_PATCH = '<EMPTY_COMMIT_PATCH>'

# check if a commit patch is valid
def is_commit_patch_valid(commit_patch):
    commit_patch = commit_patch.strip()
    if not commit_patch or commit_patch == INVALID_PATCH or commit_patch == EMPTY_COMMIT_PATCH: return False
    return True

# remove comments within c/c++ source code
# Reference: this function refers to https://stackoverflow.com/a/241506
def comment_remover(source_code):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, source_code)

# check if the line is code block header
def is_code_block_header(line):
    return line.startswith('@@') or line == '{...}'

# check if the line is diff header
def is_diff_header(line):
    return line.startswith('diff --git')

# check if the line is revision line (starting with '+' or '-')
def is_revision_line(line):
    return line.startswith('+') or line.startswith('-')

# convert list of lines to a string
def custom_join(lines_list):
    res = ''
    for line in lines_list:
        line = line.strip()
        if not line: continue
        res += line
        if not line.endswith(';') and not line.endswith('}'):
            res += ' '
    return res

# merge revisions (lines starting with '+' or '-')
def merge_revisions_helper(lines_list):
    def adding_revisions_to_new_list(revisions, sign, new_lines_list):
        revisions_str = custom_join(revisions).strip()
        if revisions_str: new_lines_list.append(f'{sign}{{{revisions_str}}}')

    new_lines_list = []
    i = 0
    while i < len(lines_list):
        if lines_list[i].startswith('+'):
            revisions = []
            while i < len(lines_list) and lines_list[i].startswith('+'):
                line = lines_list[i][1:].strip()
                if line: revisions.append(line)
                i += 1
            if revisions: adding_revisions_to_new_list(revisions, '+', new_lines_list)
            continue

        if lines_list[i].startswith('-'):
            revisions = []
            while i < len(lines_list) and lines_list[i].startswith('-'):
                line = lines_list[i][1:].strip()
                if line: revisions.append(line)
                i += 1
            if revisions: adding_revisions_to_new_list(revisions, '-', new_lines_list)
            continue

        line = lines_list[i].strip()
        if line: new_lines_list.append(line)
        i += 1

    return new_lines_list

# central diffusion algorithm
def central_diffusion_helper(config, code_block_body):
    def get_idx_of_first_revision_line(lines_list):
        for i in range(len(lines_list)):
            if is_revision_line(lines_list[i]):
                return i
        return -1

    # configuration
    central_diffusion_range = config['central_diffusion_range'] # only work for processing mode 1

    # remove trivial lines
    lines_list = code_block_body
    tmp_lines_list = []
    for line in lines_list:
        line = line.strip()
        if len(line) > 1:
            tmp_lines_list.append(line)
        elif len(line) == 1 and not is_revision_line(line):
            tmp_lines_list.append(line)
    lines_list = tmp_lines_list

    # add context lines
    new_lines_list = []
    cur_i = get_idx_of_first_revision_line(lines_list)
    while cur_i >= 0:
        new_lines_list.append(lines_list[cur_i])

        idx_to_insert = len(new_lines_list) - 1
        count = central_diffusion_range
        upper_i = cur_i - 1
        while upper_i >= 0 and count > 0 and not is_revision_line(lines_list[upper_i]):
            new_lines_list.insert(idx_to_insert, lines_list[upper_i])
            count -= 1
            upper_i -= 1

        count = central_diffusion_range
        lower_i = cur_i + 1
        while lower_i < len(lines_list) and count > 0 and not is_revision_line(lines_list[lower_i]):
            new_lines_list.append(lines_list[lower_i])
            count -= 1
            lower_i += 1

        lines_list = lines_list[lower_i:]
        cur_i = get_idx_of_first_revision_line(lines_list)

    return new_lines_list

# get first diff of commit patch
def get_first_diff(commit_patch):
    lines_list = commit_patch.split('\n')
    for i in range(len(lines_list)):
        if is_code_block_header(lines_list[i]):
            lines_list = lines_list[i:]
            break
    for i in range(len(lines_list)):
        if is_diff_header(lines_list[i]):
            lines_list = lines_list[:i]
            break
    return '\n'.join(lines_list)

# remove all diff information within commit patch
def remove_diff(commit_patch):
    lines_list = commit_patch.split('\n')
    new_lines_list = []
    is_diff_info = False
    for line in lines_list:
        if is_diff_header(line):
            is_diff_info = True
            continue
        if is_diff_info and is_code_block_header(line):
            is_diff_info = False
        if not is_diff_info:
            new_lines_list.append(line)
    return '\n'.join(new_lines_list)

# preprocess a single commit patch
def preprocess_single_commit_patch(config, commit_patch):
    # return EMPTY_COMMIT_PATCH if commit patch is invalid
    if not is_commit_patch_valid(commit_patch): return EMPTY_COMMIT_PATCH

    # configuration
    maximum_lines = config['maximum_lines']
    only_using_first_diff = config['only_using_first_diff']
    removing_comments = config['removing_comments']
    removing_redundant_whitespace = config['removing_redundant_whitespace']
    keeping_code_block_headers = config['keeping_code_block_headers'] # line starting with '@@'
    merging_revisions = config['merging_revisions']
    commit_patch_processing_mode = config['commit_patch_processing_mode'] # mode 0 or 1

    # process diff
    if only_using_first_diff:
        commit_patch = get_first_diff(commit_patch)
    else:
        commit_patch = remove_diff(commit_patch)

    # remove comments
    if removing_comments:
        commit_patch = comment_remover(commit_patch)

    # remove redundant whitespace
    if removing_redundant_whitespace:
        commit_patch = re.sub(r'\t', '', commit_patch)
        commit_patch = re.sub('[ ]+', ' ', commit_patch)
        commit_patch = commit_patch.strip()

    # convert commit patch to list of lines
    lines_list = commit_patch.split('\n')
    for i in range(len(lines_list)):
        if is_code_block_header(lines_list[i]):
            lines_list = lines_list[i:]
            break
    if not lines_list: return EMPTY_COMMIT_PATCH
    if len(lines_list) == 1 and is_code_block_header(lines_list[0]): return EMPTY_COMMIT_PATCH

    # keep code block headers
    for i in range(len(lines_list)):
        if is_code_block_header(lines_list[i]) and not keeping_code_block_headers:
            lines_list[i] = '{...}'

    # merge revisions
    if merging_revisions:
        lines_list = merge_revisions_helper(lines_list)
        if not lines_list: return EMPTY_COMMIT_PATCH

    # return if processing mode is 0
    if not is_code_block_header(lines_list[0]): return EMPTY_COMMIT_PATCH
    if commit_patch_processing_mode == 0:
        lines_list = lines_list[:maximum_lines]
        commit_patch = custom_join(lines_list)
        if removing_redundant_whitespace:
            commit_patch = re.sub('[ ]+', ' ', commit_patch)
            commit_patch = commit_patch.strip()
        if not is_commit_patch_valid(commit_patch): return EMPTY_COMMIT_PATCH
        return commit_patch

    # convert list of lines to list of code blocks
    code_block_list = []
    code_block_header_idx = 0
    for i in range(len(lines_list)):
        line = lines_list[i]
        if is_code_block_header(line):
            code_block_header_idx = i
            continue
        if i + 1 == len(lines_list) or is_code_block_header(lines_list[i + 1]):
            if i > code_block_header_idx:
                code_block = [lines_list[code_block_header_idx]]
                code_block_body = central_diffusion_helper(config, lines_list[code_block_header_idx + 1 : i + 1])
                if code_block_body:
                    code_block.extend(code_block_body)
                    code_block_list.append(code_block)
    if not code_block_list: return EMPTY_COMMIT_PATCH

    # convert list of code blocks to string
    lines_list = []
    for code_block in code_block_list:
        lines_list.extend(code_block)
    lines_list = lines_list[:maximum_lines]
    commit_patch = custom_join(lines_list)
    if removing_redundant_whitespace:
        commit_patch = re.sub('[ ]+', ' ', commit_patch)
        commit_patch = commit_patch.strip()
    if not is_commit_patch_valid(commit_patch): return EMPTY_COMMIT_PATCH
    return commit_patch

# unit testing
if __name__ == '__main__':
    config = {
               'only_using_first_diff': False,
               'removing_comments': True,
               'removing_redundant_whitespace': True,
               'keeping_code_block_headers': False,
               'merging_revisions': True,
               'commit_patch_processing_mode': 1,
               'central_diffusion_range': 4,
               'maximum_lines': 50,
             }

    path = 'SPI_init_csv_dataset/ffmpeg.csv'
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        commit_patch = row['patch']
        processed_commit_patch = preprocess_single_commit_patch(config, commit_patch)
        print(processed_commit_patch)
        print('---------------------------------')
        assert(False)



