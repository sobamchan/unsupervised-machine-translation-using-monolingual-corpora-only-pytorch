import re
from tqdm import tqdm


def main(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    # header = lines[0]
    data = lines[1:]
    dic = {}
    for l in tqdm(data):
        values_str = []
        values_float = []
        for val in re.findall(r"[-+]?\d*\.\d+|\d+", l):
            values_str.append(val)
            values_float.append(float(val))

        word = l.replace(' '.join(values_str), '').replace('\n', '')[:-1]
        dic[word] = values_float
    return dic


if __name__ == '__main__':
    d = main('./input/vectors-en.txt')
