import os
import json
import numpy as np
import csv
import optparse
from copy import deepcopy
import sys


def parsing(json_path):
    with open(json_path) as json_file:
        parsed_data = json.load(json_file, encoding='UTF-8')

        polygon_shapes = parsed_data['shapes']

        # EACH SHAPE
        for polygon_data in polygon_shapes:
            label = polygon_data['label']
            points = np.array(polygon_data['points'])
            path = deepcopy(json_path).replace('.json', '.jpg')
            return path, points, label


def search(path):
    print(f'searching path : {path}')
    # check is directory.
    for target in os.listdir(path):

        target_path = os.path.join(path, target)

        # if path.
        if os.path.isdir(target_path):
            print(f'? next target : {target_path}')
            # search.
            search(deepcopy(target_path))

        # not directory and jsonfile.
        elif target_path.endswith('.json'):
            print(f'?  ?processing : {target}')
            # parsing.
            img_path, points, target_label = parsing(deepcopy(target_path))
            wr.writerow([img_path, points[0, 0], points[0, 1], points[1, 0], points[1, 1], target_label])
            print(f'?  ?path : {img_path}, label {target_label}')
        else:
            pass


if __name__ == '__main__':
    parser = optparse.OptionParser("LabelMeParser", version='1.0.0')
    parser.add_option('-p', '--target_path', action='store', dest='target_path', help='Set parent directory of target path.')
    parser.add_option('-o', '--output_path', action='store', dest='output_path', help='Set output path.')
    (options, args) = parser.parse_args(sys.argv)

    assert options.target_path is not None, 'Please set target path. See -h, --help options.'
    assert options.output_path is not None, 'Please set output path. See -h, --help options.'

    parent_path = options.target_path
    f = open(options.output_path, 'w', newline='', encoding='utf-8')
    wr = csv.writer(f)
    search(parent_path)
    f.close()
    print(f'? Finish.')
