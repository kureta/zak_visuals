import os
from multiprocessing import Pool


def do_multiprocess(function, args_list, num_processes=8):
    with Pool(num_processes) as p:
        results = list(p.map(function, args_list))
    return results


def recursive_file_paths(directory, extensions):
    file_paths = []
    for dirname, dirnames, filenames in os.walk(directory):
        # print path to all filenames.
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in extensions:
                file_paths.append(os.path.join(dirname, filename))

    return file_paths
