#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path

def get_filesize(path):
    return os.path.getsize(path)

def pick_up_retry_setting(filename):
    output_filename_sucess = filename + '.sucess'
    output_filename = filename + '.retry'
    f = open(output_filename, 'w')
    f_s = open(output_filename_sucess, 'w')
    for l in open(filename):
        l = l.strip()

        log_filename = l.split('> ')[-1]
        size = 0
        if os.path.exists(log_filename):
            size = get_filesize(log_filename)

        if size < 1000:
            print log_filename
            print size
            cmd = l
            # cmd = cmd.replace('--gpu=0', '--gpu=1')
            f.write(cmd + '\n')
        else:
            f_s.write(log_filename + '\n')


def main():
    import sys
    filename = sys.argv[1]
    pick_up_retry_setting(filename)

if __name__ == '__main__':
    main()