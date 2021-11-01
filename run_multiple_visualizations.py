# Written in python2.7, meant to be run in the saliency conda env.
import os
from cfg import *
import numpy as np
import argparse

cfg = flickr_cfg()

with open(cfg.val_file, "r") as f:
    lines = f.readlines()
    val_ex = []
    for line in lines:
        val_ex.append(int(line.rstrip().split(".jpg")[0]))

with open(cfg.test_file, "r") as f:
    lines = f.readlines()
    test_ex = []
    for line in lines:
        test_ex.append(int(line.rstrip().split(".jpg")[0]))

with open(cfg.annotations_path, "r") as f:
    lines = f.readlines()
    file_id_to_annotation_map = {} # int: str
    for example in lines:
        filename, annotation = example.split("\t")
        file_id = int(filename.split(".jpg")[0]) # removes the .jpg
        if file_id in test_ex:
            file_id_to_annotation_map[file_id] = annotation.rstrip()

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, required=False)
parser.add_argument("--checkpoint", type=int, required=True)
parser.add_argument("--num-evals", type=int, required=True)
args = parser.parse_args()

np.random.seed(0)
file_ids_to_eval = np.random.choice(list(file_id_to_annotation_map.keys()), args.num_evals)
print("file_ids_to_eval", file_ids_to_eval)

for eval_id in file_ids_to_eval:
    caption = file_id_to_annotation_map[eval_id]
    command = 'python visualization.py --dataset Flickr30k --media_id {}  --checkpoint {} --sentence "{}" --gpu {}'.format(eval_id, args.checkpoint, caption, args.gpu)
    print(command)
    os.system(command)

# produce local copy commands
commands = []
for eval_id in file_ids_to_eval:
    output_path = "/scratch/cluster/albertyu/dev/caption-guided-saliency/output_samples"
    command = "scp titan1:{}/{}/*.png .".format(output_path, eval_id)
    commands.append(command)
print("Copy commands")
print(commands)
