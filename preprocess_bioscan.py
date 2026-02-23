import csv
from tqdm import tqdm
from collections import defaultdict
import argparse
import numpy as np
import random
from copy import deepcopy
import os

def get_scientific_name(row):
    _phylum, _class1, _order, _family, _subfamily, _genus, species = row[3], row[4], row[5], row[6], row[7], row[8], row[9]

    # name = '_'.join([x for x in [phylum, class1, order, family, subfamily, genus, species] if len(x)>0])
    name = '_'.join(species.split())

    return name

def is_species(row):
    return len(row[9])>0

def find_images(root_dir):
    image_map = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            os.path.splitext(filename)[1].lower()
            name_without_ext = os.path.splitext(filename)[0]
            full_path = os.path.join(dirpath, filename)
            image_map[name_without_ext] = full_path
    return image_map

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file', type=str, default='/fs/ess/user/bioscan-5m/bioscan5m/metadata/csv/BIOSCAN_5M_Insect_Dataset_metadata.csv')
    parser.add_argument('--seed', type=int, default=3875671)
    parser.add_argument('--image-dir', type=str, default='/fs/ess/user/bioscan-5m/bioscan5m/images/')
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--exclude-unseen', action='store_true', help='exclude unseen species')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    total = 0
    total_lines = 5150851
    species_dna_dict = defaultdict(list)

    # create a mapping from process id to filename
    process_id_to_imgfile_dict = find_images(args.image_dir)

    with open(args.csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile)

        for row_id, row in enumerate(tqdm(reader, total=total_lines)):
            # print(', '.join(row))
            # print(row[11])
            
            process_id = row[0] # uniue sample identifier

            if args.exclude_unseen:
                split = row[20]
                if split in ['train', 'val', 'test']:
                    if is_species(row):
                        species_dna_dict[get_scientific_name(row)].append(process_id)
            else:
                if is_species(row):
                    species_dna_dict[get_scientific_name(row)].append(process_id)

            total += 1

            # if row_id>10:
                # break
        
        print('total = {}'.format(total))

        print('len(species_dna_dict) = {}'.format(len(species_dna_dict)))

        all_species = list(species_dna_dict.keys())

        selected_species = all_species

        for species in selected_species:
            all_samples_species = deepcopy(species_dna_dict[species])

            # create a folder with species name
            species_dir = os.path.join(args.out_dir, species)

            os.makedirs(species_dir, exist_ok=True)

            # create a symlink with image filename

            for process_id in all_samples_species:
                train_file_src = process_id_to_imgfile_dict[process_id]
                train_file_dst = os.path.join(species_dir, '{}.jpg'.format(process_id))

                os.symlink(train_file_src, train_file_dst)
            