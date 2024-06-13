import json
from tqdm import tqdm
import random
import re
import argparse
from langdetect import detect
import sling

def is_english(sentence):
    """
    Check if a sentence contains English characters.
    :param sentence: str, the input sentence
    :return: bool, True if the sentence contains English characters, otherwise False
    """
    return bool(re.search(r'[A-Za-z]', sentence))

def main(args):
    # Load QID to name mapping from file
    id_to_name = {}
    with open(args.qid_names_file, "r") as file:
        lines = file.readlines()
        for line in tqdm(lines):
            items = line.strip().split()
            if 'Q' not in items[0]:
                continue
            id_to_name[items[0]] = ' '.join(items[1:])
    
    # Load the SLING knowledge base
    kb = sling.Store()
    kb.load(args.kb_file)
    kb.freeze()
    
    # Extract descriptions for entities
    id_to_description = {}
    for n, f in enumerate(kb):
        try:
            if f.id not in id_to_description and f.id is not None and f.description is not None:
                id_to_description[f.id] = f.description
        except Exception:
            continue

    # Define months for temporal conflict checking
    month = {'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'}

    # Read fact triples from 'kb.cfacts'
    with open(args.fact_triples_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Collect conflict-related data
    conflict_row_data = []
    for line in tqdm(lines):
        items = line.split()
        if items[0][0] != 'P' or items[1] not in id_to_name or items[2] not in id_to_name or items[1] not in id_to_description or items[2] not in id_to_description:
            continue
        if id_to_name[items[1]].lower() in id_to_name[items[2]].lower() or id_to_name[items[2]].lower() in id_to_name[items[1]].lower():
            continue
        flag = False
        for i in month:
            if i.lower() in id_to_name[items[1]].lower() or i.lower() in id_to_name[items[2]].lower():
                flag = True
                break
        if not flag:
            conflict_row_data.append(line)

    # Write conflict row data to JSON file
    with open(args.conflict_row_output_file, 'w', encoding='utf-8') as f:
        for line in tqdm(conflict_row_data):
            items = line.split()
            data = {
                'subject_id': items[1],
                'object_id': items[2],
                'relation': items[0]
            }
            json_data = json.dumps(data)
            f.write(json_data + '\n')

    # Load all conflict data and count relations
    all_data = []
    relation_cnt = {}
    with open(args.conflict_row_output_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            all_data.append(data)
            if data['relation'] not in relation_cnt:
                relation_cnt[data['relation']] = 0
            relation_cnt[data['relation']] += 1

    # Load templates and time categories
    templates = {}
    time_category = {}
    with open(args.templates_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            templates[data['relation']] = data['template']
            time_category[data['relation']] = data['category']

    # Select top 100 relations with templates
    relation_cnt = sorted(relation_cnt.items(), key=lambda x: -x[1])
    selected_relation = []
    for item in relation_cnt:
        if item[0] in templates:
            selected_relation.append(item[0])
        if len(selected_relation) == 100:
            break

    # Collect objects for each selected relation
    relation_to_object = {}
    s_r_object = {}
    for line in lines:
        items = line.split()
        if items[0] in selected_relation:
            if items[2] not in id_to_name:
                continue
            flag = False
            for i in month:
                if i.lower() in id_to_name[items[2]].lower():
                    flag = True
                    break
            if flag:
                continue
            if items[0] not in relation_to_object:
                relation_to_object[items[0]] = []
            relation_to_object[items[0]].append(items[2])
            if (items[0], items[1]) not in s_r_object:
                s_r_object[(items[0], items[1])] = []
            s_r_object[(items[0], items[1])].append(items[2])

    # Write relation to object mappings to JSON files
    with open(args.relation_to_object_output_file, 'w', encoding='utf-8') as f:
        for item in relation_to_object.items():
            data = {
                'key': item[0],
                'value': item[1]
            }
            json_data = json.dumps(data)
            f.write(json_data + '\n')

    with open(args.s_r_object_output_file, 'w', encoding='utf-8') as f:
        for item in s_r_object.items():
            data = {
                'key': item[0],
                'value': item[1]
            }
            json_data = json.dumps(data)
            f.write(json_data + '\n')

    # Define month dictionary for temporal conflict generation
    months_dict = {
        1: 'January',
        2: 'February',
        3: 'March',
        4: 'April',
        5: 'May',
        6: 'June',
        7: 'July',
        8: 'August',
        9: 'September',
        10: 'October',
        11: 'November',
        12: 'December'
    }

    def temporal_conflict(mode):
        """
        Generate a temporal conflict date.
        :param mode: str, the mode of conflict ('other' or 'point time')
        :return: tuple, the conflict type and date(s)
        """
        if mode == 'other' and random.randint(0, 2) != 0:
            s_year = random.randint(2024, 2035)
            e_year = random.randint(s_year + 1, 2045)
            if random.randint(0, 1) == 0:
                s_month = random.randint(1, 10)
                e_month = random.randint(s_month + 1, 12)
                if random.randint(0, 1) == 0:
                    s_day = random.randint(1, 28)
                    e_day = random.randint(1, 28)
                    s_date = '{0} {1}, {2}'.format(s_day, months_dict[s_month], s_year)
                    e_date = '{0} {1}, {2}'.format(e_day, months_dict[e_month], e_year)
                    return 'interval', s_date, e_date
                else:
                    s_date = '{0}, {1}'.format(months_dict[s_month], s_year)
                    e_date = '{0}, {1}'.format(months_dict[e_month], e_year)
                    return 'interval', s_date, e_date
            else:
                return 'interval', str(s_year), str(e_year)
        else:
            year = random.randint(2024, 2045)
            if random.randint(0, 1) == 0:
                month = random.randint(1, 12)
                if random.randint(0, 1) == 0:
                    day = random.randint(1, 28)
                    date = '{0} {1}, {2}'.format(day, months_dict[month], year)
                    return 'point', date, None
                else:
                    date = '{0}, {1}'.format(months_dict[month], year)
                    return 'point', date, None
            else:
                return 'point', str(year), None

    # Generate conflict data and write to JSON file
    with open(args.fact_conflict_output_file, 'w', encoding='utf-8') as f:
        for data in tqdm(all_data):
            try:
                if data['relation'] not in selected_relation:
                    continue
                template = templates[data['relation']]
                correct_pair = template.replace('<subject>', id_to_name[data['subject_id']]).replace('<object>', id_to_name[data['object_id']])
                if len(relation_to_object[data['relation']]) < 3:
                    continue
                replaced_object = random.sample(relation_to_object[data['relation']], 1)[0]
                cnt = 5000
                while (replaced_object == data['object_id'] or replaced_object not in id_to_name or replaced_object not in id_to_description or replaced_object in s_r_object[(data['relation'], data['subject_id'])]) and cnt > 0:
                    replaced_object = random.sample(relation_to_object[data['relation']], 1)[0]
                    cnt -= 1
                if cnt == 0:
                    continue
                fact_conflict_pair = template.replace('<subject>', id_to_name[data['subject_id']]).replace('<object>', id_to_name[replaced_object])
                if time_category[data['relation']] == 'other':
                    mode, st, et = temporal_conflict('other')
                    if mode == 'interval':
                        template = template.replace('.', ' from <st> to <et>.')
                        temporal_conflict_pair = template.replace('<subject>', id_to_name[data['subject_id']]).replace('<object>', id_to_name[replaced_object]).replace('<st>', st).replace('<et>', et)
                        conflict_time_span = [st, et]
                    elif mode == 'point':
                        template = template.replace('.', ' in <st>.')
                        temporal_conflict_pair = template.replace('<subject>', id_to_name[data['subject_id']]).replace('<object>', id_to_name[replaced_object]).replace('<st>', st)
                        conflict_time_span = [st]
                else:
                    mode, st, et = temporal_conflict('point time')
                    template = template.replace('.', ' in <st>.')
                    temporal_conflict_pair = template.replace('<subject>', id_to_name[data['subject_id']]).replace('<object>', id_to_name[replaced_object]).replace('<st>', st)
                    conflict_time_span = [st]
                data = {
                    'relation': data['relation'],
                    'subject': id_to_name[data['subject_id']],
                    'object': id_to_name[data['object_id']],
                    'replaced_object': id_to_name[replaced_object],
                    'correct_pair': correct_pair,
                    'fact_conflict_pair': fact_conflict_pair,
                    'temporal_conflict_pair': temporal_conflict_pair,
                    'conflict_time_span': conflict_time_span,
                    'subject_additional_information': id_to_description[data['subject_id']],
                    'object_additional_information': id_to_description[data['object_id']],
                    'replaced_object_additional_information': id_to_description[replaced_object]
                }
                json_data = json.dumps(data)
                f.write(json_data + '\n')

            except Exception:
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate conflict data for knowledge conflicts.")
    parser.add_argument("--qid_names_file", type=str, required=True, help="Path to the file containing QID names.")
    parser.add_argument("--kb_file", type=str, required=True, help="Path to the SLING knowledge base file.")
    parser.add_argument("--fact_triples_file", type=str, required=True, help="Path to the file containing fact triples.")
    parser.add_argument("--templates_file", type=str, required=True, help="Path to the file containing relation templates.")
    parser.add_argument("--conflict_row_output_file", type=str, required=True, help="Path to the output file for conflict rows.")
    parser.add_argument("--relation_to_object_output_file", type=str, required=True, help="Path to the output file for relation to object mappings.")
    parser.add_argument("--s_r_object_output_file", type=str, required=True, help="Path to the output file for subject-relation-object mappings.")
    parser.add_argument("--fact_conflict_output_file", type=str, required=True, help="Path to the output file for fact conflicts.")
    args = parser.parse_args()
    
    main(args)
