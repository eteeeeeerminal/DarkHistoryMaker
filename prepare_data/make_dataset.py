import json
from pathlib import Path
import re
import chardet

def sum_text_data(  input_dir, ext="**/*.txt",
                    split_pattern=r"[\n。]+", del_pattern=r"[\t\n 　「」『』]"):
    p = Path(input_dir)
    file_list = p.glob(ext)

    split_pattern = re.compile(split_pattern)
    del_pattern   = re.compile(del_pattern)

    paragraphs = []
    for file_path in file_list:
        with open(file_path, 'rb') as f:
            encode = chardet.detect(f.read())["encoding"]

        with open(file_path, 'r', encoding=encode, errors='ignore') as f:
            text = split_pattern.split(f.read())
            for prgrph in text:
                prgrph = del_pattern.sub("", prgrph)
                # if len(prgrph) >= 5:
                paragraphs.append(prgrph)

    return paragraphs

if __name__ == "__main__":
    config = json.load(open("../.json", "r", encoding="utf-8"))
    data_dir   = config["data_dir"]
    output_path = config["dataset_path"]

    text  = sum_text_data(data_dir)

    with open(output_path, "w", encoding='utf-8', errors='ignore') as f:
        f.write('\n\n'.join(text))