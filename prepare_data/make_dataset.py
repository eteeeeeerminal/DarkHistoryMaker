import json
from pathlib import Path
import re
import chardet

def sum_text_data(input_dir, ext="**/*.txt", split_pattern=r"[\n]{2, }"):
    p = Path(input_dir)
    file_list = p.glob(ext)

    split_pattern = re.compile(split_pattern)

    paragraphs = []
    for file_path in file_list:
        with open(file_path, 'rb') as f:
            encode = chardet.detect(f.read())["encoding"]

        with open(file_path, 'r', encoding=encode, errors='ignore') as f:
            text = split_pattern.split(f.read())
            for prgrph in text:
                if len(prgrph) >= 3:
                    paragraphs.append(prgrph)

    return paragraphs

if __name__ == "__main__":
    config = json.load(open("../.json", "r"))
    data_dir   = config["data_dir"]
    output_dir = config["output_dir"]

    text  = sum_text_data(data_dir)

    with open(output_dir, "w", encoding='utf-8', errors='ignore') as f:
        f.write('\n\n'.join(text))