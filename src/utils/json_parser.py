
"""
json parser
"""

import json
import os
import pandas as pd

data_path = "/workspace/title_generation/raw_datasets/1_essays"

T1_path = os.path.join(data_path, "TL1")
T2_path = os.path.join(data_path, "TL2")

# json을 하나씩 가져와서, 제목과 글만 추출

titles = []
contents = []

# T1, T2 json 추출
for T_path in [T1_path, T2_path]:
    for json_file_name in os.listdir( T_path ):
        T_json_path = os.path.join(
            T_path, json_file_name
        )

        json_string = open(
            T_json_path, encoding='utf-8'
        )

        json_data = json.load(json_string)

        # 제목
        titles.append( json_data['rubric']['essay_main_subject'] )

        # 본문 내용
        contents.append( json_data['paragraph'][0]['paragraph_txt'] )

df = pd.DataFrame(
        {
            "titles": titles, 
            "contents": contents
        }
    )

tsv_path = os.path.join(data_path, "ai_hub_essay.tsv")

df.to_csv(tsv_path, sep='\t')
