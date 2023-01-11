
"""
pdf에서 텍스트 추출을 위해 사용됨
"""

from tika import parser

"""
tika 사용이 안되는건 java의 문제, 7 이상의 java 버전이 있으면 되므로, 아래의 명령어 입력하면 됨
apt-get install openjdk-8-jdk
"""

import os

#https://juliea.tistory.com/44

# path는 변경하여 사용하기
data_path = "/workspace/title_generation/raw_datasets/1_essays"

# os.listdir(): 경로의 폴더 아래의 파일 이름 모두 가져오기
for file_name in os.listdir(data_path):
    
    # file_name에 pdf라는 글자가 있는지 확인하기
    if "pdf" in file_name:

        # pdf 파일의 경로
        pdf_path = os.path.join(data_path, file_name)

        # parser로 pdf에서 text 추출하기
        parsed = parser.from_file(pdf_path)
        content = parsed['content']
        content = content.strip()

        # 추출한 text 파일을 txt로 만들기
        pure_file_name = file_name.split('.')[0]

        # txt 파일 경로 + 이름
        txt_file_name = os.path.join(data_path, pure_file_name + ".txt")

        # txt 빈 파일 만들기(쓰기모드)
        txt = open(txt_file_name, 'w', encoding = 'utf-8')

        # txt 빈 파일에 content 작성하기
        txt.write(content)

        txt.close()


