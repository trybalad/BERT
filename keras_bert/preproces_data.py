import codecs

import spacy


def preprocess_file(to_read_file_path, to_save_file_path, lines_to_process=None, start_index=0):
    document = codecs.open(to_read_file_path, 'r', 'utf-8')
    data_file = codecs.open(to_save_file_path, 'w', 'utf-8')
    nlp = spacy.load('pl_core_news_md')

    line = document.readline()
    for i in range(start_index):
        line = document.readline()

    saved_lines = 0
    print("Starting on line:", start_index)

    print(line)
    print(lines_to_process)
    print(saved_lines)
    while line and (lines_to_process is None or saved_lines < lines_to_process):
        line = line.strip()
        if line != '' and not line.startswith("<") and not line.strip().endswith(">"):
            doc = nlp(line.strip())
            count = 0
            for token in doc:
                count += 1
            sentence_in_line = 0
            new_line = ""
            for sentence in doc.sents:
                if sentence_in_line == 1:
                    new_line += " "
                new_line += sentence.text
                sentence_in_line += 1
                if sentence_in_line == 2:
                    data_file.write(new_line + "\n")
                    new_line = ""
                    sentence_in_line = 0
            if sentence_in_line != 0:
                data_file.write(new_line + "\n")
        line = document.readline()
        saved_lines += 1
        if saved_lines % 100 == 0:
            print("Preprocessed line:", saved_lines, "/", lines_to_process)
