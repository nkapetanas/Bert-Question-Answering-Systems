import json

import pandas as pd

TRAIN_DATASET_PATH = "./Squad2Data/train-v2.0.json"
DEV_DATASET_PATH = "./Squad2Data/dev-v2.0.json"

DATA_FIELD = "data"
TITLE_FIELD = "title"
PARAGRAPH_FIELD = "paragraphs"
CONTEXT_FIELD = "context"
QAS_FIELD = "qas"
QUESTION_FIELD = "question"
ANSWER_FIELD = "answers"
TEXT_FIELD = "text"
ID_FIELD = "id"
ANSWER_START_FIELD = "answer_start"


def squad_json_to_dataframe(file):
    data = json.loads(open(file, "r").read())
    id_list = list()
    titles_list = list()
    context_list = list()
    questions_list = list()
    answers_list = list()
    text_list = list()

    articles = data[DATA_FIELD]

    for article in articles:

        title = article[TITLE_FIELD]

        paragraphs = article[PARAGRAPH_FIELD]

        for paragraph in paragraphs:

            context = paragraph[CONTEXT_FIELD]

            qas_list = paragraph[QAS_FIELD]

            for qas in qas_list:

                question = qas[QUESTION_FIELD]

                id = qas[ID_FIELD]

                answers = qas[ANSWER_FIELD]

                for answer in answers:
                    ans_start = answer[ANSWER_START_FIELD]

                    text = answer[TEXT_FIELD]

                    titles_list.append(title)
                    context_list.append(context)
                    questions_list.append(question)
                    id_list.append(id)
                    answers_list.append(ans_start)
                    text_list.append(text)

    ids_df = pd.DataFrame(id_list, columns=[ID_FIELD])
    titles_df = pd.DataFrame(titles_list, columns=[TITLE_FIELD])
    contexts_df = pd.DataFrame(context_list, columns=[CONTEXT_FIELD])
    questions_df = pd.DataFrame(questions_list, columns=[QUESTION_FIELD])
    answers_start_df = pd.DataFrame(answers_list, columns=[ANSWER_START_FIELD])
    text_df = pd.DataFrame(text_list, columns=[TEXT_FIELD])

    new_df = pd.concat(
        [ids_df.reset_index(drop=True), titles_df.reset_index(drop=True), contexts_df.reset_index(drop=True),
         questions_df.reset_index(drop=True), answers_start_df.reset_index(drop=True), text_df.reset_index(drop=True)],
        axis=1)

    return new_df.drop_duplicates(keep='first')


dataframe = squad_json_to_dataframe(DEV_DATASET_PATH)
