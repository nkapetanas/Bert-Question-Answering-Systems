import pandas as pd

TRAIN_DATASET_PATH = "./Squad2Data/train-v2.0.json"
DEV_DATASET_PATH = "./Squad2Data/dev-v2.0.json"

PARAGRAPH_FIELD = "paragraphs"
CONTEXT_FIELD = "context"
QAS_FIELD = "qas"
QUESTION_FIELD = "question"
ANSWER_FIELD = "answers"
TEXT_FIELD = "text"
ANSWER_START_FIELD = "answer_start"


def json_to_df(dataset):
    contexts = list()
    questions = list()
    answers_text = list()
    answers_start = list()

    for i in range(dataset.shape[0]):
        topic = dataset.iloc[i, 1][PARAGRAPH_FIELD]
        for sub_para in topic:
            for q_a in sub_para[QAS_FIELD]:

                questions.append(q_a[QUESTION_FIELD])

                if len(q_a[ANSWER_FIELD]) != 0:
                    answers_start.append(q_a[ANSWER_FIELD][0][ANSWER_START_FIELD])
                    answers_text.append(q_a[ANSWER_FIELD][0][TEXT_FIELD])
                else:
                    answers_start.append("")
                    answers_text.append("")

                contexts.append(sub_para[CONTEXT_FIELD])

    return pd.DataFrame(
        {CONTEXT_FIELD: contexts, QUESTION_FIELD: questions, ANSWER_START_FIELD: answers_start,
         TEXT_FIELD: answers_text})


train = pd.read_json(TRAIN_DATASET_PATH)
dev = pd.read_json(DEV_DATASET_PATH)

train_df = json_to_df(train)
dev_df = json_to_df(dev)

train_df.to_csv("train.csv")
dev_df.to_csv("dev.csv")
