import json
from collections import namedtuple
import pandas as pd
import os


class Dialog:
    def __init__(self, dialog_id, context, scores=None, user_is_bot=None):
        self.dialog_id = int(dialog_id)
        self.context = context
        self.messages = []
        self.scores = scores
        self.user_is_bot = user_is_bot

    def add_message(self, user_id, text):
        Message = namedtuple("Message", "user_id text")
        self.messages.append(Message(user_id, text))

    def get_user_messages(self, user_id):
        return [message.text for message in self.messages if message.user_id == user_id]

    def get_message_mask(self, user_id):
        return [int(message.user_id == user_id) for message in self.messages]

    def __str__(self):
        return str(self.dialog_id)

    def __repr__(self):
        return self.__str__()


def parse(filenames, get_df=True):
    raw_dialogs = []
    for filename in filenames:
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()
            text = '{"dialogs": ' + text + '}'
            raw_dialogs += json.loads(text)["dialogs"]
    dialogs = []
    for dialog in raw_dialogs:
        messages = dialog["thread"]
        user_is_bot = None
        scores = None
        if "evaluation" in dialog:
            scores = {user["userId"]: float(user["quality"]) for user in dialog["evaluation"]}
        if "userType" in dialog["users"][0]:
            user_is_bot = {user["id"]: user["userType"] == "Bot" for user in dialog["users"]}
        dialog = Dialog(dialog_id=dialog["dialogId"], context=dialog["context"],
                        scores=scores, user_is_bot=user_is_bot)
        for message in messages:
            dialog.add_message(message["userId"], message["text"])
        dialogs.append(dialog)
    df = pd.DataFrame()
    df["dialogId"] = [dialog.dialog_id for dialog in dialogs]
    df["context"] = [dialog.context for dialog in dialogs]
    df["messages"] = [[message.text for message in dialog.messages] for dialog in dialogs]
    df["messageUsers"] = [[message.user_id for message in dialog.messages] for dialog in dialogs]
    df["AliceMessages"] = [dialog.get_user_messages("Alice") for dialog in dialogs]
    df["BobMessages"] = [dialog.get_user_messages("Bob") for dialog in dialogs]
    df["AliceMessageMask"] = [dialog.get_message_mask("Alice") for dialog in dialogs]
    df["BobMessageMask"] = [dialog.get_message_mask("Bob") for dialog in dialogs]
    if dialogs[0].scores is not None:
        for userId in dialogs[0].scores.keys():
            df[userId+"Score"] = [dialog.scores[userId] for dialog in dialogs]
    if dialogs[0].user_is_bot is not None:
        for userId in dialogs[0].user_is_bot.keys():
            df[userId+"IsBot"] = [dialog.user_is_bot[userId] for dialog in dialogs]
    if get_df:
        return df
    return dialogs


def parse_dir(dir_name="data", df=True):
    return parse([os.path.join(dir_name, file_name) for file_name in os.listdir(dir_name)], get_df=df)
