import json
from collections import namedtuple
import pandas as pd
import os
import numpy as np


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


def parse(filenames, df=True):
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
    data = pd.DataFrame()
    data["dialogId"] = [dialog.dialog_id for dialog in dialogs]
    data["context"] = [dialog.context for dialog in dialogs]
    data["messages"] = [[message.text for message in dialog.messages] for dialog in dialogs]
    data["messageUsers"] = [[message.user_id for message in dialog.messages] for dialog in dialogs]
    data["AliceMessages"] = [dialog.get_user_messages("Alice") for dialog in dialogs]
    data["BobMessages"] = [dialog.get_user_messages("Bob") for dialog in dialogs]
    data["AliceMessageMask"] = [dialog.get_message_mask("Alice") for dialog in dialogs]
    data["BobMessageMask"] = [dialog.get_message_mask("Bob") for dialog in dialogs]
    data["AliceScore"] = [dialog.scores["Alice"] if dialog.scores is not None else np.NaN for dialog in dialogs]
    data["BobScore"] = [dialog.scores["Bob"] if dialog.scores is not None else np.NaN for dialog in dialogs]
    data["AliceIsBot"] = [int(dialog.user_is_bot["Alice"]) if dialog.user_is_bot is not None else np.NaN for dialog in dialogs]
    data["BobIsBot"] = [int(dialog.user_is_bot["Bob"]) if dialog.user_is_bot is not None else np.NaN for dialog in dialogs]
    if df:
        return data
    return dialogs


def parse_dir(dir_name="data", df=True):
    return parse([os.path.join(dir_name, file_name) for file_name in os.listdir(dir_name)], df=df)
