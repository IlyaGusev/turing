import json
from collections import namedtuple
import pandas as pd


class Dialog:
    def __init__(self, dialog_id, context, first_user_id, second_user_id,
                 first_user_is_bot=None, second_user_is_bot=None):
        self.dialog_id = int(dialog_id)
        self.first_user_id = first_user_id
        self.second_user_id = second_user_id
        self.context = context
        self.messages = []
        self.first_user_is_bot = first_user_is_bot
        self.second_user_is_bot = second_user_is_bot

    def add_message(self, user_id, text):
        Message = namedtuple("Message", "user_id text")
        self.messages.append(Message(user_id, text))

    def get_first_user_messages(self):
        return [message.text for message in self.messages if message.user_id == self.first_user_id]

    def get_second_user_messages(self):
        return [message.text for message in self.messages if message.user_id == self.second_user_id]

    def get_messages(self):
        return self.messages

    def get_context(self):
        return self.context

    def __str__(self):
        return str(self.dialog_id) + " " + self.first_user_id + " " + self.second_user_id

    def __repr__(self):
        return self.__str__()


def parse(filename, get_df=True):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
        dialogs = []
        if text[0] == "[":
            text = '{"dialogs": ' + text + '}'
            dialogs = json.loads(text)["dialogs"]
        else:
            dialogs.append(json.loads(text))

        result = []
        for dialog in dialogs:
            users = dialog["users"]
            messages = dialog["thread"]
            first_user = users[0]
            second_user = users[1]
            first_user_is_bot = None
            second_user_is_bot = None
            if "userType" in first_user:
                first_user_is_bot = first_user["userType"] != "Human"
                second_user_is_bot = second_user["userType"] != "Human"

            dialog = Dialog(dialog_id=dialog["dialogId"], context=dialog["context"],
                            first_user_id=first_user["id"], second_user_id=second_user["id"],
                            first_user_is_bot=first_user_is_bot, second_user_is_bot=second_user_is_bot)
            for message in messages:
                dialog.add_message(message["userId"], message["text"])
            result.append(dialog)
        if get_df:
            df = pd.DataFrame()
            df["dialogId"] = [dialog.dialog_id for dialog in result]
            df["context"] = [dialog.context for dialog in result]
            df["messages"] = [[message.text for message in dialog.messages] for dialog in result]
            df["message_users"] = [[message.user_id for message in dialog.messages] for dialog in result]
            print(df)
            return df
        return result