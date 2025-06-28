import re


class CharsetAdapter(object):

    def __init__(self, charset: str):
        self.lowercase_only = charset == charset.lower()
        self.uppercase_only = charset == charset.upper()
        self.unsupported = re.compile(f"[^{re.escape(charset)}]")

    def convert(self, label: str):
        if self.lowercase_only:
            label = label.lower()
        elif self.uppercase_only:
            label = label.upper()
        label = self.unsupported.sub("", label)
        return label
