# import re
# from html import unescape

# class AbstractProcessor:
#     def __init__(self):
#         self.default_patterns = [
#             r'\$\$',  # Specific special characters
#             r'\^',  # Specific special characters
#             r'\$[^$]*?\$',  # Inline math expressions enclosed in $
#             r'\\[a-zA-Z]+\{.*?\}',  # LaTeX commands
#             r'\w+_{[^}]+}',  # Subscripts
#             r"(?<!\w)'(?!\w)",  # Primes that do not follow or precede word characters
#             r'\w+_{[^}]+}\'?',  # Subscripts followed by an optional prime
#             r'\w+^{-?\d+}',  # Superscripts
#         ]

#     def clean_text(self, text):
#         # Unescape HTML entities and normalize text first
#         text = unescape(text.replace('\n', ' '))

#         # Specific substitution to handle escaped apostrophes and middle initials
#         text = re.sub(r"\\'", '', text)  # Remove escaped apostrophes
#         text = re.sub(r'\.-', '.', text)  # Correct middle initial formatting if necessary

#         # Replace math expressions and LaTeX commands
#         for pattern in self.default_patterns:
#             text = re.sub(pattern, ' [MATH_EXPR] ', text)

#         # Normalize whitespace after replacements by replacing newlines with
#         # spaces and removing excessive whitespace.
#         text = re.sub(r'\s+', ' ', text).strip()
#         return text

#     def replace_math_expressions(self, text):
#         replacement = ' [MATH_EXPR] '
#         for pattern in self.default_patterns:
#             text = re.sub(pattern, replacement, text)
#         text = re.sub(r'(\s*\[MATH_EXPR\]\s*)+', ' [MATH_EXPR] ', text)
#         return text

#     def find_special_signs(self, col, patterns=None):
#         if patterns is None:
#             patterns = self.default_patterns
#         special_signs = set()
#         if isinstance(col, pd.Series):  # Check if the input is a Series
#             for text in col:
#                 for pattern in patterns:
#                     matches = re.findall(pattern, text)
#                     special_signs.update(matches)
#         else:
#             for pattern in patterns:
#                 matches = re.findall(pattern, col)  # This assumes 'col' is a single string
#                 special_signs.update(matches)
#         return special_signs

#     def process(self, series, func=None):
#         if func:
#             return series.apply(func)
#         else:
#             return series.apply(self.clean_text)

#     def analyze_columns(self, dataframe, columns):
#         special_signs_by_column = {}
#         for column in columns:
#             results = self.process(dataframe[column], lambda x: self.find_special_signs(x))
#             column_signs = set()
#             for result in results:
#                 column_signs.update(result)
#             special_signs_by_column[column] = column_signs
#         return special_signs_by_column

#     def find_and_clean_special_signs(self, text):
#         special_signs = self.find_special_signs(text)
#         cleaned_text = self.clean_text(text)
#         return cleaned_text

# class TweetProcessor(AbstractProcessor):
#     '''Adds methods tailored to the unique characteristics of tweets, such as removing @user mentions,
#     hashtags, and URLs.'''
#     def __init__(self):
#         super().__init__()
#         self.tweet_patterns = [
#             r'@\w+',  # Remove @user mentions
#             r'http\S+|www\S+',  # Remove URLs
#         ]

#     def clean_tweet(self, text):
#         text = unescape(text.replace('\n', ' '))
#         for pattern in self.tweet_patterns:
#             text = re.sub(pattern, '', text)

#         text = re.sub(r'\$', '', text)  # Remove all dollar signs
#         text = re.sub(r'\^', '', text)  # Remove all caret symbols
#         text = re.sub(r'\'', '', text)  # Remove all single quotes
#         text = re.sub(r'\[([^\]]+)\]', r'\1', text)  # Remove square brackets but keep content
#         text = re.sub(r'\{([^}]+)\}', r'\1', text)  # Remove curly braces but keep content
#         text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters (e.g., emojis)
#         text = re.sub(r'\s+', ' ', text).strip()
#         return text

#     def clean_special_characters(self, text):
#         '''Focuses on removing special characters while preserving surrounding text.'''
#         text = unescape(text.replace('\n', ' '))
#         text = re.sub(r'\$', '', text)  # Remove all dollar signs
#         text = re.sub(r'\^', '', text)  # Remove all caret symbols
#         text = re.sub(r'\'', '', text)  # Remove all single quotes
#         text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
#         text = re.sub(r'@\w+', '', text)  # Remove @user mentions
#         text = re.sub(r'#(\w+)', r'\1', text)  # Remove hashtags but keep content
#         text = re.sub(r'[~:.,!]+', '', text)
#         text = re.sub(r'\s+', ' ', text).strip()
#         return text

#     def process(self, df, columns, func=None, output_format='series'):
#         cleaned_df = df.copy()

#         for column in columns:
#             if func:
#                 cleaned_series = df[column].apply(func)
#             else:
#                 cleaned_series = df[column].apply(self.clean_special_characters)

#             cleaned_df[column] = cleaned_series

#         if output_format == 'txt':
#             return '\n'.join(cleaned_df[columns].apply(lambda x: ' '.join(x), axis=1).tolist())
#         elif output_format == 'dataframe':
#             return cleaned_df
#         else:
#             return cleaned_df[columns]

import re
from html import unescape
import spacy
import pandas as pd

class AbstractProcessor:
    def __init__(self):
        self.default_patterns = [
            r'\$\$',  # Specific special characters
            r'\^',  # Specific special characters
            r'\$[^$]*?\$',  # Inline math expressions enclosed in $
            r'\\[a-zA-Z]+\{.*?\}',  # LaTeX commands
            r'\w+_{[^}]+}',  # Subscripts
            r"(?<!\w)'(?!\w)",  # Primes that do not follow or precede word characters
            r'\w+_{[^}]+}\'?',  # Subscripts followed by an optional prime
            r'\w+^{-?\d+}',  # Superscripts
        ]
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def clean_text(self, text):
        # Unescape HTML entities and normalize text first
        text = unescape(text.replace('\n', ' '))

        # Specific substitution to handle escaped apostrophes and middle initials
        text = re.sub(r"\\'", '', text)  # Remove escaped apostrophes
        text = re.sub(r'\.-', '.', text)  # Correct middle initial formatting if necessary

        # Replace math expressions and LaTeX commands
        for pattern in self.default_patterns:
            text = re.sub(pattern, ' [MATH_EXPR] ', text)

        # Normalize whitespace after replacements by replacing newlines with
        # spaces and removing excessive whitespace.
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def replace_math_expressions(self, text):
        replacement = ' [MATH_EXPR] '
        for pattern in self.default_patterns:
            text = re.sub(pattern, replacement, text)
        text = re.sub(r'(\s*\[MATH_EXPR\]\s*)+', ' [MATH_EXPR] ', text)
        return text

    def find_special_signs(self, col, patterns=None):
        if patterns is None:
            patterns = self.default_patterns
        special_signs = set()
        if isinstance(col, pd.Series):  # Check if the input is a Series
            for text in col:
                for pattern in patterns:
                    matches = re.findall(pattern, text)
                    special_signs.update(matches)
        else:
            for pattern in patterns:
                matches = re.findall(pattern, col)  # This assumes 'col' is a single string
                special_signs.update(matches)
        return special_signs

    def process(self, series, func=None):
        series = series.apply(lambda x: str(x) if x is not None else "")
        if func:
            return series.apply(func)
        else:
            return series.apply(self.clean_text)

    def analyze_columns(self, dataframe, columns):
        special_signs_by_column = {}
        for column in columns:
            results = self.process(dataframe[column], lambda x: self.find_special_signs(x))
            column_signs = set()
            for result in results:
                column_signs.update(result)
            special_signs_by_column[column] = column_signs
        return special_signs_by_column

    def find_and_clean_special_signs(self, text):
        special_signs = self.find_special_signs(text)
        cleaned_text = self.clean_text(text)
        return cleaned_text

class TweetProcessor(AbstractProcessor):
    '''Adds methods tailored to the unique characteristics of tweets, such as removing @user mentions,
    hashtags, and URLs.'''
    def __init__(self):
        super().__init__()
        self.tweet_patterns = [
            r'@\w+',  # Remove @user mentions
            r'http\S+|www\S+',  # Remove URLs
        ]

    def clean_tweet(self, text):
        text = unescape(text.replace('\n', ' '))
        for pattern in self.tweet_patterns:
            text = re.sub(pattern, '', text)

        text = re.sub(r'\$', '', text)  # Remove all dollar signs
        text = re.sub(r'\^', '', text)  # Remove all caret symbols
        text = re.sub(r'\'', '', text)  # Remove all single quotes
        text = re.sub(r'\[([^\]]+)\]', r'\1', text)  # Remove square brackets but keep content
        text = re.sub(r'\{([^}]+)\}', r'\1', text)  # Remove curly braces but keep content
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters (e.g., emojis)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()  # Convert text to lowercase
        return text

    def clean_special_characters(self, text):
        '''Focuses on removing special characters while preserving surrounding text.'''
        text = unescape(text.replace('\n', ' '))
        text = re.sub(r'\$', '', text)  # Remove all dollar signs
        text = re.sub(r'\^', '', text)  # Remove all caret symbols
        text = re.sub(r'\'', '', text)  # Remove all single quotes
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
        text = re.sub(r'@\w+', '', text)  # Remove @user mentions
        text = re.sub(r'#(\w+)', r'\1', text)  # Remove hashtags but keep content
        text = re.sub(r'[~:.,!]+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()  # Convert text to lowercase
        return text

    def process(self, df, columns, func=None, output_format='series'):
        cleaned_df = df.copy()

        for column in columns:
            cleaned_series = df[column].apply(lambda x: str(x) if x is not None else "")
            if func:
                cleaned_series = cleaned_series.apply(func)
            else:
                cleaned_series = cleaned_series.apply(self.clean_special_characters)

            cleaned_df[column] = cleaned_series

        if output_format == 'txt':
            return '\n'.join(cleaned_df[columns].apply(lambda x: ' '.join(x), axis=1).tolist())
        elif output_format == 'dataframe':
            return cleaned_df
        else:
            return cleaned_df[columns]

