# Generated content DO NOT EDIT
class PreTokenizer:
    """
    Base class for all pre-tokenizers

    This class is not supposed to be instantiated directly. Instead, any implementation of a
    PreTokenizer will return an instance of this class when instantiated.
    """

    def pre_tokenize(self, pretok):
        """
        Pre-tokenize a :class:`~tokenizers.PyPreTokenizedString` in-place

        This method allows to modify a :class:`~tokenizers.PreTokenizedString` to
        keep track of the pre-tokenization, and leverage the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you just want to see the result of
        the pre-tokenization of a raw string, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize_str`

        Args:
            pretok (:class:`~tokenizers.PreTokenizedString):
                The pre-tokenized string on which to apply this
                :class:`~tokenizers.pre_tokenizers.PreTokenizer`
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given string

        This method provides a way to visualize the effect of a
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` but it does not keep track of the
        alignment, nor does it provide all the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you need some of these, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize`

        Args:
            sequence (:obj:`str`):
                A string to pre-tokeize

        Returns:
            :obj:`List[Tuple[str, Offsets]]`:
                A list of tuple with the pre-tokenized parts and their offsets
        """
        pass

class BertPreTokenizer(PreTokenizer):
    """
    BertPreTokenizer

    This pre-tokenizer splits tokens on spaces, and also on punctuation.
    Each occurence of a punctuation character will be treated separately.
    """

    def __init__(self):
        pass
    def pre_tokenize(self, pretok):
        """
        Pre-tokenize a :class:`~tokenizers.PyPreTokenizedString` in-place

        This method allows to modify a :class:`~tokenizers.PreTokenizedString` to
        keep track of the pre-tokenization, and leverage the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you just want to see the result of
        the pre-tokenization of a raw string, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize_str`

        Args:
            pretok (:class:`~tokenizers.PreTokenizedString):
                The pre-tokenized string on which to apply this
                :class:`~tokenizers.pre_tokenizers.PreTokenizer`
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given string

        This method provides a way to visualize the effect of a
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` but it does not keep track of the
        alignment, nor does it provide all the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you need some of these, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize`

        Args:
            sequence (:obj:`str`):
                A string to pre-tokeize

        Returns:
            :obj:`List[Tuple[str, Offsets]]`:
                A list of tuple with the pre-tokenized parts and their offsets
        """
        pass

class ByteLevel(PreTokenizer):
    """
    ByteLevel PreTokenizer

    This pre-tokenizer takes care of replacing all bytes of the given string
    with a corresponding representation, as well as splitting into words.

    Args:
        add_prefix_space (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to add a space to the first word if there isn't already one. This
            lets us treat `hello` exactly like `say hello`.
        use_regex (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Set this to :obj:`False` to prevent this `pre_tokenizer` from using
            the GPT2 specific regexp for spliting on whitespace.
    """

    def __init__(self, add_prefix_space=True, use_regex=True):
        pass
    @staticmethod
    def alphabet():
        """
        Returns the alphabet used by this PreTokenizer.

        Since the ByteLevel works as its name suggests, at the byte level, it
        encodes each byte value to a unique visible character. This means that there is a
        total of 256 different characters composing this alphabet.

        Returns:
            :obj:`List[str]`: A list of characters that compose the alphabet
        """
        pass
    def pre_tokenize(self, pretok):
        """
        Pre-tokenize a :class:`~tokenizers.PyPreTokenizedString` in-place

        This method allows to modify a :class:`~tokenizers.PreTokenizedString` to
        keep track of the pre-tokenization, and leverage the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you just want to see the result of
        the pre-tokenization of a raw string, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize_str`

        Args:
            pretok (:class:`~tokenizers.PreTokenizedString):
                The pre-tokenized string on which to apply this
                :class:`~tokenizers.pre_tokenizers.PreTokenizer`
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given string

        This method provides a way to visualize the effect of a
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` but it does not keep track of the
        alignment, nor does it provide all the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you need some of these, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize`

        Args:
            sequence (:obj:`str`):
                A string to pre-tokeize

        Returns:
            :obj:`List[Tuple[str, Offsets]]`:
                A list of tuple with the pre-tokenized parts and their offsets
        """
        pass

class CharDelimiterSplit(PreTokenizer):
    """
    This pre-tokenizer simply splits on the provided char. Works like `.split(delimiter)`

    Args:
        delimiter: str:
            The delimiter char that will be used to split input
    """

    def pre_tokenize(self, pretok):
        """
        Pre-tokenize a :class:`~tokenizers.PyPreTokenizedString` in-place

        This method allows to modify a :class:`~tokenizers.PreTokenizedString` to
        keep track of the pre-tokenization, and leverage the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you just want to see the result of
        the pre-tokenization of a raw string, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize_str`

        Args:
            pretok (:class:`~tokenizers.PreTokenizedString):
                The pre-tokenized string on which to apply this
                :class:`~tokenizers.pre_tokenizers.PreTokenizer`
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given string

        This method provides a way to visualize the effect of a
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` but it does not keep track of the
        alignment, nor does it provide all the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you need some of these, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize`

        Args:
            sequence (:obj:`str`):
                A string to pre-tokeize

        Returns:
            :obj:`List[Tuple[str, Offsets]]`:
                A list of tuple with the pre-tokenized parts and their offsets
        """
        pass

class Digits(PreTokenizer):
    """
    This pre-tokenizer simply splits using the digits in separate tokens

    Args:
        individual_digits (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If set to True, digits will each be separated as follows::

                "Call 123 please" -> "Call ", "1", "2", "3", " please"

            If set to False, digits will grouped as follows::

                "Call 123 please" -> "Call ", "123", " please"
    """

    def __init__(self, individual_digits=False):
        pass
    def pre_tokenize(self, pretok):
        """
        Pre-tokenize a :class:`~tokenizers.PyPreTokenizedString` in-place

        This method allows to modify a :class:`~tokenizers.PreTokenizedString` to
        keep track of the pre-tokenization, and leverage the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you just want to see the result of
        the pre-tokenization of a raw string, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize_str`

        Args:
            pretok (:class:`~tokenizers.PreTokenizedString):
                The pre-tokenized string on which to apply this
                :class:`~tokenizers.pre_tokenizers.PreTokenizer`
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given string

        This method provides a way to visualize the effect of a
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` but it does not keep track of the
        alignment, nor does it provide all the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you need some of these, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize`

        Args:
            sequence (:obj:`str`):
                A string to pre-tokeize

        Returns:
            :obj:`List[Tuple[str, Offsets]]`:
                A list of tuple with the pre-tokenized parts and their offsets
        """
        pass

class Metaspace(PreTokenizer):
    """
    Metaspace pre-tokenizer

    This pre-tokenizer replaces any whitespace by the provided replacement character.
    It then tries to split on these spaces.

    Args:
        replacement (:obj:`str`, `optional`, defaults to :obj:`▁`):
            The replacement character. Must be exactly one character. By default we
            use the `▁` (U+2581) meta symbol (Same as in SentencePiece).

        add_prefix_space (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to add a space to the first word if there isn't already one. This
            lets us treat `hello` exactly like `say hello`.
    """

    def __init__(self, replacement="_", add_prefix_space=True):
        pass
    def pre_tokenize(self, pretok):
        """
        Pre-tokenize a :class:`~tokenizers.PyPreTokenizedString` in-place

        This method allows to modify a :class:`~tokenizers.PreTokenizedString` to
        keep track of the pre-tokenization, and leverage the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you just want to see the result of
        the pre-tokenization of a raw string, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize_str`

        Args:
            pretok (:class:`~tokenizers.PreTokenizedString):
                The pre-tokenized string on which to apply this
                :class:`~tokenizers.pre_tokenizers.PreTokenizer`
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given string

        This method provides a way to visualize the effect of a
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` but it does not keep track of the
        alignment, nor does it provide all the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you need some of these, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize`

        Args:
            sequence (:obj:`str`):
                A string to pre-tokeize

        Returns:
            :obj:`List[Tuple[str, Offsets]]`:
                A list of tuple with the pre-tokenized parts and their offsets
        """
        pass

class Punctuation(PreTokenizer):
    """
    This pre-tokenizer simply splits on punctuation as individual characters.

    Args:
        behavior (:class:`~tokenizers.SplitDelimiterBehavior`):
            The behavior to use when splitting.
            Choices: "removed", "isolated" (default), "merged_with_previous", "merged_with_next",
            "contiguous"
    """

    def __init__(self, behavior="isolated"):
        pass
    def pre_tokenize(self, pretok):
        """
        Pre-tokenize a :class:`~tokenizers.PyPreTokenizedString` in-place

        This method allows to modify a :class:`~tokenizers.PreTokenizedString` to
        keep track of the pre-tokenization, and leverage the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you just want to see the result of
        the pre-tokenization of a raw string, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize_str`

        Args:
            pretok (:class:`~tokenizers.PreTokenizedString):
                The pre-tokenized string on which to apply this
                :class:`~tokenizers.pre_tokenizers.PreTokenizer`
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given string

        This method provides a way to visualize the effect of a
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` but it does not keep track of the
        alignment, nor does it provide all the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you need some of these, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize`

        Args:
            sequence (:obj:`str`):
                A string to pre-tokeize

        Returns:
            :obj:`List[Tuple[str, Offsets]]`:
                A list of tuple with the pre-tokenized parts and their offsets
        """
        pass

class Sequence(PreTokenizer):
    """
    This pre-tokenizer composes other pre_tokenizers and applies them in sequence
    """

    def __init__(self, pretokenizers):
        pass
    def pre_tokenize(self, pretok):
        """
        Pre-tokenize a :class:`~tokenizers.PyPreTokenizedString` in-place

        This method allows to modify a :class:`~tokenizers.PreTokenizedString` to
        keep track of the pre-tokenization, and leverage the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you just want to see the result of
        the pre-tokenization of a raw string, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize_str`

        Args:
            pretok (:class:`~tokenizers.PreTokenizedString):
                The pre-tokenized string on which to apply this
                :class:`~tokenizers.pre_tokenizers.PreTokenizer`
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given string

        This method provides a way to visualize the effect of a
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` but it does not keep track of the
        alignment, nor does it provide all the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you need some of these, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize`

        Args:
            sequence (:obj:`str`):
                A string to pre-tokeize

        Returns:
            :obj:`List[Tuple[str, Offsets]]`:
                A list of tuple with the pre-tokenized parts and their offsets
        """
        pass

class Split(PreTokenizer):
    """
    Split PreTokenizer

    This versatile pre-tokenizer splits using the provided pattern and
    according to the provided behavior. The pattern can be inverted by
    making use of the invert flag.

    Args:
        pattern (:obj:`str` or :class:`~tokenizers.Regex`):
            A pattern used to split the string. Usually a string or a a regex built with `tokenizers.Regex`

        behavior (:class:`~tokenizers.SplitDelimiterBehavior`):
            The behavior to use when splitting.
            Choices: "removed", "isolated", "merged_with_previous", "merged_with_next",
            "contiguous"

        invert (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to invert the pattern.
    """

    def __init__(self, pattern, behavior, invert=False):
        pass
    def pre_tokenize(self, pretok):
        """
        Pre-tokenize a :class:`~tokenizers.PyPreTokenizedString` in-place

        This method allows to modify a :class:`~tokenizers.PreTokenizedString` to
        keep track of the pre-tokenization, and leverage the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you just want to see the result of
        the pre-tokenization of a raw string, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize_str`

        Args:
            pretok (:class:`~tokenizers.PreTokenizedString):
                The pre-tokenized string on which to apply this
                :class:`~tokenizers.pre_tokenizers.PreTokenizer`
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given string

        This method provides a way to visualize the effect of a
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` but it does not keep track of the
        alignment, nor does it provide all the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you need some of these, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize`

        Args:
            sequence (:obj:`str`):
                A string to pre-tokeize

        Returns:
            :obj:`List[Tuple[str, Offsets]]`:
                A list of tuple with the pre-tokenized parts and their offsets
        """
        pass

class UnicodeScripts(PreTokenizer):
    """
    This pre-tokenizer splits on characters that belong to different language family
    It roughly follows https://github.com/google/sentencepiece/blob/master/data/Scripts.txt
    Actually Hiragana and Katakana are fused with Han, and 0x30FC is Han too.
    This mimicks SentencePiece Unigram implementation.
    """

    def __init__(self):
        pass
    def pre_tokenize(self, pretok):
        """
        Pre-tokenize a :class:`~tokenizers.PyPreTokenizedString` in-place

        This method allows to modify a :class:`~tokenizers.PreTokenizedString` to
        keep track of the pre-tokenization, and leverage the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you just want to see the result of
        the pre-tokenization of a raw string, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize_str`

        Args:
            pretok (:class:`~tokenizers.PreTokenizedString):
                The pre-tokenized string on which to apply this
                :class:`~tokenizers.pre_tokenizers.PreTokenizer`
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given string

        This method provides a way to visualize the effect of a
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` but it does not keep track of the
        alignment, nor does it provide all the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you need some of these, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize`

        Args:
            sequence (:obj:`str`):
                A string to pre-tokeize

        Returns:
            :obj:`List[Tuple[str, Offsets]]`:
                A list of tuple with the pre-tokenized parts and their offsets
        """
        pass

class Whitespace(PreTokenizer):
    """
    This pre-tokenizer simply splits using the following regex: `\w+|[^\w\s]+`
    """

    def __init__(self):
        pass
    def pre_tokenize(self, pretok):
        """
        Pre-tokenize a :class:`~tokenizers.PyPreTokenizedString` in-place

        This method allows to modify a :class:`~tokenizers.PreTokenizedString` to
        keep track of the pre-tokenization, and leverage the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you just want to see the result of
        the pre-tokenization of a raw string, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize_str`

        Args:
            pretok (:class:`~tokenizers.PreTokenizedString):
                The pre-tokenized string on which to apply this
                :class:`~tokenizers.pre_tokenizers.PreTokenizer`
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given string

        This method provides a way to visualize the effect of a
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` but it does not keep track of the
        alignment, nor does it provide all the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you need some of these, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize`

        Args:
            sequence (:obj:`str`):
                A string to pre-tokeize

        Returns:
            :obj:`List[Tuple[str, Offsets]]`:
                A list of tuple with the pre-tokenized parts and their offsets
        """
        pass

class WhitespaceSplit(PreTokenizer):
    """
    This pre-tokenizer simply splits on the whitespace. Works like `.split()`
    """

    def __init__(self):
        pass
    def pre_tokenize(self, pretok):
        """
        Pre-tokenize a :class:`~tokenizers.PyPreTokenizedString` in-place

        This method allows to modify a :class:`~tokenizers.PreTokenizedString` to
        keep track of the pre-tokenization, and leverage the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you just want to see the result of
        the pre-tokenization of a raw string, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize_str`

        Args:
            pretok (:class:`~tokenizers.PreTokenizedString):
                The pre-tokenized string on which to apply this
                :class:`~tokenizers.pre_tokenizers.PreTokenizer`
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given string

        This method provides a way to visualize the effect of a
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` but it does not keep track of the
        alignment, nor does it provide all the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you need some of these, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize`

        Args:
            sequence (:obj:`str`):
                A string to pre-tokeize

        Returns:
            :obj:`List[Tuple[str, Offsets]]`:
                A list of tuple with the pre-tokenized parts and their offsets
        """
        pass
