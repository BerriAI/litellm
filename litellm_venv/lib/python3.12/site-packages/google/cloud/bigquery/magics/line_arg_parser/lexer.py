# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import namedtuple
from collections import OrderedDict
import itertools
import re

import enum


Token = namedtuple("Token", ("type_", "lexeme", "pos"))
StateTransition = namedtuple("StateTransition", ("new_state", "total_offset"))

# Pattern matching is done with regexes, and the order in which the token patterns are
# defined is important.
#
# Suppose we had the following token definitions:
#  * INT - a token matching integers,
#  * FLOAT - a token matching floating point numbers,
#  * DOT - a token matching a single literal dot character, i.e. "."
#
# The FLOAT token would have to be defined first, since we would want the input "1.23"
# to be tokenized as a single FLOAT token, and *not* three tokens (INT, DOT, INT).
#
# Sometimes, however, different tokens match too similar patterns, and it is not
# possible to define them in order that would avoid any ambiguity. One such case are
# the OPT_VAL and PY_NUMBER tokens, as both can match an integer literal, say "42".
#
# In order to avoid the dilemmas, the lexer implements a concept of STATES. States are
# used to split token definitions into subgroups, and in each lexer state only a single
# subgroup is used for tokenizing the input. Lexer states can therefore be though of as
# token namespaces.
#
# For example, while parsing the value of the "--params" option, we do not want to
# "recognize" it as a single OPT_VAL token, but instead want to parse it as a Python
# dictionary and verify its syntactial correctness. On the other hand, while parsing
# the value of an option other than "--params", we do not really care about its
# structure, and thus do not want to use any of the "Python tokens" for pattern matching.
#
# Token definition order is important, thus an OrderedDict is used. In addition, PEP 468
# guarantees us that the order of kwargs is preserved in Python 3.6+.
token_types = OrderedDict(
    state_parse_pos_args=OrderedDict(
        GOTO_PARSE_NON_PARAMS_OPTIONS=r"(?P<GOTO_PARSE_NON_PARAMS_OPTIONS>(?=--))",  # double dash - starting the options list
        DEST_VAR=r"(?P<DEST_VAR>[^\d\W]\w*)",  # essentially a Python ID
    ),
    state_parse_non_params_options=OrderedDict(
        GOTO_PARSE_PARAMS_OPTION=r"(?P<GOTO_PARSE_PARAMS_OPTION>(?=--params(?:\s|=|--|$)))",  # the --params option
        OPTION_SPEC=r"(?P<OPTION_SPEC>--\w+)",
        OPTION_EQ=r"(?P<OPTION_EQ>=)",
        OPT_VAL=r"(?P<OPT_VAL>\S+?(?=\s|--|$))",
    ),
    state_parse_params_option=OrderedDict(
        PY_STRING=r"(?P<PY_STRING>(?:{})|(?:{}))".format(  # single and double quoted strings
            r"'(?:[^'\\]|\.)*'", r'"(?:[^"\\]|\.)*"'
        ),
        PARAMS_OPT_SPEC=r"(?P<PARAMS_OPT_SPEC>--params(?=\s|=|--|$))",
        PARAMS_OPT_EQ=r"(?P<PARAMS_OPT_EQ>=)",
        GOTO_PARSE_NON_PARAMS_OPTIONS=r"(?P<GOTO_PARSE_NON_PARAMS_OPTIONS>(?=--\w+))",  # found another option spec
        PY_BOOL=r"(?P<PY_BOOL>True|False)",
        DOLLAR_PY_ID=r"(?P<DOLLAR_PY_ID>\$[^\d\W]\w*)",
        PY_NUMBER=r"(?P<PY_NUMBER>-?[1-9]\d*(?:\.\d+)?(:?[e|E][+-]?\d+)?)",
        SQUOTE=r"(?P<SQUOTE>')",
        DQUOTE=r'(?P<DQUOTE>")',
        COLON=r"(?P<COLON>:)",
        COMMA=r"(?P<COMMA>,)",
        LCURL=r"(?P<LCURL>\{)",
        RCURL=r"(?P<RCURL>})",
        LSQUARE=r"(?P<LSQUARE>\[)",
        RSQUARE=r"(?P<RSQUARE>])",
        LPAREN=r"(?P<LPAREN>\()",
        RPAREN=r"(?P<RPAREN>\))",
    ),
    common=OrderedDict(
        WS=r"(?P<WS>\s+)",
        EOL=r"(?P<EOL>$)",
        UNKNOWN=r"(?P<UNKNOWN>\S+)",  # anything not a whitespace or matched by something else
    ),
)


class AutoStrEnum(str, enum.Enum):
    """Base enum class for for name=value str enums."""

    def _generate_next_value_(name, start, count, last_values):
        return name


TokenType = AutoStrEnum(  # type: ignore  # pytype: disable=wrong-arg-types
    "TokenType",
    [
        (name, enum.auto())
        for name in itertools.chain.from_iterable(token_types.values())
        if not name.startswith("GOTO_")
    ],
)


class LexerState(AutoStrEnum):
    PARSE_POS_ARGS = enum.auto()  # parsing positional arguments
    PARSE_NON_PARAMS_OPTIONS = enum.auto()  # parsing options other than "--params"
    PARSE_PARAMS_OPTION = enum.auto()  # parsing the "--params" option
    STATE_END = enum.auto()


class Lexer(object):
    """Lexical analyzer for tokenizing the cell magic input line."""

    _GRAND_PATTERNS = {
        LexerState.PARSE_POS_ARGS: re.compile(
            "|".join(
                itertools.chain(
                    token_types["state_parse_pos_args"].values(),
                    token_types["common"].values(),
                )
            )
        ),
        LexerState.PARSE_NON_PARAMS_OPTIONS: re.compile(
            "|".join(
                itertools.chain(
                    token_types["state_parse_non_params_options"].values(),
                    token_types["common"].values(),
                )
            )
        ),
        LexerState.PARSE_PARAMS_OPTION: re.compile(
            "|".join(
                itertools.chain(
                    token_types["state_parse_params_option"].values(),
                    token_types["common"].values(),
                )
            )
        ),
    }

    def __init__(self, input_text):
        self._text = input_text

    def __iter__(self):
        # Since re.scanner does not seem to support manipulating inner scanner states,
        # we need to implement lexer state transitions manually using special
        # non-capturing lookahead token patterns to signal when a state transition
        # should be made.
        # Since we don't have "nested" states, we don't really need a stack and
        # this simple mechanism is sufficient.
        state = LexerState.PARSE_POS_ARGS
        offset = 0  # the number of characters processed so far

        while state != LexerState.STATE_END:
            token_stream = self._find_state_tokens(state, offset)

            for maybe_token in token_stream:  # pragma: NO COVER
                if isinstance(maybe_token, StateTransition):
                    state = maybe_token.new_state
                    offset = maybe_token.total_offset
                    break

                if maybe_token.type_ != TokenType.WS:
                    yield maybe_token

                if maybe_token.type_ == TokenType.EOL:
                    state = LexerState.STATE_END
                    break

    def _find_state_tokens(self, state, current_offset):
        """Scan the input for current state's tokens starting at ``current_offset``.

        Args:
            state (LexerState): The current lexer state.
            current_offset (int): The offset in the input text, i.e. the number
                of characters already scanned so far.

        Yields:
            The next ``Token`` or ``StateTransition`` instance.
        """
        pattern = self._GRAND_PATTERNS[state]
        scanner = pattern.finditer(self._text, current_offset)

        for match in scanner:  # pragma: NO COVER
            token_type = match.lastgroup

            if token_type.startswith("GOTO_"):
                yield StateTransition(
                    new_state=getattr(LexerState, token_type[5:]),  # w/o "GOTO_" prefix
                    total_offset=match.start(),
                )

            yield Token(token_type, match.group(), match.start())
