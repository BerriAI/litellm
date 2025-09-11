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

from google.cloud.bigquery.magics.line_arg_parser import DuplicateQueryParamsError
from google.cloud.bigquery.magics.line_arg_parser import ParseError
from google.cloud.bigquery.magics.line_arg_parser import QueryParamsParseError
from google.cloud.bigquery.magics.line_arg_parser import TokenType


class ParseNode(object):
    """A base class for nodes in the input parsed to an abstract syntax tree."""


class InputLine(ParseNode):
    def __init__(self, destination_var, option_list):
        self.destination_var = destination_var
        self.option_list = option_list


class DestinationVar(ParseNode):
    def __init__(self, token):
        # token type is DEST_VAR
        self.token = token
        self.name = token.lexeme if token is not None else None


class CmdOptionList(ParseNode):
    def __init__(self, option_nodes):
        self.options = [node for node in option_nodes]  # shallow copy


class CmdOption(ParseNode):
    def __init__(self, name, value):
        self.name = name  # string
        self.value = value  # CmdOptionValue node


class ParamsOption(CmdOption):
    def __init__(self, value):
        super(ParamsOption, self).__init__("params", value)


class CmdOptionValue(ParseNode):
    def __init__(self, token):
        # token type is OPT_VAL
        self.token = token
        self.value = token.lexeme


class PyVarExpansion(ParseNode):
    def __init__(self, token):
        self.token = token
        self.raw_value = token.lexeme


class PyDict(ParseNode):
    def __init__(self, dict_items):
        self.items = [item for item in dict_items]  # shallow copy


class PyDictItem(ParseNode):
    def __init__(self, key, value):
        self.key = key
        self.value = value


class PyDictKey(ParseNode):
    def __init__(self, token):
        self.token = token
        self.key_value = token.lexeme


class PyScalarValue(ParseNode):
    def __init__(self, token, raw_value):
        self.token = token
        self.raw_value = raw_value


class PyTuple(ParseNode):
    def __init__(self, tuple_items):
        self.items = [item for item in tuple_items]  # shallow copy


class PyList(ParseNode):
    def __init__(self, list_items):
        self.items = [item for item in list_items]  # shallow copy


class Parser(object):
    """Parser for the tokenized cell magic input line.

    The parser recognizes a simplified subset of Python grammar, specifically
    a dictionary representation in typical use cases when the "--params" option
    is used with the %%bigquery cell magic.

    The grammar (terminal symbols are CAPITALIZED):

        input_line       : destination_var option_list
        destination_var  : DEST_VAR | EMPTY
        option_list      : (OPTION_SPEC [OPTION_EQ] option_value)*
                           (params_option | EMPTY)
                           (OPTION_SPEC [OPTION_EQ] option_value)*

        option_value     : OPT_VAL | EMPTY

        # DOLLAR_PY_ID can occur if a variable passed to --params does not exist
        # and is thus not expanded to a dict.
        params_option    : PARAMS_OPT_SPEC [PARAMS_OPT_EQ] \
                           (DOLLAR_PY_ID | PY_STRING | py_dict)

        py_dict          : LCURL dict_items RCURL
        dict_items       : dict_item | (dict_item COMMA dict_items)
        dict_item        : (dict_key COLON py_value) | EMPTY

        # dict items are actually @parameter names in the cell body (i.e. the query),
        # thus restricting them to strings.
        dict_key         : PY_STRING

        py_value         : PY_BOOL
                         | PY_NUMBER
                         | PY_STRING
                         | py_tuple
                         | py_list
                         | py_dict

        py_tuple         : LPAREN collection_items RPAREN
        py_list          : LSQUARE collection_items RSQUARE
        collection_items : collection_item | (collection_item COMMA collection_items)
        collection_item  : py_value | EMPTY

    Args:
        lexer (line_arg_parser.lexer.Lexer):
            An iterable producing a tokenized cell magic argument line.
    """

    def __init__(self, lexer):
        self._lexer = lexer
        self._tokens_iter = iter(self._lexer)
        self.get_next_token()

    def get_next_token(self):
        """Obtain the next token from the token stream and store it as current."""
        token = next(self._tokens_iter)
        self._current_token = token

    def consume(self, expected_type, exc_type=ParseError):
        """Move to the next token in token stream if it matches the expected type.

        Args:
            expected_type (lexer.TokenType): The expected token type to be consumed.
            exc_type (Optional[ParseError]): The type of the exception to raise. Should be
                the ``ParseError`` class or one of its subclasses. Defaults to
                ``ParseError``.

        Raises:
            ParseError: If the current token does not match the expected type.
        """
        if self._current_token.type_ == expected_type:
            if expected_type != TokenType.EOL:
                self.get_next_token()
        else:
            if self._current_token.type_ == TokenType.EOL:
                msg = "Unexpected end of input, expected {}.".format(expected_type)
            else:
                msg = "Expected token type {}, but found {} at position {}.".format(
                    expected_type, self._current_token.lexeme, self._current_token.pos
                )
            self.error(message=msg, exc_type=exc_type)

    def error(self, message="Syntax error.", exc_type=ParseError):
        """Raise an error with the given message.

        Args:
            expected_type (lexer.TokenType): The expected token type to be consumed.
            exc_type (Optional[ParseError]): The type of the exception to raise. Should be
                the ``ParseError`` class or one of its subclasses. Defaults to
                ``ParseError``.

        Raises:
            ParseError: If the current token does not match the expected type.
        """
        raise exc_type(message)

    def input_line(self):
        """The top level method for parsing the cell magic arguments line.

        Implements the following grammar production rule:

            input_line : destination_var option_list
        """
        dest_var = self.destination_var()
        options = self.option_list()

        token = self._current_token

        if token.type_ != TokenType.EOL:
            msg = "Unexpected input at position {}: {}".format(token.pos, token.lexeme)
            self.error(msg)

        return InputLine(dest_var, options)

    def destination_var(self):
        """Implementation of the ``destination_var`` grammar production rule.

        Production:

            destination_var  : DEST_VAR | EMPTY
        """
        token = self._current_token

        if token.type_ == TokenType.DEST_VAR:
            self.consume(TokenType.DEST_VAR)
            result = DestinationVar(token)
        elif token.type_ == TokenType.UNKNOWN:
            msg = "Unknown input at position {}: {}".format(token.pos, token.lexeme)
            self.error(msg)
        else:
            result = DestinationVar(None)

        return result

    def option_list(self):
        """Implementation of the ``option_list`` grammar production rule.

        Production:

            option_list : (OPTION_SPEC [OPTION_EQ] option_value)*
                          (params_option | EMPTY)
                          (OPTION_SPEC [OPTION_EQ] option_value)*
        """
        all_options = []

        def parse_nonparams_options():
            while self._current_token.type_ == TokenType.OPTION_SPEC:
                token = self._current_token
                self.consume(TokenType.OPTION_SPEC)

                opt_name = token.lexeme[2:]  # cut off the "--" prefix

                # skip the optional "=" character
                if self._current_token.type_ == TokenType.OPTION_EQ:
                    self.consume(TokenType.OPTION_EQ)

                opt_value = self.option_value()
                option = CmdOption(opt_name, opt_value)
                all_options.append(option)

        parse_nonparams_options()

        token = self._current_token

        if token.type_ == TokenType.PARAMS_OPT_SPEC:
            option = self.params_option()
            all_options.append(option)

        parse_nonparams_options()

        if self._current_token.type_ == TokenType.PARAMS_OPT_SPEC:
            self.error(
                message="Duplicate --params option", exc_type=DuplicateQueryParamsError
            )

        return CmdOptionList(all_options)

    def option_value(self):
        """Implementation of the ``option_value`` grammar production rule.

        Production:

            option_value : OPT_VAL | EMPTY
        """
        token = self._current_token

        if token.type_ == TokenType.OPT_VAL:
            self.consume(TokenType.OPT_VAL)
            result = CmdOptionValue(token)
        elif token.type_ == TokenType.UNKNOWN:
            msg = "Unknown input at position {}: {}".format(token.pos, token.lexeme)
            self.error(msg)
        else:
            result = None

        return result

    def params_option(self):
        """Implementation of the ``params_option`` grammar production rule.

        Production:

            params_option : PARAMS_OPT_SPEC [PARAMS_OPT_EQ] \
                            (DOLLAR_PY_ID | PY_STRING | py_dict)
        """
        self.consume(TokenType.PARAMS_OPT_SPEC)

        # skip the optional "=" character
        if self._current_token.type_ == TokenType.PARAMS_OPT_EQ:
            self.consume(TokenType.PARAMS_OPT_EQ)

        if self._current_token.type_ == TokenType.DOLLAR_PY_ID:
            token = self._current_token
            self.consume(TokenType.DOLLAR_PY_ID)
            opt_value = PyVarExpansion(token)
        elif self._current_token.type_ == TokenType.PY_STRING:
            token = self._current_token
            self.consume(TokenType.PY_STRING, exc_type=QueryParamsParseError)
            opt_value = PyScalarValue(token, token.lexeme)
        else:
            opt_value = self.py_dict()

        result = ParamsOption(opt_value)

        return result

    def py_dict(self):
        """Implementation of the ``py_dict`` grammar production rule.

        Production:

            py_dict : LCURL dict_items RCURL
        """
        self.consume(TokenType.LCURL, exc_type=QueryParamsParseError)
        dict_items = self.dict_items()
        self.consume(TokenType.RCURL, exc_type=QueryParamsParseError)

        return PyDict(dict_items)

    def dict_items(self):
        """Implementation of the ``dict_items`` grammar production rule.

        Production:

            dict_items : dict_item | (dict_item COMMA dict_items)
        """
        result = []

        item = self.dict_item()
        if item is not None:
            result.append(item)

        while self._current_token.type_ == TokenType.COMMA:
            self.consume(TokenType.COMMA, exc_type=QueryParamsParseError)
            item = self.dict_item()
            if item is not None:
                result.append(item)

        return result

    def dict_item(self):
        """Implementation of the ``dict_item`` grammar production rule.

        Production:

            dict_item : (dict_key COLON py_value) | EMPTY
        """
        token = self._current_token

        if token.type_ == TokenType.PY_STRING:
            key = self.dict_key()
            self.consume(TokenType.COLON, exc_type=QueryParamsParseError)
            value = self.py_value()
            result = PyDictItem(key, value)
        elif token.type_ == TokenType.UNKNOWN:
            msg = "Unknown input at position {}: {}".format(token.pos, token.lexeme)
            self.error(msg, exc_type=QueryParamsParseError)
        else:
            result = None

        return result

    def dict_key(self):
        """Implementation of the ``dict_key`` grammar production rule.

        Production:

            dict_key : PY_STRING
        """
        token = self._current_token
        self.consume(TokenType.PY_STRING, exc_type=QueryParamsParseError)
        return PyDictKey(token)

    def py_value(self):
        """Implementation of the ``py_value`` grammar production rule.

        Production:

            py_value : PY_BOOL | PY_NUMBER | PY_STRING | py_tuple | py_list | py_dict
        """
        token = self._current_token

        if token.type_ == TokenType.PY_BOOL:
            self.consume(TokenType.PY_BOOL, exc_type=QueryParamsParseError)
            return PyScalarValue(token, token.lexeme)
        elif token.type_ == TokenType.PY_NUMBER:
            self.consume(TokenType.PY_NUMBER, exc_type=QueryParamsParseError)
            return PyScalarValue(token, token.lexeme)
        elif token.type_ == TokenType.PY_STRING:
            self.consume(TokenType.PY_STRING, exc_type=QueryParamsParseError)
            return PyScalarValue(token, token.lexeme)
        elif token.type_ == TokenType.LPAREN:
            tuple_node = self.py_tuple()
            return tuple_node
        elif token.type_ == TokenType.LSQUARE:
            list_node = self.py_list()
            return list_node
        elif token.type_ == TokenType.LCURL:
            dict_node = self.py_dict()
            return dict_node
        else:
            msg = "Unexpected token type {} at position {}.".format(
                token.type_, token.pos
            )
            self.error(msg, exc_type=QueryParamsParseError)

    def py_tuple(self):
        """Implementation of the ``py_tuple`` grammar production rule.

        Production:

            py_tuple : LPAREN collection_items RPAREN
        """
        self.consume(TokenType.LPAREN, exc_type=QueryParamsParseError)
        items = self.collection_items()
        self.consume(TokenType.RPAREN, exc_type=QueryParamsParseError)

        return PyTuple(items)

    def py_list(self):
        """Implementation of the ``py_list`` grammar production rule.

        Production:

            py_list : LSQUARE collection_items RSQUARE
        """
        self.consume(TokenType.LSQUARE, exc_type=QueryParamsParseError)
        items = self.collection_items()
        self.consume(TokenType.RSQUARE, exc_type=QueryParamsParseError)

        return PyList(items)

    def collection_items(self):
        """Implementation of the ``collection_items`` grammar production rule.

        Production:

            collection_items : collection_item | (collection_item COMMA collection_items)
        """
        result = []

        item = self.collection_item()
        if item is not None:
            result.append(item)

        while self._current_token.type_ == TokenType.COMMA:
            self.consume(TokenType.COMMA, exc_type=QueryParamsParseError)
            item = self.collection_item()
            if item is not None:
                result.append(item)

        return result

    def collection_item(self):
        """Implementation of the ``collection_item`` grammar production rule.

        Production:

            collection_item : py_value | EMPTY
        """
        if self._current_token.type_ not in {TokenType.RPAREN, TokenType.RSQUARE}:
            result = self.py_value()
        else:
            result = None  # end of list/tuple items

        return result
