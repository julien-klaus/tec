#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class LexerException(Exception):
    def __init__(self, message, column):
        self.message = message
        self.column = column

    def __str__(self):
        return str("Scanner exception at column %i: %s" % (self.column, self.message))


class ParserException(Exception):
    def __init__(self, message, column):
        self.message = message
        self.column = column

    def __str__(self):
        return str("Parser exception at position %i: %s" % (self.column, self.message))


class SemanticException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return str("SemanticException: %s" % self.message)
