# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------

from abc import ABCMeta, abstractmethod


class CryptoTransform(object, metaclass=ABCMeta):
    def __init__(self, key):
        self._key = key

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._key = None

    @abstractmethod
    def transform(self, data):
        raise NotImplementedError()


class BlockCryptoTransform(CryptoTransform):
    @abstractmethod
    def block_size(self):
        raise NotImplementedError()

    @abstractmethod
    def update(self, data):
        raise NotImplementedError()

    @abstractmethod
    def finalize(self):
        raise NotImplementedError()


class AuthenticatedCryptoTransform(object, metaclass=ABCMeta):
    @abstractmethod
    def tag(self):
        raise NotImplementedError()


class SignatureTransform(object, metaclass=ABCMeta):
    @abstractmethod
    def sign(self, digest):
        raise NotImplementedError()

    @abstractmethod
    def verify(self, digest, signature):
        raise NotImplementedError()


class DigestTransform(object, metaclass=ABCMeta):
    @abstractmethod
    def update(self, data):
        raise NotImplementedError()

    @abstractmethod
    def finalize(self, data):
        raise NotImplementedError()
