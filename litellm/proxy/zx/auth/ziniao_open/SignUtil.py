#!/usr/bin/python
# -*- coding: UTF-8 -*-
from Crypto.Signature import PKCS1_v1_5
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
import rsa
import base64

__pem_begin = '-----BEGIN RSA PRIVATE KEY-----\n'
__pem_end = '\n-----END RSA PRIVATE KEY-----'


def create_sign(all_params, private_key, sign_type):
    """创建签名

    :param all_params: 参数
    :type all_params: dict

    :param private_key: 私钥字符串
    :type private_key: str

    :param sign_type: 签名类型，'RSA', 'RSA2'二选一
    :type sign_type: str

    :return: 返回签名内容
    :rtype: str
    """
    sign_content = get_sign_content(all_params)
    private_key = _format_private_key(private_key)
    return sign(sign_content, private_key, sign_type)


def _format_private_key(private_key):
    if not private_key.startswith(__pem_begin):
        private_key = __pem_begin + private_key
    if not private_key.endswith(__pem_end):
        private_key = private_key + __pem_end
    return private_key


def get_sign_content(params):
    """构建签名内容

    1.筛选并排序
    获取所有请求参数，不包括字节类型参数，如文件、字节流，剔除sign字段，剔除值为空的参数，并按照参数名ASCII码递增排序（字母升序排序），
    如果遇到相同字符则按照第二个字符的键值ASCII码递增排序，以此类推。

    2.拼接
    将排序后的参数与其对应值，组合成“参数=参数值”的格式，并且把这些参数用&字符连接起来，此时生成的字符串为待签名字符串。

    :param params: 参数
    :type params: dict

    :return: 返回签名内容
    :rtype: str
    """
    keys = list(params.keys())
    keys.sort()
    result = []
    for key in keys:
        value = str(params.get(key))
        if len(value) > 0:
            result.append(key + '=' + value)

    return '&'.join(result)


def sign(content, private_key, sign_type):
    """签名

    :param content: 签名内容
    :type content: str

    :param private_key: 私钥字符串
    :type private_key: str

    :param sign_type: 签名类型，'RSA', 'RSA2'二选一
    :type sign_type: str

    :return: 返回签名内容
    :rtype: str
    """
    if sign_type.upper() == 'RSA':
        return rsa_sign(content, private_key, 'SHA-1')
    elif sign_type.upper() == 'RSA2':
        return sign_with_rsa2(private_key, content, 'utf-8')
    else:
        raise Exception('sign_type错误')


def rsa_sign(content, private_key, _hash):
    """SHAWithRSA

    :param content: 签名内容
    :type content: str

    :param private_key: 私钥
    :type private_key: str

    :param _hash: hash算法，如：SHA-1,SHA-256
    :type _hash: str

    :return: 签名内容
    :rtype: str
    """
    pri_key = rsa.PrivateKey.load_pkcs1(private_key.encode('utf-8'))
    sign_result = rsa.sign(content, pri_key, _hash)
    return base64.b64encode(sign_result)


def sign_with_rsa2(private_key, sign_content, charset):
    key = RSA.importKey(private_key)
    hash_value = SHA256.new(bytes(sign_content, encoding=charset))
    signer = PKCS1_v1_5.new(key)
    signature = signer.sign(hash_value)
    return str(base64.b64encode(signature), encoding=charset)


def fill_private_key_marker(private_key):
    return add_start_end(private_key, "-----BEGIN RSA PRIVATE KEY-----\n", "\n-----END RSA PRIVATE KEY-----")


def add_start_end(key, startMarker, endMarker):
    if key.find(startMarker) < 0:
        key = startMarker + key
    if key.find(endMarker) < 0:
        key = key + endMarker
    return key