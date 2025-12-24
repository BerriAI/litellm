#!/usr/bin/python
# -*- coding: UTF-8 -*-

import json
import time

import requests

from common import SignUtil, RequestTypes
from common.RequestType import RequestType
from response.BaseResponse import BaseResponse
from response.StreamResponse import StreamResponse

_headers = {'Accept-Encoding': 'identity'}


class OpenClient:
    """调用客户端"""
    __app_id = ''
    __private_key = ''
    __url = ''

    def __init__(self, app_id, private_key, url):
        """客户端

        :param app_id: 应用ID
        :type app_id: str

        :param private_key: 应用私钥
        :type private_key: str

        :param url: 请求URL
        :type url: str
        """
        self.__app_id = app_id
        self.__private_key = private_key
        self.__url = url

    def execute(self, request, user_token=None, app_token=None, stream=False):
        """

        :param request: 请求对象，BaseRequest的子类

        :param user_token: (Optional) 用户token
        :type user_token: str

        :param app_token: (Optional) 应用token
        :type app_token: str

        :param stream: (Optional) 是否流式请求，默认False
        :type stream: bool

        :return: 返回请求结果
        :rtype: BaseResponse | Response
        """
        request_type = request.get_request_type()
        if not isinstance(request_type, RequestType):
            raise Exception('get_request_type返回错误类型，正确方式：RequestTypes.XX')

        if request.files is not None:
            response = self._post_file(request, user_token, app_token, stream)
        elif request_type == RequestTypes.GET:
            response = self._get(request, user_token, app_token, stream)
        elif request_type == RequestTypes.POST_FORM:
            response = self._post_form(request, user_token, app_token, stream)
        elif request_type == RequestTypes.POST_JSON:
            response = self._post_json(request, user_token, app_token, stream)
        elif request_type == RequestTypes.POST_UPLOAD:
            response = self._post_file(request, user_token, app_token, stream)
        elif request_type == RequestTypes.PUT:
            response = self._put(request, user_token, app_token, stream)
        elif request_type == RequestTypes.DELETE:
            response = self._delete(request, user_token, app_token, stream)
        else:
            raise Exception('get_request_type设置错误')
        if stream:
            return self._parse_stream_response(response)
        else:
            response.encoding = "utf-8"
            return self._parse_response(response.text)

    def _get(self, request, user_token, app_token, stream=False):
        all_params = self._build_params(request, user_token, app_token)
        response = requests.get(self.__url, all_params, headers=_headers, stream=stream)
        return response

    def _post_form(self, request, user_token, app_token, stream=False):
        all_params = self._build_params(request, user_token, app_token)
        response = requests.post(self.__url, data=all_params, headers=_headers, stream=stream)
        return response

    def _post_json(self, request, user_token, app_token, stream=False):
        all_params = self._build_params(request, user_token, app_token)
        response = requests.post(self.__url, json=all_params, headers=_headers, stream=stream)
        return response

    def _post_file(self, request, user_token, app_token, stream=False):
        all_params = self._build_params(request, user_token, app_token)
        response = requests.request('POST', self.__url, data=all_params, files=request.files, headers=_headers, stream=stream)
        return response

    def _put(self, request, user_token, app_token, stream=False):
        all_params = self._build_params(request, user_token, app_token)
        response = requests.put(self.__url, json=all_params, headers=_headers, stream=stream)
        return response

    def _delete(self, request, user_token, app_token, stream=False):
        all_params = self._build_params(request, user_token, app_token)
        response = requests.delete(self.__url, json=all_params, headers=_headers, stream=stream)
        return response

    def _build_params(self, request, user_token, app_token):
        """构建所有的请求参数

        :param request: 请求对象
        :type request: request.BaseRequest

        :param params: 业务请求参数
        :type params: dict

        :param token: token
        :type token: str

        :return: 返回请求参数
        :rtype: str
        """
        all_params = {
            'app_id': self.__app_id,
            'method': request.get_method(),
            'charset': 'UTF-8',
            'sign_type': 'RSA2',
            'timestamp': int(round(time.time() * 1000)),
            'version': '1.0',
            'sdk_version': '1.0'
        }

        if user_token is not None:
            all_params['user_access_token'] = user_token
        if app_token is not None:
            all_params['app_auth_token'] = app_token

        biz_model = request.biz_model
        params_model = request.params_model

        if biz_model is None:
            biz_model = {}
        if isinstance(biz_model, str):
            biz_str = biz_model
        else:
            biz_str = json.dumps(biz_model)

        if params_model is None:
            params_model = {}
        if isinstance(params_model, str):
            params_str = params_model
        else:
            params_str = json.dumps(params_model)

        # 添加业务参数
        if biz_str is not None:
            all_params['biz_content'] = biz_str
        # url携带参数(GET请求以外有效)
        if params_str is not None:
            all_params['params_content'] = params_str

        # 构建sign
        sign = SignUtil.create_sign(all_params, self.__private_key, 'RSA2')
        all_params['sign'] = sign
        return all_params

    @staticmethod
    def _parse_response(resp):
        response_dict = json.loads(resp)
        return BaseResponse(response_dict)

    @staticmethod
    def _parse_stream_response(resp):
        return StreamResponse(resp)

