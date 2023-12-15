"""Amazon SQS Connection."""

from __future__ import annotations

import json

from botocore.serialize import Serializer
from vine import transform

from kombu.asynchronous.aws.connection import AsyncAWSQueryConnection
from kombu.asynchronous.aws.ext import AWSRequest

from .ext import boto3
from .message import AsyncMessage
from .queue import AsyncQueue

__all__ = ('AsyncSQSConnection',)


class AsyncSQSConnection(AsyncAWSQueryConnection):
    """Async SQS Connection."""

    def __init__(self, sqs_connection, debug=0, region=None, **kwargs):
        if boto3 is None:
            raise ImportError('boto3 is not installed')
        super().__init__(
            sqs_connection,
            region_name=region, debug=debug,
            **kwargs
        )

    def _create_query_request(self, operation, params, queue_url, method):
        params = params.copy()
        if operation:
            params['Action'] = operation

        # defaults for non-get
        param_payload = {'data': params}
        if method.lower() == 'get':
            # query-based opts
            param_payload = {'params': params}

        return AWSRequest(method=method, url=queue_url, **param_payload)

    def _create_json_request(self, operation, params, queue_url):
        params = params.copy()
        params['QueueUrl'] = queue_url

        service_model = self.sqs_connection.meta.service_model
        operation_model = service_model.operation_model(operation)

        url = self.sqs_connection._endpoint.host

        headers = {}
        # Content-Type
        json_version = operation_model.metadata['jsonVersion']
        content_type = f'application/x-amz-json-{json_version}'
        headers['Content-Type'] = content_type

        # X-Amz-Target
        target = '{}.{}'.format(
            operation_model.metadata['targetPrefix'],
            operation_model.name,
        )
        headers['X-Amz-Target'] = target

        param_payload = {
            'data': json.dumps(params),
            'headers': headers
        }

        method = operation_model.http.get('method', Serializer.DEFAULT_METHOD)
        return AWSRequest(
            method=method,
            url=url,
            **param_payload
        )

    def make_request(self, operation_name, params, queue_url, verb, callback=None):  # noqa
        """
        Overide make_request to support different protocols.

        botocore is soon going to change the default protocol of communicating
        with SQS backend from 'query' to 'json', so we need a special
        implementation of make_request for SQS. More information on this can
        be found in: https://github.com/celery/kombu/pull/1807.
        """
        signer = self.sqs_connection._request_signer

        service_model = self.sqs_connection.meta.service_model
        protocol = service_model.protocol

        if protocol == 'query':
            request = self._create_query_request(
                operation_name, params, queue_url, verb)
        elif protocol == 'json':
            request = self._create_json_request(
                operation_name, params, queue_url)
        else:
            raise Exception(f'Unsupported protocol: {protocol}.')

        signing_type = 'presign-url' if request.method.lower() == 'get' \
            else 'standard'

        signer.sign(operation_name, request, signing_type=signing_type)
        prepared_request = request.prepare()

        return self._mexe(prepared_request, callback=callback)

    def create_queue(self, queue_name,
                     visibility_timeout=None, callback=None):
        params = {'QueueName': queue_name}
        if visibility_timeout:
            params['DefaultVisibilityTimeout'] = format(
                visibility_timeout, 'd',
            )
        return self.get_object('CreateQueue', params,
                               callback=callback)

    def delete_queue(self, queue, force_deletion=False, callback=None):
        return self.get_status('DeleteQueue', None, queue.id,
                               callback=callback)

    def get_queue_url(self, queue):
        res = self.sqs_connection.get_queue_url(QueueName=queue)
        return res['QueueUrl']

    def get_queue_attributes(self, queue, attribute='All', callback=None):
        return self.get_object(
            'GetQueueAttributes', {'AttributeName': attribute},
            queue.id, callback=callback,
        )

    def set_queue_attribute(self, queue, attribute, value, callback=None):
        return self.get_status(
            'SetQueueAttribute',
            {'Attribute.Name': attribute, 'Attribute.Value': value},
            queue.id, callback=callback,
        )

    def receive_message(
        self, queue, queue_url, number_messages=1, visibility_timeout=None,
        attributes=('ApproximateReceiveCount',), wait_time_seconds=None,
        callback=None
    ):
        params = {'MaxNumberOfMessages': number_messages}
        if visibility_timeout:
            params['VisibilityTimeout'] = visibility_timeout
        if attributes:
            attrs = {}
            for idx, attr in enumerate(attributes):
                attrs['AttributeName.' + str(idx + 1)] = attr
            params.update(attrs)
        if wait_time_seconds is not None:
            params['WaitTimeSeconds'] = wait_time_seconds
        return self.get_list(
            'ReceiveMessage', params, [('Message', AsyncMessage)],
            queue_url, callback=callback, parent=queue,
        )

    def delete_message(self, queue, receipt_handle, callback=None):
        return self.delete_message_from_handle(
            queue, receipt_handle, callback,
        )

    def delete_message_batch(self, queue, messages, callback=None):
        params = {}
        for i, m in enumerate(messages):
            prefix = f'DeleteMessageBatchRequestEntry.{i + 1}'
            params.update({
                f'{prefix}.Id': m.id,
                f'{prefix}.ReceiptHandle': m.receipt_handle,
            })
        return self.get_object(
            'DeleteMessageBatch', params, queue.id,
            verb='POST', callback=callback,
        )

    def delete_message_from_handle(self, queue, receipt_handle,
                                   callback=None):
        return self.get_status(
            'DeleteMessage', {'ReceiptHandle': receipt_handle},
            queue, callback=callback,
        )

    def send_message(self, queue, message_content,
                     delay_seconds=None, callback=None):
        params = {'MessageBody': message_content}
        if delay_seconds:
            params['DelaySeconds'] = int(delay_seconds)
        return self.get_object(
            'SendMessage', params, queue.id,
            verb='POST', callback=callback,
        )

    def send_message_batch(self, queue, messages, callback=None):
        params = {}
        for i, msg in enumerate(messages):
            prefix = f'SendMessageBatchRequestEntry.{i + 1}'
            params.update({
                f'{prefix}.Id': msg[0],
                f'{prefix}.MessageBody': msg[1],
                f'{prefix}.DelaySeconds': msg[2],
            })
        return self.get_object(
            'SendMessageBatch', params, queue.id,
            verb='POST', callback=callback,
        )

    def change_message_visibility(self, queue, receipt_handle,
                                  visibility_timeout, callback=None):
        return self.get_status(
            'ChangeMessageVisibility',
            {'ReceiptHandle': receipt_handle,
             'VisibilityTimeout': visibility_timeout},
            queue.id, callback=callback,
        )

    def change_message_visibility_batch(self, queue, messages, callback=None):
        params = {}
        for i, t in enumerate(messages):
            pre = f'ChangeMessageVisibilityBatchRequestEntry.{i + 1}'
            params.update({
                f'{pre}.Id': t[0].id,
                f'{pre}.ReceiptHandle': t[0].receipt_handle,
                f'{pre}.VisibilityTimeout': t[1],
            })
        return self.get_object(
            'ChangeMessageVisibilityBatch', params, queue.id,
            verb='POST', callback=callback,
        )

    def get_all_queues(self, prefix='', callback=None):
        params = {}
        if prefix:
            params['QueueNamePrefix'] = prefix
        return self.get_list(
            'ListQueues', params, [('QueueUrl', AsyncQueue)],
            callback=callback,
        )

    def get_queue(self, queue_name, callback=None):
        # TODO Does not support owner_acct_id argument
        return self.get_all_queues(
            queue_name,
            transform(self._on_queue_ready, callback, queue_name),
        )
    lookup = get_queue

    def _on_queue_ready(self, name, queues):
        return next(
            (q for q in queues if q.url.endswith(name)), None,
        )

    def get_dead_letter_source_queues(self, queue, callback=None):
        return self.get_list(
            'ListDeadLetterSourceQueues', {'QueueUrl': queue.url},
            [('QueueUrl', AsyncQueue)],
            callback=callback,
        )

    def add_permission(self, queue, label, aws_account_id, action_name,
                       callback=None):
        return self.get_status(
            'AddPermission',
            {'Label': label,
             'AWSAccountId': aws_account_id,
             'ActionName': action_name},
            queue.id, callback=callback,
        )

    def remove_permission(self, queue, label, callback=None):
        return self.get_status(
            'RemovePermission', {'Label': label}, queue.id, callback=callback,
        )
