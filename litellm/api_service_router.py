class APIService:
    def __init__(self, name):
        self.name = name
        self.status = 'healthy'
        self.successful_calls = 0
        self.failed_calls = 0

    def mark_healthy(self):
        self.status = 'healthy'

    def mark_unhealthy(self):
        self.status = 'unhealthy'

    def increment_successful_calls(self):
        self.successful_calls += 1

    def increment_failed_calls(self):
        self.failed_calls += 1
        
class APIServiceRouter:
    def __init__(self):
        self.services = {
            'openai': APIService('openai'),
            'anthropic': APIService('anthropic'),
            'azure': APIService('azure')
        }
