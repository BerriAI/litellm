import hashlib
import logging


log = logging.getLogger("langfuse")


class Sampler:
    sample_rate: float

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    def sample_event(self, event: dict):
        # need to get trace_id from a given event
        # returns true if

        if "type" in event and "body" in event:
            event_type = event["type"]

            if event_type == "sdk-log":
                return True

            trace_id = None

            if event_type == "trace-create" and "id" in event["body"]:
                trace_id = event["body"]["id"]
            elif "trace_id" in event["body"]:
                trace_id = event["body"]["trace_id"]
            elif "traceId" in event["body"]:
                trace_id = event["body"]["traceId"]
            else:
                log.error("Unexpected event format: No trace id found in event")
                return True

            return self.deterministic_sample(trace_id, self.sample_rate)

        else:
            log.error("Unexpected event format: No trace id found in event")
            return True

    def deterministic_sample(self, trace_id: str, sample_rate: float):
        """Determins if an event should be sampled based on the trace_id and sample_rate. Event will be sent to server if True"""
        log.debug(
            f"Applying deterministic sampling to trace_id: {trace_id} with rate {sample_rate}"
        )

        # Use SHA-256 to hash the trace_id
        hash_object = hashlib.sha256(trace_id.encode())
        # Get the hexadecimal representation of the hash
        hash_hex = hash_object.hexdigest()

        # Take the first 8 characters of the hex digest and convert to integer
        hash_int = int(hash_hex[:8], 16)

        # Normalize the integer to a float in the range [0, 1)
        normalized_hash = hash_int / 0xFFFFFFFF

        result = normalized_hash < sample_rate

        if not result:
            log.debug(
                f"event with trace_id: {trace_id} and rate {sample_rate} was sampled and not sent to the server"
            )

        return result
