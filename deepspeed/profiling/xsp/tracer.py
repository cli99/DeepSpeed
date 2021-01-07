import logging
import time
from jaeger_client import Config
import atexit
from enum import IntEnum

import opentracing
# import opentracing


def _init(service):
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    config = Config(
        config={ # usually read from some yaml config
            'sampler': { # const type with param 1 configs all spans to be reported
                'type': 'const',
                'param': 1,
            },
            'reporter_batch_size': 10,# default 10. Number of span to submit at once to the agent or collector
            'reporter_queue_size': 1000, # default 100. Number of spans in memory buffer before batching them to the agent or collector
            'reporter_flush_interval': 0.01, # default 1s. The interval to trigger a flush, sending all in-memory spans to the agent or collector
            'local_agent': {
                'reporting_host': '10.124.76.134',
                # 'reporting_port': '16686',
            },
            'logging': False,
        },
        service_name='deepspeed',
        validate=True,
    )
    # this call also sets opentracing.tracer
    return config.initialize_tracer()


class TraceLevel(IntEnum):
    DISABLED = 0
    APPLICATION = 1
    MODEL = 2
    FRAMEWORK = 3


class NoOpSpan:
    def __init__(self, *args, **kwargs):
        pass

    def finish(*args, **kwargs):
        pass


class XSP:
    started = False

    def __init__(self, level=TraceLevel.DISABLED, service="deepspeed"):
        print("initialized tracer...")
        self.tracer = _init(service)
        self.level = level
        # atexit.register(self.close)

    def start_span(self, level, *args, **kwargs):
        if level > self.level:
            return NoOpSpan()
        return self.tracer.start_span(*args, **kwargs)

    def start_profile(self, *args, **kwargs):
        print("XXXX start xsp profiling")
        if self.started:
            print("xsp has already started")
            return
        if self.level > int(TraceLevel.DISABLED):
            self.root_span = self.tracer.start_span("root")
            self.start_time = time.time()
            self.started = True

    def end_profile(self, *args, **kwargs):
        print("XXXX end xsp profiling")
        self.started = False
        if self.level > int(TraceLevel.DISABLED):
            self.root_span.finish()
        self.close()

    def close(self):
        print("closing tracer...")
        time.sleep(1)
        self.tracer.close()


if __name__ == "__main__":
    log_level = logging.DEBUG
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(asctime)s %(message)s', level=log_level)

    # this call also sets opentracing.tracer
    tracer = _init('first-service')

    with tracer.start_span('TestSpan') as span:
        span.log_kv({'event': 'test message', 'life': 42})

        with tracer.start_span('ChildSpan', child_of=span) as child_span:
            child_span.log_kv({'event': 'down below'})

    time.sleep(
        3
    )  # yield to IOLoop to flush the spans - https://github.com/jaegertracing/jaeger-client-python/issues/50
    tracer.close()  # flush any buffered spans
