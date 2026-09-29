# Run from the repo root: PYTHONPATH=. python tests/test_log_api.py  (no cluster needed)
from types import SimpleNamespace as NS
from api.log_api import LogAPI

class FakeK8s:
    def __init__(self, logs): self.logs = logs  # {container: text}
    def read_namespaced_pod(self, name, namespace): return NS(spec=NS(containers=[NS(name=c) for c in self.logs]))
    def read_namespaced_pod_log(self, name, namespace, container, tail_lines): return self.logs[container]

def logs_of(containers, important=True):
    api = LogAPI.__new__(LogAPI)  # skip k8s init
    api.namespace, api.pods, api._k8s_client = "ns", ["pod"], FakeK8s(containers)
    return api.get_pod_logs("pod", important=important)

# Status codes inside timestamps/IDs are not important; real status codes and text keywords are
out = logs_of({"payment": '{"level":"info","time":1790674934741,"msg":"Transaction complete."}\n'
                          '{"level":"warn","msg":"Payment request failed. Invalid token."}\n'
                          'GET /api/cart 404\n'
                          'WARNING: OOMKilled'})
assert out == ('Found 3 important log entries:\n\n'
               '{"level":"warn","msg":"Payment request failed. Invalid token."}\n'
               'GET /api/cart 404\n'
               'WARNING: OOMKilled'), out

# Single-container pod: logs unchanged. Multi-container pod: every container read, lines prefixed
assert logs_of({"app": "a\nb"}, important=False) == "a\nb"
assert logs_of({"flagd": "ERROR x", "flagd-ui": "started"}, important=False) == "[flagd] ERROR x\n[flagd-ui] started"
print("ok")
