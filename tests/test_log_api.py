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
assert logs_of({"server": "ERROR x", "sidecar": "started"}, important=False) == "[server] ERROR x\n[sidecar] started"

# Astronomy Shop, real lines (kind cluster 2026-10-01; shipping's WARN from the AIOpsLab run of 2026-08-01): flag
# evaluations and talk with flagd go, flag announcements keep only their symptom, real failures stay untouched
shipping_eval = ('\x1b[2m2026-10-01T17:09:52.707820Z\x1b[0m \x1b[32m INFO\x1b[0m \x1b[2mshipping::shipping_service\x1b[0m\x1b[2m:\x1b[0m '
                 '\x1b[3mfeature_flag_key\x1b[0m\x1b[2m=\x1b[0m"intlShippingSlowdown" \x1b[3mfeature_flag_provider_name\x1b[0m\x1b[2m=\x1b[0m"flagd" '
                 '\x1b[3mfeature_flag_variant\x1b[0m\x1b[2m=\x1b[0m"off"')
shipping_flagd = "2026-08-01T12:27:20.477972Z  WARN new:new: open_feature_flagd::resolver::rpc: Connection attempt 1 failed, retrying in 1000ms: transport error"
fraud = ("2026-10-01 17:49:16 - fraud-detection - FeatureFlag 'kafkaQueueProblems' is enabled, sleeping 1 second "
         "trace_id=a3338111516b6b73f97c17a8fe211366 span_id=f04b739e4f59eeed trace_flags=03 ")
ad_failure = ("2026-10-01 17:48:35 - oteldemo.AdService - GetAds Failed with status Status{code=UNAVAILABLE, description=null, cause=null} "
              "trace_id=fd02380c3a864263820e825f306d3174 span_id=1a24202e890b351b trace_flags=01")
ad_gc = "Feature Flag adManualGc enabled, performing a manual gc now"  # demo 3.1.0 source (AdService.java:242): ad crash-looped before logging it
lines = "\n".join([shipping_eval, shipping_flagd, fraud, ad_failure, ad_gc])
assert logs_of({"app": lines}, important=False) == "\n".join([
    "2026-10-01 17:49:16 - fraud-detection - sleeping 1 second trace_id=a3338111516b6b73f97c17a8fe211366 span_id=f04b739e4f59eeed trace_flags=03",
    ad_failure,
    "performing a manual gc now",
])
# Before, flagd's WARN was among the important lines; now only the real failure is
assert logs_of({"app": lines}) == "Found 1 important log entries:\n\n" + ad_failure

# Pod and service lists (real names, AIOpsLab run 2026-08-01): no flagd, so every tool answers "does not exist"
names = lambda *n: NS(items=[NS(metadata=NS(name=x)) for x in n])
api = LogAPI.__new__(LogAPI)
api.namespace, api._pods_cache, api._services_cache = "astronomy-shop", None, None
api._k8s_client = NS(list_namespaced_pod=lambda ns: names("email-6d9bc6d666-bqqct", "flagd-6797468f77-xgwrd", "fraud-detection-5d94d88bb7-9np78"),
                     list_namespaced_service=lambda ns: names("email", "flagd", "fraud-detection"))
assert api.get_pods_list() == ["email-6d9bc6d666-bqqct", "fraud-detection-5d94d88bb7-9np78"]
assert api.get_services_list() == ["email", "fraud-detection"]
print("ok")
