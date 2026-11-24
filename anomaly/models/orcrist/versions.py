"""The versions of Orcrist, apart from the model so that they can be listed without torch (see model.py for what
the names mean: the macro area, then the version inside it)."""
VERSIONS = {
    # condition: "plain", one learned vector per service; "type+instance", e_type + e_instance, the instance part
    # dropped in training with probability INSTANCE_DROPOUT so that what services of a technology share goes in e_type;
    # "onehot", the service's one-hot. prior: "standard" N(0, I), or "service" p(z | e_s) = N(mu(e_s), sigma(e_s))
    "AA": {"condition": "plain", "prior": "standard"},  # the reference conditional VAE
    "AB": {"condition": "type+instance", "prior": "standard"},  # + the technology of the service
    "AC": {"condition": "type+instance", "prior": "service"},  # + the service's own prior
    "AD": {"condition": "onehot", "prior": "service"},  # AC with a one-hot in place of the embedding (ablation)
}
