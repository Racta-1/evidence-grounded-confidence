| run                   | dataset       | method                   |   n |   accuracy |   mean_conf |     ece |   brier |   auroc |    aurc |
|:----------------------|:--------------|:-------------------------|----:|-----------:|------------:|--------:|--------:|--------:|--------:|
| pilot_smolvlm256m_cpu | MedXpertQA-MM | Token probability        |  18 |      0.111 |       0.709 |   0.598 |   0.473 |   0.562 |   0.901 |
| pilot_smolvlm256m_cpu | MedXpertQA-MM | Verbalized               |   1 |      1     |       1     |   0     |   0     | nan     |   0     |
| pilot_smolvlm256m_cpu | MedXpertQA-MM | Sampling consistency     |  18 |      0.111 |       0.315 |   0.315 |   0.241 |   0.516 |   0.912 |
| pilot_smolvlm256m_cpu | MedXpertQA-MM | Platt-scaled token prob. |  18 |      0.111 |       0.174 |   0.063 |   0.104 |   0.562 |   0.901 |
| pilot_smolvlm256m_cpu | MedXpertQA-MM | EGC (evidence-adjusted)  |  18 |      0.111 |       0.214 |   0.183 |   0.134 |   0.25  |   0.96  |
| pilot_smolvlm256m_cpu | MedXpertQA-MM | EGC-logistic             |  18 |      0.111 |       0.164 |   0.053 |   0.103 |   0.375 |   0.939 |
| pilot_smolvlm256m_cpu | SLAKE         | Token probability        |  13 |      0.231 |       0.355 |   0.153 |   0.104 |   0.933 |   0.503 |
| pilot_smolvlm256m_cpu | SLAKE         | Verbalized               |   0 |    nan     |     nan     | nan     | nan     | nan     | nan     |
| pilot_smolvlm256m_cpu | SLAKE         | Sampling consistency     |  13 |      0.231 |       0.359 |   0.128 |   0.162 |   0.867 |   0.557 |
| pilot_smolvlm256m_cpu | SLAKE         | Platt-scaled token prob. |  13 |      0.231 |       0.077 |   0.154 |   0.154 |   0.933 |   0.503 |
| pilot_smolvlm256m_cpu | SLAKE         | EGC (evidence-adjusted)  |  13 |      0.231 |       0.212 |   0.193 |   0.08  |   0.933 |   0.523 |
| pilot_smolvlm256m_cpu | SLAKE         | EGC-logistic             |  13 |      0.231 |       0.096 |   0.135 |   0.134 |   0.933 |   0.503 |
| pilot_smolvlm256m_cpu | VQA-RAD       | Token probability        |   9 |      0.556 |       0.361 |   0.239 |   0.149 |   0.85  |   0.198 |
| pilot_smolvlm256m_cpu | VQA-RAD       | Verbalized               |   0 |    nan     |     nan     | nan     | nan     | nan     | nan     |
| pilot_smolvlm256m_cpu | VQA-RAD       | Sampling consistency     |   9 |      0.556 |       0.481 |   0.074 |   0.099 |   0.95  |   0.164 |
| pilot_smolvlm256m_cpu | VQA-RAD       | Platt-scaled token prob. |   9 |      0.556 |       0.167 |   0.43  |   0.344 |   0.85  |   0.198 |
| pilot_smolvlm256m_cpu | VQA-RAD       | EGC (evidence-adjusted)  |   9 |      0.556 |       0.32  |   0.279 |   0.192 |   0.85  |   0.198 |
| pilot_smolvlm256m_cpu | VQA-RAD       | EGC-logistic             |   9 |      0.556 |       0.15  |   0.411 |   0.362 |   0.9   |   0.182 |