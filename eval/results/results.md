# BankLens results

## Overall

| Config | Hit@5 | Recall@5 | MRR | Answer correct | Faithfulness | Cite prec. | Abstain P | Abstain R | False abstain | Verify acc | Verify macro-F1 | Latency ms | $ / run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01_baseline | 0.567 | 0.400 | 0.358 | 0.675 | 0.817 | 0.436 | 0.692 | 0.900 | 0.133 | 0.867 | 0.866 | 1154 | 0.083 |
| 02_pymupdf_tables | 0.567 | 0.367 | 0.269 | 0.625 | 0.761 | 0.442 | 0.588 | 1.000 | 0.233 | 0.833 | 0.822 | 794 | 0.059 |
| 03_hybrid | 0.533 | 0.358 | 0.227 | 0.675 | 0.833 | 0.309 | 0.625 | 1.000 | 0.200 | 0.833 | 0.839 | 933 | 0.073 |
| 04_filter | 0.667 | 0.433 | 0.278 | 0.750 | 0.896 | 0.453 | 0.625 | 1.000 | 0.200 | 0.867 | 0.866 | 912 | 0.068 |
| 05_rerank | 0.700 | 0.533 | 0.456 | 0.825 | 1.000 | 0.395 | 0.769 | 1.000 | 0.100 | 0.900 | 0.893 | 3385 | 0.078 |
| 06_ctx_header | 0.633 | 0.525 | 0.493 | 0.850 | 0.940 | 0.402 | 0.833 | 1.000 | 0.067 | 0.867 | 0.866 | 2685 | 0.079 |

## Answer correctness by question type

| Config | comparison | factual | numeric | unanswerable |
|---|---|---|---|---|
| 01_baseline | 0.250 | 0.500 | 0.917 | 0.900 |
| 02_pymupdf_tables | 0.500 | 0.300 | 0.667 | 1.000 |
| 03_hybrid | 0.375 | 0.600 | 0.667 | 1.000 |
| 04_filter | 0.500 | 0.600 | 0.833 | 1.000 |
| 05_rerank | 0.625 | 0.800 | 0.833 | 1.000 |
| 06_ctx_header | 0.625 | 0.900 | 0.833 | 1.000 |


## Hit@5 by question type

| Config | comparison | factual | numeric | unanswerable |
|---|---|---|---|---|
| 01_baseline | 0.375 | 0.500 | 0.750 | – |
| 02_pymupdf_tables | 0.500 | 0.400 | 0.750 | – |
| 03_hybrid | 0.250 | 0.600 | 0.667 | – |
| 04_filter | 0.500 | 0.600 | 0.833 | – |
| 05_rerank | 0.375 | 0.800 | 0.833 | – |
| 06_ctx_header | 0.375 | 0.900 | 0.583 | – |


## Claim verification accuracy by perturbation

| Config | bank | none | number | unverifiable | year |
|---|---|---|---|---|---|
| 01_baseline | 1.000 | 0.857 | 0.857 | 0.750 | 1.000 |
| 02_pymupdf_tables | 1.000 | 0.714 | 0.714 | 0.875 | 1.000 |
| 03_hybrid | 0.750 | 0.857 | 0.857 | 0.875 | 0.750 |
| 04_filter | 0.750 | 0.857 | 0.857 | 0.875 | 1.000 |
| 05_rerank | 1.000 | 0.857 | 0.857 | 0.875 | 1.000 |
| 06_ctx_header | 1.000 | 0.857 | 0.857 | 0.750 | 1.000 |


_Small eval set: indicative, not statistically conclusive._
