# Data

`data/raw/` is git-ignored (large files). Download the 8 PDFs and save them as `data/raw/{ticker}_fy{year}.pdf`
(e.g. `jpm_fy2025.pdf`), then fill in `source_url` for each entry in `manifest.yaml`.

| Bank | CIK | Where to get it |
|---|---|---|
| JPMorgan Chase (`jpm`) | 0000019617 | The 10-K incorporates the Annual Report by reference: download the full **Annual Report PDF** from the investor-relations site. |
| Bank of America (`bac`) | 0000070858 | 10-K PDF from investor.bankofamerica.com |
| Citigroup (`c`) | 0000831001 | 10-K PDF from citigroup.com investor relations |
| Wells Fargo (`wfc`) | 0000072971 | Financials are in the **Annual Report (Exhibit 13)**: download the Annual Report PDF. |

Verify each note when downloading (e.g. that the PDF really contains the financial statements and MD&A).
Fiscal years: 2024 and 2025 for each bank (8 files).
