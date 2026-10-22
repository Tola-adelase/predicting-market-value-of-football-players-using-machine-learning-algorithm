# Required original input

Place `players_clean.csv` here: a CSV export of the dissertation's
`PlayerListFinalPickle` **before** feature-dropping cells. This file is absent
from the original public repository. No results have been recomputed yet.

Required columns: Name, Value, Age, Overall, Potential, Int_Reputation,
Height, Weight, Wage, Release_Clause. Optional: ID, Position.

Units must match the original cleaning code: Value, Wage and Release_Clause
in EUR millions; Height in cm; Weight in kg. Do not supply raw currency strings.
Record the dataset source, snapshot/version and export date in a local
`provenance.json` before running. The notebook requires those fields.

Example provenance structure (replace every value with actual provenance):

    {"source": "original SoFIFA dissertation export",
     "snapshot": "actual dataset version or 'unknown historical snapshot'",
     "export_date": "YYYY-MM-DD"}

Only unpickle a file you trust. If you still have your own original pickle,
export it locally using pandas, then use the CSV:

    pd.read_pickle("PlayerListFinalPickle").to_csv("data/players_clean.csv", index=False)

Do not upload or commit the dataset unless its redistribution terms permit it.
The SHA-256 digest is recorded at execution. This is historical game valuation
data, not a real transfer-fee dataset.
