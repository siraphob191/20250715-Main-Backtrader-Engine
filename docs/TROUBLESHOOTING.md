# Troubleshooting

## PyFolio TypeError: unexpected keyword argument 'gross_lev'

**Symptom:**  
TypeError: create_full_tear_sheet() got an unexpected keyword argument 'gross_lev'

**Root Cause:**  
The `create_full_tear_sheet` function signature changed; it no longer accepts a `gross_lev` argument.

**Resolution:**  
1. In `src/utils/report_io.py`, remove `gross_lev=gross_lev` from the tear-sheet call.  
2. If gross leverage plots are required, generate them separately:
   ```python
   from pyfolio import plotting
   plotting.plot_leverage(returns, positions, transactions)
   ```
