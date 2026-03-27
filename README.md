# <tt>umbra</tt>

Code for performing least-squares transit searches in time-series photometry.
<tt>umbra</tt> implements versions of Box Least-Squares (BLS), Transit Least-Squares
(TLS) and Warped Least-Squares (WLS) searches. WLS is a new type of search that 
applies the effect of simple variability filters to the the transit shape 
templates, improving recovery rates for some filters. 

# Usage

A basic usage example is provided in the examples/simple-example.ipynb

# Change log
- v1.0.0 (upcoming) This release including major implementation updates and improvements to the user experience. It is recommended to wait for this version.
    - Major re-factoring, improvements to documentation, and input verification. 
    - Implementation of [Ofir (2014)](https://ui.adsabs.harvard.edu/abs/2014A%26A...561A.138O/abstract) period grid.
    - Implementation of [Talens et al. (2024)](https://ui.adsabs.harvard.edu/abs/2025RNAAS...9..319T/abstract) duration limits.
    - Option to search the full eccentric duration range.
    - Simplified duration grid to use the same log-spaced durations at all periods.
    - Clarified epoch sampling.
    - Added further example notebooks.
- v0.7.1 (2025-03-27) More user-friendly version published alongside Talens et al. (in prep). Can be used by the interested user.
    - Minor changes and fixes.
    - Renamed the code and repo to umbra.
    - Added a simple example notebook.
- v0.7.0 (2025-10-17)
    - Minor change to the duration grid.
    - Implements options at short periods where WLS templates contain multiple transits.
        - Skip short periods.
        - Use TLS templates at short periods.
        - Full evaluation of WLS templates (slow, not recommended).
- v0.6.0 (2025-08-04)
    - Major re-factor of template generation.
    - Returned transit depths are no longer scaled by an arbitrary factor.
- v0.5.0 (2025-08-04)
    - Implements BLS and TLS transit templates.
- v0.4.0 (2025-08-04)
    - Updates to YSD-Lowess implementation to improve peak/trough detection.
- v0.3.0 (2025-07-04)
    - Implements tricube weighting for WLS templates.
- v0.2.0 (2025-04-02)
    - Oldest tagged version.