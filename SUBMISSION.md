# Draft text for the hackathon form (paste, edit)
**Project submission description**
regime_audit finds silent label-regime changes in time-stamped tabular data. It fits a time-free TabPFN-3.5 model out of fold, scans every
date in every automatically discovered subgroup for a jump in the residuals, and ranks the breaks by a z-score. Because TabPFN is in-context,
it also tests a free intervention: choosing the context rows by regime. On 3,146 verbal-autopsy deaths from the Iganga-Mayuge HDSS (Uganda,
2007-2022) it recovers, from age/sex/zone alone, an abrupt change in how infant deaths are classified as climate-sensitive on 2017-06-06
(positive rate 0.97 -> 0.56, z = -17.5). Using only post-break context rows cuts held-out log-loss from 0.785 to 0.675 and fixes calibration
(mean prediction 0.594 vs true 0.586, instead of 0.800) with no retraining. A planted-break test and a no-download synthetic demo are
included. TabPFN-3.5 is the core model throughout. Apache-2.0.
