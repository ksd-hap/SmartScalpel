# SmartScalpel – ML presentation & defence guide

Everything below is checked against the real data, the saved model and the published paper. Where something is my interpretation
(not stated in the data or paper) it is marked **[inference]** so you never present a guess as a fact.

---

## 0. The 30-second pitch

> "Surgeons need to know what tissue they are cutting. A technique called REIMS analyses the smoke from an electrosurgical knife
> and produces a chemical fingerprint of the tissue. We took 689 real recorded fingerprints from 72 patients, trained a classifier
> to tell tumour, stroma and adipose apart, and validated it so that no patient is ever in both training and test data.
> It reaches about 91% accuracy, in line with the 92.3% reported by the original study. Our simulation replays real recorded
> spectra through that model and drives an LED. It is a simulation, not a medical device."

---

## 1. The science, down to the molecule

**Step by step, from tissue to numbers**

1. **Electrosurgery.** An electrosurgical knife passes high-frequency current through tissue. Resistance heats the cells; water
   boils, cell membranes rupture, and the tissue is vaporised.
2. **Aerosol.** The smoke ("surgical plume") contains the cell's contents: intact **lipids** (fats) and other small
   molecules, as charged droplets and ions. This is the "evaporative ionisation" in REIMS: *Rapid Evaporative Ionisation Mass
   Spectrometry*. No sample preparation is needed.
3. **Transport.** The smoke is sucked down a tube into a mass spectrometer.
4. **Mass spectrometry.** Ions are accelerated and separated by their **mass-to-charge ratio (m/z)**. Heavier ions arrive later
   (time-of-flight). A detector counts how many ions arrive at each m/z.
5. **The spectrum.** The result is a plot: x = m/z, y = how much signal at that m/z. Each peak is (mostly) one molecule type.
   One spectrum = one burn = one data point for us.

**Why different tissues give different spectra: the molecules**

| Tissue | Dominant chemistry | Where it shows up in YOUR data |
|---|---|---|
| **Adipose (fat)** | Fat cells are mostly **triacylglycerols** (3 fatty acids on a glycerol backbone, the body's energy store), heavy molecules | **48% of adipose signal sits at m/z 800-1000** vs 10% for tumour. Strongest adipose peaks m/z 893.8, 919.8, 891.8 |
| **Tumour** | Fast-growing cells have lots of membrane **phospholipids** (phosphatidylethanolamine etc.) | **58% of tumour signal sits at m/z 600-800** vs 20% for adipose. Top peaks 699.5, 744.5, 742.5, 725.5 |
| **Stroma** (connective tissue around the tumour) | Mix of free **fatty acids** and membrane lipids | **35% of stroma signal at m/z 200-400**, led by m/z 281.2, 255.2, 279.2, 283.2 |

**[inference] Peak identities.** The file contains only numbers, no annotations. By mass these match standard negative-ion
lipid assignments: 255.2 ≈ palmitic acid, 279.2 ≈ linoleic acid, 281.2 ≈ oleic acid, 283.2 ≈ stearic acid (all fatty-acid
anions); 742.5 and 744.5 ≈ phosphatidylethanolamines (PE 36:2 and 36:1); 893-920 is the triacylglycerol range. Present these as
"consistent with", not as verified identifications. The honest line: *"the model learns from the whole spectral pattern, not from
named molecules."*

**Why stroma matters (and why we keep it):** the source paper is titled *"Stromal vapors for real-time molecular guidance of
breast-conserving surgery"* (Vaysse et al., Sci Rep 2020). Its point is that the stroma around a tumour changes chemically, so
it helps guide a surgeon. That is why stroma is a legitimate third class.

---

## 2. The data

| Fact | Value (verified in code) |
|---|---|
| Source | Vaysse et al., *Scientific Reports* 10:20109 (2020), supplementary file 3, sheet `MAM_REIMS_A_S_T.data` |
| Spectra / patients | **689 spectra from 72 patients** |
| Features per spectrum | **8,000** m/z bins, from 200.05 to 999.95 in steps of **0.1** |
| Classes | tumour 224 spectra (50 patients), stroma 209 (47 patients), adipose 256 (55 patients) |
| Spectra per patient | 1 to 28 (very uneven) |
| Normalisation | **Every spectrum sums to 1.0** (total-ion-count normalised), so values are each m/z's *share* of the signal, not absolute amounts |
| Missing / invalid values | none; 5.3% of values are exactly zero (no ions at that bin) |
| Origin | **Ex-vivo** (tissue removed from the body), per the paper |

Key facts for defending the data:
- **Each patient contributes several spectra** (different burns/locations). That is why patient leakage is the central risk (§4).
- The file is **already processed** by the original authors, so we did not preprocess beyond standardising channels in the model.
- Labels are tissue types from the study. **[check]** I could read the abstract but not the full text, so confirm in the paper
  how ground truth was established (the usual standard is histopathology) before stating it as fact.

---

## 3. The model, step by step

`StandardScaler -> PCA(100 components) -> Linear Discriminant Analysis (shrinkage) -> temperature-scaled scores -> rejection threshold`

**Why this design, in plain terms:** we have **8,000 numbers per spectrum but only 529 training spectra** (54 patients). With more
features than samples, flexible models memorise noise. So we shrink the problem first, then use a simple, hard-to-overfit
classifier.

1. **Standardise (StandardScaler).** For each m/z channel, subtract its training mean and divide by its training standard
   deviation. Without this, big peaks drown out small but informative ones. Statistics come from training patients only.
2. **PCA to 100 components.** Many channels move together (peaks of the same molecule family rise and fall together). PCA finds
   the directions of greatest variation and keeps 100 of them: **89.4% of total variance**, with the first component alone 26.3%.
   This compresses 8,000 correlated numbers into 100 uncorrelated ones and discards mostly noise.
3. **LDA.** Models each class as a cloud of points with its own centre but a **shared spread**, then draws straight-line
   boundaries between the clouds. A new spectrum goes to the class whose cloud it is most likely from. It is linear, fast
   (about 1 ms) and has few parameters, so it is hard to overfit.
4. **Shrinkage.** Estimating the shared spread from limited data is noisy. Shrinkage blends the estimated covariance with a
   plain, spherical one (automatic Ledoit-Wolf weight). It gave a small gain over plain LDA (about +1 point) and made the
   scores less over-confident.
5. **Scores.** Plain LDA scores were wildly over-confident (0.9999998). We fitted a "temperature" on out-of-fold training data to
   rescale them. For the final model it came out at **~1.0**, meaning the shrinkage LDA scores needed essentially no softening.
   These are **model scores, not clinical probabilities**.
6. **Rejection.** If the top score is below **0.77**, the result is "uncertain" (LED off). The threshold was chosen on
   out-of-fold training scores only, never on test data.

**Does confidence mean anything? (held-out set)**

| Top score | Spectra | Actual accuracy |
|---|---|---|
| below 0.6 | 12 | 42% |
| 0.6-0.77 | 21 | 86% |
| 0.77-0.9 | 18 | 83% |
| 0.9-0.99 | 104 | 99% |
| 0.99 and up | 5 | 100% |

Low confidence really does mean "more likely wrong". The model is slightly *under*-confident overall (mean score 87.5% vs
91.2% accuracy), which is the safe direction.

**Why not deep learning?** Only 72 patients. Neural nets need far more data and would likely overfit and be harder to defend. We
compared simpler alternatives (SVM, logistic regression) by nested cross-validation and LDA won or tied, so extra complexity was
not justified. You can say that honestly.

---

## 4. Validation: how we know it is real

**The trap: patient leakage.** If you randomly split spectra, spectra from the *same patient* land in both train and test. The
model can then recognise the patient (their individual biology) instead of the tissue, and scores look inflated. Spectra from one
patient are highly similar.

**What we did instead**
1. **Patient-grouped split:** all spectra of a patient go entirely to train or entirely to test. 54 patients train, **18 patients
   (160 spectra) test**, zero overlap (asserted in code and tests).
2. **Model choice inside training only:** grouped 5-fold CV on the 54 training patients selected the model, PCA size, temperature
   and threshold. The test patients were never used for any decision.
3. **Nested grouped CV over all 72 patients** (6 outer folds, selection repeated inside each fold) for a steadier estimate.
4. **Controls:** shuffled labels give 33.8% (chance); always guessing the biggest class gives 40.6%.

**Results**

| Measure | Result |
|---|---|
| Nested grouped CV, all 72 patients (headline) | **91.1% accuracy**, balanced 91.0%, macro F1 0.910, fold-to-fold spread about +/-2 points |
| Per-class recall (nested) | tumour **95.1%**, stroma 86.6%, adipose 91.4% |
| Fixed hold-out, 18 unseen patients | 91.2% accuracy, tumour recall 93.8%, stroma 93.6%, adipose 85.4% |
| Original paper | **92.3%**, leave-one-patient-out, same 689 spectra from 72 patients |

**How to compare with the paper:** same data and the same patient-separation principle; the paper's number is slightly higher;
ours is within about a point, and our validation differs (6 outer folds with selection nested inside, vs leave-one-patient-out).
Do not claim to beat the paper. Say we **reproduced the same level of performance independently**.

**Where it errs** (hold-out): most errors are adipose called stroma (7 of 48). Stroma is the most mixed tissue (blood vessels,
fibres, some fat), so it is the hardest class.

---

## 5. Hard questions, with defensible answers

**Q: Isn't 91% too good / is the model cheating?**
No: patients never overlap between train and test; shuffled labels give chance (33.8%); and the figure matches the original
study's 92.3%.

**Q: Why not 99%?**
The tissue classes overlap chemically (stroma contains fat and tumour-adjacent cells), there are only 72 patients, and each patient
differs biologically. About 9% error is realistic for this data, as the paper itself shows.

**Q: Is this clinically usable?**
No, and we do not claim it. It is ex-vivo, from one published dataset, 72 patients, with no in-hospital validation. It is a
simulation.

**Q: Does the Apple Pencil detect cancer?**
No. The pencil only sends a position. The backend picks a real recorded spectrum for that virtual region; the model classifies that
spectrum; the LED shows the result. Nothing physical is sensed.

**Q: Isn't it circular? You choose a tumour spectrum for a tumour region, then "predict" tumour.**
The region type selects *which spectrum to show*. The model never sees the region type or the true label. It sees 8,000 numbers
and decides alone, and it is sometimes wrong or uncertain. We say openly that demo spectra are curated (see below).

**Q: Is the demo cherry-picked?**
Partly, and we disclose it. Demo "typical" spectra are held-out ones the model gets right; we also include "uncertain" ones
where it hesitates. Real performance is the 91% figure, not the demo.

**Q: Why PCA, why LDA, why not a neural network?**
8,000 features vs ~500 training samples: reduce dimensions first, then use a low-variance linear classifier. Neural networks
overfit on this size of data, and SVM and logistic regression did not beat LDA in nested CV.

**Q: What does the LED not show?**
Stroma and uncertain both show "off", so the iPad must display the class name. That is a hardware limitation of two colours.

**Q: Why does the model reject some spectra?**
When the top score is below 0.77 it says "uncertain". Those spectra are much less accurate (42-86%), so refusing to answer is
safer than guessing. Lighting the LED is therefore very reliable (red was right 53 of 53 times on the hold-out), at the cost of
about 15% of real tumour/adipose spectra staying dark.

**Q: Could it work on a new hospital's data?**
Unknown. Different instruments and protocols shift spectra, so it would need re-validation. That is future work.

**Q: How did you handle class imbalance?**
Classes are roughly balanced (224/209/256). We report balanced accuracy, and class-weighted variants gave no meaningful gain.

**Q: Are the scores probabilities?**
No. They are model scores that we calibrated with temperature scaling; they are not clinical probabilities. The check above shows
they track real accuracy reasonably.

**Q: What would you do with more time/data?**
More patients and a second centre; per-patient normalisation; peak identification to explain the model; real-time acquisition
on the instrument; prospective clinical validation.

**Q: How do you know it wasn't tuned on the test set?**
Test patients were set aside before any model comparison; all choices used grouped CV inside the training patients; the nested CV
repeats that selection inside every fold. Anyone can rerun `py -m ml.validate` to reproduce the numbers.

---

## 6. Things NOT to say

- "It detects cancer" / "it finds the tumour": it classifies one recorded spectrum.
- "The pencil senses tissue" or "the LED measures tissue."
- "Clinically validated" or "diagnostic".
- "Beats the published model": ours is comparable, not better.
- "99% accurate" / quoting the demo set as performance.
- Naming specific molecules as proven: say "consistent with".

---

## 7. Cheat sheet (memorise these)

- **689 spectra, 72 patients, 8,000 features (m/z 200-1000, 0.1 step)**, 3 classes
- **Pipeline:** standardise -> PCA(100, keeps 89% variance) -> shrinkage LDA -> reject below 0.77
- **Split:** by patient: 54 train / 18 test patients, no overlap
- **Headline:** 91.1% accuracy, tumour recall 95.1% (nested patient-grouped CV); paper: 92.3%
- **Controls:** shuffled labels 33.8%, majority class 40.6%
- **Speed:** about 1 ms per spectrum
- **Rejection:** 0.77, about 21% of held-out spectra rejected; accepted ones 96.9% accurate
- **Disclaimer:** simulation, ex-vivo research data, not clinical

## 8. How to re-check everything yourself

```bash
py -m ml.validate      # all the numbers above, in seconds
py -m ml.cv_report     # recompute the nested CV (about 1.5 min)
py -m pytest tests -v  # 11 tests
```
