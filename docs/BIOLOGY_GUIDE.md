# SmartScalpel – biology & medical guide (presenter's notes)

Written so you can introduce the project and answer medical questions confidently. Each fact is tagged:
**[solid]** = standard textbook / well-established; **[paper]** = from the source study's abstract; **[data]** = verified in our dataset;
**[inference]** = our reasoning, say "consistent with"; **[verify]** = I'm not certain of the exact figure, so quote it loosely or check it first.

---

## 0. Your introduction (about 45 seconds)

> "In breast-conserving surgery the surgeon has to remove the whole tumour but as little healthy breast as possible. The problem is
> that during the operation they can't see the cancer's edge. They rely on touch, imaging and, afterwards, a pathologist, so some
> patients need a second operation. A technique called REIMS analyses the smoke from the electrosurgical knife and reads the
> chemical fingerprint of the tissue in real time. Our project is a simulation of that idea: an iPad shows virtual tissue, the
> pencil acts as the knife, and each region retrieves a real recorded spectrum that our model classifies as tumour, stroma or
> fat, lighting an LED. I'll now hand over to explain how the model works."

---

## 1. The clinical problem

- **Breast-conserving surgery (lumpectomy)** removes the tumour plus a rim of healthy tissue (the **margin**) and keeps the breast. **[solid]**
- **Positive margin** = cancer cells at the cut edge of the removed specimen, found by the pathologist *after* surgery. It usually
  means a **second operation (re-excision)**. Re-excision rates are often quoted at roughly one in five patients, but vary by hospital
  and study. **[verify]** Say "roughly 15-25% in the literature" if asked, not an exact figure.
- **Why it's hard:** tumour edges are irregular, the surgeon works by sight and touch, and the standard check (histopathology) takes
  from tens of minutes (frozen section) to days. **[solid]**
- **The idea behind REIMS-guided surgery:** give the surgeon a tissue read-out in about a second, from the knife they are already using.
  The knife-based version was developed by Zoltan Takats and colleagues (the "iKnife"); our dataset's paper comes from the Maastricht
  group. **[solid]** for the iKnife concept; **[paper]** for Maastricht.

---

## 2. Breast anatomy in one paragraph

The breast is made of **lobules** (milk-producing glands) connected by **ducts**, embedded in **adipose tissue** (fat) and held
together by **stroma** (fibrous connective tissue). Most breast cancers arise from the epithelial cells lining the ducts and
lobules; the commonest type is invasive ductal carcinoma. **[solid]** Our three classes map onto this:

| Class | What it is | Cells / structure |
|---|---|---|
| **Adipose** | Fat | Adipocytes: each cell is mostly one giant fat droplet |
| **Stroma** | Supportive connective tissue | Fibroblasts, collagen fibres, blood vessels, immune cells |
| **Tumour** | Cancer tissue | Densely packed, fast-dividing epithelial cancer cells (plus some stroma inside) |

**[data]** The dataset has three labels only. It does not distinguish cancer subtypes (e.g. hormone-receptor or HER2 status), and the
tumour class is not split into invasive vs in-situ disease.

---

## 3. The molecules: why the three tissues smell chemically different

### 3.1 Lipids are the signal
REIMS in this setting mostly sees **lipids**: fats and membrane molecules that vaporise and ionise easily. **[solid]**

- **Fatty acid** = a carbon chain with a carboxylic acid head. Written `C:D` (carbons : double bonds): palmitic 16:0, stearic 18:0,
  oleic 18:1, linoleic 18:2. More double bonds = more "unsaturated" and kinked. **[solid]**
- **Triacylglycerol (TAG, triglyceride)** = a glycerol backbone with **three** fatty acids attached by ester bonds. It is the body's
  energy storage, kept in the lipid droplet. **[solid]**
- **Phospholipid** = glycerol with **two** fatty acids and a **phosphate head group**. The head group names the class:
  phosphatidylcholine (PC), phosphatidylethanolamine (PE), phosphatidylinositol (PI), phosphatidylserine (PS), phosphatidic acid
  (PA). Phospholipids form the **cell membrane bilayer**: heads face the water, tails face each other. **[solid]**
- **Notation:** `PE 36:2` means a PE whose two fatty-acid chains total 36 carbons and 2 double bonds (for example 18:1/18:1). **[solid]**

### 3.2 Why tissues differ

**Adipose: mostly storage fat.** A mature fat cell is one large lipid droplet, more than 90% of the cell volume, packed with
triacylglycerol and wrapped in a thin shell with few membranes. So its smoke is rich in TAG-derived ions and poor in membrane
phospholipids. **[solid]**
**[data]** 48% of adipose signal sits at m/z 800-1000, versus about 10% for tumour. Its strongest peaks are at m/z 893.8, 919.8 and
891.8 (a TAG-range signature; the exact ion form is not stated in the file **[verify]**).

**Tumour: lots of membrane.** Cancer cells divide fast, so they must build new membranes constantly. Many cancers switch on
**de-novo fatty acid synthesis** (enzymes such as ACLY, ACC and FASN) and **choline/phospholipid synthesis** (choline kinase), and
use **SCD1** to make monounsaturated fatty acids like oleic acid. This "lipogenic" phenotype is well described. **[solid]** Tumour
tissue is also densely cellular, so there is much more membrane per gram than in fat.
**[data]** 58% of tumour signal sits at m/z 600-800, the phospholipid range, versus about 20% for adipose. Top tumour peaks: 699.5,
744.5, 742.5, 725.5. **[inference]** By mass 742.5 and 744.5 match deprotonated PE 36:2 and 36:1, and 699.5 matches PA 36:2.

**Stroma: structural, with free fatty acids.** Stroma is mostly extracellular matrix (collagen, hyaluronan) with fibroblasts and
vessels, so it has relatively few lipids and its profile is dominated by small free fatty acids.
**[data]** 35% of stroma signal sits at m/z 200-400, led by m/z 281.2, 255.2, 279.2 and 283.2. **[inference]** Those masses match
the deprotonated fatty acids oleate (281.25), palmitate (255.23), linoleate (279.23) and stearate (283.26). These peaks are also
present in the other two tissues but are relatively strongest in stroma.

**Why stroma matters clinically:** near a tumour the stroma is remodelled. Fibroblasts become **cancer-associated fibroblasts**,
collagen is laid down densely (the **desmoplastic reaction**), and the stroma's chemistry changes. **[solid]** The source paper's
premise is exactly this: stromal chemistry can guide surgery, and it reports separating tumour-border and remote-tumour stroma
with 86.4% (REIMS) and 87.8% (DESI-MS) accuracy. **[paper]**

### 3.3 One-line takeaway
*Fat is storage lipid (heavy TAG ions), tumour is membrane lipid (mid-range phospholipid ions), stroma is structural with free fatty
acids (light ions). The spectrum is a fingerprint of how much of each lipid family a tissue contains.*

---

## 4. How REIMS actually makes the signal

1. **Electrosurgery.** A generator drives a high-frequency alternating current (hundreds of kHz, which is why it cuts without
   shocking nerves **[solid]**) through the knife tip into tissue.
2. **Joule heating.** Tissue resistance turns the current into heat in a tiny volume. Cell water boils, membranes burst, and
   cell contents vaporise. **[solid]**
3. **Aerosol.** The smoke contains charged droplets of cell lipids and metabolites. This self-ionising process is the "rapid
   evaporative ionisation". No chemical sample preparation is needed. **[solid]**
4. **Transfer.** The aerosol is drawn through a tube into the mass spectrometer.
5. **Mass analysis.** Ions are separated by **m/z** (mass divided by charge; for singly charged ions, just the mass). In
   **negative-ion mode** a molecule that has lost a proton is detected as `[M-H]-`, 1.007 Da lighter than the neutral molecule.
   **[solid]** Detection is typically time-of-flight (light ions arrive first). **[solid]**
6. **The spectrum.** Intensity versus m/z. Our dataset covers m/z 200-1000 in 0.1-unit bins, each spectrum scaled to sum to 1, so it
   records *relative composition*, not absolute amounts. **[data]**

**Resolution of "what is a peak?":** each peak is mostly one lipid species (or a few of the same mass). The exact identification of
each peak would need tandem MS or reference standards; we did not do that, which is why we say "consistent with".

---

## 5. Ex-vivo vs in-vivo and other limits

- **Ex-vivo** = tissue measured after removal from the body. **[paper]** The dataset is ex-vivo, so the lipid profile may differ from
  living tissue (blood flow, temperature, oxygen). The paper also tested one in-vivo surgery. **[paper]**
- **Not seen in the data:** other tissues (muscle, skin, blood), tumour subtypes, benign lesions, treated tissue, other hospitals
  and instruments. **[data]**
- **Burn variability:** how hard and how long the knife pressed changes the spectrum, and each patient contributes between 1 and 28
  spectra. **[data]**
- **Ground truth:** a spectrum is labelled by the tissue it came from. Check the paper's methods on how that was confirmed (usually
  histopathology) before stating it. **[verify]**

---

## 6. Likely questions and answers

**Q: What exactly is REIMS?**
Rapid Evaporative Ionisation Mass Spectrometry. The knife's heat vaporises tissue; the smoke carries lipid ions into a mass
spectrometer, which measures their mass-to-charge ratios, giving a fingerprint of the tissue's lipids in about a second.

**Q: Why lipids and not proteins or DNA?**
Lipids vaporise and ionise easily from the smoke, and they are abundant. Tissues differ strongly in their lipid composition, so
they make good fingerprints. DNA and proteins don't survive or ionise in this kind of aerosol in a form that is easy to read.

**Q: Why is tumour tissue chemically different from fat?**
Fat cells store energy as triacylglycerol in one droplet. Tumour cells are packed with membranes and rewire their metabolism to make
new fatty acids and phospholipids for division, so their spectrum is dominated by membrane lipids instead.

**Q: What is stroma, and why include it?**
The supportive connective tissue: collagen, fibroblasts and vessels. Near tumours it is remodelled, and the source study is about
using that signal. It is also the hardest class because it overlaps with both fat and tumour chemically.

**Q: Why does the model sometimes confuse adipose and stroma?**
Breast stroma often contains fat, and both show free fatty acid peaks (m/z 255, 281). So a fatty stroma spectrum or a fibrous fat
spectrum can look alike. 7 of the 48 held-out adipose spectra were called stroma.

**Q: Would this work in a real operating theatre?**
That is what the original research is working toward, but our project only simulates it. The study tested a single in-vivo case, and
clinical use would need prospective trials, regulatory approval and validation across instruments and hospitals.

**Q: How is this better than current practice?**
Potentially faster: a read-out in about a second at the knife instead of tens of minutes to days for pathology. It would not
replace pathology; it would add real-time guidance. We don't claim it is already better.

**Q: Does it find cancer or tell you where the tumour is?**
No. It classifies one spectrum, i.e. the tissue the knife is currently burning. Where the knife is comes from the surgeon (or
here, the simulation).

**Q: Is the LED measuring tissue?**
No. The LED shows the model's classification of a pre-recorded spectrum selected for the virtual region.

**Q: What would the oncology team worry about?**
False negatives (tumour called healthy), generalisation to new sites, tumour subtypes, and treated tissue. Our held-out tumour
recall is about 94-95%, so roughly 1 in 20 tumour spectra is missed. That is why this is a research simulation.

**Q: What about ethics and data?**
The data are anonymised patient identifiers from a published open dataset. Our work is a simulation, makes no clinical claims, and
discloses that the demo spectra are curated.

**Q: What would you do next?**
Test on independent hospitals' data, add more tissue types and tumour subtypes, identify peaks with tandem MS to explain the model,
and study in-vivo performance.

---

## 7. Glossary

- **Adipocyte:** fat cell. **Stroma:** supportive connective tissue. **Parenchyma:** the functional tissue (here, glandular).
- **Lumpectomy / breast-conserving surgery:** tumour removal that keeps the breast.
- **Margin:** rim of normal tissue around the removed tumour. **Positive margin:** cancer at the edge. **Re-excision:** second operation.
- **Histopathology:** microscope examination of stained tissue by a pathologist.
- **Frozen section:** quick intraoperative histology, takes tens of minutes.
- **Lipid / TAG / phospholipid:** fat family / storage fat / membrane fat.
- **m/z:** mass divided by charge. **Da:** dalton, unit of mass. **[M-H]-:** molecule that lost one proton, seen in negative mode.
- **Electrosurgery (diathermy):** cutting and coagulating tissue with high-frequency current.
- **Ex vivo / in vivo:** outside / inside the living body.
- **DESI-MS:** a related imaging mass spectrometry that scans a tissue slice's surface (used in the same paper for comparison). **[paper]**

---

## 8. Things not to claim

- "REIMS detects cancer cells" (it profiles lipid chemistry; cancer is inferred from the pattern).
- Specific molecule identities as proven; "consistent with" only.
- Any re-excision percentage as exact.
- That the system is a medical device, or has been tested on living patients by us.
- That the LED or pencil senses anything physical.
