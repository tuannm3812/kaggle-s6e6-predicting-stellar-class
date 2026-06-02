import json

# Load notebook
path = "notebooks/01_eda.ipynb"
print(f"Loading {path}...")
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Markdown content for the takeaways section
takeaways_markdown = """## 8. Strategic Key Takeaways & Modeling Guidelines

Based on our exploratory data analysis, we establish the following strategic directives for our modeling pipeline:

1. **Address Severe Class Imbalance (Metric Alignment)**
   * **Observation:** The target classes are heavily skewed (`GALAXY`: ~65.4%, `QSO`: ~20.3%, `STAR`: ~14.3%).
   * **Guideline:** Since our target metric is **Balanced Accuracy**, standard cross-entropy loss will overfit the majority class. We must apply inverse class weights. A misclassified `STAR` is **~4.5x** more costly than a misclassified `GALAXY`. We will configure `class_weight='balanced'` in LightGBM, `auto_class_weights='Balanced'` in CatBoost, and custom sample weights in XGBoost.

2. **Handle Redshift Cosmological Artifacts (Unphysical Shifts)**
   * **Observation:** Over 75% of the samples labeled as `STAR` have redshift $z > 0.02$, with a maximum of $5.44$ (which is physically impossible for Milky Way stars).
   * **Guideline:** The model cannot rely on a simple redshift cutoff to isolate stars. It must learn to combine redshift with color indices and spectral codes to resolve synthetic data noise.

3. **Leverage Photometric Color Indices**
   * **Observation:** Absolute magnitudes ($u, g, r, i, z$) depend on object distance and absolute luminosity. Color indices (differences between adjacent filter bands) act as distance-invariant estimators of radiation curves.
   * **Guideline:** We will engineer all 10 color index differences ($u-g, g-r, r-i, i-z, u-r, g-i, r-z, u-i, g-z, u-z$) to map physical surface properties and emission line signatures.

4. **Prevent Coordinate Wrap-Around Boundaries**
   * **Observation:** Right Ascension (`alpha`) is a circular longitude ($0^\\circ \\to 360^\\circ$), where $359.9^\\circ$ is spatially adjacent to $0.1^\\circ$. Decision tree splits treat these boundary extremes as opposites.
   * **Guideline:** We project celestial coordinates ($\\alpha, \\delta$) into 3D Cartesian coordinates ($x, y, z$) on a unit sphere to resolve boundary discontinuities."""

# Split lines with trailing newlines
source_lines = [line + "\n" for line in takeaways_markdown.strip().split("\n")]
if source_lines:
    source_lines[-1] = source_lines[-1].rstrip("\n")

# Create cell object
new_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": source_lines
}

# Append cell
data["cells"].append(new_cell)

# Write back
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)

print(f"Successfully appended Key Takeaways cell to {path}!")
