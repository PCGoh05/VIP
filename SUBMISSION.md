# CDS6334 Submission Checklist

## Main Files for Review

- Main notebook: `notebooks/CDS6334_Plant_Disease_Recognition_Project.ipynb`
- Streamlit app: `app/streamlit-app.py`
- README and run guide: `README.md`

The final Word report is kept outside GitHub and should be submitted separately through the course submission platform or shared with teammates through the group drive.

## Demo Links

- Streamlit demo: https://vip-plant-disease-recognition.streamlit.app/
- GitHub repository: https://github.com/PCGoh05/VIP

## Output Folders

- `outputs/models/`: trained Keras models and class names
- `outputs/figures/`: class distribution, training curves, confusion matrices and Grad-CAM examples
- `outputs/reports/`: model comparison table, split summaries and classification reports

## Notes

- The dataset images are not committed to GitHub because they are large.
- Word report drafts are not committed to GitHub to keep the repository clean.
- The notebook is safe to open in review mode because all `RUN_*` switches are set to `False` by default.
- To regenerate outputs, change the relevant `RUN_*` switch in the notebook to `True`.
- Do not turn on training switches unless you are ready to retrain and overwrite model outputs.
