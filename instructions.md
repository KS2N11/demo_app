Tasks:
1) I want to create a new page for forecasting and prediction. Here are some details/references for the same:

🔹 Features to Provide on Forecasting & Prediction Page
1. Forecasting (Trends & Projections)

Based on the uploaded data, provide:

📈 Enrollment Forecast

Predict how many participants will enroll per week/month.

Show best-case, worst-case, and expected trendlines.

📉 Dropout Forecast

Predict future dropout rates (e.g., “By Month 6, we expect ~15% dropout”).

🌍 Site-level Forecasts

Forecast enrollment and retention by trial site or region.

📊 Visualization

Line charts with projected values + confidence intervals.

Option to toggle between different forecast horizons (e.g., 1 month, 3 months, 6 months).

2. Prediction (User-driven “What If” Scenarios)

Allow users to ask: “What will happen if…”

🔮 Custom Predictions

User inputs assumptions like:

“Increase recruitment budget by 20%”

“Add 2 new sites in urban areas”

“Improve logistics (reduce travel time)”

Model recalculates expected enrollment/dropout.

🧾 Written Insights

AI summarizes forecasts:

“If recruitment budget is increased by 20%, predicted enrollment reaches target 3 weeks earlier, with a 10% reduction in dropout.”

3. Interactivity

📤 Upload Data → Forecast: Automatically detect time-series columns (e.g., enrollment by week).

📝 Ask Questions: Natural language queries like:

“Predict participant numbers for next 3 months.”

“What happens if dropout increases by 5%?”

🔁 Compare Scenarios: Side-by-side comparison of baseline vs. custom prediction.

🔹 Technologies to Use
Forecasting (Time-Series)

Classical Models (Quick & Reliable):

statsmodels ARIMA / SARIMA → good for small datasets.

Modern ML Models:

Facebook Prophet → easy, interpretable, great for business forecasts.

scikit-learn regression models (Linear/Random Forest) for numeric predictions.

Advanced (if you want to showcase AI muscle):

LSTM / GRU (deep learning models for time series) with TensorFlow or PyTorch.

But for a demo, Prophet or ARIMA is more than enough.

Prediction (Scenario Simulation)

ML Models:

Train simple classification/regression models:

Logistic Regression → Predict dropout likelihood (yes/no).

Random Forest → Predict probability of retention.

Allow user to adjust input features (budget, site count, demographics) and re-run predictions.

AI Summaries:

Use Azure OpenAI (or GPT-4) to:

Convert numeric results into plain English insights.

Explain “why” the forecast changed.

Visualization

Plotly / Streamlit native charts → for interactive forecasts with sliders.

Confidence intervals for forecasts (shaded bands).

Scenario comparison charts.

🔹 Suggested Workflow for Forecasting Page

Upload Data → detect time dimension (date, week, month).

Baseline Forecast → Prophet/ARIMA predicts future enrollment/dropouts.

Visualization → Show trendlines + confidence intervals.

Written Summary → LLM explains key takeaways.

Custom Prediction Section:

User adjusts assumptions (sliders/dropdowns).

Model recalculates future outcomes.

Compare baseline vs. adjusted scenario (charts + text).

🔹 Example User Flow (Demo Script)

👩‍💻 Upload dataset →
📊 Dashboard shows “Enrollment so far: 300, Dropouts: 40” →
🔮 Forecast tab: “If current trend continues, trial will reach 500 participants in 6 months with ~18% dropout risk.” →
📝 User types: “What if we add 2 new sites?” →
📈 New chart generated showing faster recruitment, with AI summary:

“Adding 2 new sites reduces time-to-target by 4 weeks and lowers dropout risk by 5%.”



But make sure that the models do not expect a fixed format and fixed columns, they should be able to accept any csv data. We want to provide some flexibility. This page should be something that is a game changer for the people working in this industry and actually helps solve their real-world problems. Keep this in mind when creating this page. 
You need not strictly stick to the reference details provided, they are for your reference.

Make sure to keep the architecture of this page robust and in line with the rest of the application.