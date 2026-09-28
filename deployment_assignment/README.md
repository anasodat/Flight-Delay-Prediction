# Car Insurance Risk Prediction API

A machine learning project that predicts car insurance risk level (**Low / Medium / High**) based on driver and vehicle attributes, deployed locally using **FastAPI**.

---

## Project Idea

Insurance companies assess risk before setting premiums. This project simulates that process: given basic driver and vehicle data, the model predicts whether the insurance risk is **Low**, **Medium**, or **High**.

---

## Dataset (`data.csv`)

- **120 rows** of synthetic data generated with `generate_data.py`
- **5 features + 1 target column**

| Column | Type | Description |
|---|---|---|
| `age` | int | Driver's age in years (18–65) |
| `driving_experience` | int | Years the driver has been licensed (0–40) |
| `accidents_last_5_years` | int | Number of accidents in the past 5 years (0–3) |
| `vehicle_age` | int | Age of the vehicle in years (0–20) |
| `annual_km` | int | Estimated kilometers driven per year (5,000–80,000) |
| `insurance_risk` | str | **Target** — Low / Medium / High |

---

## ML Model

- **Algorithm**: Random Forest Classifier (`scikit-learn`)
- **Train/Test split**: 80% / 20% (stratified)
- **Test accuracy**: ~79%
- **Saved as**: `model.pkl` (bundles model + label encoder)

---

## Project Structure

```
deployment_assignment/
├── data.csv           ← Synthetic dataset
├── generate_data.py   ← Script to regenerate dataset
├── train_model.py     ← Model training script
├── model.pkl          ← Saved trained model
├── main.py            ← FastAPI application
├── requirements.txt   ← Python dependencies
└── README.md          ← This file
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (generates `model.pkl`)
```bash
python train_model.py
```

### 3. Start the FastAPI server
```bash
uvicorn main:app --reload
```

---

## How to Test the API

Open your browser and go to:
```
http://127.0.0.1:8000/docs
```

Use **Swagger UI** to test the `/predict` endpoint.

### Example Request Body
```json
{
  "age": 28,
  "driving_experience": 5,
  "accidents_last_5_years": 2,
  "vehicle_age": 8,
  "annual_km": 45000
}
```

### Example Response
```json
{
  "risk_level": "High",
  "confidence": 0.87,
  "latency_ms": 3.2
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Predict insurance risk level |
| GET | `/docs` | Swagger UI (interactive testing) |
