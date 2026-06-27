# Lab 2B: Deploy a Real-Time Prediction Service in the New Azure ML Studio

## Goal of This Lab

In this lab, you will deploy a trained diabetes prediction model as a real-time web service using the modern Azure Machine Learning Studio experience.

By the end of this lab, you will be able to:

- Find a trained model in Azure ML Studio
- Create a Managed Online Endpoint
- Deploy a model to the endpoint
- Test the endpoint from Azure ML Studio
- Call the endpoint from a Python notebook
- Monitor and delete the endpoint to avoid charges

---

## Important Change from the Old DP-100 Lab

The old lab used this flow:

```text
Designer Training Pipeline
        ↓
Create Inference Pipeline
        ↓
Deploy to AKS
        ↓
Test Web Service
```

The new Azure ML Studio uses this modern flow:

```text
Trained Model
        ↓
Registered Model
        ↓
Managed Online Endpoint
        ↓
Deployment
        ↓
REST API Prediction Service
```

In the new Azure ML Studio, you usually deploy models using **Managed Online Endpoints**, not by manually creating an AKS inference cluster.

---

## Prerequisites

Before starting this lab, make sure you have completed the previous lab where you trained and registered a diabetes prediction model.

You should already have:

- An Azure Machine Learning workspace
- A compute instance
- A trained diabetes model
- A registered model in Azure ML Studio

Suggested model name:

```text
diabetes-model
```

If your model has a different name, use your own model name throughout this lab.

---

# Task 1: Open Azure ML Studio

1. Go to:

```text
https://ml.azure.com
```

2. Sign in with your Azure account.

3. Select your:

```text
Directory
Subscription
Workspace
```

4. Make sure you are inside the correct Azure Machine Learning workspace.

---

# Task 2: Verify That Your Model Exists

1. In Azure ML Studio, from the left menu, click:

```text
Models
```

2. Look for your registered model.

Example:

```text
diabetes-model
```

3. Click the model name.

4. Verify that the model has at least one version.

Example:

```text
Version 1
```

5. Review the model details.

You should confirm:

- Model name
- Model version
- Model type
- Creation date

If you do not see your model, go back to Lab 2A and make sure the model was trained and registered successfully.

---

# Task 3: Create a Managed Online Endpoint

An endpoint is the public prediction service. Applications send data to the endpoint and receive predictions back.

1. From the left menu, click:

```text
Endpoints
```

2. Click:

```text
+ Create
```

3. Choose:

```text
Online endpoint
```

4. Select:

```text
Managed online endpoint
```

5. Configure the endpoint:

| Setting | Value |
|---|---|
| Endpoint name | predict-diabetes |
| Authentication type | Key-based authentication |
| Description | Real-time diabetes prediction endpoint |

6. Click:

```text
Next
```

7. Continue until the endpoint creation screen is complete.

8. Click:

```text
Create
```

Wait until the endpoint is created.

---

# Task 4: Deploy the Model to the Endpoint

The endpoint is only the door. The deployment is the actual model running behind that door.

1. Open your endpoint:

```text
Endpoints → predict-diabetes
```

2. Click:

```text
+ Add deployment
```

or

```text
Deploy model
```

3. Select:

```text
Existing registered model
```

4. Choose your model.

Example:

```text
diabetes-model
```

5. Select the model version.

Example:

```text
Version 1
```

6. Configure the deployment:

| Setting | Value |
|---|---|
| Deployment name | blue |
| Virtual machine size | Standard_DS3_v2 |
| Instance count | 1 |
| Traffic | 100% |

7. For scoring method, use the option that matches your model.

If your model was logged with MLflow, choose:

```text
MLflow model
```

8. Click:

```text
Deploy
```

9. Wait until the deployment status becomes:

```text
Healthy
```

Do not continue until the deployment is healthy.

---

# Task 5: Test the Endpoint in Azure ML Studio

1. Open:

```text
Endpoints → predict-diabetes
```

2. Click the:

```text
Test
```

tab.

3. Paste the following sample JSON input:

```json
{
  "input_data": {
    "columns": [
      "Pregnancies",
      "PlasmaGlucose",
      "DiastolicBloodPressure",
      "TricepsThickness",
      "SerumInsulin",
      "BMI",
      "DiabetesPedigree",
      "Age"
    ],
    "data": [
      [9, 104, 51, 7, 24, 27.36, 1.35, 43],
      [6, 73, 61, 35, 24, 18.74, 1.07, 75],
      [4, 115, 50, 29, 243, 34.69, 0.74, 59]
    ]
  }
}
```

4. Click:

```text
Test
```

5. Review the result.

You may see output similar to:

```json
[0, 0, 1]
```

Your output may be different depending on how your model was trained.

---

# Task 6: Understand the Prediction Output

The model may return values like:

| Output | Meaning |
|---|---|
| 0 | Patient is predicted as not diabetic |
| 1 | Patient is predicted as diabetic |

Some models may return probabilities as well.

Example:

```json
[
  {"prediction": 0, "probability": 0.31},
  {"prediction": 1, "probability": 0.82}
]
```

The exact output depends on your training and scoring script.

---

# Task 7: Get the Endpoint URL and Key

1. Open:

```text
Endpoints → predict-diabetes
```

2. Click the:

```text
Consume
```

tab.

3. Copy the following values:

```text
REST endpoint URL
Primary key
```

You will use these in your Python notebook.

Example endpoint format:

```text
https://predict-diabetes.<region>.inference.ml.azure.com/score
```

Do not share your key publicly.

---

# Task 8: Open a Notebook

1. From the left menu, click:

```text
Notebooks
```

2. Open your lab folder.

Example:

```text
DP-100
```

3. Create a new notebook named:

```text
Lab_2B_Test_Endpoint.ipynb
```

4. Make sure your notebook is connected to a running compute instance.

If the compute instance is stopped, start it first.

---

# Task 9: Test the Endpoint from Python

In the first code cell, paste the following code.

Replace:

```text
<PASTE-YOUR-ENDPOINT-URL-HERE>
<PASTE-YOUR-PRIMARY-KEY-HERE>
```

with your actual endpoint URL and primary key.

```python
import requests
import json

endpoint_url = "<PASTE-YOUR-ENDPOINT-URL-HERE>"
api_key = "<PASTE-YOUR-PRIMARY-KEY-HERE>"

payload = {
    "input_data": {
        "columns": [
            "Pregnancies",
            "PlasmaGlucose",
            "DiastolicBloodPressure",
            "TricepsThickness",
            "SerumInsulin",
            "BMI",
            "DiabetesPedigree",
            "Age"
        ],
        "data": [
            [9, 104, 51, 7, 24, 27.36, 1.35, 43]
        ]
    }
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

response = requests.post(endpoint_url, headers=headers, json=payload)

print("Status Code:", response.status_code)
print("Response:")
print(response.text)
```

Run the cell.

A successful request should return:

```text
Status Code: 200
```

---

# Task 10: Test Multiple Patients

Create a new code cell and paste this code:

```python
payload = {
    "input_data": {
        "columns": [
            "Pregnancies",
            "PlasmaGlucose",
            "DiastolicBloodPressure",
            "TricepsThickness",
            "SerumInsulin",
            "BMI",
            "DiabetesPedigree",
            "Age"
        ],
        "data": [
            [9, 104, 51, 7, 24, 27.36, 1.35, 43],
            [6, 73, 61, 35, 24, 18.74, 1.07, 75],
            [4, 115, 50, 29, 243, 34.69, 0.74, 59]
        ]
    }
}

response = requests.post(endpoint_url, headers=headers, json=payload)

print("Status Code:", response.status_code)
print("Response:")
print(response.text)
```

Run the cell.

You should receive predictions for three patients.

---

# Task 11: Student Challenge

Modify the patient values and test again.

Try changing:

- PlasmaGlucose
- BMI
- Age
- DiabetesPedigree

Answer the following questions:

1. Which patient is most likely to be predicted as diabetic?
2. What happens when BMI increases?
3. What happens when PlasmaGlucose increases?
4. Does age appear to affect the prediction?
5. Why should we not blindly trust the model output?

---

# Task 12: Monitor the Endpoint

1. Go back to Azure ML Studio.

2. Open:

```text
Endpoints → predict-diabetes
```

3. Review the endpoint details.

4. Check:

```text
Metrics
```

Look for:

- Requests
- Latency
- Failed requests
- Successful requests

5. Open the deployment:

```text
Deployments → blue
```

6. Review logs if available.

Logs are useful when the endpoint fails.

---

# Task 13: Common Errors and Fixes

## Error: 401 Unauthorized

Possible reason:

```text
Wrong API key
```

Fix:

- Go to the Consume tab
- Copy the primary key again
- Make sure the Authorization header is correct

```python
"Authorization": f"Bearer {api_key}"
```

---

## Error: 404 Not Found

Possible reason:

```text
Wrong endpoint URL
```

Fix:

- Copy the endpoint URL again from the Consume tab
- Make sure you are using the scoring URL

---

## Error: 500 Internal Server Error

Possible reasons:

- Model scoring script failed
- Input data format is wrong
- Column names do not match the training data

Fix:

- Check deployment logs
- Compare your JSON input with the expected schema
- Make sure all required columns are included

---

## Error: Deployment Not Healthy

Possible reasons:

- Environment failed to build
- Model loading failed
- Compute size is unavailable

Fix:

- Open deployment logs
- Try redeploying
- Try a different VM size

---

# Task 14: Clean Up Resources

This step is very important. Online endpoints can create Azure charges.

1. Go to:

```text
Endpoints
```

2. Select:

```text
predict-diabetes
```

3. Click:

```text
Delete
```

4. Confirm deletion.

5. Go to:

```text
Compute
```

6. Stop your compute instance if you are finished.

7. Delete any compute clusters you do not need.

---

# Final Lab Checklist

Before you finish, confirm that you completed the following:

| Task | Completed |
|---|---|
| Found registered model | ☐ |
| Created managed online endpoint | ☐ |
| Deployed model as blue deployment | ☐ |
| Tested endpoint in Azure ML Studio | ☐ |
| Copied endpoint URL and key | ☐ |
| Tested endpoint from Python notebook | ☐ |
| Tested multiple patient records | ☐ |
| Reviewed metrics/logs | ☐ |
| Deleted endpoint | ☐ |
| Stopped compute instance | ☐ |

---

# Reflection Questions

Answer these in your notebook or lab document.

1. What is the difference between a model and an endpoint?
2. What is the difference between an endpoint and a deployment?
3. Why do we use a key when calling the endpoint?
4. Why is it important to test the endpoint before giving it to an application developer?
5. Why should we delete unused endpoints?
6. What could go wrong if the input columns do not match the training columns?

---

# Summary

In this lab, you deployed a trained machine learning model as a real-time prediction service using the modern Azure ML Studio.

You learned the current Azure ML workflow:

```text
Register Model
      ↓
Create Managed Online Endpoint
      ↓
Deploy Model
      ↓
Test Endpoint
      ↓
Consume from Python
      ↓
Monitor and Clean Up
```

This is the modern replacement for the older DP-100 Designer inference pipeline and AKS deployment lab.
