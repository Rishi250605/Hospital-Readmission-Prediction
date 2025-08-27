Use Case: Predictive Modeling for Hospital Readmission Risk
The use case is centered on a critical challenge in the healthcare industry: unplanned patient readmissions. A readmission occurs when a patient who has been discharged from a hospital has to be admitted again within a short period (typically 30 days). This is a significant concern because it often indicates potential gaps in patient care and leads to major financial and operational burdens for hospitals.

High readmission rates are often a symptom of underlying issues. The patient may have been discharged prematurely, may not have understood their post-discharge care instructions, or may lack the necessary support at home. This leads to a decline in the patient's health and a lower quality of care.

Unplanned readmissions strain hospital resources. They create unexpected demand for beds, staff, and medical equipment, which disrupts scheduled procedures and can lead to overcrowding and staff burnout.

The Objective: Proactive Risk Stratification

The primary objective of this project is to build a machine learning model that can accurately predict the likelihood of a patient being readmitted. The goal is not just to build a model, but to create a tool that enables a fundamental shift from reactive to proactive patient care.

Identify High-Risk Patients Early
Generate a Risk Probability
DATASET
https://www.kaggle.com/datasets/dubradave/hospital-readmissions?resource=download

Information in the Dataset

"age" - age bracket of the patient

"time_in_hospital" - days (from 1 to 14)

"n_procedures" - number of procedures performed during the hospital stay

"n_lab_procedures" - number of laboratory procedures performed during the hospital stay

"n_medications" - number of medications administered during the hospital stay

"n_outpatient"- number of outpatient visits in the year before a hospital stay

"n_inpatient" - number of inpatient visits in the year before the hospital stay

"n_emergency"- number of visits to the emergency room in the year before the hospital stay

"medical_specialty" - the specialty of the admitting physician

"diag_1"- primary diagnosis (Circulatory, Respiratory, Digestive, etc.)

"diag_2"- secondary diagnosis

"diag_3" - additional secondary diagnosis

"glucose_test" - whether the glucose serum came out as high (> 200), normal, or not performed

"A1Ctest" - whether the A1C level of the patient came out as high (> 7%), normal, or not performed

"change" - whether there was a change in the diabetes medication ('yes' or 'no')

"diabetes_med" - whether a diabetes medication was prescribed ('yes' or 'no')

"readmitted" - if the patient was readmitted at the hospital ('yes' or 'no')
