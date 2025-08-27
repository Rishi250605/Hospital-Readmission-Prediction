"""
Enhanced Explainable AI Module for Hospital Readmission Prediction
Uses SHAP for comprehensive model interpretability with detailed analysis
"""

# To install required packages, run:
# pip install shap pandas numpy matplotlib seaborn

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for web deployment
import seaborn as sns
import base64
import io
import json
from collections import OrderedDict
import warnings

# Suppress noisy sklearn pipeline FutureWarnings during SHAP/name probing
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.pipeline")

class ModelExplainer:
    """
    Enhanced SHAP explanations for the hospital readmission model with comprehensive analysis
    """
    
    def __init__(self, model_package, background_data=None):
        """
        Initialize the explainer
        
        Args:
            model_package: Dict containing pipeline, threshold, feature_names
            background_data: Background dataset for SHAP (optional)
        """
        self.pipeline = model_package['pipeline']
        self.threshold = model_package.get('threshold', 0.4)
        self.feature_names = model_package.get('feature_names', None)
        
        # Initialize SHAP explainer - using TreeExplainer for tree-based models
        try:
            # Get the classifier from pipeline
            classifier = self.pipeline.named_steps['classifier']
            self.explainer = shap.TreeExplainer(classifier)
            print("TreeExplainer initialized successfully")
        except Exception as e:
            print(f"TreeExplainer failed: {e}. Trying Explainer...")
            # Fallback to general explainer
            self.explainer = shap.Explainer(self.pipeline.predict, background_data)
    
    def preprocess_input(self, input_df):
        """
        Preprocess input through the pipeline (excluding final classifier)
        
        Args:
            input_df: Raw input DataFrame
            
        Returns:
            Preprocessed features ready for the classifier
        """
        # Apply all preprocessing steps except the final classifier
        preprocessed = input_df.copy()
        
        # Apply each step manually
        preprocessed = self.pipeline.named_steps['age_encoder'].transform(preprocessed)
        preprocessed = self.pipeline.named_steps['feature_creator'].transform(preprocessed)
        preprocessed = self.pipeline.named_steps['preprocessor'].transform(preprocessed)
        
        return preprocessed
    
    def get_feature_names_after_preprocessing(self):
        """
        Get human-readable feature names after preprocessing with enhanced mappings
        """
        if self.feature_names:
            return self.feature_names

        preprocessor = self.pipeline.named_steps['preprocessor']

        # Enhanced feature name mapping for better interpretability (layman-friendly)
        feature_name_mapping = {
            'time_in_hospital': 'Days Spent in Hospital',
            'n_procedures': 'Number of Medical Procedures',
            'n_lab_procedures': 'Number of Lab Tests',
            'n_medications': 'Number of Medications Given',
            'n_outpatient': 'Outpatient Visits (Past Year)',
            'n_inpatient': 'Previous Hospital Admissions (Past Year)',
            'n_emergency': 'Emergency Room Visits (Past Year)',
            'n_visits': 'Total Healthcare Visits',
            'proc_med_ratio': 'Procedures to Medications Ratio',
            'age_encoded': 'Patient Age Group',
            'medical_specialty': 'Doctor\'s Specialty',
            'diag_1': 'Main Reason for Hospital Stay',
            'diag_2': 'Second Health Issue',
            'diag_3': 'Third Health Issue',
            'glucose_test': 'Glucose Blood Test Result',
            'A1Ctest': 'A1C Blood Test Result',
            'change_in_med': 'Change in Diabetes Medication',
            'diabetes_med': 'Was Diabetes Medication Prescribed?'
        }

        # 1) Best case: sklearn exposes names directly
        try:
            names = preprocessor.get_feature_names_out()
            # Apply friendly name mapping
            friendly_names = []
            for name in names:
                base_name = name.split('__')[-1] if '__' in name else name
                friendly_name = feature_name_mapping.get(base_name, base_name)
                friendly_names.append(friendly_name)
            return friendly_names
        except Exception:
            pass

        # 2) Introspect ColumnTransformer
        names = []
        try:
            if hasattr(preprocessor, 'transformers_'):
                for trans_name, transformer, columns in preprocessor.transformers_:
                    if transformer == 'drop':
                        continue
                    if transformer == 'passthrough':
                        if isinstance(columns, (list, tuple)):
                            for col in columns:
                                friendly_name = feature_name_mapping.get(str(col), str(col))
                                names.append(friendly_name)
                        continue

                    # Unwrap pipeline inside transformer if present
                    inner = transformer
                    if hasattr(transformer, 'steps'):
                        inner = transformer.steps[-1][1]

                    # Prefer estimator-level names
                    if hasattr(inner, 'get_feature_names_out'):
                        try:
                            if isinstance(columns, (list, tuple)):
                                cols = list(columns)
                            else:
                                cols = None
                            out = inner.get_feature_names_out(input_features=cols) if cols is not None else inner.get_feature_names_out()
                            for name in out:
                                base_name = str(name).split('__')[-1] if '__' in str(name) else str(name)
                                friendly_name = feature_name_mapping.get(base_name, str(name))
                                names.append(friendly_name)
                            continue
                        except Exception:
                            pass

                    # Fallback: just append raw columns with mapping
                    if isinstance(columns, (list, tuple)):
                        for col in columns:
                            friendly_name = feature_name_mapping.get(str(col), str(col))
                            names.append(friendly_name)

            if names:
                return names
        except Exception:
            pass

        # 3) Last resort: length-based generic names using a small probe
        try:
            sample = {
                'age_encoded': 5,
                'time_in_hospital': 1,
                'n_procedures': 0,
                'n_lab_procedures': 0,
                'n_medications': 0,
                'n_outpatient': 0,
                'n_inpatient': 0,
                'n_emergency': 0,
                'n_visits': 0,
                'proc_med_ratio': 0.0,
                'medical_specialty': 'Other',
                'diag_1': 'Other',
                'diag_2': 'Other',
                'diag_3': 'Other',
                'glucose_test': 'no',
                'A1Ctest': 'no',
                'change_in_med': 'no',
                'diabetes_med': 'no'
            }
            n_features = preprocessor.transform(pd.DataFrame([sample])).shape[1]
        except Exception:
            n_features = 20
        
        # Use mapping for generic features
        generic_names = []
        for i in range(n_features):
            generic_names.append(f"Clinical Feature {i+1}")
        return generic_names
    
    def explain_prediction(self, input_df):
        """
        Generate comprehensive SHAP explanations for a single prediction
        
        Args:
            input_df: Input DataFrame (single row)
            
        Returns:
            Dictionary containing detailed prediction and explanations
        """
        try:
            # Get prediction probability
            prob = self.pipeline.predict_proba(input_df)[0][1]
            prediction = "yes" if prob >= self.threshold else "no"
            
            # Preprocess input for SHAP
            X_processed = self.preprocess_input(input_df)
            
            # Generate SHAP values (support both array and Explanation outputs)
            shap_values_raw = self.explainer.shap_values(X_processed)
            
            # Normalize SHAP output to a 1D array for the single sample and positive class
            if hasattr(shap_values_raw, "values"):
                values = shap_values_raw.values
            else:
                values = shap_values_raw
            
            if isinstance(values, list):
                if len(values) == 1:
                    arr = values[0]
                    if isinstance(arr, list):
                        arr = np.asarray(arr)
                    if isinstance(arr, np.ndarray):
                        shap_values_positive = arr[0] if arr.ndim == 2 else arr
                    else:
                        shap_values_positive = np.array(arr)
                else:
                    shap_values_positive = values[1][0]
            else:
                if values.ndim == 3:
                    class_index = 1 if values.shape[0] > 1 else 0
                    shap_values_positive = values[class_index, 0, :]
                elif values.ndim == 2:
                    shap_values_positive = values[0, :]
                else:
                    shap_values_positive = values
            
            # Get feature names
            feature_names = self.get_feature_names_after_preprocessing()
            
            # Ensure we have the right number of features
            if len(feature_names) != len(shap_values_positive):
                if len(feature_names) > len(shap_values_positive):
                    feature_names = feature_names[:len(shap_values_positive)]
                else:
                    feature_names = feature_names + [f"Clinical Feature {i}" for i in range(len(feature_names), len(shap_values_positive))]
            
            # Create comprehensive feature importance data
            feature_importance = []
            for i, (name, importance) in enumerate(zip(feature_names, shap_values_positive)):
                feature_importance.append({
                    'feature': name,
                    'importance': float(importance),
                    'abs_importance': abs(float(importance)),
                    'interpretation': self._interpret_feature_impact(name, importance),
                    'clinical_significance': self._get_clinical_significance(name, importance)
                })
            
            # Sort by absolute importance
            feature_importance.sort(key=lambda x: x['abs_importance'], reverse=True)
            
            # Get top features for different analyses
            top_features = feature_importance[:10]
            high_impact_features = [f for f in feature_importance if f['abs_importance'] > 0.01]
            protective_factors = [f for f in feature_importance if f['importance'] < -0.005][:5]
            risk_factors = [f for f in feature_importance if f['importance'] > 0.005][:5]
            
            # Calculate base value
            expected = self.explainer.expected_value
            if isinstance(expected, (list, tuple)):
                expected_scalar = expected[1] if len(expected) > 1 else expected[0]
            elif isinstance(expected, np.ndarray):
                if expected.ndim == 0:
                    expected_scalar = expected.item()
                else:
                    expected_scalar = expected.flat[1] if expected.size > 1 else expected.flat[0]
            else:
                expected_scalar = expected
            base_value = float(expected_scalar)
            
            # Generate comprehensive analysis
            risk_assessment = self._generate_risk_assessment(prediction, prob, self.threshold)
            clinical_insights = self._generate_clinical_insights(top_features[:5], base_value, input_df)
            recommendation = self._generate_recommendations(prediction, prob, risk_factors, protective_factors)
            
            # Generate detailed explanation
            explanation_text = self._generate_comprehensive_explanation_text(
                prediction, prob, top_features[:5], base_value, risk_assessment, clinical_insights
            )
            
            return {
                'prediction': prediction,
                'probability': round(float(prob), 4),
                'threshold': self.threshold,
                'base_value': round(base_value, 4),
                'confidence_level': self._calculate_confidence(prob, self.threshold),
                'risk_category': self._categorize_risk(prob),
                
                # Feature analysis
                'top_features': top_features,
                'all_features': feature_importance,
                'high_impact_features': high_impact_features,
                'risk_factors': risk_factors,
                'protective_factors': protective_factors,
                
                # Comprehensive explanations
                'explanation_text': explanation_text,
                'risk_assessment': risk_assessment,
                'clinical_insights': clinical_insights,
                'recommendations': recommendation,
                
                # Visualizations
                'shap_plot': self._create_enhanced_shap_plot(top_features),
                'risk_breakdown_plot': self._create_risk_breakdown_plot(risk_factors, protective_factors),
                
                # Additional metrics
                'feature_count': len(feature_importance),
                'significant_features': len(high_impact_features),
                'model_certainty': abs(prob - 0.5) * 2  # How far from uncertain (0.5)
            }
            
        except Exception as e:
            print(f"Error in explain_prediction: {e}")
            # Enhanced fallback explanation
            prob = self.pipeline.predict_proba(input_df)[0][1]
            prediction = "yes" if prob >= self.threshold else "no"
            
            return {
                'prediction': prediction,
                'probability': round(float(prob), 4),
                'threshold': self.threshold,
                'confidence_level': 'Medium',
                'risk_category': self._categorize_risk(prob),
                'explanation_text': f"""
                <div class="explanation-text">
                    <h4>Basic Prediction Available:</h4>
                    <p><strong>Prediction:</strong> <span class="prediction-{prediction}">{prediction.upper()}</span> 
                       (Probability: {prob:.1%})</p>
                    <p>The model predicts {'a higher' if prediction == 'yes' else 'a lower'} risk of readmission based on the provided patient data.</p>
                    <p><em>Detailed feature analysis temporarily unavailable due to technical issues.</em></p>
                </div>
                """,
                'error': str(e)
            }
    
    def _interpret_feature_impact(self, feature_name, importance):
        """Generate human-readable interpretation of feature impact"""
        if abs(importance) < 0.001:
            return "Minimal impact on prediction"
        
        direction = "increases" if importance > 0 else "decreases"
        magnitude = "significantly" if abs(importance) > 0.02 else "moderately" if abs(importance) > 0.01 else "slightly"
        
        return f"This factor {magnitude} {direction} readmission risk"
    
    def _get_clinical_significance(self, feature_name, importance):
        """Provide clinical context for features (layman-friendly)"""
        clinical_context = {
            'Days Spent in Hospital': 'Longer stays may mean more serious illness or better recovery time.',
            'Number of Lab Tests': 'More lab tests can mean closer monitoring or more health issues.',
            'Number of Medications Given': 'Taking more medications often means more health problems.',
            'Outpatient Visits (Past Year)': 'Regular outpatient care can help prevent problems or show ongoing issues.',
            'Previous Hospital Admissions (Past Year)': 'More past admissions can mean ongoing health challenges.',
            'Emergency Room Visits (Past Year)': 'Frequent ER visits may signal unstable health.',
            'Main Reason for Hospital Stay': 'The main illness or reason for admission affects readmission risk.',
            'Was Diabetes Medication Prescribed?': 'Managing diabetes with medication can affect readmission chances.',
            'Patient Age Group': 'Certain age groups may have higher risk of returning to the hospital.',
            'Doctor\'s Specialty': 'The type of doctor treating the patient can influence outcomes.',
            'Second Health Issue': 'Other health problems can add to the risk.',
            'Third Health Issue': 'Additional health issues may further increase risk.',
            'Glucose Blood Test Result': 'Abnormal blood sugar can increase risk.',
            'A1C Blood Test Result': 'High A1C means poor long-term blood sugar control.',
            'Change in Diabetes Medication': 'Changing diabetes medication may mean unstable diabetes.'
        }
        return clinical_context.get(feature_name, 'This factor may affect readmission risk depending on the patient.')
    
    def _generate_risk_assessment(self, prediction, probability, threshold):
        """Generate detailed risk assessment"""
        if prediction == "yes":
            if probability > 0.7:
                level = "HIGH"
                description = "This patient has a high probability of readmission within 30 days."
            elif probability > threshold + 0.1:
                level = "MODERATE-HIGH"
                description = "This patient shows elevated readmission risk requiring attention."
            else:
                level = "MODERATE"
                description = "This patient is just above the threshold for readmission risk."
        else:
            if probability < 0.2:
                level = "LOW"
                description = "This patient has a low probability of readmission."
            elif probability < threshold - 0.1:
                level = "MODERATE-LOW"
                description = "This patient shows below-average readmission risk."
            else:
                level = "BORDERLINE"
                description = "This patient is close to the readmission risk threshold."
        
        return {
            'level': level,
            'description': description,
            'probability_range': f"{probability:.1%}",
            'threshold_distance': abs(probability - threshold)
        }
    
    def _generate_clinical_insights(self, top_features, base_value, input_df):
        """Generate clinical insights from top features"""
        insights = []
        
        # Analyze top contributing factors
        for feature in top_features[:3]:
            if feature['importance'] > 0.01:
                insights.append(f"• **{feature['feature']}** is a major risk factor for this patient")
            elif feature['importance'] < -0.01:
                insights.append(f"• **{feature['feature']}** is providing significant protection against readmission")
            else:
                insights.append(f"• **{feature['feature']}** has moderate influence on the prediction")
        
        # Add contextual insights
        if base_value > 0.5:
            insights.append(f"• The baseline readmission risk for similar patients is elevated ({base_value:.1%})")
        else:
            insights.append(f"• The baseline readmission risk for similar patients is moderate ({base_value:.1%})")
        
        return insights
    
    def _generate_recommendations(self, prediction, probability, risk_factors, protective_factors):
        """Generate actionable recommendations"""
        recommendations = []
        
        if prediction == "yes":
            recommendations.append("**Immediate Actions Recommended:**")
            recommendations.append("• Enhanced discharge planning and coordination")
            recommendations.append("• Schedule early follow-up appointment (within 7 days)")
            recommendations.append("• Provide comprehensive medication reconciliation")
            recommendations.append("• Consider care transition interventions")
            
            if risk_factors:
                recommendations.append(f"**Focus Areas:** Address {', '.join([f['feature'] for f in risk_factors[:2]])}")
        else:
            recommendations.append("**Standard Discharge Process Appropriate:**")
            recommendations.append("• Standard follow-up timing is sufficient")
            recommendations.append("• Continue routine care coordination")
            
            if protective_factors:
                recommendations.append(f"**Strengths:** {', '.join([f['feature'] for f in protective_factors[:2]])} are working in patient's favor")
        
        return recommendations
    
    def _calculate_confidence(self, probability, threshold):
        """Calculate confidence level in prediction"""
        distance = abs(probability - threshold)
        if distance > 0.2:
            return "High"
        elif distance > 0.1:
            return "Medium"
        else:
            return "Low"
    
    def _categorize_risk(self, probability):
        """Categorize overall risk level"""
        if probability >= 0.7:
            return "Very High Risk"
        elif probability >= 0.5:
            return "High Risk"
        elif probability >= 0.3:
            return "Moderate Risk"
        else:
            return "Low Risk"
    
    def _generate_comprehensive_explanation_text(self, prediction, probability, top_features, base_value, risk_assessment, clinical_insights):
        """Generate comprehensive explanation text with rich analysis"""
        explanation = f"""
        <div class="explanation-text">
            <div class="prediction-summary">
                <h4>🏥 Prediction Summary</h4>
                <p><strong>Readmission Prediction:</strong> <span class="prediction-{prediction}">{prediction.upper()}</span></p>
                <p><strong>Probability:</strong> {probability:.1%} | <strong>Risk Level:</strong> {risk_assessment['level']}</p>
                <p><strong>Assessment:</strong> {risk_assessment['description']}</p>
            </div>
            
            <div class="model-reasoning">
                <h4>📊 How the Model Made This Decision</h4>
                <p><strong>Baseline Population Risk:</strong> {base_value:.1%} - This represents the average readmission rate for similar patients in the training data.</p>
                
                <div class="feature-analysis">
                    <h5>🔍 Key Contributing Factors:</h5>
                    <ul>
        """
        
        for i, feature in enumerate(top_features[:4]):
            impact_direction = "⬆️ INCREASES" if feature['importance'] > 0 else "⬇️ DECREASES"
            impact_magnitude = "Significantly" if abs(feature['importance']) > 0.02 else "Moderately" if abs(feature['importance']) > 0.01 else "Slightly"
            
            explanation += f"""
                        <li><strong>{feature['feature']}:</strong> {impact_direction} risk by {abs(feature['importance']):.3f}
                            <br><em>{impact_magnitude} influential - {feature['interpretation']}</em>
                            <br><small>Clinical Note: {feature['clinical_significance']}</small>
                        </li>
            """
        
        explanation += """
                    </ul>
                </div>
                
                <div class="clinical-insights">
                    <h5>💡 Clinical Insights:</h5>
                    <ul>
        """
        
        for insight in clinical_insights:
            explanation += f"                        <li>{insight}</li>\n"
        
        explanation += f"""
                    </ul>
                </div>
                
                <div class="confidence-note">
                    <p><strong>Model Confidence:</strong> The model is <em>{abs(probability - 0.5) * 2:.1%} certain</em> about this prediction (distance from neutral 50%).</p>
                </div>
            </div>
        </div>
        """
        
        return explanation
    
    def _create_enhanced_shap_plot(self, top_features):
        """Create an enhanced SHAP waterfall plot with better styling"""
        try:
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Extract data for plotting
            features = [f['feature'] for f in top_features]
            values = [f['importance'] for f in top_features]
            
            # Create horizontal bar plot with enhanced styling
            colors = ['#ef4444' if v > 0 else '#10b981' for v in values]
            bars = ax.barh(features, values, color=colors, alpha=0.8, height=0.6)
            
            # Enhanced customization
            ax.set_xlabel('SHAP Value (Impact on Readmission Probability)', fontsize=14, fontweight='bold')
            ax.set_title('Feature Contributions to Readmission Prediction', fontsize=16, fontweight='bold', pad=20)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1.2)
            
            # Add value labels on bars with better positioning
            for i, (bar, value) in enumerate(zip(bars, values)):
                width = bar.get_width()
                label_x = width + (0.002 if width >= 0 else -0.002)
                ax.text(label_x, bar.get_y() + bar.get_height()/2,
                       f'{value:+.3f}', ha='left' if width >= 0 else 'right', 
                       va='center', fontweight='bold', fontsize=11)
            
            # Enhanced grid and styling
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Improve y-axis labels
            ax.tick_params(axis='y', labelsize=11)
            ax.tick_params(axis='x', labelsize=11)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close()
            
            plot_base64 = base64.b64encode(plot_data).decode()
            return f"data:image/png;base64,{plot_base64}"
            
        except Exception as e:
            print(f"Error creating enhanced SHAP plot: {e}")
            return None
    
    def _create_risk_breakdown_plot(self, risk_factors, protective_factors):
        """Create a risk breakdown visualization"""
        try:
            if not risk_factors and not protective_factors:
                return None
                
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Risk factors plot
            if risk_factors:
                risk_names = [f['feature'][:20] + '...' if len(f['feature']) > 20 else f['feature'] for f in risk_factors]
                risk_values = [f['importance'] for f in risk_factors]
                ax1.barh(risk_names, risk_values, color='#ef4444', alpha=0.7)
                ax1.set_title('Risk-Increasing Factors', fontweight='bold', color='#dc2626')
                ax1.set_xlabel('Impact on Risk')
            else:
                ax1.text(0.5, 0.5, 'No significant\nrisk factors identified', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Risk-Increasing Factors', fontweight='bold')
            
            # Protective factors plot
            if protective_factors:
                prot_names = [f['feature'][:20] + '...' if len(f['feature']) > 20 else f['feature'] for f in protective_factors]
                prot_values = [abs(f['importance']) for f in protective_factors]  # Show as positive for clarity
                ax2.barh(prot_names, prot_values, color='#10b981', alpha=0.7)
                ax2.set_title('Protective Factors', fontweight='bold', color='#059669')
                ax2.set_xlabel('Protective Effect')
            else:
                ax2.text(0.5, 0.5, 'No significant\nprotective factors identified', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Protective Factors', fontweight='bold')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
            buffer.seek(0)
            plot_data = buffer.getvalue()
            buffer.close()
            plt.close()
            
            plot_base64 = base64.b64encode(plot_data).decode()
            return f"data:image/png;base64,{plot_base64}"
            
        except Exception as e:
            print(f"Error creating risk breakdown plot: {e}")
            return None
    
    def get_model_summary(self):
        """
        Get comprehensive model information
        """
        try:
            classifier = self.pipeline.named_steps['classifier']
            model_type = type(classifier).__name__
            
            # Try to get feature importance from the model
            if hasattr(classifier, 'feature_importances_'):
                feature_names = self.get_feature_names_after_preprocessing()
                importances = classifier.feature_importances_
                
                feature_importance = [
                    {'feature': name, 'importance': float(imp)}
                    for name, imp in zip(feature_names, importances)
                ]
                feature_importance.sort(key=lambda x: x['importance'], reverse=True)
                
                return {
                    'model_type': model_type,
                    'threshold': self.threshold,
                    'total_features': len(feature_names),
                    'top_global_features': feature_importance[:10],
                    'model_complexity': 'High' if len(feature_names) > 50 else 'Medium' if len(feature_names) > 20 else 'Low'
                }
            else:
                return {
                    'model_type': model_type,
                    'threshold': self.threshold,
                    'model_complexity': 'Unknown'
                }
                
        except Exception as e:
            return {'error': f"Could not generate model summary: {e}"}


def create_explainer(model_package):
    """
    Factory function to create explainer instance
    """
    return ModelExplainer(model_package)